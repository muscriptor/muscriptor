import { useRef, type Dispatch, type RefObject, type SetStateAction } from "react";
import { streamTranscribe, TranscribeError } from "../sse";
import { track } from "../analytics";
import type { AudioEngine } from "../audio";
import type { ProgressEstimator } from "../progress";
import { PianoRoll, type RollNote } from "../pianoroll";

type StartEvent = {
  type: "start";
  pitch: number;
  start_time: number;
  index: number;
  instrument: string;
};
type EndEvent = {
  type: "end";
  end_time: number;
  start_event_index: number;
};
type MidiEvent = {
  type: "midi";
  data: string; // base64-encoded .mid file
};
type ProgressMsg = {
  type: "progress";
  completed: number; // chunks transcribed so far
  total: number; // total chunks
};
type StreamedEvent = StartEvent | EndEvent | MidiEvent | ProgressMsg;

export type AppState = "idle" | "transcribing" | "done" | "error";

export interface TranscriptionDeps {
  audio: AudioEngine;
  /** Holds the PianoRoll once the canvas has mounted (may be null very early). */
  rollRef: RefObject<PianoRoll | null>;
  /** Conditioning instruments selected in the UI, read at submit time. */
  getConditioning: () => string[];
  /** Smooths chunk-completion anchors into a live progress fraction + ETA. */
  progress: ProgressEstimator;
  /** Called when a (non-superseded) transcription fails, so the UI can recover.
   *  `message` is a user-facing explanation (server-sent when available). */
  onError: (message: string) => void;
  setAppState: (s: AppState) => void;
  /** Detected-instrument names, in first-seen order. */
  setInstruments: Dispatch<SetStateAction<string[]>>;
  /** Object URL of the assembled MIDI file, or null to disable download. */
  setMidiUrl: (url: string | null) => void;
  /** Raw MIDI blob, re-uploaded to /auralize for the mix download. */
  setMidiBlob: (blob: Blob | null) => void;
  /** Source audio file, re-uploaded to /auralize alongside the MIDI. */
  setCurrentFile: (file: File | null) => void;
  setUserScrolled: (v: boolean) => void;
  /** Mutated so the download anchor can name the saved file. */
  midiFilenameRef: RefObject<string>;
}

/**
 * Returns a `transcribe(file)` function that streams the `/transcribe` SSE feed
 * into the piano roll + audio engine. Ported from the original imperative
 * `main.ts`; React state is touched only for low-frequency UI (status, detected
 * instruments, download URL) — notes themselves stay inside the two classes.
 */
export function useTranscription(deps: TranscriptionDeps) {
  const {
    audio,
    rollRef,
    getConditioning,
    progress,
    onError,
    setAppState,
    setInstruments,
    setMidiUrl,
    setMidiBlob,
    setCurrentFile,
    setUserScrolled,
    midiFilenameRef,
  } = deps;

  // Names already surfaced in the instrument list this run.
  const knownRef = useRef(new Set<string>());
  // Provisional notes awaiting their matching `end` event, keyed by start index.
  const openNotesRef = useRef(new Map<number, { startNote: RollNote }>());
  // Current MIDI object URL, kept so we can revoke it before creating a new one.
  const midiUrlRef = useRef<string | null>(null);
  // The in-flight transcription, if any. Starting a new one aborts this so its
  // (now stale) SSE stream stops feeding notes into the freshly-reset UI/audio.
  const activeRef = useRef<AbortController | null>(null);

  function addInstrument(name: string) {
    if (knownRef.current.has(name)) return;
    knownRef.current.add(name);
    setInstruments((prev) => [...prev, name]);
  }

  function reset() {
    rollRef.current?.clear();
    knownRef.current.clear();
    setInstruments([]);
    openNotesRef.current.clear();
    audio.reset();
    setUserScrolled(false);
  }

  function clearMidi() {
    if (midiUrlRef.current !== null) {
      URL.revokeObjectURL(midiUrlRef.current);
      midiUrlRef.current = null;
    }
    setMidiUrl(null);
    setMidiBlob(null);
  }

  function setMidi(base64: string) {
    const bytes = Uint8Array.from(atob(base64), (c) => c.charCodeAt(0));
    const blob = new Blob([bytes], { type: "audio/midi" });
    if (midiUrlRef.current !== null) URL.revokeObjectURL(midiUrlRef.current);
    const url = URL.createObjectURL(blob);
    midiUrlRef.current = url;
    setMidiUrl(url);
    setMidiBlob(blob);
  }

  function onEvent(ev: StartEvent | EndEvent) {
    const roll = rollRef.current;
    if (ev.type === "start") {
      addInstrument(ev.instrument);
      // Provisional end = start (will be patched on the matching end event).
      const note: RollNote = {
        pitch: ev.pitch,
        start: ev.start_time,
        end: ev.start_time,
        instrument: ev.instrument,
      };
      roll?.addNote(note);
      openNotesRef.current.set(ev.index, { startNote: note });
    } else {
      const open = openNotesRef.current.get(ev.start_event_index);
      if (!open) return;
      open.startNote.end = ev.end_time;
      // Finalize geometry in the roll too — it stamps the reveal time and
      // restacks any overlapping same-pitch notes. (The line above keeps `end`
      // correct for audio scheduling even if the canvas hasn't mounted yet.)
      roll?.setNoteEnd(open.startNote, ev.end_time);
      openNotesRef.current.delete(ev.start_event_index);
      audio.scheduleNote({
        instrument: open.startNote.instrument,
        pitch: open.startNote.pitch,
        start: open.startNote.start,
        end: open.startNote.end,
      });
    }
  }

  async function transcribe(file: File) {
    // Remember the source audio so the mix download can re-upload it.
    setCurrentFile(file);
    // Cancel any previous run before resetting state for this one.
    activeRef.current?.abort();
    const controller = new AbortController();
    activeRef.current = controller;
    // True once a *newer* transcription has superseded this one.
    const isStale = () => activeRef.current !== controller;

    reset();
    clearMidi();
    progress.reset();
    midiFilenameRef.current = file.name.replace(/\.[^/.]+$/, "") + ".mid";
    setAppState("transcribing");
    // Resume the AudioContext now, while we still have the user-gesture
    // activation from the drop / file-pick. Without this, calling play()
    // later — even from a click — sometimes can't unlock the context.
    audio.unlock().catch(() => {});
    // Decode the WAV in parallel with the transcription so it's ready by the
    // time the user hits play.
    audio.loadWav(file).catch(() => {});
    const startedAt = performance.now();
    let maxEnd = 0;
    let noteCount = 0;
    try {
      const cond = getConditioning();
      const extra = cond.length > 0 ? { instruments: cond } : undefined;
      for await (const raw of streamTranscribe(
        "/transcribe",
        file,
        extra,
        controller.signal,
      )) {
        // A newer upload took over while we were awaiting — drop this event so
        // it can't repopulate the piano roll / re-schedule old notes.
        if (isStale()) return;
        const ev = raw as StreamedEvent;
        if (ev.type === "midi") {
          // Final event: the assembled MIDI file. Enables the download button.
          setMidi(ev.data);
          continue;
        }
        if (ev.type === "progress") {
          // Coarse chunk anchor — the estimator smooths it into the live bar.
          progress.onAnchor(ev.completed, ev.total, performance.now());
          continue;
        }
        onEvent(ev);
        if (ev.type === "start") noteCount++;
        if (ev.type === "end" && ev.end_time > maxEnd) maxEnd = ev.end_time;
      }
      if (isStale()) return;
      // Stop the transport ~0.3 s after the last note ends.
      if (maxEnd > 0) audio.scheduleStop(maxEnd + 0.3);
      setAppState("done");
      track("transcription_complete", {
        // Decoded length of the uploaded audio; 0 if decoding hasn't finished
        // (or failed) by the time the transcription stream ends.
        audio_duration_s: Math.round(audio.duration),
        transcribe_time_s: Math.round((performance.now() - startedAt) / 1000),
        instruments: cond.slice().sort().join(",") || "(none)",
        instrument_count: cond.length,
        detected_instruments: Array.from(knownRef.current).sort().join(","),
        detected_count: knownRef.current.size,
        note_count: noteCount,
      });
    } catch (e) {
      // An abort (from being superseded) surfaces here as an error — ignore it
      // so it can't clobber the newer run.
      if (isStale() || controller.signal.aborted) return;
      setAppState("error");
      track("transcription_error", {
        status: e instanceof TranscribeError ? e.status : undefined,
        message: e instanceof Error ? e.message : String(e),
      });
      // Prefer the server's explanation (e.g. an undecodable file) over a
      // generic message; fall back when the failure was a network error or
      // the server gave no detail.
      const message =
        e instanceof TranscribeError && e.userMessage
          ? e.userMessage
          : "The muscriptor server is temporarily unavailable. Please try again later.";
      onError(message);
    } finally {
      if (activeRef.current === controller) activeRef.current = null;
    }
  }

  /** Abort the in-flight transcription (if any) without starting a new one.
   *  Call when the user leaves the transcribe screen, so the stale stream
   *  stops feeding notes into the audio engine — and so the server hears the
   *  disconnect while the user is still picking the next file. */
  function abort() {
    activeRef.current?.abort();
    activeRef.current = null;
  }

  return { transcribe, abort };
}
