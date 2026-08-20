import { useEffect, useRef, useState } from "react";
import clsx from "clsx";
import type { PianoRoll } from "./pianoroll";
import { useAudioEngine } from "./hooks/useAudioEngine";
import {
  useTranscription,
  type AppState,
  type TranscriptionResult,
} from "./hooks/useTranscription";
import { Controls } from "./components/Controls";
import { OutputBar } from "./components/OutputBar";
import { FeedbackLine } from "./components/FeedbackLine";
import { PianoRollCanvas } from "./components/PianoRollCanvas";
import { InstrumentList } from "./components/InstrumentList";
import { DropOverlay } from "./components/DropOverlay";
import { Footer, PartnerLogos } from "./components/Footer";
import { WelcomeScreen } from "./components/WelcomeScreen";
import { ConsentBanner } from "./components/ConsentBanner";
import { Faq } from "./components/Faq";
import { track } from "./analytics";

/**
 * A failure surfaced on the welcome screen. `server` means the backend is
 * unreachable (health probe / network) and replaces the file picker entirely;
 * `file` is a per-upload problem (e.g. an undecodable audio file) shown
 * alongside the picker so the user can choose a different file.
 */
export type AppError = { kind: "server" | "file"; message: string };
import { ProgressEstimator, formatClock } from "./progress";
import { editToWavFile, type AudioEdit } from "./audio-edit";

type Screen = "welcome" | "transcribe";

/**
 * What has happened to the upload since "Transcribe" was clicked. The welcome
 * screen stays put until the server accepts the request, so a server that is
 * busy with someone else's transcription never yanks the user into an empty
 * piano roll — the button reports the wait instead.
 *
 * `submitting` = request in flight; `busy` = refused with 503, retrying at
 * `retryAt` (a `Date.now()` timestamp, so the button can count down to it).
 */
export type SubmitState =
  | { phase: "idle" }
  | { phase: "submitting" }
  | { phase: "busy"; retryAt: number; attempt: number };

// The song is Headache by Lost Deposit. ig: @lostdeposit
const EXAMPLE = {
  // Dev gets a 10s clip so local test transcriptions are quick.
  url: import.meta.env.DEV
    ? "/headache_by_lost_deposit_10s.mp3"
    : "/headache_by_lost_deposit_1min.mp3",
  filename: "Lost Deposit - Headache (example track)",
  conditioning: [
    "drums",
    "electric_bass",
    "distorted_electric_guitar",
    "clean_electric_guitar",
    "voice",
  ],
};

export function App() {
  const audio = useAudioEngine();
  const rollRef = useRef<PianoRoll | null>(null);
  const clockRef = useRef<HTMLSpanElement | null>(null);
  // Progress estimator (stable across renders) + the DOM nodes its smoothed
  // fraction/ETA are written into each frame.
  const progressRef = useRef<ProgressEstimator | null>(null);
  const progress = (progressRef.current ??= new ProgressEstimator());
  const progressFillRef = useRef<HTMLDivElement | null>(null);
  const progressLabelRef = useRef<HTMLSpanElement | null>(null);

  const [screen, setScreen] = useState<Screen>("welcome");
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  // The waveform editor's pending trim/cut, encoded only at submit time.
  const [audioEdit, setAudioEdit] = useState<AudioEdit | null>(null);
  const [appState, setAppState] = useState<AppState>("idle");
  const [submit, setSubmit] = useState<SubmitState>({ phase: "idle" });
  // Shown on the welcome screen: a server-down notice (set when the /health
  // check fails) or a per-file error (set when a transcription is rejected).
  // null = healthy and no file error.
  const [error, setError] = useState<AppError | null>(null);
  const [instruments, setInstruments] = useState<string[]>([]);
  // The finished transcription's exports
  const [result, setResult] = useState<TranscriptionResult | null>(null);
  const [currentFile, setCurrentFile] = useState<File | null>(null);
  const [mix, setMix] = useState(0.75);
  const [stereo, setStereo] = useState(false);
  const [userScrolled, setUserScrolled] = useState(false);
  const [condSelected, setCondSelected] = useState<Set<string>>(() => new Set());
  // True while a file is being dragged over the window. On the welcome screen
  // this swaps the panel's prompt in place instead of showing the overlay.
  const [dragging, setDragging] = useState(false);
  const midiFilenameRef = useRef("transcription.mid");
  // Mirror of the selected conditioning set, read at submit time without
  // re-creating `transcribe` whenever the selection changes.
  const condRef = useRef(condSelected);
  condRef.current = condSelected;
  // Read inside the per-frame loop (which only re-subscribes on `audio`) so the
  // transcribed-so-far tint is only drawn while a transcription is running.
  const appStateRef = useRef(appState);
  appStateRef.current = appState;

  const { transcribe, abort } = useTranscription({
    audio,
    rollRef,
    getConditioning: () => Array.from(condRef.current),
    progress,
    // A failed transcription bounces back to the welcome screen with a message.
    onError: (message) => {
      setSubmit({ phase: "idle" });
      setError({ kind: "file", message });
      setScreen("welcome");
    },
    // The server took the job: only now is there something to show.
    onAccepted: () => {
      setSubmit({ phase: "idle" });
      setScreen("transcribe");
    },
    // Refused (503) — stay on the welcome screen and count down to the retry.
    onBusy: ({ attempt, retryInMs }) =>
      setSubmit({ phase: "busy", retryAt: Date.now() + retryInMs, attempt }),
    setAppState,
    setInstruments,
    setResult,
    setCurrentFile,
    setUserScrolled,
  });
  // Submit the file picked on the welcome screen. The view only switches once
  // the server has accepted the request (`onAccepted` above); until then the
  // button reports progress — including waiting out a busy server. Called from
  // a button click, so the AudioContext unlock inside `transcribe` still
  // happens under a user gesture.
  function startTranscription() {
    if (selectedFile === null || submit.phase !== "idle") return;
    const audioFile =
      audioEdit === null
        ? selectedFile
        : editToWavFile(selectedFile, audioEdit);
    track("transcription_start", {
      instruments: Array.from(condSelected).sort().join(",") || "(none)",
      instrument_count: condSelected.size,
      is_example: selectedFile.name === EXAMPLE.filename,
      file_type: (selectedFile.name.match(/\.([^./]+)$/)?.[1] ?? "unknown").toLowerCase(),
      file_size_mb: Math.round(selectedFile.size / 1e5) / 10,
      trimmed: audioEdit !== null,
    });
    // Drop any leftover file error from a previous failed attempt.
    setError(null);
    setSubmit({ phase: "submitting" });
    transcribe(audioFile);
  }

  // Give up on a submission that hasn't been accepted yet (in flight, or waiting
  // out a busy server): stop retrying and re-enable the button.
  function cancelSubmit() {
    abort();
    setSubmit({ phase: "idle" });
    setAppState("idle");
  }

  // Choosing another file drops a not-yet-accepted submission, so a request that
  // finally gets through can't open the piano roll for the old file.
  function pickFile(file: File) {
    if (submit.phase !== "idle") cancelSubmit();
    setSelectedFile(file);
    setAudioEdit(null);
  }

  // Tear down the current transcription (in-flight or finished) and return to
  // the welcome screen. The previously chosen conditioning is kept so it's easy
  // to re-run with the same settings.
  function resetToWelcome() {
    // Stop the in-flight run right away (not just when the next one starts):
    // otherwise it keeps streaming notes into the torn-down UI and keeps the
    // server transcribing — and the server lock held — while the user picks
    // the next file.
    abort();
    audio.reset();
    rollRef.current?.clear();
    setInstruments([]);
    setResult(null);
    setUserScrolled(false);
    setAppState("idle");
    setSubmit({ phase: "idle" });
    setScreen("welcome");
  }

  // "Transcribe another file" (also the wordmark, on the transcribe screen):
  // confirm (the work is about to be discarded), then tear down and go back.
  function transcribeAnother() {
    if (!window.confirm("Discard this transcription and start over?")) return;
    setSelectedFile(null);
    setAudioEdit(null);
    resetToWelcome();
  }

  // Load the bundled demo track and pre-fill conditioning with a reasonable
  // guess for it (a rock track). The user still reviews and hits "Transcribe".
  async function useExample() {
    const resp = await fetch(EXAMPLE.url);
    if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
    const blob = await resp.blob();
    const file = new File([blob], EXAMPLE.filename, { type: "audio/mpeg" });
    setCondSelected(new Set(EXAMPLE.conditioning));
    pickFile(file);
  }

  // A file dropped anywhere on the page selects it on the welcome screen, from
  // either screen — dropping while a transcription is showing returns you to the
  // welcome screen with the new file picked, so you can choose conditioning
  // before hitting "Transcribe". It does not auto-start. Routed through a ref so
  // the window-level handler (installed once) always calls the latest closure.
  function onDropFile(file: File) {
    // Dropping onto the transcribe screen abandons the current run — confirm
    // first, then tear everything down (stop playback, clear the roll) so the
    // music doesn't keep playing behind the welcome screen.
    if (screen === "transcribe") {
      if (!window.confirm("Discard this transcription and start over with the dropped file?"))
        return;
      resetToWelcome();
    }
    pickFile(file);
  }
  const dropRef = useRef(onDropFile);
  dropRef.current = onDropFile;

  // Drive the body's data-state (it lives outside the React root) from state.
  useEffect(() => {
    document.body.dataset.state = appState;
  }, [appState]);

  // Drag & drop works anywhere on the page; a fullscreen overlay (CSS, keyed off
  // `body.drag`) shows while a file is being dragged. dragenter/dragleave fire
  // on every child element, so keep a depth counter to know when the drag truly
  // left the window.
  useEffect(() => {
    let dragDepth = 0;
    const onEnter = (e: DragEvent) => {
      e.preventDefault();
      dragDepth++;
      document.body.classList.add("drag");
      setDragging(true);
    };
    const onOver = (e: DragEvent) => e.preventDefault();
    const onLeave = () => {
      dragDepth = Math.max(0, dragDepth - 1);
      if (dragDepth === 0) {
        document.body.classList.remove("drag");
        setDragging(false);
      }
    };
    const onDrop = (e: DragEvent) => {
      e.preventDefault();
      dragDepth = 0;
      document.body.classList.remove("drag");
      setDragging(false);
      const f = e.dataTransfer?.files?.[0];
      if (f) dropRef.current(f);
    };
    window.addEventListener("dragenter", onEnter);
    window.addEventListener("dragover", onOver);
    window.addEventListener("dragleave", onLeave);
    window.addEventListener("drop", onDrop);
    return () => {
      window.removeEventListener("dragenter", onEnter);
      window.removeEventListener("dragover", onOver);
      window.removeEventListener("dragleave", onLeave);
      window.removeEventListener("drop", onDrop);
    };
  }, []);

  // Space toggles play/pause on the transcribe screen. Ignored while focus is
  // on a form control (slider, checkbox, button) so its native space behavior
  // is preserved.
  useEffect(() => {
    if (screen !== "transcribe") return;
    const onKey = (e: KeyboardEvent) => {
      if (e.code !== "Space" && e.key !== " ") return;
      const tag = (e.target as HTMLElement | null)?.tagName;
      if (tag === "INPUT" || tag === "BUTTON" || tag === "SELECT" || tag === "TEXTAREA")
        return;
      e.preventDefault();
      if (audio.state === "started") audio.pause();
      else audio.play();
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [screen, audio]);

  // Per-frame: advance the playhead, redraw the canvas, and tick the clock.
  // Kept off React state so it doesn't trigger ~60fps re-renders.
  useEffect(() => {
    let raf = 0;
    const frame = () => {
      const roll = rollRef.current;
      if (roll) {
        roll.setPlayhead(audio.seconds);
        // Feed the chunk-completion estimate of the transcribed span (completed
        // fraction × audio length). The roll combines it with the latest note
        // onset and eases the frontier itself; null before transcription starts
        // disables the tint. Once done the frontier sits at the full duration so
        // the whole roll keeps the lighter "transcribed" wash.
        const dur = audio.duration;
        const state = appStateRef.current;
        roll.setDuration(dur);
        if (dur > 0 && state === "done") roll.setTranscribedUntil(dur, false);
        else
          roll.setTranscribedUntil(
            dur > 0 && state === "transcribing"
              ? progress.completedFraction() * dur
              : null,
          );
        roll.render();
        // The roll re-engages follow mode on its own when the user scrolls back
        // to the live transcription frontier and holds still — mirror that into
        // the follow toggle's state.
        if (roll.consumeAutoResumed()) setUserScrolled(false);
      }
      if (clockRef.current) clockRef.current.textContent = `${audio.seconds.toFixed(1)}s`;
      // Drive the progress bar straight to the DOM (only mounted while
      // transcribing) so the smoothing doesn't re-render the app each frame.
      if (progressFillRef.current) {
        const now = performance.now();
        const frac = progress.fraction(now);
        progressFillRef.current.style.width = `${(frac * 100).toFixed(1)}%`;
        // Estimated time transcribed (smoothed fraction × audio length) out of
        // the file's total length, plus the ETA once a pace estimate exists.
        if (progressLabelRef.current) {
          const dur = audio.duration;
          if (dur > 0) {
            let text = `${formatClock(frac * dur)}/${formatClock(dur)}`;
            const eta = progress.etaMs(now);
            if (eta != null) text += `   done in ${formatClock(eta / 1000)}`;
            progressLabelRef.current.textContent = text;
          } else {
            progressLabelRef.current.textContent = "";
          }
        }
      }
      raf = requestAnimationFrame(frame);
    };
    raf = requestAnimationFrame(frame);
    return () => cancelAnimationFrame(raf);
  }, [audio]);

  // Expose for browser-devtools debugging.
  useEffect(() => {
    (window as unknown as { __mu: unknown }).__mu = { audio, rollRef };
  }, [audio]);

  return (
    <>
      <div className="grain" aria-hidden="true" />

      <header className="mx-auto flex max-w-7xl flex-wrap items-end justify-between gap-6 px-7 py-4 animate-rise">
        {/* Brand: the v2 mark (transparent PNG) + the wordmark as real text. */}
        <div
          className={clsx(
            "flex items-center gap-3 sm:gap-5",
            screen === "transcribe" && "cursor-pointer",
          )}
          onClick={screen === "transcribe" ? transcribeAnother : undefined}
          onKeyDown={
            screen === "transcribe"
              ? (e) => {
                  if (e.key === "Enter" || e.key === " ") {
                    e.preventDefault();
                    transcribeAnother();
                  }
                }
              : undefined
          }
          role={screen === "transcribe" ? "button" : undefined}
          tabIndex={screen === "transcribe" ? 0 : undefined}
          title={screen === "transcribe" ? "Transcribe another file" : undefined}
        >
          <img
            src="/muscriptor-logo-v4.svg"
            alt="MuScriptor logo"
            className="block h-[clamp(72px,10vw,110px)] w-auto"
            draggable={false}
          />
          <div className="flex flex-col gap-1">
            <span className="text-[clamp(2.3rem,6vw,3rem)] font-bold leading-none text-white">MuScriptor</span>
            <span className="text-sm text-muted">
              Music to MIDI and sheet music
            </span>
          </div>
        </div>

        {/* Also in the footer; here it's decoration, so it goes away on narrow
            screens rather than wrapping under the wordmark. */}
        <PartnerLogos className="self-center max-sm:hidden" />
      </header>

      {screen === "welcome" ? (
        <>
          <WelcomeScreen
            selectedFile={selectedFile}
            onPickFile={pickFile}
            onUseExample={useExample}
            condSelected={condSelected}
            onCondChange={setCondSelected}
            setEdit={setAudioEdit}
            onTranscribe={startTranscription}
            submitState={submit}
            onCancelSubmit={cancelSubmit}
            dragging={dragging}
            error={error}
            setError={setError}
          />
          <Faq />
        </>
      ) : (
        <main className="mx-auto grid max-w-7xl grid-cols-[1fr_300px] gap-4 px-7 pb-12 pt-2 max-[760px]:grid-cols-1">
          {/* Above the roll: exploring the result. */}
          <Controls
            audio={audio}
            clockRef={clockRef}
            mix={mix}
            onMixChange={(v) => {
              setMix(v);
              audio.setMix(v);
            }}
            stereo={stereo}
            onStereoChange={(v) => {
              setStereo(v);
              audio.setStereo(v);
            }}
            following={!userScrolled}
            onToggleFollow={() => {
              if (userScrolled) {
                rollRef.current?.follow();
                setUserScrolled(false);
              } else {
                rollRef.current?.unfollow();
                setUserScrolled(true);
              }
            }}
          />

          <PianoRollCanvas rollRef={rollRef} audio={audio} setUserScrolled={setUserScrolled} />

          {/* Sidebar column: the instrument list, with the feedback line pinned
              to the bottom of the column so it doesn't move as instruments
              come in. */}
          <div className="col-start-2 flex flex-col gap-3 max-[760px]:col-start-1">
            <InstrumentList instruments={instruments} given={condSelected} audio={audio} rollRef={rollRef} />
            <FeedbackLine className="mt-auto px-1" />
          </div>

          {/* Below the roll: the transcription job itself — progress, export,
              and starting over. */}
          <OutputBar
            transcribing={appState === "transcribing"}
            progressFillRef={progressFillRef}
            progressLabelRef={progressLabelRef}
            result={result}
            currentFile={currentFile}
            onTranscribeAnother={transcribeAnother}
          />
        </main>
      )}

      <Footer />

      <ConsentBanner />

      {screen === "transcribe" && <DropOverlay />}
    </>
  );
}
