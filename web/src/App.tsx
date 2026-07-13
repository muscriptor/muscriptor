import { useEffect, useRef, useState } from "react";
import clsx from "clsx";
import type { PianoRoll } from "./pianoroll";
import { useAudioEngine } from "./hooks/useAudioEngine";
import { useTranscription, type AppState } from "./hooks/useTranscription";
import { Controls } from "./components/Controls";
import { OutputBar } from "./components/OutputBar";
import { PianoRollCanvas } from "./components/PianoRollCanvas";
import { InstrumentList } from "./components/InstrumentList";
import { DropOverlay } from "./components/DropOverlay";
import { Footer } from "./components/Footer";
import { WelcomeScreen } from "./components/WelcomeScreen";

/**
 * A failure surfaced on the welcome screen. `server` means the backend is
 * unreachable (health probe / network) and replaces the file picker entirely;
 * `file` is a per-upload problem (e.g. an undecodable audio file) shown
 * alongside the picker so the user can choose a different file.
 */
export type AppError = { kind: "server" | "file"; message: string };
import { ProgressEstimator, formatClock } from "./progress";

type Screen = "welcome" | "transcribe";

// The song is Headache by Lost Deposit. ig: @lostdeposit
const EXAMPLE = {
  url: "/headache_by_lost_deposit_1min.mp3",
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
  const [appState, setAppState] = useState<AppState>("idle");
  // Shown on the welcome screen: a server-down notice (set when the /health
  // check fails) or a per-file error (set when a transcription is rejected).
  // null = healthy and no file error.
  const [error, setError] = useState<AppError | null>(null);
  const [instruments, setInstruments] = useState<string[]>([]);
  const [midiUrl, setMidiUrl] = useState<string | null>(null);
  const [midiBlob, setMidiBlob] = useState<Blob | null>(null);
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
      setError({ kind: "file", message });
      setScreen("welcome");
    },
    setAppState,
    setInstruments,
    setMidiUrl,
    setMidiBlob,
    setCurrentFile,
    setUserScrolled,
    midiFilenameRef,
  });
  // Start transcribing the file picked on the welcome screen and switch views.
  // Called from a button click, so the AudioContext unlock inside `transcribe`
  // still happens under a user gesture.
  function startTranscription() {
    if (selectedFile === null) return;
    // Drop any leftover file error from a previous failed attempt.
    setError(null);
    setScreen("transcribe");
    transcribe(selectedFile);
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
    setMidiUrl(null);
    setMidiBlob(null);
    setUserScrolled(false);
    setAppState("idle");
    setScreen("welcome");
  }

  // "Transcribe another file" (also the wordmark, on the transcribe screen):
  // confirm (the work is about to be discarded), then tear down and go back.
  function transcribeAnother() {
    if (!window.confirm("Discard this transcription and start over?")) return;
    setSelectedFile(null);
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
    setSelectedFile(file);
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
    setSelectedFile(file);
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
        // onset and eases the frontier itself; null while not transcribing
        // disables the tint (once done the whole roll is covered).
        const dur = audio.duration;
        roll.setDuration(dur);
        roll.setTranscribedUntil(
          appStateRef.current === "transcribing" && dur > 0
            ? progress.completedFraction() * dur
            : null,
        );
        roll.render();
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
            if (eta != null) text += ` · done in ${formatClock(eta / 1000)}`;
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

      <header className="mx-auto flex max-w-7xl flex-wrap items-end justify-between gap-6 px-7 pb-6 pt-10 max-[760px]:pt-8 animate-rise">
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
            src="/muscriptor-logo-v4-pink.svg"
            alt="MuScriptor logo"
            className="block h-[clamp(72px,10vw,110px)] w-auto"
            draggable={false}
          />
          <div className="flex flex-col gap-1">
            <span className="text-[clamp(2rem,8vw,3rem)] font-bold leading-none text-white">MuScriptor</span>
            <span className="text-sm text-muted">
              Audio to MIDI transcription
            </span>
          </div>
        </div>
      </header>

      {screen === "welcome" ? (
        <WelcomeScreen
          selectedFile={selectedFile}
          onPickFile={setSelectedFile}
          onUseExample={useExample}
          condSelected={condSelected}
          onCondChange={setCondSelected}
          onTranscribe={startTranscription}
          dragging={dragging}
          error={error}
          setError={setError}
        />
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

          <InstrumentList instruments={instruments} given={condSelected} audio={audio} rollRef={rollRef} />

          {/* Below the roll: the transcription job itself — progress, export,
              and starting over. */}
          <OutputBar
            transcribing={appState === "transcribing"}
            progressFillRef={progressFillRef}
            progressLabelRef={progressLabelRef}
            midiUrl={midiUrl}
            midiFilename={midiFilenameRef.current}
            midiBlob={midiBlob}
            currentFile={currentFile}
            onTranscribeAnother={transcribeAnother}
          />
        </main>
      )}

      <Footer />

      {screen === "transcribe" && <DropOverlay />}
    </>
  );
}
