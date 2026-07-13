import {
  useEffect,
  useRef,
  useState,
  type Dispatch,
  type SetStateAction,
} from "react";
import clsx from "clsx";
import { ConditioningPanel } from "./ConditioningPanel";
import type { AppError } from "../App";

const SERVER_DOWN = "The muscriptor server is temporarily unavailable.";

const NUMBER_INPUT_CLS =
  "rounded-lg border border-line-strong bg-bg px-2.5 py-2 text-sm text-content " +
  "outline-none transition-colors duration-150 ease-fluid focus:border-accent " +
  "disabled:cursor-not-allowed disabled:opacity-40";

/**
 * First screen of the two-step flow: pick an audio file, then optionally choose
 * conditioning instruments, then hit "Transcribe" to hand off to the main view.
 * Transcription doesn't start until the button is clicked.
 */
export function WelcomeScreen(props: {
  selectedFile: File | null;
  onPickFile: (file: File) => void;
  /** Loads the bundled demo track + its suggested conditioning. */
  onUseExample: () => Promise<void>;
  condSelected: Set<string>;
  onCondChange: (next: Set<string>) => void;
  condStrict: boolean;
  onCondStrictChange: (next: boolean) => void;
  /** Raw text of the MIDI-tempo field (parsed/validated at submit time). */
  tempoBpm: string;
  onTempoBpmChange: (next: string) => void;
  /** Decode with temperature sampling instead of greedy. */
  useSampling: boolean;
  onUseSamplingChange: (next: boolean) => void;
  /** Raw text of the temperature field (only used with sampling). */
  temperature: string;
  onTemperatureChange: (next: string) => void;
  /** Raw text of the classifier-free-guidance field. */
  cfgCoef: string;
  onCfgCoefChange: (next: string) => void;
  /** Raw text of the beam-width field. */
  beamSize: string;
  onBeamSizeChange: (next: string) => void;
  /** Server-side beam width cap; 1 hides the beam-size control entirely. */
  maxBeamSize: number;
  onTranscribe: () => void;
  /** True while a file is dragged over the window; swaps the prompt in place. */
  dragging: boolean;
  /** A server-down notice replaces the picker; a file error sits beside it. */
  error: AppError | null;
  setError: Dispatch<SetStateAction<AppError | null>>;
}) {
  const {
    selectedFile,
    onPickFile,
    onUseExample,
    condSelected,
    onCondChange,
    condStrict,
    onCondStrictChange,
    tempoBpm,
    onTempoBpmChange,
    useSampling,
    onUseSamplingChange,
    temperature,
    onTemperatureChange,
    cfgCoef,
    onCfgCoefChange,
    beamSize,
    onBeamSizeChange,
    maxBeamSize,
    onTranscribe,
    dragging,
    error,
    setError,
  } = props;
  const fileInputRef = useRef<HTMLInputElement>(null);
  const [loadingExample, setLoadingExample] = useState(false);

  // Probe the server on mount. A failure swaps the file picker for a
  // server-down notice; success clears a stale server-down notice so the user
  // can try again once the server recovers. A file error (e.g. an undecodable
  // upload) is left alone — the server being up doesn't make a bad file good.
  useEffect(() => {
    let cancelled = false;
    const clearServerError = () =>
      setError((prev) => (prev?.kind === "server" ? null : prev));
    fetch("/health")
      .then((r) => {
        if (cancelled) return;
        if (r.ok) clearServerError();
        else setError({ kind: "server", message: SERVER_DOWN });
      })
      .catch(() => {
        if (!cancelled) setError({ kind: "server", message: SERVER_DOWN });
      });
    return () => {
      cancelled = true;
    };
  }, [setError]);

  async function handleExample() {
    setLoadingExample(true);
    try {
      await onUseExample();
    } catch (e) {
      alert("Couldn't load the example file: " + (e as Error).message);
    } finally {
      setLoadingExample(false);
    }
  }

  return (
    <main className="mx-auto flex max-w-3xl flex-col gap-4 px-7 pb-12 pt-2 animate-rise [animation-delay:0.06s]">
      <p className="text-base leading-relaxed text-muted">
        MuScriptor is the best open model for multi-instrument transcription to
        date. Give it a recording: pop, classical, metal, jazz, whatever, and
        it transcribes the notes played by every instrument into
        MIDI, for you to download or explore interactively.
      </p>
      <input
        type="file"
        accept="audio/*"
        hidden
        ref={fileInputRef}
        onChange={(e) => {
          const f = e.target.files?.[0];
          if (f) onPickFile(f);
          // Allow re-picking the same file (onChange won't fire otherwise).
          e.target.value = "";
        }}
      />

      <section className={clsx("card p-0", dragging && "animate-drag-glow")}>
        {error?.kind === "server" ? (
          <div className="flex flex-col items-center gap-3 px-8 py-16 text-center">
            <p className="m-0 font-serif text-5xl leading-none text-content">
              unavailable
            </p>
            <p className="m-0 max-w-md text-base text-muted">{error.message}</p>
          </div>
        ) : selectedFile === null ? (
          <div className="flex flex-col items-center gap-4 px-8 py-16 text-center">
            <div
              className="h-16 w-32 bg-accent"
              style={{
                maskImage: "url(/muscriptor-wave.svg)",
                WebkitMaskImage: "url(/muscriptor-wave.svg)",
                maskSize: "contain",
                WebkitMaskSize: "contain",
                maskRepeat: "no-repeat",
                WebkitMaskRepeat: "no-repeat",
                maskPosition: "center",
                WebkitMaskPosition: "center",
              }}
              aria-hidden="true"
            />
            <p className="m-0 text-base text-muted">
              {dragging ? (
                <span className="font-semibold text-content">Drop anywhere</span>
              ) : (
                <>
                  Drop an{" "}
                  <strong className="font-semibold text-content">audio file</strong> here,
                  or
                </>
              )}
            </p>
            <button
              className="rounded-xl border-transparent bg-content px-7 py-3 text-base font-semibold text-[#15151b] hover:border-transparent hover:bg-white"
              onClick={() => fileInputRef.current?.click()}
            >
              Select an audio file
            </button>
            <button
              className="border-none bg-transparent px-1 py-0.5 text-sm text-muted underline underline-offset-4 hover:border-none hover:bg-transparent enabled:hover:text-content"
              onClick={handleExample}
              disabled={loadingExample}
            >
              {loadingExample ? "Loading example…" : "or try an example track"}
            </button>
          </div>
        ) : (
          <div className="flex flex-col items-start gap-2.5 px-8 py-7">
            <p
              className="m-0 max-w-full overflow-hidden text-ellipsis whitespace-nowrap text-2xl leading-[1.1] text-content"
              title={selectedFile.name}
            >
              {selectedFile.name}
            </p>
            <button onClick={() => fileInputRef.current?.click()}>
              Choose a different file
            </button>
          </div>
        )}
      </section>

      {error?.kind === "file" && (
        <p className="m-0 rounded-xl border border-red-500/40 bg-red-500/10 px-4 py-3 text-sm text-red-300">
          {error.message}
        </p>
      )}

      {error?.kind !== "server" && selectedFile !== null && (
        <>
          <ConditioningPanel
            selected={condSelected}
            onChange={onCondChange}
            strict={condStrict}
            onStrictChange={onCondStrictChange}
          />
          <div className="flex flex-wrap items-center justify-between gap-4">
            <div className="flex flex-wrap items-center gap-x-5 gap-y-3">
              <label
                className="flex items-center gap-2 text-sm text-muted"
                title={
                  "Tempo stamped into the downloaded MIDI file. Timing stays " +
                  "wall-clock accurate at any value — set your track's real BPM " +
                  "so beats land on your DAW's grid."
                }
              >
                MIDI tempo
                <input
                  type="number"
                  className={clsx("w-24", NUMBER_INPUT_CLS)}
                  min={10}
                  max={999}
                  step={0.1}
                  value={tempoBpm}
                  onChange={(e) => onTempoBpmChange(e.target.value)}
                />
                BPM
              </label>
              <label
                className="flex cursor-pointer select-none items-center gap-2 text-sm text-muted"
                title={
                  "Use temperature sampling instead of greedy decoding. Adds " +
                  "randomness — each run can give a different transcription."
                }
              >
                <input
                  type="checkbox"
                  className="accent-current"
                  checked={useSampling}
                  onChange={(e) => onUseSamplingChange(e.target.checked)}
                />
                Sampling
              </label>
              <label
                className={clsx(
                  "flex items-center gap-2 text-sm",
                  useSampling ? "text-muted" : "cursor-not-allowed text-faint",
                )}
                title={
                  "Sampling temperature. Higher values make the decoding more " +
                  "random; only used when sampling is enabled."
                }
              >
                Temperature
                <input
                  type="number"
                  className={clsx("w-20", NUMBER_INPUT_CLS)}
                  min={0.1}
                  max={10}
                  step={0.1}
                  value={temperature}
                  disabled={!useSampling}
                  onChange={(e) => onTemperatureChange(e.target.value)}
                />
              </label>
              <label
                className="flex items-center gap-2 text-sm text-muted"
                title={
                  "Classifier-free guidance coefficient. 1 disables guidance; " +
                  "higher values follow the instrument conditioning more strongly."
                }
              >
                Guidance
                <input
                  type="number"
                  className={clsx("w-20", NUMBER_INPUT_CLS)}
                  min={0}
                  max={10}
                  step={0.1}
                  value={cfgCoef}
                  onChange={(e) => onCfgCoefChange(e.target.value)}
                />
              </label>
              {maxBeamSize > 1 && (
                <label
                  className="flex items-center gap-2 text-sm text-muted"
                  title={
                    "Beam search width. 1 = greedy/sampling; higher values " +
                    "explore several decodings and keep the best, at a " +
                    `proportional cost in time (server cap: ${maxBeamSize}).`
                  }
                >
                  Beam size
                  <input
                    type="number"
                    className={clsx("w-20", NUMBER_INPUT_CLS)}
                    min={1}
                    max={maxBeamSize}
                    step={1}
                    value={beamSize}
                    onChange={(e) => onBeamSizeChange(e.target.value)}
                  />
                </label>
              )}
            </div>
            <button
              className="btn-primary rounded-xl px-9 py-3 text-base"
              onClick={onTranscribe}
            >
              Transcribe
            </button>
          </div>
        </>
      )}
    </main>
  );
}
