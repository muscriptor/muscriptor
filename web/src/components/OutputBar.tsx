import { useEffect, useRef, useState, type RefObject } from "react";
import clsx from "clsx";
import { Button } from "./Button";
import { IconChevron, IconDownload } from "./icons";
import { track } from "../analytics";

/**
 * Output / "job" actions, shown below the piano roll: live transcription
 * progress, exporting the result (MIDI or a stereo mix), and starting over
 * with another file. Distinct from the playback bar above the roll, which is
 * about exploring the result.
 *
 * The progress bar's fill width and "Xs / Ys" label are driven imperatively
 * from App's per-frame loop (via the refs) so the smoothing never triggers
 * React re-renders — same pattern as the playback clock.
 */
export function OutputBar(props: {
  transcribing: boolean;
  progressFillRef: RefObject<HTMLDivElement | null>;
  progressLabelRef: RefObject<HTMLSpanElement | null>;
  midiUrl: string | null;
  midiFilename: string;
  /** Raw MIDI blob + source audio, re-uploaded to /auralize for the mix. */
  midiBlob: Blob | null;
  currentFile: File | null;
  onTranscribeAnother: () => void;
}) {
  const {
    transcribing,
    progressFillRef,
    progressLabelRef,
    midiUrl,
    midiFilename,
    midiBlob,
    currentFile,
    onTranscribeAnother,
  } = props;
  const [synthesizing, setSynthesizing] = useState(false);
  const [menuOpen, setMenuOpen] = useState(false);
  const menuRef = useRef<HTMLDivElement>(null);
  const ready = midiUrl !== null;

  // Dismiss the download menu on outside click / Escape.
  useEffect(() => {
    if (!menuOpen) return;
    const onDown = (e: MouseEvent) => {
      if (menuRef.current && !menuRef.current.contains(e.target as Node)) setMenuOpen(false);
    };
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") setMenuOpen(false);
    };
    document.addEventListener("mousedown", onDown);
    document.addEventListener("keydown", onKey);
    return () => {
      document.removeEventListener("mousedown", onDown);
      document.removeEventListener("keydown", onKey);
    };
  }, [menuOpen]);

  function download() {
    if (midiUrl === null) return;
    track("download", { format: "midi" });
    const a = document.createElement("a");
    a.href = midiUrl;
    a.download = midiFilename;
    a.click();
  }

  // Renders the transcription server-side with FluidSynth. "synth" downloads
  // just the synthesized MIDI (mono); "mix" blends it with the original audio
  // (L = original, R = synthesis) for easy A/B comparison.
  async function downloadWav(mode: "synth" | "mix") {
    if (midiBlob === null || currentFile === null) return;
    track("download", { format: mode === "mix" ? "wav_mix" : "wav_synth" });
    setSynthesizing(true);
    try {
      const form = new FormData();
      form.append("mode", mode);
      form.append("midi", midiBlob, "transcription.mid");
      if (mode === "mix") form.append("audio", currentFile);
      const resp = await fetch("/auralize", { method: "POST", body: form });
      if (!resp.ok) {
        const text = await resp.text();
        let detail = text;
        try {
          detail = JSON.parse(text).detail ?? text;
        } catch {
          // not JSON — keep the raw body
        }
        throw new Error(detail);
      }
      const wavBlob = await resp.blob();
      const url = URL.createObjectURL(wavBlob);
      const a = document.createElement("a");
      a.href = url;
      const stem = currentFile.name.replace(/\.[^.]+$/, "") || "transcription";
      a.download = stem + (mode === "mix" ? "_mix.wav" : "_transcription.wav");
      a.click();
      URL.revokeObjectURL(url);
    } catch (e) {
      alert("Couldn't create the audio file: " + (e as Error).message);
    } finally {
      setSynthesizing(false);
    }
  }

  const menuItem =
    "block w-full rounded-none text-left text-[13px] font-normal text-content hover:bg-accent hover:text-black";

  return (
    <div className="col-span-full flex flex-wrap items-center gap-3 rounded-card border border-line bg-surface px-3.5 py-3 max-[760px]:border-0 max-[760px]:bg-transparent max-[760px]:p-0">
      {transcribing && (
        <div className="flex min-w-48 flex-1 items-center gap-3">
          <div className="h-1 flex-1 overflow-hidden border border-line bg-bg">
            <div
              ref={progressFillRef}
              className="h-full bg-accent"
              style={{ width: "0%" }}
            />
          </div>
          <span
            ref={progressLabelRef}
            className="shrink-0 whitespace-nowrap font-mono text-xs tabular-nums text-faint"
          >
            estimating…
          </span>
        </div>
      )}

      <div className="ml-auto flex items-center gap-2.5">
        <div className="relative" ref={menuRef}>
          <Button
            kind={ready ? "primary" : "secondary"}
            className="inline-flex items-center gap-2"
            disabled={!ready || synthesizing}
            onClick={() => setMenuOpen((o) => !o)}
            aria-haspopup="menu"
            aria-expanded={menuOpen}
          >
            <IconDownload />
            {synthesizing ? "Synthesizing…" : "Download"}
            <IconChevron
              className={clsx("transition-transform", menuOpen && "rotate-180")}
            />
          </Button>
          {menuOpen && (
            <div
              role="menu"
              className="absolute left-0 z-20 mt-1.5 min-w-48 overflow-hidden rounded-md border border-line-strong bg-bg py-1 shadow-pop"
            >
              <Button
                kind="ghost"
                pad="px-4 py-2"
                role="menuitem"
                className={menuItem}
                onClick={() => {
                  setMenuOpen(false);
                  download();
                }}
              >
                MIDI file
              </Button>
              <Button
                kind="ghost"
                pad="px-4 py-2"
                role="menuitem"
                className={menuItem}
                title="Just the transcribed notes, played with a SoundFont (mono)"
                onClick={() => {
                  setMenuOpen(false);
                  downloadWav("synth");
                }}
              >
                WAV - transcription only
              </Button>
              <Button
                kind="ghost"
                pad="px-4 py-2"
                role="menuitem"
                className={menuItem}
                title="Original audio (L) + transcribed notes played with a SoundFont (R)"
                onClick={() => {
                  setMenuOpen(false);
                  downloadWav("mix");
                }}
              >
                WAV - stereo with original
              </Button>
            </div>
          )}
        </div>
        <Button
          onClick={(e) => {
            e.currentTarget.blur();
            onTranscribeAnother();
          }}
        >
          Transcribe another
        </Button>
      </div>
    </div>
  );
}
