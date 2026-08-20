import { useEffect, useRef, useState, type RefObject } from "react";
import clsx from "clsx";
import { unzipSync } from "fflate";
import { Button } from "./Button";
import { SheetsDialog, type SheetFile } from "./SheetsDialog";
import type { TranscriptionResult } from "../hooks/useTranscription";
import { IconChevron, IconDownload } from "./icons";
import { track } from "../analytics";

/** The FastAPI `detail` of a failed response, or its raw body if it isn't one. */
async function errorDetail(resp: Response): Promise<string> {
  const text = await resp.text();
  try {
    return JSON.parse(text).detail ?? text;
  } catch {
    // not JSON — keep the raw body
    return text;
  }
}

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
  /** The finished transcription's exports, or null while there is none. */
  result: TranscriptionResult | null;
  /** Source audio, re-uploaded to /auralize alongside the MIDI for the mix. */
  currentFile: File | null;
  onTranscribeAnother: () => void;
}) {
  const {
    transcribing,
    progressFillRef,
    progressLabelRef,
    result,
    currentFile,
    onTranscribeAnother,
  } = props;
  // Label shown on the Download button while a server-side render runs, or null
  // when idle — the exports below all block the menu while they work.
  const [busy, setBusy] = useState<string | null>(null);
  const [menuOpen, setMenuOpen] = useState(false);
  // The unpacked sheet-music archive, once /sheets has answered. Kept after the
  // dialog is dismissed so reopening the picker doesn't engrave the same score
  // again — the transcription it was rendered from cannot change without this
  // whole bar unmounting.
  const [sheets, setSheets] = useState<{
    files: SheetFile[];
    zipBlob: Blob;
    zipFilename: string;
  } | null>(null);
  const [sheetsOpen, setSheetsOpen] = useState(false);
  const menuRef = useRef<HTMLDivElement>(null);
  const ready = result !== null;

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
    if (result === null) return;
    track("download", { format: "midi" });
    const a = document.createElement("a");
    a.href = result.url;
    a.download = result.filename;
    a.click();
  }

  /** The uploaded file's name without its extension, for naming exports. */
  function stem(): string {
    return currentFile?.name.replace(/\.[^.]+$/, "") || "transcription";
  }

  // Renders the transcription server-side with FluidSynth. "synth" downloads
  // just the synthesized MIDI (mono); "mix" blends it with the original audio
  // (L = original, R = synthesis) for easy A/B comparison.
  async function downloadWav(mode: "synth" | "mix") {
    if (result === null || currentFile === null) return;
    track("download", { format: mode === "mix" ? "wav_mix" : "wav_synth" });
    setBusy("Synthesizing…");
    try {
      const form = new FormData();
      form.append("mode", mode);
      form.append("midi", result.midi, "transcription.mid");
      if (mode === "mix") form.append("audio", currentFile);
      const resp = await fetch("/auralize", { method: "POST", body: form });
      if (!resp.ok) throw new Error(await errorDetail(resp));
      const wavBlob = await resp.blob();
      const url = URL.createObjectURL(wavBlob);
      const a = document.createElement("a");
      a.href = url;
      a.download = stem() + (mode === "mix" ? "_mix.wav" : "_transcription.wav");
      a.click();
      URL.revokeObjectURL(url);
    } catch (e) {
      track("download_error", {
        format: mode === "mix" ? "wav_mix" : "wav_synth",
        message: (e as Error).message,
      });
      alert("Couldn't create the audio file: " + (e as Error).message);
    } finally {
      setBusy(null);
    }
  }

  // Engraves the transcription as notation (MuseScore, server-side). That takes
  // long enough that it is done in bulk: one request renders the whole set —
  // MusicXML, the full score, a PDF per instrument, tablature for the fretted
  // ones — and answers with an uncompressed zip. Unpacking it here means the
  // user picks files out of something the browser already has, with no further
  // round trip and no archive to open by hand.
  async function downloadSheets() {
    if (result === null) return;
    track("download", { format: "sheets" });
    // Already engraved this run — just show the picker again.
    if (sheets !== null) {
      setSheetsOpen(true);
      return;
    }
    setBusy("Generating…");
    try {
      const form = new FormData();
      // Engrave the grid-snapped notes when the server managed to snap them:
      // notation made from raw transcription can have spurious 128th notes etc.
      const engrave = result.quantizedMidi ?? result.midi;
      form.append("midi", engrave, "transcription.mid");
      form.append("quantized", String(result.quantizedMidi !== null));
      const resp = await fetch("/sheets", { method: "POST", body: form });
      if (!resp.ok) throw new Error(await errorDetail(resp));
      const zipBlob = await resp.blob();
      // The members are stored, not deflated, so this unpacking is a copy out
      // of the buffer rather than an inflate of every PDF.
      const unpacked = unzipSync(new Uint8Array(await zipBlob.arrayBuffer()));
      // Object keys keep insertion order for names like these, so the list
      // comes out in the order the server wrote it: MIDI, MusicXML, full score,
      // then the parts.
      const files = Object.entries(unpacked).map(([name, bytes]) => ({
        name,
        // Unpacked out of an ArrayBuffer, so never the SharedArrayBuffer that
        // fflate's looser return type leaves open (and that Blob rejects).
        bytes: bytes as Uint8Array<ArrayBuffer>,
      }));
      if (files.length === 0) throw new Error("the server returned an empty archive");
      setSheets({ files, zipBlob, zipFilename: stem() + "_sheets.zip" });
      setSheetsOpen(true);
    } catch (e) {
      track("download_error", { format: "sheets", message: (e as Error).message });
      alert("Couldn't engrave the sheet music: " + (e as Error).message);
    } finally {
      setBusy(null);
    }
  }

  const menuItem =
    "block w-full rounded-none text-left text-[13px] font-normal text-content hover:bg-[#20212b]";

  return (
    <div className="col-span-full flex flex-wrap items-center gap-3 rounded-card border border-line bg-surface px-3.5 py-3">
      {transcribing && (
        <div className="flex min-w-48 flex-1 items-center gap-3">
          <div className="h-1 flex-1 overflow-hidden rounded-full bg-bg">
            <div
              ref={progressFillRef}
              className="h-full rounded-full bg-accent"
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
            className="relative inline-flex items-center gap-2 overflow-hidden"
            disabled={!ready || busy !== null}
            onClick={() => setMenuOpen((o) => !o)}
            aria-haspopup="menu"
            aria-expanded={menuOpen}
          >
            <IconDownload />
            {busy ?? "Download"}
            <IconChevron
              className={clsx("transition-transform", menuOpen && "rotate-180")}
            />
            {/* The server renders these in one shot and reports nothing along
                the way, so this bar is honestly fake: it exists to show the
                click registered and something is still running. Keyed on the
                label so a second export restarts it from zero. */}
            {busy !== null && (
              <span
                key={busy}
                aria-hidden
                className="absolute inset-x-0 bottom-0 h-0.5 animate-creep rounded-full bg-white/70"
              />
            )}
          </Button>
          {menuOpen && (
            <div
              role="menu"
              className="absolute left-0 z-20 mt-1.5 min-w-48 overflow-hidden rounded-md border border-line-strong bg-surface-2 py-1 shadow-pop"
            >
              <Button
                kind="ghost"
                pad="px-3 py-2"
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
                pad="px-3 py-2"
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
                pad="px-3 py-2"
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
              <Button
                kind="ghost"
                pad="px-3 py-2"
                role="menuitem"
                className={menuItem}
                title="Engraved notation: PDFs per instrument, tablature, MusicXML"
                onClick={() => {
                  setMenuOpen(false);
                  downloadSheets();
                }}
              >
                Sheet music
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
          Transcribe another file
        </Button>
      </div>

      {sheets && sheetsOpen && (
        <SheetsDialog
          files={sheets.files}
          zipBlob={sheets.zipBlob}
          zipFilename={sheets.zipFilename}
          quantized={result?.quantizedMidi != null}
          onClose={() => setSheetsOpen(false)}
        />
      )}
    </div>
  );
}
