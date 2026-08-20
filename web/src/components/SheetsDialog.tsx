import { useEffect, useMemo } from "react";
import { Button } from "./Button";
import { IconDownload } from "./icons";
import { track } from "../analytics";

/** One member of the engraved-sheets archive, already unpacked in memory.
 *  The buffer is narrowed to a plain ArrayBuffer because that is what Blob
 *  accepts — fflate types its output over ArrayBufferLike. */
export type SheetFile = { name: string; bytes: Uint8Array<ArrayBuffer> };

/**
 * File picker for the engraved sheet music.
 *
 * Engraving is slow and produces a whole directory, so the server renders the
 * set in one go and answers with a zip. By the time this dialog is on screen
 * the browser already holds every file — clicking a row only saves one of the
 * blobs already in memory, which is why there is no spinner and no second
 * request here. "Download all" hands over the untouched archive instead.
 *
 * Each file gets its own object URL, all revoked together when the dialog
 * closes: they must outlive the click that starts a download, and the user is
 * expected to save several before closing.
 */
export function SheetsDialog(props: {
  files: SheetFile[];
  /** The archive as it arrived, for the "download all" row. */
  zipBlob: Blob;
  zipFilename: string;
  /** Whether the notes were snapped to a beat grid before engraving. Without
   *  one, note lengths and bar lines are guesses and the score reads badly. */
  quantized: boolean;
  onClose: () => void;
}) {
  const { files, zipBlob, zipFilename, quantized, onClose } = props;

  const urls = useMemo(
    () =>
      files.map((f) => URL.createObjectURL(new Blob([f.bytes], { type: mimeOf(f.name) }))),
    [files],
  );
  const zipUrl = useMemo(() => URL.createObjectURL(zipBlob), [zipBlob]);
  useEffect(() => {
    return () => {
      urls.forEach(URL.revokeObjectURL);
      URL.revokeObjectURL(zipUrl);
    };
  }, [urls, zipUrl]);

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") onClose();
    };
    document.addEventListener("keydown", onKey);
    return () => document.removeEventListener("keydown", onKey);
  }, [onClose]);

  function save(url: string, filename: string) {
    const a = document.createElement("a");
    a.href = url;
    a.download = filename;
    a.click();
  }

  return (
    // Above the page grain (z-100), below the drag-and-drop overlay (z-200), so
    // dropping a new file still reads as leaving this behind.
    <div
      className="fixed inset-0 z-[150] flex items-center justify-center bg-[rgba(11,12,16,0.72)] p-6 backdrop-blur-sm"
      onClick={(e) => {
        if (e.target === e.currentTarget) onClose();
      }}
    >
      <div
        role="dialog"
        aria-modal="true"
        aria-label="Download sheet music"
        className="flex max-h-full w-full max-w-lg flex-col overflow-hidden rounded-card border border-line-strong bg-surface shadow-overlay"
      >
        <div className="border-b border-line px-5 py-4">
          <h2 className="m-0 text-base font-semibold text-content">Download sheet music</h2>
        </div>

        {!quantized && (
          <div className="border-b border-line bg-accent-2/10 px-5 py-3 text-[13px] text-accent-2">
            We couldn't find a steady beat in this recording, so the bar lines are guesses.
            The score may be hard to read.
          </div>
        )}

        <div className="min-h-0 flex-1 overflow-y-auto px-2 py-2">
          {files.map((file, i) => (
            <Button
              key={file.name}
              kind="ghost"
              pad="px-3 py-2.5"
              className="flex w-full items-center gap-3 rounded-md text-left font-normal hover:bg-[#20212b]"
              onClick={() => {
                track("download", {
                  format: "sheets_file",
                  file_type: extensionOf(file.name),
                  sheet_kind: kindOf(file.name),
                });
                save(urls[i], file.name);
              }}
            >
              <span className="shrink-0 text-faint">
                <IconDownload />
              </span>
              <span className="min-w-0 flex-1">
                <span className="block truncate text-[13px] text-content">
                  {describe(file.name)}
                </span>
                <span className="block truncate font-mono text-[11px] text-faint">
                  {file.name}
                </span>
              </span>
              <span className="shrink-0 font-mono text-[11px] tabular-nums text-faint">
                {formatSize(file.bytes.length)}
              </span>
            </Button>
          ))}
        </div>

        <div className="flex items-center justify-between gap-3 border-t border-line px-5 py-3.5">
          <Button
            kind="primary"
            className="inline-flex items-center gap-2"
            onClick={() => {
              track("download", { format: "sheets_zip" });
              save(zipUrl, zipFilename);
            }}
          >
            <IconDownload />
            Download all ({files.length} files, {formatSize(zipBlob.size)})
          </Button>
          <Button onClick={onClose}>Close</Button>
        </div>
      </div>
    </div>
  );
}

function extensionOf(name: string): string {
  return name.split(".").pop()?.toLowerCase() ?? "";
}

/** Low-cardinality label for analytics: which kind of sheet was saved. */
function kindOf(name: string): string {
  if (name === "full_score.pdf") return "full_score";
  const part = name.match(/^\d+_.+?(_tab)?\.pdf$/);
  if (part) return part[1] ? "part_tab" : "part";
  return extensionOf(name);
}

// Content types for what the server engraves, so a saved file opens in the
// right application instead of arriving as an anonymous blob.
const MIME: Record<string, string> = {
  pdf: "application/pdf",
  musicxml: "application/vnd.recordare.musicxml+xml",
  mid: "audio/midi",
};

function mimeOf(name: string): string {
  return MIME[extensionOf(name)] ?? "application/octet-stream";
}

/**
 * A readable label for one of write_sheets' filenames — `full_score.pdf`,
 * `score.musicxml`, or a numbered part like `02_electric_bass_tab.pdf`. Falls
 * back to the filename itself, so an unrecognized member still lists sensibly.
 */
function describe(name: string): string {
  if (name === "score.mid") return "MIDI file";
  if (name === "score.musicxml") return "MusicXML score";
  if (name === "full_score.pdf") return "Full score - every instrument";
  const part = name.match(/^\d+_(.+?)(_tab)?\.pdf$/);
  if (!part) return name;
  const instrument = part[1].replace(/_/g, " ");
  return instrument.charAt(0).toUpperCase() + instrument.slice(1) + (part[2] ? " - tablature" : "");
}

function formatSize(bytes: number): string {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${Math.round(bytes / 1024)} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
}
