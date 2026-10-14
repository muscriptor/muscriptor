import { useEffect, useState } from "react";
import clsx from "clsx";
import { Button } from "./Button";
import { IconDownload } from "./icons";
import { track } from "../analytics";
import type { TranscriptionResult } from "../hooks/useTranscription";
import {
  correctedGrid,
  DEFAULT_CLEANUP,
  exportMidi,
  isRetimed,
  saveBlob,
  type NoteCleanup,
  type TimingOverrides,
} from "../export";

const STORAGE_KEY = "midiExportCleanup";

/** Steps per beat, as note values. */
const GRIDS: { value: number; label: string }[] = [
  { value: 1, label: "1/4" },
  { value: 2, label: "1/8" },
  { value: 3, label: "1/8 triplets" },
  { value: 4, label: "1/16" },
  { value: 6, label: "1/16 triplets" },
  { value: 8, label: "1/32" },
];
const gridLabel = (steps: number) =>
  GRIDS.find((g) => g.value === steps)?.label ?? `1/${steps} beat`;

/** What a typed-in tempo quantizes to when there is no measured subdivision. */
const FALLBACK_GRID = 4;

/** Longest rest to fill, in beats. */
const GAP_FILLS: { value: number; label: string }[] = [
  { value: 0, label: "Keep every rest" },
  { value: 0.25, label: "Fill rests up to a 1/16" },
  { value: 0.5, label: "Fill rests up to a 1/8" },
  { value: 1, label: "Fill rests up to a beat" },
];

function loadCleanup(): NoteCleanup {
  try {
    const saved = localStorage.getItem(STORAGE_KEY);
    if (saved) return { ...DEFAULT_CLEANUP, ...JSON.parse(saved) };
  } catch {
    // unavailable or corrupt — fall through to the defaults
  }
  return DEFAULT_CLEANUP;
}

function saveCleanup(cleanup: NoteCleanup) {
  try {
    localStorage.setItem(STORAGE_KEY, JSON.stringify(cleanup));
  } catch {
    // a remembered preference is a convenience; nothing to do without it
  }
}

const SELECT =
  "rounded-md border border-line-strong bg-surface-2 px-2 py-1 text-[13px] text-content outline-none focus:border-accent disabled:opacity-50";

/**
 * Options for the MIDI download: as played, or quantized onto the beat grid
 * (to what resolution, how firmly), whether to tie notes across the short rests
 * a transcription leaves between them, and whether to drop the empty bars
 * before the first note. Notation software such as Guitar Pro writes every few
 * milliseconds of slop as a rest, which is what most of these are for.
 *
 * The tempo, meter and bar lines come from the Tempo & bars panel. The
 * cleanup choices are remembered across visits.
 */
export function MidiExportDialog(props: {
  result: TranscriptionResult;
  timing: TimingOverrides;
  onClose: () => void;
}) {
  const { result, timing, onClose } = props;
  const [cleanup, setCleanup] = useState(loadCleanup);
  const [busy, setBusy] = useState(false);
  const grid = correctedGrid(result.beatGrid, timing);
  // The measured subdivision is only good for the tempo it was measured at.
  const autoGrid = isRetimed(result.beatGrid, timing)
    ? null
    : (result.beatGrid?.beat_subdivision ?? null);
  const subdivision = cleanup.subdivision ?? autoGrid ?? FALLBACK_GRID;
  const canQuantize = grid !== null;
  const quantize = cleanup.quantize && canQuantize;

  const update = (patch: Partial<NoteCleanup>) => {
    const next = { ...cleanup, ...patch };
    setCleanup(next);
    saveCleanup(next);
  };

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") onClose();
    };
    document.addEventListener("keydown", onKey);
    return () => document.removeEventListener("keydown", onKey);
  }, [onClose]);

  async function download() {
    const options: NoteCleanup = {
      ...cleanup,
      quantize,
      // "Auto" only when there is a measured subdivision to fall back on.
      subdivision: cleanup.subdivision ?? (autoGrid === null ? FALLBACK_GRID : null),
    };
    track("download", {
      format: "midi",
      quantize: options.quantize,
      grid: options.quantize ? subdivision : undefined,
      strength: options.quantize ? Math.round(options.strength * 100) : undefined,
      fill_gaps: options.fill_gaps,
      trim_leading_bars: options.trim_leading_bars,
      tempo_changed: timing.bpm !== null,
      tempo_map: grid?.steady === false,
      beat_scale: timing.beat_scale,
      meter_changed: timing.beats_per_bar !== null,
      downbeat_changed: timing.first_downbeat !== null,
    });
    setBusy(true);
    try {
      const midi = await exportMidi(result.notes, result.beatGrid, timing, options);
      saveBlob(midi, result.filename);
      onClose();
    } catch (e) {
      track("download_error", { format: "midi", message: (e as Error).message });
      alert("Couldn't create the MIDI file: " + (e as Error).message);
    } finally {
      setBusy(false);
    }
  }

  const segment = (active: boolean) =>
    clsx(
      "flex-1 rounded-md px-3 py-1.5 text-[13px]",
      active ? "bg-surface-2 text-content shadow-pop" : "text-muted hover:text-content",
    );

  return (
    // Same layering as SheetsDialog: above the page grain, below the drop overlay.
    <div
      className="fixed inset-0 z-[150] flex items-center justify-center bg-[rgba(11,12,16,0.72)] p-6 backdrop-blur-sm"
      onClick={(e) => {
        if (e.target === e.currentTarget) onClose();
      }}
    >
      <div
        role="dialog"
        aria-modal="true"
        aria-label="Download MIDI"
        className="flex max-h-full w-full max-w-md flex-col overflow-y-auto rounded-card border border-line-strong bg-surface shadow-overlay"
      >
        <div className="border-b border-line px-5 py-4">
          <h2 className="m-0 text-base font-semibold text-content">Download MIDI</h2>
          <p className="m-0 mt-1 text-xs text-faint">
            {grid
              ? `${grid.steady === false ? `Following the performance, ~${grid.bpm.toFixed(0)}` : grid.bpm.toFixed(1)} BPM${
                  grid.beats_per_bar ? `, ${grid.beats_per_bar}/4` : ""
                } — change it under Tempo & bars.`
              : "No tempo: written at 120 BPM. Set one under Tempo & bars."}
          </p>
        </div>

        <div className="flex flex-col gap-5 px-5 py-4 text-sm text-muted">
          <section>
            <h3 className="m-0 mb-2 text-[13px] font-semibold text-content">Timing</h3>
            <div className="flex gap-1 rounded-lg border border-line bg-bg p-1">
              <Button
                kind="ghost"
                className={segment(!quantize)}
                onClick={() => update({ quantize: false })}
              >
                As played
              </Button>
              <Button
                kind="ghost"
                className={clsx(segment(quantize), !canQuantize && "opacity-40")}
                disabled={!canQuantize}
                onClick={() => update({ quantize: true })}
              >
                Quantized
              </Button>
            </div>
            <p className="m-0 mt-1.5 text-xs text-faint">
              {quantize
                ? "Moves notes onto the grid. Best for notation software and editing."
                : "Keeps the timing of the performance, exactly as you hear it here."}
            </p>

            {quantize && (
              <div className="mt-3 flex flex-col gap-2.5">
                <label className="flex items-center justify-between gap-3">
                  Grid
                  <select
                    className={SELECT}
                    value={cleanup.subdivision ?? ""}
                    onChange={(e) =>
                      update({
                        subdivision: e.target.value === "" ? null : Number(e.target.value),
                      })
                    }
                  >
                    <option value="">
                      Auto ({gridLabel(autoGrid ?? FALLBACK_GRID)})
                    </option>
                    {GRIDS.map((g) => (
                      <option key={g.value} value={g.value}>
                        {g.label}
                      </option>
                    ))}
                  </select>
                </label>
                <label className="flex items-center justify-between gap-3">
                  Strength
                  <span className="flex items-center gap-2">
                    <input
                      type="range"
                      min={0}
                      max={100}
                      step={5}
                      value={Math.round(cleanup.strength * 100)}
                      onChange={(e) => update({ strength: Number(e.target.value) / 100 })}
                      className="w-32 accent-[var(--color-accent)]"
                    />
                    <span className="w-9 text-right font-mono text-xs tabular-nums">
                      {Math.round(cleanup.strength * 100)}%
                    </span>
                  </span>
                </label>
              </div>
            )}
          </section>

          <section>
            <h3 className="m-0 mb-2 text-[13px] font-semibold text-content">Rests</h3>
            <select
              className={SELECT + " w-full"}
              value={cleanup.fill_gaps}
              onChange={(e) => update({ fill_gaps: Number(e.target.value) })}
            >
              {GAP_FILLS.map((g) => (
                <option key={g.value} value={g.value}>
                  {g.label}
                </option>
              ))}
            </select>
            <p className="m-0 mt-1.5 text-xs text-faint">
              Holds a note until the next one instead of leaving a short rest
              between them.
            </p>
            <label className="mt-3 flex items-start gap-2">
              <input
                type="checkbox"
                className="mt-0.5 accent-[var(--color-accent)]"
                checked={cleanup.trim_leading_bars}
                onChange={(e) => update({ trim_leading_bars: e.target.checked })}
              />
              <span>
                Skip empty bars at the start
                <span className="block text-xs text-faint">
                  The MIDI then no longer lines up with the audio from 0:00.
                </span>
              </span>
            </label>
          </section>
        </div>

        <div className="flex items-center justify-between gap-3 border-t border-line px-5 py-3.5">
          <Button
            kind="primary"
            className="inline-flex items-center gap-2"
            disabled={busy}
            onClick={download}
          >
            <IconDownload />
            {busy ? "Writing…" : "Download .mid"}
          </Button>
          <Button onClick={onClose}>Cancel</Button>
        </div>
      </div>
    </div>
  );
}
