import { useEffect, useState } from "react";
import { Button } from "./Button";
import type { BeatGrid } from "../pianoroll";
import clsx from "clsx";
import {
  correctedGrid,
  followsBeats,
  NO_OVERRIDES,
  type TimingOverrides,
} from "../export";

const FIELD =
  "w-20 rounded-md border border-line-strong bg-surface-2 px-2 py-1 text-right font-mono text-[13px] tabular-nums text-content outline-none focus:border-accent";
const SMALL_BUTTON = "px-2 py-1 text-xs";

/** Meters the MIDI writer can express: n/4 only. */
const METERS = [2, 3, 4, 5, 6, 7];

/**
 * A number input that only reports a value once it is committed (blur or
 * Enter), so typing "1", "12", "120" doesn't redraw the roll at 1 and 12 BPM on
 * the way. Shows `value` again whenever it changes from outside.
 */
function NumberField(props: {
  value: number | null;
  digits: number;
  min: number;
  max: number;
  label: string;
  onCommit: (value: number) => void;
}) {
  const { value, digits, min, max, label, onCommit } = props;
  const shown = value === null ? "" : value.toFixed(digits);
  const [text, setText] = useState(shown);
  useEffect(() => setText(shown), [shown]);

  function commit() {
    const parsed = Number.parseFloat(text);
    if (Number.isFinite(parsed) && parsed >= min && parsed <= max) onCommit(parsed);
    else setText(shown);
  }

  return (
    <input
      type="text"
      inputMode="decimal"
      aria-label={label}
      className={FIELD}
      value={text}
      placeholder="—"
      onChange={(e) => setText(e.target.value)}
      onBlur={commit}
      onKeyDown={(e) => {
        if (e.key === "Enter") e.currentTarget.blur();
        if (e.key === "Escape") {
          setText(shown);
          e.currentTarget.blur();
        }
      }}
    />
  );
}

/**
 * Sidebar card for correcting the beat grid the export is written on: whether
 * bar lines follow the performance (a tempo map on the tracked beats) or one
 * fixed tempo, the tempo (with ×2 / ÷2 for a tracker that locked onto half or
 * double time), the meter, and where a bar line falls. The roll redraws its
 * bar lines as these change; the notes never move, since a tempo only says
 * where the bars are.
 */
export function TimingPanel(props: {
  /** The detected grid, null when no beats were found. */
  detected: BeatGrid | null;
  timing: TimingOverrides;
  onChange: (timing: TimingOverrides) => void;
  /** Current playback position, for "bar line at the playhead". */
  getPlayhead: () => number;
}) {
  const { detected, timing, onChange, getPlayhead } = props;
  const grid = correctedGrid(detected, timing);
  const follow = followsBeats(detected, timing);
  const canFollow = !!detected?.beats && detected.beats.length >= 2;
  const changed = JSON.stringify(timing) !== JSON.stringify(NO_OVERRIDES);
  const set = (patch: Partial<TimingOverrides>) => onChange({ ...timing, ...patch });
  // A typed tempo is scaled as it is; otherwise the detected beats are, which
  // also re-reads the meter (4/4 twice as fast is 8/4).
  const scaleTempo = (factor: number) => {
    if (timing.bpm !== null) {
      set({ bpm: Math.min(400, Math.max(20, timing.bpm * factor)) });
    } else {
      set({ beat_scale: Math.min(4, Math.max(0.25, timing.beat_scale * factor)) });
    }
  };
  const segment = (active: boolean) =>
    clsx(
      "flex-1 rounded-md px-2 py-1 text-xs",
      active ? "bg-surface-2 text-content shadow-pop" : "text-muted hover:text-content",
    );

  return (
    <section className="card px-4 pb-4 pt-4 animate-rise [animation-delay:0.24s]">
      <h2 className="m-0 mb-1 text-base font-semibold">Tempo &amp; bars</h2>
      <p className="m-0 mb-3 text-xs leading-snug text-faint">
        {!detected
          ? "No beat detected. Type a tempo in to get bar lines and quantized exports."
          : detected.steady === false
            ? `The tempo varies (around ${detected.bpm.toFixed(0)} BPM), so the bar lines follow the performance. Correct them here if they're off — exports follow.`
            : `Detected ${detected.bpm.toFixed(1)} BPM${
                detected.beats_per_bar ? `, ${detected.beats_per_bar}/4` : ""
              }. Correct it here if the bar lines are off — exports follow.`}
      </p>

      {canFollow && (
        <div className="mb-3 flex gap-1 rounded-lg border border-line bg-bg p-1">
          <Button
            kind="ghost"
            className={segment(follow)}
            title="A tempo change on every beat, so bar lines stay on the music when it speeds up or slows down"
            onClick={() => set({ tempo_map: true, bpm: null })}
          >
            Follow the performance
          </Button>
          <Button
            kind="ghost"
            className={segment(!follow)}
            title="One tempo for the whole file, for apps that ignore tempo changes"
            onClick={() => set({ tempo_map: false })}
          >
            One fixed tempo
          </Button>
        </div>
      )}

      <div className="flex flex-col gap-2.5 text-sm text-muted">
        <label className="flex items-center gap-2">
          <span className="w-16 shrink-0">Tempo</span>
          {follow ? (
            <span
              className="w-20 text-right font-mono text-[13px] tabular-nums text-content"
              title="Average tempo; the exported MIDI changes tempo with the music"
            >
              ~{grid?.bpm.toFixed(0)}
            </span>
          ) : (
            <NumberField
              label="Tempo in BPM"
              value={grid?.bpm ?? null}
              digits={1}
              min={20}
              max={400}
              onCommit={(bpm) => set({ bpm, tempo_map: false })}
            />
          )}
          <span className="text-xs text-faint">BPM</span>
          <span className="ml-auto flex gap-1">
            <Button
              pad={SMALL_BUTTON}
              disabled={!grid}
              title="Half the tempo: the beat tracker counted every eighth note"
              onClick={() => scaleTempo(0.5)}
            >
              ÷2
            </Button>
            <Button
              pad={SMALL_BUTTON}
              disabled={!grid}
              title="Double the tempo: the beat tracker counted every half note"
              onClick={() => scaleTempo(2)}
            >
              ×2
            </Button>
          </span>
        </label>

        <label className="flex items-center gap-2">
          <span className="w-16 shrink-0">Meter</span>
          <select
            aria-label="Beats per bar"
            className={FIELD + " w-auto text-left"}
            disabled={!grid}
            value={grid?.beats_per_bar ?? ""}
            onChange={(e) => set({ beats_per_bar: Number(e.target.value) })}
          >
            {grid?.beats_per_bar == null && <option value="">unknown</option>}
            {METERS.map((n) => (
              <option key={n} value={n}>
                {n}/4
              </option>
            ))}
          </select>
        </label>

        <label className="flex items-center gap-2">
          <span className="w-16 shrink-0">Bar line</span>
          <NumberField
            label="A bar line, in seconds"
            value={grid?.first_downbeat ?? null}
            digits={2}
            min={0}
            max={24 * 3600}
            onCommit={(first_downbeat) => set({ first_downbeat })}
          />
          <span className="text-xs text-faint">s</span>
          <Button
            pad={SMALL_BUTTON}
            className="ml-auto"
            disabled={!grid}
            title="Put a bar line where playback is: pause on the downbeat, then click"
            onClick={() => set({ first_downbeat: getPlayhead() })}
          >
            At playhead
          </Button>
        </label>
      </div>

      {changed && (
        <Button
          kind="ghost"
          className="mt-3 text-xs text-faint underline hover:text-content"
          onClick={() => onChange(NO_OVERRIDES)}
        >
          {detected ? "Back to the detected tempo" : "Clear"}
        </Button>
      )}
    </section>
  );
}
