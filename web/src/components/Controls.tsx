import { useEffect, useState, type RefObject } from "react";
import clsx from "clsx";
import type { AudioEngine } from "../audio";
import { Button } from "./Button";
import { IconPlay, IconPause } from "./icons";

export function Controls(props: {
  audio: AudioEngine;
  /** Attached to the time clock; updated imperatively by the rAF loop. */
  clockRef: RefObject<HTMLSpanElement | null>;
  mix: number;
  onMixChange: (v: number) => void;
  stereo: boolean;
  onStereoChange: (v: boolean) => void;
  /** Whether the roll auto-follows the playhead (toggled off by manual scrolling). */
  following: boolean;
  onToggleFollow: () => void;
}) {
  const { audio, clockRef, mix, onMixChange, stereo, onStereoChange, following, onToggleFollow } =
    props;
  // The transport's state isn't React state (and it can auto-stop at the end),
  // so poll it each frame to keep the toggle button's label in sync.
  const [playing, setPlaying] = useState(false);
  useEffect(() => {
    let raf = 0;
    const tick = () => {
      setPlaying(audio.state === "started");
      raf = requestAnimationFrame(tick);
    };
    raf = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(raf);
  }, [audio]);

  return (
    <div className="col-span-full flex flex-wrap items-center gap-2.5 rounded-card border border-line bg-surface px-3.5 py-3 max-[760px]:border-0 max-[760px]:bg-transparent max-[760px]:p-0 animate-rise [animation-delay:0.06s]">
      <Button
        className={clsx(
          "inline-flex items-center gap-2",
          playing && "shadow-[0_0_10px_var(--color-accent)]",
        )}
        onClick={(e) => {
          e.currentTarget.blur();
          playing ? audio.pause() : audio.play();
        }}
      >
        {playing ? <IconPause /> : <IconPlay />}
        {playing ? "Pause" : "Play"}
      </Button>
      <Button
        kind={following ? "square" : "squareOff"}
        aria-pressed={following}
        title={following ? "Stop following the playhead" : "Scroll along with the playhead"}
        onClick={(e) => {
          e.currentTarget.blur();
          onToggleFollow();
        }}
      >
        Follow playhead
      </Button>
      <span
        className="rounded-md border border-line bg-bg px-2.5 py-1 font-mono text-sm tabular-nums text-muted"
        ref={clockRef}
      >
        0.0s
      </span>
      {/* Mix slider + stereo toggle share one non-wrapping row; on mobile the
          row takes the full width and the slider shrinks to whatever is left. */}
      <div className="ml-auto flex items-center max-[760px]:w-full">
        <label
          className={clsx(
            "inline-flex items-center gap-2.5 text-sm text-muted max-[760px]:min-w-0 max-[760px]:flex-1",
            stereo && "opacity-40",
          )}
        >
          <span
            className={clsx(
              "min-w-8 text-center transition-colors",
              !stereo && "cursor-pointer hover:text-content",
            )}
            onClick={() => !stereo && onMixChange(0)}
          >
            Original
          </span>
          <input
            className="mix-slider"
            type="range"
            min="0"
            max="1"
            step="0.01"
            value={mix}
            disabled={stereo}
            onChange={(e) => onMixChange(parseFloat(e.target.value))}
            // Drop focus once the drag ends so Space keeps toggling play/pause
            // (the global handler ignores Space while an input is focused).
            // Range inputs implicitly capture the pointer, so this fires even
            // when the drag is released outside the slider.
            onPointerUp={(e) => e.currentTarget.blur()}
            // Clicks on the label's Original/MIDI spans focus the slider via a
            // forwarded click with no pointer event, so blur on click too.
            onClick={(e) => e.currentTarget.blur()}
          />
          <span
            className={clsx(
              "min-w-8 text-center transition-colors",
              !stereo && "cursor-pointer hover:text-content",
            )}
            onClick={() => !stereo && onMixChange(1)}
          >
            MIDI
          </span>
        </label>
        <label className="inline-flex shrink-0 cursor-pointer select-none items-center gap-1.5 text-sm text-muted px-3 max-[760px]:pr-0">
          <input
            className="cursor-pointer accent-accent"
            type="checkbox"
            checked={stereo}
            onChange={(e) => onStereoChange(e.target.checked)}
            // Same as the slider: keep Space bound to play/pause after clicking.
            // click (not pointerup) also catches clicks on the wrapping label,
            // which the browser forwards to the checkbox.
            onClick={(e) => e.currentTarget.blur()}
          />
          <span>Stereo</span>
        </label>
      </div>
    </div>
  );
}
