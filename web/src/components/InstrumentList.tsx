import { useEffect, useState, type RefObject } from "react";
import clsx from "clsx";
import type { AudioEngine } from "../audio";
import { Button } from "./Button";
import { instrumentColor, type PianoRoll } from "../pianoroll";
import { label } from "../instruments";
import { IconSound, IconSoundOff } from "./icons";

/** A circled "?" that reveals an explanatory tooltip on hover/focus. */
function HelpHint(props: { children: string }) {
  return (
    <span className="group/help relative ml-1.5 inline-flex align-middle">
      <span
        tabIndex={0}
        className="flex size-4 cursor-help items-center justify-center rounded-full border border-line-strong text-[10px] font-semibold text-muted outline-none transition-colors duration-150 hover:border-accent hover:text-content focus-visible:border-accent focus-visible:text-content"
        aria-label="What does this mean?"
      >
        ?
      </span>
      <span
        role="tooltip"
        className="pointer-events-none absolute right-0 top-[calc(100%+8px)] z-30 w-60 rounded-lg border border-line-strong bg-surface-2 px-3 py-2.5 text-sm font-normal leading-snug text-muted opacity-0 shadow-pop transition-opacity duration-150 group-hover/help:opacity-100 group-focus-within/help:opacity-100"
      >
        {props.children}
      </span>
    </span>
  );
}

/** A given instrument that wasn't detected: gray, struck-through, no controls. */
function UndetectedRow(props: { name: string }) {
  return (
    <li className="flex items-center gap-2.5 rounded-lg px-2.5 py-2 text-muted opacity-40 [animation:rise_0.4s_var(--ease-fluid)_both]">
      <span className="size-3 shrink-0 rounded-sm bg-faint" />
      <span className="min-w-0 flex-1 overflow-hidden text-ellipsis whitespace-nowrap line-through">
        {label(props.name)}
      </span>
      <span className="shrink-0 text-xs italic text-faint">not detected</span>
    </li>
  );
}

/** An interactive detected instrument with mute + solo controls. */
function InstrumentRow(props: {
  name: string;
  muted: boolean;
  soloed: boolean;
  onToggleMute: () => void;
  onToggleSolo: () => void;
  /** Hovering the row spotlights this instrument's notes on the piano roll. */
  onHover: (name: string | null) => void;
}) {
  const { name, muted, soloed, onToggleMute, onToggleSolo, onHover } = props;
  return (
    <li
      className="group flex items-center gap-2.5 rounded-lg px-2.5 py-2 text-muted transition-colors duration-150 ease-fluid hover:bg-white/[0.04] hover:text-content [animation:rise_0.4s_var(--ease-fluid)_both]"
      onMouseEnter={() => onHover(name)}
      onMouseLeave={() => onHover(null)}
    >
      <div
        className={clsx(
          "flex min-w-0 flex-1 items-center gap-2.5 transition-opacity duration-150 ease-fluid",
          muted && "opacity-10",
        )}
      >
        <span
          className="size-3 shrink-0 rounded-sm shadow-glow"
          style={{ background: instrumentColor(name) }}
        />
        <span className="min-w-0 flex-1 overflow-hidden text-ellipsis whitespace-nowrap">
          {label(name)}
        </span>
      </div>
      <div className="flex items-center gap-0.5">
        <Button
          type="button"
          kind="ghost"
          className={clsx(
            "-my-1 flex size-6 shrink-0 items-center justify-center rounded-md text-xs font-semibold transition-[opacity,background,color] duration-150 ease-fluid hover:bg-white/[0.08]",
            soloed
              ? "text-accent-2 opacity-100"
              : "text-muted opacity-70 group-hover:opacity-100 hover:text-content",
          )}
          title={soloed ? "Unsolo" : "Solo (mute everything else)"}
          aria-pressed={soloed}
          onClick={onToggleSolo}
        >
          S
        </Button>
        <Button
          type="button"
          kind="ghost"
          className={clsx(
            "-my-1 flex size-6 shrink-0 items-center justify-center rounded-md transition-[opacity,background,color] duration-150 ease-fluid hover:bg-white/[0.08]",
            muted
              ? "text-red opacity-100"
              : "text-muted opacity-70 group-hover:opacity-100 hover:text-content",
          )}
          title={muted ? "Unmute on MIDI track" : "Mute on MIDI track"}
          aria-pressed={muted}
          onClick={onToggleMute}
        >
          {muted ? <IconSoundOff /> : <IconSound />}
        </Button>
      </div>
    </li>
  );
}

export function InstrumentList(props: {
  instruments: string[];
  given: Set<string>;
  audio: AudioEngine;
  rollRef: RefObject<PianoRoll | null>;
}) {
  const { instruments, given, audio, rollRef } = props;
  const [muted, setMuted] = useState<Set<string>>(() => new Set());
  const [soloed, setSoloed] = useState<string | null>(null);

  // Keep the audio engine and piano roll in sync with the muted set.
  useEffect(() => {
    for (const name of instruments) {
      audio.setInstrumentMuted(name, muted.has(name));
      rollRef.current?.setInstrumentMuted(name, muted.has(name));
    }
  }, [instruments, audio, rollRef, muted]);

  const toggleMute = (name: string) => {
    setSoloed(null);
    setMuted((prev) => {
      const next = new Set(prev);
      if (next.has(name)) next.delete(name);
      else next.add(name);
      return next;
    });
  };

  const toggleSolo = (name: string) => {
    if (soloed === name) {
      setSoloed(null);
      setMuted(new Set());
    } else {
      setSoloed(name);
      setMuted(new Set(instruments.filter((other) => other !== name)));
    }
  };

  const row = (name: string) => (
    <InstrumentRow
      key={name}
      name={name}
      muted={muted.has(name)}
      soloed={soloed === name}
      onToggleMute={() => toggleMute(name)}
      onToggleSolo={() => toggleSolo(name)}
      onHover={(n) => rollRef.current?.setHighlightedInstrument(n)}
    />
  );

  const detectedSet = new Set(instruments);
  const hasGiven = given.size > 0;
  // Detected instruments that weren't in the given list.
  const extra = instruments.filter((name) => !given.has(name));

  return (
    <aside className="card col-start-2 self-start px-4 pb-5 pt-4 max-[760px]:col-start-1 animate-rise [animation-delay:0.18s]">
      {hasGiven ? (
        <>
          <h2 className="m-0 mb-3 flex items-center text-base font-semibold">
            Instruments
            <HelpHint>
              The instruments you specified. Greyed-out ones weren't detected in
              the audio.
            </HelpHint>
          </h2>
          <ul className="m-0 flex list-none flex-col gap-0.5 p-0 text-sm">
            {Array.from(given).map((name) =>
              detectedSet.has(name) ? (
                row(name)
              ) : (
                <UndetectedRow key={name} name={name} />
              ),
            )}
          </ul>
          {/* Now we don't allow non-specified instruments to appear, so probably dead code.
            * Keeping for now in case we go back/make it configurable */}
          {extra.length > 0 && (
            <>
              <h2 className="m-0 mb-3 mt-5 text-base font-semibold">
                More instruments{" "}
                <HelpHint>
                  More instruments that the model detected in the audio, even
                  without them being explicitly given.
                </HelpHint>
              </h2>
              <ul className="m-0 flex list-none flex-col gap-0.5 p-0 text-sm">
                {extra.map(row)}
              </ul>
            </>
          )}
        </>
      ) : (
        <>
          <h2 className="m-0 mb-3 text-base font-semibold">Instruments</h2>
          <ul className="m-0 flex list-none flex-col gap-0.5 p-0 text-sm">
            {instruments.map(row)}
          </ul>
        </>
      )}
    </aside>
  );
}
