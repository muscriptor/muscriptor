import { useEffect, useState } from "react";

/** Undo depth. Each entry is a whole decoded buffer — around 100 MB for five
 *  minutes of stereo — so the stacks stay bounded. */
const MAX_HISTORY = 20;

function push(stack: AudioBuffer[], buffer: AudioBuffer): AudioBuffer[] {
  return [...stack, buffer].slice(-MAX_HISTORY);
}

function isTyping(target: EventTarget | null): boolean {
  const el = target as HTMLElement | null;
  return (
    el?.isContentEditable === true ||
    ["INPUT", "TEXTAREA", "SELECT"].includes(el?.tagName ?? "")
  );
}

/** Undo/redo over the edited buffer, on cmd/ctrl-Z. The buffer itself lives in
 *  the editor; this only remembers the states either side of it. */
export function useAudioHistory({
  buffer,
  file,
  showAudio,
}: {
  buffer: AudioBuffer | null;
  /** Picking another file starts a fresh history. */
  file: File;
  showAudio: (next: AudioBuffer) => void;
}) {
  const [past, setPast] = useState<AudioBuffer[]>([]);
  const [future, setFuture] = useState<AudioBuffer[]>([]);

  useEffect(() => {
    setPast([]);
    setFuture([]);
  }, [file]);

  /** Moves one state off `past` onto `future`, or the reverse. */
  function step(back: boolean): boolean {
    const [from, setFrom, setTo] = back
      ? ([past, setPast, setFuture] as const)
      : ([future, setFuture, setPast] as const);
    if (buffer === null || from.length === 0) return false;
    setFrom(from.slice(0, -1));
    setTo((to) => push(to, buffer));
    showAudio(from[from.length - 1]);
    return true;
  }

  useEffect(() => {
    const onKeyDown = (event: KeyboardEvent) => {
      if (
        (!event.metaKey && !event.ctrlKey) ||
        event.altKey ||
        event.key.toLowerCase() !== "z" ||
        isTyping(event.target)
      ) {
        return;
      }
      // Leave the browser's own undo alone when ours would be a no-op.
      if (step(!event.shiftKey)) event.preventDefault();
    };
    window.addEventListener("keydown", onKeyDown);
    return () => window.removeEventListener("keydown", onKeyDown);
  }, [buffer, past, future]);

  return function replaceAudio(next: AudioBuffer) {
    if (buffer === null) return;
    setPast((p) => push(p, buffer));
    setFuture([]);
    showAudio(next);
  };
}
