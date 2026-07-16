import { useEffect, useRef, type Dispatch, type RefObject, type SetStateAction } from "react";
import type { AudioEngine } from "../audio";
import { PianoRoll, KEY_WIDTH } from "../pianoroll";

export function PianoRollCanvas(props: {
  rollRef: RefObject<PianoRoll | null>;
  audio: AudioEngine;
  setUserScrolled: Dispatch<SetStateAction<boolean>>;
}) {
  const { rollRef, audio, setUserScrolled } = props;
  const canvasRef = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    const canvas = canvasRef.current!;
    const roll = new PianoRoll(canvas);
    rollRef.current = roll;

    // Wheel bindings follow the DAW convention (Ableton / FL / Reaper):
    //   ctrl/cmd + scroll → zoom the time axis (also what a macOS trackpad
    //                       pinch fires, so pinch zooms the timeline);
    //   alt/option + scroll → zoom the pitch axis;
    //   shift + scroll → pan the time axis;
    //   plain scroll → vertical wheel pans the pitch axis, horizontal wheel
    //                  (trackpad) pans time.
    // (Pitch zoom via the key strip is the click-drag gesture below.)
    const onWheel = (e: WheelEvent) => {
      e.preventDefault();
      const rect = canvas.getBoundingClientRect();
      const factor = Math.exp(-e.deltaY * 0.01);
      if (e.altKey) {
        roll.zoomPitch(factor, e.clientY - rect.top);
        setUserScrolled(true);
        return;
      }
      if (e.ctrlKey || e.metaKey) {
        roll.zoomTime(factor, e.clientX - rect.left);
        setUserScrolled(true);
        return;
      }
      if (e.shiftKey) {
        const d = e.deltaX !== 0 ? e.deltaX : e.deltaY;
        if (d !== 0) roll.scrollBy(d / roll.pxPerSec);
      } else {
        if (e.deltaX !== 0) roll.scrollBy(e.deltaX / roll.pxPerSec);
        if (e.deltaY !== 0) roll.scrollPitchBy(-e.deltaY);
      }
      setUserScrolled(true);
    };

    // Scrubbing: press-and-drag in the note area moves the playhead live under
    // the cursor (with follow mode on, the view scrolls along). Playback pauses
    // for the duration of the drag and resumes from the release point. The
    // move/up handlers live on the window so the scrub survives the pointer
    // leaving the canvas — dragging out past the left edge clamps to the start.
    // A plain click is just a zero-length scrub, so click-to-seek still works.
    let scrub: { wasPlaying: boolean } | null = null;
    const scrubTo = (clientX: number) => {
      const rect = canvas.getBoundingClientRect();
      // Leaving the canvas on the left means "to the start" — a stationary
      // cursor gets no more mousemove events, so it could never out-scroll
      // the view to reach 0 otherwise.
      const t =
        clientX < rect.left ? 0 : Math.max(0, roll.xToSeconds(clientX - rect.left));
      // Clock-only seek per move (a full seek() re-schedules every note); the
      // rAF loop reads it back into the roll's playhead each frame.
      audio.scrubTo(t);
      roll.setPlayhead(t);
    };

    // Ableton-style key-strip gesture: click-drag on the keyboard zooms the
    // pitch axis horizontally (drag right = zoom in) and pans it vertically
    // (grab-style). Anchored on the cursor's pitch.
    let keyDrag: { x: number; y: number } | null = null;
    const onMouseDown = (e: MouseEvent) => {
      const rect = canvas.getBoundingClientRect();
      const x = e.clientX - rect.left;
      if (x >= KEY_WIDTH) {
        scrub = { wasPlaying: audio.state === "started" };
        if (scrub.wasPlaying) audio.pause();
        scrubTo(e.clientX);
      } else {
        keyDrag = { x, y: e.clientY - rect.top };
      }
      e.preventDefault();
    };
    const onMouseMove = (e: MouseEvent) => {
      if (scrub) {
        scrubTo(e.clientX);
        return;
      }
      const rect = canvas.getBoundingClientRect();
      if (keyDrag) {
        const x = e.clientX - rect.left;
        const y = e.clientY - rect.top;
        const dx = x - keyDrag.x;
        const dy = y - keyDrag.y;
        if (dx !== 0) roll.zoomPitch(Math.exp(dx * 0.01), y);
        if (dy !== 0) roll.scrollPitchBy(dy);
        keyDrag = { x, y };
        setUserScrolled(true);
        return;
      }
      // Hint the gesture with a resize cursor while hovering the key strip.
      canvas.style.cursor =
        e.clientX - rect.left < KEY_WIDTH ? "ew-resize" : "default";
    };
    const onMouseUp = () => {
      if (scrub) {
        // One real seek to rebuild the (one-shot) note schedule from the
        // release point, then resume if the drag interrupted playback.
        audio.seek(audio.seconds);
        if (scrub.wasPlaying) audio.play();
        scrub = null;
      }
      keyDrag = null;
    };

    // Touch: one finger drags the view (left/right scrolls time, up/down pans
    // the pitch axis when zoomed); two fingers pinch to zoom — horizontal
    // spread zooms time, vertical spread zooms the pitch axis.
    let pan: { x: number; y: number } | null = null;
    let pinch: { dx: number; dy: number } | null = null;
    const pinchSpan = (t: TouchList) => ({
      dx: Math.abs(t[0].clientX - t[1].clientX),
      dy: Math.abs(t[0].clientY - t[1].clientY),
    });

    const onTouchStart = (e: TouchEvent) => {
      if (e.touches.length === 2) {
        pinch = pinchSpan(e.touches);
        pan = null;
        e.preventDefault();
      } else if (e.touches.length === 1) {
        pan = { x: e.touches[0].clientX, y: e.touches[0].clientY };
      }
    };

    const onTouchMove = (e: TouchEvent) => {
      const rect = canvas.getBoundingClientRect();
      if (pinch && e.touches.length === 2) {
        e.preventDefault();
        const span = pinchSpan(e.touches);
        const cx = (e.touches[0].clientX + e.touches[1].clientX) / 2 - rect.left;
        const cy = (e.touches[0].clientY + e.touches[1].clientY) / 2 - rect.top;
        const MIN = 12; // ignore tiny spans where the ratio gets noisy
        if (pinch.dx > MIN && span.dx > MIN) roll.zoomTime(span.dx / pinch.dx, cx);
        if (pinch.dy > MIN && span.dy > MIN) roll.zoomPitch(span.dy / pinch.dy, cy);
        pinch = span;
        setUserScrolled(true);
      } else if (pan && e.touches.length === 1) {
        e.preventDefault();
        const t = e.touches[0];
        roll.scrollBy(-(t.clientX - pan.x) / roll.pxPerSec);
        roll.scrollPitchBy(t.clientY - pan.y);
        pan = { x: t.clientX, y: t.clientY };
        setUserScrolled(true);
      }
    };

    const onTouchEnd = (e: TouchEvent) => {
      if (e.touches.length < 2) pinch = null;
      if (e.touches.length === 0) pan = null;
    };

    canvas.addEventListener("wheel", onWheel, { passive: false });
    canvas.addEventListener("touchstart", onTouchStart, { passive: false });
    canvas.addEventListener("touchmove", onTouchMove, { passive: false });
    canvas.addEventListener("touchend", onTouchEnd, { passive: false });
    canvas.addEventListener("mousedown", onMouseDown);
    window.addEventListener("mousemove", onMouseMove);
    window.addEventListener("mouseup", onMouseUp);
    return () => {
      canvas.removeEventListener("wheel", onWheel);
      canvas.removeEventListener("touchstart", onTouchStart);
      canvas.removeEventListener("touchmove", onTouchMove);
      canvas.removeEventListener("touchend", onTouchEnd);
      canvas.removeEventListener("mousedown", onMouseDown);
      window.removeEventListener("mousemove", onMouseMove);
      window.removeEventListener("mouseup", onMouseUp);
    };
  }, [audio, rollRef, setUserScrolled]);

  return (
    <section className="relative col-start-1 overflow-hidden rounded-card border border-line bg-[linear-gradient(180deg,rgba(255,255,255,0.02),transparent_60px),#0a0b0e] p-0 shadow-canvas animate-rise [animation-delay:0.12s]">
      <canvas className="block h-[420px] w-full" width={1200} height={400} ref={canvasRef} />
    </section>
  );
}
