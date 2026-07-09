export interface RollNote {
  pitch: number;
  start: number;
  end: number;
  instrument: string;
  /**
   * `performance.now()` at which this note's geometry was finalized (its `end`
   * event arrived), driving the left-to-right reveal. Unset notes are drawn
   * with no reveal (full width), e.g. when loading a finished MIDI.
   */
  spawn?: number;
  /**
   * Number of same-pitch notes drawn in front of this one that overlap it in
   * time. The note is nudged up by this many steps so it peeks out from behind
   * them. Maintained in {@link PianoRoll.setNoteEnd}.
   */
  stackOffset?: number;
}

/** Reveal duration in ms — how long a note takes to grow to full width. */
const REVEAL_MS = 280;

/** Shave this off each note's drawn width so adjacent notes keep a visible gap. */
const NOTE_GAP_PX = 1.5;

/** Upward nudge per overlapping same-pitch note drawn in front, so it peeks out. */
const NOTE_STACK_PX = 1.5;

/** Ease-out cubic: fast start, slowing as it reaches the end. */
function easeOut(t: number): number {
  const u = 1 - t;
  return 1 - u * u * u;
}

const PX_PER_SEC = 80;
const MIN_PX_PER_SEC = 10;
const MAX_PX_PER_SEC = 2000;
const MAX_PX_PER_PITCH = 160;
const PITCH_MIN = 21; // A0
const PITCH_MAX = 108; // C8
/** Top of the auto-fit view; higher pitches are reachable only by zooming. */
const DEFAULT_PITCH_TOP = 84; // C6
const DEFAULT_PITCH_RANGE = DEFAULT_PITCH_TOP - PITCH_MIN + 1; // 64

/** Width in px of the piano-key strip on the left edge. */
export const KEY_WIDTH = 56;

/** Black keys within an octave (C=0): C#, D#, F#, G#, A#. */
function isBlackKey(pitch: number): boolean {
  const n = ((pitch % 12) + 12) % 12;
  return n === 1 || n === 3 || n === 6 || n === 8 || n === 10;
}

/**
 * Golden-angle hue step (≈137.5°). Stepping the hue by this each time keeps any
 * prefix of the generated sequence as evenly spaced around the wheel as
 * possible, so the palette stays well-separated however many instruments appear.
 */
const GOLDEN_ANGLE = 137.508;

/**
 * Color index per instrument, assigned in first-seen order and remembered for
 * the lifetime of the page. Persisting it keeps a given instrument on the same
 * color across transcriptions even as the set of instruments changes.
 */
const instrumentIndex = new Map<string, number>();

export function instrumentColor(name: string): string {
  let idx = instrumentIndex.get(name);
  if (idx === undefined) {
    idx = instrumentIndex.size;
    instrumentIndex.set(name, idx);
  }
  const hue = (idx * GOLDEN_ANGLE) % 360;
  return `hsl(${hue}, 70%, 55%)`;
}

export class PianoRoll {
  private canvas: HTMLCanvasElement;
  private ctx: CanvasRenderingContext2D;
  private notes: RollNote[] = [];
  private hiddenInstruments = new Set<string>();
  private mutedInstruments = new Set<string>();
  /** Instrument hovered in the instrument list; all others draw faded. */
  private highlightedInstrument: string | null = null;
  private playhead = 0;
  /**
   * Chunk-completion estimate of how far transcription has reached (seconds), or
   * null when not transcribing. Combined with `latestNoteStart` into the tinted
   * [0, t) frontier; the chunk estimate is what advances the frontier across
   * silent spans where no notes arrive.
   */
  private chunkFrontier: number | null = null;
  /** Start time of the latest note seen — a precise lower bound on the frontier. */
  private latestNoteStart = 0;
  /** Frontier actually drawn; eases toward the target each frame for a smooth glide. */
  private transcribedSmooth = 0;
  /** Seconds-from-start of the left edge when the user has taken over scrolling. */
  private userOffset: number | null = null;
  /** Last offset we rendered with — handy so external code knows what's visible. */
  private lastOffset = 0;
  /** Time-axis zoom, in pixels per second. */
  private _pxPerSec = PX_PER_SEC;
  /** Row height in px when the user has zoomed the pitch axis; null = auto-fit all 88 keys. */
  private pxPerPitch: number | null = null;
  /** Pitch value at the top edge, used only while pitch-zoomed. */
  private pitchTop = DEFAULT_PITCH_TOP;
  /** Loaded audio length in seconds; caps how far the time axis can scroll. */
  private contentDuration = 0;

  get pxPerSec(): number {
    return this._pxPerSec;
  }

  constructor(canvas: HTMLCanvasElement) {
    this.canvas = canvas;
    this.ctx = canvas.getContext("2d")!;
    this.resize();
    window.addEventListener("resize", () => this.resize());
  }

  private resize() {
    const dpr = window.devicePixelRatio || 1;
    const rect = this.canvas.getBoundingClientRect();
    this.canvas.width = rect.width * dpr;
    this.canvas.height = rect.height * dpr;
    this.ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  }

  addNote(n: RollNote) {
    this.notes.push(n);
    // The transcribed frontier never lags behind the most recent note's onset.
    if (n.start > this.latestNoteStart) this.latestNoteStart = n.start;
  }

  /**
   * Record a note's final end time (its `end` event arrived). Beyond driving the
   * left-to-right reveal, this restacks same-pitch notes that overlap `n` in
   * time: whichever of an overlapping pair is drawn behind gets one more upward
   * step ({@link NOTE_STACK_PX}) so it peeks out above the note in front.
   *
   * Iteration order matches draw order (insertion order), so notes seen before
   * `n` are behind it and notes seen after are in front. Each overlapping pair
   * is settled exactly once — when the later of the two to arrive finalizes —
   * since only already-finalized notes (`spawn` set) take part.
   */
  setNoteEnd(n: RollNote, end: number) {
    n.end = end;
    n.spawn = performance.now();
    let behindN = true;
    for (const e of this.notes) {
      if (e === n) {
        behindN = false;
        continue;
      }
      if (e.spawn === undefined || e.pitch !== n.pitch) continue;
      // Half-open overlap test on the final time ranges.
      if (e.start < n.end && n.start < e.end) {
        if (behindN) e.stackOffset = (e.stackOffset ?? 0) + 1;
        else n.stackOffset = (n.stackOffset ?? 0) + 1;
      }
    }
  }

  /** Spotlight `name`'s notes (fading all other instruments); null clears it. */
  setHighlightedInstrument(name: string | null) {
    this.highlightedInstrument = name;
  }

  setPlayhead(seconds: number) {
    this.playhead = seconds;
  }

  /** Tell the roll the loaded audio length, which bounds the time-axis scroll. */
  setDuration(seconds: number) {
    this.contentDuration = seconds;
  }

  /**
   * Feed the chunk-completion estimate of the transcribed span, or null when not
   * transcribing (which disables the tint). The drawn frontier is the max of this
   * and the latest note onset, eased over a few frames.
   */
  setTranscribedUntil(seconds: number | null) {
    this.chunkFrontier = seconds;
  }

  /** Show or hide every note belonging to `instrument`. */
  setInstrumentVisible(instrument: string, visible: boolean) {
    if (visible) this.hiddenInstruments.delete(instrument);
    else this.hiddenInstruments.add(instrument);
  }

  /** Dim every note belonging to `instrument` (drawn at 10% alpha) when muted. */
  setInstrumentMuted(instrument: string, muted: boolean) {
    if (muted) this.mutedInstruments.add(instrument);
    else this.mutedInstruments.delete(instrument);
  }

  clear() {
    this.notes = [];
    this.hiddenInstruments.clear();
    // Unmounting the instrument list mid-hover fires no mouseleave, so drop
    // any stale spotlight here.
    this.highlightedInstrument = null;
    this.playhead = 0;
    this.chunkFrontier = null;
    this.latestNoteStart = 0;
    this.transcribedSmooth = 0;
    this.userOffset = null;
    this.lastOffset = 0;
    this._pxPerSec = PX_PER_SEC;
    this.pxPerPitch = null;
    this.pitchTop = DEFAULT_PITCH_TOP;
    this.contentDuration = 0;
  }

  /** Add `deltaSeconds` to the left-edge offset, taking the view out of follow mode. */
  scrollBy(deltaSeconds: number) {
    const base = this.userOffset ?? this.lastOffset;
    this.userOffset = this.clampOffset(base + deltaSeconds, this._pxPerSec);
  }

  /** Pan the pitch axis by `deltaPx` pixels. */
  scrollPitchBy(deltaPx: number) {
    const H = this.canvas.getBoundingClientRect().height;
    if (this.pxPerPitch === null) {
      // Auto-fit hides everything above C6; entering a panned window at the
      // same row height lets the user scroll up to it without zooming first.
      this.pxPerPitch = H / DEFAULT_PITCH_RANGE;
      this.pitchTop = DEFAULT_PITCH_TOP;
    }
    this.pitchTop = this.clampPitchTop(
      this.pitchTop + deltaPx / this.pxPerPitch,
      this.pxPerPitch,
      H,
    );
  }

  /**
   * Zoom the time axis by `factor`, keeping the second under `anchorX` fixed.
   * `anchorX` is canvas-relative (including the key strip); the gutter is
   * subtracted here so the content area starts at zero seconds-from-offset.
   */
  zoomTime(factor: number, anchorX: number) {
    const old = this._pxPerSec;
    const next = Math.min(MAX_PX_PER_SEC, Math.max(MIN_PX_PER_SEC, old * factor));
    if (next === old) return;
    const ax = anchorX - KEY_WIDTH;
    const base = this.userOffset ?? this.lastOffset;
    const anchorSec = base + ax / old;
    this._pxPerSec = next;
    this.userOffset = this.clampOffset(anchorSec - ax / next, next);
  }

  /** Zoom the pitch axis by `factor`, keeping the pitch under `anchorY` (px) fixed. */
  zoomPitch(factor: number, anchorY: number) {
    const H = this.canvas.getBoundingClientRect().height;
    const fit = H / DEFAULT_PITCH_RANGE;
    const oldRowH = this.pxPerPitch ?? fit;
    const oldTop = this.pxPerPitch === null ? DEFAULT_PITCH_TOP : this.pitchTop;
    const next = Math.min(MAX_PX_PER_PITCH, oldRowH * factor);
    if (next <= fit) {
      // Zoomed back out to where all 88 keys fit — return to auto-fit mode.
      this.pxPerPitch = null;
      return;
    }
    const anchorPitch = oldTop - anchorY / oldRowH;
    this.pxPerPitch = next;
    this.pitchTop = this.clampPitchTop(anchorPitch + anchorY / next, next, H);
  }

  /** Keep the zoomed pitch window inside [A0, C8]. */
  private clampPitchTop(top: number, rowH: number, H: number): number {
    const visible = H / rowH;
    const min = Math.min(PITCH_MIN + visible, PITCH_MAX);
    return Math.min(PITCH_MAX, Math.max(min, top));
  }

  /** Resume auto-following the playhead. */
  follow() {
    this.userOffset = null;
  }

  /** Take over scrolling: freeze the view at its current offset. */
  unfollow() {
    this.userOffset = this.lastOffset;
  }

  get isFollowing(): boolean {
    return this.userOffset === null;
  }

  /** Translate a canvas-relative x pixel coordinate to a transport-time in seconds. */
  xToSeconds(x: number): number {
    return Math.max(0, this.lastOffset + (x - KEY_WIDTH) / this._pxPerSec);
  }

  /** Latest note end across all notes — drives x-axis scroll. */
  private maxEnd(): number {
    let m = 0;
    for (const n of this.notes) if (n.end > m) m = n.end;
    return m;
  }

  /** End of the scrollable content: the further of the audio length and last note. */
  private contentEnd(): number {
    return Math.max(this.contentDuration, this.maxEnd());
  }

  /**
   * Clamp a left-edge offset (seconds) to [0, contentEnd - viewSec] so the view
   * can't scroll past the start or beyond the end of the audio. `pxPerSec` is
   * passed in because callers (zoomTime) may be mid-change to it.
   */
  private clampOffset(offsetSec: number, pxPerSec: number): number {
    const contentW = this.canvas.getBoundingClientRect().width - KEY_WIDTH;
    const viewSec = contentW / pxPerSec;
    const max = Math.max(0, this.contentEnd() - viewSec);
    return Math.max(0, Math.min(offsetSec, max));
  }

  render() {
    const ctx = this.ctx;
    const rect = this.canvas.getBoundingClientRect();
    const W = rect.width;
    const H = rect.height;
    ctx.clearRect(0, 0, W, H);

    // Auto-scroll: keep the view still while the playhead is inside it; only
    // scroll once the playhead reaches the right third (or leaves the view to
    // the left, e.g. on stop). Manual scrolling (userOffset !== null) wins.
    // The note area starts after the key strip; only that width shows time.
    const contentW = W - KEY_WIDTH;
    const maxT = Math.max(this.maxEnd(), this.playhead + 1);
    const viewSec = contentW / this._pxPerSec;
    let offsetSec: number;
    if (this.userOffset !== null) {
      // Re-clamp each frame so a window resize or newly-streamed notes can't
      // leave the frozen view stranded past the content end.
      offsetSec = Math.max(0, Math.min(this.userOffset, Math.max(0, this.contentEnd() - viewSec)));
    } else {
      offsetSec = this.lastOffset;
      if (this.playhead > offsetSec + viewSec * 0.66) {
        offsetSec = this.playhead - viewSec * 0.66;
      } else if (this.playhead < offsetSec) {
        offsetSec = this.playhead;
      }
      offsetSec = Math.max(0, Math.min(offsetSec, Math.max(0, maxT - viewSec)));
    }
    this.lastOffset = offsetSec;
    const pxPerSec = this._pxPerSec;

    // Pitch axis: auto-fit all 88 keys, or use the zoomed window if the user
    // has pinched vertically.
    const rowH = this.pxPerPitch ?? H / DEFAULT_PITCH_RANGE;
    const pitchTop = this.pxPerPitch === null ? DEFAULT_PITCH_TOP : this.pitchTop;

    // Everything time-indexed lives in the content area to the right of the
    // key strip; clip so notes/grid can't bleed under the keyboard.
    ctx.save();
    ctx.beginPath();
    ctx.rect(KEY_WIDTH, 0, contentW, H);
    ctx.clip();

    // Pitch grid: faint horizontal stripes for C notes
    ctx.fillStyle = "#121212";
    for (let p = PITCH_MIN; p <= PITCH_MAX; p++) {
      if (p % 12 === 0) {
        const y = (pitchTop - p) * rowH;
        ctx.fillRect(KEY_WIDTH, y, contentW, rowH);
      }
    }

    // Time grid: vertical lines every second.
    ctx.strokeStyle = "#282828";
    ctx.lineWidth = 1;
    const startSec = Math.floor(offsetSec);
    const endSec = Math.ceil(offsetSec + viewSec);
    for (let s = startSec; s <= endSec; s++) {
      const x = KEY_WIDTH + (s - offsetSec) * pxPerSec;
      ctx.beginPath();
      ctx.moveTo(x + 0.5, 0);
      ctx.lineTo(x + 0.5, H);
      ctx.stroke();
    }

    // Transcribed-so-far overlay: a faint light wash over the [0, t) time span
    // the backend has already processed. The frontier is the furthest of the
    // chunk-completion estimate (which advances across silent spans) and the
    // latest note onset (precise where notes exist), eased toward each frame so
    // it glides instead of snapping when an anchor lands or a note starts. Drawn
    // over the grid but under the notes so they stay crisp; clipped to content.
    if (this.chunkFrontier !== null) {
      const target = Math.max(this.chunkFrontier, this.latestNoteStart);
      // ~12%/frame ≈ a 150 ms ease at 60 fps; snap once effectively there.
      this.transcribedSmooth += (target - this.transcribedSmooth) * 0.12;
      if (Math.abs(target - this.transcribedSmooth) < 0.01) {
        this.transcribedSmooth = target;
      }
      const x0 = KEY_WIDTH + (0 - offsetSec) * pxPerSec;
      const x1 = KEY_WIDTH + (this.transcribedSmooth - offsetSec) * pxPerSec;
      ctx.fillStyle = "rgba(255, 255, 255, 0.05)";
      ctx.fillRect(x0, 0, x1 - x0, H);
    } else {
      this.transcribedSmooth = 0; // tint off once transcription finishes
    }

    // Notes
    const now = performance.now();
    // Sliver of background between adjacent semitones, so notes on
    // consecutive pitches read as separate bars at any zoom level.
    const rowGap = Math.max(0.5, Math.min(4, rowH * 0.1));
    // Stack-shift budget for overlapping same-pitch notes: at most 3px, and
    // never past the row gap (minus a 1px border) — so a shifted note can't
    // collide with a note one semitone up. At the unzoomed row height this
    // rounds to no shift at all.
    const maxShift = Math.min(3, Math.max(0, rowGap - 1));
    for (const n of this.notes) {
      if (n.pitch < PITCH_MIN || n.pitch > PITCH_MAX) continue;
      if (this.hiddenInstruments.has(n.instrument)) continue;
      const x = KEY_WIDTH + (n.start - offsetSec) * pxPerSec;
      const full = Math.max(2, (n.end - n.start) * pxPerSec - NOTE_GAP_PX);
      // Left-to-right reveal: grow from the note's onset to its full width,
      // easing out over REVEAL_MS. Unstamped notes draw at full width.
      const reveal =
        n.spawn === undefined ? 1 : easeOut(Math.min(1, (now - n.spawn) / REVEAL_MS));
      const w = full * reveal;
      if (x + w < KEY_WIDTH || x > W) continue;
      // Nudge up so notes hidden behind an overlapping same-pitch note peek
      // out. stackOffset counts overlapping notes, so it can grow large (a
      // sustained note under a run of staccato hits collects +1 per hit) —
      // cap the TOTAL shift at the budget above.
      const shift = Math.min((n.stackOffset ?? 0) * NOTE_STACK_PX, maxShift);
      const y = (pitchTop - n.pitch) * rowH - shift;
      if (y + rowH < 0 || y > H) continue;
      // Muting dims a track; while an instrument is highlighted (hovered in
      // the instrument list), every other track fades even further back.
      let alpha = this.mutedInstruments.has(n.instrument) ? 0.2 : 1;
      if (this.highlightedInstrument !== null && n.instrument !== this.highlightedInstrument) {
        alpha = Math.min(alpha, 0.15);
      }
      ctx.globalAlpha = alpha;
      ctx.fillStyle = instrumentColor(n.instrument);
      ctx.fillRect(x, y, w, Math.max(2, rowH - rowGap));
    }
    ctx.globalAlpha = 1;

    // Playhead
    const px = KEY_WIDTH + (this.playhead - offsetSec) * pxPerSec;
    ctx.strokeStyle = "#39f2ae";
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.moveTo(px, 0);
    ctx.lineTo(px, H);
    ctx.stroke();

    ctx.restore();

    this.drawKeyboard(ctx, H, rowH, pitchTop);
  }

  /** Draw the piano-key strip down the left edge, aligned to the pitch axis. */
  private drawKeyboard(
    ctx: CanvasRenderingContext2D,
    H: number,
    rowH: number,
    pitchTop: number,
  ) {
    const showLabels = rowH >= 8;
    if (showLabels) {
      ctx.font = "10px Satoshi-Variable, sans-serif";
      ctx.textBaseline = "middle";
    }
    for (let p = PITCH_MIN; p <= PITCH_MAX; p++) {
      const y = (pitchTop - p) * rowH;
      if (y + rowH < 0 || y > H) continue;
      ctx.fillStyle = isBlackKey(p) ? "#232323" : "#efefef";
      ctx.fillRect(0, y, KEY_WIDTH, rowH);
      // Thin separator between adjacent keys.
      ctx.strokeStyle = "#121212";
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.moveTo(0, y + 0.5);
      ctx.lineTo(KEY_WIDTH, y + 0.5);
      ctx.stroke();
      // Octave label on each C (MIDI 60 = C4).
      if (showLabels && p % 12 === 0) {
        ctx.fillStyle = "#121212";
        ctx.fillText(`C${p / 12 - 1}`, 4, y + rowH / 2);
      }
    }
    // Boundary between keyboard and note area.
    ctx.strokeStyle = "#000";
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(KEY_WIDTH + 0.5, 0);
    ctx.lineTo(KEY_WIDTH + 0.5, H);
    ctx.stroke();
  }
}
