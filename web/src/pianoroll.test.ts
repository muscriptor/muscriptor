import { describe, it, expect, vi } from "vitest";
import { PianoRoll, noteLabel, durationLabel, KEY_WIDTH } from "./pianoroll";

// ---------------------------------------------------------------------------
// noteLabel
// ---------------------------------------------------------------------------

describe("noteLabel", () => {
  it("names middle C correctly", () => {
    expect(noteLabel(60)).toBe("C4");
  });

  it("names semitones above middle C", () => {
    expect(noteLabel(61)).toBe("C♯4");
    expect(noteLabel(62)).toBe("D4");
  });

  it("names the lowest and highest MIDI notes on an 88-key piano", () => {
    expect(noteLabel(21)).toBe("A0");
    expect(noteLabel(108)).toBe("C8");
  });

  it("handles octave boundaries correctly", () => {
    expect(noteLabel(59)).toBe("B3");
    expect(noteLabel(72)).toBe("C5");
  });
});

// ---------------------------------------------------------------------------
// durationLabel
// ---------------------------------------------------------------------------

describe("durationLabel", () => {
  it("shows only seconds when BPM is unknown", () => {
    expect(durationLabel(0.5, null)).toBe("0.50 s");
  });

  it("appends singular 'beat' for exactly one beat", () => {
    // 0.5 s * 120 BPM / 60 = 1 beat
    expect(durationLabel(0.5, 120)).toBe("0.50 s · 1 beat");
  });

  it("appends plural 'beats' for other durations", () => {
    // 0.25 s * 120 BPM / 60 = 0.5 beats
    expect(durationLabel(0.25, 120)).toBe("0.25 s · 0.5 beats");
    // 1.0 s * 120 BPM / 60 = 2 beats
    expect(durationLabel(1.0, 120)).toBe("1.00 s · 2 beats");
  });

  it("rounds beat count to two decimal places", () => {
    // 1/3 s * 120 / 60 = 0.6666… → rounds to 0.67
    expect(durationLabel(1 / 3, 120)).toBe("0.33 s · 0.67 beats");
  });
});

// ---------------------------------------------------------------------------
// PianoRoll.noteAt – hit testing
//
// Canvas 800 × 420 (CSS px). After construction:
//   pxPerPitch = null  → rowH = 420 / 64 ≈ 6.5625
//   pitchTop   = 84 (DEFAULT_PITCH_TOP)
//   pxPerSec   = 80  (PX_PER_SEC)
//   lastOffset = 0
//
// For a note at pitch=60, start=0, end=1:
//   left  = KEY_WIDTH + 0 * 80 = 56
//   width = max(2, 1.0 * 80 − 1.5) = 78.5
//   top   = (84 − 60) * 6.5625 = 157.5
//   noteH = max(2, 6.5625 − 0.65625) ≈ 5.91
//   → hit zone x ∈ [56, 134.5], y ∈ [157.5, 163.4]
// ---------------------------------------------------------------------------

function makeRoll(width = 800, height = 420): PianoRoll {
  const canvas = document.createElement("canvas");
  vi.spyOn(canvas, "getBoundingClientRect").mockReturnValue({
    width,
    height,
    top: 0,
    left: 0,
    right: width,
    bottom: height,
    x: 0,
    y: 0,
    toJSON() {
      return {};
    },
  });
  vi.spyOn(canvas, "getContext").mockReturnValue({
    setTransform: vi.fn(),
  } as unknown as CanvasRenderingContext2D);
  return new PianoRoll(canvas);
}

describe("PianoRoll.noteAt", () => {
  it("returns null when there are no notes", () => {
    const roll = makeRoll();
    expect(roll.noteAt(100, 160)).toBeNull();
  });

  it("returns null when x is inside the key strip", () => {
    const roll = makeRoll();
    roll.addNote({ pitch: 60, start: 0, end: 1, instrument: "piano" });
    expect(roll.noteAt(KEY_WIDTH - 1, 160)).toBeNull();
  });

  it("returns the note when the cursor is inside its bounds", () => {
    const roll = makeRoll();
    const note = { pitch: 60, start: 0, end: 1, instrument: "piano" };
    roll.addNote(note);
    // midpoint of the hit zone: x≈95, y≈160
    expect(roll.noteAt(95, 160)).toBe(note);
  });

  it("returns null when cursor is outside the note bounds", () => {
    const roll = makeRoll();
    roll.addNote({ pitch: 60, start: 0, end: 1, instrument: "piano" });
    // hit zone x ∈ [56, 134.5], y ∈ [157.5, 163.4]
    expect(roll.noteAt(136, 160)).toBeNull(); // right of note
    expect(roll.noteAt(95, 165)).toBeNull();  // below note
    expect(roll.noteAt(95, 156)).toBeNull();  // above note
  });

  it("skips notes whose instrument is hidden", () => {
    const roll = makeRoll();
    const note = { pitch: 60, start: 0, end: 1, instrument: "piano" };
    roll.addNote(note);
    roll.setInstrumentVisible("piano", false);
    expect(roll.noteAt(95, 160)).toBeNull();
  });

  it("returns topmost (last-drawn) note when two notes overlap at the cursor", () => {
    const roll = makeRoll();
    const bottom = { pitch: 60, start: 0, end: 1, instrument: "piano" };
    const top = { pitch: 60, start: 0, end: 1, instrument: "strings" };
    roll.addNote(bottom);
    roll.addNote(top);
    // Both notes occupy the same geometry; top was added last (drawn on top).
    expect(roll.noteAt(95, 160)).toBe(top);
  });

  it("hits a note offset in time", () => {
    const roll = makeRoll();
    // start=2 → left = 56 + 2*80 = 216
    // end=3   → width = max(2, 80-1.5) = 78.5
    // hit zone x ∈ [216, 294.5], y same as pitch=60
    const note = { pitch: 60, start: 2, end: 3, instrument: "piano" };
    roll.addNote(note);
    expect(roll.noteAt(250, 160)).toBe(note);
    expect(roll.noteAt(95, 160)).toBeNull(); // first second is empty
  });
});
