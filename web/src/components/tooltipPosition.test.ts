import { describe, it, expect } from "vitest";
import { tooltipPosition } from "./tooltipPosition";

const W = 800;
const H = 420;

describe("tooltipPosition", () => {
  it("places tooltip to the right and below in the upper-left quadrant", () => {
    expect(tooltipPosition(100, 100, W, H)).toEqual({ left: 112, top: 112 });
  });

  it("flips left when cursor is in the right half", () => {
    // right = W - x + 12 = 800 - 700 + 12 = 112
    expect(tooltipPosition(700, 100, W, H)).toEqual({ right: 112, top: 112 });
  });

  it("flips above when cursor is in the bottom half", () => {
    // bottom = H - y + 12 = 420 - 350 + 12 = 82
    expect(tooltipPosition(100, 350, W, H)).toEqual({ left: 112, bottom: 82 });
  });

  it("flips both when cursor is in the lower-right quadrant", () => {
    expect(tooltipPosition(700, 350, W, H)).toEqual({ right: 112, bottom: 82 });
  });

  it("uses left/top at exactly the midpoint (not strictly greater)", () => {
    // x === W/2 → not > W/2, so left wins
    expect(tooltipPosition(400, 210, W, H)).toEqual({ left: 412, top: 222 });
  });

  it("flips right/bottom one pixel past the midpoint", () => {
    expect(tooltipPosition(401, 211, W, H)).toEqual({ right: 411, bottom: 221 });
  });
});
