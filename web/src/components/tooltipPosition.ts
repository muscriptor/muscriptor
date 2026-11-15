export type TooltipPos = {
  left?: number;
  right?: number;
  top?: number;
  bottom?: number;
};

/**
 * Compute absolute CSS position for a hover tooltip so it stays inside a
 * container of size W×H. Flips to the left / above the cursor when the
 * cursor is in the right or bottom half of the container.
 */
export function tooltipPosition(
  x: number,
  y: number,
  containerWidth: number,
  containerHeight: number,
): TooltipPos {
  return {
    ...(x > containerWidth / 2 ? { right: containerWidth - x + 12 } : { left: x + 12 }),
    ...(y > containerHeight / 2 ? { bottom: containerHeight - y + 12 } : { top: y + 12 }),
  };
}
