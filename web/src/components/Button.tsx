import clsx from "clsx";
import type { ButtonHTMLAttributes } from "react";

type ButtonKind = "square" | "squareOff" | "slanted" | "ghost";

// The kyutai.org button styles plus a chrome-less reset:
//   square    — SquareButton: black, 1px dashed green border, glows on hover.
//   squareOff — SquareButton's primaryOff: the gray "off" half of a toggle.
//   slanted   — SlantedButton: green skewed fill behind black text; on hover
//               the fill retracts leftwards, leaving the outline + white text.
//   ghost     — no chrome (inline icons, links, menu rows); Tailwind's preflight
//               already strips button backgrounds/borders, so this is minimal.
const kindToClass: Record<ButtonKind, string> = {
  square: clsx(
    "border border-dashed border-accent bg-black font-medium text-accent",
    "drop-shadow-[0_0.2rem_0.15rem_var(--color-surface)]",
    "enabled:hover:shadow-[0_0_10px_var(--color-accent)]",
    "disabled:border-line disabled:text-line",
  ),
  squareOff: clsx(
    "border border-dashed border-faint bg-black font-medium text-muted",
    "drop-shadow-[0_0.2rem_0.15rem_var(--color-surface)]",
    "enabled:hover:shadow-[0_0_10px_var(--color-faint)]",
    "disabled:border-line disabled:text-line",
  ),
  slanted: clsx(
    // isolate keeps the -z-10 fill behind the label without lifting the
    // button above sibling popovers (e.g. the instrument suggestions).
    "relative isolate font-semibold text-black",
    "after:absolute after:inset-y-0 after:left-0 after:-z-10 after:w-full after:content-['']",
    "after:-skew-x-10 after:border-2 after:border-accent after:bg-accent",
    "after:transition-[width] after:duration-300 after:ease-in-out",
    "enabled:hover:text-white enabled:hover:after:w-0",
    "disabled:text-faint disabled:after:border-dashed disabled:after:border-faint disabled:after:bg-surface",
  ),
  ghost: "p-0",
};

/**
 * Shared button element. `size` and `pad` replace the kind's defaults (instead
 * of fighting them in the class list, where Tailwind's stylesheet order — not
 * class order — would decide the winner); `className` is for everything else.
 */
export function Button({
  kind = "square",
  size,
  pad,
  className,
  ...rest
}: {
  kind?: ButtonKind;
  /** Font-size utility, e.g. "text-base". Defaults to text-[13px] (ghost: inherit). */
  size?: string;
  /** Padding utilities, e.g. "px-9 py-3". Defaults to px-4 py-2 (ghost: none). */
  pad?: string;
} & ButtonHTMLAttributes<HTMLButtonElement>) {
  return (
    <button
      {...rest}
      className={clsx(
        "cursor-pointer transition duration-150 ease-fluid",
        "focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-accent",
        "disabled:cursor-not-allowed",
        kindToClass[kind],
        size ?? (kind === "ghost" ? undefined : "text-[13px]"),
        pad ?? (kind === "ghost" ? undefined : "px-4 py-2"),
        className,
      )}
    />
  );
}
