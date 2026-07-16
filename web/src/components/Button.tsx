import clsx from "clsx";
import type { ButtonHTMLAttributes } from "react";

type ButtonKind = "primary" | "secondary" | "secondaryOff" | "ghost";

// This skin keeps the default button chrome in style.css — @layer base for the
// resting look, the btn-primary utility for the CTA gradient — so the kinds
// mostly just name the roles:
//   primary      — the glowing accent-gradient CTA (Transcribe, Download).
//   secondary    — the default chrome from @layer base; nothing to add.
//   secondaryOff — the "off" half of a toggle: same as secondary here (this
//                  skin marks the "on" half at the call site instead).
//   ghost        — no chrome (inline icons, links, menu rows): undo the
//                  @layer base fill/border/padding.
const kindToClass: Record<ButtonKind, string> = {
  primary: "btn-primary rounded-xl",
  secondary: "",
  secondaryOff: "",
  ghost: "border-none bg-transparent p-0",
};

/**
 * Shared button element. `size` and `pad` replace the kind's defaults (instead
 * of fighting them in the class list, where Tailwind's stylesheet order — not
 * class order — would decide the winner); `className` is for everything else.
 */
export function Button({
  kind = "secondary",
  size,
  pad,
  className,
  ...rest
}: {
  kind?: ButtonKind;
  /** Font-size utility, e.g. "text-base". Defaults to the base chrome's 13px. */
  size?: string;
  /** Padding utilities, e.g. "px-9 py-3". Defaults to the base chrome's 8/16px. */
  pad?: string;
} & ButtonHTMLAttributes<HTMLButtonElement>) {
  return (
    <button {...rest} className={clsx(kindToClass[kind], size, pad, className)} />
  );
}
