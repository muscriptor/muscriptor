/** Fullscreen overlay shown (via the `body.drag` class) while a file is dragged. */
export function DropOverlay() {
  return (
    <div
      className="drop-overlay pointer-events-none fixed inset-0 z-200 flex items-center justify-center bg-[rgba(18,18,18,0.8)] opacity-0 backdrop-blur-sm transition-opacity duration-150 ease-fluid"
      aria-hidden="true"
    >
      <div className="flex flex-col items-center gap-3.5 rounded-card border-2 border-dashed border-accent bg-bg px-16 py-11 text-center shadow-overlay">
        <svg
          className="h-8 w-32 fill-none stroke-accent stroke-2 [stroke-linecap:round] [stroke-linejoin:round]"
          viewBox="0 0 120 32"
          aria-hidden="true"
          preserveAspectRatio="none"
        >
          <path d="M0 16 L8 16 L12 6 L18 26 L24 10 L30 22 L36 4 L42 28 L48 14 L54 18 L60 8 L66 24 L72 12 L78 20 L84 6 L90 26 L96 16 L104 16 L120 16" />
        </svg>
        <p className="m-0 text-base text-muted">
          Drop an <strong className="font-semibold text-content">audio file</strong> to
          transcribe
        </p>
      </div>
    </div>
  );
}
