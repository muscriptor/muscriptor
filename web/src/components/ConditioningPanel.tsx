import { useEffect, useMemo, useRef, useState } from "react";
import clsx from "clsx";
import { Button } from "./Button";
import { label, scoreInstrument } from "../instruments";

/**
 * Conditioning instruments: optional pre-selection that overrides the model's
 * auto-detect. Names come from the backend `/instruments` endpoint; the panel
 * silently renders empty if the server doesn't expose it.
 *
 * Selected instruments are shown as removable tags. Typing in the field offers
 * a filtered list of suggestions to add.
 */
export function ConditioningPanel(props: {
  selected: Set<string>;
  onChange: (next: Set<string>) => void;
}) {
  const { selected, onChange } = props;
  const [available, setAvailable] = useState<string[]>([]);
  const [query, setQuery] = useState("");
  const [open, setOpen] = useState(false);
  const [highlight, setHighlight] = useState(0);
  const rootRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    let cancelled = false;
    (async () => {
      try {
        const resp = await fetch("/instruments");
        if (!resp.ok) return;
        const data = (await resp.json()) as { instruments: string[] };
        if (!cancelled) setAvailable(data.instruments);
      } catch {
        /* ignore — server may not expose endpoint */
      }
    })();
    return () => {
      cancelled = true;
    };
  }, []);

  // Close the suggestion list when clicking outside the panel.
  useEffect(() => {
    if (!open) return;
    function onDocClick(e: MouseEvent) {
      if (!rootRef.current?.contains(e.target as Node)) setOpen(false);
    }
    document.addEventListener("mousedown", onDocClick);
    return () => document.removeEventListener("mousedown", onDocClick);
  }, [open]);

  const suggestions = useMemo(() => {
    return (
      available
        .filter((n) => !selected.has(n))
        .map((n, i) => ({ n, i, score: scoreInstrument(n, query) }))
        .filter((c) => c.score > 0)
        // Best score first; ties keep the backend ordering for stability.
        .sort((a, b) => b.score - a.score || a.i - b.i)
        .map((c) => c.n)
    );
  }, [available, selected, query]);

  // Keep the highlighted index in range as the suggestion list changes.
  useEffect(() => {
    setHighlight((h) => Math.min(h, Math.max(0, suggestions.length - 1)));
  }, [suggestions.length]);

  function add(name: string) {
    if (!available.includes(name) || selected.has(name)) return;
    const next = new Set(selected);
    next.add(name);
    onChange(next);
    setQuery("");
    setHighlight(0);
  }

  function remove(name: string) {
    const next = new Set(selected);
    next.delete(name);
    onChange(next);
  }

  function onKeyDown(e: React.KeyboardEvent<HTMLInputElement>) {
    if (e.key === "ArrowDown") {
      e.preventDefault();
      setOpen(true);
      setHighlight((h) => Math.min(h + 1, suggestions.length - 1));
    } else if (e.key === "ArrowUp") {
      e.preventDefault();
      setHighlight((h) => Math.max(h - 1, 0));
    } else if (e.key === "Enter") {
      e.preventDefault();
      if (open && suggestions[highlight]) add(suggestions[highlight]);
    } else if (e.key === "Escape") {
      setOpen(false);
    } else if (e.key === "Backspace" && query === "" && selected.size > 0) {
      // Backspace on an empty field removes the most recently added tag.
      remove(Array.from(selected)[selected.size - 1]);
    }
  }

  return (
    <section className="card col-span-full px-5 pb-5 pt-4 animate-rise [animation-delay:0.24s]">
      <div className="mb-3.5 flex items-start gap-3">
        <div className="flex flex-col gap-2">
          <h2 className="m-0 text-lg font-semibold">
            What instruments are there in this track?
          </h2>
          <p className="text-faint">
            Optional. Leave empty to let the model detect anything; listing
            instruments here forbids every other instrument from appearing.
          </p>
        </div>
        <div className="ml-auto">
          <Button
            type="button"
            size="text-xs"
            pad="px-3.5 py-1"
            onClick={() => onChange(new Set())}
            disabled={selected.size === 0}
          >
            Clear
          </Button>
        </div>
      </div>

      <div className="relative" ref={rootRef}>
        <div
          className="flex min-h-11 cursor-text flex-wrap items-center gap-2 rounded-lg border border-line-strong bg-bg px-2.5 py-2 transition-colors duration-150 ease-fluid focus-within:border-accent"
          onClick={() => rootRef.current?.querySelector("input")?.focus()}
        >
          {Array.from(selected).map((name) => (
            <span
              className="inline-flex select-none items-center gap-1.5 rounded-md border border-accent-glow bg-accent-soft py-1 pl-3 pr-1.5 text-sm text-content"
              key={name}
            >
              {label(name)}
              <Button
                type="button"
                kind="ghost"
                className="grid size-4 place-content-center rounded text-sm leading-none text-muted hover:bg-white/10 hover:text-content"
                aria-label={`Remove ${label(name)}`}
                onClick={(e) => {
                  e.stopPropagation();
                  remove(name);
                }}
              >
                ×
              </Button>
            </span>
          ))}
          <input
            type="text"
            className="m-0 min-w-32 flex-1 border-none bg-transparent px-0.5 py-1 font-sans text-sm text-content outline-none placeholder:text-faint"
            placeholder={selected.size === 0 ? "Add an instrument…" : ""}
            value={query}
            autoComplete="off"
            spellCheck={false}
            onChange={(e) => {
              setQuery(e.target.value);
              setOpen(true);
            }}
            onFocus={() => setOpen(true)}
            onKeyDown={onKeyDown}
          />
        </div>

        {open && suggestions.length > 0 && (
          <ul
            className="absolute inset-x-0 top-[calc(100%+5px)] z-20 m-0 max-h-60 list-none overflow-y-auto rounded-lg border border-line-strong bg-surface-2 p-1 shadow-pop"
            role="listbox"
          >
            {suggestions.map((name, i) => (
              <li
                key={name}
                role="option"
                aria-selected={i === highlight}
                className={clsx(
                  "cursor-pointer rounded-md px-2.5 py-2 text-sm",
                  i === highlight ? "bg-white/[0.06] text-content" : "text-muted",
                )}
                onMouseEnter={() => setHighlight(i)}
                onMouseDown={(e) => {
                  // mousedown (not click) so we add before the input blurs.
                  e.preventDefault();
                  add(name);
                }}
              >
                {label(name)}
              </li>
            ))}
          </ul>
        )}
      </div>
    </section>
  );
}
