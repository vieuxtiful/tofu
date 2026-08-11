import { useEffect, useRef, useState } from "react";
import { ChevronDown } from "lucide-react";
import type { CaptureMode } from "./api";

/**
 * Auto / Guided / Manual, in the Project Manager card.
 *
 * Styled as the language pickers are (transparent trigger, Dropdown Morph
 * panel) rather than as a native <select>, so the Project Manager reads as
 * one control family instead of two.
 *
 * The order is fixed: Auto first because it is the default and what every
 * existing project already does, Manual last because it turns detection off.
 */

type Props = {
  value: CaptureMode;
  onChange: (mode: CaptureMode) => void;
  disabled?: boolean;
};

export const CAPTURE_MODES: { id: CaptureMode; label: string; hint: string }[] = [
  { id: "auto", label: "Auto", hint: "Detect all text. Any terms you add are extra evidence." },
  { id: "guided", label: "Guided", hint: "Locate the source text you supply." },
  { id: "manual", label: "Manual", hint: "No detection — draw regions yourself." },
];

export default function CaptureModeSelect({ value, onChange, disabled }: Props) {
  const [open, setOpen] = useState(false);
  const wrapperRef = useRef<HTMLDivElement | null>(null);
  const current = CAPTURE_MODES.find((mode) => mode.id === value) ?? CAPTURE_MODES[0];

  useEffect(() => {
    if (!open) return;
    const close = (event: MouseEvent) => {
      if (wrapperRef.current && !wrapperRef.current.contains(event.target as Node)) {
        setOpen(false);
      }
    };
    document.addEventListener("mousedown", close);
    return () => document.removeEventListener("mousedown", close);
  }, [open]);

  return (
    <div>
      <p className="subtext mb-1 text-xs uppercase tracking-wider text-zinc-500 dark:text-zinc-600">
        capture mode
      </p>
      <div ref={wrapperRef} className="relative">
        <button
          type="button"
          aria-label="Capture mode"
          aria-haspopup="listbox"
          aria-expanded={open}
          onClick={(e) => { e.stopPropagation(); if (!disabled) setOpen((v) => !v); }}
          className={`flex w-full items-center gap-1 rounded border border-transparent px-1 py-0.5 text-left text-xs transition ${
            disabled
              ? "cursor-not-allowed opacity-50"
              : "hover:border-zinc-300 hover:bg-zinc-200 dark:hover:border-zinc-700 dark:hover:bg-zinc-800"
          } text-zinc-800 dark:text-zinc-200`}
          title={current.hint}
        >
          <span className="min-w-0 flex-1 truncate">{current.label}</span>
          <ChevronDown size={10} className="shrink-0 text-zinc-500" />
        </button>
        {/* Dropdown Morph — reusable expand/collapse animation (see .dropdown-morph in uikit.css) */}
        <div
          role="listbox"
          onClick={(e) => e.stopPropagation()}
          className={`dropdown-morph bezier-card absolute left-0 top-full z-100 mt-1 w-44 overflow-hidden rounded-lg border border-zinc-300 bg-white dark:border-zinc-700 dark:bg-zinc-900${open ? " expanded" : ""}`}
          style={open ? { boxShadow: "4px 4px 0 var(--bc-shadow), 8px 8px 16px rgba(0,0,0,0.18)" } : undefined}
        >
          {CAPTURE_MODES.map((mode) => (
            <button
              key={mode.id}
              type="button"
              role="option"
              aria-selected={mode.id === value}
              onClick={() => { onChange(mode.id); setOpen(false); }}
              className={`flex w-full flex-col gap-0.5 px-3 py-1.5 text-left text-xs transition hover:bg-zinc-100 dark:hover:bg-zinc-800 ${
                mode.id === value ? "text-cyan-600 dark:text-cyan-400" : "text-zinc-700 dark:text-zinc-300"
              }`}
            >
              <span>{mode.label}</span>
              <span className="subtext text-[10px] leading-snug text-zinc-500">{mode.hint}</span>
            </button>
          ))}
        </div>
      </div>
      <p className="subtext mt-1 text-[11px] leading-snug text-zinc-500">{current.hint}</p>
    </div>
  );
}
