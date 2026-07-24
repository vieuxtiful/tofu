import { useCallback, useEffect, useRef, useState } from "react";
import { ChevronDown, Check } from "lucide-react";

export interface BBoxColorOption {
  label: string;
  value: string;
}

export const BBOX_COLORS: BBoxColorOption[] = [
  { label: "cyan",     value: "#22d3ee" },
  { label: "emerald",  value: "#22c55e" },
  { label: "amber",    value: "#f59e0b" },
  { label: "rose",     value: "#f43f5e" },
  { label: "violet",   value: "#a855f7" },
  { label: "sky",      value: "#0ea5e9" },
  { label: "orange",   value: "#f97316" },
  { label: "lime",     value: "#84cc16" },
];

interface Props {
  value: string;
  onChange: (color: string) => void;
}

export default function BBoxColorDropdown({ value, onChange }: Props) {
  const [open, setOpen] = useState(false);
  const wrapperRef = useRef<HTMLDivElement>(null);

  const openPanel = useCallback(() => setOpen(true), []);

  useEffect(() => {
    if (!open) return;
    const onDown = (e: MouseEvent) => {
      if (wrapperRef.current && !wrapperRef.current.contains(e.target as Node)) {
        setOpen(false);
      }
    };
    const onKey = (e: KeyboardEvent) => { if (e.key === "Escape") setOpen(false); };
    document.addEventListener("mousedown", onDown);
    document.addEventListener("keydown", onKey);
    return () => {
      document.removeEventListener("mousedown", onDown);
      document.removeEventListener("keydown", onKey);
    };
  }, [open]);

  const current = BBOX_COLORS.find((c) => c.value === value) ?? BBOX_COLORS[0];

  return (
    <div ref={wrapperRef} className="relative">
      <button
        onClick={(e) => { e.stopPropagation(); open ? setOpen(false) : openPanel(); }}
        className="flex w-full items-center gap-1.5 rounded-sm border border-transparent px-1 py-0.5 text-left text-xs transition hover:border-zinc-300 hover:bg-zinc-200 dark:hover:border-zinc-700 dark:hover:bg-zinc-800 text-zinc-800 dark:text-zinc-200"
        title={`bbox color: ${current.label}`}
      >
        <span className="h-3.5 w-3.5 shrink-0 rounded-xs border border-zinc-400/50" style={{ background: current.value }} />
        <span className="min-w-0 flex-1 truncate">{current.label}</span>
        <ChevronDown size={10} className="shrink-0 text-zinc-500" />
      </button>
      <div
        onClick={(e) => e.stopPropagation()}
        className={`dropdown-morph bezier-card absolute left-0 top-full z-100 mt-1 w-40 overflow-hidden rounded-lg border border-zinc-300 bg-white dark:border-zinc-700 dark:bg-zinc-900${open ? " expanded" : ""}`}
        style={open ? { boxShadow: "4px 4px 0 var(--bc-shadow), 8px 8px 16px rgba(0,0,0,0.18)" } : undefined}
      >
        <div className="subtext border-b border-zinc-200 px-2 py-1.5 text-[10px] uppercase tracking-wider text-zinc-500 dark:border-zinc-800">
          bbox color
        </div>
        <div className="grid grid-cols-4 gap-1 p-2">
          {BBOX_COLORS.map((c) => (
            <button
              key={c.value}
              onClick={() => { onChange(c.value); setOpen(false); }}
              className={`relative flex h-8 items-center justify-center rounded-md border transition ${
                c.value === value
                  ? "border-zinc-400 dark:border-zinc-500 ring-1 ring-zinc-400/50"
                  : "border-zinc-200 hover:border-zinc-400 dark:border-zinc-700 dark:hover:border-zinc-500"
              }`}
              title={c.label}
            >
              <span className="h-5 w-5 rounded-xs" style={{ background: c.value }} />
              {c.value === value && (
                <Check size={10} className="absolute -right-0.5 -top-0.5 rounded-full bg-white text-zinc-800 dark:bg-zinc-900 dark:text-zinc-200" />
              )}
            </button>
          ))}
        </div>
      </div>
    </div>
  );
}
