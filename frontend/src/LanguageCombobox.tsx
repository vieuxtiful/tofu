import { useCallback, useEffect, useRef, useState } from "react";
import { ChevronDown } from "lucide-react";
import { LANGUAGES, LANGUAGE_REGIONS, REGION_ORDER, REGION_ABBR, langDisplayName, langFlag } from "./languageData";

interface LanguageComboboxProps {
  value: string | null;
  onChange: (code: string) => void;
  availableCodes: string[];
  placeholder?: string;
  /** subtle styling when the value is inherited rather than explicit */
  inherited?: boolean;
  /** when set, only show languages from this region */
  regionFilter?: string | null;
  disabled?: boolean;
  /** show only the language code (e.g. ko-KR) instead of flag + full name */
  compact?: boolean;
}

/** filterable language picker: collapsed state shows flag + name; typing
 * filters by name, native name, or code; the arrow opens the full
 * region-grouped list. dropdown renders position:absolute below the button. */
export default function LanguageCombobox({
  value, onChange, availableCodes, placeholder = "language", inherited = false, regionFilter = null, disabled = false, compact = false,
}: LanguageComboboxProps) {
  const [open, setOpen] = useState(false);
  const [filter, setFilter] = useState("");
  const [selectedRegions, setSelectedRegions] = useState<Set<string>>(new Set());
  const wrapperRef = useRef<HTMLDivElement>(null);
  const panelRef = useRef<HTMLDivElement>(null);

  const available = new Set(availableCodes);

  const openPanel = useCallback(() => {
    setFilter("");
    setSelectedRegions(new Set());
    setOpen(true);
  }, []);

  // outside click / escape closes
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

  const q = filter.trim().toLowerCase();
  const matches = (code: string): boolean => {
    if (!q) return true;
    const meta = LANGUAGES.find((l) => l.code === code);
    return (
      code.toLowerCase().includes(q)
      || (meta?.name.toLowerCase().includes(q) ?? false)
      || (meta?.nativeName.toLowerCase().includes(q) ?? false)
    );
  };
  const regionMatches = (region: string): boolean => {
    if (!q) return true;
    return (
      region.toLowerCase().includes(q)
      || (REGION_ABBR[region] ?? "").toLowerCase().includes(q)
    );
  };

  const pick = (code: string) => {
    onChange(code);
    setOpen(false);
  };

  return (
    <div ref={wrapperRef} className="relative">
      <button
        onClick={(e) => { e.stopPropagation(); if (!disabled) open ? setOpen(false) : openPanel(); }}
        className={`flex w-full items-center gap-1 rounded border border-transparent px-1 py-0.5 text-left text-xs transition ${
          disabled ? "cursor-not-allowed opacity-50" : "hover:border-zinc-300 hover:bg-zinc-200 dark:hover:border-zinc-700 dark:hover:bg-zinc-800"
        } ${
          inherited ? "text-zinc-500" : "text-zinc-800 dark:text-zinc-200"
        }`}
        title={value ? `${langDisplayName(value)} (${value})` : placeholder}
      >
        <span className="min-w-0 flex-1 truncate">
          {value ? (compact ? value : `${langFlag(value)} ${langDisplayName(value)}`) : placeholder}
        </span>
        <ChevronDown size={10} className="shrink-0 text-zinc-500" />
      </button>
      {/* Dropdown Morph — reusable expand/collapse animation (see .dropdown-morph in uikit.css) */}
      <div
        ref={panelRef}
        onClick={(e) => e.stopPropagation()}
        className={`dropdown-morph bezier-card absolute left-0 top-full z-[100] mt-1 w-44 overflow-hidden rounded-lg border border-zinc-300 bg-white dark:border-zinc-700 dark:bg-zinc-900${open ? " expanded" : ""}`}
        style={open ? { boxShadow: "4px 4px 0 var(--bc-shadow), 8px 8px 16px rgba(0,0,0,0.18)" } : undefined}
      >
        <input
          autoFocus={open}
          value={filter}
          onChange={(e) => setFilter(e.target.value)}
          placeholder="filter…"
          className="w-full border-b border-zinc-200 bg-zinc-50 px-2 py-1.5 text-xs text-zinc-800 outline-none dark:border-zinc-800 dark:bg-zinc-950 dark:text-zinc-200"
        />
        {/* region filter chips */}
        <div className="flex flex-wrap gap-0.5 border-b border-zinc-200 px-2 py-1.5 dark:border-zinc-800">
          {REGION_ORDER.map((region) => {
            const count = (LANGUAGE_REGIONS[region] || []).filter((c) => available.has(c)).length;
            if (count === 0 || !regionMatches(region)) return null;
            const isActive = selectedRegions.has(region);
            return (
              <button
                key={region}
                onClick={(e) => {
                  e.stopPropagation();
                  setSelectedRegions((prev) => {
                    const next = new Set(prev);
                    if (next.has(region)) next.delete(region);
                    else next.add(region);
                    return next;
                  });
                }}
                className={`subtext rounded px-1.5 py-0.5 text-[10px] font-medium transition ${
                  isActive
                    ? "bg-cyan-600 text-white"
                    : "bg-zinc-100 text-zinc-500 hover:bg-zinc-200 dark:bg-zinc-800 dark:text-zinc-400 dark:hover:bg-zinc-700"
                }`}
                title={region}
              >
                {REGION_ABBR[region] ?? region}
              </button>
            );
          })}
        </div>
        <div className="max-h-56 overflow-y-auto py-1">
          {REGION_ORDER.map((region) => {
            if (selectedRegions.size > 0 && !selectedRegions.has(region)) return null;
            if (regionFilter && region !== regionFilter) return null;
            const allCodes = (LANGUAGE_REGIONS[region] || []).filter((c) => available.has(c));
            const matchedCodes = allCodes.filter((c) => matches(c));
            const rMatch = regionMatches(region);
            // show region if its name matches the filter (show all langs) or if specific langs match
            if (!rMatch && matchedCodes.length === 0) return null;
            const displayCodes = rMatch ? allCodes : matchedCodes;
            return (
              <div key={region}>
                <div className="subtext px-2 pb-0.5 pt-1 text-[10px] uppercase tracking-wider text-zinc-500 dark:text-zinc-600">
                  {region}
                </div>
                {displayCodes.map((code) => (
                  <button
                    key={code}
                    onClick={() => pick(code)}
                    className={`flex w-full items-center gap-1.5 px-2 py-1 text-left text-xs hover:bg-zinc-100 dark:hover:bg-zinc-800 ${
                      code === value ? "text-cyan-600 dark:text-cyan-400" : "text-zinc-700 dark:text-zinc-300"
                    }`}
                  >
                    <span>{langFlag(code)}</span>
                    <span className="min-w-0 flex-1 truncate">{langDisplayName(code)}</span>
                    <span className="text-[10px] text-zinc-600">{code}</span>
                  </button>
                ))}
              </div>
            );
          })}
        </div>
      </div>
    </div>
  );
}
