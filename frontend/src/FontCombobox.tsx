import React, { useCallback, useEffect, useRef, useState } from "react";
import { ChevronDown, ChevronRight } from "lucide-react";
import { FontFamily, FontWeight } from "./api";

interface FontComboboxProps {
  value: string | null;
  onChange: (path: string, family: string, weight: FontWeight | null) => void;
  families: FontFamily[];
  placeholder?: string;
}

// caches the LOAD PROMISE (not just a "started" flag) so repeat callers —
// including ones that arrive after the first load already resolved —
// can always `await` the same result instead of getting nothing back.
// textFit.ts's canvas measurements are meaningless against a font whose
// @font-face hasn't actually finished loading yet (the browser silently
// falls back to a system font mid-measurement), so callers that need
// accurate metrics (the Translate-tab preview's wrap/fit) must await this.
const LOADED_FONTS = new Map<string, Promise<void>>();

export function loadFontPreview(path: string, family: string): Promise<void> {
  const key = path;
  const cached = LOADED_FONTS.get(key);
  if (cached) return cached;
  const fontName = `tofu-preview-${key.replace(/[^a-zA-Z0-9]/g, "")}`;
  const url = `/api/font-file?path=${encodeURIComponent(path)}`;
  const style = document.createElement("style");
  style.textContent = `@font-face { font-family: "${fontName}"; src: url("${url}") format("truetype"); }`;
  document.head.appendChild(style);
  const promise = document.fonts.load(`16px "${fontName}"`).then(() => undefined).catch(() => undefined);
  LOADED_FONTS.set(key, promise);
  return promise;
}

export function fontNameForPath(path: string): string {
  return `tofu-preview-${path.replace(/[^a-zA-Z0-9]/g, "")}`;
}

export function weightLabel(w: FontWeight): string {
  // Font naming is not a fixed CSS scale.  In particular, many families use
  // "Heavy" where others use "Black" for the same OS/2 weight class.  Keep
  // the foundry's meaningful name in the source of truth instead of relabeling
  // a selected Heavy face as Black in the editor.
  const subfamily = (w.subfamily || "").trim();
  if (/\b(?:heavy|black|ultra|extra[ -]?bold|semi[ -]?bold|demi[ -]?bold|bold|medium|regular|book|light|thin)\b/i.test(subfamily)) {
    return subfamily;
  }
  const wc = w.weight_class;
  if (wc <= 100) return "Thin";
  if (wc <= 200) return "Extra Light";
  if (wc <= 300) return "Light";
  if (wc <= 400) return "Regular";
  if (wc <= 500) return "Medium";
  if (wc <= 600) return "SemiBold";
  if (wc <= 700) return "Bold";
  if (wc <= 800) return "Extra Bold";
  return "Black";
}

function weightCss(w: FontWeight): React.CSSProperties {
  const wc = w.weight_class;
  const sub = (w.subfamily || "").toLowerCase();
  const style: React.CSSProperties = {};
  if (wc <= 100) style.fontWeight = 100;
  else if (wc <= 200) style.fontWeight = 200;
  else if (wc <= 300) style.fontWeight = 300;
  else if (wc <= 400) style.fontWeight = 400;
  else if (wc <= 500) style.fontWeight = 500;
  else if (wc <= 600) style.fontWeight = 600;
  else if (wc <= 700) style.fontWeight = 700;
  else if (wc <= 800) style.fontWeight = 800;
  else style.fontWeight = 900;
  if (sub.includes("italic") || sub.includes("oblique")) style.fontStyle = "italic";
  return style;
}

export default function FontCombobox({
  value, onChange, families, placeholder = "font",
}: FontComboboxProps) {
  const [open, setOpen] = useState(false);
  const [filter, setFilter] = useState("");
  const [expandedFamily, setExpandedFamily] = useState<string | null>(null);
  const wrapperRef = useRef<HTMLDivElement>(null);

  const selectedFamily = families.find((f) =>
    f.weights.some((w) => w.path === value) || f.best_path === value
  );
  const selectedWeight = selectedFamily?.weights.find((w) => w.path === value) ?? null;

  const displayLabel = value
    ? selectedWeight
      ? `${selectedFamily?.family ?? ""} ${weightLabel(selectedWeight)}`
      : selectedFamily?.family ?? value.split(/[\\/]/).pop()?.replace(/\.(ttf|otf|ttc|otc)$/i, "") ?? value
    : placeholder;

  const openPanel = useCallback(() => {
    setFilter("");
    setExpandedFamily(null);
    setOpen(true);
  }, []);

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

  // preload font previews for all families when panel opens
  useEffect(() => {
    if (!open) return;
    families.forEach((f) => {
      loadFontPreview(f.best_path, f.family);
    });
  }, [open, families]);

  // preload all weight files for a family when it is expanded so each
  // weight label renders in its own typeface immediately
  useEffect(() => {
    if (!expandedFamily) return;
    const fam = families.find((f) => f.family === expandedFamily);
    if (!fam) return;
    fam.weights.forEach((w) => loadFontPreview(w.path, fam.family));
  }, [expandedFamily, families]);

  const q = filter.trim().toLowerCase();
  const matches = (family: string): boolean => {
    if (!q) return true;
    return family.toLowerCase().includes(q);
  };

  const pick = (path: string, family: string, weight: FontWeight | null) => {
    onChange(path, family, weight);
    setOpen(false);
  };

  return (
    <div ref={wrapperRef} className="relative">
      <button
        onClick={(e) => { e.stopPropagation(); open ? setOpen(false) : openPanel(); }}
        className="flex w-full items-center gap-1 rounded border border-transparent px-1 py-0.5 text-left text-xs transition hover:border-zinc-300 hover:bg-zinc-200 dark:hover:border-zinc-700 dark:hover:bg-zinc-800 text-zinc-800 dark:text-zinc-200"
        title={displayLabel}
      >
        <span
          className="min-w-0 flex-1 truncate"
          style={value && selectedFamily ? { fontFamily: fontNameForPath(selectedWeight?.path ?? selectedFamily.best_path) } : undefined}
        >
          {displayLabel}
        </span>
        <ChevronDown size={10} className="shrink-0 text-zinc-500" />
      </button>
      {/* Dropdown Morph — reusable expand/collapse animation (see .dropdown-morph in uikit.css) */}
      <div
        onClick={(e) => e.stopPropagation()}
        className={`dropdown-morph bezier-card absolute left-0 top-full z-[100] mt-1 w-64 overflow-hidden rounded-lg border border-zinc-300 bg-white dark:border-zinc-700 dark:bg-zinc-900${open ? " expanded" : ""}`}
        style={open ? { boxShadow: "4px 4px 0 var(--bc-shadow), 8px 8px 16px rgba(0,0,0,0.18)" } : undefined}
      >
        <input
          autoFocus={open}
          value={filter}
          onChange={(e) => setFilter(e.target.value)}
          placeholder="filter…"
          className="w-full border-b border-zinc-200 bg-zinc-50 px-2 py-1.5 text-xs text-zinc-800 outline-none dark:border-zinc-800 dark:bg-zinc-950 dark:text-zinc-200"
        />
        <div className="max-h-56 overflow-y-auto py-1">
          {/* Auto option */}
          <button
            onClick={() => pick("", "", null)}
            className={`flex w-full items-center px-2 py-1 text-left text-xs hover:bg-zinc-100 dark:hover:bg-zinc-800 ${
              !value ? "text-cyan-600 dark:text-cyan-400" : "text-zinc-700 dark:text-zinc-300"
            }`}
          >
            <span className="min-w-0 flex-1 truncate">auto (best coverage)</span>
          </button>
          {families.filter((f) => matches(f.family)).map((fam) => {
            const isExpanded = expandedFamily === fam.family;
            const isSelected = selectedFamily?.family === fam.family;
            return (
              <div key={fam.family}>
                <button
                  onClick={() => setExpandedFamily(isExpanded ? null : fam.family)}
                  className={`flex w-full items-center gap-1 px-2 py-1 text-left text-xs hover:bg-zinc-100 dark:hover:bg-zinc-800 ${
                    isSelected ? "text-cyan-600 dark:text-cyan-400" : "text-zinc-700 dark:text-zinc-300"
                  }`}
                  style={{ fontFamily: fontNameForPath(fam.best_path) }}
                >
                  <ChevronRight
                    size={10}
                    className={`shrink-0 text-zinc-500 transition-transform ${isExpanded ? "rotate-90" : ""}`}
                  />
                  <span className="min-w-0 flex-1 truncate">{fam.family}</span>
                  <span className="text-[10px] text-zinc-500">{(fam.best_coverage * 100).toFixed(0)}%</span>
                </button>
                {isExpanded && (
                  <div className="ml-4 border-l border-zinc-200 dark:border-zinc-800">
                    {fam.weights.map((w) => (
                      <button
                        key={w.path}
                        onClick={() => pick(w.path, fam.family, w)}
                        className={`flex w-full items-center gap-1 px-2 py-1 text-left text-xs hover:bg-zinc-100 dark:hover:bg-zinc-800 ${
                          value === w.path ? "text-cyan-600 dark:text-cyan-400" : "text-zinc-600 dark:text-zinc-400"
                        }`}
                        style={{ fontFamily: fontNameForPath(w.path) }}
                      >
                        <span className="min-w-0 flex-1 truncate" style={weightCss(w)}>{weightLabel(w)}</span>
                        <span className="text-[10px] text-zinc-500">{(w.coverage * 100).toFixed(0)}%</span>
                      </button>
                    ))}
                  </div>
                )}
              </div>
            );
          })}
          {families.length === 0 && (
            <p className="px-2 py-2 text-xs text-zinc-500">no fonts available for this language</p>
          )}
        </div>
      </div>
    </div>
  );
}
