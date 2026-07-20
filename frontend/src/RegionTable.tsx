import { Fragment, useCallback, useEffect, useRef, useState, type CSSProperties, type ReactNode } from "react";
import { FontFamily, FontOption, InstText, LanguageOption } from "./api";
import { BookmarkCheck, Eye, EyeOff, Loader2, ScanText, Trash2 } from "lucide-react";
import { langDisplayName } from "./languageData";
import LanguageCombobox from "./LanguageCombobox";
import { loadFontPreview, fontNameForPath } from "./FontCombobox";
import { HiLockClosed, HiLockOpen } from "react-icons/hi";
import "./bbox.css";

/** capture: bbox/string registry (source text, source lang, OCR, delete).
 *  translate: translation work (target lang, font, target-text expansion,
 *  apply-to-selected); source text is read-only reference. */
export type TableMode = "capture" | "translate";

interface RegionTableProps {
  mode: TableMode;
  regions: InstText[];
  selectedId: string | null;
  hoveredId: string | null;
  onSelect: (id: string) => void;
  onHover: (id: string | null) => void;
  onTextChange: (id: string, text: string) => void;
  onTargetChange: (id: string, target: string) => void;
  onDelete: (id: string) => void;
  onOcr: (id: string) => void;
  onToggleDnt: (id: string) => void;
  onTargetLangChange: (id: string, lang: string) => void;
  onSrcLangChange: (id: string, lang: string) => void;
  onFontChange: (id: string, font: string) => void;
  onApplyTargetLang: (ids: string[], lang: string) => void;
  ocrLoading: string | null;
  languages: LanguageOption[];
  defaultTargLang: string;
  /** asset-level source language: regions with no per-region language
   * evidence (digits, symbols) inherit it for display */
  defaultSrcLang?: string | null;
  fontsByLang: Record<string, FontOption[]>;
  familiesByLang?: Record<string, FontFamily[]>;
  onNeedFonts: (lang: string) => void;
  lockedLangs?: Set<string>;
  onToggleLangLock?: (id: string) => void;
  footer?: ReactNode;
}

function confColor(conf: number | null): string {
  if (conf === null) return "text-zinc-500";
  if (conf >= 0.8) return "text-emerald-400";
  if (conf >= 0.6) return "text-amber-400";
  return "text-red-400";
}

function fontName(path: string): string {
  const base = path.split(/[\\/]/).pop() ?? path;
  return base.replace(/\.(ttf|otf|ttc|otc)(#\d+)?$/i, "");
}

// column model: label collapses to `short` below `narrowAt` px
interface Col {
  key: string;
  label: string;
  short?: string;
  w: number;
  min: number;
  narrowAt?: number;
  resizable?: boolean;
}
const ALL_COLS: Col[] = [
  { key: "check", label: "", w: 28, min: 28 },
  { key: "num", label: "#", w: 28, min: 24 },
  { key: "id", label: "ID", w: 38, min: 32 },
  { key: "source", label: "Source", short: "SRC", w: 86, min: 50, narrowAt: 70, resizable: true },
  { key: "srctgt", label: "Source / Target", short: "S/T", w: 220, min: 80, narrowAt: 120, resizable: true },
  { key: "lang", label: "Lang", w: 68, min: 50, resizable: true },
  { key: "font", label: "Font", w: 104, min: 56, resizable: true },
  { key: "conf", label: "Conf", w: 46, min: 40 },
  { key: "actions", label: "", w: 78, min: 78 },
];

// translate drops the per-row target-language dropdown: source + target
// text live together in one column, and the expanded row is where target
// text is edited (bulk target-language changes via the selection bar)
const MODE_COLS: Record<TableMode, string[]> = {
  capture: ["check", "num", "id", "source", "lang", "conf", "actions"],
  translate: ["check", "num", "id", "srctgt", "font", "conf", "actions"],
};

export default function RegionTable({
  mode, regions, selectedId, hoveredId, onSelect, onHover, onTextChange, onTargetChange,
  onDelete, onOcr, onToggleDnt, onTargetLangChange, onSrcLangChange, onFontChange,
  onApplyTargetLang, ocrLoading, languages, defaultTargLang, defaultSrcLang, fontsByLang, familiesByLang, onNeedFonts,
  lockedLangs, onToggleLangLock, footer,
}: RegionTableProps) {
  const COLS = ALL_COLS.filter((c) => MODE_COLS[mode].includes(c.key));
  const rowRefs = useRef<Record<string, HTMLTableRowElement | null>>({});
  const scrollRef = useRef<HTMLDivElement>(null);
  const syncBarRef = useRef<HTMLDivElement>(null);
  const [checked, setChecked] = useState<Set<string>>(new Set());
  const [expandedId, setExpandedId] = useState<string | null>(null);
  const [applyLang, setApplyLang] = useState<string>(defaultTargLang);
  const [colWidths, setColWidths] = useState<Record<string, number>>(
    () => Object.fromEntries(ALL_COLS.map((c) => [c.key, c.w]))
  );
  const resizeRef = useRef<{ key: string; startX: number; startW: number } | null>(null);

  const availableCodes = languages.map((l) => l.code);

  // scroll selected row into view
  useEffect(() => {
    if (selectedId && rowRefs.current[selectedId]) {
      rowRefs.current[selectedId]?.scrollIntoView({ behavior: "smooth", block: "nearest" });
    }
  }, [selectedId]);

  // capture mode: scroll to bottom so latest entries are visible
  useEffect(() => {
    if (mode === "capture" && scrollRef.current) {
      scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
    }
  }, [mode, regions.length]);

  // sync horizontal scroll between table container and sticky bottom scrollbar
  useEffect(() => {
    const el = scrollRef.current;
    const bar = syncBarRef.current;
    if (!el || !bar) return;
    const sync = () => { if (bar.scrollLeft !== el.scrollLeft) bar.scrollLeft = el.scrollLeft; };
    const onBarScroll = () => { if (el.scrollLeft !== bar.scrollLeft) el.scrollLeft = bar.scrollLeft; };
    el.addEventListener("scroll", sync);
    bar.addEventListener("scroll", onBarScroll);
    return () => {
      el.removeEventListener("scroll", sync);
      bar.removeEventListener("scroll", onBarScroll);
    };
  }, []);

  // prefetch font lists for every effective target language in the table
  useEffect(() => {
    const langs = new Set(regions.map((r) => r.target_language ?? defaultTargLang));
    langs.forEach((l) => onNeedFonts(l));
  }, [regions, defaultTargLang, onNeedFonts]);

  // column resize drag
  useEffect(() => {
    const onMove = (e: MouseEvent) => {
      const r = resizeRef.current;
      if (!r) return;
      const col = ALL_COLS.find((c) => c.key === r.key);
      const w = Math.max(col?.min ?? 40, r.startW + (e.clientX - r.startX));
      setColWidths((prev) => ({ ...prev, [r.key]: w }));
    };
    const onUp = () => { resizeRef.current = null; };
    window.addEventListener("mousemove", onMove);
    window.addEventListener("mouseup", onUp);
    return () => {
      window.removeEventListener("mousemove", onMove);
      window.removeEventListener("mouseup", onUp);
    };
  }, []);

  const toggleCheck = (id: string) => {
    setChecked((prev) => {
      const next = new Set(prev);
      if (next.has(id)) next.delete(id);
      else next.add(id);
      return next;
    });
  };
  const allChecked = regions.length > 0 && checked.size === regions.length;
  const toggleAll = () => {
    setChecked(allChecked ? new Set() : new Set(regions.map((r) => r.id)));
  };
  const tableWidth = COLS.reduce((sum, c) => sum + (colWidths[c.key] ?? c.w), 0);

  const translated = regions.filter((r) => r.target_text).length;
  const avgConf = regions.filter((r) => r.confidence !== null).reduce((a, r) => a + (r.confidence ?? 0), 0) /
    (regions.filter((r) => r.confidence !== null).length || 1);

  const headerLabel = (c: Col) => {
    const w = colWidths[c.key] ?? c.w;
    return c.short && c.narrowAt && w < c.narrowAt ? c.short : c.label;
  };

  return (
    <div className={`bezier-card soft-shadow flex h-full flex-col rounded-lg bg-white/60 dark:bg-zinc-900/60 ${footer ? "overflow-visible" : "overflow-hidden"}`}>
      {/* stats bar */}
      <div className="subtext flex items-center justify-between border-b border-zinc-300 px-3 py-2 text-xs text-zinc-600 dark:border-zinc-800 dark:text-zinc-400">
        <span>regions: {regions.length}</span>
        {regions.some((r) => r.confidence !== null) && (
          <span>avg conf {(avgConf * 100).toFixed(0)}%</span>
        )}
      </div>

      {/* apply-to-selected bar (translation work only) */}
      {mode === "translate" && checked.size > 0 && (
        <div className="flex items-center gap-2 border-b border-cyan-300/60 bg-cyan-50 px-3 py-1.5 text-xs dark:border-cyan-900/50 dark:bg-cyan-950/30">
          <span className="text-cyan-700 dark:text-cyan-300">{checked.size} selected</span>
          <div className="w-40">
            <LanguageCombobox
              value={applyLang}
              onChange={setApplyLang}
              availableCodes={availableCodes}
            />
          </div>
          <button
            onClick={() => {
              onApplyTargetLang([...checked], applyLang);
              setChecked(new Set());
            }}
            className="rounded bg-cyan-700 px-2 py-1 font-medium text-white hover:bg-cyan-600"
          >
            Apply target language
          </button>
          <button
            onClick={() => setChecked(new Set())}
            className="rounded px-2 py-1 text-zinc-500 hover:text-zinc-300"
          >
            Clear
          </button>
        </div>
      )}

      {/* table */}
      <div ref={scrollRef} className="bbox-canvas-scroll flex-1 min-h-0">
       <table className="w-full text-sm" style={{ tableLayout: "fixed" }}>
          <colgroup>
            {COLS.map((c) => (
              <col key={c.key} style={{ width: colWidths[c.key] ?? c.w }} />
            ))}
          </colgroup>
          <thead className="sticky top-0 z-10 bg-zinc-100 text-xs text-zinc-500 dark:bg-zinc-900">
            <tr>
              {COLS.map((c) => (
                <th key={c.key} className="relative select-none px-2 py-1 text-left font-medium">
                  {c.key === "check" ? (
                    <input
                      type="checkbox"
                      checked={allChecked}
                      onChange={toggleAll}
                      className="accent-cyan-600"
                      title="Select all"
                    />
                  ) : c.key === "font" && mode === "translate" ? (
                    <span className="flex items-center gap-2">
                      {headerLabel(c)}
                      {regions.some((r) => r.characteristics?.font_style || r.style_profile?.font_family) && (
                        <span className="flex items-center gap-1 text-[10px] text-cyan-700 dark:text-cyan-300">
                          <ScanText size={11} className="shrink-0" />
                          detected
                        </span>
                      )}
                    </span>
                  ) : c.key === "srctgt" && mode === "translate" ? (
                    <span className="flex items-center gap-2">
                      {headerLabel(c)}
                      <span className={`text-[10px] ${translated === regions.length && regions.length > 0 ? "text-emerald-600 dark:text-emerald-400" : "text-red-500 dark:text-red-400"}`}>
                        {translated}/{regions.length}
                      </span>
                    </span>
                  ) : c.key === "lang" && mode === "capture" && lockedLangs && regions.length > 0 && lockedLangs.size === regions.length ? (
                    <span className="flex items-center gap-1">
                      <HiLockClosed size={12} className="text-cyan-600 dark:text-cyan-400" />
                      {headerLabel(c)}
                    </span>
                  ) : (
                    headerLabel(c)
                  )}
                  {c.resizable && (
                    <span
                      onMouseDown={(e) => {
                        e.preventDefault();
                        resizeRef.current = {
                          key: c.key, startX: e.clientX,
                          startW: colWidths[c.key] ?? c.w,
                        };
                      }}
                      className="absolute -right-0.5 top-0 z-10 h-full w-1.5 cursor-col-resize hover:bg-cyan-600/50"
                      title="Drag to resize"
                    />
                  )}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {regions.map((inst) => {
              const isSel = inst.id === selectedId;
              const isHovered = inst.id === hoveredId;
              const isExpanded = inst.id === expandedId;
              const effectiveTarg = inst.target_language ?? defaultTargLang;
              // per-region language, else the asset's source language —
              // digits/symbols carry no language evidence of their own but
              // still belong to the asset's source language
              const effectiveSrc = inst.language ?? inst.detected_language ?? defaultSrcLang ?? null;
              const srcInherited = inst.language == null && inst.detected_language == null;
              const fonts = fontsByLang[effectiveTarg] ?? [];
              const currentFont = inst.style_profile?.font_family ?? null;
              return (
                <Fragment key={inst.id}>
                  <tr
                    ref={(el) => { rowRefs.current[inst.id] = el; }}
                    onClick={() => {
                      onSelect(inst.id);
                      if (mode === "translate") setExpandedId(isExpanded ? null : inst.id);
                    }}
                    onMouseEnter={() => onHover(inst.id)}
                    onMouseLeave={() => onHover(null)}
                    className={`cursor-pointer border-b border-zinc-200 dark:border-zinc-800/50 ${
                      isSel
                        ? "bg-zinc-200/70 dark:bg-zinc-800/50"
                        : isHovered
                        ? "bg-zinc-200/40 dark:bg-zinc-800/30"
                        : "hover:bg-zinc-200/40 dark:hover:bg-zinc-800/30"
                    }`}
                  >
                    <td className="px-2 py-1" onClick={(e) => e.stopPropagation()}>
                      <input
                        type="checkbox"
                        checked={checked.has(inst.id)}
                        onChange={() => toggleCheck(inst.id)}
                        className="accent-cyan-600"
                      />
                    </td>
                    <td className="px-2 py-1 text-xs text-zinc-500">{(inst.reading_order ?? 0) + 1}</td>
                    <td className="truncate px-2 py-1 font-mono text-xs text-zinc-400">
                      <span className="flex items-center gap-1">
                        {inst.id}
                        {inst.tm_suggestion && (
                          <span title={`seen before (${inst.tm_suggestion.method} match, ${(inst.tm_suggestion.score * 100).toFixed(0)}%)`}>
                            <BookmarkCheck size={11} className="shrink-0 text-cyan-600 dark:text-cyan-400" />
                          </span>
                        )}
                      </span>
                    </td>
                    <td className="px-2 py-1">
                      {mode === "capture" ? (
                        <input
                          value={inst.text ?? ""}
                          onChange={(e) => onTextChange(inst.id, e.target.value)}
                          onClick={(e) => e.stopPropagation()}
                          className="w-full rounded bg-transparent px-1 py-0.5 text-xs text-zinc-800 outline-none focus:bg-zinc-200 dark:text-zinc-200 dark:focus:bg-zinc-800"
                          placeholder="—"
                        />
                      ) : (
                        // source (read-only outside Capture) with target
                        // text stacked beneath — click to expand and edit
                        <div className="min-w-0 px-1 py-0.5">
                          <span className="block truncate text-xs text-zinc-700 dark:text-zinc-300" title={inst.text ?? ""}>
                            {inst.text || <span className="text-zinc-600">—</span>}
                          </span>
                          <span
                            className={`block truncate text-xs ${inst.target_text ? "text-emerald-600 dark:text-emerald-300" : "text-zinc-500 dark:text-zinc-600"}`}
                            title={inst.target_text ?? ""}
                          >
                            {inst.target_text || "no translation yet"}
                          </span>
                          {!inst.target_text && inst.tm_suggestion && (
                            <button
                              onClick={(e) => {
                                e.stopPropagation();
                                onTargetChange(inst.id, inst.tm_suggestion!.target_text);
                              }}
                              title={`from memory: "${inst.tm_suggestion.target_text}"`}
                              className="mt-0.5 flex items-center gap-1 rounded bg-cyan-500/15 px-1.5 py-0.5 text-[10px] font-medium text-cyan-700 transition hover:bg-cyan-500/25 dark:text-cyan-300"
                            >
                              <BookmarkCheck size={10} className="shrink-0" />
                              TM {(inst.tm_suggestion.score * 100).toFixed(0)}% — apply
                            </button>
                          )}
                        </div>
                      )}
                    </td>
                    {mode === "capture" && (
                      <td className="px-2 py-1" onClick={(e) => e.stopPropagation()}>
                        <div className="flex items-center gap-0.5">
                          {lockedLangs && onToggleLangLock && (
                            <button
                              onClick={() => onToggleLangLock(inst.id)}
                              title={lockedLangs.has(inst.id) ? "unlock source language" : "lock source language"}
                              className="rounded p-0.5 text-zinc-500 hover:text-cyan-600 dark:text-zinc-400 dark:hover:text-cyan-400"
                            >
                              {lockedLangs.has(inst.id)
                                ? <HiLockClosed size={12} className="text-cyan-600 dark:text-cyan-400" />
                                : <HiLockOpen size={12} />}
                            </button>
                          )}
                          <LanguageCombobox
                            value={effectiveSrc}
                            onChange={(code) => onSrcLangChange(inst.id, code)}
                            availableCodes={availableCodes}
                            placeholder="detect"
                            inherited={srcInherited || inst.language == null}
                            disabled={lockedLangs?.has(inst.id)}
                            compact
                          />
                        </div>
                      </td>
                    )}
                    {mode === "translate" && (
                      <td className="px-2 py-1" onClick={(e) => e.stopPropagation()}>
                        {inst.characteristics?.font_style || inst.style_profile?.font_family ? (
                          <div className="flex items-center gap-2">
                            <ScanText size={11} className="shrink-0 text-cyan-600 dark:text-cyan-400" />
                            <span
                              className="block truncate text-xs text-zinc-500 dark:text-zinc-400"
                              style={(() => {
                                const style: CSSProperties = {};
                                const fontPath = inst.style_profile?.font_family;
                                const fontStyle = inst.characteristics?.font_style ?? "";
                                const fs = fontStyle.toLowerCase();
                                if (fontPath && familiesByLang) {
                                  const lang = inst.target_language ?? defaultTargLang;
                                  const families = familiesByLang[lang] ?? [];
                                  const fam = families.find((f) => f.weights.some((w) => w.path === fontPath) || f.best_path === fontPath);
                                  const weight = fam?.weights.find((w) => w.path === fontPath);
                                  if (weight) {
                                    const wc = weight.weight_class;
                                    if (wc <= 400) style.fontWeight = 400;
                                    else if (wc <= 500) style.fontWeight = 500;
                                    else if (wc <= 600) style.fontWeight = 600;
                                    else if (wc <= 700) style.fontWeight = 700;
                                    else style.fontWeight = 800;
                                    if ((weight.subfamily || "").toLowerCase().includes("italic")) style.fontStyle = "italic";
                                    style.fontFamily = fontNameForPath(fontPath);
                                    loadFontPreview(fontPath, fam?.family ?? "");
                                  } else {
                                    if (fs.includes("bold")) style.fontWeight = 700;
                                    if (fs.includes("italic")) style.fontStyle = "italic";
                                  }
                                } else {
                                  if (fs.includes("bold")) style.fontWeight = 700;
                                  if (fs.includes("italic")) style.fontStyle = "italic";
                                }
                                return style;
                              })()}
                              title={[
                                inst.style_profile?.font_family ? fontName(inst.style_profile.font_family) : null,
                                inst.characteristics?.font_style && inst.characteristics.font_style !== "regular" ? inst.characteristics.font_style : null,
                              ].filter(Boolean).join(" ") || "auto"}
                            >
                              {[
                                inst.style_profile?.font_family ? fontName(inst.style_profile.font_family) : null,
                                inst.characteristics?.font_style && inst.characteristics.font_style !== "regular" ? inst.characteristics.font_style : null,
                              ].filter(Boolean).join(" ") || "auto"}
                            </span>
                          </div>
                        ) : (
                          <span className="text-xs text-zinc-400 dark:text-zinc-600">—</span>
                        )}
                      </td>
                    )}
                    <td className={`px-2 py-1 text-xs ${confColor(inst.confidence)}`}>
                      {inst.confidence !== null ? `${(inst.confidence * 100).toFixed(0)}%` : "man"}
                    </td>
                    <td className="px-2 py-1">
                      <div className="flex items-center gap-1" onClick={(e) => e.stopPropagation()}>
                        {mode === "capture" && (
                          <button
                            onClick={() => onOcr(inst.id)}
                            disabled={ocrLoading === inst.id}
                            title="Recognize text"
                            className="rounded p-1 text-zinc-500 hover:bg-zinc-300 hover:text-cyan-600 dark:hover:bg-zinc-700 dark:hover:text-cyan-400"
                          >
                            {ocrLoading === inst.id ? <Loader2 size={12} className="animate-spin" /> : <ScanText size={12} />}
                          </button>
                        )}
                        <button
                          onClick={() => onToggleDnt(inst.id)}
                          title={inst.dnt ? "Unmark DNT" : "Mark DNT"}
                          className={`rounded p-1 hover:bg-zinc-300 dark:hover:bg-zinc-700 ${inst.dnt ? "text-amber-500 dark:text-amber-400" : "text-zinc-500 hover:text-amber-500 dark:hover:text-amber-400"}`}
                        >
                          {inst.dnt ? <EyeOff size={12} /> : <Eye size={12} />}
                        </button>
                        {mode === "capture" && (
                          <button
                            onClick={() => onDelete(inst.id)}
                            title="Delete region"
                            className="rounded p-1 text-zinc-500 hover:bg-zinc-300 hover:text-red-500 dark:hover:bg-zinc-700 dark:hover:text-red-400"
                          >
                            <Trash2 size={12} />
                          </button>
                        )}
                      </div>
                    </td>
                  </tr>
                  {mode === "translate" && isExpanded && (
                    <tr className="border-b border-zinc-200 bg-zinc-100/80 dark:border-zinc-800/50 dark:bg-zinc-950/60">
                      <td colSpan={COLS.length} className="px-4 py-2">
                        <div className="grid gap-2 md:grid-cols-2">
                          <div>
                            <p className="subtext mb-1 text-[10px] uppercase tracking-wider text-zinc-500 dark:text-zinc-600">
                              source · {effectiveSrc ? langDisplayName(effectiveSrc) : "unknown"}
                            </p>
                            <div className="rounded bg-white px-2 py-1.5 text-xs text-zinc-700 dark:bg-zinc-900 dark:text-zinc-300">
                              {inst.text || <span className="text-zinc-600">no source text</span>}
                            </div>
                          </div>
                          <div>
                            <p className="subtext mb-1 text-[10px] uppercase tracking-wider text-zinc-500 dark:text-zinc-600">
                              target · {langDisplayName(effectiveTarg)}
                            </p>
                            <textarea
                              value={inst.target_text ?? ""}
                              onChange={(e) => onTargetChange(inst.id, e.target.value)}
                              onClick={(e) => e.stopPropagation()}
                              rows={2}
                              placeholder="enter translation…"
                              className={`w-full resize-y rounded border border-zinc-300 bg-white px-2 py-1.5 text-xs outline-none focus:border-cyan-600 dark:border-zinc-800 dark:bg-zinc-900 dark:focus:border-cyan-800 ${
                                inst.target_text ? "text-emerald-600 dark:text-emerald-300" : "text-zinc-600 dark:text-zinc-400"
                              }`}
                            />
                          </div>
                        </div>
                      </td>
                    </tr>
                  )}
                </Fragment>
              );
            })}
            {regions.length === 0 && (
              <tr>
                <td colSpan={COLS.length} className="subtext px-3 py-6 text-center text-xs text-zinc-500 dark:text-zinc-600">
                  No regions detected. Click "Add Region" to draw a bounding box.
                </td>
              </tr>
            )}
            {/* ghost rows to fill remaining space for UI seamlessness */}
            {regions.length > 0 && regions.length < 20 && (
              Array.from({ length: Math.max(3, 20 - regions.length) }).map((_, i) => (
                <tr key={`ghost-${i}`} className="border-b border-zinc-100 dark:border-zinc-900/30" style={{ opacity: 0.3 }}>
                  {COLS.map((c) => (
                    <td key={c.key} className="px-2 py-1 text-xs text-zinc-300 dark:text-zinc-700">&nbsp;</td>
                  ))}
                </tr>
              ))
            )}
          </tbody>
        </table>
      </div>
      {/* synced horizontal scrollbar — pinned to the bottom of the card */}
      <div
        ref={syncBarRef}
        style={{ overflowX: "auto", overflowY: "hidden", flexShrink: 0 }}
      >
        <div style={{ width: tableWidth, height: 1 }} />
      </div>
      {footer && (
        <div className="shrink-0 overflow-visible border-t border-zinc-300 dark:border-zinc-800">
          {footer}
        </div>
      )}
    </div>
  );
}
