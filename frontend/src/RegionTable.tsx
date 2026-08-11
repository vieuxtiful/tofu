import { Fragment, useCallback, useEffect, useRef, useState, type CSSProperties, type ReactNode } from "react";
import { FontFamily, FontOption, InstText, LanguageOption } from "./api";
import { AlertTriangle, AlertCircle, BookmarkCheck, ChevronDown, ChevronLeft, ChevronRight, Eye, EyeOff, GripVertical, Loader2, ScanText, Trash2, AlignLeft, AlignVerticalJustifyCenter, ArrowLeftRight } from "lucide-react";
import { LuReplace } from "react-icons/lu";
import { BsTranslate } from "react-icons/bs";
import { langDisplayName } from "./languageData";
import { textLangMatchesTarget } from "./detectLanguage";
import LanguageCombobox from "./LanguageCombobox";
import { loadFontPreview, weightLabel, fontNameForPath } from "./FontCombobox";
import { fontIdentity } from "./doppelganger";
import { HiLockClosed, HiLockOpen } from "react-icons/hi";
import { FaSearch } from "react-icons/fa";
import { PiArrowsMergeBold } from "react-icons/pi";
import { RiFunctionAiFill, RiFunctionAiLine } from "react-icons/ri";
import { TbLanguageOff, TbLeafFilled, TbAlertSquare, TbAlertSquareFilled } from "react-icons/tb";
import { MdFontDownload, MdOutlineFontDownload } from "react-icons/md";
import type { Theme } from "./theme";
import "./bbox.css";
import AnimatedCaretTextarea from "./AnimatedCaretTextarea";

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
  /** fold several regions into one. cicerone re-reads the union box and
   * falls back to joining the parts in reading order. */
  onMergeRegions?: (ids: string[]) => void;
  mergeLoading?: boolean;
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
  onOrientationToggle?: (id: string) => void;
  onWordOrderToggle?: (id: string) => void;
  /** Refresh visual font evidence.  External catalog lookup is opt-in. */
  onFontMatch?: (id: string, allowExternal?: boolean) => void;
  fontMatchingId?: string | null;
  formerTargLang?: string | null;
  targLang?: string;
  footer?: ReactNode;
  onReorder?: (fromId: string, toId: string) => void;
  onBatchBegin?: () => void;
  onBatchEnd?: () => void;
  /** Region IDs whose source text changed via a Capture-tab merge while they
   *  already carried a translation.  Shown as a re-translation alert icon in
   *  the actions column so the user knows the target text may be stale. */
  retranslationNeededIds?: string[];
  onDismissRetranslation?: (id: string) => void;
  bboxColor?: string;
  hideRegionCounter?: boolean;
  /** Passed rather than read from useTheme(): that hook holds its own state
   * per caller, so a second copy here would not follow the toggle. */
  theme?: Theme;
  sourceSuggestions?: string[];
  /** Open the preview Font Manager dialog (preview-only font override).
   *  The dialog itself lives in App.tsx; this callback opens it. */
  onOpenPreviewFontManager?: () => void;
  /** Current preview font overrides, keyed by region ID. */
  previewFontOverride?: Record<string, string>;
  /** Clear the preview font override for a region (reset to auto). */
  onClearPreviewFont?: (id: string) => void;
  /** Full font families by language, for resolving override labels. */
  fullFamiliesByLang?: Record<string, FontFamily[]>;
}

function confColor(conf: number | null): string {
  if (conf === null) return "text-zinc-500";
  if (conf >= 0.8) return "text-emerald-400";
  if (conf >= 0.6) return "text-amber-400";
  return "text-red-400";
}

/** What ToFU's arbitration read for this region, when it has a reading.
 *
 * Arbitration scores every observation of a region -- each detection pass
 * plus the independent verifier -- and its reading is now loaded into the
 * manifest rather than merely recorded. This attributes it: the marker says
 * which text came from ToFU rather than from the recogniser's first pass.
 *
 * The per-signal breakdown (cross_backend, confidence, stability, geometry,
 * language, glyph, language_model) is deliberately NOT here. Six numbers in
 * a hover tooltip is a debugging readout, not something a localiser can act
 * on, and it buried the one line that matters. It stays where it belongs, in
 * `ocr_provenance.hypothesis.score_breakdown` on the manifest -- written on
 * every region, exported with the project, and readable whenever a decision
 * has to be audited.
 *
 * Returns null when there is nothing to attribute, so the caller can use it
 * as the render condition.
 */
export function arbitrationReading(inst: InstText): string | null {
  if (inst.source_override?.kind === "tofu_arbitration"
      && inst.source_override.text === inst.text && inst.text) {
    return `ToFU read: "${inst.text}"`;
  }
  const correction = inst.ocr_correction;
  const legacyTofuReason = correction?.reason === "versioned cross-engine OCR arbitration"
    || correction?.reason === "arbitration selected the independent verifier; difference is a glyph confusion";
  if (!correction?.applied
      || (correction.correction_resource?.kind !== "tofu_arbitration" && !legacyTofuReason)
      || !correction.corrected_text
      || correction.corrected_text !== inst.text) return null;
  return `ToFU read: "${inst.text}"`;
}

export function groundTruthReading(inst: InstText): string | null {
  if (inst.source_override?.kind === "ground_truth"
      && inst.source_override.text === inst.text && inst.text) {
    return `Ground Truth override: "${inst.text}"`;
  }
  const correction = inst.ocr_correction;
  if (!correction?.applied
      || correction.correction_resource?.kind !== "ground_truth"
      || !correction.corrected_text
      || correction.corrected_text !== inst.text) return null;
  return `Ground Truth override: "${inst.text}"`;
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
  { key: "conf", label: "Conf.", w: 46, min: 40 },
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
  onApplyTargetLang, onMergeRegions, mergeLoading, ocrLoading, languages, defaultTargLang, defaultSrcLang, fontsByLang, familiesByLang, onNeedFonts,
  lockedLangs, onToggleLangLock, onOrientationToggle, onWordOrderToggle, onFontMatch, fontMatchingId, formerTargLang, targLang, footer, onReorder, onBatchBegin, onBatchEnd, retranslationNeededIds, onDismissRetranslation, bboxColor, hideRegionCounter, theme, sourceSuggestions = [],
  onOpenPreviewFontManager, previewFontOverride = {}, onClearPreviewFont, fullFamiliesByLang = {},
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
  const [rejectTooltip, setRejectTooltip] = useState<{ id: string; pulse: number } | null>(null);
  const composingTargetsRef = useRef<Set<string>>(new Set());
  const justCommittedTargetsRef = useRef<Set<string>>(new Set());
  const [sortKey, setSortKey] = useState<"num" | "id" | null>(null);
  const [sortDir, setSortDir] = useState<"asc" | "desc">("asc");
  const [searchQuery, setSearchQuery] = useState("");
  const [searchOpen, setSearchOpen] = useState(false);
  const [draggedId, setDraggedId] = useState<string | null>(null);
  const draggedIdRef = useRef<string | null>(null);
  const [dragOverId, setDragOverId] = useState<string | null>(null);
  const canReorder = mode === "capture" && !!onReorder;

  // pagination — Capture defaults to 15/page, Translate defaults to 10/page
  const pageLimitOptions = mode === "translate" ? [10, 15, 20, 25, 30] : [15, 20, 25, 30];
  const [rowsPerPage, setRowsPerPage] = useState(mode === "translate" ? 10 : 15);
  const [currentPage, setCurrentPage] = useState(0);
  const [rowsOpen, setRowsOpen] = useState(false);
  const rowsDropdownRef = useRef<HTMLDivElement>(null);

  const sortedRegions = (() => {
    let r = regions;
    if (searchQuery.trim()) {
      const q = searchQuery.toLowerCase();
      r = r.filter((inst) => (inst.text ?? "").toLowerCase().includes(q) || (inst.target_text ?? "").toLowerCase().includes(q));
    }
    if (sortKey) {
      r = [...r].sort((a, b) => {
        let cmp: number;
        if (sortKey === "num") {
          const ai = regions.indexOf(a);
          const bi = regions.indexOf(b);
          cmp = ai - bi;
        } else {
          cmp = a.id.localeCompare(b.id, undefined, { numeric: true });
        }
        return sortDir === "asc" ? cmp : -cmp;
      });
    }
    return r;
  })();

  const totalPages = Math.max(1, Math.ceil(sortedRegions.length / rowsPerPage));
  const clampedPage = Math.min(currentPage, totalPages - 1);
  const pageStart = clampedPage * rowsPerPage;
  const pageEnd = Math.min(pageStart + rowsPerPage, sortedRegions.length);
  const paginatedRegions = sortedRegions.slice(pageStart, pageEnd);

  const availableCodes = languages.map((l) => l.code);

  // Basil plating lookup: map semantic_region_id → assigned text so each
  // cube can show what was replaced when cross-cube plating occurred.
  const platingOriginals = (() => {
    const map: Record<string, string> = {};
    for (const r of regions) {
      if (r.semantic_assignment?.semantic_region_id) {
        map[r.semantic_assignment.semantic_region_id] = r.semantic_assignment.text;
      }
    }
    return map;
  })();

  // scroll selected row into view
  useEffect(() => {
    if (selectedId && rowRefs.current[selectedId]) {
      rowRefs.current[selectedId]?.scrollIntoView({ behavior: "smooth", block: "nearest" });
    }
  }, [selectedId]);

  // clamp page when total pages change
  useEffect(() => {
    setCurrentPage((p) => Math.min(p, totalPages - 1));
  }, [totalPages]);

  // capture mode: jump to the last page when new regions are added
  const prevCountRef = useRef(regions.length);
  useEffect(() => {
    if (mode === "capture" && regions.length > prevCountRef.current) {
      setCurrentPage(Math.max(0, totalPages - 1));
    }
    prevCountRef.current = regions.length;
  }, [mode, regions.length, totalPages]);

  // jump to the page containing the selected region
  useEffect(() => {
    if (!selectedId) return;
    const idx = sortedRegions.findIndex((r) => r.id === selectedId);
    if (idx >= 0) setCurrentPage(Math.floor(idx / rowsPerPage));
  }, [selectedId]);

  // close rows-per-page dropdown on outside click
  useEffect(() => {
    if (!rowsOpen) return;
    const onDown = (e: MouseEvent) => {
      if (rowsDropdownRef.current && !rowsDropdownRef.current.contains(e.target as Node)) {
        setRowsOpen(false);
      }
    };
    document.addEventListener("mousedown", onDown);
    return () => document.removeEventListener("mousedown", onDown);
  }, [rowsOpen]);

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
    <div className={`bezier-card soft-shadow region-table-card flex h-full flex-col rounded-lg bg-white/60 dark:bg-zinc-900/60 ${footer || mode === "translate" ? "overflow-visible" : "overflow-hidden"}${hoveredId ? " region-hovering" : ""}`}>
      {mode === "translate" && (
        <h2 className="subtext mx-3 mb-3 mt-3 flex items-center justify-between text-sm font-semibold uppercase tracking-wider text-zinc-500 dark:text-zinc-400">
          <span className="flex items-center gap-2">
            <BsTranslate size={14} />
            Translate
          </span>
          <span className="flex items-center gap-2 normal-case tracking-normal">
            <div className="relative" ref={rowsDropdownRef}>
              <button
                type="button"
                onClick={() => setRowsOpen((v) => !v)}
                className="flex items-center gap-1 rounded-md border border-zinc-300 bg-white px-1.5 py-0.5 text-[10px] text-zinc-600 transition hover:border-cyan-600 dark:border-zinc-700 dark:bg-zinc-800 dark:text-zinc-400"
                title="Rows per page"
              >
                {rowsPerPage}/page
                <ChevronDown size={10} className={`transition-transform duration-200 ${rowsOpen ? "rotate-180" : ""}`} />
              </button>
              <div
                className={`dropdown-morph absolute right-0 top-full z-200 mt-1 w-20 rounded-lg border border-zinc-300 bg-white p-1 dark:border-zinc-700 dark:bg-zinc-900${rowsOpen ? " expanded" : ""}`}
                style={rowsOpen ? { boxShadow: "1px 1px 0 var(--bc-shadow), 2px 2px 6px rgba(0,0,0,0.06)" } : undefined}
              >
                {pageLimitOptions.map((n) => (
                  <button
                    key={n}
                    type="button"
                    onClick={() => { setRowsPerPage(n); setCurrentPage(0); setRowsOpen(false); }}
                    className={`flex w-full rounded-md px-2 py-1 text-[10px] transition hover:bg-zinc-100 dark:hover:bg-zinc-800 ${rowsPerPage === n ? "bg-zinc-100 font-medium text-zinc-900 dark:bg-zinc-800 dark:text-zinc-100" : "text-zinc-600 dark:text-zinc-400"}`}
                  >
                    {n}/page
                  </button>
                ))}
              </div>
            </div>
            {totalPages > 1 && (
              <span className="flex items-center gap-0.5">
                <button
                  onClick={() => setCurrentPage((p) => Math.max(0, p - 1))}
                  disabled={clampedPage === 0}
                  title="Previous page"
                  className="rounded-sm p-0.5 text-zinc-500 hover:text-cyan-600 disabled:opacity-30 dark:text-zinc-400 dark:hover:text-cyan-400"
                >
                  <ChevronLeft size={14} />
                </button>
                <span className="tabular-nums">{clampedPage + 1}/{totalPages}</span>
                <button
                  onClick={() => setCurrentPage((p) => Math.min(totalPages - 1, p + 1))}
                  disabled={clampedPage >= totalPages - 1}
                  title="Next page"
                  className="rounded-sm p-0.5 text-zinc-500 hover:text-cyan-600 disabled:opacity-30 dark:text-zinc-400 dark:hover:text-cyan-400"
                >
                  <ChevronRight size={14} />
                </button>
              </span>
            )}
          </span>
        </h2>
      )}
      {/* stats bar */}
      {mode === "capture" && (
        <div className="subtext flex items-center justify-between border-b border-zinc-300 px-3 py-2 text-xs text-zinc-600 dark:border-zinc-800 dark:text-zinc-400">
          <span>regions: {regions.length}</span>
          <div className="flex items-center gap-2">
            {/* Detection already joins the words of a line on its own and
              * declines where the geometry is ambiguous (across a column
              * gutter, over a gap wider than a word space). This is the
              * manual door for those, and for groupings only a person knows
              * are one unit. Two is the minimum that means anything.
              *
              * Reserved in a fixed-width slot (rather than only rendering
              * when checked.size >= 2) so the page-number control beside
              * it never jumps left/right as the button appears and
              * disappears. */}
            <span className="flex w-[96px] shrink-0 justify-end">
              {onMergeRegions && checked.size >= 2 && (
                <button
                  onClick={() => {
                    onMergeRegions([...checked]);
                    setChecked(new Set());
                  }}
                  disabled={!!mergeLoading}
                  title={`merge ${checked.size} regions into one`}
                  className="flex items-center gap-1 rounded-lg bg-zinc-800 px-2 py-1 text-xs font-medium text-zinc-100 transition hover:bg-zinc-700 disabled:opacity-40 dark:bg-zinc-100 dark:text-zinc-900 dark:hover:bg-white"
                >
                  {mergeLoading
                    ? <Loader2 size={11} className="animate-spin" />
                    : <PiArrowsMergeBold size={11} />}
                  merge {checked.size}
                </button>
              )}
            </span>
            <div className="relative" ref={rowsDropdownRef}>
              <button
                type="button"
                onClick={() => setRowsOpen((v) => !v)}
                className="flex items-center gap-1 rounded-md border border-zinc-300 bg-white px-1.5 py-0.5 text-[10px] text-zinc-600 transition hover:border-cyan-600 dark:border-zinc-700 dark:bg-zinc-800 dark:text-zinc-400"
                title="Rows per page"
              >
                {rowsPerPage}/page
                <ChevronDown size={10} className={`transition-transform duration-200 ${rowsOpen ? "rotate-180" : ""}`} />
              </button>
              <div
                className={`dropdown-morph absolute right-0 top-full z-200 mt-1 w-20 rounded-lg border border-zinc-300 bg-white p-1 dark:border-zinc-700 dark:bg-zinc-900${rowsOpen ? " expanded" : ""}`}
                style={rowsOpen ? { boxShadow: "1px 1px 0 var(--bc-shadow), 2px 2px 6px rgba(0,0,0,0.06)" } : undefined}
              >
                {pageLimitOptions.map((n) => (
                  <button
                    key={n}
                    type="button"
                    onClick={() => { setRowsPerPage(n); setCurrentPage(0); setRowsOpen(false); }}
                    className={`flex w-full rounded-md px-2 py-1 text-[10px] transition hover:bg-zinc-100 dark:hover:bg-zinc-800 ${rowsPerPage === n ? "bg-zinc-100 font-medium text-zinc-900 dark:bg-zinc-800 dark:text-zinc-100" : "text-zinc-600 dark:text-zinc-400"}`}
                  >
                    {n}/page
                  </button>
                ))}
              </div>
            </div>
            {totalPages > 1 && (
              <span className="flex items-center gap-0.5">
                <button
                  onClick={() => setCurrentPage((p) => Math.max(0, p - 1))}
                  disabled={clampedPage === 0}
                  title="Previous page"
                  className="rounded-sm p-0.5 text-zinc-500 hover:text-cyan-600 disabled:opacity-30 dark:text-zinc-400 dark:hover:text-cyan-400"
                >
                  <ChevronLeft size={14} />
                </button>
                <span className="tabular-nums">{clampedPage + 1}/{totalPages}</span>
                <button
                  onClick={() => setCurrentPage((p) => Math.min(totalPages - 1, p + 1))}
                  disabled={clampedPage >= totalPages - 1}
                  title="Next page"
                  className="rounded-sm p-0.5 text-zinc-500 hover:text-cyan-600 disabled:opacity-30 dark:text-zinc-400 dark:hover:text-cyan-400"
                >
                  <ChevronRight size={14} />
                </button>
              </span>
            )}
          </div>
        </div>
      )}

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
            className="rounded-sm bg-cyan-700 px-2 py-1 font-medium text-white hover:bg-cyan-600"
          >
            Apply target language
          </button>
          <button
            onClick={() => setChecked(new Set())}
            className="rounded-sm px-2 py-1 text-zinc-500 hover:text-zinc-300"
          >
            Clear
          </button>
        </div>
      )}

      {/* table */}
      <div ref={scrollRef} className={`bbox-canvas-scroll flex-1 min-h-0${mode === "translate" ? " region-table-scroll" : ""}`}>
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
                  ) : c.key === "conf" && regions.some((r) => r.confidence !== null) ? (
                    <span className="flex items-center gap-2">
                      {headerLabel(c)}
                      <span className={`text-[10px] ${confColor(avgConf)}`}>
                        {(avgConf * 100).toFixed(0)}%
                      </span>
                    </span>
                  ) : c.key === "lang" && mode === "capture" && lockedLangs && regions.length > 0 && lockedLangs.size === regions.length ? (
                    <span className="flex items-center gap-1">
                      <HiLockClosed size={12} className="text-cyan-600 dark:text-cyan-400" />
                      {headerLabel(c)}
                    </span>
                  ) : c.key === "num" || c.key === "id" ? (
                    // The region count rides beside ID rather than up in the
                    // Translate heading, alongside the other header readouts
                    // (Source/Target, font detected, average confidence). It
                    // sits OUTSIDE the sort button on purpose: it is a
                    // readout, and clicking a number to re-sort the table by
                    // something else reads as a bug.
                    <span className="flex items-center gap-1">
                      <button
                        onClick={() => {
                          if (sortKey === c.key) {
                            setSortDir((d) => d === "asc" ? "desc" : "asc");
                          } else {
                            setSortKey(c.key as "num" | "id");
                            setSortDir("asc");
                          }
                        }}
                        className="group flex items-center gap-0.5 hover:text-zinc-700 dark:hover:text-zinc-300"
                        title={`sort by ${c.label}`}
                      >
                        {headerLabel(c)}
                        <ChevronDown
                          size={10}
                          className={`transition-opacity ${
                            sortKey === c.key
                              ? "opacity-100"
                              : "opacity-0 group-hover:opacity-50"
                          }`}
                          style={{ transform: sortKey === c.key && sortDir === "desc" ? "rotate(180deg)" : "none" }}
                        />
                      </button>
                      {c.key === "id" && mode === "translate" && !hideRegionCounter && (
                        <span
                          className="text-[10px] font-medium normal-case tracking-normal"
                          style={{ color: bboxColor ?? "#22d3ee" }}
                          title={`${regions.length} region${regions.length === 1 ? "" : "s"}`}
                        >
                          {regions.length}
                        </span>
                      )}
                    </span>
                  ) : c.key === "source" || c.key === "srctgt" ? (
                    <span className="flex items-center gap-1">
                      <button
                        onClick={() => setSearchOpen((v) => !v)}
                        className="text-zinc-400 hover:text-cyan-600 dark:hover:text-cyan-400"
                        title="search source text"
                      >
                        <FaSearch size={9} />
                      </button>
                      {headerLabel(c)}
                      {searchOpen && (
                        <input
                          autoFocus
                          value={searchQuery}
                          onChange={(e) => setSearchQuery(e.target.value)}
                          onClick={(e) => e.stopPropagation()}
                          onKeyDown={(e) => { if (e.key === "Escape") { setSearchOpen(false); setSearchQuery(""); } }}
                          placeholder="filter…"
                          className="w-16 rounded-sm border border-zinc-300 bg-white px-1 py-0.5 text-[10px] text-zinc-700 outline-hidden focus:border-cyan-600 dark:border-zinc-700 dark:bg-zinc-800 dark:text-zinc-300"
                        />
                      )}
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
            {paginatedRegions.map((inst, pageRowIndex) => {
              const rowIndex = pageStart + pageRowIndex;
              const isSel = inst.id === selectedId;
              const isHovered = inst.id === hoveredId;
              const isExpanded = inst.id === expandedId;
              const effectiveTarg = inst.target_language ?? defaultTargLang;
              // per-region language, else the asset's source language —
              // digits/symbols carry no language evidence of their own but
              // still belong to the asset's source language
              const effectiveSrc = inst.language ?? inst.detected_language ?? defaultSrcLang ?? null;
              const srcInherited = inst.language == null && inst.detected_language == null;
              const sourceProvenanceIcon = groundTruthReading(inst)
                ? <TbLeafFilled size={11} className="shrink-0 text-emerald-600 dark:text-emerald-400" />
                : arbitrationReading(inst)
                  ? (theme === "dark"
                    ? <RiFunctionAiFill size={11} className="shrink-0 text-amber-600 dark:text-amber-400" />
                    : <RiFunctionAiLine size={11} className="shrink-0 text-amber-600 dark:text-amber-400" />)
                  : null;
              const sourceProvenanceTitle = groundTruthReading(inst) ?? arbitrationReading(inst) ?? undefined;
              const wrongTargetLanguage = mode === "translate"
                && Boolean(inst.target_text?.trim() && effectiveTarg && !textLangMatchesTarget(inst.target_text, effectiveTarg));
              const fonts = fontsByLang[effectiveTarg] ?? [];
              const currentFont = inst.style_profile?.font_family ?? null;
              return (
                <Fragment key={inst.id}>
                  <tr
                    ref={(el) => { rowRefs.current[inst.id] = el; }}
                    draggable={canReorder}
                    onDragStart={(e) => {
                      if (!canReorder) return;
                      draggedIdRef.current = inst.id;
                      setDraggedId(inst.id);
                      e.dataTransfer.effectAllowed = "move";
                      e.dataTransfer.setData("text/plain", inst.id);
                    }}
                    onDragOver={(e) => {
                      if (!canReorder) return;
                      e.preventDefault();
                      e.dataTransfer.dropEffect = "move";
                      if (draggedIdRef.current && inst.id !== draggedIdRef.current) setDragOverId(inst.id);
                    }}
                    onDragLeave={() => {
                      if (dragOverId === inst.id) setDragOverId(null);
                    }}
                    onDrop={(e) => {
                      if (!canReorder) return;
                      e.preventDefault();
                      const fromId = draggedIdRef.current;
                      if (fromId && inst.id !== fromId) onReorder!(fromId, inst.id);
                      draggedIdRef.current = null;
                      setDraggedId(null);
                      setDragOverId(null);
                    }}
                    onDragEnd={() => { draggedIdRef.current = null; setDraggedId(null); setDragOverId(null); }}
                    onClick={() => {
                      onSelect(inst.id);
                      if (mode === "translate") setExpandedId(isExpanded ? null : inst.id);
                    }}
                    onMouseEnter={() => onHover(inst.id)}
                    onMouseLeave={() => onHover(null)}
                    className={`cursor-pointer border-b border-zinc-200 dark:border-zinc-800/50 ${
                      mode === "translate" ? "region-row " : ""
                    }${
                      draggedId === inst.id
                        ? "opacity-40"
                        : dragOverId === inst.id
                        ? "border-t-2 border-t-cyan-500 bg-cyan-50/50 dark:bg-cyan-950/20"
                        : isSel
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
                    {/* ``regions`` is supplied in visual reading order by
                        App.  Persisted reading_order can be stale on older
                        manifests, but the immutable rN identity stays in its
                        own column and is never remapped. */}
                    <td className="px-2 py-1 text-xs text-zinc-500">
                      <span className={`flex items-center gap-0.5 ${canReorder ? "cursor-grab active:cursor-grabbing" : ""}`}>
                        {canReorder && <GripVertical size={10} className="shrink-0 text-zinc-400 dark:text-zinc-600" />}
                        {rowIndex + 1}
                      </span>
                    </td>
                    <td className="truncate px-2 py-1 font-mono text-xs text-zinc-400">
                      <span className="flex items-center gap-1">
                        {inst.id}
                        {inst.tm_suggestion && (
                          <span title={`seen before (${inst.tm_suggestion.method} match, ${(inst.tm_suggestion.score * 100).toFixed(0)}%)`}>
                            <BookmarkCheck size={11} className="shrink-0 text-cyan-600 dark:text-cyan-400" />
                          </span>
                        )}
                        {mode === "capture" && groundTruthReading(inst) && (
                          <span title={groundTruthReading(inst)!}>
                            <TbLeafFilled size={11} className="shrink-0 text-emerald-600 dark:text-emerald-400" />
                          </span>
                        )}
                        {mode === "capture" && !groundTruthReading(inst) && arbitrationReading(inst) && (
                          <span title={arbitrationReading(inst)!}>
                            {theme === "dark"
                              ? <RiFunctionAiFill size={11} className="shrink-0 text-amber-600 dark:text-amber-400" />
                              : <RiFunctionAiLine size={11} className="shrink-0 text-amber-600 dark:text-amber-400" />}
                          </span>
                        )}
                        {mode === "translate" && wrongTargetLanguage && (
                          <span title="Target text is in the wrong language">
                            <TbLanguageOff size={12} className="shrink-0 text-red-500 dark:text-red-400" />
                          </span>
                        )}
                      </span>
                    </td>
                    <td className="px-2 py-1">
                      {mode === "capture" ? (
                        <div className="flex min-w-0 items-center gap-1">
                        <AnimatedCaretTextarea
                          label={`Source text for ${inst.id}`}
                          value={inst.text ?? ""}
                          onChange={(text) => onTextChange(inst.id, text)}
                          onFocus={onBatchBegin}
                          onBlur={onBatchEnd}
                          onClick={(event) => event.stopPropagation()}
                          rows={1}
                          expandable
                          suggestions={/^(ja|zh|ko)(-|$)/i.test(effectiveSrc ?? "") ? sourceSuggestions : []}
                          language={effectiveSrc}
                          className={`capture-source-caret min-w-0 flex-1 text-xs ${
                            groundTruthReading(inst)
                              ? "ground-truth-value source-override-ground-truth"
                              : arbitrationReading(inst)
                              ? "tofu-value source-override-tofu"
                              : ""
                          }`}
                          placeholder="—"
                        />
                        </div>
                      ) : (
                        // source (read-only outside Capture) with target
                        // text stacked beneath — click to expand and edit
                        <div className="min-w-0 px-1 py-0.5">
                          <span className={`flex items-center gap-1 text-xs ${
                            groundTruthReading(inst)
                              ? "source-override-ground-truth rounded-sm px-1 text-emerald-600 dark:text-emerald-400"
                              : arbitrationReading(inst)
                              ? "source-override-tofu rounded-sm px-1 text-amber-600 dark:text-amber-400"
                              : "text-zinc-700 dark:text-zinc-300"
                          }`} title={inst.text ?? ""}>
                            <span className="min-w-0 flex-1 truncate">{inst.text || <span className="text-zinc-600">—</span>}</span>
                            {sourceProvenanceIcon && <span title={sourceProvenanceTitle}>{sourceProvenanceIcon}</span>}
                          </span>
                          {inst.semantic_assignment && inst.semantic_assignment.semantic_region_id && inst.semantic_assignment.semantic_region_id !== inst.id ? (
                            <span className="block truncate text-xs text-emerald-600 dark:text-emerald-300" title={inst.target_text ?? ""}>
                              *{inst.target_text} <LuReplace size={10} className="inline shrink-0 text-cyan-600 dark:text-cyan-400" /> ({platingOriginals[inst.id] ?? inst.target_text})
                            </span>
                          ) : (
                            <span
                              className={`block truncate text-xs ${inst.target_text ? "text-emerald-600 dark:text-emerald-300" : "text-zinc-500 dark:text-zinc-600"}`}
                              title={inst.target_text ?? ""}
                            >
                              {inst.target_text || "no translation yet"}
                            </span>
                          )}
                          {!inst.target_text && inst.tm_suggestion && (
                            <button
                              onClick={(e) => {
                                e.stopPropagation();
                                onTargetChange(inst.id, inst.tm_suggestion!.target_text);
                              }}
                              title={`from memory: "${inst.tm_suggestion.target_text}"`}
                              className="mt-0.5 flex items-center gap-1 rounded-sm bg-cyan-500/15 px-1.5 py-0.5 text-[10px] font-medium text-cyan-700 transition hover:bg-cyan-500/25 dark:text-cyan-300"
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
                              className="rounded-sm p-0.5 text-zinc-500 hover:text-cyan-600 dark:text-zinc-400 dark:hover:text-cyan-400"
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
                    {mode === "translate" && (() => {
                      // Resolved through doppelganger.fontIdentity(), the same
                      // ladder TargetPreviewCanvas paints with.  This column
                      // used to run its own `explicit ?? resolved_font_family`
                      // and so skipped the nearest-neighbour rung entirely,
                      // listing la-rue-sans-nom as "Arial Regular" beside a
                      // preview drawn in Baskerville Old Face.
                      const identity = fontIdentity(inst, { familiesByLang, defaultTargLang });
                      const fontStyle = inst.characteristics?.font_style ?? "";
                      const fs = fontStyle.toLowerCase();

                      const style: CSSProperties = {};
                      if (identity.cssFontFamily && identity.path) {
                        // The alias carries the face's own weight and slant,
                        // so asking for them again would double-apply them.
                        style.fontFamily = identity.cssFontFamily;
                        loadFontPreview(identity.path);
                      } else {
                        if (fs.includes("bold")) style.fontWeight = 700;
                        if (fs.includes("italic")) style.fontStyle = "italic";
                      }

                      const inferred = identity.provenance === "nearest_neighbor";
                      const title = inferred
                        ? `${identity.label} — closest installed face by glyph shape. `
                          + `The render keeps its own auto font until you accept this below.`
                        : identity.label;
                      const hasContent = fontStyle || identity.path;
                      return (
                        <td className="px-2 py-1" onClick={(e) => e.stopPropagation()}>
                          {hasContent ? (
                            <div className="flex items-center gap-2">
                              <ScanText size={11} className="shrink-0 text-cyan-600 dark:text-cyan-400" />
                              <span className="block truncate text-xs text-zinc-500 dark:text-zinc-400" style={style} title={title}>
                                {identity.label}
                                {/* an inferred face is not a selected one, and
                                    the column must not imply otherwise */}
                                {inferred && <span className="ml-1 text-amber-600 dark:text-amber-500">~</span>}
                              </span>
                            </div>
                          ) : (
                            <span className="text-xs text-zinc-400 dark:text-zinc-600">—</span>
                          )}
                        </td>
                      );
                    })()}
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
                            className="rounded-sm p-1 text-zinc-500 hover:bg-zinc-300 hover:text-cyan-600 dark:hover:bg-zinc-700 dark:hover:text-cyan-400"
                          >
                            {ocrLoading === inst.id ? <Loader2 size={12} className="animate-spin" /> : <ScanText size={12} />}
                          </button>
                        )}
                        <button
                          onClick={() => onToggleDnt(inst.id)}
                          title={inst.dnt ? "Unmark DNT" : "Mark DNT"}
                          className={`rounded-sm p-1 hover:bg-zinc-300 dark:hover:bg-zinc-700 ${inst.dnt ? "text-amber-500 dark:text-amber-400" : "text-zinc-500 hover:text-amber-500 dark:hover:text-amber-400"}`}
                        >
                          {inst.dnt ? <EyeOff size={12} /> : <Eye size={12} />}
                        </button>
                        {mode === "capture" && (
                          <button
                            onClick={() => onDelete(inst.id)}
                            title="Delete region"
                            className="rounded-sm p-1 text-zinc-500 hover:bg-zinc-300 hover:text-red-500 dark:hover:bg-zinc-700 dark:hover:text-red-400"
                          >
                            <Trash2 size={12} />
                          </button>
                        )}
                        {retranslationNeededIds?.includes(inst.id) && (
                          <button
                            onClick={() => onDismissRetranslation?.(inst.id)}
                            title="re-translation needed — source text changed by a merge; click to dismiss"
                            className="rounded-sm p-1 text-amber-600 hover:bg-amber-100 dark:text-amber-400 dark:hover:bg-amber-900/30"
                          >
                            {theme === "dark"
                              ? <TbAlertSquareFilled size={12} className="shrink-0" />
                              : <TbAlertSquare size={12} className="shrink-0" />}
                          </button>
                        )}
                      </div>
                    </td>
                  </tr>
                  {mode === "translate" && (
                    <tr className={`translate-detail-row ${isExpanded ? "expanded border-b border-zinc-200 dark:border-zinc-800/50" : ""}`}>
                      <td colSpan={COLS.length} className="p-0">
                        <div className={`translate-detail-morph bg-zinc-100/80 dark:bg-zinc-950/60${isExpanded ? " expanded" : ""}`}>
                        <div className="px-4 py-2">
                        <div className="grid gap-2 md:grid-cols-2">
                          <div>
                            <p className="subtext mb-1 text-[10px] uppercase tracking-wider text-zinc-500 dark:text-zinc-600">
                              source · {effectiveSrc ? langDisplayName(effectiveSrc) : "unknown"}
                            </p>
                            <AnimatedCaretTextarea
                              label={`Source text for ${inst.id}`}
                              value={inst.text ?? ""}
                              onChange={(text) => onTextChange(inst.id, text)}
                              onFocus={onBatchBegin}
                              onBlur={onBatchEnd}
                              onClick={(event) => event.stopPropagation()}
                              rows={1}
                              placeholder="no source text"
                              suggestions={/^(ja|zh|ko)(-|$)/i.test(effectiveSrc ?? "") ? sourceSuggestions : []}
                              language={effectiveSrc}
                              trailingIcon={sourceProvenanceIcon ? <span title={sourceProvenanceTitle}>{sourceProvenanceIcon}</span> : null}
                              className={
                                groundTruthReading(inst)
                                  ? "ground-truth-value source-override-ground-truth"
                                  : arbitrationReading(inst)
                                  ? "tofu-value source-override-tofu"
                                  : ""
                              }
                            />
                          </div>
                          <div>
                            <div className="mb-1 flex items-center justify-between">
                              <p className="subtext text-[10px] uppercase tracking-wider text-zinc-500 dark:text-zinc-600">
                                target · {langDisplayName(effectiveTarg)}
                              </p>
                              <div className="flex items-center gap-0.5">
                              {onOrientationToggle && (
                                <button
                                  onClick={(e) => { e.stopPropagation(); onOrientationToggle(inst.id); }}
                                  title={inst.style_profile?.target_orientation === "vertical" ? "vertical text — click for horizontal" : "horizontal text — click for vertical"}
                                  className={`flex items-center gap-1 rounded px-1.5 py-0.5 text-[10px] transition ${
                                    inst.style_profile?.target_orientation === "vertical"
                                      ? "bg-cyan-500/15 text-cyan-700 dark:text-cyan-300"
                                      : "text-zinc-500 hover:bg-zinc-200 dark:text-zinc-400 dark:hover:bg-zinc-800"
                                  }`}
                                >
                                  {inst.style_profile?.target_orientation === "vertical"
                                    ? <AlignVerticalJustifyCenter size={12} />
                                    : <AlignLeft size={12} />}
                                </button>
                              )}
                              {onWordOrderToggle && (
                                <button
                                  onClick={(e) => { e.stopPropagation(); onWordOrderToggle(inst.id); }}
                                  disabled={inst.style_profile?.target_orientation !== "vertical"}
                                  title={inst.style_profile?.target_orientation !== "vertical" ? "only available for vertical text" : inst.style_profile?.word_order === "rtl" ? "right-to-left word order — click to reset" : "reverse word order (right-to-left)"}
                                  className={`flex items-center gap-1 rounded px-1.5 py-0.5 text-[10px] transition ${
                                    inst.style_profile?.target_orientation !== "vertical"
                                      ? "cursor-not-allowed text-zinc-300 dark:text-zinc-700"
                                      : inst.style_profile?.word_order === "rtl"
                                        ? "bg-cyan-500/15 text-cyan-700 dark:text-cyan-300"
                                        : "text-zinc-500 hover:bg-zinc-200 dark:text-zinc-400 dark:hover:bg-zinc-800"
                                  }`}
                                >
                                  <ArrowLeftRight size={12} />
                                </button>
                              )}
                              </div>
                            </div>
                            <div className="relative">
                              <AnimatedCaretTextarea
                                label={`Target text for ${inst.id}`}
                                value={inst.target_text ?? ""}
                                onFocus={onBatchBegin}
                                onBlur={() => {
                                  if (rejectTooltip?.id === inst.id) setRejectTooltip(null);
                                  onBatchEnd?.();
                                }}
                                onChange={(val, meta) => {
                                  // Flag a wrong-language entry, never delete it.
                                  // This used to clear the field on mismatch,
                                  // which threw away real typing on a heuristic.
                                  onTargetChange(inst.id, val);
                                  if (meta?.isComposing || composingTargetsRef.current.has(inst.id)) return;
                                  if (justCommittedTargetsRef.current.delete(inst.id)) return;
                                  if (val.trim() && effectiveTarg && !textLangMatchesTarget(val, effectiveTarg)) {
                                    setRejectTooltip((current) => ({ id: inst.id, pulse: (current?.pulse ?? 0) + 1 }));
                                  } else if (rejectTooltip?.id === inst.id) {
                                    setRejectTooltip(null);
                                  }
                                }}
                                onClick={(e) => e.stopPropagation()}
                                rows={1}
                                placeholder="enter translation…"
                                language={effectiveTarg}
                                statusMessage={rejectTooltip?.id === inst.id ? "incorrect target language" : null}
                                statusPulse={rejectTooltip?.id === inst.id ? rejectTooltip.pulse : 0}
                                onCompositionStateChange={(composing, committedValue) => {
                                  if (composing) {
                                    composingTargetsRef.current.add(inst.id);
                                    return;
                                  }
                                  composingTargetsRef.current.delete(inst.id);
                                  justCommittedTargetsRef.current.add(inst.id);
                                  setTimeout(() => justCommittedTargetsRef.current.delete(inst.id), 0);
                                  if (committedValue.trim() && effectiveTarg && !textLangMatchesTarget(committedValue, effectiveTarg)) {
                                    setRejectTooltip((current) => ({ id: inst.id, pulse: (current?.pulse ?? 0) + 1 }));
                                  } else if (rejectTooltip?.id === inst.id) {
                                    setRejectTooltip(null);
                                  }
                                }}
                                className={`${
                                  formerTargLang && inst.target_text && inst.target_language === formerTargLang
                                    ? "stale-target"
                                    : ""
                                } ${
                                  inst.target_text ? "has-value" : ""
                                }`}
                              />
                              {formerTargLang && inst.target_text && inst.target_language === formerTargLang && (
                                <div className="group pointer-events-none absolute inset-y-0 right-0 flex items-center pr-2">
                                  <AlertCircle size={14} className="text-red-500 dark:text-red-400" />
                                  <div className="pointer-events-none absolute bottom-full right-0 mb-2 hidden whitespace-nowrap rounded-lg border border-zinc-300 bg-white px-3 py-2 text-xs text-zinc-700 shadow-lg group-hover:block dark:border-zinc-700 dark:bg-zinc-900 dark:text-zinc-300 z-50">
                                    <span className="flex items-center gap-1.5">
                                      <AlertCircle size={12} className="shrink-0 text-red-500" />
                                      <span>must edit to new target language</span>
                                    </span>
                                  </div>
                                </div>
                              )}
                            </div>
                          </div>
                        </div>
                        {/* Font preview: show the target text rendered in the
                            resolved font, mirroring the Render step's preview.
                            If a preview font override is set for this region,
                            it takes precedence so the user sees what they're
                            trying. */}
                        {(() => {
                          const sp = inst.style_profile;
                          const overridePath = previewFontOverride[inst.id];
                          const identity = fontIdentity(inst, { familiesByLang: familiesByLang ?? {}, defaultTargLang: targLang ?? "en" });
                          const fontPath = overridePath ?? identity.path;
                          const fontFamily = fontPath ? fontNameForPath(fontPath) : identity.cssFontFamily;
                          if (fontPath) loadFontPreview(fontPath);
                          // The detected source-text color (sp?.color) is intentionally
                          // NOT applied here: it can be a dark color on the dark preview
                          // background (or a light color on the light one), rendering the
                          // preview text invisible. The preview is about the font shape;
                          // the readable Tailwind color classes on the <p> below win.
                          const previewStyle: CSSProperties = {
                            fontFamily: fontFamily ?? undefined,
                            fontStyle: sp?.italic ? "italic" : undefined,
                            textDecoration: sp?.underline ? "underline" : undefined,
                          };
                          if (sp?.font_weight) {
                            const fw = sp.font_weight.toLowerCase();
                            if (fw.includes("bold")) previewStyle.fontWeight = 700;
                            else if (fw.includes("light")) previewStyle.fontWeight = 300;
                          }
                          return (
                            <div className="mt-2 rounded-sm bg-zinc-100 p-2 dark:bg-zinc-950">
                              <p className="subtext mb-0.5 text-[10px] uppercase tracking-wider text-zinc-500">preview</p>
                              <p className="text-sm text-zinc-800 dark:text-zinc-100" style={previewStyle}>
                                {inst.target_text || "(no translation)"}
                              </p>
                            </div>
                          );
                        })()}
                        {(() => {
                          const match = inst.font_match;
                          const licensed = [
                            ...(match?.candidates ?? []),
                            ...(match?.external_candidates ?? []),
                          ].filter((candidate) => candidate.license === "commercial");
                          const substitute = match?.recommended_substitute;
                          const matching = fontMatchingId === inst.id;
                          return (
                            <div className="mt-2 rounded-sm border border-zinc-200 bg-white/70 px-2 py-1.5 dark:border-zinc-800 dark:bg-[#202021]">
                              <div className="flex flex-wrap items-center justify-between gap-2">
                                <div className="flex min-w-0 items-center gap-1.5 text-[10px]">
                                  <ScanText size={12} className="shrink-0 text-cyan-600 dark:text-cyan-400" />
                                  <span className="font-medium text-zinc-700 dark:text-zinc-300">Glyph-shape font match</span>
                                  {match ? (
                                    <span className={match.status === "matched" ? "text-emerald-600 dark:text-emerald-300" : "text-amber-600 dark:text-amber-300"}>
                                      {Math.round(match.confidence * 100)}% {match.status === "matched" ? "evidence" : "— review"}
                                    </span>
                                  ) : <span className="text-zinc-500 dark:text-zinc-500">not analyzed</span>}
                                </div>
                                <div className="flex items-center gap-1">
                                  {onOpenPreviewFontManager && (
                                    <button
                                      onClick={(e) => { e.stopPropagation(); onOpenPreviewFontManager(); }}
                                      title="Preview font from Font Manager (preview only)"
                                      aria-label="Open preview Font Manager"
                                      className="shrink-0 rounded-sm border border-transparent p-1 text-zinc-500 transition hover:border-zinc-300 hover:bg-zinc-200 hover:text-zinc-700 dark:hover:border-zinc-700 dark:hover:bg-zinc-800 dark:hover:text-zinc-300"
                                    >
                                      {theme === "dark" ? <MdFontDownload size={13} /> : <MdOutlineFontDownload size={13} />}
                                    </button>
                                  )}
                                  <button
                                    onClick={(e) => { e.stopPropagation(); onFontMatch?.(inst.id, false); }}
                                    disabled={matching || !onFontMatch}
                                    title="Compare the detected glyph silhouettes with installed fonts; it never changes your selected font automatically."
                                    className="rounded-sm px-1.5 py-0.5 text-[10px] text-cyan-700 transition hover:bg-cyan-500/15 disabled:opacity-50 dark:text-cyan-300"
                                  >
                                    {matching ? <Loader2 size={11} className="animate-spin" /> : "Analyze"}
                                  </button>
                                  <button
                                    onClick={(e) => { e.stopPropagation(); onFontMatch?.(inst.id, true); }}
                                    disabled={matching || !onFontMatch}
                                    title="Optional: search a configured font catalog. ToFU will ask before sending this text crop outside the workspace."
                                    className="rounded-sm px-1.5 py-0.5 text-[10px] text-zinc-600 transition hover:bg-zinc-200 disabled:opacity-50 dark:text-zinc-400 dark:hover:bg-zinc-800"
                                  >
                                    Catalog…
                                  </button>
                                </div>
                              </div>
                              {substitute && (
                                <div className="mt-1 flex flex-wrap items-center gap-1.5 text-[10px] text-zinc-600 dark:text-zinc-400">
                                  <span>{match?.status === "matched" ? "Installed match:" : "Nearest installed substitute (review):"} <strong className="font-medium text-zinc-800 dark:text-zinc-200">{substitute.family}{substitute.subfamily ? ` ${substitute.subfamily}` : ""}</strong></span>
                                  <button
                                    onClick={(e) => { e.stopPropagation(); onFontChange(inst.id, substitute.font_path); }}
                                    title="Use this installed recommendation. This is an undoable style change."
                                    className="rounded-sm bg-cyan-500/15 px-1.5 py-0.5 text-cyan-700 transition hover:bg-cyan-500/25 dark:text-cyan-300"
                                  >Use recommendation</button>
                                </div>
                              )}
                              {previewFontOverride[inst.id] && (() => {
                                const overridePath = previewFontOverride[inst.id];
                                const allFamilies = [...(fullFamiliesByLang[targLang ?? ""] ?? []), ...(familiesByLang ?? {})[targLang ?? ""] ?? []];
                                const fam = allFamilies.find((f) => f.weights.some((w) => w.path === overridePath) || f.best_path === overridePath);
                                const weight = fam?.weights.find((w) => w.path === overridePath) ?? null;
                                const label = fam
                                  ? `${fam.family}${weight && weight.subfamily && weight.subfamily !== "Regular" ? ` ${weight.subfamily}` : ""}`
                                  : overridePath.split(/[\\/]/).pop()?.replace(/\.(ttf|otf|ttc|otc)$/i, "") ?? overridePath;
                                loadFontPreview(overridePath);
                                return (
                                  <div className="mt-1 flex items-center gap-1.5 text-[10px] text-zinc-600 dark:text-zinc-400">
                                    <ScanText size={11} className="shrink-0 text-cyan-600 dark:text-cyan-400" />
                                    <span
                                      className="truncate"
                                      style={{ fontFamily: fontNameForPath(overridePath) }}
                                      title={`${label} — preview only; the render is unaffected`}
                                    >
                                      {label}<span className="ml-0.5 text-amber-600 dark:text-amber-500">~</span>
                                    </span>
                                    {onClearPreviewFont && (
                                      <button
                                        onClick={(e) => { e.stopPropagation(); onClearPreviewFont(inst.id); }}
                                        title="reset to auto"
                                        className="rounded-sm bg-cyan-500/15 px-1.5 py-0.5 text-cyan-700 transition hover:bg-cyan-500/25 dark:text-cyan-300"
                                      >auto</button>
                                    )}
                                  </div>
                                );
                              })()}
                              {licensed.map((candidate, index) => (
                                <div key={`${candidate.family}-${index}`} className="mt-1 flex flex-wrap items-center gap-1.5 text-[10px] text-amber-700 dark:text-amber-300">
                                  <AlertTriangle size={11} className="shrink-0" />
                                  <span><strong className="font-medium">{candidate.family}{candidate.subfamily ? ` ${candidate.subfamily}` : ""}</strong>{candidate.source === "contextual_style_reference" ? " is a licensed style reference, not a detected match." : " is a licensed font candidate; it is not bundled with ToFU."}</span>
                                  {candidate.url && <a href={candidate.url} target="_blank" rel="noreferrer" className="underline underline-offset-2 hover:text-amber-900 dark:hover:text-amber-100">Review license</a>}
                                  {candidate.reason && <span className="basis-full text-amber-600/80 dark:text-amber-400/80">{candidate.reason}</span>}
                                </div>
                              ))}
                              {match?.external_provider && !match.external_provider.enabled && (
                                <p className="mt-1 text-[10px] text-zinc-500 dark:text-zinc-500">Catalog lookup unavailable: {match.external_provider.reason}</p>
                              )}
                            </div>
                          );
                        })()}
                        </div>
                        </div>
                      </td>
                    </tr>
                  )}
                </Fragment>
              );
            })}
            {sortedRegions.length === 0 && (
              <tr>
                <td colSpan={COLS.length} className="subtext px-3 py-6 text-left text-xs text-zinc-500 dark:text-zinc-600">
                  Click "draw bbox" (+) to begin.
                </td>
              </tr>
            )}
            {/* ghost rows to fill remaining space for UI seamlessness */}
            {paginatedRegions.length > 0 && paginatedRegions.length < rowsPerPage && (
              Array.from({ length: Math.max(3, rowsPerPage - paginatedRegions.length) }).map((_, i) => (
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
      {/* synced horizontal scrollbar — pinned to the bottom of the card.
          contain:paint isolates it from the card's box-shadow transition
          so it doesn't flicker on row hover. */}
      <div
        ref={syncBarRef}
        style={{ overflowX: "auto", overflowY: "hidden", flexShrink: 0, contain: "paint" }}
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
