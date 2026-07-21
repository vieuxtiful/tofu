import { useMemo, useState, useRef, useCallback } from "react";
import { ScanText } from "lucide-react";
import { MdTipsAndUpdates } from "react-icons/md";
import { TiChevronLeft, TiChevronRight, TiChevronLeftOutline, TiChevronRightOutline } from "react-icons/ti";
import { LANGUAGES, LANGUAGE_REGIONS, REGION_ORDER, REGION_ABBR, type Language } from "./languageData";

interface SourceLangPickerProps {
  selected: string | null | undefined;
  onSelect: (code: string | null) => void;
  availableCodes: string[];
  showAuto?: boolean;
}

const PAGE_SIZE = 9;

export default function SourceLangPicker({ selected, onSelect, availableCodes, showAuto = true }: SourceLangPickerProps) {
  const [searchQuery, setSearchQuery] = useState("");
  const [selectedRegions, setSelectedRegions] = useState<Set<string>>(new Set());
  const [scrollIndex, setScrollIndex] = useState(0);
  const [cardLeaving, setCardLeaving] = useState(false);
  const [cardDir, setCardDir] = useState<"left" | "right">("right");
  const leaveTimer = useRef<ReturnType<typeof setTimeout> | null>(null);
  const pageRef = useRef(0);

  const available = new Set(availableCodes);

  const filtered = useMemo<Language[]>(() => {
    const q = searchQuery.trim().toLowerCase();
    let list = LANGUAGES.filter((l) => available.has(l.code));

    if (selectedRegions.size > 0) {
      list = list.filter((l) => selectedRegions.has(l.region));
    }

    if (q) {
      list = list.filter((l) =>
        l.name.toLowerCase().includes(q) ||
        l.nativeName.toLowerCase().includes(q) ||
        l.code.toLowerCase().includes(q) ||
        l.region.toLowerCase().includes(q) ||
        (REGION_ABBR[l.region] ?? "").toLowerCase().includes(q)
      );
    }

    return list;
  }, [searchQuery, selectedRegions, available]);

  const totalPages = Math.ceil(filtered.length / PAGE_SIZE);
  const currentPage = Math.min(scrollIndex, Math.max(0, totalPages - 1));
  const pageItems = filtered.slice(currentPage * PAGE_SIZE, (currentPage + 1) * PAGE_SIZE);
  const atStart = currentPage === 0;
  const atEnd = currentPage >= totalPages - 1;
  pageRef.current = currentPage;

  const goToPage = useCallback((page: number, dir: "left" | "right") => {
    const clamped = Math.max(0, Math.min(totalPages - 1, page));
    if (clamped === pageRef.current) return;
    if (leaveTimer.current) clearTimeout(leaveTimer.current);
    setCardDir(dir);
    setCardLeaving(true);
    leaveTimer.current = setTimeout(() => {
      setCardLeaving(false);
      setScrollIndex(clamped);
    }, 200);
  }, [totalPages]);

  const scrollLeft = () => goToPage(currentPage - 1, "left");
  const scrollRight = () => goToPage(currentPage + 1, "right");

  const toggleRegion = (region: string) => {
    setSelectedRegions((prev) => {
      const next = new Set(prev);
      if (next.has(region)) next.delete(region);
      else next.add(region);
      return next;
    });
    setScrollIndex(0);
  };

  const isAuto = selected === null;

  return (
    <div className="space-y-3">
      {/* Asset Scan (Auto) — only for source language */}
      {showAuto && (
        <button
          onClick={() => onSelect(null)}
          className={`flex w-full items-center gap-3 rounded-lg border p-4 text-left transition ${
            isAuto
              ? "border-cyan-500 bg-cyan-50 dark:bg-cyan-950/30"
              : "border-zinc-300 bg-zinc-100 hover:border-cyan-600 dark:border-zinc-700 dark:bg-zinc-800/50"
          }`}
        >
          <ScanText size={20} className="text-cyan-600 dark:text-cyan-400" />
          <div>
            <p className="text-sm font-medium text-zinc-800 dark:text-zinc-200">Asset Scan (Auto)</p>
            <p className="subtext text-xs text-zinc-500">
              ToFU automatic language detection
            </p>
          </div>
        </button>
      )}

      {/* Search box */}
      <div className="relative">
        <input
          type="text"
          placeholder="search language…"
          value={searchQuery}
          onChange={(e) => { setSearchQuery(e.target.value); setScrollIndex(0); }}
          className="w-full rounded-lg border border-zinc-300 bg-zinc-50 px-3 py-2 text-sm text-zinc-800 outline-none focus:border-cyan-600 dark:border-zinc-700 dark:bg-zinc-950 dark:text-zinc-200"
        />
      </div>

      {/* Region filter chips — same style as LanguageCombobox */}
      <div className="flex flex-wrap gap-0.5">
        {REGION_ORDER.map((region) => {
          const count = (LANGUAGE_REGIONS[region] || []).filter((c) => available.has(c)).length;
          if (count === 0) return null;
          const isActive = selectedRegions.has(region);
          return (
            <button
              key={region}
              onClick={() => toggleRegion(region)}
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
        {selectedRegions.size > 0 && (
          <button
            onClick={() => setSelectedRegions(new Set())}
            className="subtext rounded px-1.5 py-0.5 text-[10px] font-medium text-zinc-400 hover:text-zinc-600 dark:hover:text-zinc-200"
          >
            clear
          </button>
        )}
      </div>

      {/* Language card grid with arrow navigation */}
      <div className="flex items-center gap-2">
        {/* Left arrow */}
        <button
          onClick={scrollLeft}
          disabled={atStart}
          className={`shrink-0 rounded-lg p-1.5 transition ${
            atStart
              ? "text-zinc-300 dark:text-zinc-700 cursor-default"
              : "text-zinc-600 hover:bg-zinc-200 dark:text-zinc-400 dark:hover:bg-zinc-800"
          }`}
        >
          {atStart ? <TiChevronLeftOutline size={20} /> : <TiChevronLeft size={20} />}
        </button>

        {/* Card grid — 3x3, rendered on the surface (no overflow clipping) */}
        <div className="relative flex-1">
          <div
            className={`grid grid-cols-3 gap-2 ${cardLeaving ? (cardDir === "right" ? "lang-card-leave-left" : "lang-card-leave-right") : cardDir === "right" ? "lang-card-enter-right" : "lang-card-enter-left"}`}
          >
            {pageItems.map((lang) => {
              const isSel = lang.code === selected;
              return (
                <button
                  key={lang.code}
                  onClick={() => onSelect(lang.code)}
                  className={`soft-shadow flex flex-col items-center justify-center gap-0.5 rounded-lg px-2 py-2.5 text-center transition ${
                    isSel
                      ? "border border-cyan-500 bg-cyan-50 dark:bg-cyan-950/30"
                      : "border border-transparent bg-transparent hover:border-cyan-400"
                  }`}
                >
                  <span className={`text-sm font-semibold ${isSel ? "text-cyan-700 dark:text-cyan-300" : "text-zinc-800 dark:text-zinc-200"}`}>
                    {lang.code}
                  </span>
                  <span className="subtext text-[10px] text-zinc-500">
                    {REGION_ABBR[lang.region] ?? lang.region}
                  </span>
                </button>
              );
            })}
            {pageItems.length === 0 && (
              <div className="col-span-3 py-6 text-center text-sm text-zinc-500">
                No languages found
              </div>
            )}
          </div>
        </div>

        {/* Right arrow */}
        <button
          onClick={scrollRight}
          disabled={atEnd}
          className={`shrink-0 rounded-lg p-1.5 transition ${
            atEnd
              ? "text-zinc-300 dark:text-zinc-700 cursor-default"
              : "text-zinc-600 hover:bg-zinc-200 dark:text-zinc-400 dark:hover:bg-zinc-800"
          }`}
        >
          {atEnd ? <TiChevronRightOutline size={20} /> : <TiChevronRight size={20} />}
        </button>
      </div>

      {/* Tip: uploads in any other language will be flagged and blocked */}
      <p className="bezier-impression subtext flex items-start gap-1 px-3 py-2 text-xs text-zinc-500 dark:text-zinc-600">
        <MdTipsAndUpdates size={14} className="mt-0.5 shrink-0 text-[#2d8cf0]" />
        Uploads in any other language will be flagged and blocked.
      </p>
    </div>
  );
}
