// 🍢 ToFU — Font Manager
//
// The whole installed font library, browsable. The ordinary dropdown shows
// the server's ranked top-24 for the target language, which is the right
// list when you want the best coverage and the wrong one when you know the
// typeface you want. This panel is the second case.

import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { ChevronRight, X } from "lucide-react";
import { FontCategory, FontFamily, FontWeight } from "./api";
import {
  CATEGORY_LABELS,
  FILTERABLE_CATEGORIES,
  RecentFont,
  SOURCE_LABELS,
  SortMode,
  filterByCategory,
  fontSource,
  readRecent,
  searchFamilies,
  sortFamilies,
} from "./fontCatalog";
import {
  FontPack, fetchFontPacks, installFontPack, removeFontPack, uploadFont,
} from "./api";
import { fontNameForPath, loadFontPreview, weightLabel } from "./FontCombobox";

interface FontManagerProps {
  open: boolean;
  onClose: () => void;
  families: FontFamily[];
  loading: boolean;
  /** face path currently applied to the selected region, if any */
  value: string | null;
  targLang: string;
  recent: RecentFont[];
  onPick: (family: FontFamily, weight: FontWeight) => void;
  /** re-fetch the catalog after the library changes on disk */
  onLibraryChanged: () => void;
}

const SORT_LABELS: Record<SortMode, string> = {
  az: "A–Z",
  za: "Z–A",
  coverage: "best coverage",
  recent: "recently used",
};

/** shared by the search field and every pill so the card reads as one
 * control surface rather than a pile of borrowed widgets */
const PILL =
  "rounded-full border px-2.5 py-1 text-[11px] transition select-none";
const PILL_OFF =
  "border-zinc-300 text-zinc-600 hover:border-zinc-400 hover:bg-zinc-100 dark:border-zinc-700 dark:text-zinc-400 dark:hover:border-zinc-600 dark:hover:bg-zinc-800";
const PILL_ON =
  "border-cyan-500 bg-cyan-500/10 text-cyan-700 dark:border-cyan-400 dark:text-cyan-300";

/** How far outside the scroll viewport a row still counts as worth
 * preloading, so a face is ready by the time it is scrolled to. */
const PREVIEW_MARGIN = 160;

/** Loads preview faces for the rows currently on screen.
 *
 * The combobox can afford to preload every family the moment it opens — it
 * has 24. This panel has the whole library, and loadFontPreview() appends a
 * <style> element and fetches a font file per path, so eager loading here
 * would mean hundreds of requests for rows nobody scrolled to.
 *
 * Measured with getBoundingClientRect rather than IntersectionObserver on
 * purpose. IO computes intersections off the compositor, so in an
 * environment that is not painting frames its callback simply never fires
 * — and the failure is silent: no error, no request, every family rendered
 * in the fallback face. Reading rects is synchronous and always answers.
 * loadFontPreview is idempotent, so re-scanning rows costs nothing. */
function loadVisiblePreviews(list: HTMLElement | null) {
  if (!list) return;
  const view = list.getBoundingClientRect();
  const top = view.top - PREVIEW_MARGIN;
  const bottom = view.bottom + PREVIEW_MARGIN;
  for (const el of Array.from(list.querySelectorAll<HTMLElement>("[data-font-path]"))) {
    const r = el.getBoundingClientRect();
    if (r.bottom >= top && r.top <= bottom) loadFontPreview(el.dataset.fontPath!);
  }
}

function weightCss(w: FontWeight): React.CSSProperties {
  const wc = Math.max(100, Math.min(900, Math.round(w.weight_class / 100) * 100));
  const style: React.CSSProperties = { fontWeight: wc };
  if (w.italic ?? /italic|oblique/i.test(w.subfamily || "")) style.fontStyle = "italic";
  return style;
}

function FamilyRow({
  fam, expanded, onToggle, value, onPick,
}: {
  fam: FontFamily;
  expanded: boolean;
  onToggle: () => void;
  value: string | null;
  onPick: (weight: FontWeight) => void;
}) {
  const isCurrent =
    value !== null &&
    (fam.best_path === value || fam.weights.some((w) => w.path === value));

  // expanding a family is an explicit request for its weights, so this is
  // the one place eager loading is right
  useEffect(() => {
    if (expanded) fam.weights.forEach((w) => loadFontPreview(w.path));
  }, [expanded, fam]);

  return (
    <div
      data-font-path={fam.best_path}
      className="border-b border-zinc-200/60 last:border-b-0 dark:border-zinc-800/60"
    >
      <button
        onClick={onToggle}
        className={`flex w-full items-center gap-2 px-2 py-1.5 text-left transition hover:bg-zinc-100 dark:hover:bg-zinc-800 ${
          isCurrent ? "text-cyan-600 dark:text-cyan-400" : "text-zinc-700 dark:text-zinc-300"
        }`}
      >
        <ChevronRight
          size={11}
          className={`shrink-0 text-zinc-500 transition-transform ${expanded ? "rotate-90" : ""}`}
        />
        <span
          className="min-w-0 flex-1 truncate text-sm"
          style={{ fontFamily: fontNameForPath(fam.best_path) }}
        >
          {fam.family}
        </span>
        {isCurrent && (
          <span className="shrink-0 text-[9px] uppercase tracking-wider text-cyan-600 dark:text-cyan-400">
            current
          </span>
        )}
        {/* A family can legitimately appear twice now — a bundled Arial and
            the system Arial are different files under one name — so the
            source is what tells them apart. Only shown when it is not the
            ordinary case, to keep the row quiet. */}
        {fontSource(fam.best_path) !== "system" && (
          <span className="shrink-0 rounded-sm bg-cyan-500/15 px-1 text-[9px] text-cyan-700 dark:text-cyan-300">
            {SOURCE_LABELS[fontSource(fam.best_path)]}
          </span>
        )}
        <span className="shrink-0 text-[10px] text-zinc-500">
          {CATEGORY_LABELS[fam.category ?? "unknown"]}
        </span>
        <span className="w-9 shrink-0 text-right text-[10px] text-zinc-500">
          {(fam.best_coverage * 100).toFixed(0)}%
        </span>
      </button>
      {expanded && (
        <div className="ml-5 border-l border-zinc-200 dark:border-zinc-800">
          {fam.weights.map((w) => (
            <button
              key={w.path}
              onClick={() => onPick(w)}
              className={`flex w-full items-center gap-2 px-2 py-1 text-left text-xs transition hover:bg-zinc-100 dark:hover:bg-zinc-800 ${
                value === w.path
                  ? "text-cyan-600 dark:text-cyan-400"
                  : "text-zinc-600 dark:text-zinc-400"
              }`}
              style={{ fontFamily: fontNameForPath(w.path) }}
            >
              <span className="min-w-0 flex-1 truncate" style={weightCss(w)}>
                {weightLabel(w)}
              </span>
              <span className="shrink-0 text-[10px] text-zinc-500">
                {(w.coverage * 100).toFixed(0)}%
              </span>
            </button>
          ))}
        </div>
      )}
    </div>
  );
}

export default function FontManager({
  open, onClose, families, loading, value, targLang, recent, onPick, onLibraryChanged,
}: FontManagerProps) {
  const [leaving, setLeaving] = useState(false);
  const [packs, setPacks] = useState<FontPack[]>([]);
  const [busy, setBusy] = useState<string | null>(null);
  const [notice, setNotice] = useState<string | null>(null);
  const fileRef = useRef<HTMLInputElement>(null);
  const packRef = useRef<HTMLInputElement>(null);

  useEffect(() => {
    if (!open) return;
    fetchFontPacks().then(setPacks).catch(() => setPacks([]));
  }, [open]);

  /** Nothing here installs a font at OS level, and nothing needs to:
   * scribe renders from a path, the preview is served over /api/font-file,
   * and the registry keys on absolute paths. */
  const runInstall = useCallback(async (label: string, work: () => Promise<string>) => {
    setBusy(label);
    setNotice(null);
    try {
      setNotice(await work());
      onLibraryChanged();
      setPacks(await fetchFontPacks().catch(() => []));
    } catch (error) {
      setNotice(error instanceof Error ? error.message : String(error));
    } finally {
      setBusy(null);
    }
  }, [onLibraryChanged]);
  const [query, setQuery] = useState("");
  const [categories, setCategories] = useState<FontCategory[]>([]);
  const [sort, setSort] = useState<SortMode>("coverage");
  const [expanded, setExpanded] = useState<string | null>(null);

  // matches the .title-confirm-backdrop exit animation exactly (tco-fade-out
  // is 300ms); closing sooner clips it, later leaves a dead card on screen
  const close = useCallback(() => {
    setLeaving(true);
    setTimeout(() => {
      setLeaving(false);
      onClose();
    }, 300);
  }, [onClose]);

  useEffect(() => {
    if (!open) return;
    setQuery("");
    setCategories([]);
    setExpanded(null);
  }, [open]);

  // Escape as well as backdrop click. The app's other pop-ups close on the
  // backdrop only; a panel you type into needs the keyboard exit.
  useEffect(() => {
    if (!open) return;
    const onKey = (e: KeyboardEvent) => { if (e.key === "Escape") close(); };
    document.addEventListener("keydown", onKey);
    return () => document.removeEventListener("keydown", onKey);
  }, [open, close]);

  const visible = useMemo(
    () => sortFamilies(filterByCategory(searchFamilies(families, query), categories), sort, recent),
    [families, query, categories, sort, recent],
  );

  // Rescan after every change that moves rows under the viewport: the panel
  // opening, the list scrolling, and any search/filter/sort that replaces
  // which families are rendered at all.
  const listRef = useRef<HTMLDivElement>(null);
  useEffect(() => {
    if (!open) return;
    const scan = () => loadVisiblePreviews(listRef.current);
    scan();
    const list = listRef.current;
    if (!list) return;
    let timer: number | null = null;
    const onScroll = () => {
      if (timer !== null) window.clearTimeout(timer);
      timer = window.setTimeout(scan, 80);
    };
    list.addEventListener("scroll", onScroll, { passive: true });
    return () => {
      if (timer !== null) window.clearTimeout(timer);
      list.removeEventListener("scroll", onScroll);
    };
  }, [open, visible]);

  const toggleCategory = (c: FontCategory) =>
    setCategories((prev) => (prev.includes(c) ? prev.filter((x) => x !== c) : [...prev, c]));

  const hasFilters = query.trim() !== "" || categories.length > 0 || sort !== "coverage";
  const reset = () => { setQuery(""); setCategories([]); setSort("coverage"); };

  if (!open) return null;

  return (
    <div
      className={`title-confirm-backdrop${leaving ? " leaving" : ""}`}
      onClick={close}
    >
      <div
        className="bezier-card step-fade flex flex-col rounded-lg bg-white p-5 dark:bg-zinc-900"
        style={{
          maxWidth: "620px",
          width: "min(620px, calc(100vw - 32px))",
          boxShadow: "4px 4px 0 var(--bc-shadow), 8px 8px 16px rgba(0,0,0,0.18)",
        }}
        onClick={(e) => e.stopPropagation()}
      >
        <div className="mb-3 flex w-full items-center gap-2">
          <p className="text-sm font-semibold text-zinc-800 dark:text-zinc-200">font manager</p>
          <span className="subtext text-xs text-zinc-500">
            {loading ? "loading catalog…" : `${visible.length} of ${families.length}`}
            {" · "}
            {targLang}
          </span>
          <button
            onClick={close}
            aria-label="Close font manager"
            className="ml-auto rounded-md p-1 text-zinc-500 transition hover:bg-zinc-200 hover:text-zinc-700 dark:hover:bg-zinc-800 dark:hover:text-zinc-300"
          >
            <X size={14} />
          </button>
        </div>

        {/* Search. The original markup used daisyUI's `input` class, which
            this project deliberately leaves uninstalled (see index.css) —
            the structure is kept, the classes are ours. */}
        <label className="mb-3 flex w-full items-center gap-2 rounded-full border border-zinc-300 bg-zinc-50 px-3 py-1.5 focus-within:border-cyan-500 dark:border-zinc-700 dark:bg-zinc-950 dark:focus-within:border-cyan-400">
          <svg
            className="h-[1em] w-[1em] shrink-0 opacity-50"
            xmlns="http://www.w3.org/2000/svg"
            viewBox="0 0 24 24"
          >
            <g
              strokeLinejoin="round"
              strokeLinecap="round"
              strokeWidth="2.5"
              fill="none"
              stroke="currentColor"
            >
              <circle cx="11" cy="11" r="8" />
              <path d="m21 21-4.3-4.3" />
            </g>
          </svg>
          <input
            type="search"
            autoFocus
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            placeholder="Search fonts"
            className="min-w-0 flex-1 bg-transparent text-xs text-zinc-800 outline-hidden dark:text-zinc-200"
          />
        </label>

        <div className="mb-2 flex flex-wrap items-center gap-1.5">
          <button
            onClick={() => setCategories([])}
            className={`${PILL} ${categories.length === 0 ? PILL_ON : PILL_OFF}`}
          >
            all typefaces
          </button>
          {FILTERABLE_CATEGORIES.map((c) => (
            <button
              key={c}
              onClick={() => toggleCategory(c)}
              className={`${PILL} ${categories.includes(c) ? PILL_ON : PILL_OFF}`}
            >
              {CATEGORY_LABELS[c]}
            </button>
          ))}
        </div>

        <div className="mb-3 flex flex-wrap items-center gap-1.5">
          {(Object.keys(SORT_LABELS) as SortMode[]).map((m) => (
            <button
              key={m}
              onClick={() => setSort(m)}
              className={`${PILL} ${sort === m ? PILL_ON : PILL_OFF}`}
            >
              {SORT_LABELS[m]}
            </button>
          ))}
        </div>

        {hasFilters && (
          <div className="mb-2 flex flex-wrap items-center gap-1.5">
            {query.trim() !== "" && (
              <button
                onClick={() => setQuery("")}
                className={`${PILL} ${PILL_ON} flex items-center gap-1`}
              >
                “{query.trim()}” <X size={9} />
              </button>
            )}
            {categories.map((c) => (
              <button
                key={c}
                onClick={() => toggleCategory(c)}
                className={`${PILL} ${PILL_ON} flex items-center gap-1`}
              >
                {CATEGORY_LABELS[c]} <X size={9} />
              </button>
            ))}
            {sort !== "coverage" && (
              <button
                onClick={() => setSort("coverage")}
                className={`${PILL} ${PILL_ON} flex items-center gap-1`}
              >
                {SORT_LABELS[sort]} <X size={9} />
              </button>
            )}
            <button
              onClick={reset}
              className="subtext text-[11px] text-zinc-500 underline-offset-2 hover:underline"
            >
              reset
            </button>
          </div>
        )}

        <div
          ref={listRef}
          className="max-h-[46vh] w-full overflow-y-auto rounded-lg border border-zinc-200 dark:border-zinc-800"
        >
          {visible.map((fam) => (
            <FamilyRow
              key={fam.family}
              fam={fam}
              expanded={expanded === fam.family}
              onToggle={() => setExpanded(expanded === fam.family ? null : fam.family)}
              value={value}
              onPick={(w) => { onPick(fam, w); close(); }}
            />
          ))}
          {!loading && visible.length === 0 && (
            <p className="px-3 py-4 text-center text-xs text-zinc-500">
              {families.length === 0
                ? "no fonts available for this language"
                : "nothing matches those filters"}
            </p>
          )}
          {loading && families.length === 0 && (
            <p className="px-3 py-4 text-center text-xs text-zinc-500">reading the font library…</p>
          )}
        </div>

        {/* --- install ------------------------------------------------- */}
        <div className="mt-3 w-full border-t border-zinc-200 pt-3 dark:border-zinc-800">
          <div className="flex flex-wrap items-center gap-2">
            <input
              ref={fileRef} type="file" accept=".ttf,.otf,.ttc,.otc" className="hidden"
              onChange={(e) => {
                const file = e.target.files?.[0];
                e.target.value = "";
                if (!file) return;
                runInstall("font", async () => {
                  const result = await uploadFont(file);
                  return result.preview_blocked
                    // installed and renders locally, but its vendor's fsType
                    // forbids embedding, so it is not served for preview
                    ? `${result.installed} installed — its licence blocks web preview`
                    : `${result.installed} installed (${result.faces} face(s))`;
                });
              }}
            />
            <input
              ref={packRef} type="file" accept=".zip" className="hidden"
              onChange={(e) => {
                const file = e.target.files?.[0];
                e.target.value = "";
                if (!file) return;
                runInstall("pack", async () => {
                  const result = await installFontPack(file);
                  return `pack ${result.installed} installed (${result.faces} face(s))`;
                });
              }}
            />
            <button
              onClick={() => fileRef.current?.click()} disabled={busy !== null}
              className={`${PILL} ${PILL_OFF} disabled:opacity-40`}
            >
              {busy === "font" ? "installing…" : "install a font"}
            </button>
            <button
              onClick={() => packRef.current?.click()} disabled={busy !== null}
              className={`${PILL} ${PILL_OFF} disabled:opacity-40`}
            >
              {busy === "pack" ? "installing…" : "install a pack"}
            </button>
            {notice && (
              <span className="subtext text-[11px] text-zinc-600 dark:text-zinc-400">{notice}</span>
            )}
          </div>

          {packs.length > 0 && (
            <div className="mt-2 flex flex-wrap items-center gap-1.5">
              {packs.map((pack) => (
                <span
                  key={pack.name}
                  className={`${PILL} ${PILL_OFF} flex items-center gap-1`}
                  title={[pack.license, pack.version].filter(Boolean).join(" · ") || undefined}
                >
                  {pack.name}
                  <span className="text-zinc-400">{pack.face_files ?? 0}</span>
                  <button
                    aria-label={`remove pack ${pack.name}`}
                    disabled={busy !== null}
                    onClick={() => runInstall("pack", async () => {
                      const result = await removeFontPack(pack.name);
                      return `pack ${result.removed} removed`;
                    })}
                    className="text-zinc-500 hover:text-red-500"
                  >
                    <X size={9} />
                  </button>
                </span>
              ))}
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
