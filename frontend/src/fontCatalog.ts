// 🍢 ToFU — Font Manager catalog logic
//
// Everything the Font Manager does to a list of families that is not
// rendering: searching, filtering, sorting, remembering. All pure, so none
// of it needs a DOM, a font file, or a server — the same reason
// doppelganger.ts is shaped this way.

import { FontCategory, FontFamily } from "./api";

export type SortMode = "az" | "za" | "coverage" | "recent";

export interface RecentFont {
  family: string;
  path: string;
}

/** How many recently-used faces we keep. Long enough to cover a working
 * session's palette, short enough that the "recently used" sort stays a
 * shortlist rather than a second full catalog. */
export const RECENT_LIMIT = 12;

export const RECENT_KEY = "tofu.recentFonts";

/** the subset of Storage we actually use, so tests can pass a plain object
 * instead of standing up a DOM */
export interface KeyValueStore {
  getItem(key: string): string | null;
  setItem(key: string, value: string): void;
}

export const CATEGORY_LABELS: Record<FontCategory, string> = {
  serif: "serif",
  sans: "sans serif",
  mono: "monospace",
  display: "display",
  unknown: "uncategorized",
};

/** The order the filter chips appear in. `unknown` is deliberately absent:
 * it is a non-answer, not a category someone browses for. Families that
 * sift() could not classify still list and still search — they just do not
 * satisfy a category filter. */
export const FILTERABLE_CATEGORIES: FontCategory[] = ["serif", "sans", "mono", "display"];

/** Case-insensitive match over the family name AND every face's subfamily,
 * so "italic" or "condensed" finds families whose name never says so. The
 * ordinary dropdown filter matches family names only; this is the panel you
 * open when that was not enough. */
export function searchFamilies(families: FontFamily[], query: string): FontFamily[] {
  const q = query.trim().toLowerCase();
  if (!q) return families;
  return families.filter(
    (f) =>
      f.family.toLowerCase().includes(q) ||
      f.weights.some((w) => (w.subfamily || "").toLowerCase().includes(q)),
  );
}

/** An empty selection means "all typefaces" — the chips are additive, so
 * selecting none and selecting every one both show everything. */
export function filterByCategory(
  families: FontFamily[],
  categories: FontCategory[],
): FontFamily[] {
  if (categories.length === 0) return families;
  const wanted = new Set(categories);
  return families.filter((f) => wanted.has(f.category ?? "unknown"));
}

function byName(a: FontFamily, b: FontFamily): number {
  return a.family.localeCompare(b.family, undefined, { sensitivity: "base" });
}

/** Never sorts in place: the caller's array is the cached catalog, and the
 * sort control must not permanently reorder it. */
export function sortFamilies(
  families: FontFamily[],
  mode: SortMode,
  recent: RecentFont[] = [],
): FontFamily[] {
  const out = [...families];
  switch (mode) {
    case "az":
      return out.sort(byName);
    case "za":
      return out.sort((a, b) => byName(b, a));
    case "coverage":
      // ties broken by name so the list is stable rather than arbitrary —
      // whole swathes of the library score exactly 1.0 for Latin
      return out.sort((a, b) => b.best_coverage - a.best_coverage || byName(a, b));
    case "recent": {
      const rank = new Map(recent.map((r, i) => [r.family, i]));
      const at = (f: FontFamily) => rank.get(f.family) ?? Number.MAX_SAFE_INTEGER;
      // unused families keep alphabetical order behind the used ones,
      // rather than falling back to discovery order
      return out.sort((a, b) => at(a) - at(b) || byName(a, b));
    }
    default:
      return out;
  }
}

/** Add one family to the session's ordinary dropdown list.
 *
 * The dropdown is fed the server's ranked top-24; picking font #97 in the
 * manager has to put it there too, or the combobox cannot resolve the path
 * back to a family and falls back to showing a bare filename. */
export function mergeFamily(families: FontFamily[], fam: FontFamily): FontFamily[] {
  const i = families.findIndex((f) => f.family === fam.family);
  if (i === -1) return [...families, fam];
  const next = [...families];
  next[i] = fam;
  return next;
}

export function readRecent(store: KeyValueStore): RecentFont[] {
  try {
    const raw = store.getItem(RECENT_KEY);
    if (!raw) return [];
    const parsed = JSON.parse(raw);
    if (!Array.isArray(parsed)) return [];
    return parsed
      .filter((r): r is RecentFont =>
        Boolean(r) && typeof r.family === "string" && typeof r.path === "string")
      .slice(0, RECENT_LIMIT);
  } catch {
    // a corrupt or foreign value in localStorage must not take the panel
    // down with it — an empty history is a fine answer
    return [];
  }
}

/** Most-recent-first, deduped by family. Re-picking a family you already
 * used moves it to the front rather than adding a second entry. */
export function pushRecent(
  store: KeyValueStore,
  entry: RecentFont,
): RecentFont[] {
  const next = [entry, ...readRecent(store).filter((r) => r.family !== entry.family)]
    .slice(0, RECENT_LIMIT);
  try {
    store.setItem(RECENT_KEY, JSON.stringify(next));
  } catch {
    // private-browsing quota failures are not worth losing a font pick over
  }
  return next;
}

/** Where a face came from, read off its path.
 *
 * The server installs into `server/font-library/user/` and
 * `server/font-library/packs/<name>/`; anything else is a directory the
 * deployment or the OS provided. Derived here rather than sent per family
 * because the path already says it and the catalog payload is large enough.
 *
 * It matters because a font can now legitimately appear TWICE under one
 * family name — a bundled Arial and the system Arial are different files
 * and different registry keys, and presenting them as duplicates with no
 * way to tell them apart would be worse than showing one.
 */
export type FontSource = "user" | "pack" | "system";

export function fontSource(path: string | null | undefined): FontSource {
  const normalised = (path ?? "").split("\\").join("/").toLowerCase();
  if (normalised.includes("/font-library/user/")) return "user";
  if (normalised.includes("/font-library/packs/")) return "pack";
  return "system";
}

export const SOURCE_LABELS: Record<FontSource, string> = {
  user: "uploaded",
  pack: "pack",
  system: "installed",
};
