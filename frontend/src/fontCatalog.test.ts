// 🍢 ToFU — Font Manager catalog logic tests
//
// fontCatalog.ts is pure by design, so none of this needs a DOM, a font
// file, or a server — the same arrangement doppelganger.test.ts uses.

import { describe, it, expect } from "vitest";
import { FontCategory, FontFamily } from "./api";
import {
  KeyValueStore,
  RECENT_KEY,
  RECENT_LIMIT,
  filterByCategory,
  mergeFamily,
  pushRecent,
  readRecent,
  searchFamilies,
  sortFamilies,
  fontSource,
} from "./fontCatalog";

function fam(
  family: string,
  category?: FontCategory,
  coverage = 1,
  subfamilies: string[] = ["Regular"],
): FontFamily {
  return {
    family,
    best_path: `C:/fonts/${family}.ttf`,
    best_coverage: coverage,
    category,
    weights: subfamilies.map((subfamily, i) => ({
      path: `C:/fonts/${family}-${subfamily}.ttf`,
      subfamily,
      weight_class: 400 + i * 100,
      coverage,
    })),
  };
}

/** stands in for localStorage. A plain object is enough because
 * fontCatalog only ever calls getItem/setItem. */
function fakeStore(initial?: string): KeyValueStore & { data: Record<string, string> } {
  const data: Record<string, string> = initial ? { [RECENT_KEY]: initial } : {};
  return {
    data,
    getItem: (k) => (k in data ? data[k] : null),
    setItem: (k, v) => { data[k] = v; },
  };
}

describe("searchFamilies", () => {
  const list = [
    fam("Georgia", "serif"),
    fam("Consolas", "mono"),
    fam("Arial", "sans", 1, ["Regular", "Bold Italic"]),
  ];

  it("returns everything for an empty or whitespace query", () => {
    expect(searchFamilies(list, "")).toHaveLength(3);
    expect(searchFamilies(list, "   ")).toHaveLength(3);
  });

  it("matches family names case-insensitively", () => {
    expect(searchFamilies(list, "GEORG").map((f) => f.family)).toEqual(["Georgia"]);
  });

  it("matches subfamily names, which the ordinary dropdown filter cannot", () => {
    expect(searchFamilies(list, "italic").map((f) => f.family)).toEqual(["Arial"]);
  });

  it("returns an empty list rather than everything when nothing matches", () => {
    expect(searchFamilies(list, "zzzz")).toEqual([]);
  });
});

describe("filterByCategory", () => {
  const list = [
    fam("Georgia", "serif"),
    fam("Consolas", "mono"),
    fam("Arial", "sans"),
    fam("Mystery"), // sift() declined to classify this one
  ];

  it("treats an empty selection as 'all typefaces'", () => {
    expect(filterByCategory(list, [])).toHaveLength(4);
  });

  it("filters to one category", () => {
    expect(filterByCategory(list, ["mono"]).map((f) => f.family)).toEqual(["Consolas"]);
  });

  it("is additive across categories", () => {
    expect(filterByCategory(list, ["serif", "sans"]).map((f) => f.family))
      .toEqual(["Georgia", "Arial"]);
  });

  it("counts an absent category as unknown, so unclassified families stay reachable", () => {
    expect(filterByCategory(list, ["unknown"]).map((f) => f.family)).toEqual(["Mystery"]);
  });
});

describe("sortFamilies", () => {
  const list = [fam("Charlie", "serif", 0.5), fam("alpha", "sans", 1), fam("Bravo", "mono", 1)];

  it("sorts A–Z ignoring case", () => {
    expect(sortFamilies(list, "az").map((f) => f.family)).toEqual(["alpha", "Bravo", "Charlie"]);
  });

  it("sorts Z–A", () => {
    expect(sortFamilies(list, "za").map((f) => f.family)).toEqual(["Charlie", "Bravo", "alpha"]);
  });

  it("sorts by coverage, breaking ties by name so the order is stable", () => {
    expect(sortFamilies(list, "coverage").map((f) => f.family))
      .toEqual(["alpha", "Bravo", "Charlie"]);
  });

  it("puts recently used first and leaves the rest alphabetical", () => {
    const recent = [{ family: "Charlie", path: "x" }];
    expect(sortFamilies(list, "recent", recent).map((f) => f.family))
      .toEqual(["Charlie", "alpha", "Bravo"]);
  });

  it("never mutates the caller's array — it is the cached catalog", () => {
    const original = [...list];
    sortFamilies(list, "za");
    expect(list).toEqual(original);
  });
});

describe("mergeFamily", () => {
  it("appends a family the ranked list did not contain", () => {
    const merged = mergeFamily([fam("Arial")], fam("Papyrus"));
    expect(merged.map((f) => f.family)).toEqual(["Arial", "Papyrus"]);
  });

  it("replaces in place rather than duplicating a family already present", () => {
    const merged = mergeFamily([fam("Arial", "sans", 0.5), fam("Georgia")], fam("Arial", "sans", 1));
    expect(merged).toHaveLength(2);
    expect(merged[0].best_coverage).toBe(1);
  });

  it("does not mutate the list it was given", () => {
    const list = [fam("Arial")];
    mergeFamily(list, fam("Papyrus"));
    expect(list).toHaveLength(1);
  });
});

describe("recently used", () => {
  it("reads an empty history from a fresh store", () => {
    expect(readRecent(fakeStore())).toEqual([]);
  });

  it("survives a corrupt or foreign stored value", () => {
    expect(readRecent(fakeStore("not json"))).toEqual([]);
    expect(readRecent(fakeStore('{"nope":1}'))).toEqual([]);
    expect(readRecent(fakeStore('[{"family":"ok","path":"p"},{"bad":true}]')))
      .toEqual([{ family: "ok", path: "p" }]);
  });

  it("stores most-recent-first", () => {
    const store = fakeStore();
    pushRecent(store, { family: "A", path: "a" });
    const after = pushRecent(store, { family: "B", path: "b" });
    expect(after.map((r) => r.family)).toEqual(["B", "A"]);
    expect(readRecent(store).map((r) => r.family)).toEqual(["B", "A"]);
  });

  it("moves a re-picked family to the front instead of duplicating it", () => {
    const store = fakeStore();
    pushRecent(store, { family: "A", path: "a-regular" });
    pushRecent(store, { family: "B", path: "b" });
    const after = pushRecent(store, { family: "A", path: "a-bold" });
    expect(after.map((r) => r.family)).toEqual(["A", "B"]);
    expect(after[0].path).toBe("a-bold");
  });

  it("caps the history", () => {
    const store = fakeStore();
    let last: { family: string }[] = [];
    for (let i = 0; i < RECENT_LIMIT + 5; i++) {
      last = pushRecent(store, { family: `F${i}`, path: `p${i}` });
    }
    expect(last).toHaveLength(RECENT_LIMIT);
    expect(last[0].family).toBe(`F${RECENT_LIMIT + 4}`);
  });

  it("does not throw when the store rejects writes", () => {
    const store: KeyValueStore = {
      getItem: () => null,
      setItem: () => { throw new Error("quota exceeded"); },
    };
    expect(() => pushRecent(store, { family: "A", path: "a" })).not.toThrow();
  });
});

describe("fontSource", () => {
  it("recognises an uploaded face", () => {
    expect(fontSource(String.raw`C:\repo\server\font-library\user\My.ttf`)).toBe("user");
    expect(fontSource("/srv/tofu/server/font-library/user/My.ttf")).toBe("user");
  });

  it("recognises a pack face", () => {
    expect(fontSource("/srv/server/font-library/packs/noto-cjk/A.otf")).toBe("pack");
  });

  it("treats bundled and OS directories alike as installed", () => {
    expect(fontSource(String.raw`C:\Windows\Fonts\arial.ttf`)).toBe("system");
    expect(fontSource("/usr/share/fonts/truetype/DejaVuSans.ttf")).toBe("system");
    expect(fontSource("/srv/tofu/server/fonts/Bundled.ttf")).toBe("system");
  });

  it("is safe on a missing path", () => {
    expect(fontSource(null)).toBe("system");
    expect(fontSource(undefined)).toBe("system");
    expect(fontSource("")).toBe("system");
  });

  it("distinguishes two same-named families by source", () => {
    // the case that makes the badge necessary: one family name, two files
    const a = fontSource(String.raw`C:\Windows\Fonts\arial.ttf`);
    const b = fontSource(String.raw`C:\repo\server\font-library\user\arial.ttf`);
    expect(a).not.toBe(b);
  });
});
