// 🍢 doppelganger — the dupe ticket's two decisions, tested on their own.
//
// Both are pure, so neither needs a DOM, a canvas, or a server.  The
// properties asserted here are the ones that make the preview honest:
// a real font pick is never overridden by an inference, and the one
// place the preview deliberately does MORE than scribe never does it
// where a human's typographic intent would be flattened.

import { describe, it, expect } from "vitest";
import { InstText, SemanticTextUnit, FontFamily } from "./api";
import { FitResult } from "./textFit";
import {
  resolveFontPath, dupeSpec, adviseShrink, shouldRenderVertical, fontIdentity,
  SHRINK_WARN,
} from "./doppelganger";

const CTX = { defaultTargLang: "it", familiesByLang: {} as Record<string, FontFamily[]> };

function region(over: Partial<InstText> = {}): InstText {
  return {
    id: "r1",
    bounding_box: { x: 0, y: 0, width: 200, height: 40 },
    text: "VIEUX MURS",
    target_text: "MURI VECCHI",
    confidence: 0.9,
    detected_language: "fr",
    reading_order: 0,
    dnt: false,
    target_language: "it",
    ...over,
  } as InstText;
}

describe("resolveFontPath — the ladder", () => {
  it("rung 1: an explicit pick wins over everything below it", () => {
    const r = region({
      style_profile: { font_family: "picked.ttf" } as InstText["style_profile"],
      resolved_font_family: "auto.ttf",
      font_match: {
        status: "matched", confidence: 0.99,
        recommended_substitute: { font_path: "nn.ttf", family: "NN", score: 0.99 },
      } as InstText["font_match"],
    });
    expect(resolveFontPath(r, CTX)).toEqual({ path: "picked.ttf", provenance: "explicit" });
  });

  it("rung 2: a glyph-shape match outranks the generic auto default", () => {
    const r = region({
      resolved_font_family: "auto.ttf",
      font_match: {
        status: "matched", confidence: 0.99,
        recommended_substitute: { font_path: "nn.ttf", family: "NN", score: 0.99 },
      } as InstText["font_match"],
    });
    // resolved_font_family is _default_fallback_path() — arial.ttf for
    // every Latin region regardless of what the sign looks like. Ranking
    // that constant first made this rung unreachable: the la-rue-sans-nom
    // plaque matched Centaur at 0.784 and still rendered in Arial.
    expect(resolveFontPath(r, CTX)).toEqual({ path: "nn.ttf", provenance: "nearest_neighbor" });
  });

  it("rung 2 accepts a review-grade match — the rung below is a constant, not a rival opinion", () => {
    const r = region({
      resolved_font_family: "auto.ttf",
      font_match: {
        status: "review", confidence: 0.5,
        recommended_substitute: { font_path: "nn.ttf", family: "NN", score: 0.5 },
      } as InstText["font_match"],
    });
    expect(resolveFontPath(r, CTX).provenance).toBe("nearest_neighbor");
  });

  it("rung 3: falls to the auto default when the match is unavailable", () => {
    const r = region({
      resolved_font_family: "auto.ttf",
      font_match: {
        status: "unavailable", confidence: 0,
        recommended_substitute: { font_path: "nn.ttf", family: "NN", score: 0.1 },
      } as InstText["font_match"],
    });
    expect(resolveFontPath(r, CTX)).toEqual({ path: "auto.ttf", provenance: "auto" });
  });

  it("rung 3: falls to the auto default when there is no substitute to recommend", () => {
    const r = region({
      resolved_font_family: "auto.ttf",
      font_match: { status: "review", confidence: 0.4 } as InstText["font_match"],
    });
    expect(resolveFontPath(r, CTX)).toEqual({ path: "auto.ttf", provenance: "auto" });
  });

  it("rung 4: falls to the highest-coverage family for the target script", () => {
    const ctx = {
      defaultTargLang: "it",
      familiesByLang: {
        "it": [
          { family: "Best", best_path: "best.ttf", best_coverage: 1, weights: [] },
          { family: "Worse", best_path: "worse.ttf", best_coverage: 0.9, weights: [] },
        ],
      },
    };
    expect(resolveFontPath(region(), ctx)).toEqual({ path: "best.ttf", provenance: "script_default" });
  });

  it("rung 4 uses the REGION's own target language, not the global one", () => {
    const ctx = {
      defaultTargLang: "it",
      familiesByLang: {
        "it": [{ family: "Latin", best_path: "latin.ttf", best_coverage: 1, weights: [] }],
        "ja": [{ family: "Gothic", best_path: "gothic.ttc#0", best_coverage: 1, weights: [] }],
      },
    };
    expect(resolveFontPath(region({ target_language: "ja" }), ctx).path).toBe("gothic.ttc#0");
  });

  it("rung 5: nothing resolved is reported, not disguised", () => {
    expect(resolveFontPath(region(), CTX)).toEqual({ path: null, provenance: "unresolved" });
  });
});

describe("dupeSpec — scribe defaults", () => {
  it("defaults align_h to center, as scribe does, not to left", () => {
    expect(dupeSpec(region(), CTX).alignH).toBe("center");
  });

  it("adds kerning to tracking — scribe widens a line by both", () => {
    const r = region({ style_profile: { tracking: 3, kerning: 2 } as InstText["style_profile"] });
    expect(dupeSpec(r, CTX).letterSpacingPx).toBe(5);
  });

  it("draws no shadow unless the region carries one", () => {
    expect(dupeSpec(region(), CTX).shadow).toBeNull();
  });

  it("fills a partial shadow from scribe's DEFAULT_SHADOW", () => {
    const r = region({ style_profile: { shadow: { blur: 9 } } as InstText["style_profile"] });
    expect(dupeSpec(r, CTX).shadow).toEqual({
      offset_x: 2, offset_y: 2, blur: 9, color: "#00000080",
    });
  });

  it("never takes the detected SOURCE size as an explicit target size", () => {
    // characteristics.size is tuned for the source string's length; using
    // it would skip the auto-fit that adapts to the translation's length
    const r = region({ characteristics: { size: 31 } as InstText["characteristics"] });
    expect(dupeSpec(r, CTX).explicitSizePx).toBeNull();
  });

  it("resolves a path to an @font-face alias with no family-list lookup", () => {
    const r = region({ resolved_font_family: "C:\\Windows\\Fonts\\arial.ttf" });
    expect(dupeSpec(r, CTX).cssFontFamily).toBe('"tofu-preview-CWindowsFontsarialttf"');
  });
});

describe("shouldRenderVertical", () => {
  const tall = { width: 30, height: 120 };

  it("stacks a tall CJK region even with no explicit orientation", () => {
    expect(shouldRenderVertical(tall, "ja", "居酒屋")).toBe(true);
  });

  it("does not stack a wide CJK region", () => {
    expect(shouldRenderVertical({ width: 200, height: 40 }, "ja", "居酒屋")).toBe(false);
  });

  it("does not stack a single character — the aspect ratio says nothing", () => {
    expect(shouldRenderVertical(tall, "ja", "居")).toBe(false);
  });

  it("does not stack a non-CJK language however tall the box", () => {
    expect(shouldRenderVertical(tall, "it", "MURI")).toBe(false);
  });

  it("an explicit orientation decides it outright, both ways", () => {
    expect(shouldRenderVertical({ width: 200, height: 40 }, "it", "MURI", "vertical")).toBe(true);
    expect(shouldRenderVertical(tall, "ja", "居酒屋", "horizontal")).toBe(false);
  });
});

describe("adviseShrink — report the constraint, never repair it", () => {
  const fit = (fontSizePx: number): FitResult => ({ fontSizePx, lines: ["x"], lineAdvancePx: fontSizePx * 1.2 });
  const pair = (srcA: number, srcB: number) => [
    region({ id: "r1", characteristics: { size: srcA } as InstText["characteristics"] }),
    region({ id: "r2", characteristics: { size: srcB } as InstText["characteristics"] }),
  ];
  const unit = (): SemanticTextUnit => ({ id: "u1", region_ids: ["r1", "r2"] } as SemanticTextUnit);
  /** stand in for canvas ink measurement: ink = em * factor per region */
  const ink = (map: Record<string, number>) => (inst: InstText) => map[inst.id] ?? 0;

  it("the la-rue-sans-nom case: flags the expansion-pinned line, not its healthy neighbour", () => {
    // real measured numbers — source ink 81/84, target ink 85/61
    const out = adviseShrink(
      { r1: fit(117), r2: fit(82) }, pair(81, 84), [unit()],
      ink({ r1: 85, r2: 61 }),
    );
    expect(out.r1).toBeUndefined();               // ink 85 vs source 81 — fine
    expect(out.r2.ratio).toBeCloseTo(0.73, 2);    // ink 61 vs source 84
    expect(out.r2.sourceInkPx).toBe(84);
    expect(out.r2.targetInkPx).toBe(61);
  });

  it("names the unit peers that kept full size", () => {
    const out = adviseShrink(
      { r1: fit(117), r2: fit(82) }, pair(81, 84), [unit()], ink({ r1: 85, r2: 61 }),
    );
    expect(out.r2.groupId).toBe("u1");
    expect(out.r2.unshrunkPeers).toEqual(["r1"]);
  });

  it("never changes a fit — advisory only", () => {
    const fits = { r1: fit(117), r2: fit(82) };
    const before = JSON.stringify(fits);
    adviseShrink(fits, pair(81, 84), [unit()], ink({ r1: 85, r2: 61 }));
    expect(JSON.stringify(fits)).toBe(before);
  });

  it("stays quiet when the target holds its source's optical size", () => {
    const out = adviseShrink({ r1: fit(40) }, pair(50, 50).slice(0, 1), [], ink({ r1: 50 }));
    expect(out).toEqual({});
  });

  it("fires just past the threshold, not at it", () => {
    const at = adviseShrink({ r1: fit(40) }, pair(100, 100).slice(0, 1), [], ink({ r1: 100 * SHRINK_WARN }));
    expect(at.r1).toBeUndefined();
    const under = adviseShrink({ r1: fit(40) }, pair(100, 100).slice(0, 1), [], ink({ r1: 100 * SHRINK_WARN - 1 }));
    expect(under.r1).toBeDefined();
  });

  it("says nothing about a region the user sized explicitly", () => {
    const members = pair(81, 84);
    members[1].style_profile = { font_size: 20 } as InstText["style_profile"];
    const out = adviseShrink({ r1: fit(117), r2: fit(20) }, members, [unit()], ink({ r1: 85, r2: 14 }));
    expect(out.r2).toBeUndefined();
  });

  it("needs source evidence — no detected size means no claim", () => {
    const members = [region({ id: "r1" })];
    expect(adviseShrink({ r1: fit(40) }, members, [], ink({ r1: 10 }))).toEqual({});
  });

  it("advises outside a semantic unit too, just without peers", () => {
    const out = adviseShrink({ r1: fit(40) }, pair(100, 100).slice(0, 1), [], ink({ r1: 50 }));
    expect(out.r1.ratio).toBeCloseTo(0.5, 2);
    expect(out.r1.groupId).toBeUndefined();
    expect(out.r1.unshrunkPeers).toBeUndefined();
  });
});

describe("fontIdentity — one answer per question, everywhere it is asked", () => {
  const matched = (over = {}) => region({
    resolved_font_family: "C:/Windows/Fonts/arial.ttf",
    font_match: {
      status: "review", confidence: 0.53,
      recommended_substitute: {
        font_path: "C:/Windows/Fonts/BASKVILL.TTF",
        family: "Baskerville Old Face", subfamily: "Regular", score: 0.78,
      },
    } as InstText["font_match"],
    ...over,
  });

  it("the region table and the preview name the same face", () => {
    // the la-rue-sans-nom defect: the Font column read
    // `explicit ?? resolved_font_family` and printed "Arial Regular" while
    // the canvas beside it painted Baskerville Old Face
    const r = matched();
    expect(fontIdentity(r, CTX).path).toBe(dupeSpec(r, CTX).fontPath);
  });

  it("names the substitute's own family rather than a file stem", () => {
    // font_match carries family/subfamily; familiesByLang is truncated to
    // 24 families and would degrade this to "BASKVILL"
    expect(fontIdentity(matched(), CTX).label).toBe("Baskerville Old Face");
  });

  it("marks an inferred face as inferred", () => {
    expect(fontIdentity(matched(), CTX).provenance).toBe("nearest_neighbor");
  });

  it("one ladder now answers for the render too", () => {
    // This used to assert the opposite: `useMatch: false` returned the auto
    // fallback because /api/render read only style_profile.font_family, so a
    // localiser approved Baskerville and shipped Arial. The server now lends
    // an auto region its matched face for the duration of the render
    // (_matched_faces_applied), so there is one answer, not two.
    const identity = fontIdentity(matched(), CTX);
    expect(identity.path).toBe("C:/Windows/Fonts/BASKVILL.TTF");
    expect(identity.provenance).toBe("nearest_neighbor");
  });

  it("a human's explicit pick outranks the match", () => {
    const r = matched({
      style_profile: { font_family: "C:/Windows/Fonts/times.ttf" } as InstText["style_profile"],
    });
    expect(fontIdentity(r, CTX).path).toBe("C:/Windows/Fonts/times.ttf");
    expect(fontIdentity(r, CTX).provenance).toBe("explicit");
  });

  it("falls back to the file's name, never to a bare style word", () => {
    const r = region({ resolved_font_family: "C:/Windows/Fonts/GARA.TTF" });
    expect(fontIdentity(r, CTX).label).toBe("GARA");
  });

  it("says 'auto' only when nothing at all resolved", () => {
    expect(fontIdentity(region(), CTX).label).toBe("auto");
  });

  it("keeps a detected style word when there is no face to name", () => {
    const r = region({ characteristics: { font_style: "bold italic" } as InstText["characteristics"] });
    expect(fontIdentity(r, CTX).label).toBe("bold italic");
  });

  it("strips a collection index so the label is readable", () => {
    const r = region({ resolved_font_family: "C:/Windows/Fonts/batang.ttc#1" });
    expect(fontIdentity(r, CTX).label).toBe("batang");
  });
});
