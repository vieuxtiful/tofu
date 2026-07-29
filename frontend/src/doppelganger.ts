// 🍢 doppelganger — the Translate tab's "dupe".
//
// In a kitchen the DUPE is the duplicate ticket clipped over the pass: the
// slip that tells the line exactly what the plate has to look like before
// anyone starts cooking it.  This module is that ticket for a localized
// asset.  It resolves, per region, the same face / size / style decisions
// scribe.render() will make, so the Translate-tab preview stops being a
// rough guide and starts being a prediction — a doppelgänger of the
// pre-Garnish render.
//
// Two responsibilities, both pure (no DOM, no canvas, no fetch) so they
// can be reasoned about and tested on their own:
//
//   dupeSpec()       — resolve one region's typographic ticket, including
//                      the font-resolution LADDER (below).
//   harmonizeUnits() — the one place this preview deliberately does MORE
//                      than scribe: keep a semantic unit's regions
//                      optically consistent when the target language
//                      changes their lengths.
//
// Everything that mirrors a server rule cites the rule it mirrors.  When
// scribe changes, these are the places that have to change with it.

import { InstText, FontFamily, SemanticTextUnit } from "./api";
import { FitResult } from "./textFit";
import { fontNameForPath, weightLabel } from "./FontCombobox";

// ── constants mirrored from src/tofu/layers/scribe.py ────────────────────

/** scribe.CJK_VERTICAL_LANGS — languages whose signage stacks vertically. */
export const CJK_VERTICAL_LANGS = new Set([
  "ja", "zh-cn", "zh-tw", "zh-hk", "zh-mo", "zh-sg",
]);
/** scribe.VERTICAL_ASPECT_MIN */
export const VERTICAL_ASPECT_MIN = 1.3;
/** scribe.VERTICAL_ROW_FACTOR — row height as a multiple of font size. */
export const VERTICAL_ROW_FACTOR = 1.15;
/** scribe.ITALIC_SHEAR (0.2) as a CSS angle: atan(0.2) ≈ 11.31°. */
export const ITALIC_SHEAR_DEG = 11.3099;
/** scribe.DEFAULT_SHADOW — merged under any partial style_profile.shadow. */
export const DEFAULT_SHADOW = { offset_x: 2, offset_y: 2, blur: 2, color: "#00000080" };
/** scribe re-fits super/subscript at round(size * 0.65), auto-fit only. */
export const SUB_SUPER_RATIO = 0.65;

/**
 * scribe._should_render_vertical().  Note the important half: an explicit
 * orientation decides it outright, but with NO orientation set a CJK
 * target in a narrow-and-tall box still stacks — the shape cicerone's
 * merge_vertical_columns() produces for genuine stacked signage.  A
 * preview that only honoured the explicit flag showed such regions
 * horizontally and then the render turned them vertical.
 */
export function shouldRenderVertical(
  box: { width: number; height: number },
  lang: string | null | undefined,
  text: string,
  orientation?: string | null,
): boolean {
  if (orientation === "vertical") return true;
  if (orientation === "horizontal") return false;
  if (!lang || !CJK_VERTICAL_LANGS.has(lang)) return false;
  if (box.width <= 0 || text.length < 2) return false;
  return box.height >= box.width * VERTICAL_ASPECT_MIN;
}

// ── font resolution ──────────────────────────────────────────────────────

/**
 * Where a region's previewed face came from.  Surfaced in the UI because
 * the bottom two rungs mean something the user needs to know: the preview
 * is showing a face the RENDER will not use.
 */
export type FontProvenance =
  | "explicit"          // style_profile.font_family — the user picked it
  | "nearest_neighbor"  // font_match glyph-shape substitute; render still uses auto
  | "auto"              // resolved_font_family — the generic FALLBACK_FONTS default
  | "script_default"    // highest-coverage family for the target script
  | "unresolved";       // nothing resolved; sizes here are not trustworthy

/** confidence floor is deliberately absent — see resolveFontPath(). */
export interface DupeContext {
  familiesByLang?: Record<string, FontFamily[]>;
  defaultTargLang: string;
}

/**
 * The ladder.  Each rung is reached ONLY when every rung above it yields
 * nothing, so a HUMAN's pick is never overridden by an inference.
 *
 *   1 explicit         style_profile.font_family
 *   2 nearest_neighbor font_match.recommended_substitute.font_path
 *   3 auto             resolved_font_family (the generic default)
 *   4 script_default   best-coverage family for the target script
 *   5 unresolved       generic sans-serif
 *
 * Rung 2 is the nearest-neighbour rung.  font_matching.py scores installed
 * faces against the SOURCE glyph outlines (Dice + chamfer + projection
 * profile + aspect) and marks the winner "matched" only at confidence
 * >= 0.82 with margin >= 0.075, otherwise "review".  We deliberately
 * accept "review" too, because the rung below is not a competing opinion
 * about this region — see next paragraph — and a review-grade shape match
 * is strictly better information than a constant.  The matched/review
 * distinction is carried in the badge, not used to suppress the match.
 *
 * Why nearest-neighbour outranks "auto": resolved_font_family is NOT an
 * adaptation to this region.  scribe.resolve_auto_font() falls through to
 * _default_fallback_path(), which is the first installed FALLBACK_FONTS
 * entry — arial.ttf for every Latin region on a Windows box, regardless
 * of what the sign actually looks like.  Ranking that constant above
 * per-region glyph evidence made rung 2 unreachable for Latin text
 * entirely: measured on the la-rue-sans-nom plaque, whose serif lettering
 * matched Centaur at 0.784 and was drawn in Arial anyway.  A generic
 * default is a floor, not a verdict, so it sits below the evidence.
 *
 * This ladder now describes the render as well as the preview.  It used to
 * take a `useMatch: false` option, because /api/render read only
 * style_profile.font_family and so drew the auto fallback where this
 * preview drew the match — one face approved, a different face shipped.
 * The server closed that from its own side (_matched_faces_applied in
 * server/main.py lets an auto region render with the matched face for the
 * duration of the render, without persisting it), so there is no longer a
 * second question to answer and no second answer to pass around.
 *
 * Nothing here writes back to the manifest, so the invariant asserted by
 * tests/test_font_matching.py ("visual font retrieval remains evidence
 * only; it never mutates the style") still holds — a preview mutates
 * nothing, and neither does the render.
 */
export function resolveFontPath(
  inst: InstText, ctx: DupeContext,
): { path: string | null; provenance: FontProvenance } {
  const explicit = inst.style_profile?.font_family;
  if (explicit) return { path: explicit, provenance: "explicit" };

  const match = inst.font_match;
  const substitute = match?.recommended_substitute;
  if (match && match.status !== "unavailable" && substitute?.font_path) {
    return { path: substitute.font_path, provenance: "nearest_neighbor" };
  }

  if (inst.resolved_font_family) {
    return { path: inst.resolved_font_family, provenance: "auto" };
  }

  const lang = inst.target_language ?? ctx.defaultTargLang;
  // families arrive already ranked by best-coverage face (fonts.py's
  // families_with_weights), so [0] is the highest-coverage family for
  // this script — the same ordering ToFU's own recommend() uses.
  const best = ctx.familiesByLang?.[lang]?.[0]?.best_path;
  if (best) return { path: best, provenance: "script_default" };

  return { path: null, provenance: "unresolved" };
}

/** How a resolved face should be named and drawn wherever it is shown. */
export interface FontIdentity {
  path: string | null;
  provenance: FontProvenance;
  /** "{Family} {Weight}", or the file's own name when the family list has
   *  not got it — never a bare style word like "bold". */
  label: string;
  /** CSS family for previewing the label in its own face, when resolvable. */
  cssFontFamily?: string;
}

/**
 * One answer to "which face is this region going to be shown in", for every
 * surface that has to say so.
 *
 * The Translate tab used to answer this twice: the region table resolved
 * `explicit ?? resolved_font_family` while the preview canvas ran the full
 * ladder, so a region matched to Baskerville Old Face was listed as "Arial
 * Regular" beside a preview drawn in Baskerville.  Two answers to one
 * question is worse than either answer, so both now come through here.
 *
 * The family NAME is taken from the match itself where there is one, rather
 * than from familiesByLang: that list is truncated to 24 families and only
 * fetched for languages some other panel already asked for, so looking a
 * substitute up in it degrades a real family name to a bare file stem.
 */
export function fontIdentity(
  inst: InstText, ctx: DupeContext,
): FontIdentity {
  const { path, provenance } = resolveFontPath(inst, ctx);
  if (!path) {
    const detected = (inst.characteristics?.font_style ?? "").trim();
    return {
      path: null,
      provenance,
      label: detected && detected !== "regular" ? detected : "auto",
    };
  }

  const cssFontFamily = `"${fontNameForPath(path)}"`;
  const substitute = inst.font_match?.recommended_substitute;
  if (provenance === "nearest_neighbor" && substitute?.family) {
    const subfamily = substitute.subfamily && substitute.subfamily !== "Regular"
      ? ` ${substitute.subfamily}` : "";
    return { path, provenance, label: `${substitute.family}${subfamily}`, cssFontFamily };
  }

  const lang = inst.target_language ?? ctx.defaultTargLang;
  const families = ctx.familiesByLang?.[lang] ?? [];
  const family = families.find(
    (f) => f.best_path === path || f.weights.some((w) => w.path === path),
  );
  const weight = family?.weights.find((w) => w.path === path);
  if (family) {
    const suffix = weight ? ` ${weightLabel(weight)}` : "";
    return { path, provenance, label: `${family.family}${suffix}`, cssFontFamily };
  }
  return { path, provenance, label: fontFileLabel(path), cssFontFamily };
}

/** Readable name for a font file when no family record is available. */
export function fontFileLabel(path: string): string {
  const base = path.split("#")[0].split(/[\\/]/).pop() ?? path;
  return base.replace(/\.(ttf|otf|ttc|otc)$/i, "");
}

export interface DupeSpec {
  fontPath: string | null;
  provenance: FontProvenance;
  /** CSS font-family: an @font-face alias for a real path, else generic. */
  cssFontFamily: string;
  /** render() will shear an upright face rather than load an italic one. */
  syntheticItalic: boolean;
  color?: string;
  /** style_profile.font_size ALONE — null means run the binary search. */
  explicitSizePx: number | null;
  alignH: string;
  alignV: string | null;
  justification: string | null;
  indentPx: number;
  /** tracking + kerning: scribe adds BOTH to a line's width. */
  letterSpacingPx: number;
  leadingPx: number | null;
  baselineShiftPx: number;
  strokeColor: string | null;
  strokeWidthPx: number;
  strokePosition: string;
  subscript: boolean;
  superscript: boolean;
  underline: boolean;
  underlineOffsetPx: number | null;
  underlineWidthPx: number | null;
  shadow: { offset_x: number; offset_y: number; blur: number; color: string } | null;
  rotationDeg: number;
  isRtl: boolean;
  isVertical: boolean;
  wrapText: boolean;
  tsume: number;
}

export function dupeSpec(inst: InstText, ctx: DupeContext): DupeSpec {
  const { path, provenance } = resolveFontPath(inst, ctx);
  const sp = inst.style_profile;
  const targetText = inst.target_text ?? "";
  const lang = inst.target_language ?? ctx.defaultTargLang;

  // Shadow is drawn only when the region actually asks for one; scribe
  // guards on `if s.shadow:` and merges DEFAULT_SHADOW into whatever
  // subset of keys is present.
  const shadow = sp?.shadow ? { ...DEFAULT_SHADOW, ...sp.shadow } : null;

  return {
    fontPath: path,
    provenance,
    // A resolved path always becomes an alias, with no dependency on the
    // /api/fonts family list: that list is truncated to 24 families and
    // is only fetched for languages some other panel already asked for,
    // so gating on it silently degraded both the MEASUREMENT and the
    // paint to sans-serif for any font outside the cut.
    cssFontFamily: path ? `"${fontNameForPath(path)}"` : "sans-serif",
    syntheticItalic: Boolean(inst.resolved_synthetic_italic),
    color: sp?.color ?? inst.characteristics?.color ?? undefined,
    // render() passes style_profile.font_size ALONE as _fit_wrapped's
    // explicit_size — None means true binary-search auto-fit.  Falling
    // back to characteristics.size (the DETECTED SOURCE size) would skip
    // the auto-fit and draw at a size tuned for the source string's
    // length, not the translation's.
    explicitSizePx: sp?.font_size ?? null,
    alignH: sp?.align_h ?? "center",   // scribe: `ah = s.align_h or "center"`
    alignV: sp?.align_v ?? null,        // scribe: `av = s.align_v or "middle"`
    justification: sp?.justification ?? null,
    indentPx: sp?.indent ?? 0,
    letterSpacingPx: (sp?.tracking ?? 0) + (sp?.kerning ?? 0),
    leadingPx: sp?.leading ?? null,
    baselineShiftPx: sp?.baseline_shift ?? 0,
    strokeColor: sp?.stroke_color ?? null,
    strokeWidthPx: sp?.stroke_width ?? 0,
    strokePosition: sp?.stroke_position ?? "outer",
    subscript: Boolean(sp?.subscript),
    superscript: Boolean(sp?.superscript),
    underline: Boolean(sp?.underline),
    underlineOffsetPx: sp?.underline_offset ?? null,
    underlineWidthPx: sp?.underline_width ?? null,
    shadow,
    rotationDeg: sp?.transform?.rotation ?? inst.characteristics?.positioning?.rotation_deg ?? 0,
    isRtl: sp?.word_order === "rtl",
    isVertical: shouldRenderVertical(
      inst.bounding_box, lang, targetText, sp?.target_orientation,
    ),
    wrapText: Boolean(sp?.transform?.wrap_text),
    tsume: sp?.tsume ?? 0,
  };
}

/**
 * The CSS font shorthand `fitWrappedText` measures with.
 *
 * Always normal/normal.  Weight and italic are properties of the resolved
 * FILE, not of the request: scribe picks a bold or italic face by
 * resolve_face()'s sibling search and then draws it plainly.  Asking the
 * browser for 700 on top of an already-bold file makes it synthesize a
 * second layer of boldness, widening glyphs past what PIL draws.  And a
 * synthetic italic is a post-layout SHEAR in scribe (applied in
 * _render_line_layer, after _fit_wrapped has measured), so it must not
 * enter the measurement here either.
 */
export function measureFontSpec(spec: DupeSpec): string {
  return `{size}px ${spec.cssFontFamily}`;
}

// ── cross-region proportionality ─────────────────────────────────────────

/**
 * A target region whose glyphs came out materially smaller than the source
 * glyphs they replace.
 */
export interface SizeAdvisory {
  /** cicerone's measured source ink height (ascender-to-descender). */
  sourceInkPx: number;
  /** the target's ink height at its fitted size, same quantity. */
  targetInkPx: number;
  /** targetInkPx / sourceInkPx; < 1 means the translation lost height. */
  ratio: number;
  /** Basil unit whose other members did NOT shrink, when there is one. */
  groupId?: string;
  unshrunkPeers?: string[];
}

/**
 * Below this, the drop is visible as a broken hierarchy rather than as
 * normal fitting slack.  Measured on the la-rue-sans-nom plaque:
 * "SANS-NOM" → "SENZA NOME" lands at 0.73 (two extra characters in the
 * same 543px width, hard against the box edge) while its unit-mate
 * "La rue" → "La via" lands at 1.05 and needs no attention at all.
 */
export const SHRINK_WARN = 0.85;

/**
 * Report — never repair — regions whose translation could not keep the
 * source's optical size.
 *
 * This replaces an earlier pass that equalized fitted sizes across a
 * Basil unit.  That was wrong twice over, and the la-rue-sans-nom plaque
 * shows both faults in one asset:
 *
 *   1. It gated on source INK heights (81px / 84px, "coherent") and then
 *      equalized EM sizes (117 → 82).  Those are different quantities:
 *      equal em is not equal optical size, and here the ems legitimately
 *      differed because the strings differ in length and box width.
 *   2. It harmonized DOWN, toward the member whose size is pinned by
 *      translation expansion.  "SENZA NOME" is 10 characters where
 *      "SANS-NOM" was 8, in the same width, so it MUST shrink; dragging
 *      "La via" down to meet it took a region that had reproduced its
 *      source almost exactly (ink 85px vs 81px) and made it 26% too small.
 *      The expansion damage got copied onto the healthy neighbour instead
 *      of being contained.
 *
 * A preview cannot fix an unavoidable constraint, so it should say what
 * the constraint is and let a human decide — shorten the copy, widen the
 * box, or accept the smaller line.  Dropping the resize also restores
 * exact parity with scribe._fit_wrapped(), which fits every region
 * independently.
 *
 * `measureInkPx` is injected so this stays free of canvas access.
 */
export function adviseShrink(
  fits: Record<string, FitResult>,
  manifest: InstText[],
  semanticUnits: SemanticTextUnit[],
  measureInkPx: (inst: InstText, fit: FitResult) => number,
): Record<string, SizeAdvisory> {
  const out: Record<string, SizeAdvisory> = {};
  const ratios = new Map<string, number>();

  for (const inst of manifest) {
    const fit = fits[inst.id];
    const sourceInkPx = inst.characteristics?.size ?? 0;
    if (!fit || sourceInkPx <= 0) continue;
    // An explicitly sized region is a stated intent, not a fitted guess —
    // its size is the user's answer, so there is nothing to advise about.
    if ((inst.style_profile?.font_size ?? 0) > 0) continue;
    const targetInkPx = measureInkPx(inst, fit);
    if (targetInkPx <= 0) continue;
    const ratio = targetInkPx / sourceInkPx;
    ratios.set(inst.id, ratio);
    if (ratio >= SHRINK_WARN) continue;
    out[inst.id] = {
      sourceInkPx,
      targetInkPx: Math.round(targetInkPx),
      ratio: Math.round(ratio * 100) / 100,
    };
  }

  // Naming the peers that held their size turns "this line shrank" into
  // "this line shrank and its neighbour on the same sign did not", which
  // is the form a person can actually act on.
  for (const unit of semanticUnits) {
    const ids = unit.region_ids ?? [];
    const peers = ids.filter((id) => (ratios.get(id) ?? 0) >= SHRINK_WARN);
    for (const id of ids) {
      if (!out[id]) continue;
      out[id].groupId = unit.id;
      if (peers.length) out[id].unshrunkPeers = peers;
    }
  }
  return out;
}
