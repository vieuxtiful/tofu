// 🍢 textFit — client-side mirror of src/tofu/layers/scribe.py's
// _fit_wrapped(): greedy word-wrap (char-wrap for space-less scripts like
// CJK) + binary-search the largest font size that fits the wrapped block
// into a bbox. exists so the Translate-tab preview can show roughly what
// the server will actually render — a translation longer or shorter than
// the source text changes the server's chosen size/line-count, and the
// old single-line `overflow: hidden` preview never reflected that.
//
// not pixel-parity with PIL (a browser's text shaper and PIL's are
// different engines) — the goal is "close enough to plan around", with
// final pixel-perfect fidelity coming from the actual Render tab, exactly
// as the product vision describes.

export const MIN_FONT_PX = 6;

export interface FitResult {
  fontSizePx: number;
  lines: string[];
  /** vertical distance between successive line BASELINES, i.e. one
   * line's nominal height plus the leading gap — mirrors scribe's
   * `_line_height(font) + spacing` used when it composites each
   * wrapped line. */
  lineAdvancePx: number;
}

/** greedy wrap targeting maxWidth: word-wrap when the text has
 * space-delimited words, character-wrap otherwise (CJK and other
 * scripts that don't use spaces between words) — mirrors scribe's
 * `_wrap_lines()` exactly, including its "always keep at least one unit
 * on an empty line" rule so a single word wider than the box doesn't
 * vanish. */
function wrapLines(ctx: CanvasRenderingContext2D, text: string, maxWidth: number): string[] {
  if (!text) return [""];
  const useWords = text.trim().includes(" ");
  const units = useWords ? text.split(" ") : Array.from(text);
  const sep = useWords ? " " : "";
  const lines: string[] = [];
  let cur = "";
  for (const unit of units) {
    const candidate = cur ? `${cur}${sep}${unit}` : unit;
    const width = ctx.measureText(candidate).width;
    if (width <= maxWidth || !cur) {
      cur = candidate;
    } else {
      lines.push(cur);
      cur = unit;
    }
  }
  if (cur || lines.length === 0) lines.push(cur);
  return lines;
}

/** the font's own NOMINAL ascent+descent at the current ctx.font size —
 * text-content-independent, like PIL's `font.getmetrics()`. scribe uses
 * this (x 0.2) as the default leading between lines. `fontBoundingBox*`
 * is the canvas TextMetrics field matching this ("bounding box of the
 * font", not of the specific glyphs measured); falls back to a 1.2x
 * multiple of the font's pixel size on engines that don't expose it. */
function nominalLineHeight(ctx: CanvasRenderingContext2D, fontSizePx: number): number {
  const m = ctx.measureText("Hg");
  const ascent = m.fontBoundingBoxAscent;
  const descent = m.fontBoundingBoxDescent;
  if (ascent != null && descent != null) return ascent + descent;
  return fontSizePx * 1.2;
}

/** the wrapped block's ACTUAL ink bounding box — width of the widest
 * line, height as the sum of each line's own tight ascent/descent plus
 * inter-line spacing. mirrors scribe's fit-check, which deliberately
 * uses PIL's `multiline_textbbox` (real ink extent) instead of nominal
 * font metrics: nominal metrics over-estimate height for no-descender
 * text ("SALE" et al) and pick an unnecessarily smaller font — a real
 * bug this project measured and fixed in scribe.py itself. */
function actualBlockSize(
  ctx: CanvasRenderingContext2D, lines: string[], spacingPx: number,
): { width: number; height: number } {
  let width = 0;
  let height = 0;
  for (const line of lines) {
    const m = ctx.measureText(line || " ");
    width = Math.max(width, m.width);
    const asc = m.actualBoundingBoxAscent;
    const desc = m.actualBoundingBoxDescent;
    height += asc != null && desc != null ? asc + desc : parseFloat(ctx.font) || 12;
  }
  height += Math.max(0, lines.length - 1) * spacingPx;
  return { width, height };
}

/**
 * binary-search the largest font size whose greedy-wrapped `text` fits
 * (boxWidth, boxHeight), or use `explicitSizePx` verbatim when given —
 * the same two modes as `scribe._fit_wrapped()`.
 *
 * `ctx` must already have had its font FAMILY/weight/style set via
 * `ctx.font` conventions are irrelevant here — this function sets
 * `ctx.font`'s SIZE itself on every candidate, but the caller is
 * responsible for the family/weight/style portion being correct
 * beforehand (pass them in via `fontSpec`, a CSS font shorthand suffix
 * like `"italic 700 {size}px \"MyFont\""` with `{size}` as a literal
 * placeholder this function substitutes).
 */
export function fitWrappedText(
  ctx: CanvasRenderingContext2D,
  text: string,
  boxWidth: number,
  boxHeight: number,
  fontSpec: string,  // e.g. `normal 700 {size}px "tofu-preview-xyz"`
  explicitSizePx?: number | null,
  /** style_profile.leading override, natural px — mirrors _fit_wrapped's
   * `leading` param: a fixed inter-line gap instead of the font-size-
   * derived default. */
  leadingOverridePx?: number | null,
  /** Explicit layout choice from Localized Asset Canvas.  Off keeps one
   * glyph run even if it crosses the cube; on enables real wrap. */
  wrapText = false,
): FitResult {
  const setSize = (size: number) => { ctx.font = fontSpec.replace("{size}", String(size)); };
  const safeBoxWidth = Math.max(1, boxWidth);
  const safeBoxHeight = Math.max(1, boxHeight);
  const spacingFor = (size: number) => leadingOverridePx ?? nominalLineHeight(ctx, size) * 0.2;

  if (explicitSizePx && explicitSizePx > 0) {
    setSize(explicitSizePx);
    const lines = wrapText ? wrapLines(ctx, text, safeBoxWidth) : [text];
    const spacing = spacingFor(explicitSizePx);
    return { fontSizePx: explicitSizePx, lines, lineAdvancePx: nominalLineHeight(ctx, explicitSizePx) + spacing };
  }

  let lo = MIN_FONT_PX;
  let hi = Math.max(MIN_FONT_PX + 1, Math.floor(safeBoxHeight * 2));
  setSize(MIN_FONT_PX);
  let best: FitResult = {
    fontSizePx: MIN_FONT_PX,
    lines: wrapText ? wrapLines(ctx, text, safeBoxWidth) : [text],
    lineAdvancePx: nominalLineHeight(ctx, MIN_FONT_PX) * 1.2,
  };

  while (lo <= hi) {
    const mid = Math.floor((lo + hi) / 2);
    setSize(mid);
    const lines = wrapText ? wrapLines(ctx, text, safeBoxWidth) : [text];
    const spacing = spacingFor(mid);
    const { width, height } = actualBlockSize(ctx, lines, spacing);
    if (width <= safeBoxWidth && height <= safeBoxHeight) {
      best = { fontSizePx: mid, lines, lineAdvancePx: nominalLineHeight(ctx, mid) + spacing };
      lo = mid + 1;
    } else {
      hi = mid - 1;
    }
  }
  return best;
}

let sharedCtx: CanvasRenderingContext2D | null = null;
/** one offscreen 2D context reused for every measurement — creating a
 * canvas per call is wasteful and unnecessary since `measureText` never
 * paints anything. */
export function getMeasureContext(): CanvasRenderingContext2D {
  if (!sharedCtx) {
    const canvas = document.createElement("canvas");
    const ctx = canvas.getContext("2d");
    if (!ctx) throw new Error("2D canvas context unavailable");
    sharedCtx = ctx;
  }
  return sharedCtx;
}
