import type { BBox } from "./api";

/** What a freshly drawn rectangle turns into before it is posted.
 *
 * Extracted from the draw handler because the rule it encodes is the whole
 * of two defects, and an inline `if` in a 6000-line component is not
 * something a test can hold still.
 *
 * GUIDED DOES NOT READ. The user typed the string and was asked only where
 * it is, so a recognition pass has nothing to contribute and two ways to do
 * harm. It did both, visibly, in the prem-sais-gt manifest:
 *
 * * its text overwrote a string that was never in doubt -- a Block reading
 *   `鼜㒪` was stored as `孝支`, matched no atom, and was asked for again
 *   over a box that was already correct;
 * * its highest-confidence sub-detection replaced the drawn rectangle, so a
 *   box drawn around a whole phrase came back around one component of it.
 *
 * Auto and Manual keep refinement: there is no declaration there, so
 * snapping a rough rectangle to the text inside it is the whole service.
 */
export type DrawResolution = { bbox: BBox; text?: string };

export async function resolveDraw(
  bbox: BBox,
  {
    guided,
    assetId,
    refine,
    onRefineFailed,
  }: {
    guided: boolean;
    assetId: string;
    /** Injected so the guided path can be proven never to reach it. */
    refine: (assetId: string, bbox: BBox) => Promise<{
      regions: { bbox: BBox; text: string }[];
    }>;
    onRefineFailed?: () => void;
  },
): Promise<DrawResolution> {
  // The drawn rectangle, verbatim. The server clamps it to the asset and
  // that is the only adjustment anything makes to it.
  if (guided) return { bbox };

  try {
    const refined = await refine(assetId, bbox);
    const best = refined.regions[0];
    if (best && best.bbox.width > 0 && best.bbox.height > 0) {
      return {
        bbox: {
          x: best.bbox.x, y: best.bbox.y,
          width: best.bbox.width, height: best.bbox.height,
        },
        text: best.text,
      };
    }
  } catch {
    // Refinement is a convenience, not a prerequisite for manual capture. A
    // missing OCR engine or a hard crop must never discard the user's box.
    onRefineFailed?.();
  }
  return { bbox };
}
