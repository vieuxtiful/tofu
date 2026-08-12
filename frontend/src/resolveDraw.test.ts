import { describe, it, expect, vi } from "vitest";
import { resolveDraw } from "./resolveDraw";

const drawn = { x: 12, y: 34, width: 210, height: 56 };

/** What a refinement pass on the prem-sais phrase actually returned: the
 *  highest-confidence SUB-detection, not the phrase, and a misread of it. */
const refineToAFragment = vi.fn(async () => ({
  regions: [{ bbox: { x: 20, y: 40, width: 30, height: 20 }, text: "孝支" }],
}));

describe("resolveDraw in Guided capture", () => {
  it("never calls the recogniser", async () => {
    const refine = vi.fn(refineToAFragment);
    await resolveDraw(drawn, { guided: true, assetId: "a1", refine });
    expect(refine).not.toHaveBeenCalled();
  });

  it("keeps the drawn rectangle exactly", async () => {
    const result = await resolveDraw(drawn, {
      guided: true, assetId: "a1", refine: refineToAFragment,
    });
    expect(result.bbox).toEqual(drawn);
  });

  it("sends no text, leaving the Block to speak for itself", async () => {
    // The server fills this in from the stored Block. A client that named
    // the text could contradict the declaration the box was drawn under.
    const result = await resolveDraw(drawn, {
      guided: true, assetId: "a1", refine: refineToAFragment,
    });
    expect(result.text).toBeUndefined();
  });
});

describe("resolveDraw outside Guided capture", () => {
  it("still snaps to the refined box and keeps its text", async () => {
    // Auto and Manual have no declaration, so snapping a rough rectangle to
    // the text inside it is the whole service and must not regress.
    const result = await resolveDraw(drawn, {
      guided: false, assetId: "a1", refine: refineToAFragment,
    });
    expect(result.bbox).toEqual({ x: 20, y: 40, width: 30, height: 20 });
    expect(result.text).toBe("孝支");
  });

  it("keeps the drawn box when refinement fails, and says so once", async () => {
    const onRefineFailed = vi.fn();
    const result = await resolveDraw(drawn, {
      guided: false, assetId: "a1", onRefineFailed,
      refine: async () => { throw new Error("EasyOCR is not installed"); },
    });
    expect(result.bbox).toEqual(drawn);
    expect(result.text).toBeUndefined();
    expect(onRefineFailed).toHaveBeenCalledTimes(1);
  });

  it("keeps the drawn box when refinement returns a degenerate rectangle", async () => {
    const result = await resolveDraw(drawn, {
      guided: false, assetId: "a1",
      refine: async () => ({ regions: [{ bbox: { x: 0, y: 0, width: 0, height: 0 }, text: "" }] }),
    });
    expect(result.bbox).toEqual(drawn);
  });
});
