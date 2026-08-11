import { describe, expect, it } from "vitest";
import { insertedRange, isRecommendationActivationKey } from "./AnimatedCaretTextarea";

describe("insertedRange", () => {
  it("finds an insertion at the caret", () => {
    expect(insertedRange("abcd", "abXYcd")).toEqual({ start: 2, end: 4 });
  });

  it("treats deletion as immediate", () => {
    expect(insertedRange("abcd", "acd")).toBeNull();
  });

  it("reveals the inserted side of a replacement", () => {
    expect(insertedRange("周屋", "湯屋追加")).toEqual({ start: 0, end: 4 });
  });
});

describe("Windows IME-adjacent recommendation ownership", () => {
  it("requires an explicit ToFU activation gesture", () => {
    expect(isRecommendationActivationKey("ArrowDown")).toBe(false);
    expect(isRecommendationActivationKey("Enter")).toBe(false);
    expect(isRecommendationActivationKey(" ")).toBe(false);
    expect(isRecommendationActivationKey(" ", true, false)).toBe(true);
    expect(isRecommendationActivationKey("ArrowDown", false, true)).toBe(true);
  });
});
