import { describe, expect, it } from "vitest";
import { textLangMatchesTarget } from "./detectLanguage";

describe("target language guard", () => {
  it("distinguishes Traditional from Simplified Chinese", () => {
    expect(textLangMatchesTarget("飛驒小坂車站", "zh-CN")).toBe(false);
    expect(textLangMatchesTarget("飞驒小坂车站", "zh-CN")).toBe(true);
    expect(textLangMatchesTarget("飞驒小坂车站", "zh-TW")).toBe(false);
    expect(textLangMatchesTarget("飛驒小坂車站", "zh-TW")).toBe(true);
  });

  it("does not apply Chinese variant policy to Japanese", () => {
    expect(textLangMatchesTarget("丸太造りの駅舎飛騨小坂駅", "ja-JP")).toBe(true);
  });

  it("normalizes regional language tags before script checks", () => {
    expect(textLangMatchesTarget("日本語", "en-GB")).toBe(false);
    expect(textLangMatchesTarget("English", "en-GB")).toBe(true);
  });

  it("rejects Latin typing in CJK target fields", () => {
    expect(textLangMatchesTarget("English entered here", "zh-CN")).toBe(false);
    expect(textLangMatchesTarget("English entered here", "ja-JP")).toBe(false);
    expect(textLangMatchesTarget("English entered here", "ko-KR")).toBe(false);
  });
});
