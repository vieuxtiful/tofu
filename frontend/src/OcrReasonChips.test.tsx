import { describe, it, expect } from "vitest";
import { render, screen } from "@testing-library/react";
import OcrReasonChips from "./OcrReasonChips";
import { ocrReasonInfo, OCR_REASON_INFO } from "./ocrReasons";

describe("ocrReasonInfo", () => {
  it("returns a mapped label and description for known codes", () => {
    const info = ocrReasonInfo("below_mark_resolution");
    expect(info.label).toBe("Glyphs too small");
    expect(info.description).toContain("3px");
  });

  it("falls back to space-replaced label with empty description for unknown codes", () => {
    const info = ocrReasonInfo("some_new_backend_code");
    expect(info.label).toBe("some new backend code");
    expect(info.description).toBe("");
  });

  it("covers all nine backend reason codes", () => {
    const known = [
      "asset_unreadable",
      "region_too_small",
      "geometry_unavailable",
      "below_mark_resolution",
      "low_effective_glyph_height",
      "component_count_mismatch",
      "ambiguous_segmentation",
      "engine_unavailable",
      "no_recognized_glyphs",
    ];
    for (const code of known) {
      expect(OCR_REASON_INFO[code]).toBeDefined();
      expect(OCR_REASON_INFO[code].label.length).toBeGreaterThan(0);
      expect(OCR_REASON_INFO[code].description.length).toBeGreaterThan(0);
    }
  });
});

describe("OcrReasonChips", () => {
  it("renders one chip per reason with the mapped label", () => {
    const { container } = render(<OcrReasonChips reasons={["below_mark_resolution", "component_count_mismatch"]} />);
    const chips = container.querySelectorAll(".ocr-reason-chip");
    expect(chips.length).toBe(2);
    expect(chips[0]!.textContent).toContain("Glyphs too small");
    expect(chips[1]!.textContent).toContain("Component mismatch");
  });

  it("renders an info icon on chips that have a description (hoverable)", () => {
    const { container } = render(<OcrReasonChips reasons={["low_effective_glyph_height"]} />);
    const chip = container.querySelector(".ocr-reason-chip")!;
    expect(chip.querySelector(".ocr-reason-chip-icon")).not.toBeNull();
  });

  it("falls back to space-replaced label for an unknown code with no info icon", () => {
    const { container } = render(<OcrReasonChips reasons={["unknown_future_code"]} />);
    const chip = container.querySelector(".ocr-reason-chip")!;
    expect(chip.textContent).toContain("unknown future code");
    // No tooltip => no info icon rendered.
    expect(chip.querySelector(".ocr-reason-chip-icon")).toBeNull();
  });

  it("renders nothing visible when reasons is empty", () => {
    const { container } = render(<OcrReasonChips reasons={[]} />);
    expect(container.querySelectorAll(".ocr-reason-chip").length).toBe(0);
  });
});
