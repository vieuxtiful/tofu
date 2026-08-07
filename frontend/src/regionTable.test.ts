// 🍢 ToFU — attributing a reading to arbitration
//
// Arbitration scores every observation of a region and its reading is now
// loaded into the manifest rather than only recorded. The workspace marks
// which text came from ToFU and says nothing else: the per-signal breakdown
// is six numbers a localiser cannot act on, and it buried the one line that
// could be. Those stay on ocr_provenance for audit.

import { describe, it, expect } from "vitest";
import { InstText } from "./api";
import { arbitrationReading, groundTruthReading } from "./RegionTable";

type Hypothesis = NonNullable<NonNullable<InstText["ocr_provenance"]>["hypothesis"]>;

function region(hypothesis: Hypothesis | null, text = "nos rues 4"): InstText {
  return {
    id: "r2",
    bounding_box: { x: 0, y: 0, width: 10, height: 10 },
    text,
    target_text: null,
    confidence: 0.9,
    detected_language: "fr",
    reading_order: 1,
    dnt: false,
    target_language: null,
    ocr_provenance: hypothesis ? { hypothesis } : null,
    ocr_correction: hypothesis?.promoted ? {
      applied: true,
      original_text: "nos rues 4",
      corrected_text: text,
      correction_resource: { kind: "tofu_arbitration", revision: "test" },
    } : null,
  } as InstText;
}

const BASE = {
  verified: true,
  observations_scored: 4,
  selected_text: "nos rues !",
  auto_accepted: true,
  review_required: false,
  score_breakdown: { cross_backend: 0.82, stability: 1, language_model: null },
  agrees_with_pairwise: false,
};

describe("arbitrationReading", () => {
  it("does not misrepresent an unapplied arbitration proposal as the manifest reading", () => {
    expect(arbitrationReading(region(BASE))).toBeNull();
  });

  it("carries none of the per-signal breakdown", () => {
    const promoted = { ...BASE, agrees_with_pairwise: true, promoted: true };
    const message = arbitrationReading(region(promoted, "nos rues !"))!;
    for (const signal of ["cross_backend", "stability", "language_model",
                          "observations", "verifier", "pairwise"]) {
      expect(message).not.toContain(signal);
    }
    expect(message.split("\n")).toHaveLength(1);
  });

  it("attributes a promoted reading, which now agrees with the shown text", () => {
    // promotion sets agrees_with_pairwise true, so the old disagreement
    // condition would have gone silent on exactly the regions ToFU decided
    const promoted = { ...BASE, agrees_with_pairwise: true, promoted: true,
                       selected_text: "nos rues !" };
    expect(arbitrationReading(region(promoted, "nos rues !")))
      .toBe('ToFU read: "nos rues !"');
  });

  it("uses the final manifest text after later normalization", () => {
    const promoted = { ...BASE, agrees_with_pairwise: true, promoted: true };
    expect(arbitrationReading(region(promoted, "nos rues !")))
      .toBe('ToFU read: "nos rues !"');
  });

  it("does not attribute a later correction to an older promoted hypothesis", () => {
    const promoted = { ...BASE, agrees_with_pairwise: true, promoted: true };
    const inst = region(promoted, "湯屋下島濁河温泉");
    inst.ocr_correction = {
      applied: true,
      original_text: "周屋下島周河温泉",
      corrected_text: "湯屋下島濁河温泉",
      correction_resource: { kind: "gazetteer" },
    };
    expect(arbitrationReading(inst)).toBeNull();
  });

  it("is silent when arbitration simply agreed and changed nothing", () => {
    expect(arbitrationReading(region({ ...BASE, agrees_with_pairwise: true }))).toBeNull();
  });

  it("is silent for a region that was never arbitrated", () => {
    expect(arbitrationReading(region(null))).toBeNull();
  });

  it("is silent when arbitration produced no reading", () => {
    expect(arbitrationReading(region({ ...BASE, selected_text: "" }))).toBeNull();
  });
});

describe("groundTruthReading", () => {
  it("attributes an applied end-user Ground Truth override", () => {
    const inst = region(null, "RÉPUBLIQUE");
    inst.ocr_correction = {
      applied: true,
      original_text: "REPUBLIOUE",
      corrected_text: "RÉPUBLIQUE",
      correction_resource: { kind: "ground_truth", scope: "asset", terms: ["RÉPUBLIQUE"] },
    };
    expect(groundTruthReading(inst)).toBe('Ground Truth override: "RÉPUBLIQUE"');
  });

  it("is silent for a review-only Ground Truth proposal", () => {
    const inst = region(null);
    inst.ocr_correction = {
      applied: false,
      candidate_text: "nos rues !",
      correction_resource: { kind: "ground_truth", scope: "project" },
    };
    expect(groundTruthReading(inst)).toBeNull();
  });

  it("uses durable Ground Truth attribution after the correction scratch slot changes", () => {
    const inst = region(null, "RÉPUBLIQUE");
    inst.source_override = { kind: "ground_truth", text: "RÉPUBLIQUE", icon: "leaf", color: "emerald" };
    inst.ocr_correction = null;
    expect(groundTruthReading(inst)).toBe('Ground Truth override: "RÉPUBLIQUE"');
  });
});
