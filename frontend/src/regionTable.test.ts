// 🍢 ToFU — the arbitration hypothesis has to reach a reader
//
// Arbitration scores every observation of a region and records what it would
// have chosen. The region keeps the older pairwise reading on purpose, so the
// hypothesis is evidence rather than a decision -- but it was written and
// read by nothing at all, in the server, the client, or any test. These pin
// the one thing that makes it reviewable: a disagreement produces a message,
// and agreement produces silence.

import { describe, it, expect } from "vitest";
import { InstText } from "./api";
import { hypothesisDissent } from "./RegionTable";

function region(hypothesis: InstText["ocr_provenance"] extends null ? never : NonNullable<InstText["ocr_provenance"]>["hypothesis"]): InstText {
  return {
    id: "r2",
    bounding_box: { x: 0, y: 0, width: 10, height: 10 },
    text: "nos rues 4",
    target_text: null,
    confidence: 0.9,
    detected_language: "fr",
    reading_order: 1,
    dnt: false,
    target_language: null,
    ocr_provenance: hypothesis ? { hypothesis } : null,
  } as InstText;
}

const DISSENTING = {
  verified: true,
  observations_scored: 4,
  selected_text: "nos rues !",
  auto_accepted: true,
  review_required: false,
  score_breakdown: { cross_backend: 0.82, stability: 1, language_model: null },
  agrees_with_pairwise: false,
};

describe("hypothesisDissent", () => {
  it("reports the reading arbitration would have chosen instead", () => {
    const message = hypothesisDissent(region(DISSENTING));
    expect(message).not.toBeNull();
    // Both readings, so the disagreement can be judged without opening
    // anything: what it wanted, and what is actually shown.
    expect(message).toContain('arbitration read this as "nos rues !"');
    expect(message).toContain('shown: "nos rues 4"');
  });

  it("carries the per-signal breakdown, skipping signals that had no value", () => {
    const message = hypothesisDissent(region(DISSENTING))!;
    expect(message).toContain("cross_backend 0.82");
    expect(message).toContain("stability 1.00");
    // language_model is null whenever no n-gram artifact is installed. A
    // signal that did not weigh in must not be shown as if it scored zero.
    expect(message).not.toContain("language_model");
  });

  it("says whether the verifier corroborated the reading", () => {
    expect(hypothesisDissent(region(DISSENTING))).toContain("corroborated by the verifier");
    expect(hypothesisDissent(region({ ...DISSENTING, verified: false })))
      .toContain("not verified");
  });

  it("is silent when the hypothesis agrees, so only disagreement is marked", () => {
    expect(hypothesisDissent(region({ ...DISSENTING, agrees_with_pairwise: true }))).toBeNull();
  });

  it("is silent for a region that was never arbitrated", () => {
    // Every region detected before the hypothesis wiring, and every region
    // under an 'off' policy, arrives with no hypothesis at all.
    expect(hypothesisDissent(region(null))).toBeNull();
    expect(hypothesisDissent(region(undefined))).toBeNull();
  });
});
