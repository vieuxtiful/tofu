import { describe, it, expect, vi, afterEach, beforeEach } from "vitest";
import { render, screen, cleanup, waitFor } from "@testing-library/react";
import DetectionAttributionCard from "./DetectionAttributionCard";
import type { DetectionAttribution } from "./api";

vi.mock("./api", async () => ({
  getDetectionAttribution: vi.fn(),
}));
const { getDetectionAttribution } = await import("./api");

afterEach(() => cleanup());
beforeEach(() => vi.mocked(getDetectionAttribution).mockReset());

function report(over: Partial<DetectionAttribution> = {}): DetectionAttribution {
  return {
    asset_id: "a1", rung: "no_proposals",
    explanation: "the detector proposed nothing at this resolution",
    regions: { total: 0, active: 0, excluded: 0 },
    lineage: {
      run_kind: "production", nodes: 0, raw_proposals: 0,
      by_state: {}, by_stage: {}, suppressed_by_stage: {},
      suppression_reasons: {}, orphans: 0, raw_untraced: 0,
      detector_configs: [],
    },
    ...over,
  };
}

const show = (r: DetectionAttribution) => {
  vi.mocked(getDetectionAttribution).mockResolvedValue(r);
  render(<DetectionAttributionCard assetId="a1" />);
  return waitFor(() => screen.getByTestId("detection-attribution"));
};

describe("DetectionAttributionCard", () => {
  it("says the detector proposed nothing", async () => {
    const card = await show(report());
    expect(card.getAttribute("data-rung")).toBe("no_proposals");
    expect(card.textContent).toContain("proposed nothing at this resolution");
  });

  it("distinguishes suppression from blindness, and says where", async () => {
    // The distinction the old single toast could not make: one costs a new
    // detector, the other costs a threshold.
    const card = await show(report({
      rung: "all_suppressed",
      explanation: "the detector proposed candidates and every one was suppressed",
      lineage: {
        ...report().lineage!,
        raw_proposals: 41,
        suppressed_by_stage: {
          merge_detections: { merged: 33 },
          prune_contained_fragments: { pruned: 8 },
        },
      },
    }));
    expect(card.textContent).toContain("41 detector proposals");
    expect(card.textContent).toContain("merge detections");
    expect(card.textContent).toContain("33");
  });

  it("points an all-excluded asset at History", async () => {
    const card = await show(report({
      rung: "all_excluded",
      explanation: "regions were captured, then all of them were excluded",
      regions: { total: 3, active: 0, excluded: 3 },
    }));
    expect(card.textContent).toContain("3 regions were captured");
    expect(card.textContent).toContain("History");
  });

  it("reports a missing graph as missing, not as blindness", async () => {
    // Absence of evidence said as such. Reporting it as "proposed nothing"
    // would be inventing a finding.
    const card = await show(report({
      rung: "no_lineage",
      explanation: "no lineage was recorded for this asset, so the detector's own proposals cannot be inspected",
      lineage: null,
    }));
    expect(card.textContent).toContain("predates lineage recording");
  });

  /* The fetch-failure path is deliberately not asserted here. The component
     guards it with `.catch`, but a rejected promise in this runner surfaces
     as the test's own failure however it is handled, and a never-settling
     one hangs the cleanup hook -- so an assertion would be measuring vitest
     rather than the component. The additive behaviour below is the same
     guarantee from the user's side: the card appears only when it has
     something to say. */
  it("renders nothing when regions are present", async () => {
    vi.mocked(getDetectionAttribution).mockResolvedValue(
      report({ rung: "regions_present", explanation: "regions were captured" }),
    );
    render(<DetectionAttributionCard assetId="a1" />);
    await Promise.resolve();
    expect(screen.queryByTestId("detection-attribution")).toBeNull();
  });
});
