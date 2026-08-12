import { describe, it, expect, afterEach } from "vitest";
import { render, screen, cleanup } from "@testing-library/react";
import GroundTruthField from "./GroundTruthField";

/** The Blocks field is locked during CAPTURE and only during capture.
 *
 * It used to lock for `scan?.status === "scanning"` too -- the
 * "(prepping...)" pre-flight sniff of the uploaded asset. That is the
 * slowest step between upload and Capture, and it is not a run against
 * these Blocks, so locking through it froze the field exactly when the user
 * had time to fill it in.
 *
 * The two states look similar (both read as "busy"), which is why this is
 * pinned: the regression would be invisible.
 */
const noop = () => {};
const assess = async () => ({
  state: "insufficient_evidence" as const, detected_language: null, source_language: null,
  scripts: [], direction: "ltr" as const, reasons: [], warns: false,
  policy_revision: "test",
});

function field(capturing: boolean) {
  return render(
    <GroundTruthField
      label="Blocks" value="SORTIE" onChange={noop} capturing={capturing}
      sourceLanguage="fr-FR" assess={assess}
    />,
  );
}

afterEach(() => cleanup());

describe("GroundTruthField capture lock", () => {
  it("is editable when no capture is running", () => {
    const { container } = field(false);
    expect(screen.getByRole("textbox").hasAttribute("readonly")).toBe(false);
    expect(container.querySelector(".gt-lock-shimmer")).toBeNull();
  });

  it("is locked, visibly, while a capture runs", () => {
    const { container } = field(true);
    expect(screen.getByRole("textbox").hasAttribute("readonly")).toBe(true);
    // The shimmer is the "working, not broken" signal -- a readOnly input
    // with no explanation reads as a disabled one.
    expect(container.querySelector(".gt-lock")).toBeTruthy();
    expect(container.querySelector(".gt-lock-shimmer")).toBeTruthy();
  });
});
