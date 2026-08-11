import { describe, it, expect, vi, afterEach } from "vitest";
import { render, screen, fireEvent, cleanup, waitFor } from "@testing-library/react";
import CaptureModeSelect, { CAPTURE_MODES } from "./CaptureModeSelect";
import GroundTruthField from "./GroundTruthField";
import type { LanguageAssessment } from "./api";

afterEach(() => cleanup());

const assessment = (over: Partial<LanguageAssessment> = {}): LanguageAssessment => ({
  state: "agree", source_language: "en-US", detected_language: null,
  scripts: ["Latn"], direction: "ltr", reasons: [], warns: false,
  policy_revision: "palate-1", ...over,
});

describe("CaptureModeSelect", () => {
  it("offers Auto, Guided and Manual in that order", () => {
    render(<CaptureModeSelect value="auto" onChange={() => {}} />);
    fireEvent.click(screen.getByLabelText("Capture mode"));
    const options = screen.getAllByRole("option");
    expect(options.map((o) => o.textContent?.startsWith("Auto") ? "auto"
      : o.textContent?.startsWith("Guided") ? "guided" : "manual"))
      .toEqual(["auto", "guided", "manual"]);
  });

  it("defaults to Auto, which is what existing projects already do", () => {
    expect(CAPTURE_MODES[0].id).toBe("auto");
  });

  it("reports the chosen mode", () => {
    const onChange = vi.fn();
    render(<CaptureModeSelect value="auto" onChange={onChange} />);
    fireEvent.click(screen.getByLabelText("Capture mode"));
    fireEvent.click(screen.getAllByRole("option")[1]);
    expect(onChange).toHaveBeenCalledWith("guided");
  });

  it("marks the current mode as selected", () => {
    render(<CaptureModeSelect value="guided" onChange={() => {}} />);
    fireEvent.click(screen.getByLabelText("Capture mode"));
    const selected = screen.getAllByRole("option").filter(
      (o) => o.getAttribute("aria-selected") === "true",
    );
    expect(selected).toHaveLength(1);
    expect(selected[0].textContent).toContain("Guided");
  });

  it("uses the language-picker trigger styling, not a native select", () => {
    render(<CaptureModeSelect value="auto" onChange={() => {}} />);
    const trigger = screen.getByLabelText("Capture mode");
    expect(trigger.tagName).toBe("BUTTON");
    expect(document.querySelector("select")).toBeNull();
    for (const cls of ["rounded", "border-transparent", "text-xs"]) {
      expect(trigger.className).toContain(cls);
    }
    expect(document.querySelector(".dropdown-morph")).not.toBeNull();
  });

  it("explains each mode without naming internals", () => {
    render(<CaptureModeSelect value="manual" onChange={() => {}} />);
    expect(screen.getAllByText(/no detection/i).length).toBeGreaterThan(0);
    for (const forbidden of ["LayerMode", "HYBRID", "cicerone", "okara"]) {
      expect(document.body.textContent).not.toContain(forbidden);
    }
  });
});

describe("GroundTruthField source-language warning", () => {
  it("warns when the entry reads as another language", async () => {
    const assess = vi.fn(async () => assessment({
      state: "probable_mismatch", detected_language: "fr", warns: true,
      reasons: ["reads as fr, not en"],
    }));
    render(
      <GroundTruthField
        label="Blocks" value="Rue des Martyrs" onChange={() => {}}
        sourceLanguage="en-US" assess={assess}
      />,
    );
    const warning = await screen.findByTestId("language-warning", {}, { timeout: 2000 });
    expect(warning.getAttribute("data-state")).toBe("probable_mismatch");
    expect(warning.textContent).toContain("fr");
  });

  it("places the warning at the logical trailing edge for RTL", async () => {
    const assess = vi.fn(async () => assessment({
      state: "strong_mismatch", direction: "rtl", warns: true,
      reasons: ["written in Arab, which en does not use"],
    }));
    render(
      <GroundTruthField
        label="Blocks" value="مرحبا بالعالم" onChange={() => {}}
        sourceLanguage="en-US" assess={assess}
      />,
    );
    const warning = await screen.findByTestId("language-warning", {}, { timeout: 2000 });
    // `dir` drives the side, so the icon follows the writing direction
    // instead of being pinned to one edge.
    expect(warning.getAttribute("dir")).toBe("rtl");
    expect(warning.getAttribute("data-direction")).toBe("rtl");
  });

  it("stays silent when the entry agrees", async () => {
    const assess = vi.fn(async () => assessment());
    render(
      <GroundTruthField
        label="Blocks" value="Exit" onChange={() => {}}
        sourceLanguage="en-US" assess={assess}
      />,
    );
    await waitFor(() => expect(assess).toHaveBeenCalled(), { timeout: 2000 });
    expect(screen.queryByTestId("language-warning")).toBeNull();
  });

  it("does not accuse short entries — nothing is assessed without a source language", async () => {
    const assess = vi.fn(async () => assessment());
    render(<GroundTruthField label="Blocks" value="FRI" onChange={() => {}} assess={assess} />);
    await new Promise((r) => setTimeout(r, 600));
    expect(assess).not.toHaveBeenCalled();
    expect(screen.queryByTestId("language-warning")).toBeNull();
  });

  it("never blocks: a warning is advisory and the value is untouched", async () => {
    const assess = vi.fn(async () => assessment({
      state: "strong_mismatch", warns: true, reasons: ["written in Arab"],
    }));
    const onChange = vi.fn();
    render(
      <GroundTruthField
        label="Blocks" value="مرحبا بالعالم" onChange={onChange}
        sourceLanguage="en-US" assess={assess}
      />,
    );
    await screen.findByTestId("language-warning", {}, { timeout: 2000 });
    expect(onChange).not.toHaveBeenCalled();
    const input = document.querySelector("input") as HTMLInputElement;
    expect(input.value).toBe("مرحبا بالعالم");
    expect(input.disabled).toBe(false);
  });
});
