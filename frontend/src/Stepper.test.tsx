import { describe, it, expect, vi, beforeEach } from "vitest";
import { render, screen, fireEvent, within } from "@testing-library/react";
import Stepper, { type Step } from "./Stepper";

const baseProps = {
  current: 0 as Step,
  onStep: vi.fn(),
  canCapture: true,
  canTranslate: true,
  canRender: true,
  canVerify: true,
  theme: "light" as const,
  flagStep: null as Step | null,
};

function renderStepper(overrides: Partial<typeof baseProps> = {}) {
  const onStep = vi.fn();
  const props = { ...baseProps, onStep, ...overrides };
  const utils = render(<Stepper {...props} />);
  return { ...utils, onStep };
}

/** Find a step button by its data-step attribute. */
function stepButton(container: HTMLElement, step: Step): HTMLButtonElement {
  return container.querySelector(`button[data-step="${step}"]`)!;
}

describe("Stepper", () => {
  it("renders all 5 step labels", () => {
    const { container } = renderStepper();
    const labels = ["Upload", "Capture", "Translate", "Render", "Verify"];
    for (const label of labels) {
      const btn = Array.from(container.querySelectorAll("button")).find(
        (b) => b.textContent?.includes(label),
      );
      expect(btn).toBeTruthy();
    }
  });

  it("calls onStep when clicking an enabled step", () => {
    const { container, onStep } = renderStepper({ current: 0 });
    fireEvent.click(stepButton(container, 1));
    expect(onStep).toHaveBeenCalledWith(1);
  });

  it("disables Capture when canCapture is false", () => {
    const { container, onStep } = renderStepper({ canCapture: false });
    const btn = stepButton(container, 1);
    expect(btn.disabled).toBe(true);
    expect(btn.title).toBe("upload an asset first");
    fireEvent.click(btn);
    expect(onStep).not.toHaveBeenCalled();
  });

  it("disables Translate when canTranslate is false", () => {
    const { container, onStep } = renderStepper({ canTranslate: false });
    const btn = stepButton(container, 2);
    expect(btn.disabled).toBe(true);
    expect(btn.title).toBe("capture at least one region first");
    fireEvent.click(btn);
    expect(onStep).not.toHaveBeenCalled();
  });

  it("disables Render when canRender is false", () => {
    const { container, onStep } = renderStepper({ canRender: false });
    const btn = stepButton(container, 3);
    expect(btn.disabled).toBe(true);
    expect(btn.title).toBe("translate at least one region first");
    fireEvent.click(btn);
    expect(onStep).not.toHaveBeenCalled();
  });

  it("disables Verify when canVerify is false", () => {
    const { container, onStep } = renderStepper({ canVerify: false });
    const btn = stepButton(container, 4);
    expect(btn.disabled).toBe(true);
    expect(btn.title).toBe("complete a render first");
    fireEvent.click(btn);
    expect(onStep).not.toHaveBeenCalled();
  });

  it("allows clicking Upload (step 0) regardless of gating flags", () => {
    const { container, onStep } = renderStepper({
      canCapture: false, canTranslate: false, canRender: false, canVerify: false,
    });
    const btn = stepButton(container, 0);
    expect(btn.disabled).toBe(false);
    fireEvent.click(btn);
    expect(onStep).toHaveBeenCalledWith(0);
  });

  it("shows flag indicator on the flagged step", () => {
    const { container } = renderStepper({ current: 2, flagStep: 2 });
    const flag = container.querySelector('[title="target language changed — translations need review"]');
    expect(flag).toBeTruthy();
  });

  it("calls onStep with the current step when re-clicking it", () => {
    const { container, onStep } = renderStepper({ current: 2 });
    fireEvent.click(stepButton(container, 2));
    expect(onStep).toHaveBeenCalledWith(2);
  });
});
