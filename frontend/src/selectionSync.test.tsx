import { describe, it, expect, vi, afterEach } from "vitest";
import { render, screen, cleanup, act } from "@testing-library/react";
import AnimatedCaretTextarea from "./AnimatedCaretTextarea";

// Helper: set native selection on a textarea/input and dispatch selectionchange
function setNativeSelection(
  el: HTMLTextAreaElement | HTMLInputElement,
  start: number,
  end: number,
) {
  el.setSelectionRange(start, end);
  document.dispatchEvent(new Event("selectionchange"));
}

// Helper: flush pending requestAnimationFrame callbacks.
// happy-dom's rAF is async; we need to let microtasks + timers settle.
function flushRaf() {
  return new Promise<void>((resolve) => {
    requestAnimationFrame(() => {
      requestAnimationFrame(() => resolve());
    });
  });
}

// Focus the field and wait for React to process the onFocus state update so
// the selectionchange listener is attached before we dispatch events.
async function focusField(el: HTMLElement) {
  await act(async () => {
    el.focus();
  });
}

afterEach(() => cleanup());

describe("AnimatedCaretTextarea selection tracking", () => {
  it("renders the selection highlight when a non-collapsed selection is set", async () => {
    const onChange = vi.fn();
    render(
      <AnimatedCaretTextarea
        label="test-field"
        value="hello world"
        onChange={onChange}
      />,
    );
    const textarea = screen.getByLabelText("test-field") as HTMLTextAreaElement;
    await focusField(textarea);
    await act(async () => {
      setNativeSelection(textarea, 0, 5);
    });
    await flushRaf();
    const selectionSpan = document.querySelector(".animated-caret-selection");
    expect(selectionSpan).not.toBeNull();
    expect(selectionSpan?.textContent).toBe("hello");
  });

  it("updates the selection range as the caret moves during drag (forward)", async () => {
    const onChange = vi.fn();
    render(
      <AnimatedCaretTextarea
        label="test-field"
        value="abcdef"
        onChange={onChange}
      />,
    );
    const textarea = screen.getByLabelText("test-field") as HTMLTextAreaElement;
    await focusField(textarea);
    // Simulate drag: selection grows from 0-2 to 0-4
    await act(async () => { setNativeSelection(textarea, 0, 2); });
    await flushRaf();
    expect(document.querySelector(".animated-caret-selection")?.textContent).toBe("ab");
    await act(async () => { setNativeSelection(textarea, 0, 4); });
    await flushRaf();
    expect(document.querySelector(".animated-caret-selection")?.textContent).toBe("abcd");
  });

  it("updates the selection range for reverse drag", async () => {
    const onChange = vi.fn();
    render(
      <AnimatedCaretTextarea
        label="test-field"
        value="abcdef"
        onChange={onChange}
      />,
    );
    const textarea = screen.getByLabelText("test-field") as HTMLTextAreaElement;
    await focusField(textarea);
    await act(async () => { setNativeSelection(textarea, 4, 6); });
    await flushRaf();
    expect(document.querySelector(".animated-caret-selection")?.textContent).toBe("ef");
    // Reverse drag: selectionStart moves back
    await act(async () => { setNativeSelection(textarea, 2, 6); });
    await flushRaf();
    expect(document.querySelector(".animated-caret-selection")?.textContent).toBe("cdef");
  });

  it("handles drag contraction", async () => {
    const onChange = vi.fn();
    render(
      <AnimatedCaretTextarea
        label="test-field"
        value="abcdef"
        onChange={onChange}
      />,
    );
    const textarea = screen.getByLabelText("test-field") as HTMLTextAreaElement;
    await focusField(textarea);
    await act(async () => { setNativeSelection(textarea, 0, 5); });
    await flushRaf();
    expect(document.querySelector(".animated-caret-selection")?.textContent).toBe("abcde");
    // Contract: drag back to 3
    await act(async () => { setNativeSelection(textarea, 0, 3); });
    await flushRaf();
    expect(document.querySelector(".animated-caret-selection")?.textContent).toBe("abc");
  });

  it("shows a leaving span when selection collapses", async () => {
    const onChange = vi.fn();
    render(
      <AnimatedCaretTextarea
        label="test-field"
        value="abcdef"
        onChange={onChange}
      />,
    );
    const textarea = screen.getByLabelText("test-field") as HTMLTextAreaElement;
    await focusField(textarea);
    await act(async () => { setNativeSelection(textarea, 0, 3); });
    await flushRaf();
    expect(document.querySelector(".animated-caret-selection:not(.leaving)")).not.toBeNull();
    // Collapse selection
    await act(async () => { setNativeSelection(textarea, 2, 2); });
    await flushRaf();
    // Leaving span should appear briefly
    const leavingSpan = document.querySelector(".animated-caret-selection.leaving");
    expect(leavingSpan).not.toBeNull();
    expect(leavingSpan?.textContent).toBe("abc");
  });

  it("does not call onChange when selection changes", async () => {
    const onChange = vi.fn();
    render(
      <AnimatedCaretTextarea
        label="test-field"
        value="abcdef"
        onChange={onChange}
      />,
    );
    const textarea = screen.getByLabelText("test-field") as HTMLTextAreaElement;
    await focusField(textarea);
    await act(async () => { setNativeSelection(textarea, 0, 3); });
    await flushRaf();
    await act(async () => { setNativeSelection(textarea, 0, 5); });
    await flushRaf();
    expect(onChange).not.toHaveBeenCalled();
  });

  it("handles multiline selection", async () => {
    const onChange = vi.fn();
    render(
      <AnimatedCaretTextarea
        label="test-field"
        value={"line1\nline2"}
        onChange={onChange}
      />,
    );
    const textarea = screen.getByLabelText("test-field") as HTMLTextAreaElement;
    await focusField(textarea);
    // Select across the newline
    await act(async () => { setNativeSelection(textarea, 3, 9); });
    await flushRaf();
    expect(document.querySelector(".animated-caret-selection")?.textContent).toBe("e1\nlin");
  });

  it("keeps a stable selection span (no remount) as range changes", async () => {
    const onChange = vi.fn();
    render(
      <AnimatedCaretTextarea
        label="test-field"
        value="abcdef"
        onChange={onChange}
      />,
    );
    const textarea = screen.getByLabelText("test-field") as HTMLTextAreaElement;
    await focusField(textarea);
    await act(async () => { setNativeSelection(textarea, 0, 2); });
    await flushRaf();
    const span1 = document.querySelector(".animated-caret-selection:not(.leaving)");
    expect(span1).not.toBeNull();
    await act(async () => { setNativeSelection(textarea, 0, 4); });
    await flushRaf();
    const span2 = document.querySelector(".animated-caret-selection:not(.leaving)");
    expect(span2).not.toBeNull();
    // The element should be the same DOM node (not remounted) — same identity
    expect(span2).toBe(span1);
  });
});
