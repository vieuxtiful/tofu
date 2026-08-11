import { describe, it, expect, vi } from "vitest";
import { renderHook, act } from "@testing-library/react";
import { useSelectionSync } from "./useSelectionSync";

// Helper: create a mock element with selectionStart/selectionEnd
function mockElement(start: number | null, end: number | null): HTMLElement {
  const el = {
    selectionStart: start,
    selectionEnd: end,
    setSelectionRange: vi.fn(),
  } as unknown as HTMLElement;
  return el;
}

describe("useSelectionSync", () => {
  it("initializes with a collapsed selection at valueLength", () => {
    const inputRef = { current: null as HTMLTextAreaElement | null };
    const { result } = renderHook(() =>
      useSelectionSync(inputRef, 10, false, false),
    );
    expect(result.current[0]).toEqual({ start: 10, end: 10, collapsed: true });
  });

  it("setSelectionManually updates the selection state immediately", () => {
    const inputRef = { current: null as HTMLTextAreaElement | null };
    const { result } = renderHook(() =>
      useSelectionSync(inputRef, 10, false, false),
    );
    act(() => {
      result.current[1](2, 5);
    });
    expect(result.current[0]).toEqual({ start: 2, end: 5, collapsed: false });
  });

  it("setSelectionManually with start === end sets collapsed to true", () => {
    const inputRef = { current: null as HTMLTextAreaElement | null };
    const { result } = renderHook(() =>
      useSelectionSync(inputRef, 10, false, false),
    );
    act(() => {
      result.current[1](3, 3);
    });
    expect(result.current[0]).toEqual({ start: 3, end: 3, collapsed: true });
  });

  it("reads selection from the native element on selectionchange when focused", async () => {
    const el = mockElement(2, 6);
    const inputRef = { current: el as HTMLTextAreaElement | null };
    Object.defineProperty(document, "activeElement", {
      value: el,
      configurable: true,
    });
    const { result } = renderHook(() =>
      useSelectionSync(inputRef, 10, true, false),
    );
    // Dispatch selectionchange
    await act(async () => {
      document.dispatchEvent(new Event("selectionchange"));
      // Let rAF flush
      await new Promise<void>((r) => requestAnimationFrame(() => requestAnimationFrame(() => r())));
    });
    expect(result.current[0]).toEqual({ start: 2, end: 6, collapsed: false });
  });

  it("does not react to selectionchange when not focused", async () => {
    const el = mockElement(2, 6);
    const inputRef = { current: el as HTMLTextAreaElement | null };
    const { result } = renderHook(() =>
      useSelectionSync(inputRef, 10, false, false),
    );
    await act(async () => {
      document.dispatchEvent(new Event("selectionchange"));
      await new Promise<void>((r) => requestAnimationFrame(() => requestAnimationFrame(() => r())));
    });
    // Should remain at initial state
    expect(result.current[0]).toEqual({ start: 10, end: 10, collapsed: true });
  });

  it("does not react to selectionchange during IME composition", async () => {
    const el = mockElement(2, 6);
    const inputRef = { current: el as HTMLTextAreaElement | null };
    const { result } = renderHook(() =>
      useSelectionSync(inputRef, 10, true, true), // composing = true
    );
    await act(async () => {
      document.dispatchEvent(new Event("selectionchange"));
      await new Promise<void>((r) => requestAnimationFrame(() => requestAnimationFrame(() => r())));
    });
    expect(result.current[0]).toEqual({ start: 10, end: 10, collapsed: true });
  });

  it("does not update when the active element is not our input", async () => {
    const el = mockElement(2, 6);
    const inputRef = { current: el as HTMLTextAreaElement | null };
    // Set document.activeElement to something else
    const otherEl = mockElement(0, 0);
    Object.defineProperty(document, "activeElement", {
      value: otherEl,
      configurable: true,
    });
    const { result } = renderHook(() =>
      useSelectionSync(inputRef, 10, true, false),
    );
    await act(async () => {
      document.dispatchEvent(new Event("selectionchange"));
      await new Promise<void>((r) => requestAnimationFrame(() => requestAnimationFrame(() => r())));
    });
    expect(result.current[0]).toEqual({ start: 10, end: 10, collapsed: true });
    // Restore
    Object.defineProperty(document, "activeElement", {
      value: el,
      configurable: true,
    });
  });

  it("coalesces multiple selectionchange events into one rAF read", async () => {
    const el = mockElement(0, 0);
    const inputRef = { current: el as HTMLTextAreaElement | null };
    Object.defineProperty(document, "activeElement", {
      value: el,
      configurable: true,
    });
    const { result } = renderHook(() =>
      useSelectionSync(inputRef, 10, true, false),
    );
    // Rapidly change selection and dispatch multiple events before rAF can fire
    await act(async () => {
      (el as any).selectionStart = 1;
      (el as any).selectionEnd = 3;
      document.dispatchEvent(new Event("selectionchange"));
      (el as any).selectionStart = 1;
      (el as any).selectionEnd = 5;
      document.dispatchEvent(new Event("selectionchange"));
      (el as any).selectionStart = 1;
      (el as any).selectionEnd = 7;
      document.dispatchEvent(new Event("selectionchange"));
      await new Promise<void>((r) => requestAnimationFrame(() => requestAnimationFrame(() => r())));
    });
    // Should reflect the last value read, not intermediate ones
    expect(result.current[0]).toEqual({ start: 1, end: 7, collapsed: false });
  });
});
