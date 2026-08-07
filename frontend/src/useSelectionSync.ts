import { useCallback, useEffect, useRef, useState } from "react";

export type SelectionState = {
  start: number;
  end: number;
  collapsed: boolean;
};

/**
 * Tracks the native selection of a textarea/input in real time.
 *
 * Uses a document-level `selectionchange` listener (fires on every caret move,
 * including during drag) coalesced through `requestAnimationFrame` to avoid
 * excessive React renders. The native control remains authoritative for
 * selection, accessibility, clipboard, undo, and IME — this hook only mirrors
 * the selection range into React state for the visual mirror layer.
 *
 * @param inputRef     ref to the native <textarea> or <input>
 * @param valueLength  current value length (used as fallback when the element
 *                     has no selectionStart, e.g. before mount)
 * @param focused      whether the field is currently focused (the listener is
 *                     only active while focused, to avoid cross-field noise)
 * @param composing    whether an IME composition is in progress (selection
 *                     tracking is paused during composition so the candidate
 *                     window stays authoritative)
 */
export function useSelectionSync(
  inputRef: React.RefObject<HTMLTextAreaElement | HTMLInputElement | null>,
  valueLength: number,
  focused: boolean,
  composing: boolean,
): readonly [SelectionState, (start: number, end: number) => void] {
  const [selection, setSelection] = useState<SelectionState>({
    start: valueLength,
    end: valueLength,
    collapsed: true,
  });
  const rafRef = useRef<number | null>(null);

  const readSelection = useCallback(() => {
    const el = inputRef.current;
    if (!el) return;
    const start = el.selectionStart ?? valueLength;
    const end = el.selectionEnd ?? valueLength;
    setSelection((prev) =>
      prev.start === start && prev.end === end && prev.collapsed === (start === end)
        ? prev
        : { start, end, collapsed: start === end },
    );
  }, [inputRef, valueLength]);

  const scheduleRead = useCallback(() => {
    if (rafRef.current !== null) return; // already scheduled for this frame
    rafRef.current = requestAnimationFrame(() => {
      rafRef.current = null;
      readSelection();
    });
  }, [readSelection]);

  // selectionchange listener — only while focused and not composing
  useEffect(() => {
    if (!focused || composing) return;
    const handler = () => {
      // Only react if the active element is our input
      if (document.activeElement !== inputRef.current) return;
      scheduleRead();
    };
    document.addEventListener("selectionchange", handler);
    return () => document.removeEventListener("selectionchange", handler);
  }, [focused, composing, inputRef, scheduleRead]);

  // Clean up any pending rAF on unmount
  useEffect(() => () => {
    if (rafRef.current !== null) cancelAnimationFrame(rafRef.current);
  }, []);

  // Manual setter for programmatic selection (e.g. acceptSuggestion calls
  // setSelectionRange then needs the mirror to sync immediately, without
  // waiting for the next selectionchange event).
  const setSelectionManually = useCallback((start: number, end: number) => {
    if (rafRef.current !== null) {
      cancelAnimationFrame(rafRef.current);
      rafRef.current = null;
    }
    setSelection({ start, end, collapsed: start === end });
  }, []);

  return [selection, setSelectionManually] as const;
}
