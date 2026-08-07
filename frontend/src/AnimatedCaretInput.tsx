import {
  useEffect, useLayoutEffect, useRef, useState,
  type CompositionEvent, type FocusEventHandler,
} from "react";
import { insertedRange } from "./AnimatedCaretTextarea";
import { useSelectionSync } from "./useSelectionSync";

type Props = {
  value: string;
  onChange: (value: string) => void;
  label: string;
  placeholder?: string;
  className?: string;
  onFocus?: FocusEventHandler<HTMLInputElement>;
  onBlur?: () => void;
  /** optional leading icon node (e.g. a Search icon) rendered inside the field */
  leadingIcon?: React.ReactNode;
  /** typewriter placeholder text — when set, the placeholder populates
   *  per-character with a blinking block cursor while the field is empty,
   *  matching the basil-placeholder-typewriter pattern. */
  typewriterPlaceholder?: string;
};

type Reveal = { start: number; end: number; key: number } | null;

/** Single-line animated-caret input — the same transparent-ink + mirror +
 *  motion caret + insertion reveal + selection highlight pattern used by
 *  AnimatedCaretTextarea (Translate) and GroundTruthField (Capture), but for a
 *  one-line `<input>`.  The native control keeps selection, undo and IME;
 *  only its ink is transparent.  A synchronized mirror overlay renders the
 *  visible text, a blinking bar caret tracks the real caret position, and
 *  inserted text reveals with the animated-caret-reveal animation. */
export default function AnimatedCaretInput({
  value, onChange, label, placeholder, className = "", onFocus, onBlur,
  leadingIcon, typewriterPlaceholder,
}: Props) {
  const fieldRef = useRef<HTMLDivElement>(null);
  const mirrorRef = useRef<HTMLDivElement>(null);
  const measureRef = useRef<HTMLDivElement>(null);
  const motionCaretMarkerRef = useRef<HTMLSpanElement>(null);
  const inputRef = useRef<HTMLInputElement>(null);
  const previousRef = useRef(value);
  const composingRef = useRef(false);
  const compositionBaseRef = useRef(value);
  const timerRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const sequenceRef = useRef(0);
  const [focused, setFocused] = useState(false);
  const [reveal, setReveal] = useState<Reveal>(null);
  const [selection, setSelectionManually] = useSelectionSync(inputRef, value.length, focused, composingRef.current);
  const selectionCollapsed = selection.collapsed;
  const selectionRange = selection;
  const [selectionLeaving, setSelectionLeaving] = useState<{ start: number; end: number } | null>(null);
  const prevCollapsedRef = useRef(true);
  const lastSelectionRef = useRef<{ start: number; end: number }>({ start: value.length, end: value.length });
  const [caretPosition, setCaretPosition] = useState(value.length);
  const [motionCaret, setMotionCaret] = useState({ left: 0, top: 0, height: 16 });
  const [typedPlaceholder, setTypedPlaceholder] = useState(0);

  // typewriter placeholder — per-character reveal with blinking block cursor
  // while the field is empty (same pattern as SemanticSubstitutionPanel).
  useEffect(() => {
    if (!typewriterPlaceholder) return;
    if (value) { setTypedPlaceholder(0); return; }
    const timers: ReturnType<typeof setTimeout>[] = [];
    for (let i = 1; i <= typewriterPlaceholder.length; i++) {
      timers.push(setTimeout(() => setTypedPlaceholder(i), i * 70));
    }
    return () => timers.forEach(clearTimeout);
  }, [value, typewriterPlaceholder]);

  useEffect(() => {
    if (!composingRef.current) previousRef.current = value;
  }, [value]);

  useEffect(() => () => {
    if (timerRef.current) clearTimeout(timerRef.current);
  }, []);

  // selection-leaving: when the selection collapses, keep the highlight span
  // mounted briefly with a `.leaving` class so the CSS transition can fade it
  // out smoothly instead of vanishing instantly.
  useEffect(() => {
    if (!selectionCollapsed) {
      lastSelectionRef.current = { start: selectionRange.start, end: selectionRange.end };
    }
    const wasCollapsed = prevCollapsedRef.current;
    prevCollapsedRef.current = selectionCollapsed;
    if (wasCollapsed && !selectionCollapsed) {
      setSelectionLeaving(null);
    } else if (!wasCollapsed && selectionCollapsed) {
      setSelectionLeaving({ ...lastSelectionRef.current });
      const t = setTimeout(() => setSelectionLeaving(null), 140);
      return () => clearTimeout(t);
    }
  }, [selectionCollapsed, selectionRange.start, selectionRange.end]);

  // motion caret positioning — measure the invisible marker span in the
  // measure layer and translate it to field-relative coordinates.  The bar
  // caret is an absolutely-positioned 2px-wide span that tracks the real
  // caret position with a CSS transition for smooth movement.
  useLayoutEffect(() => {
    const field = fieldRef.current;
    const marker = motionCaretMarkerRef.current;
    if (!field || !marker) return;
    const fieldRect = field.getBoundingClientRect();
    const markerRect = marker.getBoundingClientRect();
    setMotionCaret({
      left: markerRect.left - fieldRect.left - field.clientLeft,
      top: markerRect.top - fieldRect.top - field.clientTop,
      height: markerRect.height || 16,
    });
  }, [value, caretPosition]);

  const showInsertion = (previous: string, next: string) => {
    const range = insertedRange(previous, next);
    if (!range) { setReveal(null); return; }
    setReveal({ ...range, key: ++sequenceRef.current });
    if (timerRef.current) clearTimeout(timerRef.current);
    timerRef.current = setTimeout(() => setReveal(null), 160);
  };

  const start = reveal?.start ?? value.length;
  const end = reveal?.end ?? value.length;

  return (
    <div ref={fieldRef} className={`animated-caret-field ${leadingIcon ? "has-leading-icon" : ""} ${className}`}>
      {leadingIcon && <span className="animated-caret-leading-icon">{leadingIcon}</span>}
      <div className="animated-caret-mirror" aria-hidden="true" ref={mirrorRef}>
        {!value && !typewriterPlaceholder && <span className="animated-caret-placeholder">{placeholder}</span>}
        {!value && typewriterPlaceholder && (
          <span className="basil-placeholder-typewriter" style={{ color: "#a1a1aa" }}>
            {typewriterPlaceholder.slice(0, typedPlaceholder)}
            <span className="basil-placeholder-cursor" />
          </span>
        )}
        {!selectionCollapsed ? <>
          {value.slice(0, selectionRange.start)}
          <span key="selection" className="animated-caret-selection">{value.slice(selectionRange.start, selectionRange.end)}</span>
          {value.slice(selectionRange.end)}
        </> : selectionLeaving ? <>
          {value.slice(0, selectionLeaving.start)}
          <span key="selection-leaving" className="animated-caret-selection leaving">{value.slice(selectionLeaving.start, selectionLeaving.end)}</span>
          {value.slice(selectionLeaving.end)}
        </> : <>
          {value.slice(0, start)}
          {reveal && <span key={reveal.key} className="animated-caret-insert">{value.slice(start, end)}</span>}
          {value.slice(end)}
        </>}
      </div>
      <div className="animated-caret-measure" aria-hidden="true" ref={measureRef}>
        {value.slice(0, caretPosition)}<span ref={motionCaretMarkerRef} className="animated-motion-caret-marker" />{value.slice(caretPosition)}
      </div>
      {focused && selectionCollapsed && (
        <span className="animated-motion-caret" aria-hidden="true" style={motionCaret} />
      )}
      <input
        ref={inputRef}
        aria-label={label}
        inputMode="text"
        autoCapitalize="none"
        value={value}
        onFocus={(event) => { setFocused(true); onFocus?.(event); }}
        onBlur={() => { setFocused(false); onBlur?.(); }}
        onCompositionStart={() => {
          composingRef.current = true;
          compositionBaseRef.current = previousRef.current;
          setReveal(null);
        }}
        onCompositionEnd={(event: CompositionEvent<HTMLInputElement>) => {
          composingRef.current = false;
          const next = event.currentTarget.value;
          showInsertion(compositionBaseRef.current, next);
          previousRef.current = next;
          setCaretPosition(event.currentTarget.selectionStart ?? next.length);
        }}
        onChange={(event) => {
          const next = event.target.value;
          if (!composingRef.current) showInsertion(previousRef.current, next);
          previousRef.current = next;
          setCaretPosition(event.currentTarget.selectionStart ?? next.length);
          onChange(next);
        }}
        onClick={(event) => setCaretPosition(event.currentTarget.selectionStart ?? value.length)}
        onKeyUp={(event) => setCaretPosition(event.currentTarget.selectionStart ?? value.length)}
        onScroll={(event) => {
          if (mirrorRef.current) mirrorRef.current.scrollLeft = event.currentTarget.scrollLeft;
          if (measureRef.current) measureRef.current.scrollLeft = event.currentTarget.scrollLeft;
        }}
        className="animated-caret-input"
        spellCheck={false}
        autoComplete="off"
      />
    </div>
  );
}
