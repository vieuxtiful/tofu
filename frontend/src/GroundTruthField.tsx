import {
  useEffect, useLayoutEffect, useMemo, useRef, useState,
  type CompositionEvent, type CSSProperties, type ReactNode,
} from "react";
import { createPortal } from "react-dom";
import { TbLanguageOff } from "react-icons/tb";
import { regionColorMap } from "./regionPalette";
import { insertedRange } from "./AnimatedCaretTextarea";
import { useSelectionSync } from "./useSelectionSync";
import { textLangMatchesTarget } from "./detectLanguage";

type Props = {
  value: string;
  onChange: (value: string) => void;
  onBlur?: () => void;
  label: string;
  placeholder?: string;
  language?: string | null;
};

/** IME-safe native input with Basil's delimiter-settled colour mirror. */
export default function GroundTruthField({ value, onChange, onBlur, label, placeholder, language }: Props) {
  const mirrorRef = useRef<HTMLDivElement>(null);
  const fieldRef = useRef<HTMLDivElement>(null);
  const measureRef = useRef<HTMLDivElement>(null);
  const markerRef = useRef<HTMLSpanElement>(null);
  const inputRef = useRef<HTMLInputElement>(null);
  const previousRef = useRef(value);
  const composingRef = useRef(false);
  const compositionBaseRef = useRef(value);
  const timerRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const sequenceRef = useRef(0);
  const internalChangeRef = useRef(false);
  const [focused, setFocused] = useState(false);
  const [reveal, setReveal] = useState<{ start: number; end: number; key: number } | null>(null);
  const [selection, setSelectionManually] = useSelectionSync(inputRef, value.length, focused, composingRef.current);
  const selectionRange = selection;
  const [selectionLeaving, setSelectionLeaving] = useState<{ start: number; end: number } | null>(null);
  const prevCollapsedRef = useRef(true);
  const lastSelectionRef = useRef<{ start: number; end: number }>({ start: value.length, end: value.length });
  const [caretPosition, setCaretPosition] = useState(value.length);
  const [warning, setWarning] = useState<{ pulse: number } | null>(null);
  const [warningRect, setWarningRect] = useState<DOMRect | null>(null);
  const pieces = useMemo(() => value.split(/(\s+)/), [value]);
  const words = useMemo(() => pieces.filter((piece) => piece && !/^\s+$/.test(piece)), [pieces]);
  const ids = useMemo(() => words.map((_, index) => `gt${index}`), [words]);
  const colors = useMemo(() => regionColorMap(ids), [ids]);
  let wordIndex = -1;

  useEffect(() => {
    if (!composingRef.current) previousRef.current = value;
  }, [value]);

  // when the value changes externally (e.g. loaded from the DB with a trailing
  // space), snap the caret/selection to the end so the warning tooltip marker
  // and any reveal animation align with the padded cursor position.
  useEffect(() => {
    if (internalChangeRef.current) { internalChangeRef.current = false; return; }
    setCaretPosition(value.length);
    setSelectionManually(value.length, value.length);
  }, [value, setSelectionManually]);

  useEffect(() => () => {
    if (timerRef.current) clearTimeout(timerRef.current);
  }, []);

  // selection-leaving: when the selection collapses, keep the highlight span
  // mounted briefly with a `.leaving` class so the CSS transition can fade it
  // out smoothly instead of vanishing instantly. We capture the last
  // non-collapsed range in a ref because by the time the effect runs, the
  // selection state has already been updated to the collapsed position.
  useEffect(() => {
    if (!selection.collapsed) {
      lastSelectionRef.current = { start: selection.start, end: selection.end };
    }
    const wasCollapsed = prevCollapsedRef.current;
    prevCollapsedRef.current = selection.collapsed;
    if (wasCollapsed && !selection.collapsed) {
      setSelectionLeaving(null);
    } else if (!wasCollapsed && selection.collapsed) {
      setSelectionLeaving({ ...lastSelectionRef.current });
      const t = setTimeout(() => setSelectionLeaving(null), 140);
      return () => clearTimeout(t);
    }
  }, [selection.collapsed, selection.start, selection.end]);

  useLayoutEffect(() => {
    setWarningRect(warning ? markerRef.current?.getBoundingClientRect() ?? null : null);
  }, [warning, value, caretPosition]);

  const validateLanguage = (next: string) => {
    if (next.trim() && language && !textLangMatchesTarget(next, language)) {
      setWarning((current) => ({ pulse: (current?.pulse ?? 0) + 1 }));
    } else {
      setWarning(null);
    }
  };

  const showInsertion = (previous: string, next: string) => {
    const range = insertedRange(previous, next);
    if (!range) { setReveal(null); return; }
    setReveal({ ...range, key: ++sequenceRef.current });
    if (timerRef.current) clearTimeout(timerRef.current);
    timerRef.current = setTimeout(() => setReveal(null), 260);
  };

  const animatedPiece = (piece: string, pieceStart: number): ReactNode => {
    const selectionStart = Math.max(0, selectionRange.start - pieceStart);
    const selectionEnd = Math.min(piece.length, selectionRange.end - pieceStart);
    if (selectionStart < selectionEnd) {
      return <>{piece.slice(0, selectionStart)}<span key="selection" className="animated-caret-selection">{piece.slice(selectionStart, selectionEnd)}</span>{piece.slice(selectionEnd)}</>;
    }
    if (selectionLeaving) {
      const leaveStart = Math.max(0, selectionLeaving.start - pieceStart);
      const leaveEnd = Math.min(piece.length, selectionLeaving.end - pieceStart);
      if (leaveStart < leaveEnd) {
        return <>{piece.slice(0, leaveStart)}<span key="selection-leaving" className="animated-caret-selection leaving">{piece.slice(leaveStart, leaveEnd)}</span>{piece.slice(leaveEnd)}</>;
      }
    }
    if (!reveal) return piece;
    const localStart = Math.max(0, reveal.start - pieceStart);
    const localEnd = Math.min(piece.length, reveal.end - pieceStart);
    if (localStart >= localEnd) return piece;
    return <>{piece.slice(0, localStart)}<span key={reveal.key} className="animated-caret-insert">{piece.slice(localStart, localEnd)}</span>{piece.slice(localEnd)}</>;
  };

  let pieceOffset = 0;

  return (
    <div ref={fieldRef} className={`basil-plate ground-truth-field animated-caret-field${warning ? " ground-truth-language-warning" : ""}`}>
      <div className="basil-plate-mirror" aria-hidden="true" ref={mirrorRef}>
        {!value && <span className="font-mono text-zinc-400 dark:text-zinc-600">{placeholder}</span>}
        {pieces.map((piece, index) => {
          if (!piece) return null;
          const start = pieceOffset;
          pieceOffset += piece.length;
          if (/^\s+$/.test(piece)) return <span key={index}>{animatedPiece(piece, start)}</span>;
          wordIndex += 1;
          return (
            <span
              key={index}
              className="basil-token mapped"
              style={{ "--bh-ink": colors[`gt${wordIndex}`] } as CSSProperties}
            >
              {animatedPiece(piece, start)}
            </span>
          );
        })}
      </div>
      <div className="ground-truth-caret-measure" aria-hidden="true" ref={measureRef}>
        {value.slice(0, caretPosition)}<span ref={markerRef} className="animated-caret-status-anchor" />{value.slice(caretPosition)}
      </div>
      <input
        ref={inputRef}
        aria-label={label}
        lang={language ?? undefined}
        inputMode="text"
        value={value}
        onFocus={() => setFocused(true)}
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
          internalChangeRef.current = true;
          setCaretPosition(event.currentTarget.selectionStart ?? next.length);
          validateLanguage(next);
        }}
        onChange={(event) => {
          const next = event.target.value;
          if (!composingRef.current) showInsertion(previousRef.current, next);
          previousRef.current = next;
          internalChangeRef.current = true;
          setCaretPosition(event.currentTarget.selectionStart ?? next.length);
          onChange(next);
          if (!composingRef.current && !(event.nativeEvent as InputEvent).isComposing) validateLanguage(next);
        }}
        onBlur={() => { setFocused(false); setWarning(null); onBlur?.(); }}
        onScroll={(event) => {
          if (mirrorRef.current) mirrorRef.current.scrollLeft = event.currentTarget.scrollLeft;
          if (measureRef.current) measureRef.current.scrollLeft = event.currentTarget.scrollLeft;
          if (warning) setWarningRect(markerRef.current?.getBoundingClientRect() ?? null);
        }}
        className="basil-plate-input"
        spellCheck={false}
        autoComplete="off"
      />
      {warning && <TbLanguageOff size={13} className="ground-truth-language-icon text-red-500 dark:text-red-400" aria-label="Ground Truth contains text in the wrong language" />}
      {warning && warningRect && createPortal(
        <div
          key={warning.pulse}
          role="status"
          className="lang-reject-tooltip lang-reject-tooltip-pulse pointer-events-none fixed z-500 whitespace-nowrap rounded-lg border border-red-300 bg-white px-3 py-2 text-xs text-red-700 shadow-lg dark:border-red-800 dark:bg-zinc-900 dark:text-red-300"
          style={{ left: warningRect.left, top: warningRect.top - 38 }}
        >incorrect source language</div>, document.body,
      )}
    </div>
  );
}
