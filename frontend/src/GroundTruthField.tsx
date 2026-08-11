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
import { assessLanguage, type LanguageAssessment } from "./api";

type Props = {
  value: string;
  onChange: (value: string) => void;
  onBlur?: () => void;
  label: string;
  placeholder?: string;
  language?: string | null;
  /** The project's confirmed SOURCE language.  Entered text has to exist in
   *  the asset being localized, so source is what it must agree with --
   *  target only governs whether the control is shown at all. */
  sourceLanguage?: string | null;
  /** A capture is running: the field is read-only and says so visibly. */
  capturing?: boolean;
  /** Injectable for tests; defaults to the server assessment so there is
   *  exactly one lexicon. */
  assess?: (text: string, sourceLanguage: string | null) => Promise<LanguageAssessment>;
  /** Enter was pressed on a non-empty entry.  Guided uses this to commit
   *  one Block; Auto leaves it undefined and Enter does nothing, as before.
   *
   *  Never fires mid-composition.  On a Japanese or Chinese IME the first
   *  Enter CONFIRMS the candidate rather than ending the entry, so
   *  committing on it would truncate 東京都渋谷区 to whatever was converted
   *  so far -- and the languages most likely to need Guided are exactly the
   *  ones that compose. */
  onCommit?: (value: string) => void;
  /** Backspace at a collapsed caret in an empty entry.  The tag-input
   *  idiom: nothing left to delete here, so remove the chip behind. */
  onBackspaceEmpty?: () => void;
};

/** IME-safe native input with Basil's delimiter-settled colour mirror. */
export default function GroundTruthField({ value, onChange, onBlur, label, placeholder, language, sourceLanguage, assess, capturing, onCommit, onBackspaceEmpty }: Props) {
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
  const [assessment, setAssessment] = useState<LanguageAssessment | null>(null);
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

  // Debounced because it is a server call, and settled-only because warning
  // mid-word ("Rue" before "des Martyrs" is typed) accuses the user of a
  // mistake they are in the middle of not making.
  useEffect(() => {
    const text = value.trim();
    if (!text || !sourceLanguage) { setAssessment(null); return; }
    const run = assess ?? assessLanguage;
    let cancelled = false;
    const timer = setTimeout(() => {
      run(text, sourceLanguage)
        .then((result) => { if (!cancelled) setAssessment(result); })
        .catch(() => { if (!cancelled) setAssessment(null); });
    }, 400);
    return () => { cancelled = true; clearTimeout(timer); };
  }, [value, sourceLanguage, assess]);

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
    <div ref={fieldRef} className={`basil-plate ground-truth-field animated-caret-field${warning ? " ground-truth-language-warning" : ""}${capturing ? " gt-locked" : ""}`}>
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
        onKeyDown={(event) => {
          // `isComposing` is checked on the native event as well as our own
          // flag: Chromium fires keydown with keyCode 229 during
          // composition and Safari/Firefox differ on whether
          // compositionend lands first, so neither signal alone is
          // reliable across engines.
          const composing = composingRef.current
            || (event.nativeEvent as unknown as { isComposing?: boolean }).isComposing;
          if (composing) return;
          if (event.key === "Enter" && onCommit) {
            event.preventDefault();
            if (value.trim()) onCommit(value);
            return;
          }
          if (
            event.key === "Backspace" && onBackspaceEmpty && !value
            && event.currentTarget.selectionStart === 0
            && event.currentTarget.selectionEnd === 0
          ) {
            event.preventDefault();
            onBackspaceEmpty();
          }
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
        readOnly={capturing}
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
      {capturing && (
        /* Locked, and shown to be locked: editing Blocks mid-capture would
           mean the run no longer matches the request that started it. The
           stripes travel horizontally so the state reads as "working", not
           "disabled". */
        <div className="gt-lock" aria-hidden="true">
          <div className="gt-lock-stripes" />
        </div>
      )}
      {assessment?.warns && (
        // `dir` puts the icon at the LOGICAL trailing edge -- visually right
        // for LTR, visually left for RTL -- instead of hard-coding a side.
        <div
          dir={assessment.direction}
          role="status"
          data-testid="language-warning"
          data-state={assessment.state}
          data-direction={assessment.direction}
          className="mt-1 flex items-center gap-1 text-[11px] text-amber-600 dark:text-amber-400"
        >
          <span className="min-w-0 flex-1 truncate">
            {assessment.detected_language
              ? `"${value.trim()}" appears to be ${assessment.detected_language}, not ${assessment.source_language}.`
              : (assessment.reasons[0] ?? "This may not be the source language.")}
          </span>
          <TbLanguageOff size={12} className="shrink-0" aria-hidden="true" />
        </div>
      )}
    </div>
  );
}
