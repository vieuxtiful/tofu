import {
  useEffect, useLayoutEffect, useRef, useState,
  type CompositionEvent, type FocusEventHandler, type KeyboardEvent,
  type MouseEventHandler, type ReactNode,
} from "react";
import { createPortal } from "react-dom";
import { ChevronLeft, ChevronRight } from "lucide-react";
import { TbChevronsDown } from "react-icons/tb";
import { useSelectionSync } from "./useSelectionSync";

type Props = {
  value: string;
  onChange: (value: string, meta?: { isComposing: boolean }) => void;
  label: string;
  placeholder?: string;
  rows?: number;
  className?: string;
  onFocus?: FocusEventHandler<HTMLTextAreaElement>;
  onBlur?: FocusEventHandler<HTMLTextAreaElement>;
  onClick?: MouseEventHandler<HTMLTextAreaElement>;
  suggestions?: string[];
  /** BCP-47 hint for the browser/OS IME. It does not attempt to switch the
   * user's active Windows input method. */
  language?: string | null;
  trailingIcon?: ReactNode;
  /** Display-only: the text stays selectable and copyable, and the caret
   *  animation and suggestion popover stay out of the way. `readOnly`
   *  rather than `disabled` on purpose — a disabled field cannot be focused
   *  or its contents copied, and reading the source is the whole reason it
   *  is on screen in Translate. */
  readOnly?: boolean;
  statusMessage?: string | null;
  statusPulse?: number;
  onCompositionStateChange?: (composing: boolean, value: string) => void;
  expandable?: boolean;
};

type Reveal = { start: number; end: number; key: number } | null;

export function isRecommendationActivationKey(key: string, ctrlKey = false, altKey = false): boolean {
  return (ctrlKey && key === " ") || (altKey && key === "ArrowDown");
}

export function insertedRange(previous: string, next: string): { start: number; end: number } | null {
  if (next.length <= previous.length) return null;
  let start = 0;
  while (start < previous.length && start < next.length && previous[start] === next[start]) start += 1;
  let suffix = 0;
  while (
    suffix < previous.length - start && suffix < next.length - start
    && previous[previous.length - 1 - suffix] === next[next.length - 1 - suffix]
  ) suffix += 1;
  const end = next.length - suffix;
  return end > start ? { start, end } : null;
}

/** Native textarea interaction over a synchronized animated text mirror.
 * Windows/browser IME composition stays authoritative. The ToFU popup is a
 * post-composition Ground Truth recommendation surface, never an OS candidate
 * window replacement. */
export default function AnimatedCaretTextarea({
  value, onChange, label, placeholder, rows = 1, className = "", onFocus, onBlur, onClick,
  suggestions = [], language, trailingIcon, statusMessage, statusPulse = 0, onCompositionStateChange, expandable = false,
  readOnly = false,
}: Props) {
  const mirrorRef = useRef<HTMLDivElement>(null);
  const fieldRef = useRef<HTMLDivElement>(null);
  const caretMeasureRef = useRef<HTMLDivElement>(null);
  const motionCaretMarkerRef = useRef<HTMLSpanElement>(null);
  const inputRef = useRef<HTMLTextAreaElement>(null);
  const previousRef = useRef(value);
  const composingRef = useRef(false);
  const compositionBaseRef = useRef(value);
  const timerRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const sequenceRef = useRef(0);
  const [reveal, setReveal] = useState<Reveal>(null);
  const [suggestionOpen, setSuggestionOpen] = useState(false);
  const [suggestionKeyboardActive, setSuggestionKeyboardActive] = useState(false);
  const [suggestionIndex, setSuggestionIndex] = useState(0);
  const [suggestionPage, setSuggestionPage] = useState(0);
  const [menuRect, setMenuRect] = useState<DOMRect | null>(null);
  const caretMarkerRef = useRef<HTMLSpanElement>(null);
  const [caretPosition, setCaretPosition] = useState(value.length);
  const [statusRect, setStatusRect] = useState<DOMRect | null>(null);
  const [focused, setFocused] = useState(false);
  const [selection, setSelectionManually] = useSelectionSync(inputRef, value.length, focused, composingRef.current);
  const selectionCollapsed = selection.collapsed;
  const selectionRange = selection;
  const [selectionLeaving, setSelectionLeaving] = useState<{ start: number; end: number } | null>(null);
  const prevCollapsedRef = useRef(true);
  const lastSelectionRef = useRef<{ start: number; end: number }>({ start: value.length, end: value.length });
  const [motionCaret, setMotionCaret] = useState({ left: 0, top: 0, height: 16 });
  const [expanded, setExpanded] = useState(false);

  const caret = inputRef.current?.selectionStart ?? value.length;
  const tokenStart = Math.max(value.lastIndexOf(" ", caret - 1), value.lastIndexOf("\n", caret - 1), value.lastIndexOf("\t", caret - 1)) + 1;
  const token = value.slice(tokenStart, caret);
  const allMatches = suggestions
    .filter((term, index, all) => term && all.indexOf(term) === index)
    .filter((term) => !token || term.includes(token));
  const suggestionPageSize = 9;
  const suggestionPageCount = Math.max(1, Math.ceil(allMatches.length / suggestionPageSize));
  const currentSuggestionPage = Math.min(suggestionPage, suggestionPageCount - 1);
  const matches = allMatches.slice(
    currentSuggestionPage * suggestionPageSize,
    (currentSuggestionPage + 1) * suggestionPageSize,
  );

  useEffect(() => {
    if (!composingRef.current) previousRef.current = value;
  }, [value]);

  useEffect(() => () => {
    if (timerRef.current) clearTimeout(timerRef.current);
  }, []);

  // selection-leaving: when the selection collapses, keep the highlight span
  // mounted briefly with a `.leaving` class so the CSS transition can fade it
  // out smoothly instead of vanishing instantly. We capture the last
  // non-collapsed range in a ref because by the time the effect runs, the
  // selection state has already been updated to the collapsed position.
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

  useEffect(() => {
    if (!suggestionOpen || !inputRef.current) return;
    const update = () => setMenuRect(inputRef.current?.getBoundingClientRect() ?? null);
    update();
    window.addEventListener("resize", update);
    window.addEventListener("scroll", update, true);
    return () => {
      window.removeEventListener("resize", update);
      window.removeEventListener("scroll", update, true);
    };
  }, [suggestionOpen]);

  useLayoutEffect(() => {
    setStatusRect(statusMessage ? caretMarkerRef.current?.getBoundingClientRect() ?? null : null);
  }, [statusMessage, value, caretPosition]);

  useLayoutEffect(() => {
    const field = fieldRef.current;
    const marker = motionCaretMarkerRef.current;
    if (!field || !marker) return;
    const fieldRect = field.getBoundingClientRect();
    const markerRect = marker.getBoundingClientRect();
    setMotionCaret({
      // Absolute children are positioned from the field's padding edge;
      // DOMRects include its border. Remove that border once so the visual
      // bar lands on the glyph position measured by the mirror.
      left: markerRect.left - fieldRect.left - field.clientLeft,
      top: markerRect.top - fieldRect.top - field.clientTop,
      height: markerRect.height || 16,
    });
  }, [value, caretPosition, rows]);

  const showInsertion = (previous: string, next: string) => {
    const range = insertedRange(previous, next);
    if (!range) { setReveal(null); return; }
    const current = { ...range, key: ++sequenceRef.current };
    setReveal(current);
    if (timerRef.current) clearTimeout(timerRef.current);
    timerRef.current = setTimeout(() => setReveal(null), 160);
  };

  const handleCompositionStart = () => {
    composingRef.current = true;
    compositionBaseRef.current = previousRef.current;
    setReveal(null);
    setSuggestionOpen(false);
    setSuggestionKeyboardActive(false);
    onCompositionStateChange?.(true, previousRef.current);
  };

  const handleCompositionEnd = (event: CompositionEvent<HTMLTextAreaElement>) => {
    composingRef.current = false;
    const next = event.currentTarget.value;
    showInsertion(compositionBaseRef.current, next);
    previousRef.current = next;
    onCompositionStateChange?.(false, next);
    if (suggestions.length) {
      setSuggestionIndex(0);
      setSuggestionPage(0);
      setSuggestionOpen(true);
      setSuggestionKeyboardActive(false);
    }
  };

  const acceptSuggestion = (term: string) => {
    const field = inputRef.current;
    const selectionStart = field?.selectionStart ?? value.length;
    const selectionEnd = field?.selectionEnd ?? selectionStart;
    const start = Math.max(value.lastIndexOf(" ", selectionStart - 1), value.lastIndexOf("\n", selectionStart - 1), value.lastIndexOf("\t", selectionStart - 1)) + 1;
    const next = value.slice(0, start) + term + value.slice(selectionEnd);
    showInsertion(value, next);
    previousRef.current = next;
    onChange(next);
    setSuggestionOpen(false);
    setSuggestionKeyboardActive(false);
    requestAnimationFrame(() => {
      field?.focus();
      field?.setSelectionRange(start + term.length, start + term.length);
      setSelectionManually(start + term.length, start + term.length);
    });
  };

  const handleKeyDown = (event: KeyboardEvent<HTMLTextAreaElement>) => {
    // A read-only field has nothing to accept a suggestion INTO, and offering
    // one that silently does nothing is worse than offering none.
    if (readOnly) return;
    if (composingRef.current || event.nativeEvent.isComposing || !suggestions.length) return;
    if (isRecommendationActivationKey(event.key, event.ctrlKey, event.altKey)) {
      event.preventDefault();
      setSuggestionOpen(true);
      setSuggestionKeyboardActive(true);
      setSuggestionIndex(0);
      return;
    }
    // A visible recommendation list must not capture the arrows, Enter or
    // Space that Windows IME uses. Keyboard ownership transfers to ToFU only
    // after the explicit Ctrl+Space / Alt+Down gesture above.
    if (!suggestionKeyboardActive) return;
    if (suggestionOpen && /^[1-9]$/.test(event.key)) {
      const numbered = matches[Number(event.key) - 1];
      if (numbered) {
        event.preventDefault();
        acceptSuggestion(numbered);
      }
    } else if (event.key === "ArrowDown") {
      event.preventDefault();
      setSuggestionOpen(true);
      setSuggestionIndex((current) => matches.length ? Math.min(current + (suggestionOpen ? 1 : 0), matches.length - 1) : 0);
    } else if (event.key === "ArrowUp" && suggestionOpen) {
      event.preventDefault();
      setSuggestionIndex((current) => Math.max(0, current - 1));
    } else if (event.key === "PageDown" && suggestionOpen) {
      event.preventDefault();
      setSuggestionPage((current) => Math.min(suggestionPageCount - 1, current + 1));
      setSuggestionIndex(0);
    } else if (event.key === "PageUp" && suggestionOpen) {
      event.preventDefault();
      setSuggestionPage((current) => Math.max(0, current - 1));
      setSuggestionIndex(0);
    } else if ((event.key === "Enter" || event.key === "Tab") && suggestionOpen && matches[suggestionIndex]) {
      event.preventDefault();
      acceptSuggestion(matches[suggestionIndex]);
    } else if (event.key === "Escape") {
      setSuggestionOpen(false);
      setSuggestionKeyboardActive(false);
    }
  };

  const start = reveal?.start ?? value.length;
  const end = reveal?.end ?? value.length;

  return (
    <div ref={fieldRef} className={`animated-caret-field ${trailingIcon ? "has-trailing-icon" : ""}${expandable ? " is-expandable" : ""}${expanded ? " is-expanded" : ""} ${className}`}>
      <div className="animated-caret-mirror" aria-hidden="true" ref={mirrorRef}>
        {!value && <span className="animated-caret-placeholder">{placeholder}</span>}
        {!selectionCollapsed ? <>
          {value.slice(0, selectionRange.start)}
          <span key="selection" className="animated-caret-selection">{value.slice(selectionRange.start, selectionRange.end)}</span>
          {value.slice(selectionRange.end)}
        </> : selectionLeaving ? <>
          {value.slice(0, selectionLeaving.start)}
          <span key="selection-leaving" className="animated-caret-selection leaving">{value.slice(selectionLeaving.start, selectionLeaving.end)}</span>
          {value.slice(selectionLeaving.end)}
        </> : statusMessage ? <>
          {value.slice(0, caretPosition)}<span ref={caretMarkerRef} className="animated-caret-status-anchor" />{value.slice(caretPosition)}
        </> : <>
          {value.slice(0, start)}
          {reveal && <span key={reveal.key} className="animated-caret-insert">{value.slice(start, end)}</span>}
          {value.slice(end)}
        </>}
        {value.endsWith("\n") && "\u200b"}
      </div>
      <div className="animated-caret-measure" aria-hidden="true" ref={caretMeasureRef}>
        {value.slice(0, caretPosition)}<span ref={motionCaretMarkerRef} className="animated-motion-caret-marker" />{value.slice(caretPosition)}
        {value.endsWith("\n") && "\u200b"}
      </div>
      {focused && selectionCollapsed && (
        <span className="animated-motion-caret" aria-hidden="true" style={motionCaret} />
      )}
      <textarea
        ref={inputRef}
        aria-label={label}
        lang={language ?? undefined}
        inputMode="text"
        autoCapitalize="none"
        value={value}
        rows={rows}
        readOnly={readOnly}
        placeholder={placeholder}
        onFocus={(event) => { setFocused(true); onFocus?.(event); }}
        onBlur={(event) => { setFocused(false); setSuggestionOpen(false); onBlur?.(event); }}
        onClick={(event) => { setCaretPosition(event.currentTarget.selectionStart ?? value.length); onClick?.(event); }}
        onCompositionStart={handleCompositionStart}
        onCompositionEnd={handleCompositionEnd}
        onKeyDown={handleKeyDown}
        onChange={(event) => {
          const next = event.target.value;
          if (!composingRef.current) showInsertion(previousRef.current, next);
          previousRef.current = next;
          setCaretPosition(event.currentTarget.selectionStart);
          onChange(next, { isComposing: composingRef.current || Boolean((event.nativeEvent as InputEvent).isComposing) });
          if (!composingRef.current && suggestions.length) {
            setSuggestionIndex(0);
            setSuggestionPage(0);
            setSuggestionOpen(true);
            setSuggestionKeyboardActive(false);
          }
        }}
        onScroll={(event) => {
          if (!mirrorRef.current) return;
          mirrorRef.current.scrollTop = event.currentTarget.scrollTop;
          mirrorRef.current.scrollLeft = event.currentTarget.scrollLeft;
          if (caretMeasureRef.current) {
            caretMeasureRef.current.scrollTop = event.currentTarget.scrollTop;
            caretMeasureRef.current.scrollLeft = event.currentTarget.scrollLeft;
          }
          if (statusMessage) setStatusRect(caretMarkerRef.current?.getBoundingClientRect() ?? null);
        }}
        className="animated-caret-input"
        spellCheck={false}
      />
      {trailingIcon && <span className="animated-caret-trailing-icon">{trailingIcon}</span>}
      {focused && (
        <span className="animated-caret-nav">
          <button
            type="button"
            title="Move caret left"
            aria-label="Move caret left"
            onMouseDown={(event) => event.preventDefault()}
            onClick={(event) => {
              event.stopPropagation();
              const field = inputRef.current;
              if (!field) return;
              const selStart = field.selectionStart ?? value.length;
              const selEnd = field.selectionEnd ?? selStart;
              const pos = selStart === selEnd ? Math.max(0, selStart - 1) : selStart;
              field.focus();
              field.setSelectionRange(pos, pos);
              setCaretPosition(pos);
              setSelectionManually(pos, pos);
            }}
          >
            <ChevronLeft size={10} />
          </button>
          <button
            type="button"
            title="Move caret right"
            aria-label="Move caret right"
            onMouseDown={(event) => event.preventDefault()}
            onClick={(event) => {
              event.stopPropagation();
              const field = inputRef.current;
              if (!field) return;
              const selStart = field.selectionStart ?? value.length;
              const selEnd = field.selectionEnd ?? selStart;
              const pos = selStart === selEnd ? Math.min(value.length, selEnd + 1) : selEnd;
              field.focus();
              field.setSelectionRange(pos, pos);
              setCaretPosition(pos);
              setSelectionManually(pos, pos);
            }}
          >
            <ChevronRight size={10} />
          </button>
        </span>
      )}
      {expandable && (
        <button
          type="button"
          className="animated-caret-expand"
          title={expanded ? "Collapse source field" : "Expand source field"}
          aria-label={expanded ? "Collapse source field" : "Expand source field"}
          aria-expanded={expanded}
          onMouseDown={(event) => event.preventDefault()}
          onClick={(event) => { event.stopPropagation(); setExpanded((current) => !current); }}
        >
          <TbChevronsDown size={10} className={expanded ? "rotate-180" : ""} />
        </button>
      )}
      {statusMessage && statusRect && createPortal(
        <div
          role="status"
          key={statusPulse}
          className="lang-reject-tooltip lang-reject-tooltip-pulse pointer-events-none fixed z-500 whitespace-nowrap rounded-lg border border-red-300 bg-white px-3 py-2 text-xs text-red-700 shadow-lg dark:border-red-800 dark:bg-zinc-900 dark:text-red-300"
          style={{ left: statusRect.left, top: statusRect.top - 38 }}
        >{statusMessage}</div>, document.body,
      )}
      {suggestionOpen && menuRect && matches.length > 0 && createPortal(
        <div
          role="listbox"
          aria-label={`${label} Ground Truth recommendations`}
          className="dropdown-morph expanded bezier-card cjk-entry-suggestions"
          data-keyboard-active={suggestionKeyboardActive ? "true" : "false"}
          style={{ left: menuRect.left, top: menuRect.bottom + 4, width: Math.max(menuRect.width, 176) }}
          onMouseDown={(event) => event.preventDefault()}
        >
          <div className="tofu-ime-reading">User dictionary<span>{suggestionKeyboardActive ? "keyboard active" : "Ctrl+Space"}</span></div>
          {matches.map((term, index) => (
            <button
              key={term}
              type="button"
              role="option"
              aria-selected={index === suggestionIndex}
              className={`cjk-entry-suggestion ${index === suggestionIndex ? "active" : ""}`}
              onMouseEnter={() => setSuggestionIndex(index)}
              onClick={() => acceptSuggestion(term)}
            >
              <span className="cjk-entry-candidate-number">{index + 1}</span>
              <span className="min-w-0 flex-1 truncate">{term}</span>
              <span className="cjk-entry-candidate-source ground-truth">Ground Truth</span>
            </button>
          ))}
          {suggestionPageCount > 1 && (
            <div className="cjk-entry-pagination" aria-label="Suggestion pages">
              <button type="button" aria-label="Previous suggestion page" disabled={currentSuggestionPage === 0}
                onClick={() => { setSuggestionPage((page) => Math.max(0, page - 1)); setSuggestionIndex(0); }}><ChevronLeft size={14} /></button>
              <span>{currentSuggestionPage + 1} / {suggestionPageCount}</span>
              <button type="button" aria-label="Next suggestion page" disabled={currentSuggestionPage >= suggestionPageCount - 1}
                onClick={() => { setSuggestionPage((page) => Math.min(suggestionPageCount - 1, page + 1)); setSuggestionIndex(0); }}><ChevronRight size={14} /></button>
            </div>
          )}
        </div>, document.body,
      )}
    </div>
  );
}
