import {
  useEffect, useRef, useState,
  type CompositionEvent, type FocusEventHandler, type MouseEventHandler,
} from "react";

type Props = {
  value: string;
  onChange: (value: string) => void;
  label: string;
  placeholder?: string;
  rows?: number;
  className?: string;
  onFocus?: FocusEventHandler<HTMLTextAreaElement>;
  onBlur?: FocusEventHandler<HTMLTextAreaElement>;
  onClick?: MouseEventHandler<HTMLTextAreaElement>;
};

type Reveal = { start: number; end: number; key: number } | null;

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

/** Native textarea interaction over a synchronized animated text mirror. */
export default function AnimatedCaretTextarea({
  value, onChange, label, placeholder, rows = 1, className = "", onFocus, onBlur, onClick,
}: Props) {
  const mirrorRef = useRef<HTMLDivElement>(null);
  const previousRef = useRef(value);
  const composingRef = useRef(false);
  const compositionBaseRef = useRef(value);
  const timerRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const sequenceRef = useRef(0);
  const [reveal, setReveal] = useState<Reveal>(null);

  useEffect(() => {
    if (!composingRef.current) previousRef.current = value;
  }, [value]);

  useEffect(() => () => {
    if (timerRef.current) clearTimeout(timerRef.current);
  }, []);

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
  };

  const handleCompositionEnd = (event: CompositionEvent<HTMLTextAreaElement>) => {
    composingRef.current = false;
    const next = event.currentTarget.value;
    showInsertion(compositionBaseRef.current, next);
    previousRef.current = next;
  };

  const start = reveal?.start ?? value.length;
  const end = reveal?.end ?? value.length;

  return (
    <div className={`animated-caret-field ${className}`}>
      <div className="animated-caret-mirror" aria-hidden="true" ref={mirrorRef}>
        {!value && <span className="animated-caret-placeholder">{placeholder}</span>}
        {value.slice(0, start)}
        {reveal && <span key={reveal.key} className="animated-caret-insert">{value.slice(start, end)}</span>}
        {value.slice(end)}
        {value.endsWith("\n") && "\u200b"}
      </div>
      <textarea
        aria-label={label}
        value={value}
        rows={rows}
        placeholder={placeholder}
        onFocus={onFocus}
        onBlur={onBlur}
        onClick={onClick}
        onCompositionStart={handleCompositionStart}
        onCompositionEnd={handleCompositionEnd}
        onChange={(event) => {
          const next = event.target.value;
          if (!composingRef.current) showInsertion(previousRef.current, next);
          previousRef.current = next;
          onChange(next);
        }}
        onScroll={(event) => {
          if (!mirrorRef.current) return;
          mirrorRef.current.scrollTop = event.currentTarget.scrollTop;
          mirrorRef.current.scrollLeft = event.currentTarget.scrollLeft;
        }}
        className="animated-caret-input"
        spellCheck={false}
      />
    </div>
  );
}
