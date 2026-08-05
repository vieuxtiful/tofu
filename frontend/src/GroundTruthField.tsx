import {
  useEffect, useMemo, useRef, useState,
  type CompositionEvent, type CSSProperties, type ReactNode,
} from "react";
import { regionColorMap } from "./regionPalette";
import { insertedRange } from "./AnimatedCaretTextarea";

type Props = {
  value: string;
  onChange: (value: string) => void;
  label: string;
  placeholder?: string;
};

/** IME-safe native input with Basil's delimiter-settled colour mirror. */
export default function GroundTruthField({ value, onChange, label, placeholder }: Props) {
  const mirrorRef = useRef<HTMLDivElement>(null);
  const previousRef = useRef(value);
  const composingRef = useRef(false);
  const compositionBaseRef = useRef(value);
  const timerRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const sequenceRef = useRef(0);
  const [reveal, setReveal] = useState<{ start: number; end: number; key: number } | null>(null);
  const pieces = useMemo(() => value.split(/(\s+)/), [value]);
  const words = useMemo(() => pieces.filter((piece) => piece && !/^\s+$/.test(piece)), [pieces]);
  const ids = useMemo(() => words.map((_, index) => `gt${index}`), [words]);
  const colors = useMemo(() => regionColorMap(ids), [ids]);
  let wordIndex = -1;
  const finalSettled = /\s$/.test(value);

  useEffect(() => {
    if (!composingRef.current) previousRef.current = value;
  }, [value]);

  useEffect(() => () => {
    if (timerRef.current) clearTimeout(timerRef.current);
  }, []);

  const showInsertion = (previous: string, next: string) => {
    const range = insertedRange(previous, next);
    if (!range) { setReveal(null); return; }
    setReveal({ ...range, key: ++sequenceRef.current });
    if (timerRef.current) clearTimeout(timerRef.current);
    timerRef.current = setTimeout(() => setReveal(null), 160);
  };

  const animatedPiece = (piece: string, pieceStart: number): ReactNode => {
    if (!reveal) return piece;
    const localStart = Math.max(0, reveal.start - pieceStart);
    const localEnd = Math.min(piece.length, reveal.end - pieceStart);
    if (localStart >= localEnd) return piece;
    return <>{piece.slice(0, localStart)}<span key={reveal.key} className="animated-caret-insert">{piece.slice(localStart, localEnd)}</span>{piece.slice(localEnd)}</>;
  };

  let pieceOffset = 0;

  return (
    <div className="basil-plate ground-truth-field animated-caret-field">
      <div className="basil-plate-mirror" aria-hidden="true" ref={mirrorRef}>
        {!value && <span className="text-zinc-400 dark:text-zinc-600">{placeholder}</span>}
        {pieces.map((piece, index) => {
          if (!piece) return null;
          const start = pieceOffset;
          pieceOffset += piece.length;
          if (/^\s+$/.test(piece)) return <span key={index}>{animatedPiece(piece, start)}</span>;
          wordIndex += 1;
          const pending = wordIndex === words.length - 1 && !finalSettled;
          return (
            <span
              key={index}
              className={`basil-token${pending ? "" : " mapped"}`}
              style={pending ? undefined : { "--bh-ink": colors[`gt${wordIndex}`] } as CSSProperties}
            >
              {animatedPiece(piece, start)}
            </span>
          );
        })}
      </div>
      <input
        aria-label={label}
        value={value}
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
        }}
        onChange={(event) => {
          const next = event.target.value;
          if (!composingRef.current) showInsertion(previousRef.current, next);
          previousRef.current = next;
          onChange(next);
        }}
        onScroll={(event) => {
          if (mirrorRef.current) mirrorRef.current.scrollLeft = event.currentTarget.scrollLeft;
        }}
        className="basil-plate-input"
        spellCheck={false}
        autoComplete="off"
      />
    </div>
  );
}
