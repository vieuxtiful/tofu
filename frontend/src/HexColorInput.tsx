import { useEffect, useState } from "react";

type HexColorInputProps = {
  value: string | null;
  onChange: (value: string | null) => void;
  placeholder?: string;
};

const HEX = /^#(?:[\da-f]{3}|[\da-f]{4}|[\da-f]{6}|[\da-f]{8})$/i;
const SHORT_HEX = /^#([\da-f])([\da-f])([\da-f])([\da-f])?$/i;

function normalize(value: string) {
  const short = value.match(SHORT_HEX);
  if (!short) return value.toLowerCase();
  return `#${short[1]}${short[1]}${short[2]}${short[2]}${short[3]}${short[3]}${short[4] ? `${short[4]}${short[4]}` : ""}`.toLowerCase();
}

/** Allows a person to type a hex value without losing intermediate keystrokes. */
export default function HexColorInput({ value, onChange, placeholder = "#000000" }: HexColorInputProps) {
  const [draft, setDraft] = useState(value ?? "");
  useEffect(() => setDraft(value ?? ""), [value]);

  return (
    <input
      type="text" value={draft} placeholder={placeholder} aria-label="hex color"
      onChange={(event) => {
        const next = event.target.value.trim();
        setDraft(next);
        if (!next) onChange(null);
        else if (HEX.test(next)) onChange(next.toLowerCase());
      }}
      onBlur={() => {
        const next = draft.trim();
        if (!next) { onChange(null); return; }
        if (HEX.test(next)) {
          const canonical = normalize(next);
          setDraft(canonical);
          onChange(canonical);
        } else {
          setDraft(value ?? "");
        }
      }}
      className="w-24 rounded border border-zinc-300 bg-white px-2 py-0.5 font-mono text-xs text-zinc-800 dark:border-zinc-700 dark:bg-zinc-900 dark:text-zinc-200"
    />
  );
}
