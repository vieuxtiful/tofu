import { useEffect, useRef, useState, type ComponentType } from "react";
import { ChevronDown } from "lucide-react";
import { PiCubeFocusFill, PiCubeFocusLight, PiUserFocus, PiUserFocusFill } from "react-icons/pi";
import StatusFlag from "./StatusFlag";
import type { CaptureMode } from "./api";
import type { Theme } from "./theme";

/**
 * Auto / Guided / Manual, in the Project Manager card.
 *
 * Styled as the language pickers are (transparent trigger, Dropdown Morph
 * panel) rather than as a native <select>, so the Project Manager reads as
 * one control family instead of two.
 *
 * The order is fixed: Auto first because it is the default and what every
 * existing project already does, Manual last because it turns detection off.
 */

type Props = {
  value: CaptureMode;
  onChange: (mode: CaptureMode) => void;
  disabled?: boolean;
  theme?: Theme;
};

type ModeIcon = ComponentType<{ size?: number; className?: string }>;

/** The three modes, each with the icon that carries its idea.
 *
 * Auto and Guided share the cube-focus glyph at two weights -- both point the
 * detector at the image, and the FILLED one is Guided because the user has
 * supplied the text, so the mode is the more specified of the two. Manual is
 * a different glyph, not a third weight: nothing is detected there at all,
 * and a weight change would imply it sits on the same scale as the others.
 *
 * `iconDark` is set only where the shape needs a heavier form to hold up on
 * a dark ground. Where it is absent the single icon is used in both themes.
 */
export const CAPTURE_MODES: {
  id: CaptureMode; label: string; hint: string; icon: ModeIcon; iconDark?: ModeIcon;
  /** Shown in the OPEN LIST only, never on the trigger. On the trigger it
   *  would qualify whatever mode happened to be selected, so a project set
   *  to Manual would read "Manual best". */
  flag?: string;
}[] = [
  { id: "auto", label: "Auto", hint: "Detect all text. Any terms you add are extra evidence.", icon: PiCubeFocusLight },
  { id: "guided", label: "Guided", hint: "Locate the source text you supply.", icon: PiCubeFocusFill, flag: "best" },
  { id: "manual", label: "Manual", hint: "No detection — draw regions yourself.", icon: PiUserFocus, iconDark: PiUserFocusFill },
];

function iconFor(mode: (typeof CAPTURE_MODES)[number], theme?: Theme): ModeIcon {
  return theme === "dark" && mode.iconDark ? mode.iconDark : mode.icon;
}

export default function CaptureModeSelect({ value, onChange, disabled, theme }: Props) {
  const [open, setOpen] = useState(false);
  const wrapperRef = useRef<HTMLDivElement | null>(null);
  const current = CAPTURE_MODES.find((mode) => mode.id === value) ?? CAPTURE_MODES[0];

  useEffect(() => {
    if (!open) return;
    const close = (event: MouseEvent) => {
      if (wrapperRef.current && !wrapperRef.current.contains(event.target as Node)) {
        setOpen(false);
      }
    };
    document.addEventListener("mousedown", close);
    return () => document.removeEventListener("mousedown", close);
  }, [open]);

  return (
    <div>
      <p className="subtext mb-1 text-xs uppercase tracking-wider text-zinc-500 dark:text-zinc-600">
        capture mode
      </p>
      {/* Same width as the panel it opens, and as the collapsed nav menu. */}
      <div ref={wrapperRef} className="relative w-44">
        <button
          type="button"
          aria-label="Capture mode"
          aria-haspopup="listbox"
          aria-expanded={open}
          onClick={(e) => { e.stopPropagation(); if (!disabled) setOpen((v) => !v); }}
          className={`flex w-full items-center gap-1 rounded border border-transparent px-1 py-0.5 text-left text-xs transition ${
            disabled
              ? "cursor-not-allowed opacity-50"
              : "hover:border-zinc-300 hover:bg-zinc-200 dark:hover:border-zinc-700 dark:hover:bg-zinc-800"
          } text-zinc-800 dark:text-zinc-200`}
          title={current.hint}
        >
          {(() => {
            const Icon = iconFor(current, theme);
            return <Icon size={13} className="shrink-0" />;
          })()}
          <span className="min-w-0 flex-1 truncate">{current.label}</span>
          <ChevronDown size={10} className="shrink-0 text-zinc-500" />
        </button>
        {/* Dropdown Morph — reusable expand/collapse animation (see .dropdown-morph in uikit.css) */}
        <div
          role="listbox"
          onClick={(e) => e.stopPropagation()}
          className={`dropdown-morph bezier-card absolute left-0 top-full z-100 mt-1 w-44 overflow-hidden rounded-lg border border-zinc-300 bg-white dark:border-zinc-700 dark:bg-zinc-900${open ? " expanded" : ""}`}
          style={open ? { boxShadow: "4px 4px 0 var(--bc-shadow), 8px 8px 16px rgba(0,0,0,0.18)" } : undefined}
        >
          {CAPTURE_MODES.map((mode) => {
            const Icon = iconFor(mode, theme);
            return (
              <button
                key={mode.id}
                type="button"
                role="option"
                aria-selected={mode.id === value}
                onClick={() => { onChange(mode.id); setOpen(false); }}
                className={`flex w-full items-start gap-2 px-3 py-1.5 text-left text-xs transition hover:bg-zinc-100 dark:hover:bg-zinc-800 ${
                  mode.id === value ? "text-cyan-600 dark:text-cyan-400" : "text-zinc-700 dark:text-zinc-300"
                }`}
              >
                {/* Aligned to the LABEL's line, not centred on the whole
                    option: the hint below wraps to two lines for Auto, and a
                    vertically-centred icon drifts off the name it belongs to. */}
                <Icon size={13} className="mt-0.5 shrink-0" />
                <span className="flex min-w-0 flex-col gap-0.5">
                  <span className="flex items-center gap-1.5">
                    {mode.label}
                    {mode.flag && <StatusFlag>{mode.flag}</StatusFlag>}
                  </span>
                  <span className="subtext text-[10px] leading-snug text-zinc-500">{mode.hint}</span>
                </span>
              </button>
            );
          })}
        </div>
      </div>
      <p className="subtext mt-1 text-[11px] leading-snug text-zinc-500">{current.hint}</p>
    </div>
  );
}
