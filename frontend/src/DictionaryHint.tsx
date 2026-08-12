import { useState } from "react";
import { X } from "lucide-react";
import { Kbd, KbdGroup } from "@/components/ui/kbd";

/** Where the dismissal is remembered.
 *
 * Versioned, and in localStorage rather than component state: a hint that
 * returns on every reload has not been dismissed, it has been postponed,
 * and users read that as a bug. The version suffix is the way to bring it
 * back deliberately if the shortcut ever changes. */
const DISMISS_KEY = "tofu.hint.dictionary.v1";

function alreadyDismissed(): boolean {
  try {
    return window.localStorage.getItem(DISMISS_KEY) === "1";
  } catch {
    // Private browsing and locked-down profiles throw on access rather than
    // returning null. A hint is not worth a crash; show it and move on.
    return false;
  }
}

/** "⌨ Ctrl + Space — User Dictionary", collapsible rather than destroyable.
 *
 *  Alt+ArrowDown does the same thing (see `isRecommendationActivationKey`
 *  in AnimatedCaretTextarea). It is named in the accessible label rather
 *  than shown, because Ctrl+Space is taken by the input-source switcher on
 *  macOS and a hint that names only a shortcut the OS intercepts is worse
 *  than no hint at all.
 *
 *  DISMISSAL COLLAPSES; IT DOES NOT DELETE. The hint used to return `null`,
 *  so a user who dismissed it had no way back short of clearing site data --
 *  a permanent consequence for a click that looks provisional. It now
 *  collapses to its own ✕, which rotates a quarter turn to read as a `+`
 *  affordance and expands the hint again on click. The stored flag persists
 *  the COLLAPSED state, so the choice still survives a reload.
 */
export default function DictionaryHint() {
  const [collapsed, setCollapsed] = useState(alreadyDismissed);

  const remember = (value: boolean) => {
    setCollapsed(value);
    try {
      window.localStorage.setItem(DISMISS_KEY, value ? "1" : "0");
    } catch { /* nothing to remember it with */ }
  };

  return (
    <p
      /* 8.4px to match the status readouts it shares a line with. */
      className="subtext flex items-center gap-1 text-[8.4px] text-zinc-500 dark:text-zinc-400"
      data-testid="dictionary-hint"
      data-collapsed={collapsed ? "true" : "false"}
    >
      <button
        type="button"
        onClick={() => remember(!collapsed)}
        title={collapsed ? "Show the dictionary shortcut" : "Hide the dictionary shortcut"}
        aria-label={collapsed ? "Show the dictionary shortcut hint" : "Dismiss the dictionary shortcut hint"}
        aria-expanded={!collapsed}
        className="dictionary-hint-toggle shrink-0 rounded p-0.5 opacity-60 transition hover:opacity-100"
      >
        {/* One glyph, two meanings. Rotating the ✕ a quarter turn lands it on
            a `+`, so the control that closed the hint is visibly the control
            that reopens it -- rather than a second, unrelated affordance
            appearing where the first one used to be. */}
        <X size={12} className="dictionary-hint-x" aria-hidden="true" />
      </button>
      <span
        className="dictionary-hint-body"
        /* `inert` while collapsed: the width transition leaves the content in
           the DOM, and a keyboard user must not tab into a hint that is not
           on screen. */
        aria-hidden={collapsed ? "true" : undefined}
      >
        <span
          className="flex flex-nowrap items-center gap-1.5 whitespace-nowrap"
          aria-label="User dictionary: press Control plus Space, or Alt plus Down Arrow"
        >
          {/* Keys first, then what they do — the same order the user performs
              it in, and the shape every other shortcut readout on this row
              already uses ("A draw", "Esc deselect"). */}
          <KbdGroup aria-hidden="true">
            <Kbd className="h-4 min-w-4 px-1 text-[10px] font-normal">Ctrl</Kbd>
            <span className="text-zinc-400 dark:text-zinc-600">+</span>
            <Kbd className="h-4 min-w-4 px-1 text-[10px] font-normal">Space</Kbd>
          </KbdGroup>
          <span>User Dictionary</span>
        </span>
      </span>
    </p>
  );
}
