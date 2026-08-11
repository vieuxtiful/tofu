import { useState } from "react";
import { MdTipsAndUpdates } from "react-icons/md";
import { X } from "lucide-react";
import { Kbd, KbdGroup } from "@/components/ui/kbd";
import KeyboardIcon from "./KeyboardIcon";

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

/** "Dictionary = ⌨ Ctrl + Space" — the user dictionary's activation
 *  shortcut, which until now was only discoverable by opening the
 *  suggestion popover that the shortcut opens.
 *
 *  Alt+ArrowDown does the same thing (see `isRecommendationActivationKey`
 *  in AnimatedCaretTextarea). It is named in the accessible label rather
 *  than shown, because Ctrl+Space is taken by the input-source switcher on
 *  macOS and a hint that names only a shortcut the OS intercepts is worse
 *  than no hint at all. */
export default function DictionaryHint() {
  const [dismissed, setDismissed] = useState(alreadyDismissed);

  if (dismissed) return null;

  const dismiss = () => {
    setDismissed(true);
    try { window.localStorage.setItem(DISMISS_KEY, "1"); } catch { /* nothing to remember it with */ }
  };

  return (
    <p
      className="bezier-impression subtext flex items-start gap-1 px-3 py-2 text-xs text-zinc-500 dark:text-zinc-600"
      data-testid="dictionary-hint"
    >
      <MdTipsAndUpdates size={14} className="mt-0.5 shrink-0 text-[#2d8cf0]" />
      <span
        className="flex flex-1 flex-wrap items-center gap-1.5"
        aria-label="User dictionary: press Control plus Space, or Alt plus Down Arrow"
      >
        <span>Dictionary =</span>
        <KeyboardIcon size={14} aria-hidden="true" />
        <KbdGroup aria-hidden="true">
          <Kbd>Ctrl</Kbd>
          <span className="text-zinc-400 dark:text-zinc-600">+</span>
          <Kbd>Space</Kbd>
        </KbdGroup>
      </span>
      <button
        type="button"
        onClick={dismiss}
        title="dismiss"
        aria-label="Dismiss the dictionary shortcut hint"
        className="shrink-0 rounded p-0.5 opacity-60 transition hover:opacity-100"
      >
        <X size={12} />
      </button>
    </p>
  );
}
