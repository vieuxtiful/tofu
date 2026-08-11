import type { GuidedPrompt } from "./useGuidedCapture";

/** What the canvas asks for next, over the image it is asked about.
 *
 * Positioned inside the canvas card's own stacking context so it stays put
 * when the canvas is resized, expanded or scrolled — anything anchored from
 * outside would drift. Deliberately measured with `absolute` offsets rather
 * than an IntersectionObserver: the observer never fires in the in-app
 * browser pane, silently, so the behaviour could not be verified locally.
 *
 * Presentation only. It receives a finished sentence and some flags; it
 * never decides what to prompt, which is what keeps the prompt derivable
 * from persisted state after a reload.
 */
export default function GuidedPromptBubble({
  prompt, onComplete, onSkip,
}: {
  prompt: GuidedPrompt;
  onComplete?: () => void;
  onSkip?: () => void;
}) {
  const saving = prompt.status === "saving";
  const partial = prompt.status === "partial" || prompt.status === "review";
  const verb = prompt.plural ? "Draw box(es) for" : "Draw box for";

  return (
    <div
      className="pointer-events-none absolute left-3 top-3 z-20 max-w-[min(28rem,calc(100%-1.5rem))]"
      data-testid="guided-prompt"
      data-status={prompt.status}
    >
      <div
        role="status"
        aria-live="polite"
        /* The accessible name always carries the WHOLE Block. A screen-reader
           user hearing `Draw box for "saisie"` has been told something
           false: saisie is an atom of "la première saisie", not a thing to
           find on its own. */
        aria-label={
          partial
            ? `${verb} “${prompt.text}”. Found ${prompt.matchedText}. Still needed: ${prompt.remainingText}.`
            : `${verb} “${prompt.text}”.`
        }
        className="soft-shadow rounded-lg border border-pink-300 bg-pink-50 px-3 py-2 text-xs text-pink-900 dark:border-pink-800 dark:bg-pink-950/40 dark:text-pink-200"
      >
        <span aria-hidden="true" className="flex flex-wrap items-center gap-1.5">
          <span className="font-medium">{verb}</span>
          {partial ? (
            <>
              {/* Matched atoms are struck through in place rather than
                  removed, so the phrase stays legible as ONE thing while
                  showing what is left of it. */}
              <span className="line-through opacity-50">{prompt.matchedText}</span>
              {/* .guided-term reads --bbox-color from the canvas card, so
                  the term being hunted matches the boxes being drawn. */}
              <span className="guided-term">{prompt.remainingText}</span>
            </>
          ) : (
            <span className="font-medium">“{prompt.text}”</span>
          )}
          {saving && <span className="opacity-60">· reading…</span>}
        </span>
        {(onComplete || onSkip) && !saving && (
          /* The escape hatches, and they are not polish: some text is
             genuinely unreadable, and without an explicit way out one bad
             sign holds the workflow open indefinitely. */
          <span className="pointer-events-auto mt-1.5 flex items-center gap-2">
            {onComplete && (
              <button
                type="button"
                onClick={onComplete}
                className="rounded px-1.5 py-0.5 text-[10px] underline-offset-2 opacity-70 transition hover:opacity-100 hover:underline"
              >
                mark found
              </button>
            )}
            {onSkip && (
              <button
                type="button"
                onClick={onSkip}
                className="rounded px-1.5 py-0.5 text-[10px] underline-offset-2 opacity-70 transition hover:opacity-100 hover:underline"
              >
                skip
              </button>
            )}
          </span>
        )}
      </div>
    </div>
  );
}
