import { useCallback, useEffect, useRef, useState } from "react";
import type { GuidedPrompt } from "./useGuidedCapture";

/** Opaque hides the image behind the hint; fill lets it show through. */
export type HintFill = "opaque" | "translucent";

/** What the canvas asks for next, over the image it is asked about.
 *
 * Positioned inside the canvas card's own stacking context so it stays put
 * when the canvas is resized, expanded or scrolled — anything anchored from
 * outside would drift. Deliberately measured with `absolute` offsets rather
 * than an IntersectionObserver: the observer never fires in the in-app
 * browser pane, silently, so the behaviour could not be verified locally.
 *
 * DRAGGABLE, because it sits over the very image the user has to draw on.
 * Text in the top-left corner of a sign is unreachable while a fixed hint
 * covers it, and the hint cannot be dismissed without losing the prompt.
 * The offset is deliberately NOT persisted: it is a nudge to get at what is
 * underneath right now, and a hint that reopens 300px from where it belongs
 * on the next asset reads as a bug.
 *
 * Presentation only. It receives a finished sentence and some flags; it
 * never decides what to prompt, which is what keeps the prompt derivable
 * from persisted state after a reload.
 */
export default function GuidedPromptBubble({
  prompt, onSkip, fill = "opaque",
}: {
  prompt: GuidedPrompt;
  onSkip?: () => void;
  fill?: HintFill;
}) {
  const saving = prompt.status === "saving";
  const partial = prompt.status === "partial" || prompt.status === "review";
  const verb = prompt.plural ? "Draw box(es) for" : "Draw box for";

  const [offset, setOffset] = useState({ x: 0, y: 0 });
  const [dragging, setDragging] = useState(false);
  const dragRef = useRef<{ startX: number; startY: number; originX: number; originY: number } | null>(null);

  const onPointerDown = useCallback((event: React.PointerEvent) => {
    // Buttons inside the bubble stop propagation, so this only fires on the
    // body: "skip" must stay clickable rather than becoming a drag handle.
    if (event.button !== 0) return;
    dragRef.current = {
      startX: event.clientX, startY: event.clientY,
      originX: offset.x, originY: offset.y,
    };
    setDragging(true);
  }, [offset.x, offset.y]);

  // Listeners on WINDOW, not the element: a fast drag outruns the pointer
  // and the bubble would be dropped the moment the cursor left it.
  useEffect(() => {
    if (!dragging) return;
    const move = (event: PointerEvent) => {
      const start = dragRef.current;
      if (!start) return;
      setOffset({
        x: start.originX + (event.clientX - start.startX),
        y: start.originY + (event.clientY - start.startY),
      });
    };
    const up = () => { dragRef.current = null; setDragging(false); };
    window.addEventListener("pointermove", move);
    window.addEventListener("pointerup", up);
    window.addEventListener("pointercancel", up);
    return () => {
      window.removeEventListener("pointermove", move);
      window.removeEventListener("pointerup", up);
      window.removeEventListener("pointercancel", up);
    };
  }, [dragging]);

  // A new Block is a new question; putting it back where it belongs beats
  // leaving it wherever the last one was dragged to.
  useEffect(() => { setOffset({ x: 0, y: 0 }); }, [prompt.blockId]);

  return (
    <div
      className="pointer-events-none absolute left-3 top-3 z-20 max-w-[min(28rem,calc(100%-1.5rem))]"
      style={{
        transform: `translate(${offset.x}px, ${offset.y}px)`,
        // No transition while dragging, or the bubble lags the pointer.
        transition: dragging ? "none" : "transform 160ms cubic-bezier(0.4, 0, 0.2, 1)",
      }}
      data-testid="guided-prompt"
      data-status={prompt.status}
      data-dragging={dragging ? "true" : undefined}
    >
      <div
        role="status"
        aria-live="polite"
        onPointerDown={onPointerDown}
        /* The accessible name always carries the WHOLE Block. A screen-reader
           user hearing `Draw box for "saisie"` has been told something
           false: saisie is an atom of "la première saisie", not a thing to
           find on its own. */
        aria-label={
          partial
            ? `${verb} “${prompt.text}”. Found ${prompt.matchedText}. Still needed: ${prompt.remainingText}.`
            : `${verb} “${prompt.text}”.`
        }
        title="Drag to move this hint out of the way"
        /* `pointer-events-auto` so the body is grabbable; the wrapper stays
           inert so the rest of the canvas is still drawable through it. */
        className={`soft-shadow pointer-events-auto select-none rounded-lg border border-pink-300 px-3 py-2 text-xs text-pink-900 dark:border-pink-800 dark:text-pink-200 ${
          fill === "opaque"
            /* Dark mode was `bg-pink-950/40` — translucent, so the image
               showed through and the hint was unreadable over busy signage,
               while light mode's `bg-pink-50` was solid. Opaque now means
               solid in BOTH themes, matching the app's dark surface. */
            ? "bg-pink-50 dark:bg-zinc-900"
            : "bg-pink-50/80 dark:bg-pink-950/40"
        } ${dragging ? "cursor-grabbing" : "cursor-grab"}`}
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
          {/* "saving", not "reading": Guided runs no recogniser. The word
              described a refinement pass that overwrote the string the user
              had already typed, and naming it honestly is also the only
              signal that the draw tool is briefly locked. */}
          {saving && <span className="opacity-60">· saving…</span>}
        </span>
        {onSkip && !saving && (
          /* `skip` is the only escape hatch now. `mark found` was removed:
             it settled a Block by DECREE while contributing no matched
             atoms, so the label read "10/10" against a bar that could never
             fill — `progress()` compensates skipped Blocks' atoms and had no
             equivalent for user_complete. With Guided declaring its own text,
             drawing the box completes the Block honestly, and a button that
             claims completion without evidence has nothing left to do. */
          <span className="mt-1.5 flex items-center gap-2">
            <button
              type="button"
              onPointerDown={(event) => event.stopPropagation()}
              onClick={onSkip}
              className="cursor-pointer rounded px-1.5 py-0.5 text-[10px] underline-offset-2 opacity-70 transition hover:opacity-100 hover:underline"
            >
              skip
            </button>
          </span>
        )}
      </div>
    </div>
  );
}
