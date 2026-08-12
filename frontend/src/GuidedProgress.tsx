import type { GuidedProgress as Progress } from "./useGuidedCapture";

/** Two numbers with two different jobs, computed independently.
 *
 * The BAR moves on atom coverage, so drawing a box always advances
 * something visible. The LABEL counts Blocks, because "6/8" means what a
 * user thinks it means and a weighted fraction does not.
 *
 * Deriving either from the other is what produces the nearly-full bar
 * labelled `0/1` that a five-atom Block would otherwise show at four atoms —
 * and a user who reads that twice stops trusting both.
 */
export default function GuidedProgress({ progress }: { progress: Progress }) {
  if (!progress.blocks_total) return null;

  const percent = Math.round(progress.fraction * 100);
  const skipped = progress.blocks_skipped;

  // Red at 0%, yellow at 50%, green at 100%. One linear sweep through hue
  // rather than three colour stops: HSL puts yellow at exactly half of the
  // 0→120 arc, so the midpoint the user is told about is the midpoint they
  // see, with no banding where discrete stops would meet.
  const barHue = progress.fraction * 120;

  // Skipped runs the same arc backwards, and against blocks_total rather
  // than a fixed ceiling: one skip out of two is most of the work abandoned,
  // one out of thirty is a rounding error, and a fixed scale calls both the
  // same colour.
  const skippedHue = progress.blocks_total
    ? 60 * (1 - Math.min(1, skipped / progress.blocks_total))
    : 60;

  return (
    <span
      className="subtext flex items-center gap-1.5 text-[8.4px] text-zinc-500 dark:text-zinc-400"
      data-testid="guided-progress"
      role="status"
      aria-live="polite"
      /* Both values, plus the skipped count: a bar that reads "full" because
         Blocks were skipped is not the same as one that reads full because
         everything was located, and only the label can say which. */
      aria-label={
        `${progress.blocks_resolved} of ${progress.blocks_total} Blocks resolved`
        + (skipped ? `, ${skipped} skipped` : "")
        + `, ${percent}% of terms located`
      }
    >
      {/* A real bar, not eight block glyphs. The glyphs could only move in
          whole eighths, so the first box of a nine-Block run advanced
          nothing visible, and they cannot animate or carry a gradient. */}
      <span
        aria-hidden="true"
        className="inline-block h-1 w-14 shrink-0 overflow-hidden rounded-full bg-zinc-200 dark:bg-zinc-700"
      >
        <span
          className="block h-full rounded-full"
          style={{
            width: `${percent}%`,
            backgroundColor: `hsl(${barHue}, 70%, 45%)`,
            // Width and colour travel together, so the bar arrives at its
            // new length already the right colour instead of snapping.
            transition: "width 320ms cubic-bezier(0.4, 0, 0.2, 1), background-color 320ms linear",
          }}
        />
      </span>
      <span aria-hidden="true" className="text-zinc-700 dark:text-zinc-300">
        {progress.blocks_resolved}/{progress.blocks_total}
      </span>
      {skipped > 0 && (
        <span
          aria-hidden="true"
          style={{
            color: `hsl(${skippedHue}, 75%, 45%)`,
            transition: "color 320ms linear",
          }}
        >
          ({skipped} skipped)
        </span>
      )}
    </span>
  );
}
