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

  const filled = Math.round(progress.fraction * 8);
  const percent = Math.round(progress.fraction * 100);
  const skipped = progress.blocks_skipped;

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
      <span aria-hidden="true" className="font-mono tracking-tighter">
        {"█".repeat(filled)}{"░".repeat(Math.max(0, 8 - filled))}
      </span>
      <span aria-hidden="true" className="text-zinc-700 dark:text-zinc-300">
        {progress.blocks_resolved}/{progress.blocks_total}
      </span>
      {skipped > 0 && (
        <span aria-hidden="true" className="text-amber-600 dark:text-amber-400">
          ({skipped} skipped)
        </span>
      )}
    </span>
  );
}
