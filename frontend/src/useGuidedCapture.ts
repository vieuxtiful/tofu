import { useMemo } from "react";
import type { GuidedBlockRecord, GuidedState } from "./api";

export type GuidedPromptStatus = "pending" | "partial" | "review" | "saving" | "complete";

/** Presentation-only description of what to ask for next.
 *
 *  Deliberately flat and derived: the canvas is handed a finished sentence
 *  and some flags, never the workflow. Keeping the state machine out of the
 *  component is what lets the prompt be re-derived from persisted state
 *  after a reload instead of replayed from a counter the reload lost. */
export type GuidedPrompt = {
  blockId: string;
  /** The whole Block, always — the accessible name uses this even when only
   *  the remaining part is emphasised. */
  text: string;
  /** What is still missing. Equal to `text` until something matches. */
  remainingText: string;
  /** What has already been found, for the struck-through chips. */
  matchedText: string;
  plural: boolean;
  status: GuidedPromptStatus;
};

export type GuidedProgress = GuidedState["progress"];

const EMPTY_PROGRESS: GuidedProgress = {
  blocks_total: 0, blocks_complete: 0, blocks_skipped: 0, blocks_review: 0,
  blocks_resolved: 0, atoms_total: 0, atoms_matched: 0, fraction: 0,
};

/** How the prompt is worded.
 *
 *  A Block that can only be one box says "Draw box for"; one that may take
 *  several says "Draw box(es) for". `find_all` means every occurrence, so it
 *  is plural regardless of how the phrase divides.
 *
 *  The phrase is NEVER re-worded down to what is missing. Showing
 *  `Draw box for "saisie"` while the Block is "la première saisie" tells the
 *  user that `saisie` is a thing to find in its own right, which is the one
 *  claim the whole Block model exists to deny. */
function isPlural(block: GuidedBlockRecord): boolean {
  return block.find_all || block.atoms.filter((a) => a.kind !== "punctuation").length > 1;
}

export function promptFor(
  block: GuidedBlockRecord | undefined,
  saving: boolean,
): GuidedPrompt | null {
  if (!block) return null;
  const assessment = (block.detection_assessment ?? {}) as {
    status?: string; matched_text?: string; remaining_text?: string;
  };
  const status = (assessment.status ?? "pending") as string;
  return {
    blockId: block.id,
    text: block.raw_text,
    remainingText: assessment.remaining_text || block.raw_text,
    matchedText: assessment.matched_text || "",
    plural: isPlural(block),
    status: saving
      ? "saving"
      : status === "partial" || status === "review"
        ? (status as GuidedPromptStatus)
        : "pending",
  };
}

/** Derive the Guided view from what the SERVER last said.
 *
 *  Nothing here advances anything. The active Block, the coverage and the
 *  statuses all come from the persisted assessment, because an optimistic
 *  local advance that the server then contradicts leaves the screen and the
 *  stored state disagreeing — with the user believing the wrong one. */
export function useGuidedCapture({
  enabled, blocks, activeBlockId, progress, saving,
}: {
  enabled: boolean;
  blocks: GuidedBlockRecord[];
  activeBlockId: string | null;
  progress: GuidedProgress | null;
  saving: boolean;
}) {
  return useMemo(() => {
    if (!enabled || blocks.length === 0) {
      return {
        prompt: null as GuidedPrompt | null,
        progress: EMPTY_PROGRESS,
        activeBlock: undefined as GuidedBlockRecord | undefined,
        finished: false,
      };
    }
    const activeBlock = blocks.find((b) => b.id === activeBlockId);
    const state = progress ?? EMPTY_PROGRESS;
    return {
      prompt: promptFor(activeBlock, saving),
      progress: state,
      activeBlock,
      // "Nothing is asking any more", which includes Blocks the user
      // skipped. It is deliberately NOT "everything was located" — those
      // are different questions and `blocks_complete` answers the second.
      finished: state.blocks_total > 0 && state.blocks_resolved >= state.blocks_total,
    };
  }, [enabled, blocks, activeBlockId, progress, saving]);
}
