import { X } from "lucide-react";
import { Kbd } from "@/components/ui/kbd";
import GroundTruthField from "./GroundTruthField";
import { regionColorMap } from "./regionPalette";
import type { GuidedBlockRecord, LanguageAssessment } from "./api";

type Props = {
  /** Committed Blocks, in order, duplicates preserved. */
  blocks: string[];
  onChange: (blocks: string[]) => void;
  /** The uncommitted entry in the input line. */
  draft: string;
  onDraftChange: (value: string) => void;
  /** Server-side records for the committed Blocks, when they have arrived.
   *  Read-only here: atoms come from `mise` or they do not exist. */
  records?: GuidedBlockRecord[];
  language?: string | null;
  sourceLanguage?: string | null;
  capturing?: boolean;
  assess?: (text: string, sourceLanguage: string | null) => Promise<LanguageAssessment>;
};

/** The Guided entry surface: committed Blocks as chips, one entry line.
 *
 * WHY THIS IS NOT THE AUTO FIELD WITH A DIFFERENT SPLIT.
 *
 * Auto's Ground Truth is a bag of terms -- whitespace-delimited, order
 * irrelevant, duplicates meaningless -- and its field renders one palette
 * colour per whitespace-separated word to show exactly that. A Guided Block
 * is the opposite: `la première saisie` is ONE thing to locate, and drawing
 * it as three differently-coloured words tells the user the system has
 * already broken it apart. Whichever way that field's tokenizer was pointed,
 * it would be lying to one of the two modes.
 *
 * So Guided commits on Enter and shows each Block whole, in one colour.
 * Order is entry order; duplicates are kept and are independently
 * fulfillable, because a user who typed the same term twice is asking for
 * two occurrences.
 *
 * Reordering is deliberately not implemented yet: delete-and-retype is the
 * available route, and a drag affordance is worth building only once the
 * reconciliation in E2 can follow a Block's progress across the move.
 */
export default function BlocksField({
  blocks, onChange, draft, onDraftChange, records,
  language, sourceLanguage, capturing, assess,
}: Props) {
  // Keyed by index, not by text: two identical Blocks are two chips and must
  // not collapse into one colour bucket. One extra key past the end is the
  // colour the DRAFT wears, so the entry line already looks like the chip it
  // becomes on Enter.
  const colors = regionColorMap(
    blocks.map((_, index) => `b${index}`).concat(`b${blocks.length}`),
  );

  const commit = (value: string) => {
    const entry = value.trim();
    if (!entry) return;
    onChange([...blocks, entry]);
    onDraftChange("");
  };

  const removeAt = (index: number) =>
    onChange(blocks.filter((_, position) => position !== index));

  return (
    <div className="space-y-1.5" data-testid="blocks-field">
      {blocks.length > 0 && (
        <ul className="flex flex-wrap items-center gap-1.5" aria-label="Blocks to locate">
          {blocks.map((block, index) => {
            return (
              <li
                key={`${block}-${index}`}
                className="basil-token mapped flex max-w-full items-center gap-1 rounded-md px-2 py-0.5 text-xs"
                style={{ "--bh-ink": colors[`b${index}`] } as React.CSSProperties}
                /* Just the Block. The atom count was an honesty check while
                   whitespace splitting was in doubt; now that a Block is
                   committed whole and located by one box, "3 atoms" only
                   invites the reader to wonder whether it was split. */
                title={`Block ${index + 1}`}
              >
                <span className="truncate">{block}</span>
                {!capturing && (
                  <button
                    type="button"
                    onClick={() => removeAt(index)}
                    aria-label={`Remove Block ${index + 1}: ${block}`}
                    className="shrink-0 opacity-50 transition hover:opacity-100"
                  >
                    <X size={10} />
                  </button>
                )}
              </li>
            );
          })}
        </ul>
      )}
      <GroundTruthField
        label="Blocks"
        language={language}
        sourceLanguage={sourceLanguage}
        capturing={capturing}
        assess={assess}
        value={draft}
        onChange={onDraftChange}
        onCommit={commit}
        onBackspaceEmpty={() => blocks.length && removeAt(blocks.length - 1)}
        /* Commit on blur too. An entry typed and then clicked away from is
           still an entry the user meant, and losing it silently is worse
           than committing something they can delete in one click. */
        onBlur={() => commit(draft)}
        /* Spaces are ordinary text here and always were; the old field just
           coloured each word separately and so appeared to split them. The
           placeholder now states the rule rather than leaving it to be
           inferred from the colouring. */
        singleToken
        tokenColor={colors[`b${blocks.length}`]}
        placeholder="Type source text; press 'Enter' to save."
      />
      {/* BELOW the field, left-justified. */}
      <span
        className="flex items-center gap-1 text-[10px] text-zinc-500 dark:text-zinc-500"
        data-testid="blocks-space-hint"
      >
      </span>
    </div>
  );
}
