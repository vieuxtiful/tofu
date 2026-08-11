import { X } from "lucide-react";
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
  // not collapse into one colour bucket.
  const colors = regionColorMap(blocks.map((_, index) => `b${index}`));

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
            const record = records?.[index];
            const atoms = record?.atoms?.length ?? 0;
            return (
              <li
                key={`${block}-${index}`}
                className="basil-token mapped flex max-w-full items-center gap-1 rounded-md px-2 py-0.5 text-xs"
                style={{ "--bh-ink": colors[`b${index}`] } as React.CSSProperties}
                /* The atom count is the honest way to show that a phrase
                   stayed whole: it comes from `mise`, so if the server ever
                   did split it the chip would say so. */
                title={atoms
                  ? `Block ${index + 1} — ${atoms} atom${atoms === 1 ? "" : "s"}`
                  : `Block ${index + 1}`}
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
        placeholder="Add source text ToFU should locate, then press Enter."
      />
    </div>
  );
}
