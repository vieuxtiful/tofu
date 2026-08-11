# Guided corpus review protocol

**Status: not started.** `evidence/guided-corpus-v1.json` carries
`review_status: machine_derived`, and every harness that could certify Gate 2
refuses until that changes.

This document is the plan for changing it. It is deliberately a *protocol*
rather than a task: the corpus cannot certify itself, and a review that is
improvised while looking at the scores it will validate is worth nothing.

---

## Why this blocks everything downstream

The corpus's Blocks were **derived from existing ground-truth annotations by a
script** (`scripts/build_guided_corpus.py`), not written by anyone asking ToFU
to find something. So a passing Gate 2 score on it today would measure *the
derivation rule*, not guidance. `eval_guided_corpus.py` and
`compare_guided_arms.py` both refuse on `review_status`, and that refusal must
not be weakened to unblock a schedule — it is the only thing standing between
a number and a claim.

There is precedent for the failure this guards against. The first attempt at
strata assignment read **file names**, and produced a corpus that claimed RTL
coverage it did not have: `rtl-sign` is the Latin word "Welcome" (a sign to be
localized *into* RTL), `indic-shaping` is LTR Devanagari/Thai/Tamil, and
`cjk-horizontal` is annotated "City Library". Derivation rules look right until
somebody reads the actual content.

## What a reviewer is deciding

For each of the **162 Blocks** across **46 assets**, one question:

> Is `requested_text` a thing a person localizing this asset would plausibly
> type into the Blocks field and ask ToFU to find?

That is a judgement about *the request*, not about the detector. A reviewer
should never see, and never be told, how either arm scored on the Block.

Four verdicts:

| Verdict | Meaning |
|---|---|
| `plausible` | A real request. Keep as-is. |
| `implausible` | Not something anyone would ask for (an artefact of the derivation — a fragment, a stray glyph, a merge remnant). Drop the Block. |
| `mis-scoped` | Real text, wrong boundary — e.g. one word extracted from a phrase somebody would request whole, or two independent signs fused. Record the correct text; the Block is re-derived, not deleted. |
| `unsure` | Send to adjudication without an opinion. |

Plus one field-level check per Block: does `expected_occurrences` match what is
actually visible in the image? Occurrence *count* is what recall is measured
against, and a wrong count corrupts both arms equally and silently.

## Procedure

1. **Two independent reviewers.** Both review all 162 Blocks. Neither sees the
   other's verdicts until both are complete, and neither sees any arm's score
   at any point.
2. **Blind to the split.** Dev/holdout membership is withheld during review.
   The split is deterministic and content-blind by construction
   (`split_rule` in the corpus file); a reviewer who knows which assets are
   held out can no longer be trusted not to have been influenced by it.
3. **Adjudication.** Every disagreement is resolved by discussion between the
   two reviewers, and the resolution is recorded with its reasoning. An
   adjudication that is not written down did not happen.
4. **Re-derivation.** `mis-scoped` Blocks are rewritten and re-atomized through
   `mise`; `implausible` Blocks are dropped. Both change the corpus, so the
   corpus is re-frozen and re-hashed afterwards.
5. **Re-check the minimums.** After drops, the corpus must still clear
   assets ≥ 30, blocks ≥ 120, occurrences ≥ 180, strata ≥ 6. Current totals are
   46 / 162 / 189 / 6, so the occurrence budget has **9 to spare** — a review
   that drops more than nine occurrences puts the corpus below its own
   declared floor and more fixtures have to be generated before Gate 2 can run
   at all. This is the most likely way the review fails, and it should be
   watched from the first session rather than discovered at the end.
6. **Record, then flip.** Write reviewers, dates, per-Block verdicts and
   adjudications into the corpus's `review` block, then set
   `review_status: human_reviewed`.

## What gets recorded

The corpus file already reserves the shape (`review.annotators`,
`review.adjudications`). Fill it:

```jsonc
"review": {
  "required": [ ... unchanged ... ],
  "annotators": [
    {"name": "...", "reviewed": 162, "completed": "2026-08-.."},
    {"name": "...", "reviewed": 162, "completed": "2026-08-.."}
  ],
  "agreement": {"blocks": 162, "agreed": 0, "rate": 0.0},
  "adjudications": [
    {"block_id": "...", "asset": "...", "verdicts": ["plausible", "mis-scoped"],
     "resolution": "mis-scoped", "new_text": "...", "reason": "..."}
  ],
  "dropped_blocks": [{"block_id": "...", "reason": "..."}]
}
```

**Report the agreement rate.** Two reviewers who agree on 161 of 162 Blocks have
told you the derivation was already fine; two who agree on 120 have told you the
corpus needed this badly. Either is useful; only the number distinguishes them,
and it is free to compute once both passes exist.

## Effort

162 Blocks, each a look at one crop and one string. At a conservative minute per
Block that is under three hours per reviewer, plus adjudication. It is **calendar
time, not engineering time** — which is exactly why it should be scheduled now
and run alongside the remaining implementation rather than after it.

## Sequencing against the rest of the work

Nothing in Guided capture waits on this. `compare_guided_arms.py` runs today and
reports its numbers with the refusal attached; the review converts those numbers
from *indicative* to *certifying*. The one result with veto power over the
feature is the oracle arm (does perfect localization even help?), and that does
not need the review either.

So: start the review now, keep building, and let certification land when the
review does.
