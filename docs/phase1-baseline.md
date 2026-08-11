# Phase 1 baseline — can silhouette matching rescue a misread?

Measured 2026-08-11. Harness: `scripts/eval_proof_matcher.py`. Layer:
`src/tofu/layers/proof.py`. Evidence: `evidence/proof-baseline-v1.json`.

**The headline: the deterministic baseline scores 10.7%, and the reason is not
degradation. It is typeface.** That reframes what Phase 1's encoder should be
learning.

---

## The question

A region has been located and misread. Ground truth offers candidate strings.
Can rendering each candidate and comparing silhouettes recover the right one
from ink the recognizer could not read?

This is the cheapest thing that could possibly work — no training, no weights,
nothing that could memorise the evaluation set — and it is the number a trained
encoder has to beat. `docs/measured-dead-ends.md` lists nine interventions that
looked obviously right in the abstract and lost on measurement; an encoder
proposed without a baseline would be the tenth.

## The control: the method itself is sound

Rendered cleanly and matched against distinct candidates **in the same face**:

| truth | picked | support | margin |
|---|---|---:|---:|
| `SORTIE` | `SORTIE` | 1.000 | 0.423 |
| `1789` | `1789` | 1.000 | 0.511 |
| `PARIS` | `PARIS` | 1.000 | 0.493 |

And it survives every degradation in the catalogue — `blur`, `fade`, `abrasion`,
`resample` all still pick correctly at moderate severity. So silhouette
comparison discriminates strings, decisively, with wide margins, **and
degradation does not break it.**

That control matters: without it, the result below would be indistinguishable
from a bug in the harness.

## The result on real ink

56 ground-truth regions across 11 fixtures — every region whose recognizer read
was wrong and whose ink could be segmented. Regions already read correctly are
excluded; counting them would inflate the figure with work already done.

| condition | top-1 |
|---|---:|
| 2 faces (Arial, MS Gothic) | **10.7%** (6/56) |
| 14 faces, best-support wins | **17.9%** (10/56) |
| 14 faces, **any** face correct | **46.4%** (26/56) |

Median support 0.441, median margin 0.029.

**The margin carries no signal.** Median margin when the matcher is right:
0.018. When it is wrong: **0.029**. Confidence is not merely weak — it is
slightly inverted. Any decision rule built on this margin would be worse than
random at knowing when to trust itself.

## What this says

**1. Degradation is not the binding constraint.** The control is degradation-
robust and scores 1.000. Real ink scores 0.441. The difference between those
conditions is not that the real ink is degraded — it is that the real ink is in
a face the renderer does not have.

**2. Typeface is.** Going from 2 faces to 14 nearly doubles accuracy
(10.7% → 17.9%), and an oracle over faces reaches 46.4%. The information is
present in nearly half of all cases and the *face-selection rule throws it
away*.

**3. The selection rule looked like the defect, and is not fixable.** Picking
the face with the highest support jointly maximises over face and string, so a
wrong string in a well-matched face beats the right string in a poorly-matched
one — two questions answered by one number, the confound `decant` refuses
elsewhere. But separating them does not help: both aggregation rules tested
below score *worse* than the joint argmax.

**4. Even the oracle tops out below half.** 46.4% with perfect face choice is
the honest ceiling for pure silhouette matching on this corpus. That is a real
target for an encoder rather than a rhetorical one.

## Two aggregation rules, tested and rejected

If the right face is in the set for 46.4% of regions, the obvious question is
whether a better *rule* recovers it without knowing which face is right.
Absolute scores are not comparable across faces — a well-matched face scores
everything highly — but ranks within a face should be. Two ways to exploit
that, on the same 56 regions:

| rule | top-1 |
|---|---:|
| argmax over (face, string) — the current rule | 16.1% |
| **Borda**: rank strings within each face, sum ranks | **10.7%** |
| **z-norm**: normalise each face's scores, then sum | **8.9%** |
| oracle over faces | 46.4% |

**Both aggregations are worse than doing nothing.** The reason is
straightforward in hindsight: most faces are the wrong face, and a wrong face
ranks strings close to arbitrarily. Averaging over fourteen faces of which one
or two are right drowns the signal that argmax at least concentrates on.

(The 16.1% here versus 17.9% above is one region, from tie-breaking. At n=56
that difference is noise and neither figure should be quoted to a decimal.)

**This is the finding that argues FOR the encoder rather than against it.**
Face selection cannot be finessed by a scoring rule, and it cannot be solved
by identification either — `font_matching.local_match` needs the region's TEXT
to identify its face, and the text is precisely what is being recovered. The
circularity is real.

What breaks a circularity is a representation in which the face does not
matter. That is a typeface-invariant embedding — which is an encoder, and now
one with a measured reason to exist and a measured target to beat.

## Consequences for Phase 1 as proposed

The program's A1 specifies an encoder trained on *degradation* pairs —
canonical render versus synthetically degraded render, with hard negatives from
confusables. On this evidence **that trains the wrong invariance.** Degradation
is what the control shows the method already survives; typeface is what breaks
it.

Three adjustments were considered. Two are now closed:

- **Using the face ToFU already identifies: ruled out.**
  `font_matching.local_match` requires the region's `text` to rank faces, and
  the text is what is being recovered. The circularity is not incidental —
  identification works by rendering the known string in each face.
- **Separating the two decisions by rank aggregation: measured and rejected.**
  Borda 10.7%, z-norm 8.9%, both below the 16.1% argmax baseline. See above.
- **Train typeface-invariance, not degradation-invariance.** Same string across
  many faces → same embedding; different strings → separated. This is now the
  surviving option rather than the preferred one, which is a stronger position
  to build from: the two cheaper alternatives were tried and lost. Degradation
  pairs remain worth including — `proof.degrade()` is how any arm gets tested
  against controlled damage — but they are not the axis the measurement points
  at.

## Honest limits

- **n = 56**, from 11 fixtures, 9 of which carry partial annotations. Small.
- **Segmentation is upstream of everything here.** The observed mask comes from
  `imaging.text_mask`; where it fails, this measures segmentation rather than
  matching. Not separated in these numbers.
- **The candidate pool is per-image annotated strings plus mined confusables**,
  which is a realistic Ground Truth pool but not a fixed-size one — top-1 rates
  are not comparable across images with different pool sizes.
- **The face-oracle figure is an oracle.** 46.4% is not achievable without
  knowing the right face; it bounds the approach, it does not describe a system.

## Reproducing

```bash
.venv/Scripts/python scripts/eval_proof_matcher.py --out evidence/proof-baseline-v1.json
```

Roughly 12 minutes, real OCR for the "what did the recognizer read" column.
