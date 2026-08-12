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

## Encoder increment

The first trainable increment now lives in `src/tofu/layers/proof_encoder.py`,
with `scripts/train_proof_encoder.py` as its deliberately separate training
entry point.  It is not wired into Cicerone and it has no production verdict:
an uncalibrated embedding may not turn a region green.

The experiment holds out two things at once:

- strings are assigned deterministically to train or holdout, so the score
  cannot be earned by memorising the 56 labels in this baseline; and
- holdout strings are rendered only in holdout faces, so the score measures
  the typeface invariance the failed aggregation arms say is needed.

The model is a small convolutional encoder with unit-normalised embeddings and
a supervised contrastive loss.  Degradation remains a secondary augmentation
(`blur`, `resample`, `abrasion`), while cross-face identity is the primary
positive relation.  Torch remains optional at import time and the checkpoint
is an experiment artifact, not package data.

This is infrastructure, not a result.  The next recorded number must be
held-out cross-face retrieval, followed by the frozen real-ink harness after
the annotation debt is cleared.  Until both exist, the deterministic 10.7--
17.9% interval remains the only real-ink baseline and 46.4% remains an oracle,
not a claimed target achieved by the encoder.

The first Traditional Chinese cohort is
`evidence/proof-encoder-zh-hant-v1.txt`.  Its target `靄籔蠱爨鑿` and each
constituent glyph are passed through `--ground-truth`; the trainer reserves
them from training even if they are accidentally added to the corpus later.
The cohort contains related dense forms, shared radicals, common material for
cross-face invariance, and strings of several lengths.  It is a synthetic
rendering cohort, not a linguistic corpus.

### Traditional Chinese run 1

Measured with seed 17, 98 training strings, three training faces (MingLiU,
DFKai-SB, Microsoft JhengHei), and two unseen faces (Noto Sans TC, Noto Serif
TC).  The corrected sequence-preserving encoder and supervised contrastive
loss moved held-out cross-face top-1 from the collapsed run's 20.3% to
**37.5%**; loss fell from 8.394 to 6.793 over 30 epochs.

On the separately reserved target set, the complete string `靄籔蠱爨鑿` ranks
first in both directions between the two unseen Noto families.  The five
constituent glyphs score 4/5 Sans-to-Serif and 3/5 Serif-to-Sans, or **70%**
over the ten directional trials.  This is encouraging retrieval evidence but
not calibration evidence: most single-glyph winning margins are below 0.004,
and `鑿` fails in both directions.  Nothing in this run may produce a green
decision.

Evidence:

- `evidence/proof-encoder-zh-hant-v1.json`
- `evidence/proof-encoder-zh-hant-v1-target-sans-to-serif.json`
- `evidence/proof-encoder-zh-hant-v1-target-serif-to-sans.json`

## Phase 2: evidence survival and calibration

`layers/decant.py` remains the authority on whether positive identity evidence
survives.  `layers/proof_calibration.py` now calibrates the encoder's winning
margin separately and gates decisions in this order:

1. `absent` or `weak` survival is unresolvable regardless of model score;
2. `partial` or `unknown` evidence requires review;
3. an unfitted margin model requires review;
4. a fitted model without a measured acceptance threshold requires review;
5. only `present` evidence above that threshold can be accepted.

The calibrator fits a two-parameter logistic map by negative-log-likelihood and
derives its acceptance threshold from the requested precision, rather than
declaring a similarity cutoff.  ECE and Brier are recorded before and after
fitting.  The first artifact contains 12 directional target trials: 9 correct,
3 incorrect, ECE 0.238 and Brier 0.239.  It correctly remains unfitted because
30 labelled trials are required, and therefore contains no acceptance
threshold.  Evidence: `evidence/proof-margin-calibration-zh-hant-v1.json`.

## Phase 3: real-ink pilot

`scripts/eval_proof_real_ink.py` compares the learned embedding and the
deterministic silhouette matcher on identical masks, candidate pools and font
cohorts. Candidate embeddings are averaged across five TC faces; this is the
operation the typeface-invariant representation exists to make meaningful.

The first `prem-sais` pilot scores five Han-labelled manifest regions:

| arm | top-1 |
|---|---:|
| deterministic silhouette | 2/5 (40%) |
| typeface-invariant encoder | **4/5 (80%)** |

Both methods recover `鼜㒪` and `試驗會合`. The encoder additionally recovers
`鬣` and `蠹`; both miss `釁`. This is a pilot, not the Phase 3 gate: the image
is partially annotated, labels are aligned from an existing manifest rather
than a frozen two-reviewer annotation, and two discordant wins are not enough
for a meaningful paired significance claim. These five rows are therefore
excluded from the margin-calibration artifact.

Evidence: `evidence/proof-real-ink-zh-hant-v1.json`.

### Phase 3 corpus freeze

`evidence/proof-real-ink-corpus-v1.json` freezes the annotation boundary rather
than treating every available transcription as interchangeable ground truth.
It requires crop-verified text on a photographed surface for diagnostic
scoring, independently reviewed Traditional Chinese for the primary stratum,
and two independent reviews plus adjudication before a row can calibrate an
acceptance threshold.

The repository currently supplies nine scoreable transfer rows but **zero
primary rows and zero calibration rows**. Five are Simplified Chinese and four
are Japanese. The generated `gemini-street` scene is excluded from real ink,
and the five `prem-sais` Traditional Chinese candidates remain excluded until
their manifest transcriptions are independently reviewed.

On the two diagnostic photographs, both arms recover 1/9 overall. The encoder
is 1/5 versus 0/5 for the deterministic arm on the Simplified Chinese stratum;
it is 0/4 versus 1/4 on the Japanese stratum. These results neither confirm nor
reject the Phase 3 hypothesis. They reveal a missing nuisance contract instead:
most rows are vertical or contain mixed-length strings, while the current
sequence-preserving representation assumes a common reading direction. That
must be measured and normalized before a larger real-ink score is interpretable.

Evidence:

- `evidence/proof-real-ink-corpus-v1.json`
- `evidence/proof-real-ink-corpus-eval-v1.json`

## Phase 4: reading-direction normalization

The encoder was trained on horizontal renders, while seven of the nine frozen
diagnostic rows are top-to-bottom signs. `proof_encoder.canonical_reading_direction`
now rotates a declared vertical observation counter-clockwise, preserving its
top-to-bottom order as left-to-right before either arm sees it. This is metadata
ToFU already has; it does not infer orientation from the answer.

On the unchanged corpus and candidate pools:

| arm | raw | reading-normalized |
|---|---:|---:|
| deterministic silhouette | 1/9 | 3/9 |
| typeface-invariant encoder | 1/9 | **4/9** |

This is a useful ablation, not a promotion result: all nine rows remain
cross-script or regional-form transfer diagnostics, with zero primary zh-Hant
rows. The encoder is 2/5 on Simplified Chinese and 2/4 on Japanese after
normalization.

Candidate length is recorded as a stratum rather than used to filter the pool.
Filtering candidates to the known answer length would leak ground truth into
the matcher. The observed strata are: 2/2 for length 3, 1/3 for length 4, 0/1
for length 5, 1/2 for length 6, and 0/1 for length 7. The sparse cells cannot
support a learned length correction; they show that orientation normalization
does not solve longer-string generalization.

## Phase 5: vertical augmentation and auxiliary length prediction

The matched v2 arm adds clockwise-rotated training observations and a length
classification head to the same convolutional trunk. Its objective is
supervised contrastive identity loss plus 0.25 times length cross-entropy. The
length estimate therefore comes from pixels; no candidate is filtered using
its ground-truth length. Schema-1 checkpoints still load through the unchanged
embedding architecture.

The auxiliary task does not transfer and v2 is rejected:

| measure | v1 incumbent | v2 multitask |
|---|---:|---:|
| held-out cross-face identity | **37.5%** | 14.1% |
| held-out synthetic length | n/a | 87.5% |
| reserved full string | 2/2 | 2/2 |
| reserved constituent glyphs | **7/10** | 3/10 |
| normalized real-ink diagnostic identity | **4/9** | 2/9 |
| real-ink diagnostic length | n/a | 1/9 |

This is negative transfer, not an improvement hidden by a secondary metric.
The shared trunk learns synthetic layout/length cues at the expense of the
fine identity structure Proof needs, and those layout cues fail on loose,
processed scene masks. The v1 checkpoint remains the incumbent. A future
length arm would need a detached head or independently encoded geometric
features and real reviewed supervision; it must not share gradients with the
identity embedding on this evidence.

Evidence:

- `evidence/proof-encoder-zh-hant-v2.json`
- `evidence/proof-encoder-zh-hant-v2-target-sans-to-serif.json`
- `evidence/proof-encoder-zh-hant-v2-target-serif-to-sans.json`
- `evidence/proof-real-ink-corpus-eval-v2.json`

## Phase 6: detached geometry evidence

The geometry arm is now a separate convolutional network, optimizer and
checkpoint. It never loads the identity checkpoint, has no identity projection,
and its prediction is recorded beside retrieval rather than fused into the
rank. Training spans clean, blurred, resampled, abraded, occluded, loose-crop,
horizontal and vertical renders with class-balanced cross-entropy.

The structural intervention works but the evidence does not transfer. Synthetic
held-out length accuracy is 78.4% (307/384 horizontal and 295/384 vertical).
On frozen real ink it remains **1/9**, the same exact-match result as the failed
shared head, with mean absolute error 1.78 characters after reading-direction
normalization. Predictions systematically undercount longer processed strings.

Crucially, v1 identity remains exactly 4/9 and the deterministic arm 3/9. This
confirms gradient isolation, but it also closes the hypothesis that isolation
alone solves length. Scene segmentation and loose/noisy character boundaries
are the likely missing domain variables. The detached output remains diagnostic
only and may not filter candidates, alter identity ranks, or influence a green
decision.

Evidence:

- `evidence/proof-geometry-zh-hant-v1.json`
- `evidence/proof-geometry-zh-hant-v1.pt`
- `evidence/proof-real-ink-corpus-eval-v3.json`

## Phase 7: observational Scene material evidence

Scene now records a provenance-bearing `material_evidence` object on candidate
surfaces and propagates the same observation to contained text instances. The
object separates a compact taxonomy from raw descriptors (edge density,
luminance variation, colour dispersion and sample size), lists degradation
processes only as `plausible_not_measured`, records that glyphs have not yet
been excluded, and fixes `decision_eligible` to false.

This is deliberately separate from `ReconstructionProfile`: it cannot select
an inpainting provider, veto a detection, rerank a glyph candidate, calibrate a
margin or turn a recommendation green. It round-trips through the manifest so
future reviewer labels can join to the observation that produced them.

The first inventory covers 44 candidate surfaces across `china-street`,
`japan-street` and `prem-sais`: 36 unknown, five heuristic masonry, one painted
panel and two textured-unknown. The 81.8% unknown rate is an honest baseline,
not a recall failure. None of the five masonry hypotheses is a reviewed
material label; all remain uncalibrated observations. The next measurement must
use glyph-excluded substrate and reviewed surface labels before material can
condition restoration or identity inference.

Evidence: `evidence/scene-material-evidence-v1.json`.

## Phase 8: glyph-excluded substrate ablation

`scene.substrate_material_evidence` now constructs a surface crop from the
Scene polygon or box, removes every known glyph mask with a three-pixel safety
growth, falls back visibly to text bounding boxes when masks are absent, fills
excluded pixels with the observed substrate median, and reruns the same
conservative material taxonomy. It records coverage, sample trustworthiness,
fallback count and the median-fill limitation. It mutates neither surfaces nor
text instances.

On the same 44 photographed candidate surfaces, 39 retain measurable substrate
and five become unmeasurable because the available annotation boxes consume
the sample. Among measurable surfaces, four material classes change: one of
five contaminated masonry calls becomes unknown, while three prior unknowns
become masonry. Four masonry calls remain stable. This is not evidence that
glyph exclusion improves material accuracy because no reviewed material labels
exist; it is evidence that the present heuristic is sensitive to glyph pixels
and exclusion geometry.

`evidence/scene-material-review-schema-v1.json` defines the annotation debt:
two independent reviewers, visible-evidence rationales, a bounded material and
degradation taxonomy, and explicit agreement/adjudication/unresolvable states.
Only adjudicated labels can support a later calibration experiment. The paired
ablation remains observational and decision-ineligible.

Evidence: `evidence/scene-material-substrate-ablation-v1.json`.

## Phase 9: material review packet

`scripts/build_scene_material_review_packet.py` generates a reviewer-facing
packet from the frozen photographic cases. Each of 44 surface cards contains a
context crop with the reviewed region outlined, the original contaminated
surface crop, a glyph-excluded analysis crop when substrate survives, source
and exclusion provenance, two empty independent-review slots, and pending
adjudication. No label is prefilled and every record starts ineligible for
scoring.

The packet contains 44 context crops, 44 contaminated crops and 39
glyph-excluded crops. Five cards explicitly show that no substrate survived.
Visual QA also confirms why mask provenance matters: tracked GT fixtures supply
boxes rather than glyph outlines, so some exclusions remove most of a sign and
leave large median-filled areas. The packet labels those blocks as exclusions,
not reconstructed material. Such cards may be judged unresolvable; they must
not be treated as negative material labels.

Review artifact: `evidence/scene-material-review-v1/index.html` and
`evidence/scene-material-review-v1/review-queue.json`.

## Phase 10: review ingestion and agreement gate

`material_review.validate` applies the published JSON schema plus semantic
checks the schema cannot express cleanly: reviewer identities must be distinct,
timestamps must parse, matching reviews require a matching `agreed` resolution,
and disagreements require either a named, reasoned adjudication or an explicit
`unresolvable` outcome. Invalid and duplicate files remain visible and make the
ingestion command fail.

Agreement is reported both as the raw proportion and Cohen's kappa. Kappa stays
undefined for an empty set or a degenerate single-class population rather than
inventing reliability. Label eligibility and substrate-scoring eligibility are
separate; neither implies production decision eligibility.

The initial real status is intentionally empty: 44 queued, zero submitted,
zero valid, zero label-eligible, zero substrate-scoring-eligible, 44 pending,
and no agreement statistic. `production_decision_eligible` is false. Completed
records belong in `evidence/scene-material-review-v1/completed/`; the ingestion
tool never creates judgments.

Evidence: `evidence/scene-material-review-status-v1.json`.
