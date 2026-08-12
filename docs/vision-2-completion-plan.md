# ToFU Vision 2 Completion Plan (revised)

Revision 2, 2026-08-11. Supersedes revision 1. Changes are confined to the
measurement design; the governance, rollout and safety structure of revision 1
survives essentially intact, because it was the strongest part of it.

Review that produced these changes: `docs/vision-2-completion-plan-review.md`.

## What changed, and why

| # | Revision 1 | Problem | Revision 2 |
|---|---|---|---|
| 1 | 500 regions, 20% certification, ≥149 error-free greens | **Impossible.** 20% of 500 is 100 certification regions; greens are a subset, so the ceiling is 100 < 149 | Precision target is tied to the ROLLOUT STAGE, and the certification split is sized from the target by a pre-registered rule (§4) |
| 2 | Per stratum: ≥10 greens, none below 98% observed | With n=10 that means **zero errors per stratum**; a system at exactly 98% fails ~87% of the time across 10 strata | Pooled lower bound is the gate; strata are tested for **non-inferiority to pooled** at corrected alpha, with a real minimum n |
| 3 | "No stratum's top-1 regresses by more than 2 points", ≥30 per stratum | One region is 3.3 points, so "2 points" is zero tolerance stated in misleading units | Stated in **regions**, with the per-stratum n that makes points meaningful |
| 4 | "Evaluate it once on untouched certification data" | Right, but no failure policy — and §§1–3 made failure likely, so the discipline erodes on first contact | Sealed second split for exactly one retry; after that, new assets or nothing |
| 5 | Paired evaluation on "same crops" for both arms | On CJK a per-crop read scores 22% where the pipeline scores 65–78%, so the incumbent would be handicapped 3:1 and +5 would be meaningless | The incumbent is the **pipeline's shipped read**; crops are fixed for Vision 2's input only |
| 6 | zh-Hant corpus alongside the outstanding Guided review | Two large blind reviews competing for the same people, one of which already blocks Gate 2 | Explicit sequencing, with the Guided review first |

---

## Summary

Complete Vision 2 through a gated, shadow-first program for Traditional Chinese
glyph recovery.

Success requires:

- Curated, independently reviewed zh-Hant ground truth.
- Asset-disjoint training, calibration, and certification sets.
- A one-sided 95% precision lower bound meeting the **stage-appropriate target**
  (§4): **95% for `guided_review`, 98% for `auto_gt_recommend`**.
- At least +5 percentage points paired top-1 improvement over the incumbent,
  with one-sided exact McNemar `p < 0.05`.
- No material regression in OCR, Scene, Cleanse, rendering, or declared corpus
  strata.
- Explicit abstention whenever evidence, calibration, or supported-domain
  coverage is insufficient.
- Guided review rollout before Auto + GT; Auto + GT remains recommendation-only.

If a gate fails, Vision 2 remains shadow or review-only. No threshold may be
relaxed automatically.

---

## 1. Stabilize evidence and persistence contracts

*(Unchanged from revision 1 except where noted.)*

- Add one server-owned per-instance `glyph_match_evidence` object containing:
  - Candidate ranking and candidate-pool revision.
  - Encoder/checkpoint revision.
  - Raw support and winning margin.
  - Calibrated probability and calibration revision.
  - Evidence-survival state and measurements.
  - Orientation normalization and material/degradation context.
  - Decision: `shadow`, `review_required`, `unresolvable`, or `recommended`.
  - Reason codes, supported-domain status, and provenance.
- **Add `glyph_match_evidence` and `material_evidence` to
  `manifest_store.SERVER_OWNED_INSTANCE_FIELDS`** — the per-instance analogue of
  the manifest-level tuple, which exists as of this week. The serializer audit
  test parametrises over the `InstText` dataclass, so a field added without a
  serializer entry fails a test rather than losing a user's work. This is not
  hypothetical: eight per-region fields were being dropped on every save until
  they were caught, including `lineage_candidate_id`.
- Link each observation to the existing candidate-lineage node rather than
  creating a second provenance graph. Note `lineage_candidate_id` only began
  persisting this week; before that the join did not survive a reload.
- Keep source OCR text immutable unless the user explicitly accepts a
  recommendation.
- Preserve schema-1 encoder checkpoints and old manifests; missing Vision 2
  fields mean "not evaluated", never rejection.
- **Freeze and commit the v1 checkpoint before any arm is compared against it.**
  `proof_encoder.py`, `proof_calibration.py` and `proof_geometry.py` are in the
  tree; until they are committed with a pinned checkpoint and revision string,
  "incumbent" is not a fixed reference and no paired result means anything.

## 2. Complete the primary zh-Hant corpus

- Curate real photographed text regions across independent assets. **Size is
  determined by §4's sizing rule, not chosen up front.** The staged defaults:

  | stage | precision target | error-free greens needed | certification regions at 60% coverage | total corpus at 20% split |
  |---|---:|---:|---:|---:|
  | `guided_review` | 95% | 59 | 99 | **~500** |
  | `auto_gt_recommend` | 98% | 149 | 249 | **~1,250** |

  **Revision 1's 500 regions are sufficient for the first stage and not for the
  second.** That is the honest statement, and it lets Stage A proceed now while
  the corpus extension is scheduled for what it actually unlocks.

- Include marginal strata with **at least 60 reviewed regions each** (raised
  from 30 — see §4.3):
  - Horizontal and vertical.
  - Length 1, 2–4, and 5+ code points.
  - Planar/painted, textured/masonry, and unknown material.
  - No/light and moderate/severe visible degradation.
- Include rare and confusable Traditional Chinese glyphs, including the reserved
  `靄籔蠱爨鑿` cohort.
- Complete all text boxes and transcriptions; partial annotations are excluded
  from precision certification.
- Require two blind reviewers per transcription and surface/material label, with
  adjudication or `unresolvable`.
- Record inter-review raw agreement and Cohen's κ. Unresolved rows remain
  diagnostic only.
- Freeze asset-disjoint splits before model comparison: 60% training, 20%
  calibration/model selection, 20% certification — **plus a sealed retry split
  carved from the certification portion (§4.4)**.
- Prevent the same asset, sign, near-duplicate crop, string rendering, or video
  track from crossing splits.

**Sequencing (new).** The zh-Hant corpus does not exist; this is new collection,
not an extension. The Guided corpus review (162 blocks, transcription only) is
**still outstanding and already blocks Gate 2**. Run it first — it is smaller,
it unblocks a separate certification, and it calibrates how long blind review
actually takes here before a several-times-larger review is committed to.

## 3. Finish the retrieval model and evidence-survival path

*(Unchanged from revision 1, with one addition.)*

- Retain v1 as the incumbent encoder until another arm wins the frozen
  comparison.
- Train typeface-invariant identity using cross-face positives, physical
  degradation augmentation, vertical observations normalized to the canonical
  reading axis, and hard negatives mined from actual reviewed corrections.
- Keep length/geometry gradients detached from identity; do not use predicted
  length as a hard candidate filter.
- Evaluate pre-registered ablations: v1 Euclidean contrastive incumbent;
  batch-hard plus supervised contrastive; entropic component alignment for
  fragmented glyphs; hyperbolic embedding as an ablation only.
- Select the model using calibration-set top-1, paired wins, calibration
  quality, and abstention behaviour — not synthetic accuracy alone.
- Extend evidence survival with measured detector activation, glyph extent,
  segmentation completeness, and OCR-quality evidence.
- Enforce: `absent`/`weak` → `unresolvable`; `partial`/`unknown` →
  `review_required`; only `present` may proceed to calibrated recommendation;
  candidate elimination alone can never produce green certainty.

**Why typeface-invariance and not degradation-invariance (measured).** The
deterministic baseline scores 1.000 with 0.42–0.51 margins on a clean render in
the *matching* face, and survives blur, fade, abrasion and resample. On real ink
it scores 10.7% at two faces and 17.9% at fourteen, against a **46.4% face
oracle**. Degradation is what the method already survives; typeface is what
breaks it. Two cheaper alternatives were then measured and closed — identifying
the face first is circular (`local_match` needs the text to rank faces), and
rank aggregation across faces is worse than doing nothing (Borda 10.7%, z-norm
8.9%). `docs/phase1-baseline.md`.

## 4. Calibrate and certify *(substantially revised)*

### 4.1 Asset-grouped calibration

- Fit the probability mapping on the calibration split.
- Select the green threshold on calibration data only.
- Evaluate once on certification data (§4.4 governs what "once" means).
- Report ECE, Brier score, reliability bins, coverage, top-1, top-3, abstention
  rate, and precision/recall of recommendations.

### 4.2 Sizing rule, pre-registered

The certification split must be large enough that the gate is *reachable* before
any model is trained. With zero errors, the exact one-sided 95%
Clopper–Pearson lower bound is `0.05^(1/n)`, so:

```
greens_needed(T) = ceil( ln(0.05) / ln(T) )
certification_regions >= greens_needed(T) / expected_green_coverage
```

| target T | error-free greens | at 50% coverage | at 60% | at 75% |
|---:|---:|---:|---:|---:|
| 95% | 59 | 118 | 99 | 79 |
| 97% | 99 | 198 | 165 | 132 |
| 98% | **149** | 298 | 249 | 199 |

`expected_green_coverage` is registered **before** certification, from the
calibration split. If observed coverage on certification falls below it, the
run is undersized and produces review-only output — it does not get re-gated at
a lower target after the fact.

### 4.3 Green certification requires

- **Pooled**: exact one-sided 95% Clopper–Pearson lower bound ≥ the stage target
  (95% for `guided_review`, 98% for `auto_gt_recommend`).
- **Per stratum**: at least **60 evaluated regions and 30 green predictions**.
  Raised from revision 1's 30/10 so that a single error is not automatically a
  failure and so a percentage-point gate has resolution.
- **Per stratum, non-inferiority rather than a point estimate.** A stratum fails
  only if its one-sided lower bound is below the pooled target at
  `alpha = 0.05 / n_strata` (Bonferroni). Revision 1's rule — "no stratum below
  98% observed" at n=10 — required zero errors everywhere and would have failed a
  perfectly-conforming system **~87% of the time** across ten marginal strata
  (`0.98¹⁰ = 0.817` per stratum, `0.817¹⁰ = 0.133` overall).
- **Regression, stated in regions**: no stratum may lose more than **one region**
  of top-1 versus the incumbent. At 60 evaluated regions that is ≈1.7 points, and
  the unit is one the data can actually express.

### 4.4 Failure policy, pre-registered *(new)*

Certification is spent by looking at it. Revision 1 said "evaluate once" and
gave no rule for what follows a failure, which is exactly when the discipline
would have been abandoned.

- The certification portion is split into a **primary** and a **sealed retry**
  set at freeze time, asset-disjoint from each other.
- A failed certification returns the feature to shadow. The team may retrain and
  attempt the **sealed set exactly once**, at `alpha = 0.025` each to preserve a
  family-wise 0.05.
- After a second failure, no further certification is possible **until the corpus
  is extended with new assets**. Re-reading either set is leakage and is not
  permitted regardless of how much the model has changed.

### 4.5 Paired model promotion

- At least **+5 absolute points top-1** over the incumbent.
- **One-sided exact McNemar `p < 0.05`.**
- Positive paired effect in both horizontal and vertical strata.
- Report "is it real" (significance) and "is it enough" (margin) as **separate
  verdicts**. This is the fix for the instrument that just cost Gate 2 its
  certification: +7.4 points at `p = 0.00012` failed an +8-point bar, and a
  single conflated verdict cannot express that.

### 4.6 The incumbent is the shipped pipeline read *(new)*

Crops, masks, candidate pools and fonts are fixed across arms so the comparison
is paired. **But the incumbent's score is its result in the shipped pipeline,
not a fresh read of the fixed crop.**

Measured this week: a per-crop read reaches 22% on Japanese strata where the
pipeline reaches 65–78%, because multipass, zoom, surface probes, edge rescue
and the correction layers carry almost the whole CJK read. Scoring the incumbent
from crops would hand Vision 2 a +40-point head start and make the +5 gate
meaningless. `docs/gate2-status.md`.

This project has now recorded **four** instances of measuring a configuration the
product does not run. It is the most expensive recurring error here, and it is
always discovered after the number has been quoted.

### 4.7 Registration

Register every learned threshold, revision, corpus hash, expected coverage and
provenance in `docs/threshold-register.md`, **each with a provenance label**.
The register's detection tally is currently 4 `measured`, 10 `reasoned`,
3 `inherited`, majority `invented`; a feature whose safety argument is
calibration must not add to the last column.

## 5. Introduce material context safely

*(Unchanged from revision 1.)*

- Finish the existing 44-surface review packet, then add reviewed surfaces from
  the primary corpus.
- Measure contaminated versus glyph-excluded material accuracy against
  adjudicated labels. `scene.substrate()` provides the glyph-excluded sample.
- Calibrate material classification separately; `unknown` remains a valid
  abstention.
- Treat material/degradation evidence initially as descriptive model input and
  stratification metadata.
- Add material-conditioned retrieval only as a paired ablation: it may alter
  candidate scores only after improving certification performance; it may never
  override weak/absent evidence survival; generated restoration pixels may not
  be treated as observed glyph evidence.
- Defer material-conditioned Cleanse routing and forward restoration until a
  separate repair-quality gate passes.

## 6. Production integration and rollout

*(Unchanged from revision 1, which is the strongest section of the plan.)*

Feature states `off` / `shadow` / `guided_review` / `auto_gt_recommend`, all
environments defaulting to `off`; `shadow` only with a valid checkpoint,
calibration artifact and matching corpus/revision metadata.

- **Shadow**: compute and persist evidence; change nothing the user sees; log
  recommendation, abstention reason, incumbent result, latency and later
  correction.
- **Guided review**: show the recommendation, calibrated probability,
  evidence-survival state and reason chips; require explicit accept/reject;
  rejections enter the hard-negative ledger only after human confirmation.
  **Gated at 95% certified precision (§4.2).**
- **Auto + GT recommendation**: show certified green recommendations
  prominently; never auto-write source text; require explicit acceptance with
  undo and immutable provenance. **Gated at 98%, which requires the extended
  corpus.**
- Roll back automatically to shadow on checkpoint/calibration revision mismatch,
  unavailable evidence, or monitored precision below the registered gate.

## 7. UI behavior

*(Unchanged from revision 1.)*

Reuse existing states — green/recommended only for certified `recommended`,
amber for `review_required`, indeterminate for missing evidence, unresolvable
for absent/weak identity evidence. Add proposed string, calibrated probability,
evidence-survival state, typeface/material context, abstention reasons and model
revision to RegionTable and bbox tooltips. Clearly distinguish OCR confidence
from Vision 2 match probability. Provide Accept, Reject and "cannot determine".
Keep the second reviewer blind in corpus-review tooling and prevent machine
hypotheses from prefilling human labels.

---

## Test and Acceptance Plan

*(Revision 1's plan, with additions marked.)*

- **Unit**: encoder ranking, orientation normalization, entropic alignment,
  calibration, exact confidence bounds, survival gates, reason codes. Old
  checkpoints/manifests load without Vision 2 fields. Weak, absent, partial,
  unknown, unfitted, unsupported-stratum and revision-mismatch cases abstain.
  **New**: the sizing rule refuses an undersized certification split before a
  run rather than after.
- **Persistence/API**: server-owned evidence survives repeated autosaves and all
  region mutations; accept/reject is revision-guarded, auditable, reversible;
  Guided Blocks and legacy GT keep separate semantics. **New**: the serializer
  audit parametrised over `InstText` covers the new fields.
- **Corpus**: hashes and split membership frozen; no asset or near-duplicate
  leakage; two distinct reviewers and adjudication mandatory; strata counts
  enforced. **New**: the sealed retry split is asset-disjoint from the primary
  and is inaccessible until a documented first failure.
- **Paired evaluation**: same crops, masks, candidate pools and fonts for both
  arms; McNemar significance and effect size reported separately; certification
  data cannot participate in training, threshold selection or model selection.
  **New**: a test asserts the incumbent's score comes from the pipeline path,
  not a crop read (§4.6).
- **Regression**: full backend pytest, frontend tests/build, mypy, ruff, package
  build, regression images. OCR recall/CER, Scene surface recall, Cleanse repair
  quality, render containment, latency and memory within registered tolerances.
- **UI**: state colours and language correct; OCR confidence and match
  probability not conflated; Auto + GT never changes text without acceptance;
  keyboard, screen-reader, reload, stale-revision and undo flows work.
- **Rollout monitoring**: coverage, abstention, accept/reject/cannot-determine
  outcomes, post-accept corrections, per-stratum precision, latency. Any
  certified-stratum precision estimate below the registered boundary disables
  recommendations and returns the feature to shadow.

## Assumptions and Defaults

- Primary scope is static-image Traditional Chinese; video and other scripts
  remain shadow-only until separately certified.
- **Green precision targets are staged: 95% for `guided_review`, 98% for
  `auto_gt_recommend`**, each as an exact one-sided 95% lower bound. Revision 1's
  flat 98% is retained as the eventual target, not the entry bar.
- The paired effect gate is +5 percentage points and exact one-sided McNemar
  `p < 0.05`.
- Stratum gates are marginal rather than every cross-product combination, and
  are **non-inferiority tests against pooled** at Bonferroni-corrected alpha.
- Vision 2 remains recommendation-only in both Guided and Auto + GT.
- Material evidence cannot influence production decisions until reviewed and
  separately calibrated.
- The existing v1 identity encoder remains the incumbent unless a frozen paired
  experiment promotes another arm — **and must be committed with a pinned
  checkpoint before the first comparison**.

## Open decision for the owner

§2's staging assumes shipping `guided_review` at 95% is acceptable, on the
grounds that every recommendation there is explicitly accepted or rejected by a
human. If it is not — if 98% is the entry bar — then the corpus must be ~1,250
regions before *anything* ships, and the annotation effort roughly triples
before the first user sees a recommendation. That is a product call, not a
measurement one, and it is the single largest cost lever in this plan.
