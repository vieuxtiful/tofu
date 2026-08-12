# Review — ToFU Vision 2 Completion Plan

Reviewed 2026-08-11 against the Phase 1 measurements taken this week.

**Verdict: the governance is the strongest part of this plan and should be kept
almost verbatim. The statistics contain one blocking contradiction and two
gates that are self-defeating at the sample sizes proposed.** Fix the sizing
and this is executable.

---

## 1. BLOCKER — the corpus cannot satisfy the certification gate

The plan asks for both of these:

> 500 real photographed text regions … 60% training, 20% calibration, 20% untouched certification

> At least 149 green certification predictions if error-free

500 × 20% = **100 certification regions.** Green predictions are a *subset* of
certification predictions, so the ceiling is 100 — and the gate needs 149.
**Even at 100% green coverage with zero abstention the gate cannot be met.**

The 149 itself is correct, and worth saying so: with zero errors the one-sided
95% Clopper–Pearson lower bound is `0.05^(1/n)`, and `0.05^(1/149) = 0.9801`,
the first n clearing 0.98. The arithmetic is right; it just was not reconciled
with the corpus size.

What it takes to close, at a realistic green coverage (the fraction of
certification regions that reach `recommended` rather than abstaining):

| green coverage | certification regions needed | implied corpus at 20% |
|---:|---:|---:|
| 100% | 149 | 745 |
| 75% | 199 | 995 |
| 60% | 249 | 1,245 |
| 50% | 298 | 1,490 |

**Pick one of three, explicitly:**

- **Grow the corpus to ~1,250–1,500 regions.** Honest, and roughly triples the
  annotation cost — which is already the dominant cost of this plan (§5).
- **Re-split.** 40/20/40 on 500 gives 200 certification regions; enough at
  ~75% coverage, not at 50%. Costs training data, which is the thing a
  typeface-invariant encoder is most short of.
- **Lower the certified precision target,** and say so in the product. 95% with
  zero errors needs n=59; 97% needs n=99. A 97% green with a 20%-split 500-region
  corpus is achievable and defensible. 98% is not, at this size.

This is arithmetic, not judgement, and everything downstream of certification
depends on which option is chosen.

## 2. The per-stratum precision gate fails ~87% of the time even when the system is exactly right

> Each declared stratum has … at least 10 green predictions
> No stratum has observed green precision below 98%

With 10 green predictions, observed precision is 100% or ≤90%. There is no
value between. So the gate is really **"zero errors in every stratum"**.

If the true green precision is exactly 98% — i.e. the system meets its target
perfectly — then per stratum `P(0 errors in 10) = 0.98¹⁰ = 0.817`, and across
the ten marginal strata the plan declares (2 orientations + 3 lengths + 3
materials + 2 degradation levels):

| marginal strata | P(all pass) |
|---:|---:|
| 8 | 0.199 |
| **10** | **0.133** |
| 12 | 0.089 |

**A system that exactly meets spec fails this gate seven times in eight.** That
is a multiple-comparisons problem, and it will read in the room as "the model
is not good enough" when what happened is that the gate was mis-specified.

Fixes, any one of which works:

- Gate each stratum on its own **lower bound** rather than its point estimate,
  with an alpha corrected for the number of strata — the honest version of what
  the plan is reaching for.
- Or gate the pooled precision (§1) and require only that **no stratum is
  significantly worse than pooled**, which is the question actually being asked.
- Or raise the per-stratum green minimum enough that one error is not
  automatically a failure — that means ~50+ greens per stratum, which pushes
  the corpus further in the §1 direction.

## 3. The 2-point stratum regression gate is finer than the data's resolution

> at least 30 evaluated regions [per stratum] … No stratum's top-1 regresses by
> more than 2 percentage points

At 30 regions, one region is **3.3 points**. A 2-point tolerance is therefore
"no regression at all, not even one region", stated in units that imply
tolerance exists. Either raise the per-stratum n to ~100 (1 region = 1pp) or
state the gate in regions: "no stratum loses more than one region."

Same class of error as §2: a threshold chosen in percentage points without
checking what one sample is worth.

## 4. Certification can only be spent once, and there is no failure policy

> Select the green threshold on calibration data only. Evaluate it once on
> untouched certification data.

Correct, and it is the right instinct. But the plan has no answer for **what
happens when it fails** — and §§1–3 make failure likely. Retrain, then evaluate
again, and the certification set is no longer untouched; the second look is
already a mild form of the leakage the split exists to prevent.

Pre-register one of:

- a fixed number of certification attempts with an alpha correction across
  them, or
- a **second, sealed certification split** held for exactly one retry, or
- a rule that a failed certification returns to shadow and cannot be re-attempted
  until the corpus is extended with new assets.

Without this, the "evaluate once" discipline quietly erodes on the first
failure, which is the moment it matters most.

## 5. Scope: zh-Hant is a from-scratch corpus, and the review is the critical path

The existing Guided corpus has `japanese_horizontal` and `japanese_vertical`
strata and no Traditional Chinese stratum. So "curate at least 500 real
photographed text regions" is **new collection**, not an extension of anything
in the repository.

Then: two blind reviewers per region, over transcription *and* surface/material
labels. The existing Guided corpus review — 162 blocks, transcription only —
has **not happened yet** and is already the blocker on Gate 2
(`docs/guided-corpus-review-protocol.md`). This plan adds a second review that
is several times larger, on data that does not exist yet.

Sequence these deliberately rather than letting them collide. And note that
§1's fix multiplies this cost: a 1,250-region corpus with two reviewers over
two label types is the single largest line item in the plan by a wide margin,
and nothing about the modelling can be certified without it.

## 6. A trap the measurements this week point straight at

> Paired evaluation: Same crops, masks, candidate pools, and fonts for incumbent
> and Vision 2.

Fixing the crops is right for pairing. But **if the incumbent is scored by
reading those crops, on CJK that handicaps it by roughly three to one.**
Measured this week (`docs/gate2-status.md`): a per-crop read reaches 22% on
Japanese strata where the shipped pipeline reaches 65–78%, because multipass,
zoom, surface probes, edge rescue and the correction layers carry almost the
entire CJK read.

Vision 2 would clear a +5-point gate against that baseline trivially and
meaninglessly. **The incumbent must be the pipeline's shipped read for the
region, not a fresh crop read** — with crops fixed only for Vision 2's own
input. Otherwise this repeats the error `docs/measured-dead-ends.md` names as
the most expensive available here: measuring a configuration the product does
not run. It has now happened four times in this repository, three of them this
week.

## 7. What is right, and should not be negotiated away

- **The 149 arithmetic.** Correct, and correctly one-sided.
- **The +5 / exact one-sided McNemar structure.** This is the fix for the
  instrument that just cost Gate 2 a certification: +7.4 points at p = 0.00012
  failed an +8-point bar. Reporting "is it real" and "is it enough" separately
  is exactly right, and +5 is a defensible bar given the deterministic baseline
  sits at 10.7–17.9% against a 46.4% face-oracle ceiling.
- **Shadow → guided_review → auto_gt_recommend, defaulting to `off`, with
  automatic rollback on revision mismatch or monitored precision drop.** This is
  better than anything currently in the codebase and should be the template for
  future risky features.
- **Never auto-writing source text.** Combined with immutable provenance and
  undo, this is the right posture for a system whose whole failure mode is
  confident wrongness.
- **Abstention gates keyed to evidence survival**, with `absent`/`weak` →
  `unresolvable` and elimination alone never producing green. This matches
  `layers/decant.py` exactly and inherits its reasoning.
- **"Generated restoration pixels may not be treated as observed glyph
  evidence."** The single most important sentence in the plan.
- **Material as descriptive input and stratification metadata first**, gated
  behind its own paired ablation. Correctly sequenced.

## 8. Smaller notes

- `glyph_match_evidence` should join **`manifest_store.SERVER_OWNED_INSTANCE_FIELDS`**,
  which now exists — the per-instance analogue of the manifest-level tuple. The
  serializer audit test parametrises over the dataclass, so a field added to
  `InstText` without a serializer entry fails a test rather than losing data.
- "Link each observation to the existing candidate-lineage node" — good, and
  note `lineage_candidate_id` only started being persisted this week; before
  that the join did not survive a reload.
- Every learned threshold goes in `docs/threshold-register.md` **with a
  provenance label**. The register's current detection tally is 4 `measured`,
  10 `reasoned`, 3 `inherited`, majority `invented`; a calibrated feature must
  not add to the last column.
- The plan says "Retain v1 as the incumbent encoder". `proof_encoder.py`,
  `proof_calibration.py` and `proof_geometry.py` exist but are **uncommitted and
  untested in my tree**. Freeze and commit the v1 checkpoint before any arm is
  compared against it, or "incumbent" is not a fixed reference.

## Recommended order

1. **Resolve §1 first — it is a decision, not a task.** Corpus size, split, or
   precision target. Everything else is scheduled around the answer.
2. Fix the §2/§3 gate arithmetic in the same pass.
3. Write the §4 failure policy before any certification data is touched.
4. Sequence the zh-Hant review against the outstanding Guided review (§5).
5. Fix the §6 incumbent definition before the first paired run.
6. Then execute §§1–7 of the plan as written. The engineering is sound; it is
   the measurement design that needs the work.
