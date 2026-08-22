# ToFU Vision 2 — supervised distributional objective and channel provenance

Revision 1, 2026-08-16.

Engineering plan for the correction described in the technical-paper revision 2
(`tofu-technical-paper-rev-2.pdf`), its channel-index supplement
(`tofu-channel-index.pdf`), the integration ordering
(`paper-part-integration-plan.pdf`), and the diagnosis that preceded them
(`hist-reprint.pdf`).

`docs/vision-2-fusion-implementation-plan.md` (Phases 0–8) remains the authority
for the surrounding programme; this document is additive and inherits every
invariant there. `docs/vision-2-completion-plan.md` remains the authority for
corpus sizing and statistical certification. Where two documents disagree, the
stricter safety or certification rule wins.

---

## 1. What the papers actually change

Two corrections, in the order the integration plan insists on.

**Measurement first.** The system does not observe a reading through one
immutable channel. It applies a family of decoders — the fixed crop path, the
multipass threshold rungs, zoom, vertical-column and baseline-row merges, the
Paddle rescue — and the reading obtained is `t̂_{e,c} = f_{e,c}(X)`, so the
measurement is `D_{e,c}(X) = NED(f_{e,c}(X), t⋆)`, a property of the
image–channel pair rather than of the scene. Restoring the suppressed index `c`
is a measurement-theoretic correction that must **precede** supervision;
introducing it inside the learning objective would frame it as a modelling
trick.

**Supervision second.** Alignment and candidate selection become one supervised,
distributional objective: softmax distributions over the candidate pool make
cross-entropy definable; NED-graded targets encode *how wrong* a wrong candidate
is; summation over attempts supervises the multi-view machinery that today is
averaged post hoc; a differentiable partial Sinkhorn couples transport to
prediction; and survival gating carries the fail-closed rule into the gradient.

Neither correction replaces the hybrid. Cicerone stays discriminative, Scene
stays contextual and conditions the *candidate representation* rather than the
verdict, Fusion stays responsible for calibrated fail-closed acceptance.

---

## 2. Architectural foci

Seven foci, each stated as the gap between what ships and what the objective
requires. These are the load-bearing decisions; §3 is the sequence.

> **Read as a diagnosis, not as current state.** Everything below describes the
> codebase on **2026-08-16, before any of this work landed**, and is kept in the
> present tense deliberately: it is the record of what was wrong and how it was
> found, which is the part that would be lost by rewriting it. Each focus now
> carries a **Resolved** line saying what actually happened, and §8 has the
> programme-level summary. Where a resolution disagrees with the diagnosis, the
> resolution is what the code does.

### F1 — The measurement channel is not a first-class identifier

**Ships today.** The ingredients exist and are unjoined. `OCRObservation`
(`core/types.py:194`) carries `backend`, `backend_revision`, `pass_tag`.
`okara.DetectorConfig` (`layers/okara.py:113`) fingerprints engine, languages,
pass tag, and four detector thresholds. `recognition_history` records a `stage`
and a `pass`. `_score_region_hypothesis` (`layers/cicerone.py:5446`) synthesises
`detection_pass_{n}`, `selected_existing_pipeline`, `independent_verifier` tags
on the spot.

**Missing.** A closed, registered channel vocabulary; a stable channel
identifier stamped on every read, NED value, candidate distribution, alignment
plan, and survival state; persistence of that identifier through the manifest;
channel-stratified reporting. Every consumer downstream of OCR silently assumes
one channel, which is exactly the assumption the CJK crop confound violates.

**Decision.** A new layer owns the vocabulary and the fingerprint. Existing
tags are *mapped onto* it rather than replaced, so no producer changes its
behaviour in the provenance stage.

**Resolved (S0).** `layers/flight.py` owns the vocabulary and fingerprint;
existing tags are mapped onto it. Channel provenance is stamped at manifest
assembly, persisted as server-owned, and carried into the survival verdict and
retrieval diagnostics. Corpus-wide, 47/47 localized reads name a registered
route — which took fixing a second defect: every transform that mints a fresh
detection was dropping `provenance`.

### F2 — NED is re-implemented per consumer and registered nowhere

**Ships today.** `scripts/eval_detector_evidence.py:224` has a private
`_norm_ed`; `layers/verify.py:1564` has its own two-row Levenshtein;
`scripts/eval_expansion_oracle.py` and `tests/test_error_rates_and_calibration.py`
each carry another. The unreadability convention `NED ≥ 0.8` lives in source
comments and appears in no register row.

**Missing.** One implementation with the explicit empty-string cases the paper
specifies, one registered `τ_unreadable` labelled as an *evaluation convention
and not an acceptance rule*, and a reporting shape that is `D_{e,c}` rather than
a scalar. Soft targets, the channel-regret target, and the evaluation metric all
read from the same function or they measure different things.

**Resolved (S1).** One implementation in `utils/distance.py`; `verify` and the
eval harnesses delegate to it and keep their own normalizations. `τ_unreadable`
registered as `preregistered`. Reporting shape is `D_{e,c}`, enforced by
`flight.measure` taking engine and channel keyword-only and required.

### F3 — There is no probability object, only a ranking

**Ships today.** `proof_runtime._score` (`layers/proof_runtime.py:176-183`)
collects per-(view × font) cosines into `per_text`, takes the **mean** as
`support` and the **variance** as `transformation_variance`, then ranks. The
only calibrated quantity downstream is the top-2 `margin`
(`layers/proof_calibration.py`), so everything below rank 2 is discarded before
any decision sees it.

**Missing.** `p_θ^(n,c)(g_i | X)` — a softmax over the pool at temperature τ, per
attempt and channel. Without it cross-entropy is undefined at training *and*
inference, and calibration cannot act on the distribution's shape.

**Decision.** Rev 2 defines the softmax over a **fused similarity score**
`s_θ^(n,c)`, not over raw cosine (rev 1 used cosine). The head therefore takes a
score function with cosine as its default, so Scene-conditioned or fusion terms
can enter later without a schema change. The head is numpy-only: runtime must
not import torch to produce a distribution.

**Resolved (S3).** `layers/proof_distribution.py`, numpy-free, per-attempt
softmax at τ = 0.07 with consensus, normalized entropy, top-k mass and each
attempt's KL from the consensus. `probability` sits beside `support`; ranking
and every fusion decision are unchanged, and the three new fusion features
enter at zero weight.

### F4 — The only trained objective is NED-blind, attempt-blind, and survival-blind

**Ships today.** `scripts/train_proof_encoder.py:105-116` runs
`supervised_contrastive_loss` over atomic identity labels, full-batch, one
`zero_grad`/`step` per epoch across every image at once. A negative at edit
distance 1 (`0`/`O`) and a negative at distance 7 incur identical penalty. The
multi-view attempts exist only at inference. `layers/decant.py` computes a
survival state that no training code reads.

**Missing.** `L_CE + μ₁R_cons + μ₂R_OT` with `ω(X) = 𝟙[q(X) ∈ {present,
partial}]`; an attempt-level corpus carrying `(region, pool, t⋆, survival,
channel)`; and minibatching, because an attempt-level corpus multiplies the
sample count by views × fonts and the current full-batch loop will not hold it.

**Constraint.** The new objective is added *alongside* the contrastive path and
selected by flag. The incumbent is not replaced before a paired gate.

**Resolved as built, NOT as promoted (S4).** `layers/proof_objective.py` has
all four pieces and minibatching landed in the trainer. It lost its paired
real-ink gate (2/9 against the incumbent's 4/9) and the contrastive path
remains the default. The mechanism was never exercised: on a corpus of single
CJK characters the NED-graded target is 98.9% one-hot. `R_OT` is in the module
but unused — see F5.

### F5 — The alignment layer is orphaned twice over

**Ships today.** `layers/proof_alignment.py` is gradient-free and
parameter-free: cost weights `(0.55, 0.25, 0.20)` at line 49, `ε = 0.08`,
dustbin mass `0.25`, dustbin cost `0.35`, 50 iterations — all hand-set, none
registered, none swept.

**And worse than the paper records.** `align()` returns
`alignment_cost`, `matched_mass`, `observed_unmatched_mass`,
`candidate_unmatched_mass`. `fusion._features` (`layers/fusion.py:84`) reads
`alignment.get("support")` — a key that is never produced. The feature is
therefore *always* `None`, and two tests
(`tests/test_vision2_fusion.py:101`, `tests/test_manifest_store.py:365`) assert
exactly that by expecting `feature_missing:component_alignment_support`. The
transport plan does not merely lack a gradient; its output has never reached a
decision at all.

**Missing.** `C_η = η₁C_pos + η₂C_area + η₃C_aspect` with η loaded from a
registered artifact; a torch twin of the solver for training; `R_OT` weighted by
the model's own candidate distribution; and a fusion feature that is actually
populated.

**Constraint.** The default η artifact must reproduce `(0.55, 0.25, 0.20)`
bit-identically, so the numpy runtime is unchanged until a fit is certified.

**Resolved (S5), and then measured.** `align()` now emits what
`fusion._features` reads, as three measured quantities rather than one invented
composite; `FEATURE_SCHEMA` is v2 so a calibration fitted on the unpopulatable
v1 vector fails closed. η is parameterized behind a registered artifact whose
default is bit-identical, and the torch twin agrees to 1e-9 with gradients
reaching η. The first real ablation: the feature separates (48/66,
p = 0.00029) but **only on alphabetic scripts** — 0 top-1 across all 25 Han,
Hangul and Japanese regions. η stays frozen.

### F6 — Scene conditions nothing, by construction

**Ships today.** `scene._material_evidence` (`layers/scene.py:788`) emits a
`material_class` from a heuristic taxonomy plus `degradation_priors` marked
`plausible_not_measured`; `_normalize_surface_observation` stamps
`calibration_status: "unfitted"`, `decision_weight: 0.0`,
`decision_eligible: False`. `proof.degrade` (`layers/proof.py:114`) implements
eight named processes × three severities, deterministically seeded.
`train_proof_encoder` samples three of them (`blur`, `resample`, `abrasion`) at
`MODERATE`, uniformly, ignoring material entirely.

**Missing.** The indexed operator family `D_{M,𝒫}(R(g_i); z_D)` of eq. (8): a
registered map from material class and inscription process to the subset of
`proof.PROCESSES` and severity that material actually produces. The
`degradation_priors` table is already this map in embryo — `painted_panel →
[fade, abrasion, bleed]`, `masonry → [abrasion, speckle, occlusion]` — it is
simply never consumed.

**Constraint.** Training-time only. Unknown or uncalibrated Scene evidence
degrades to the unconditioned operator set; it never vetoes, never manufactures
a corruption hypothesis, and generated pixels never re-enter as observations.

**Resolved as built, inert in practice (S6).** `layers/marinade.py` registers
the operator family and imports Scene's own priors table rather than restating
it. It conditions **0 of 66** regions: 16 land on no surface, 50 on a surface
classified `unknown`, and all 50 also carry `calibration_status: "unfitted"`.
No training arm was built on a path that provably cannot execute.

### F7 — The oracle channel is a training instrument, and nothing enforces that

**Ships today.** Nothing computes `D⋆(X) = min_c D_{e,c}(X)`, so there is
nothing to leak — but there is also no guard, and the programme has already
recorded one invalid oracle arm (Gate 2's 42.3%).

**Missing.** The envelope as a *benchmark*; a learned reliability model
`r_φ(c|X)` and the mixture `p̂ = Σ_c r_φ(c|X) p̄_θ^(c)` that replaces the hard
minimum at inference; channel-aware survival `ω(X,c)`; the acceptance rule
`max_c r_φ(c|X) ≥ δ_r ∧ q_s(X, ĉ) ∈ {present, partial}`; and an executable guard
that makes `D_{e,c}` unreachable under a non-training configuration — the same
class of protection as the existing no-ground-truth-length-leakage rule.

**Consistency is licensed, not assumed.** `R_cons` may only tie channels inside
a preregistered Blackwell-equivalence class for that stratum. A crop and a
multi-pass pipeline that are not equivalent on CJK must be *allowed* to
disagree; their disagreement is evidence about channel quality.

**Resolved (S2).** `flight.oracle_envelope` takes `context` keyword-only with
no default and raises `OracleLeak` outside training, benchmarking and error
analysis; a test asserts no module in the shipped package references it at all.
Consistency is licensed only inside certified-equivalent classes — and S2
certified none, so `R_cons` ties nothing across channels.

---

## 3. Naming

New modules follow the house convention (`docs` and every layer header): a
tofu-making or culinary pun, with a docstring that explains the pun and the
boundary.

| Module | Pun | Owns |
|---|---|---|
| `layers/flight.py` | a tasting flight — the same pour served several ways, side by side, so they can be compared | channel vocabulary, channel id, `D_{e,c}` reporting, reliability model `r_φ`, channel-aware gate |
| `layers/proof_distribution.py` | stays in the established `proof_*` family (bread proofing / proof of identity) | numpy softmax head, τ, per-attempt distributions, entropy features |
| `layers/proof_objective.py` | same family | torch-only: NED soft targets, `L_CE`, `R_cons`, `R_OT`, differentiable partial Sinkhorn, survival weight |

`normalized_edit_distance` goes into a new `utils/distance.py`, unpunned like
the rest of `utils/`. **Revised during S0/S1 from an earlier draft that put it
in `utils/textmatch.py`:** that module's docstring records a deliberate decision
to keep TM matching separate from `verify`'s scorer so the two can diverge, and
hanging a shared primitive off it would contradict a design note that is still
correct. `textmatch` is unchanged.

---

## 4. Staged implementation

Stages S0–S7. Each ends with a promotion gate; a failed gate leaves the work
available for experimentation but prevents it from influencing the next stage's
production state. Stage format matches the Phase format in the fusion plan.

**Status, 2026-08-16.** S0, S1, S2 and S3 are implemented (`layers/flight.py`,
`utils/distance.py`, `scripts/eval_channel_envelope.py`,
`layers/proof_distribution.py`). S1's exit gate is met by equivalence tests on
the delegation rather than by re-running the corpus reports — the refactor is
proven identical function-by-function, and the baseline eval has not been
re-executed against fixtures.

**S3 is complete, and its prerequisite was closed first.** Every transform that
emits a fresh detection was dropping `provenance`, so 9 of 47 localized reads
— all CJK — could not name their channel. `cicerone._tag_transform_read` now
records the read wherever the lineage node is recorded. Corpus-wide route
coverage went 38/47 → **47/47 with every region's NED unchanged**, which is
what makes it a provenance fix and not a behaviour change.

The softmax head then landed on top: per-attempt distributions over the
candidate pool at τ = 0.07, summarized to a consensus distribution, normalized
entropy, top-k mass, and each attempt's KL divergence from the consensus — the
λ₃ consistency quantity, now measured at inference before it becomes a training
signal. `probability` sits beside `support` on every candidate record;
`support`, `margin`, the rank order and every fusion decision are unchanged,
and the three new fusion features are declared at zero weight.

**S2 is complete and its verdict changes the plan.** See
`docs/channel-envelope-report.md` and `evidence/channel-envelope-v1.json`.

- **S7 is deferred.** The envelope is near-flat: 0.019 NED of selectable
  headroom over 47 localized regions, and exactly zero on six of seven script
  strata. Latin's apparent 0.033 is a localization artifact (0.0015 once
  unlocalized regions are excluded). Only Korean survives, at 0.100 from n=6
  — below our own preregistered power floor.
- **S4's cross-channel consistency term is not licensed.** No stratum is
  certified equivalent. `R_cons` is restricted to attempt variation *within* a
  channel — the blur, morphological and typeface variants of one observation,
  which is what the originally primed λ₃ term was about.
- **S3 had a prerequisite, now closed.** 9 of 47 localized reads (19%, biased
  entirely toward CJK) carried no traceable route. The cause was not untagged
  detector calls but transforms dropping provenance when they mint a fresh
  detection; see §S3 status above.

**S4 is implemented and did not pass its gate.** `layers/proof_objective.py`
ships the NED-graded target, attempt-summed cross-entropy, the licensed
consistency term, and the survival gate; `train_proof_encoder.py
--objective distributional` trains it in minibatches. All four pieces behave as
specified and are tested — the survival gate produces *exactly* zero gradient
from `absent`/`weak`, σ→0 recovers hard cross-entropy, and `R_cons` refuses to
tie channels S2 did not certify.

On the frozen real-ink corpus it scores **2/9 against the incumbent's 4/9**
(3 lost, 1 gained; sign test p = 0.625). The exit gate asked for +5 paired
top-1 at p < 0.05; the arm is directionally worse and n = 9 cannot resolve it
either way. **The contrastive objective remains the default.**

The diagnosis matters more than the number. The only encoder corpus is 124
single CJK characters, so **98.7% of non-truth pool members sit at NED exactly
1.0** and the graded target is 98.9% one-hot. The mechanism the objective
exists to add never engaged: it degenerated to hard cross-entropy over a random
pool, plus a consistency term. That is a corpus limitation, and it cannot be
fixed by moving σ — moving σ on a one-hot target changes nothing.

**Prerequisite for retrying S4:** a training corpus of multi-character
readings, where NED between pool members is not almost always 1. The attempt
corpus builder over real regions (S4 work item 1) is the path to it and is not
yet built; the synthetic path was implemented first because it is what could be
run against the existing frozen artifacts.

**S5 is implemented; the feature is repaired and measured, η stays frozen.**
The orphaned feature is fixed — `align()` now emits what `fusion._features`
reads, as three measured quantities (`component_alignment_cost`,
`_matched_mass`, `_unmatched`) rather than one invented composite, and
`FEATURE_SCHEMA` is bumped to v2 so a calibration fitted on the unpopulatable
v1 vector fails closed. The cost is parameterized behind a registered artifact
whose default is bit-identical to the shipped weights, and
`proof_objective.partial_sinkhorn_torch` is a differentiable twin held to
agreeing with the numpy solver to 1e-9, with gradients reaching η.

The ablation that Phase 5 claimed and could not have run now exists
(`evidence/alignment-ablation-frozen-v1.json`, 66 regions):

- **The feature separates**: truth cheaper than the average wrong candidate on
  48/66, exact sign test **p = 0.00029**.
- **The signal is entirely alphabetic**: Latin 26/31 (p = 0.0002), Cyrillic 6/6
  (p = 0.031), and **0 top-1 across all 25 Han, Hangul and Japanese regions**.
- **No incremental gain over the encoder**: 12 rescues against 10 breaks,
  p = 0.83.

Per the exit gate, that is not a statistically supported incremental gain, so
alignment stays diagnostic-only and enters the feature vector at zero weight
until a calibration is fitted against it. η is **not** fitted: three parameters
against the corpus they would be evaluated on is the `MAX_ZOOM_INFLATION`
failure, and more fundamentally a re-weighting cannot repair CJK — the
component features are connected components, a Han glyph is many radicals and
strokes, and `MAX_COMPONENTS = 64` truncates them. The failure is in the
features, not their weights.

**S6 is implemented and inert.** `layers/marinade.py` registers the operator
family `D_{M,P}` — Scene's own `DEGRADATION_PRIORS` table, imported rather than
restated, plus an inscription-process axis and the fallback. It conditions
nothing: across the 66 annotated regions of the frozen corpus, **0 clear the
bar** (`evidence/scene-conditioning-coverage-v1.json`).

Three independent gates close, and the artifact counts each:

- 16 regions land on no scene surface at all;
- 50 land on a surface Scene classified `unknown`;
- all 50 of those also carry `calibration_status: "unfitted"` and
  `substrate_trust: "unmeasured"`.

The calibration gate alone would close all 66 even with perfect material
classification, because uncalibrated Scene evidence is *missing* evidence by
standing rule. **No training arm was built**: an arm whose conditioning path
provably cannot execute would measure the fallback, not the hypothesis, and
report the result as though it had tested the idea.

The prerequisite is reviewed material labels and a fitted classifier — the
material review packet and `scripts/ingest_scene_material_reviews.py` already
exist for exactly this — not more code.

### S0 — Channel index as provenance

**Purpose:** make the measurement experiment visible before anything learns from
it. No behaviour changes.

**Work**

1. `layers/flight.py`: a frozen `Channel` record (engine, route, view,
   transform sequence, crop geometry, merge state) with a stable
   `fingerprint()` in the style of `okara.DetectorConfig.fingerprint`.
2. Register the closed route vocabulary from what the pipeline already runs:
   `full_frame`, `detection_pass_{0,1,2}`, `zoom`, `vertical_column_merge`,
   `baseline_row_merge`, `paddle_rescue`, `region_crop` (the
   `detect_in_regions` path), `guided_block`, `independent_verifier`.
   Unrecognised routes map to `unregistered` and are reported, never dropped.
3. Stamp the channel id on `OCRObservation`, `recognition_history` entries,
   `EvidenceSurvival`, `GlyphMatchEvidence.diagnostics`, and any alignment
   record.
4. Persist it: add the fields to the server-owned declarations *and* the
   explicit serializer/deserializer in `utils/manifest_store.py`.
5. Report it: channel counts and per-channel coverage in the vision2 metrics
   endpoint.

**Conflict protection**

- No detector, recognizer, or merge behaviour changes; this stage only labels.
- Unknown channels are `unregistered`, never `unknown`-as-absent.
- The frontend cannot own or clear channel fields.
- **Stamping is not gated on the Vision 2 state.** The incumbent arm is the
  baseline every channel comparison is measured against, so an index that
  existed only when Vision 2 was on could never say what the shipped pipeline
  measured. The cost is that `off` no longer produces byte-identical serialized
  output; `channel` and `channel_id` are registered here as the explicitly
  excluded provenance fields, on the same footing as the timing fields Phase 0
  already excludes.
- Stamping happens at manifest **assembly**, not only in `detect()`. The first
  cut stamped in `detect()` alone and left every `build_manifest` caller — the
  streaming endpoint, every harness — unstamped while the unit tests passed.

**Regression tests**

- Frozen deterministic fixtures produce byte-equivalent detection output apart
  from the registered new provenance fields.
- Manifest round trip: populated, absent, partial, and unregistered channels
  survive serialize → deserialize → serialize. Repeated autosave, region edit,
  add, delete, undo, export, and import retain the channel — this is the failure
  mode `docs` already records for server-owned state.
- Every read in a frozen fixture carries a channel id; every `OCRObservation`
  channel maps to a registered route or to `unregistered`.
- Legacy manifests without channel fields load as `not_evaluated`.

**Exit gate**

- Zero output drift on frozen fixtures. Channel coverage 100% of reads.

### S1 — One NED, registered

**Purpose:** one measurement function, so targets, metrics, and channel regret
cannot silently diverge.

**Work**

1. `utils/textmatch.normalized_edit_distance(a, b)` implementing eq. (6)
   exactly, including both empty-string cases.
2. Refactor `eval_detector_evidence._norm_ed`, `verify.py`'s Levenshtein,
   `eval_expansion_oracle`, and the test helper onto it.
3. Register `TAU_UNREADABLE = 0.8` as `preregistered`, annotated in
   `docs/threshold-register.md` as an evaluation convention that is *not* a
   calibrated probability and *not* an acceptance rule.
4. Change the reporting shape from `NED` to `D_{e,c}`: every emitted NED value
   carries its engine and channel.

**Conflict protection**

- Pure refactor. Any numeric change on a frozen fixture is a bug, not an
  improvement.

**Regression tests**

- Recomputation of every committed baseline NED figure is identical to the
  digit.
- Property tests for the empty-string cases and the `[0,1]` range.
- A NED value cannot be emitted without an engine and channel.

**Exit gate**

- Byte-identical baseline reports, plus the new channel columns.

### S2 — The channel envelope (honest measurement, no learning)

**Purpose:** find out whether the channel family contains a materially better
measurement per stratum. This is a real decision gate: if the envelope is flat,
S7 is not worth building.

**Work**

1. `scripts/eval_channel_envelope.py`: for each corpus region, run the
   registered channel family, record `D_{e,c}`, compute `D⋆(X)` and `c⋆(X)`.
2. Report per-stratum channel curves and the oracle envelope **separately**,
   both labelled oracle/training-only.
3. Preregister Blackwell-equivalence classes per stratum on the validation
   split: which channels may be tied by `R_cons` in S4, and which may not.
4. Record the crop-versus-pipeline gap on CJK strata explicitly — the confound
   that motivates the whole index.

**Conflict protection**

- The report may not present a crop score as a scene-level property.
- The envelope is never written into a manifest, a feature vector, or a
  calibration artifact.

**Regression tests**

- The envelope helper refuses to run when the configuration is not a training or
  benchmarking configuration.
- Equivalence classes are derived from the validation split only and are
  content-derived, not filename-derived.

**Exit gate**

- Report committed with per-stratum channel curves, envelope, and equivalence
  classes. A materially non-flat envelope on at least one preregistered hard
  stratum licenses S7; a flat envelope defers it.

### S3 — A distribution at inference

**Purpose:** create the probability object. No training change, no decision
change.

**Work**

1. `layers/proof_distribution.py`: numpy softmax over the pool at temperature
   τ, taking a score function (cosine by default) so a fused `s_θ` can be
   substituted later.
2. `proof_runtime._score` retains the per-(view × font) matrix and emits a
   distribution per attempt and channel, plus the mean distribution. `support`,
   `margin`, `transformation_variance`, and the resulting rank order are
   untouched.
3. New fields on `GlyphCandidateEvidence` (`probability`) and
   `GlyphMatchEvidence` (distribution revision, entropy, mass on top-k).
4. Declare the distribution features in the fusion feature schema at **zero
   weight**, the same discipline observational Scene received.
5. Register τ as `invented` until fitted.

**Conflict protection**

- Torch is not imported on this path.
- Ranking and every fusion decision are identical to the incumbent with the new
  fields present.

**Regression tests**

- Decision-equality test across the full fixture set: incumbent versus
  distribution-enabled arm produce identical decisions and reason codes.
- Distribution sums to 1, is invariant to candidate ordering, and is
  deterministic under pinned seeds, checkpoint, pool, and fonts.
- Degenerate pools (size 1, identical scores) do not produce NaN.

**Exit gate**

- Zero decision drift; τ registered; distribution persisted and round-tripped.

### S4 — The supervised objective

**Purpose:** train against graded targets, supervised attempts, and a survival
gate.

**Work**

1. `scripts/build_proof_attempt_corpus.py` (**not built** — the synthetic path
   was implemented first because it runs against existing frozen artifacts;
   this is the prerequisite for retrying the S4 gate): attempt-level rows
   carrying region, candidate pool and its revision, `t⋆`, survival state,
   channel, and the full-pipeline read where the stratum demands it.
2. `layers/proof_objective.py` (torch, training-only):
   `ned_soft_target` (eq. 10), `attempt_cross_entropy` (eq. 11),
   `consistency_kl` (eq. 12) **restricted to the S2 equivalence classes**,
   `survival_weight` (eq. 15).
3. `train_proof_encoder.py --objective distributional`, alongside the default
   `contrastive`. Add minibatching; the current full-batch loop cannot hold an
   attempt-level corpus.
4. Register τ, σ, μ₁, μ₂ with provenance.

**Conflict protection**

- The contrastive path remains the default and the incumbent.
- `t⋆` enters target construction only. The candidate generator never observes
  the answer or its length.
- `absent`/`weak` instances contribute no gradient — asserted, not assumed.
- CJK attempt labels come from the full-pipeline read, or the targets inherit
  the crop's measurement bias.

**Regression tests**

- Leakage: extend the existing shuffle/sentinel tests so a poisoned `t⋆` cannot
  change candidate generation; assert `q_i` is unreachable from the inference
  path.
- Survival gate: an `absent`/`weak` batch produces exactly zero gradient norm.
- `σ → 0` recovers hard cross-entropy when an exact match is in the pool;
  a candidate at NED 1 receives near-zero target mass.
- `R_cons` refuses to tie channels outside their equivalence class.
- Determinism under pinned seeds; schema-1 checkpoints still load.

**Exit gate**

- Paired real-ink top-1 improvement over the incumbent encoder on the
  preregistered strata, exact one-sided McNemar `p < 0.05`, with no marginal
  stratum losing more than the registered tolerance. Synthetic-only improvement
  fails the gate.

### S5 — Fitted alignment cost, coupled transport

**Purpose:** promote Sinkhorn from an unread diagnostic to a fitted, coupled
regularizer — and connect its output to a decision at all.

**Work**

1. Fix the orphaned feature: `align()` emits the value `fusion._features`
   actually reads, or `_features` reads the value `align()` actually emits.
   Update the two tests that currently assert its absence.
2. Parameterize `_cost_matrix` with η from a registered artifact; the default
   artifact reproduces `(0.55, 0.25, 0.20)` exactly.
3. `proof_objective.partial_sinkhorn_torch` + `transport_regularizer` (eq. 14),
   fitted jointly in S4 training.
4. Write `alignment-cost-vN.json` with provenance; flip the register rows from
   `invented` to `measured` only on a certified fit.

**Conflict protection**

- The numpy runtime stays torch-free and bit-identical under the default
  artifact.
- Alignment still cannot independently recommend a candidate; solver failure,
  timeout, or degenerate masks produce *missing* evidence, not a negative
  identity verdict.
- No η enters production calibration until the calibration artifact revision
  includes it.

**Regression tests**

- Default artifact reproduces current alignment costs to the digit on the split,
  merged, missing, spurious, and detached-mark fixtures.
- Permutation invariance of component ordering; empty and single-component
  masks; excessive component counts; bounded deterministic runtime.
- Gradient flows to η through the scaling iterations (finite-difference check).
- Ablation shows no loss on intact glyphs beyond the registered tolerance.

**Exit gate**

- Statistically supported incremental gain on damaged-segmentation strata.
  Otherwise η stays frozen at incumbent values and the attempt is recorded in
  `docs/measured-dead-ends.md`.

### S6 — Scene-conditioned corruption (training-time only)

**Purpose:** make the gradient carry the *reason* for the corruption, not only
its cost.

**Work**

1. Promote `scene._material_evidence`'s `degradation_priors` into a registered,
   revisioned operator family: `(material_class, inscription_process) →
   subset of proof.PROCESSES × severity`, consumed by the training corpus
   builder as `D_{M,𝒫}(R(g_i); z_D)`.
2. Feed measured degradation evidence `z_D` where the surface has it; degrade to
   the unconditioned eight-process set where it does not.
3. Version the family; record which family revision produced each training row.

**Conflict protection**

- Training-time only. Generated pixels are tagged synthetic and cannot enter
  observed-evidence APIs.
- Unknown or uncalibrated material is *missing* evidence, not weak evidence: it
  selects the unconditioned set and never vetoes.
- No Cleanse provider, inpainting strategy, or Scene decision weight changes.

**Regression tests**

- Unknown material yields exactly the unconditioned operator set.
- A synthetic corruption cannot reach an observed-evidence API (existing Phase 7
  rule, extended to this path).
- Operator family revision is recorded on every training row and mismatch is
  detected.

**Exit gate**

- Paired gain on physically-degraded strata over the S4 arm. Otherwise Scene
  conditioning stays observational and the result is registered as a dead end.

### S7 — Learned channel selection and a channel-aware gate

**Purpose:** replace the oracle minimum at deployment without weakening
fail-closed behaviour. Gated on a non-flat S2 envelope.

**Work**

1. `flight.reliability`: `r_φ(c|X)` trained on the channel-regret target
   `ρ_c(X)` (eq. 9) from the **training split only**; register κ.
2. Inference mixture `p̂_{θ,φ}(g_i|X) = Σ_c r_φ(c|X) p̄_θ^(c)(g_i|X)`, preferred
   over a hard minimum because it preserves uncertainty when two channels are
   plausible.
3. Channel-aware survival `ω(X,c)` and the acceptance rule
   `max_c r_φ(c|X) ≥ δ_r ∧ q_s(X, ĉ) ∈ {present, partial}`; register δ_r.
4. Report channel-specific performance, the oracle envelope, and learned
   selector performance as three separate verdicts.

**Conflict protection**

- An executable guard makes `D_{e,c}` and `D⋆` unreachable under a non-training
  configuration; a test asserts the oracle path raises when called at inference.
- A better channel may reduce review load; no channel may manufacture
  acceptance when identity-bearing evidence is absent.
- Candidate elimination alone still cannot produce acceptance.

**Regression tests**

- Oracle-leakage test: the selector's inputs contain no function of `t⋆`.
- Selector output is a distribution over registered channels only.
- Survival `absent`/`weak` on the selected channel forces abstention regardless
  of `r_φ`.
- Selector performance is reported below the envelope; a selector that matches
  the envelope exactly is treated as a leak, not a result.

**Exit gate**

- Paired improvement over the fixed-channel incumbent, strictly below the oracle
  envelope, with review-load reduction reported separately from precision.

---

## 5. Constants introduced

All rows land in `docs/threshold-register.md` at the stage that introduces them,
labelled `invented` or `preregistered` until a certified fit moves them to
`measured`. Production gates may not use undocumented constants.

| Constant | Meaning | Introduced | Initial label |
|---|---|---|---|
| `τ` | softmax temperature over the candidate pool | S3 | `invented` |
| `τ_unreadable` = 0.8 | evaluation-only unreadability convention | S1 | `preregistered` |
| `σ` | NED soft-target temperature | S4 | `invented` |
| `μ₁`, `μ₂` | consistency and transport regularizer weights | S4/S5 | `invented` |
| `η₁, η₂, η₃` | alignment cost weights (currently 0.55/0.25/0.20) | S5 | `invented` → artifact |
| `κ` | channel-regret temperature | S7 | `invented` |
| `δ_r` | minimum channel reliability for acceptance | S7 | `preregistered` |

---

## 6. Invariants carried forward

Inherited from the fusion plan and the papers; none may be weakened.

- **No target leakage.** `q_i` is built from `t⋆` at training time only. The
  candidate generator never observes the answer or its length.
- **No oracle at inference.** `D⋆` is a training and benchmarking envelope. The
  guard is executable, not documentary.
- **Recommendation-only.** A trained `p_θ` ranks and calibrates. Acceptance
  remains with the fusion cascade and review semantics.
- **Generated pixels are hypotheses.** Scene-conditioned corruptions are
  training-time renderings of candidates, never observations.
- **Fail-closed in the gradient.** `absent`/`weak` instances contribute no
  recognition gradient, on the same grounds that they cannot be recommended.
- **Measurement honesty.** On CJK strata a per-crop read understates the shipped
  pipeline; attempt labels come from the full-pipeline read where the strata
  demand it. A crop score is never reported as a scene-level property.
- **Paired certification.** Every improvement claim is a paired, preregistered
  comparison against the shipped incumbent, with significance and margin
  reported as independent verdicts.
- **Torch stays optional.** No runtime path imports torch to produce a
  distribution, a channel id, or an alignment cost.

---

## 7. Known risks

1. **Corpus size is the binding constraint.** `L_CE` needs
   `(region, pool, t⋆, survival, channel)` rows, and the guided-corpus holdout
   is already recorded as underpowered. If the sample cannot support fitted τ
   and σ, the honest outcome is registered-not-fitted constants and no
   promotion — not a fit on the certification set.
2. **The envelope may be flat.** S2 can show the channel family contains no
   materially better route on any stratum. That defers S7 and is a result worth
   recording, not a failure to route around.
3. **Full-batch training will not survive attempt-level data.** Minibatching in
   S4 is not optional; the current loop stacks every image into one tensor.
4. **The orphaned alignment feature means Phase 5's ablation never measured
   what it claimed.** Any prior conclusion about component alignment's fusion
   value should be treated as unmeasured until S5 fixes the key mismatch and
   re-runs it.
5. **Equivalence classes are a judgement with teeth.** Tying a crop to a
   multi-pass pipeline inside `R_cons` on a stratum where they are not
   equivalent trains the encoder to agree with the weaker channel.

---

## 8. Where the programme stands

Every stage is now either implemented or closed by measurement. Four of the six
were closed by measuring rather than by building, which is the outcome the
gates existed to produce:

| Stage | State | What decided it |
|---|---|---|
| S0 channel provenance | **shipped** | 47/47 localized reads name a route |
| S1 registered NED | **shipped** | one implementation, τ_unreadable registered |
| S2 channel envelope | **shipped, S7 deferred** | 0.019 NED headroom; envelope near-flat |
| S3 distribution head | **shipped, zero-weight** | decisions provably unchanged |
| S4 supervised objective | **built, gate not met** | 2/9 vs 4/9; target 98.9% one-hot |
| S5 alignment | **repaired, η frozen** | separates on alphabets, 0/25 on CJK |
| S6 Scene conditioning | **built, inert** | 0/66 regions conditionable |
| S7 channel selector | **not built** | deferred by S2's envelope |

Three distinct corpus gaps now block the remaining work, and they are not the
same gap:

1. **Multi-character readings** for S4 — the encoder corpus is 124 single CJK
   characters, so NED-graded targets degenerate to one-hot.
2. **A held-out set separate from the measurement set** for S5's η — three
   parameters fitted on the 66 regions they would be scored against is the
   `MAX_ZOOM_INFLATION` failure verbatim.
3. **Reviewed material labels and a calibrated classifier** for S6.

None is closed by writing more code, and each is cheap to re-test: the
harnesses (`eval_channel_envelope`, `eval_alignment_ablation`,
`eval_scene_conditioning`) re-run against a new corpus and say immediately
whether the stage has become viable.
