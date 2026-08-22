# Threshold register

Every hard-coded numeric constant in the detection path, with an honest label
for where its value came from.

**Why this exists.** The suspicion that hard thresholds are driving detection
outcomes is correct in the sense that matters: there are roughly forty of them,
they interact, and until now nothing recorded which ones were *measured* and
which were simply chosen. That distinction is the whole content of this file.
A number that was fitted on evidence and a number somebody picked because it
looked reasonable are indistinguishable in source, and they warrant completely
different amounts of trust when a result disappoints.

**Provenance labels**, exactly one per row:

| Label | Meaning |
|---|---|
| `measured` | A recorded sweep or ablation exists; the row cites it. |
| `reasoned` | Derived from a stated observation about the data (e.g. "every GT region is ≥500px²"), but not swept. |
| `inherited` | A library default carried forward without examination. |
| `invented` | Chosen by judgement. No sweep, no cited observation. |

**The rule this file enforces.** No constant moves without held-out evidence,
or an explicit note in the changelog that it has none. This project has already
paid for the alternative twice: `MAX_ZOOM_INFLATION` was fitted on a single
fixture, shipped, and regressed the corpus (0.738 → 0.706); `TWIN_FLOOR` was set
from three points. Both are recorded in `docs/measured-dead-ends.md`.

**What this file does not do.** It changes no value. Part A of the guided-capture
plan is instrumentation and honesty; retuning is separate work that needs the
attribution in `docs/detection-attribution-report.md` to say which constants are
actually killing candidates before anything is swept.

---

## Detection — `src/tofu/layers/cicerone.py`

| Constant | Value | Provenance | Basis / note |
|---|---:|---|---|
| `PASS_THRESHOLDS` | 3 rungs | `invented` | The multipass ladder. Never swept as a set; `link_threshold` alone was swept and found **inert** (mean IoU identical to 3 d.p. across 0.10–0.70). |
| `LATIN_POOL_MIN_CONFIDENCE` | 0.5 | `invented` | — |
| `LATIN_POOL_RESCUE_MIN_CONFIDENCE` | 0.35 | `invented` | — |
| `MIN_DIACRITIC_WORD_LEN` | 3 | `reasoned` | Short tokens carry too little evidence to accuse of a language. |
| `DECLARED_LANGUAGE_MARGIN` | 2.0 | `invented` | — |
| `SCENE_FILTER_VETOES` | off | **`measured`** | Veto removal: 0.710 → 0.738 mean IoU, 47 → 51 matched, at 99 → 134 candidates and review rate 0.444 → 0.622 on gemini-street. A deliberate recall-for-reviewer-time trade. |
| `COLUMN_MAX_ASPECT` | 1.6 | `reasoned` | Members must be char-like/tall, not wide lines. |
| `COLUMN_X_ALIGN` | 0.5 | `invented` | Relaxing the column gates was swept and was **monotonically worse** (0.365 / 0.362 / 0.332 vs 0.381 shipped), so the gate is not under-merging — but the specific value was not fitted. |
| `COLUMN_WIDTH_RATIO` | 1.7 | `invented` | as above |
| `COLUMN_MAX_GAP` | 0.8 | `invented` | as above |
| `COLUMN_COLOR_MAX_DIST` | 90.0 | `invented` | Colour gate off scored 0.365 vs 0.381; the threshold itself unswept. |
| `COLUMN_LATIN_MIN_CONF` / `_MIN_LEN` | 0.5 / 2 | `invented` | — |
| `VERTICAL_STACK_MIN_ASPECT` | 3.0 | `reasoned` | Height ≥ 3× width is too tall for one glyph. |
| `MIN_BAND_HEIGHT_PX` | 12 | `reasoned` | Below this a band cannot hold a legible glyph. |
| `ROW_BASELINE_ALIGN` | 0.5 | `invented` | — |
| `ROW_HEIGHT_RATIO` | 1.7 | `invented` | — |
| `ROW_MAX_GAP` | 0.6 | **`measured`**, narrow | Source note records the separating cases and a margin of only ~13% either side. Flagged: a layout with tighter columns or looser word spacing than the corpus defeats it. |
| `ROW_MIN_MEMBERS` | 2 | `reasoned` | A row of one is not a row. |
| `ROW_MERGE_MIN_CONFIDENCE` | 0.5 | `invented` | — |
| `MIN_DISAMBIGUATION_EVIDENCE` | 2 | `reasoned` | One hit flipped a scene on "ET"; the second hit is the guard. |
| `MIN_KANA_CONFIDENCE` | 0.3 | `reasoned` | A near-zero-confidence misread of a blurry crop can shape-match a kana glyph and must not be a definitive Japanese signal. |
| `MIN_SYMBOL_JUNK_AREA` | 450 | **`measured`** | Every GT region across the fixtures is ≥500px²; measured false positives were 100px² and 440px². Fitted on the *evaluation* corpus — no held-out evidence. |
| `SURFACE_PROBE_MAX` | 8 | `invented` | Runtime cap. |
| `PADDLE_RESCUE_CONF_FLOOR` | 0.6 | `invented` | — |
| `PADDLE_OVERLAP_WIN_FLOOR` | 0.5 | `invented` | — |
| `PADDLE_ESCALATED_DROP_SCORE` | 0.15 | `inherited` | Below DBNet's default, deliberately; the specific value is Paddle-side convention. |
| `PADDLE_ESCALATED_UNCLIP_RATIO` | 1.9 | `inherited` | Source note says DBNet's default under-clipped. |
| `SURFACE_COVERAGE_FLOOR` | 0.85 | `invented` | Truncation heuristic against a surface's dominant-axis extent. |
| `SUBDIVIDE_MAX_DIM` | 120 | `invented` | — |
| `SUBDIVIDE_OVERLAP_PX` | 10 | `reasoned` | So a sign spanning a tile edge is not cut. |
| `SUBDIVIDE_SPLIT_TOLERANCE` | 1.5 | `invented` | — |
| `LOAF_CONTAINMENT` | 0.8 | `reasoned` | Deliberately below `ZOOM_FRAGMENT_CONTAINMENT`'s 0.9: a zoom box maps back from an upscaled crop, and at 0.9 the rounding disowns a genuine crumb of its own line. |
| `LOAF_CRUMB_MIN_CONFIDENCE` | 0.3 | `invented` | — |
| `LOAF_SPAN_SIMILARITY` | 0.6 | `invented` | — |
| `LOAF_DUPLICATE_SIMILARITY` | 0.9 | `invented` | — |
| `LOAF_DUPLICATE_RESCUE_CEILING` | 0.5 | `invented` | — |
| `ZOOM_MAX_SURFACE_FRAC` | 0.5 | `reasoned` | A surface larger than half the frame *is* the frame; re-detecting buys nothing. |
| `ZOOM_MAX_SURFACES` | 14 | `invented` | Runtime cap against pathological surface counts. |
| `ZOOM_SCALE` / `ZOOM_PAD` | 2 / 8 | `invented` | — |
| `ZOOM_FRAGMENT_CONTAINMENT` | 0.9 | `invented` | — |
| `MAX_ZOOM_INFLATION` | *removed* | **`measured`** | Fitted on one fixture, shipped, **regressed the corpus** 0.738 → 0.706, matched 51 → 48. No threshold window exists. The cautionary case for this whole file. |
| `EDGE_RESCUE_PAD_RATIO` | 0.16 | `reasoned` | A blanket pad merged `AVENUE` with `de la RÉPUBLIQUE`, halving recall on that fixture; the per-region evidence gate is the fix. |
| `EDGE_RESCUE_CONF_SLACK` | 0.10 | `invented` | — |
| `FRAGMENT_OVERLAP` | 0.5 | `invented` | — |
| `mag_ratio` | 1.0 (default) | **`measured`** | 1.5–3.0 swept: japan-street −0.143, la-bastille −0.061, ~1.8× wall clock, curve non-monotonic. Rejected. |
| `rotation_info` | unset | **`measured`** | `[90,270]` gave zero benefit, twice. |

## Evidence survival — `src/tofu/layers/decant.py`

Added 2026-08-11 with Vision 2 Track A. Registered on creation rather than
retrospectively, which is the point of this file.

| Constant | Value | Provenance | Basis / note |
|---|---:|---|---|
| `LEGIBLE_GLYPH_PX` | 12.0 | `reasoned` | Two observations agree here and neither was swept: `EasyOCRBackend.MIN_CROP_HEIGHT` is 40px because reads degrade below it, and `la-bastille`'s annotation note records ±3px box error at 13–14px line height — error large relative to the box. **Not swept.** |
| `LOW_CONFIDENCE` | 0.3 | `invented` | Deliberately may only ever contribute a *reason* and downgrade to `partial`, never decide a state alone: cicerone records a correctly-scripted CJK read at 0.015 against a wrong-charset garbage read at 0.087, so confidence is not trusted to condemn. |
| `NO_ACTIVATION` | 0.10 | `measured` | Inherited from `eval_detector_evidence`, where it is set below the pipeline's loosest `low_text` so stone grain is not counted as activation. |
| `PROPOSAL_FLOOR` | 0.20 | `measured` | `PASS_THRESHOLDS[-1]`'s `low_text` — the loosest rung the pipeline ever runs. A peak below it was never going to form a proposal however anything else is tuned. |

No weights, and deliberately no scalar: `decant` returns a state plus the
measurements behind it. Combining these into a score is the ranker's job, and
`layers/ticket.py` records why the ranker cannot be priced yet.

## Measurement — `src/tofu/utils/distance.py`

Added 2026-08-16 with the supervised-objective work (stage S1 of
`docs/vision-2-supervised-objective-plan.md`). The value is not new; its
registration is. It had lived in a source comment in
`scripts/eval_detector_evidence.py` and in the prose of
`docs/vision-2-assessment.md`, which is precisely the situation this file
exists to end.

| Constant | Value | Provenance | Basis / note |
|---|---:|---|---|
| `TAU_UNREADABLE` | 0.8 | `preregistered` | The NED at or above which a region is scored unreadable **from a crop already known to be geometrically valid**. An evaluation convention, not a calibrated probability and **not an acceptance rule**: a production gate still requires surviving evidence and may abstain far below it. Never swept — it is a reporting cut, and moving it would silently restate every published unreadability count. |

NED itself is a definition rather than a threshold, but two properties of it
are load-bearing and belong on the record: it normalizes by the **longer**
string (symmetric, so it can score candidate-against-candidate as readily as
reading-against-truth), and both empty-string cases are explicit (a missed
detection and an empty OCR output are the observations that would otherwise
divide by zero). `verify.error_rates` keeps its own normalization — CER
divides by the reference and clamps — and shares only the primitive.

## Measurement channel — `src/tofu/layers/flight.py`

Added 2026-08-16 with stage S0. No thresholds: the layer names routes and
fingerprints them, and a vocabulary is not a cut point. One registered
constant of judgement all the same.

| Constant | Value | Provenance | Basis / note |
|---|---:|---|---|
| crop fingerprint granularity | 4 px | `reasoned` | Crop geometry is part of the channel — two reads through different boxes are different measurements — but boxes jitter by a pixel or two between runs, and fingerprinting them exactly would make every channel a singleton and every cross-run comparison empty. 4px is below the smallest box difference that changes what a recognizer sees and above the observed jitter. **Not swept.** |
| `_lineage_transforms` max depth | 32 | `invented` | Runtime bound against a malformed parent cycle. Costs a truncated transform list, never a hang. |

### Channel-equivalence criterion (stage S2)

Fixed **before** the first envelope report was read, which is the only thing
that makes them preregistered rather than fitted. They decide which channel
pairs the S4 consistency regularizer may tie: a strict garbling must not be
forced to agree with the richer channel, because its disagreement is evidence
about channel quality rather than a violation.

| Constant | Value | Provenance | Basis / note |
|---|---:|---|---|
| `EQUIV_MARGIN` | 0.05 NED | `preregistered` | Largest mean paired NED difference still called equivalent. Chosen as roughly one character in a twenty-character reading — below the granularity at which a difference could change a reviewer's decision. **Not swept**; a sweep would be fitting the criterion to the answer. |
| `EQUIV_ALPHA` | 0.05 | `preregistered` | Two-sided exact sign test. Exact rather than normal-approximated because the strata here have single-digit region counts, where the approximation is anticonservative. |
| `EQUIV_MIN_PAIRS` | 8 | `preregistered` | Below this the verdict is **`underpowered`**, which is a distinct verdict and *not* a synonym for `equivalent` — the same rule `decant` holds between `unknown` and `absent`. The floor sits above the n at which the exact sign test *can* reject: n=6 is the first (2/2⁶ = 0.031), so at n=6 or 7 the only reachable "not equivalent" verdict is a unanimous one, and anything short of unanimity would be reported as equivalence by arithmetic rather than by evidence. 8 gives the test one dissenting pair of headroom. **Not swept.** |
| `MATCH_IOU` | 0.5 | `inherited` | IoU at which a shipped region is taken to *be* the annotated one, in `scripts/eval_channel_envelope.py`. Inherited from the detection harness's primary threshold so "the pipeline found this region" means the same thing in both reports. |

Equivalence requires **both** a small effect and an undetectable direction. A
channel reliably worse by less than the margin is still worse, and the sign
test is what notices.

`UNREGISTERED` is deliberately not a member of `REGISTERED_ROUTES`: a marker
that reported itself as registered would let untraceable reads disappear into
the coverage figure they exist to expose.

## Candidate distribution — `src/tofu/layers/proof_distribution.py`

Added 2026-08-16 with stage S3. The softmax head that gives the retrieval
stack a probability object where it previously had only a ranking.

| Constant | Value | Provenance | Basis / note |
|---|---:|---|---|
| `TEMPERATURE` (τ) | 0.07 | `invented` | Sets how sharply cosine similarity converts into belief. As τ→0 the distribution collapses onto the argmax and reproduces the ranking it sits beside; as τ→∞ it flattens to uniform and says nothing. Taken from the temperature the encoder's own supervised-contrastive loss uses over the same cosine geometry (`proof_encoder.supervised_contrastive_loss` defaults to 0.1), rounded toward the sharper end on the grounds that a retrieval distribution should be no flatter than the loss that shaped the embedding space. **Not swept.** It decides nothing yet: the distribution features enter fusion at zero weight, and the fitted value belongs to the supervised objective that will train against it. |

The derived quantities carry no thresholds of their own. Entropy is normalized
by `log K` so a larger pool does not read as a more uncertain one; `top_mass`
defaults to k=3; `max_attempt_divergence` is `0.0` — not `None` — when a single
attempt was scored, so "no disagreement measurable" cannot be confused with
"measured, and none found".

## Component alignment — `src/tofu/layers/proof_alignment.py`

Registered 2026-08-17 with stage S5. Every one of these was hand-set when the
layer was written and none has been swept. They are now loadable from an
artifact (`load_cost_weights`) so a fitted set can replace them without a
source edit; the default reproduces the original values bit for bit and stays
that way until a fit is certified.

| Constant | Value | Provenance | Basis / note |
|---|---:|---|---|
| `DEFAULT_COST_WEIGHTS` (η) | (0.55, 0.25, 0.20) | `invented` | Weights on centroid position, log-area and log-aspect. **Not fitted, and deliberately not fitted here**: the corpus that could fit them is the corpus they would be evaluated on, which is how `MAX_ZOOM_INFLATION` shipped and regressed. There is also a mechanical reason a fit would not help where it is most wanted — see the ablation note below. |
| `ASPECT_CLIP` | 2.0 | `reasoned` | Log-aspect differences above this are already maximally unlike; without the clip one wildly elongated component dominates a whole plan. |
| `ENTROPY_EPSILON` (ε) | 0.08 | `invented` | Sinkhorn smoothing. |
| `DUSTBIN_MASS` (ρ) | 0.25 | `invented` | Mass reserved for unmatched components on both sides. |
| `DUSTBIN_COST` | 0.35 | `invented` | Price of routing a component to the dustbin rather than matching it. |
| `SINKHORN_ITERATIONS` | 50 | `invented` | Runtime bound. |
| `MAX_COMPONENTS` | 64 | `invented` | Runtime bound — and, per the ablation, the point at which the whole method stops applying to CJK: a single Han glyph is many radical and stroke components, so the cap truncates the very structure it is trying to match. |

**Measured 2026-08-17** (`evidence/alignment-ablation-frozen-v1.json`, 66
regions): the truth transports more cheaply than the average wrong candidate on
**48/66** regions, exact sign test **p = 0.00029** — the feature separates. But
the effect is entirely alphabetic: Latin 26/31 (p = 0.0002) and Cyrillic 6/6
(p = 0.031), against **0 top-1 across all 25 Han, Hangul and Japanese regions**.
Re-weighting η cannot fix that, because the failure is in the component
features themselves, not their weighting.

## Scene-conditioned corruption — `src/tofu/layers/marinade.py`

Added 2026-08-17 with stage S6. **Training-only, and inert**: across the 66
annotated regions of the frozen corpus, zero inherit a conditionable material,
so the conditioned branch is unreachable on the only corpus that exists.

| Constant | Value | Provenance | Basis / note |
|---|---:|---|---|
| `DEGRADATION_PRIORS` | 6 classes | `invented` | Owned by `layers/scene.py` and imported, not restated — a corruption family drifting from the observation that justifies it would mean the encoder learned a decay Scene never claimed to see. Every entry is `plausible_not_measured`. |
| `PROCESS_FAMILIES` | 6 processes | `reasoned` | What each inscription process physically does to a mark: a cut mark cannot bleed, an emissive one has no substrate chemistry at all. **Unreachable today** — Scene does not infer inscription process. |
| `CONDITIONABLE` | 4 of 6 classes | `reasoned` | Excludes `unknown` and `textured_unknown`. A class whose name says it could not be identified is not a material, and conditioning on it would dress a failed classification as physical knowledge. |
| `UNCONDITIONED` | 8 processes | `inherited` | The full `proof.PROCESSES` set — what training already samples. Conditioning may only ever narrow it. |

**Measured 2026-08-17** (`evidence/scene-conditioning-coverage-v1.json`): 0/66
regions conditioned. Three gates each close on their own — 16 regions land on
no surface, 50 land on a surface classified `unknown`, and all 50 of those also
carry `calibration_status: "unfitted"` and `substrate_trust: "unmeasured"`.
The calibration gate alone would close every region even if material
classification were perfect, which is correct behaviour and not a defect:
uncalibrated Scene evidence is missing evidence.

## Supervised objective — `src/tofu/layers/proof_objective.py`

Added 2026-08-16 with stage S4. **Training-only, and not promoted**: the
distributional arm lost its paired real-ink gate (2/9 against the incumbent's
4/9), so the contrastive objective remains the default and these constants
decide nothing that ships. They are registered because they were used to
produce a measured result, not because anything depends on them.

| Constant | Value | Provenance | Basis / note |
|---|---:|---|---|
| `SIGMA` (σ) | 0.15 | `invented` | Temperature of the NED-graded soft target. As σ→0 the target collapses onto the closest candidate and the loss becomes ordinary hard cross-entropy; as σ grows every candidate looks equally acceptable and the graded signal disappears. 0.15 sits just under one edit in a seven-character reading. **Not swept** — and on the only corpus available it was also **not exercised**: 98.7% of non-truth pool members sat at NED exactly 1.0, so the target was 98.9% one-hot regardless of σ. |
| `MU_CONSISTENCY` (μ₁) | 0.1 | `invented` | Weight on the attempt-consistency regularizer, deliberately an order of magnitude below the cross-entropy it accompanies: agreement between views is evidence, not the objective, and a consistency term strong enough to dominate is satisfied perfectly by an encoder that ignores the ink and says the same thing every time. **Not swept.** |
| `DISTRIBUTIONAL_POOL` | 8 | `reasoned` | Candidates per training instance, in `scripts/train_proof_encoder.py`. A cap rather than a fitted size: `proof.hard_negatives` returns every silhouette confusion of a string, and a long string has many, so an uncapped pool would make one long region cost what fifty short ones do. |

μ₂ (the transport regularizer weight) is deliberately absent: it belongs with
the fitted alignment cost in stage S5, and coupling an unfitted η to the
candidate distribution would train against three frozen guesses.

## Glyph-margin calibration — `src/tofu/layers/proof_calibration.py`

| Constant | Value | Provenance | Basis / note |
|---|---:|---|---|
| `MIN_CALIBRATION_SAMPLES` | 30 | `inherited` | Same floor as `temper.MIN_FIT_SAMPLES`. Below it, a fitted map records accidents in a small corpus; the first zh-Hant artifact has 12 and correctly remains unfitted. |
| `MIN_ACCEPTED_SAMPLES` | 5 | `reasoned` | Prevents one lucky high-margin match from manufacturing a green threshold. Not a quality threshold and not yet swept. |
| `TARGET_GREEN_PRECISION` | 0.98 | `preregistered` | Phase 1 safety gate. It is a fitting target: the actual probability threshold remains `None` until labelled evidence reaches it. |

The logistic slope, intercept and acceptance threshold are fitted artifact
values, never source constants. Unfitted or insufficient evidence may only
produce `review_required` or `unresolvable`.

## Recognition — `src/tofu/utils/textmatch.py`

| Constant | Value | Provenance | Basis / note |
|---|---:|---|---|
| `FUZZY_MATCH_THRESHOLD` | 0.85 | `inherited` | "plan-specified minimum edit-distance ratio". Now also gates Guided reconciliation (`layers/aboyeur.py`), where it may only produce `review`, never `complete` — precisely because it is uncalibrated. |
| `TWIN_FLOOR` | — | **`measured`**, thin | Set from three points. Recorded in the dead-ends file as a fitting caution. |

## Verification — `src/tofu/layers/verify.py`

| Constant | Value | Provenance |
|---|---:|---|
| `OCR_WEIGHT` / `SSIM_WEIGHT` / `STYLE_WEIGHT` / `GARNISH_WEIGHT` | 0.7 / 0.3 / 0.2 / 0.15 | `invented` |
| `CONTENT_PASS_THRESHOLD` / `CONTENT_REVIEW_THRESHOLD` | 0.85 / 0.50 | `invented` |
| `RESIDUAL_PENALTY_THRESHOLD` | 0.3 | `invented` |
| `OCR_CONFIDENCE_REVIEW` | 0.50 | `invented` |
| `COLOR_DELTA_E_SCALE` | 40.0 | `invented` |
| `QUAD_OVERFLOW_TOLERANCE` | 0.02 | `invented` |

## Reconstruction — `src/tofu/layers/cleanse.py`

| Constant | Value | Provenance | Note |
|---|---:|---|---|
| `DILATE_ITER` | 3 | `reasoned` | ~3px, to catch anti-aliased glyph edges. |
| `FEATHER_PX` | 3.0 | `invented` | |
| `RING_PX` | 14 | `invented` | |
| `MIN_RING_PIXELS` | 20 | `reasoned` | Below this a gradient fit degrades to flat fill. |
| `MIN_MASK_PIXELS` | 6 | `reasoned` | A tiny Otsu component is not safe evidence alone. |
| `GROUP_GAP_PX` | 12 | `invented` | |
| `TELEA_RADIUS_DEFAULT` | 3 | `inherited` | OpenCV convention. |

---

## What the tally says

Counting the detection table alone: **4 `measured`**, 10 `reasoned`,
3 `inherited`, and **the clear majority `invented`**.

That is the finding, and it is worth more than any individual retune. It does
not mean the values are wrong — most were chosen by someone looking at real
output, and the corpus result is what it is *with* them. It means that when a
detection result disappoints, "a threshold is miscalibrated" is a hypothesis
with roughly thirty untested candidates behind it, and picking one to nudge is
guessing unless the lineage graph says that constant is what killed the
candidate.

Which is exactly what `GET /api/manifest/{id}/detection-attribution` and
`scripts/eval_detector_evidence.py` are for: instrument first, sweep only what
the instrument implicates, and hold out evidence before shipping the change.

## Vision 2 shadow-runtime budgets

Added 2026-08-13. These are resource ceilings, not quality or acceptance
thresholds; exceeding one records an explicit omission or abstention.

| Constant / field | Value | Provenance | Basis / note |
|---|---:|---|---|
| `DEFAULT_MAX_CANDIDATES` | 256 | `reasoned` | Bounds the persisted recall ledger on dense scenes while retaining candidates round-robin across source stages. |
| `DEFAULT_MAX_PER_SOURCE` | 64 | `reasoned` | Prevents one detector/pass from consuming the entire ledger. |
| `Vision2Config.max_retrieval_candidates` | 64 | `reasoned` | Per-region encoder candidate ceiling; candidate generation is provenance-prioritized and never filtered by answer length. |
| `Vision2Config.max_retrieval_fonts` | 5 | `reasoned` | Bounds cross-face rendering while retaining the highest candidate-coverage faces. |
| `Vision2Config.max_retrieval_regions` | 24 | `reasoned` | Per-image encoder invocation ceiling. Remaining regions record `retrieval_budget_exhausted`; they are not silently omitted. |
| proof-runtime views | 3 | `reasoned` | Original, mild blur, and morphological close. Variance is diagnostic only and cannot produce a recommendation. |
| `Vision2Config.alignment_top_k` | 3 | `reasoned` | Component alignment is restricted to the leading retrieval hypotheses and cannot rerank them. |
| `MAX_COMPONENTS` | 64 | `reasoned` | Bounds component-cost matrix growth; the largest components are retained deterministically. |
| `SINKHORN_ITERATIONS` | 50 | `reasoned` | Fixed solver budget for deterministic shadow latency. Non-convergence yields missing diagnostic evidence. |
| `ENTROPY_EPSILON` | 0.08 | `invented` | Initial entropy regularization for the diagnostic arm; not calibrated and not decision-eligible. |
| `DUSTBIN_MASS` | 0.25 | `invented` | Initial partial-mass allowance for missing or spurious components; must be ablated before promotion. |
| `DUSTBIN_COST` | 0.35 | `invented` | Initial unmatched-component cost; diagnostic only and excluded from production Fusion calibration. |
| `Vision2Config.counterfactual_top_k` | 3 | `reasoned` | Synthetic Scene hypotheses are diagnostic-only and restricted to the leading retrieval candidates. |
| `Vision2Config.counterfactual_beam` | 4 | `reasoned` | The explicit v1 operator set is identity, abrasion, bleed, and blur; callers cannot expand the beam beyond it. |

## Vision 2 Fusion calibration

Fusion has no source-coded recommendation, rejection, margin, or contradiction
threshold. `recommendation_threshold`, `rejection_threshold`,
`minimum_margin`, and `maximum_contradiction` must all be present in the
revision-compatible calibration artifact. Missing, malformed, or mismatched
artifacts fail closed to review (or remain shadow-only).
