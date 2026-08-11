# Output Dependability Roadmap

Status: working reference for moving ToFU from approximately 78% overall
completion to at least 90%, while taking MVP feature completeness above 95%.

This document defines the remaining work in terms of observable output quality,
not the number of implemented screens, endpoints, or classes. A feature is not
complete merely because a code path exists. It is complete when representative
inputs exercise it, failures are detected, unsafe output is held for review,
and the result meets a recorded acceptance threshold.

The deeper inventory in `docs/cross-layer-architecture.md` is the main source.
Its ToFU and Cicerone sections were re-audited in July 2026; its downstream
scores are explicitly provisional. The OCR and reconstruction work below also
incorporates `docs/multi-candidate-ocr-inpainting-implementation-plan.md`.

## Target and scoring contract

The target is two related milestones:

- **95%+ MVP completeness**: every promised static-image workflow works from
  upload through verified export; optional capabilities report their real
  availability; known unsupported cases fail visibly or enter review.
- **90% overall completeness**: the static-image pipeline is dependable on a
  representative held-out corpus, the video path is honestly beta-quality,
  and release evidence is reproducible from the tracked checkout.

Suggested weighted score for future assessments:

| Dimension | Weight | 90%-complete requirement |
|---|---:|---|
| Detection and transcription | 22% | High recall and usable text accuracy across the declared corpus |
| Erasure and reconstruction | 18% | Unsafe repairs are rejected; accepted repairs are usually clean |
| Rendering and scene integration | 16% | Correct shaping, fit, direction, geometry, and defensible style match |
| Verification and review safety | 14% | Independent QA catches seeded and natural failures reliably |
| Pre-flight and scene understanding | 8% | Warnings predict actual risk and surface evidence is trustworthy |
| Translation, memory, and interchange | 7% | Complete assisted workflow with stable round trips and useful reuse |
| Video | 5% | Explicit beta contract with stable tracking and bounded flicker |
| Product, tests, packaging, and operations | 10% | Clean checkout reproduces tests, builds, capabilities, and release |

Percentages must not be raised solely because a model or metric was added.
Raise them only after its corpus result, failure behavior, latency, capability
state, and provenance are recorded.

## Starting position and expected outcome

Left column is measured from the tracked checkout on 2026-08-10, not estimated.
A plan that does not record where it started cannot demonstrate that it moved.

| Metric | Measured today | Target | Source |
|---|---:|---:|---|
| Aggregate detection recall | **0.708** (51/72) | >= 0.90 @ IoU 0.5 | `baselines/detection.json` |
| Mean best IoU | **0.739** | reported @ 0.5 and 0.75 | `baselines/detection.json` |
| Worst stratum (`la-bastille-1789`) | **0.333** | >= 0.80 | `baselines/detection.json` |
| Vertical CJK (`japan-street`) | **0.571** | >= 0.85 | `baselines/detection.json` |
| Garbage fraction | 0.049 | <= 0.10 held-out | `baselines/detection.json` |
| Review rate | **0.404** | ceiling published (P1.31) | `baselines/detection.json` |
| Corpus size | **11 fixtures / 72 regions** | 100 images / 500 regions | P0.3 |
| Backend tests collecting | **168, +1 collection error** | 0 errors, count published | `pytest tests/ -q` |
| Harnesses with a real baseline | **1 of 9** | 9 of 9 | `tests/regression/baselines/` |
| Auto-accept precision | unmeasured | >= 0.95, ECE <= 5% | P1.28 |
| Residual-source false negatives | unmeasured | <= 2%, bound published | P1.17 |

Estimated dimension movement, using the weight table above and the provisional
per-layer scores in `cross-layer-architecture.md`. These are estimates and are
marked as such until P0.1 replaces them:

| Dimension | Wt | Now (prov.) | After | Δ pts |
|---|---:|---:|---:|---:|
| Detection and transcription | 22% | ~70 | 90 | **+4.4** |
| Erasure and reconstruction | 18% | ~72 | 90 | **+3.2** |
| Rendering and scene integration | 16% | ~78 | 90 | +1.9 |
| Verification and review safety | 14% | ~65 | 90 | **+3.5** |
| Pre-flight and scene understanding | 8% | ~75 | 88 | +1.0 |
| Translation, memory, and interchange | 7% | ~73 | 88 | +1.1 |
| Video | 5% | ~60 | 70 (beta) | +0.5 |
| Product, tests, packaging, operations | 10% | ~80 | 95 | +1.5 |
| **Overall** | | **~78** | **~95** | **+17** |

Three observations follow. First, the plan **overshoots** the 90% target by
about five points, so it can be cut — see "Scope control". Second, the largest
single block of gain is verification and review safety, which is also the
cheapest. Third, two things this plan makes *worse* are not tracked anywhere in
it: wall-clock latency and review rate. P1.30 and P1.31 exist to close that.

## P0 — Establish trustworthy release evidence

This is the prerequisite for every quality claim.

> **Promoted into P0.** Three items numbered in P1 below have no corpus
> dependency, are cheap, and are pure output-safety: **P1.9** (fresh independent
> verification), **P1.25** (split verifier from generator evidence), and
> **P1.29** (non-negotiable coverage gates). They keep their numbers for
> cross-reference but are scheduled here, ahead of P0.3b. P1.29 in particular is
> the enforcement mechanism several exit gates depend on, so building it late
> means the gates are unenforceable until late.

- [ ] **P0.1 Re-audit all downstream architecture sections.** Check Scene,
  Cleanse, Scribe, Basil, Wasabi, Menu, Garnish, Memory, Savor, Verify,
  typography, font matching, font registry, Knead, provider routing, imaging,
  and video against the current code. Replace provisional scores and stale
  line counts in `cross-layer-architecture.md`.
  - Done when every claimed mechanism has a code location and test/evaluation
    reference, and every known incomplete item is either open here or marked
    closed with evidence.

- [ ] **P0.2 Restore the local test suite and reconcile its evidence.**
  `tests/` is gitignored **deliberately** — the suite is local-only and is not
  published to the remote. That intent is not in question here and must not be
  "fixed" by tracking the directory. Three separate defects follow from it:
  - **(a) Missing modules.** `tests/regression/tolerances.py`,
    `tests/regression/generate_baseline.py`, `tests/regression/baseline_metrics.json`
    and `tests/conftest.py` are absent from disk, so collection errors. The nine
    `test_*_regression.py` harness tests named in `tests/regression/__pycache__`
    are gone too. Restore the infrastructure; let the coverage test report the
    missing harnesses rather than papering over them.
  - **(b) CI asserts on a path that cannot exist.** Tracked
    `.github/workflows/ci.yml` runs `python -m pytest tests/ -q` and
    `ruff check src/ tests/ server/` on GitHub runners, where `tests/` is absent
    by design. Drop `tests/` from the CI test and lint jobs; CI covers `src/`,
    `server/`, and the frontend. Release test evidence comes instead from a
    recorded local run committed as `evidence/test-run.json` (commit, count,
    pass/fail, Python, OS) — the artifact is public, the tests stay private.
  - **(c) Published count is fiction.** `tofu-final-summary.txt` claims 1,544
    backend tests; the tree collects 168. Republish the real number.
  - Done when local `pytest tests/ -q` collects and passes with zero errors, CI
    is green without referencing `tests/`, and every published test count is
    generated from a recorded run rather than typed.

- [x] **P0.3a Freeze what already exists.** *Done 2026-08-10.*
  `evidence/corpus-v1.json` pins all 11 fixtures / 72 regions by SHA-256,
  assigns each to a declared stratum, and records a deterministic holdout.
  Generated and checked by `scripts/freeze_corpus.py`; enforced by
  `tests/test_corpus_freeze.py` (15 tests).
  - **Every P1 target below is gated on P0.3a, not P0.3b.** Strata with too few
    samples report `insufficient sample`, never a fabricated number.
  - **Split rule: deterministic and content-blind** — within each stratum, sort
    by name and hold out the last. Chosen so the split cannot have been picked
    after seeing which fixtures scored well. Holdout is `la-bastille-1789`,
    `rue-des-martyrs`, `russian-billboard-2` — 3 fixtures / 15 regions (21%).
    Split hash `d542cbde8d4a647f…` covers holdout membership *and* holdout
    annotation content, and is cross-checked against the recorded baseline.
  - **Strata**: `latin_plaque` (4 fixtures / 11 regions), `latin_poster` (2/14),
    `cyrillic` (2/14), `cjk_horizontal` (1/8), `cjk_vertical` (1/7),
    `mixed_dense` (1/18).

  Four findings the freeze surfaced, all recorded in the manifest:

  1. **The baseline's `gt_sha256` held a 32-character MD5.** A hash whose field
     name misstates its algorithm can never verify and reads as tamper evidence
     when it fails. Renamed to `gt_md5_superseded`; per-asset SHA-256 now lives
     in the manifest. A regression test asserts the mislabel cannot return.
  2. **No asset has a resolved licence**, and these files are tracked in a
     public MIT-licensed repository. `russian-billboard-2` is the urgent one:
     its own annotation transcribes a `Valentina Ursu (RFE/RL)` watermark, which
     is positive evidence of third-party copyright. Clear or replace it before
     any release that advertises the corpus. `gemini-street` is AI-generated
     with unrecorded terms.
  3. **Four of six strata hold a single fixture**, so they contribute no holdout
     evidence at all — their recall floors are measured on tuned data. This is
     a limitation of an 11-fixture corpus, not something the split rule can fix.
  4. **Nine of eleven annotations are `partial`**, so unannotated text is
     neither credited nor penalised. Precision and F1 are therefore not
     meaningful corpus-wide; only recall is comparable. P1.1's precision target
     cannot be evaluated until P0.3b adds complete annotations.

- [ ] **P0.3b Grow the corpus to release size (weeks).** Extend to Latin, CJK,
  Arabic/RTL, Indic, Thai/Khmer, mixed scripts, horizontal/vertical text,
  small text, low contrast, perspective, curved signs, reflections, masonry,
  fabric, gradients, dense signage, partial occlusion, and negative/no-text
  regions. Record asset licence/source and prevent train/tune/test leakage.
  - Minimum: 100 static images, 500 annotated text regions, at least 20
    negative scenes, and at least 15 samples for each declared difficult
    category. Use a held-out release split that thresholds are never tuned on.
  - This is the single largest cost item in this document. Sizing it as one
    checkbox was an error; treat it as its own workstream.

- [ ] **P0.4 Record reproducible baselines.** For every corpus run store commit,
  OS, Python version, model names and hashes, provider config revision, device,
  seed, capability state, thresholds, per-stage latency, and raw per-region
  observations.
  - Done when two runs with fixed inputs/configuration produce identical
    decisions and equivalent metrics within documented numeric tolerance.

- [ ] **P0.5 Define supported-quality tiers.** Publish which language/script,
  orientation, material, resolution, and provider combinations are Supported,
  Beta, or Review-required. Avoid implying that optional degradation is equal
  to the neural/hybrid configuration.
  - Done when the API and UI expose the tier and missing capability before a
    user starts a run.

## P1 — Detection coverage and box quality

Detection is the highest-leverage bottleneck: a missed, clipped, or merged
region contaminates every downstream stage. Per-fixture recall from the tracked
baseline (`tests/regression/baselines/detection.json`,
`detection-2026-08-09-scene-guide-only`) is:

| Fixture | Matched / GT | Recall | Mean best IoU |
|---|---:|---:|---:|
| `la-bastille-1789` | 3 / 9 | **0.333** | 0.365 |
| `gemini-street` | 8 / 18 | **0.444** | 0.477 |
| `japan-street` | 4 / 7 | **0.571** | 0.472 |
| `china-street` | 7 / 8 | 0.875 | 0.628 |
| `russian-billboard` | 10 / 11 | 0.909 | 0.763 |
| aggregate | 51 / 72 | **0.708** | 0.739 |

Aggregate tests alone are therefore insufficient: the aggregate is 0.708 while
the worst declared stratum is 0.333.

> **Provenance warning.** `cross-layer-architecture.md` reports 0.333 for
> `gemini-street`, which the tracked baseline attributes to `la-bastille-1789`.
> Two different recall conventions are live under the same fixture names. The
> table above uses the tracked baseline. Reconciling the two is in scope for
> P0.1, and the conflict is the clearest available argument for P0.4.

- [ ] **P1.1 Make detection evaluation polygon-aware and stratified.** Report
  precision/recall/F1 at IoU 0.5 and 0.75, split by script, orientation, text
  height, density, material, and engine. Separately score missed, clipped,
  fragmented, and over-merged regions.
  - Target: held-out recall >= 0.90 at IoU 0.5, precision >= 0.90, and no
    declared stratum below 0.80 recall. Report confidence intervals.
  - **Precision is not yet measurable.** Nine of eleven annotations are
    `partial` (P0.3a), so unannotated text is neither credited nor penalised
    and corpus-wide precision/F1 are undefined. Until P0.3b adds complete
    annotations, report recall and garbage fraction, and mark precision
    `insufficient evidence` rather than computing a number the corpus cannot
    support. The strata are the six declared in `evidence/corpus-v1.json`.
  - Implemented in `scripts/detection_metrics.py` (pure, 36 unit tests) and
    `scripts/eval_detect_corpus.py` (runs the frozen corpus, refuses to report
    if it has drifted). Failure taxonomy: `matched`, `missed`, `clipped`,
    `fragmented`, `over_merged_policy`, `over_merged_detector`,
    `over_merged_unattributed`, `displaced`.

  **First stratified result, 2026-08-10** (`scripts/eval_out/corpus-detection.json`).
  Aggregate 51/72 = 0.708, reproducing the recorded baseline exactly while
  adding the structure it lacked. Lineage 100% on all 11 fixtures.

  | Stratum | n | R@0.5 | CI95 | R@0.75 |
  |---|---:|---:|---|---:|
  | `latin_plaque` | 11 | 1.000 | [1.00, 1.00] | 1.000 |
  | `cyrillic` | 14 | 0.929 | [0.79, 1.00] | 0.571 |
  | `cjk_horizontal` | 8 | 0.875 | [0.62, 1.00] | 0.375 |
  | `cjk_vertical` | 7 | **0.571** | [0.29, 0.86] | 0.429 |
  | `latin_poster` | 14 | **0.571** | [0.29, 0.79] | 0.357 |
  | `mixed_dense` | 18 | **0.444** | [0.22, 0.67] | **0.000** |

  | Split | Recall | CI95 | Losses |
  |---|---:|---|---|
  | dev (57 regions) | 0.737 | [0.61, 0.84] | clipped 5, displaced 5, missed 4, over-merge/policy 1 |
  | holdout (15 regions) | 0.600 | [0.33, 0.87] | over-merge/detector 4, displaced 2 |

  Four things this says that the previous single-number harness could not:
  1. **`mixed_dense` scores 0.000 at IoU 0.75** while reaching 0.444 at 0.5 —
     every box it does find is loose. Recall at 0.5 alone hides this
     completely, and the same pattern is visible in `cyrillic` (0.929 → 0.571)
     and `cjk_horizontal` (0.875 → 0.375). Box *quality* is a distinct problem
     from box *presence*, and it is the larger one.
  2. **The losses are not the same defect.** Dev loses regions mostly to
     clipping and displacement; holdout loses 4 of 6 to over-merging. Those
     need different fixes (P1.3 vs P1.2), and a recall number cannot
     distinguish them.
  3. **All 4 holdout over-merges attribute to the detector, not the merge
     policy** — a single detector proposal spanning two annotated regions, with
     no merge operation involved. So the fix is detection-side. That
     attribution is only possible because of P1.1a; before it, all four read
     `over_merged_unattributed`.
  4. **Every interval is too wide to act on.** `cjk_vertical` at [0.29, 0.86]
     cannot support or refute an 0.80 floor. All six strata are flagged
     `insufficient sample`. This is P0.3b's justification stated in numbers
     rather than in principle.

  Scoring note: ground truth is axis-aligned throughout, so candidates are
  compared in box space (`scoring_mode: axis_aligned`). Scoring tight quads
  against axis-aligned annotations was measured and rejected — it dropped
  `latin_plaque` at IoU 0.75 from 1.000 to 0.455 by penalising the detector
  for the annotation's empty corners. Polygon-vs-polygon scoring unlocks with
  P0.3b.

- [x] **P1.1a Close the candidate-lineage gap to shipped regions. (new)**
  *Done 2026-08-10 — lineage coverage 0% → **100%**, backstop firing zero times.*

  P1.1 found that `okara`'s graph did not reach the regions that ship, so
  over-merges could not be attributed and were reported
  `over_merged_unattributed`. The root cause was not one missing link but a
  chain that ended in six different places. Each was invisible for the same
  reason: the graph *looked* populated — full of intermediate passes — while
  the regions that actually shipped were the ones missing from it.

  Fixed, all by identity rather than by matching coordinates:
  1. `RawDetection` now carries `candidate_id`, stamped when recorded, and
     `InstText.lineage_candidate_id` carries it to the manifest. The final
     node links by id, so no coordinate join is involved — which matters
     because the transforms are precisely what reshape geometry.
  2. `final` nodes had no parents at all (lineage close-out).
  3. **Only the *second* detection pass was recorded.** `merge_detections`
     recorded `extra` and never `base`, so every region surviving from pass
     one had no node. The largest single hole.
  4. `_split_tall_detections` and `merge_baseline_runs` emitted detections
     without recording anything, terminating the chain.
  5. `_compose_crop_text` merged without recording its inputs — exactly the
     alternatives a reviewer needs when a crop spans two signs.
  6. `merge_vertical_columns` recorded its node but never stamped it onto the
     detection, so vertical-CJK chains ended at the merge.
  7. `_corroborated` copied a detection (same polygon, promoted confidence)
     and dropped the id.

  8. **`run_paddle_rescue` recorded nothing at all** — its full-frame
     `backend.detect()` and both `detect_in_regions()` probes. This was the
     whole CJK gap. The asymmetry (Latin fixtures at 100%, CJK at ~25%) was
     never a property of the scripts: it was a property of which code path
     each takes. Every EasyOCR route happens to pass through a recording
     transform; the Paddle route passed through none.
  9. The single-pass branch and the language-refined second pass likewise
     bypassed `merge_detections`, the only place recording happened by
     accident.

  **Result**: 109/109 shipped regions across the five hard fixtures trace to a
  recorded detector origin, through observed transforms. `okara.ORIGIN_STAGES`
  now names the three stages that originate proposals (`raw_craft`, `zoom`,
  `surface_probe`) so the evaluator cannot drift from what the graph records.

  A backstop in `build_manifest` guarantees no shipped region is ever
  unreachable, and **logs when it fires** — a silent backstop would let the
  next unrecorded emit hide behind 100% coverage, with ancestry invented at
  the backstop rather than observed upstream. It currently fires zero times,
  which is what makes the 100% meaningful rather than merely true.

  Nothing here was closed by fuzzy-matching geometry: inference from final
  geometry produced five phantom defects in this codebase before, which is why
  the evaluator still reports `over_merged_unattributed` rather than guessing
  whenever ancestry is genuinely absent.

- [ ] **P1.2 Add a vertical-native detection path.** Stop relying primarily on
  horizontal CRAFT grouping plus merge/split repair for vertical CJK. Preserve
  native polygons and orientation evidence from the selected backend.
  - Target: vertical-text recall >= 0.85 with fragmentation and over-merge
    rates each <= 10% on the held-out vertical subset.

- [ ] **P1.3 Correct box clipping and adjacent-sign over-merging.** Add polygon
  completeness checks, border-contact evidence, targeted expansion/re-detect,
  and separation tests using ink gaps, surface boundaries, script/orientation,
  and independent detections.
  - Target: >= 90% of accepted boxes retain all annotated glyph ink while
    excluding adjacent annotated regions; manually seeded clipping and merge
    cases must be detected or sent to review.

- [ ] **P1.4 Replace special-case hybrid arbitration with general evidence.**
  Preserve all OCR observations, cluster compatible region hypotheses, calibrate
  provider confidence, and score script-aware agreement, language fit, crop
  completeness, and independent verification. Fix terminal-dash evidence to
  require separation from the last glyph and vertical centring.
  - Target: arbitration lowers CER relative to the best single default engine
    without reducing detection precision by more than 1 percentage point.

- [ ] **P1.5 Add learned or calibrated language identification.** Retain the
  rules as evidence/fallback, but calibrate script-compatible language choices
  and mixed-script cases using labelled crops.
  - Target: >= 95% language accuracy on supported monolingual crops and >= 90%
    on the declared mixed-script subset; uncertainty must be explicit.

- [ ] **P1.6 Implement device selection and resource policy.** Detect usable
  CPU/GPU providers, expose the selected device, cap memory/concurrency, and
  prove that fallback does not silently change the advertised quality tier.

- [ ] **P1.7 Add detection failure UX.** Surface uncovered likely-text areas,
  low-confidence boxes, engine disagreements, and suspected clipping/merging in
  the editor with one-click re-detect, split, merge, and manual correction.
  - Target: every region below the auto-accept threshold has a visible reason
    and an actionable recovery path.

## P1 — Transcription and OCR repair

- [ ] **P1.8 Complete the observation/provenance contract.** Store proposals,
  rejected alternatives, calibration revision, verification result, correction
  sequence, and final decision without leaking local paths. Preserve backward
  manifest compatibility.

- [ ] **P1.9 Use fresh independent verification before correction.** *(promoted
  to P0 — highest safety value in this document.)* A Paddle proposal and Paddle
  verification of that proposal must be separate observations.
  Unavailable/error/no-text states must never mean agreement.
  - Enforce at the type level: verification takes a distinct observation object,
    so a proposal cannot be passed as its own verifier. A convention that says
    "don't do this" is not a fix; a signature that cannot express it is.
  - Unavailable/error/no-text map to an explicit `INDETERMINATE` that routes to
    review. Today these can read as agreement, which is a correctness hole
    wearing the appearance of a passing check.
  - Target: supported auto-accepted text CER <= 0.05 overall, <= 0.10 in every
    declared difficult stratum; otherwise route the region to review.

- [ ] **P1.10 Expand Savor using measured confusions.** Add multi-glyph and
  touching-glyph segmentation, detected-face references, and independent shape
  evidence such as SSIM/Hu or contour distance. Expand context grammars only
  from corpus errors, not intuition.

- [ ] **P1.11 Expand Wasabi safely.** Grow beyond the three curated pairs using
  versioned language-specific mappings and tests; distinguish Japanese,
  simplified/traditional Chinese, and Korean Hanja. Apply only when language
  and engine provenance support the conversion.

- [ ] **P1.12 Replace the tiny manual Menu knowledge base.** Add versioned,
  locale-scoped gazetteer import from user/project data and optionally licensed
  geographic sources. Support spaces and stronger matching than raw
  `SequenceMatcher`; preserve evidence and never let a gazetteer silently
  override strong pixel evidence.

## P1 — Cleanse reconstruction and pre-Scribe safety

- [ ] **P1.13 Generate multiple reconstruction candidates from the untouched
  source.** Run eligible flat/gradient/Telea/neural strategies with a bounded
  candidate cap. Each retry must restart from the source and original mask,
  never from a previously damaged candidate.

- [ ] **P1.14 Separate hard rejection from ranking.** Hard-reject protected-area
  changes, residual source glyphs, hallucinated text, malformed output,
  structural breaks, and excessive seams before aesthetic scoring. Missing
  evidence cannot become a passing score through weight renormalization.
  - Target: >= 98% precision for auto-accepted repairs and <= 2% protected-area
    violation rate on the held-out corpus.

- [ ] **P1.15 Improve glyph masks for difficult ink.** Add polarity ensembles,
  multi-colour foreground handling, component ownership for touching glyphs,
  anti-aliased edge estimation, and mask-confidence output. Retain box fallback
  only as an explicit review-triggering degradation.
  - Target: mask recall >= 0.95 and precision >= 0.90 on an annotated mask
    subset, stratified by text colour count and background texture.

- [ ] **P1.16 Make reconstruction perspective- and material-aware.** Validate
  quads, rectify before candidate generation, classify material with confidence,
  tune the fill policy by evidence, inverse-warp, then recheck seams and
  protected pixels.

- [ ] **P1.17 Add final residual verification and bounded retry.** Test the
  final cleansed pixels using an OCR family independent of the proposal when
  available plus OCR-free glyph persistence. Try the next eligible candidate
  or mask variant; exhausted retries become review-required and block Scribe.
  - Target: residual-source false-negative rate <= 2% for auto-accepted
    repairs and hallucinated-text rate <= 1%.

- [ ] **P1.18 Promote a learned scene-text remover only by benchmark.** Pin the
  model/revision, run offline/startup/timeout tests, record seed/reproducibility,
  and provide a configuration kill switch. Promote DiffSTR/another provider
  only if it beats the LaMa baseline without protected-area regression.

## P1 — Rendering, shaping, fit, and style fidelity

- [ ] **P1.19 Make HarfBuzz/FreeType the dependable shaping path.** Package and
  exercise it for every supported script rather than depending on Pillow Raqm
  availability. Add bidi itemization for mixed Arabic/Hebrew and Latin/numbers,
  plus Tibetan and declared Southeast Asian coverage.
  - Target: 100% pass on script-specific shaping fixtures covering joining,
    combining marks, conjuncts, reordering, punctuation, and mixed direction.

- [ ] **P1.20 Add OpenType and variable-font support.** *(partial cut
  candidate.)* Discover axes/features, preserve `.ttc` face identity, select
  real weight/slant/width variants, and expose feature controls where source
  evidence supports them.
  - Keep the `.ttc` face-identity fix regardless — a silently wrong face inside
    a collection is a correctness bug, not a refinement. The variable-axis work
    is the deferrable half.

- [ ] **P1.21 Improve font-family matching.** *(partial cut candidate.)*
  Establish a licensed reference corpus, compare the current silhouette/Chamfer
  system with an embedding-based retrieval model, calibrate confidence, and
  remove or complete stub-like commercial catalog integration.
  - Do the two cheap halves: build the reference corpus (without it, top-5
    recall is not even measurable) and **delete** the stub catalog integration
    rather than completing it. Defer the embedding model.
  - Respect `docs/measured-dead-ends.md` before adding features here — five
    approaches are already recorded as measured and rejected, including
    serif/contrast features that fail on opposite strings.
  - Target: top-5 family/near-equivalent recall >= 90% on supported faces;
    low-confidence matches must be labelled approximate.

- [ ] **P1.22 Guarantee fit without illegibility.** Evaluate line breaking,
  expansion, minimum readable size, tracking compression, vertical layout, and
  overflow on real translations. Do not satisfy fit by shrinking below a
  configurable legibility floor.
  - Target: >= 98% of auto-accepted renders stay inside their polygon and above
    the script-specific minimum size; all failures are review-visible.

- [ ] **P1.23 Complete source style reconstruction.** *(partial cut candidate.)*
  Implement evidence-based underline/strikethrough, shadow casting,
  colour-temperature adaptation, gradient/multicolour text, and better
  perspective/curved baselines. Record which attributes are measured, inferred,
  user-set, or defaulted.
  - The provenance half is the valuable half and is cheap: tag every attribute
    `measured | inferred | user-set | defaulted` and surface it in QA. Effects
    can then be added incrementally without inflating confidence. Do the
    tagging; defer the effects.

- [ ] **P1.24 Upgrade Garnish from generic effects to source-conditioned
  integration.** *(cut candidate — see "Scope control" below.)* Measure
  weathering, blur, grain, edge wear, illumination, and shadow from source
  glyphs/surface; constrain changes to rendered coverage.
  - Target: blinded human preference >= 70% over the current deterministic
    baseline, with zero material increase in outside-mask changes.
  - **Protocol required before this target means anything.** As previously
    written it was unfalsifiable — no rater count, no pairing scheme, no power
    analysis. Fix: forced-choice paired comparison, source image shown,
    condition order randomized, raters blind to condition; **at least 50–60
    paired judgements** to distinguish 70% from 50% at conventional power;
    report the interval, not just the point estimate. If that study is not
    going to be run, downgrade this target to non-gating and say so.

## P1 — Independent verification and honest acceptance

- [ ] **P1.25 Split verifier evidence from generator evidence.** *(promoted to
  P0; implement together with P1.9 as one observation type.)* Prefer a different
  OCR family/model revision for round-trip and residual checks. Mark
  shared-family verification as correlated and lower its auto-accept authority.

- [ ] **P1.26 Implement spatial quality metrics correctly.** Replace the
  single-window SSIM approximation with windowed mean-map SSIM; add a validated
  perceptual metric such as LPIPS for referenced synthetic/clean-background
  cases and multiscale seam/structure tests for unreferenced scenes.

- [ ] **P1.27 Expand style verification.** Score family/shape similarity,
  weight, slant, tracking, leading, baseline/orientation, effects, and local
  illumination—not only colour and height. Treat small crops as insufficient
  evidence rather than neutral success.

- [ ] **P1.28 Calibrate QA scores to real failure probability.** Run blinded
  human labels, reliability diagrams, expected calibration error, and threshold
  selection by failure cost. Report false accept and false reject rates.
  - Target: >= 95% precision for overall auto-accept, >= 90% recall of
    unacceptable outputs, and <= 5% expected calibration error on held-out
    data. Thresholds must be versioned by capability tier.
  - **Sequencing.** This is the one P1 item that genuinely cannot run on P0.3a.
    Calibration needs held-out volume *and* a blinded labelling protocol that
    does not exist yet. Schedule it after P0.3b, not with the rest of P1, and
    define the labelling protocol as its first deliverable — same requirements
    as P1.24 (blind, randomized order, reported intervals), plus a written
    rubric for "unacceptable" so labels are reproducible across raters.
  - **Overlap to resolve.** Exit gate "95% of auto-accepted outputs judged
    usable" and this item's "95% auto-accept precision" measure different
    populations and will not produce the same number. Pick one as the gate and
    make the other diagnostic.

- [ ] **P1.29 Make coverage gates non-negotiable.** *(promoted to P0.)* Missed
  source regions, untranslated non-DNT regions, unavailable required checks,
  exhausted cleanse retries, overflow, and unsupported shaping must prevent a
  green overall status even if other regions score highly.
  - Implement as a hard veto list evaluated **after** scoring: one function, one
    test per gate. Cheap, dependency-free, and the mechanism several exit gates
    below assume already exists.

- [ ] **P1.30 Hold a latency budget. (new)** Multi-candidate cleanse (P1.13),
  independent verification (P1.9, P1.25), windowed SSIM (P1.26), and rectified
  reconstruction (P1.16) each add wall-clock cost, and nothing in this document
  catches the sum. P0.4 records per-stage latency but no gate reads it.
  - Target: publish a per-stage budget at the point P0.4 baselines are frozen,
    and fail the release gate on a regression beyond a documented multiple of
    that budget. A 1.5–3x slowdown is an acceptable price for the safety work
    only if it is a decision rather than a discovery.

- [ ] **P1.31 Hold a review-rate ceiling. (new)** The tracked baseline already
  measures `mean_review_rate` at **0.404**. P1.15 (box-fallback flags), P1.17
  (exhausted retries), P1.22 (overflow), P1.27 (insufficient evidence) and
  P1.29 itself all route *more* work to review by design. Nothing bounds the
  total.
  - Target: treat review rate as a first-class release metric with a published
    ceiling. An output pipeline that is provably safe and refers most regions to
    a human has not reached 90% completeness — it has moved the failure from the
    image to the operator. If the ceiling is exceeded, the correct response is
    better detection and masking, never a relaxed gate.

## P2 — Pre-flight and Scene prediction

- [ ] **P2.1 Calibrate the render-quality score against outcomes.**
  > **Premise corrected.** An earlier draft called this "a constant stub
  > returning 0.75", inherited from `cross-layer-architecture.md:168`.
  > `_predict_render_quality()` (`src/tofu/layers/tofu.py:672`) is **not**
  > constant — it sums `_render_quality_evidence()`, a weighted formula over
  > `font_coverage`, `font_px`, `effects`, `plane_fit`, `text_contrast`,
  > `repair_confidence`, and a verification prior. The arch-doc line is stale,
  > and it sits inside a section marked re-audited, which is itself a P0.1
  > finding.

  The real gap is that those weights are hand-set and have never been checked
  against downstream accept/reject outcomes. Log
  `(preflight_features, downstream_outcome)` pairs from every corpus run — start
  this during P0.4, since data collection is the long pole — then fit a model on
  preflight-available features only. Keep the current weighted formula as a
  transparent fallback.
  - Target: risk predictions separate accepted from rejected outcomes with
    AUROC >= 0.85 and are calibrated within 5 percentage points.

- [ ] **P2.2 Predict detection/segmentation difficulty from scene evidence.**
  Include density, contrast, occlusion, reflections, texture, orientation,
  estimated glyph size, and candidate-surface ambiguity. The current geometric
  proxy misses dense-scene failures such as `gemini-street`.

- [ ] **P2.3 Retire SAM integration** (or finish it, but retire is recommended).
  `SAMBackend` (`src/tofu/layers/scene.py:494`) requires a manually downloaded
  checkpoint the user must supply via `PipelineCfg.scene_model_path`, and is
  described in the architecture inventory as stub-like — yet it is advertised as
  a capability. **Removing an unproven capability raises honest completeness;
  keeping it lowers it.** If it is kept instead, pin a supported model, prove it
  improves text-surface recall/precision *and* downstream output, and add model
  availability, hash, device, and load failure to the capability endpoint.

- [ ] **P2.4 Improve surface geometry and material classification.** *(cut
  candidate.)* Produce confidence-bearing quads/planes and a measured material
  taxonomy useful to Cleanse and Scribe. Evaluate the downstream benefit, not
  classifier accuracy alone.
  - Only worth doing if Cleanse's fill policy actually branches on material.
    Measure by cleanse acceptance rate, never by classifier accuracy — this is
    the item in the document most likely to be measured wrong and declared a
    success.

- [ ] **P2.5 Resolve font subfamilies during coverage checks.** Validate the
  actual face/axis/features that will render, including light/italic/variable
  variants, instead of family-level coverage alone.

## P2 — Translation, semantic grouping, and Memory

- [ ] **P2.6 Define the translation boundary explicitly.** State whether ToFU
  provides assisted target entry only or a selectable MT provider interface.
  If MT is included, add provider provenance, glossary/placeholder protection,
  language validation, retry/timeouts, and human approval; never present the
  small fr-to-it glossary as general MT.

- [ ] **P2.7 Harden Basil on a multilingual grouping/alignment corpus.** Cover
  reordered targets, split/merged visual regions, mixed scripts, vertical text,
  punctuation, entities, placeholders, and insufficient-evidence fallback.
  - Target: >= 90% correct grouping/alignment on supported cases and zero
    geometry mutation; uncertain alignments require review.

- [ ] **P2.8 Add useful semantic and visual retrieval to Memory.** *(cut
  candidate — improves reuse, not output correctness.)* Combine text
  embeddings, stronger visual descriptors, colour/style/geometry evidence, and
  language-pair constraints. Add incremental indexing, collision handling,
  hit-rate/acceptance analytics, and tenant/project isolation.
  - Target: top-5 useful-match recall >= 90% on a held-out reuse set while
    wrong auto-applied matches remain zero; suggestions require approval until
    evidence supports a stricter policy.

- [ ] **P2.9 Expand interchange round-trip tests.** Prove VTM/XLIFF preserve
  geometry, language overrides, provenance, style, DNT state, Unicode,
  extensions, and old-manifest compatibility across export/import/export.

## P2 — Video beta dependability

Video should not block the static-image 90% milestone, but it must not inherit a
production-quality label it has not earned.

- [ ] **P2.10 Define a video beta corpus.** Cover pans/zooms, camera shake,
  occlusion, cuts, motion blur, perspective change, illumination change,
  appearing/cycling text, multiple tracks, and variable frame rate.

- [ ] **P2.11 Add temporal reconstruction.** Use neighbouring clean evidence or
  flow/track-aligned fills so background repair does not change independently
  per frame. Preserve shot boundaries and handle occlusion explicitly.

- [ ] **P2.12 Stabilize render geometry and appearance over time.** Smooth
  position, scale, rotation, colour, opacity, and style using confidence-aware
  track state; keyframes and user overrides remain authoritative.

- [ ] **P2.13 Add temporal verification.** Measure flicker, track drift,
  intermittent residual source text, missed frames, boundary popping, and
  preview/export parity.
  - Target: no dropped/duplicated frames; preview/export pixel parity within
    documented encoding tolerance; track recall >= 0.90; ID switches <= 2 per
    1,000 annotated track frames; temporal QA threshold set from blinded review.

## P2 — Product and operational finish

- [ ] **P2.14 Make capability reporting end-to-end accurate.** Report OCR,
  shaping, SAM, inpainting, GPU, FFmpeg, fonts, verifier independence, model
  revisions, and quality tier. Exercise every combination in API/UI tests.

- [ ] **P2.15 Remove current dependency warnings and upgrade debt.** Replace
  Pydantic `dict()`/`__fields_set__` usage, execute the tracked dependency
  upgrade plan, and test oldest/newest supported library ranges where the
  application does not pin exact versions.

- [ ] **P2.16 Add production resilience tests.** Cover concurrency, cancellation,
  restart/recovery, corrupt assets, decompression bombs, disk exhaustion,
  provider crashes/timeouts, large projects, SSE disconnects, and cleanup of
  temporary artifacts without data loss.

- [ ] **P2.17 Reconcile release metadata with evidence.** Generate test counts,
  endpoint counts, supported tiers, and benchmark results rather than copying
  them into prose. Do not label a release Production/Stable until the clean
  release commit passes all quality and operational gates.
  - Motivating example, already present: `tofu-final-summary.txt` publishes
    1,544 backend tests against a tree that collects 168. Hand-copied numbers
    drift silently and are indistinguishable from measured ones once written.

- [ ] **P2.18 Audit model licensing and redistribution. (new)** LaMa, DiffSTR,
  SAM, PaddleOCR and EasyOCR weights each carry their own licence, and this
  document promotes models (P1.18, P2.3) without asking whether the project may
  ship or fetch them. Record, per model: licence, redistribution permission,
  commercial-use status, provenance URL, and pinned revision hash. A model that
  cannot be redistributed must be fetched at runtime with the licence surfaced,
  or dropped from the advertised capability set.

## Exit gates for 90% overall / 95%+ MVP

All of the following are required; averages cannot hide a failed critical gate.

- [ ] **Split by what CI can honestly see.** Frontend tests, type checks, lint
  of `src/` and `server/`, builds, package installation, and API-doc freshness
  pass in CI from a clean clone. Backend and regression suites are **local-only
  by design** and are evidenced by a recorded run at the release commit
  (`evidence/test-run.json`), not by CI. The previous wording — "clean-clone
  backend tests pass in CI" — was unachievable given that `tests/` is
  deliberately unpublished, and would have been satisfied only by breaking that
  intent.
- [ ] Static held-out corpus meets detection, OCR, cleanse, render-fit, and QA
  targets above, with no declared supported stratum below its floor.
- [ ] At least 95% of automatically accepted static outputs are judged usable
  without repair in blinded review; failures are reproducibly routed to review.
- [ ] No known source text remains in an auto-accepted output; the measured
  residual false-negative bound is published with sample size/confidence.
- [ ] Pixel changes outside authorized masks remain within a documented codec
  or numerical tolerance and never alter protected areas materially.
- [ ] Every unsupported or unavailable capability is visible before execution
  and represented honestly in the result/provenance.
- [ ] Static-image upload-to-verified-export succeeds without manual recovery
  for at least 95% of supported-corpus jobs; manual intervention is measured,
  categorized, and never concealed.
- [ ] VTM/XLIFF and stored-manifest backward-compatibility suites pass.
- [ ] Video is either within its published beta thresholds or clearly separated
  from the 90%-complete static production claim.
- [ ] The architecture inventory, supported-quality matrix, benchmark report,
  release notes, and UI/API capability claims agree with the release commit.

## Scope control

Estimated against the weight table, the full list overshoots the 90% target by
roughly five points. That means it can be cut, and the cuts should come from the
items with the highest cost per point rather than from wherever work stalls
first.

**Cut candidates**, in order of willingness: **P1.24** (Garnish
source-conditioning — needs a human study to even score), **P2.8** (Memory
retrieval — improves reuse, not output correctness), **P1.21** (font-family
matching — 1,472 lines with five recorded dead ends; keep only the reference
corpus and the deletion of the stub catalog integration), **P1.23** (style
reconstruction — keep the provenance tagging, defer the effects), **P1.20**
(variable fonts — keep the `.ttc` face-identity fix, defer the rest), **P2.4**
(material classification — worthless unless Cleanse actually branches on it).

**Never cut**: P1.9, P1.25, P1.29, P1.14, P1.17. These are the items that stop
wrong output from reaching a user, and every one of them is cheap.

## Dependency structure

The largest sequencing risk in the original draft was invisible: most P1
acceptance targets are held-out-corpus numbers, so they silently inherited the
corpus schedule.

```
P0.1 (audit) ─────────────► P2.17 (generated metadata)
P0.2 (tests) ─────────────► P0.4 ─────► everything measurable
P0.3a (freeze 11/72) ─────► P1.1–P1.7, P1.13–P1.24, P1.26, P1.27
P0.3b (grow to 100/500) ──► P1.28 (calibration) ──► P0.5 tier thresholds
P0.4 (baselines) ─────────► P1.30 (latency), P2.1 (outcome logging)
P1.9 + P1.25 ─────────────► P1.29 ─────► exit gates 3, 4, 6
P1.20 ────────────────────► P2.5 (subfamily coverage)
```

Independent of all corpus work — start immediately: **P0.2, P1.9, P1.25, P1.29,
P2.6** (declare the translation boundary), **P2.7's geometry-mutation
assertion**, **P2.15** (Pydantic), **P2.3** (retire SAM).

## Recommended execution order

1. **P0.2 + P1.9 + P1.25 + P1.29**: restore the suite and close the output-safety
   holes. No corpus dependency; start here.
2. **P0.1, P0.3a, P0.4, P0.5**: restore trustworthy measurement and define the
   promise. Start P0.3b in parallel as its own workstream.
3. P1.1–P1.12: improve detection, boxes, transcription, and review UX.
4. P1.13–P1.18: make erasure candidate-based, independently checked, and safe.
5. P1.19–P1.27, P1.30, P1.31: close shaping/style gaps; hold latency and review
   rate.
6. P1.28 once P0.3b lands: calibrate acceptance.
7. P2.1–P2.9: improve predictive context, translation alignment, and reuse.
8. P2.10–P2.18: mature video and operational/release evidence.

Groups 1, 3 and 4 should produce the largest improvement in perceived output
dependability. Video, learned style transfer, and broad catalog integration
should not displace static detection, reconstruction, shaping, and verification
work until the critical exit gates are green.
