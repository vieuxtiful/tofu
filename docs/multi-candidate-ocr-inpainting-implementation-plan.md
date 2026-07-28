# Multi-Candidate OCR and Inpainting

## Implementation-ready plan for static images

Status: refined design for subsequent implementation  
Scope: Cicerone detection, Cleanse reconstruction, and pre-Scribe verification  
Explicitly out of scope: Tesseract, video processing, tracking, and temporal inpainting

## 1. Outcome

For every eligible static-image text region, ToFU will:

1. collect OCR observations from the available recognition paths;
2. form region hypotheses without prematurely discarding disagreeing boxes;
3. rank recognition hypotheses using calibrated, cross-pass evidence;
4. use PaddleOCR as the required independent verification engine;
5. generate multiple reconstruction candidates for neural-routed regions;
6. reject unsafe candidates using hard gates;
7. rank the remaining candidates using evidence that is independent of provider confidence;
8. select a candidate only when its evidence clears a calibrated acceptance threshold; and
9. re-OCR the cleansed pixels before Scribe, retrying with a revised mask or another candidate when residual source text is detected.

The system must fail closed: unavailable verification, complete candidate failure, or ambiguous evidence must produce review-required provenance rather than a silent auto-accept.

## 2. Design corrections to the starting plan

### 2.1 Separate localization, transcription, and verification

Bounding-box agreement and text agreement are different signals. A provider can find the correct region and transcribe it incorrectly, or transcribe a fragment correctly while missing part of the region. The implementation must therefore maintain:

- a region hypothesis, created from geometrically related observations;
- one or more transcription hypotheses within that region; and
- an independent PaddleOCR verification observation.

Do not replace the surviving `InstText` box merely because one transcription wins.

### 2.2 Do not restrict multi-candidate OCR to regions already routed as neural

Cleanse routing occurs after detection and depends on scene/background evidence. OCR arbitration must be usable before that decision. Run it for:

- all text hypotheses intersecting a scene surface marked textured or complex;
- low-confidence or unstable hypotheses;
- hypotheses with material transcription disagreement;
- vertical, stylized, or CJK hypotheses; and
- optionally all hypotheses in an explicit exhaustive mode.

The default policy should be `risk_based`, with `off` and `exhaustive` modes for testing and evaluation.

### 2.3 PaddleOCR verification is required, not a fallback alias

Tesseract is prohibited. EasyOCR may remain a proposal source, but it must not substitute for PaddleOCR in any field labeled `independent_verification`.

If PaddleOCR is unavailable or errors:

- retain proposal candidates;
- set verification state to `unavailable`;
- do not apply verification-derived confidence boosts;
- do not auto-accept a residual-sensitive cleanse result; and
- mark the affected region for review according to policy.

### 2.4 OCR silence is not proof that glyphs are gone

Residual glyphs may be visually obvious but no longer form a readable word. Post-cleanse rejection therefore needs two complementary gates:

- PaddleOCR residual-source/readable-text evidence; and
- non-OCR glyph-likeness evidence based on source-to-candidate edge/stroke persistence.

The first is the required independent OCR check. The second prevents false acceptance when OCR returns no text.

### 2.5 Separate hard rejection from ranking

A high composite score must never compensate for:

- detected residual source text;
- unacceptable change outside the permitted blend band;
- invalid dimensions or pixels;
- a severe structural discontinuity; or
- an unavailable required verification result under auto mode.

Run hard gates first. Rank only the candidates that remain eligible.

### 2.6 “Dependable default” is a promotion status

`diffstr_experimental` already exists in provider configuration. It becomes the preferred learned scene-text remover only after a pinned model/revision passes the promotion benchmark in Section 11. Until then:

- LaMa is the promoted general learned baseline when installed and validated;
- DiffSTR remains an evaluated candidate, not an automatic default;
- Telea remains a deterministic last resort and requires review on neural regions; and
- analytic flat/gradient reconstruction remains preferred where scene evidence supports it.

LaMa is not prompt-conditioned; do not add prompt or negative-prompt settings to its configuration.

## 3. Operating modes and policy

Add policy objects rather than more unrelated booleans:

```python
@dataclass(frozen=True)
class OCRAssessmentPolicy:
    mode: Literal["off", "risk_based", "exhaustive"] = "risk_based"
    verifier: Literal["paddleocr"] = "paddleocr"
    require_verifier_for_auto_accept: bool = True
    max_region_proposals: int = 8
    calibration_revision: str = "ocr-v1"


@dataclass(frozen=True)
class InpaintAssessmentPolicy:
    mode: Literal["single", "multi"] = "multi"
    max_neural_candidates: int = 3
    retry_budget: int = 2
    require_residual_verification: bool = True
    calibration_revision: str = "inpaint-v1"
```

Expose the policies through pipeline configuration and preserve compatibility by giving them defaults. Avoid adding `multi_candidate=True` and `residual_check=True` independently to several layers.

## 4. Provenance contract

Use JSON-serializable records in `InstText.ocr_provenance` and the existing `InstText.repair_provenance`. Keep runtime arrays and model objects out of manifests.

### 4.1 OCR observation

```python
@dataclass
class OCRObservation:
    observation_id: str
    backend: str
    backend_revision: str
    pass_tag: str
    text: str
    raw_confidence: float
    calibrated_confidence: Optional[float]
    bbox: BBox
    polygon: Optional[Polygon]
    language_hint: Optional[str]
    detected_script: Optional[str]
    runtime_ms: Optional[int]
    error: Optional[str] = None
```

### 4.2 Region hypothesis and decision

```python
@dataclass
class OCRHypothesisDecision:
    region_id: str
    member_observation_ids: List[str]
    selected_observation_id: Optional[str]
    selected_text: Optional[str]
    geometry_score: float
    transcription_score: float
    verification_state: Literal["agree", "disagree", "no_text", "unavailable", "error"]
    verification_observation_id: Optional[str]
    auto_accepted: bool
    review_required: bool
    reason_codes: List[str]
    score_breakdown: Dict[str, Optional[float]]
    policy_revision: str
```

### 4.3 Inpaint candidate decision

```python
@dataclass
class InpaintCandidateDecision:
    candidate_id: str
    provider: str
    provider_revision: str
    seed: Optional[int]
    elapsed_ms: int
    hard_rejections: List[str]
    metrics: Dict[str, Optional[float]]
    rank_score: Optional[float]
    eligible: bool
    selected: bool
    error: Optional[str] = None
```

The provenance must record thresholds, calibration revision, provider availability, attempted candidates, rejected candidates, selected candidate, retry history, and final review state.

## 5. Multi-candidate OCR assessment

### 5.1 Preserve observations from existing passes

Refactor existing detection paths so `run_multipass`, adaptive recognition, zoom detection, vertical splitting, Paddle rescue, and hybrid audit can emit `OCRObservation` records before union logic discards alternatives.

The initial implementation should use:

- EasyOCR standard/stylized/hard threshold passes as proposal observations;
- any existing crop/zoom reads as separately tagged proposal observations; and
- PaddleOCR as a proposal source when available and as a new, clean verification call.

A Paddle proposal read must not be reused as its own verification result. Verification uses a separately prepared crop and a separate invocation tagged `verification`.

### 5.2 Form region hypotheses

Build a small observation graph. Connect observations only when both geometry and layout permit it:

- polygon IoU or bbox IoU at or above a calibrated threshold;
- containment/coverage compatibility for fragment versus whole-region reads;
- compatible orientation; and
- compatible scene-surface membership.

Do not rely on `IoU > 0.3` alone. That can merge adjacent words or fail on a fragment contained in a larger sign box.

Use connected components followed by a split check for components containing mutually exclusive observations. Preserve:

- the consensus/union geometry for masking;
- the selected observation geometry for traceability; and
- all discarded alternatives in provenance.

### 5.3 Normalize text for comparison only

Create `normalize_for_ocr_agreement(text, language)` for scoring. It may normalize Unicode, whitespace, and script-appropriate punctuation/case. It must not overwrite the stored raw text.

Use character error rate or normalized edit similarity. For multi-token Latin text, word error rate may be an additional signal. Avoid a single global `SequenceMatcher` threshold for all scripts.

### 5.4 Calibrate provider confidence

Raw EasyOCR and PaddleOCR confidence values are not directly comparable. Fit per-backend calibration on held-out ground truth and store a calibration revision. Until calibration data exists:

- use confidence only as a within-backend ranking signal;
- cap its contribution to the cross-backend decision; and
- never describe the composite as a probability.

### 5.5 Score transcription hypotheses

Use a score breakdown whose unavailable terms are omitted and whose remaining weights are renormalized:

```text
proposal_score =
    calibrated backend confidence
  + cross-pass transcription stability
  + cross-backend agreement
  + geometry stability
  + language/script consistency
  + source-crop glyph support
```

Initial weights are hypotheses to benchmark, not fixed truth:

| Signal | Initial weight |
|---|---:|
| Cross-backend agreement | 0.30 |
| Calibrated backend confidence | 0.20 |
| Cross-pass stability | 0.20 |
| Geometry stability | 0.15 |
| Language/script consistency | 0.10 |
| Source-crop glyph support | 0.05 |

Do not use containing-surface coverage as a generic positive signal. Text is often intentionally much smaller than its sign or panel. Use surface membership and truncation/fragment evidence instead.

### 5.6 PaddleOCR verification

For each selected hypothesis:

1. derive a padded crop from consensus geometry;
2. retain the polygon/quad mask where available;
3. optionally rectify a high-confidence quad for recognition only;
4. run a fresh PaddleOCR crop-level read;
5. compare normalized verification text with the selected transcription;
6. compare Paddle geometry with the consensus region; and
7. write an explicit verification state.

Decision guidance:

- strong text and geometry agreement: eligible for auto-accept;
- partial agreement consistent with a fragment/whole relationship: keep but review unless calibrated otherwise;
- material disagreement: review, preserving both readings;
- Paddle detects no text on a clearly glyph-bearing source crop: review;
- Paddle unavailable/error: unverified, never silently “passed.”

Verification may boost a decision score but must not mutate a backend’s original confidence.

### 5.7 Detection integration point

Integrate after the existing adaptive/zoom/vertical proposal generation but before downstream correction stages that mutate text (`savor`, `wasabi`, and `menu`). Corrections must operate on the selected OCR read and append to `recognition_history`.

Retain CJK Paddle rescue for coverage recovery, but deduplicate its execution with the new assessor so Paddle is not unnecessarily initialized or called twice for the same proposal purpose.

## 6. Multi-candidate reconstruction

### 6.1 Candidate generation

For each compatible neural repair group:

1. freeze the untouched source image and exact group mask;
2. enumerate available, enabled providers;
3. order them by promoted status, task specialization, and resource policy;
4. run up to `max_neural_candidates`;
5. validate output shape, dtype, finite/range values, and protected-pixel preservation;
6. evaluate every valid candidate against the same evidence functions; and
7. select the highest-ranked eligible candidate.

Candidates must all see the same source pixels. A later provider must never consume an earlier provider’s output.

The existing `candidate_observer` should fire once per candidate with a stable `candidate_id`, followed by a group selection event. Preserve the current server-owned artifact boundary.

### 6.2 Provider set

For neural-routed static regions:

- promoted DiffSTR: preferred task-specific candidate after promotion;
- promoted LaMa: dependable general learned candidate;
- promoted BrushNet: optional generated candidate for complex texture;
- unpromoted providers: evidence/review candidates only;
- Telea: deterministic fallback candidate, not evidence that neural quality passed.

Provider errors are candidate-local. One timeout must not abort the group while another candidate can still run.

### 6.3 Hard rejection gates

Reject before ranking when any of the following occurs:

- invalid output, wrong size, invalid dtype, non-finite pixels, or missing image;
- protected-area change exceeds the calibrated tolerance outside the mask plus permitted feather band;
- PaddleOCR finds source-like residual text above the calibrated detection/similarity rule;
- glyph-persistence detector exceeds its calibrated limit;
- severe boundary seam;
- severe structural break with adequate structural evidence; or
- required verification is unavailable in an auto-accept path.

Do not auto-reject merely because OCR detects any text. Backgrounds can legitimately contain nearby or newly hallucinated text-like content. Rejection should consider overlap with the erased mask, confidence, geometry, and similarity to the source read. Novel high-confidence text should also reject, but under a distinct `hallucinated_text` reason.

### 6.4 Ranking metrics

Rank only eligible candidates. Suggested initial metrics:

- boundary color/gradient compatibility;
- outside-mask preservation;
- texture-spectrum compatibility across the fill and context ring;
- structural continuation;
- patch self-similarity/repetition consistency;
- residual glyph margin; and
- provider promotion/reliability prior as a small tie-breaker.

```text
rank_score =
    0.25 boundary_compatibility
  + 0.20 outside_preservation
  + 0.20 texture_compatibility
  + 0.20 structural_continuity
  + 0.10 residual_glyph_margin
  + 0.05 reliability_prior
```

Missing metrics are omitted and weights renormalized. A candidate needs a minimum evidence count; otherwise it is review-only.

### 6.5 Repeated texture seams

Use more than edge-density delta:

- compare gradient magnitude/orientation histograms across inner and outer boundary rings;
- compare local frequency or Gabor responses for brick, fabric, and regular patterns;
- detect duplicated patches by normalized cross-correlation against nearby source context; and
- penalize periodic phase discontinuities across the mask.

Keep these functions deterministic and provider-agnostic.

### 6.6 Structural breaks

Use boundary-crossing line/curve support rather than a raw count of Hough segments:

1. detect source edges in a context band;
2. find edge tracks that enter the mask;
3. predict their exit location/orientation;
4. measure candidate support near the prediction; and
5. report evidence coverage as well as continuity.

When the source contains no reliable crossing structure, return `None`, not a perfect score.

## 7. Perspective- and material-aware reconstruction

### 7.1 Perspective

Use the normalized quad already supported in `StyleProfil.transform["quad"]` when present and valid. Do not add a second generic `SceneRegion.perspective` field.

For a high-confidence, non-degenerate quad:

1. enlarge the source polygon using stroke- and scale-aware margins;
2. rectify source crop and mask with one homography;
3. run providers in rectified space;
4. run candidate gates in rectified and original image space where relevant;
5. warp the selected fill back;
6. composite only through the original-space mask and feather band; and
7. re-run outside-preservation and seam checks after inverse warping.

Fall back to original-space repair when the quad is missing, near-identity, degenerate, or excessively distorted.

### 7.2 Material

`SceneRegion.material` and `BgProfil.material` are free-form presentational descriptors today; `BgProfil.material` explicitly must not drive routing. Introduce a separate evidence-bearing profile:

```python
@dataclass
class ReconstructionProfile:
    material_class: Literal[
        "unknown", "painted_flat", "glass", "metal", "masonry",
        "wood", "fabric", "paper", "foliage"
    ] = "unknown"
    material_confidence: float = 0.0
    planar_confidence: float = 0.0
    periodic_texture_confidence: float = 0.0
    perspective_quad: Optional[Polygon] = None
    perspective_confidence: float = 0.0
    evidence: Dict[str, Any] = field(default_factory=dict)
```

Use this profile to tune:

- provider ordering;
- rectification eligibility;
- mask expansion;
- texture/structure metric selection; and
- review thresholds.

Material class must not directly auto-accept a provider. Low-confidence or unknown material uses the generic path.

Poisson blending is an optional candidate/post-process, not an automatic fix: it can damage texture or color gradients and must pass the same final gates.

## 8. Post-cleanse residual verification and retry

Run after all selected fills are composited and before Scribe.

For each cleansed instance:

1. crop the final cleansed pixels with context padding;
2. run a fresh PaddleOCR verification read;
3. test overlap, confidence, and similarity to the original source text;
4. compute source-to-cleanse glyph persistence inside the erase footprint;
5. store the full evidence;
6. if rejected, choose the next action from the retry policy; and
7. re-run all final gates after each retry.

Retry order:

1. select the next already-generated eligible candidate, if one exists;
2. expand the mask based on residual location/stroke evidence and re-run candidates;
3. use the safe deterministic fallback when policy permits;
4. stop at `retry_budget`; then retain the safest available result and mark review required.

Do not repeatedly apply inpainting to already-inpainted pixels. Each retry begins from the untouched source with a revised mask.

The Verify-layer residual metric remains a second line of defense, but it should call the shared Paddle residual service. It must not initialize a separate EasyOCR-only path and call it independent verification.

## 9. Shared services and dependency boundaries

Avoid importing the full Cicerone layer into Cleanse. Extract a narrow OCR service/interface used by both:

```python
class OCRRegionVerifier(Protocol):
    def verify_regions(
        self, asset: Any, regions: Sequence[RegionRequest]
    ) -> Sequence[OCRVerificationResult]: ...
```

Provide a Paddle-backed implementation using the existing isolated worker/bridge. Inject a fake verifier in tests.

Likewise, keep candidate evaluation pure where possible:

```python
evaluate_candidate(
    source, candidate, mask, context, verifier, policy
) -> InpaintCandidateDecision
```

This prevents circular layer imports and makes failure behavior testable.

## 10. Concrete file changes

| File | Required change |
|---|---|
| `src/tofu/core/types.py` | Add OCR observation/decision records, `ocr_provenance`, and `ReconstructionProfile`; retain JSON compatibility |
| `src/tofu/layers/cicerone.py` | Preserve pass observations; form hypotheses; arbitrate; invoke fresh Paddle verification before correction stages |
| `src/tofu/layers/ocr_verification.py` (new) | Narrow Paddle verification service, normalization, residual-source comparison, availability/error states |
| `src/tofu/layers/inpaint_providers.py` | Enumerate candidates; add per-candidate execution; split hard gates from rank metrics; select eligible candidate |
| `src/tofu/layers/cleanse.py` | Integrate multi-candidate groups, perspective wrapper, final residual verification, source-based retries, provenance |
| `src/tofu/layers/verify.py` | Reuse shared Paddle residual verifier; preserve second-line QA semantics |
| `src/tofu/core/pipeline.py` | Thread assessment policies and enforce pre-Scribe checkpoint |
| `src/tofu/utils/manifest_store.py` | Serialize/deserialize new optional provenance fields without breaking old manifests |
| `server/inpainting-providers.example.json` | Pin provider revision/hash fields; keep DiffSTR experimental until promotion |
| `server/main.py` | Expose candidate evidence and final selection without leaking local paths |

## 11. Evaluation and promotion gates

### 11.1 OCR dataset

Use a held-out corpus covering:

- Latin, CJK, mixed script, and vertical text;
- flat panels, textured surfaces, glass/reflections, masonry, fabric, and perspective signs;
- small, stylized, low-contrast, curved, and partially occluded text; and
- negative regions with no text.

Measure:

- detection precision/recall at polygon or bbox IoU thresholds;
- character error rate and word error rate where applicable;
- expected calibration error per backend;
- arbitration win/loss/tie versus best single backend;
- verification false-agreement and false-disagreement rates;
- percent sent to review; and
- latency by mode.

Promotion requirement: multi-candidate assessment must reduce transcription error without an unacceptable detection precision or review-rate regression. Set numeric release thresholds from the current baseline before implementation is declared complete.

### 11.2 Inpainting dataset

Use source images with manually prepared clean-background references where possible, plus blinded human review for real scenes without ground truth.

Measure:

- residual source text false-negative rate;
- hallucinated-text rate;
- boundary seam detection precision/recall;
- structural-break detection precision/recall;
- protected-area change;
- LPIPS/SSIM or task-appropriate reference metrics where ground truth exists;
- blinded preference against LaMa-only and Telea baselines;
- auto-accept precision; and
- p50/p95 latency and provider failure rate.

### 11.3 Learned scene-text remover promotion

Promote the pinned DiffSTR adapter/revision to the dependable default only when it:

- beats the promoted LaMa baseline on the scene-text corpus;
- does not regress protected-area preservation;
- meets the residual-text false-negative target;
- meets the hallucinated-text target;
- is deterministic or seed-recorded and reproducible;
- passes offline install/startup/timeout tests; and
- has an explicit rollback/config kill switch.

Promotion changes configuration/status, not hard-coded provider order.

## 12. Test plan

### Unit tests

- observation clustering: overlap, containment, adjacent-word separation, and orientation mismatch;
- text normalization and script-aware similarity;
- confidence calibration lookup and missing-calibration behavior;
- OCR scoring with missing signals and weight renormalization;
- Paddle states: agree, disagree, no text, unavailable, and error;
- candidate hard gates cannot be overridden by rank score;
- source-like residual versus novel hallucinated text;
- texture repetition seam and structural continuity with absent-evidence cases;
- output validation and protected feather-band semantics;
- deterministic provider ordering and candidate cap;
- perspective round trip, degenerate quad fallback, and post-warp recheck;
- retry always starts from source;
- retry budget exhaustion and review-required outcome; and
- old-manifest round trip with absent new fields.

### Integration tests

- EasyOCR proposals plus Paddle verification select the ground-truth read;
- Paddle proposal and fresh Paddle verification remain separate observations;
- provider timeout does not block other candidates;
- best eligible candidate is selected and all candidates reach `candidate_observer`;
- all neural candidates rejected leads to reviewed Telea fallback;
- residual found after compositing selects the next candidate;
- residual persists after retries and blocks auto-accept;
- Scribe never receives an auto-approved asset when required Cleanse verification is unavailable; and
- existing detection, Cleanse, Verify, API, and manifest tests remain unchanged and pass.

### Evaluation tests

Keep calibration/evaluation corpus tests separate from fast unit tests. Store model revision, provider configuration revision, thresholds, environment capability state, and random seed with every result.

## 13. Implementation sequence

### Phase 0 — Baseline and contracts

- freeze current OCR/inpainting metrics and latency;
- define policy and provenance schemas;
- add backward-compatible serialization tests;
- extract the Paddle region-verification service.

Exit: no behavior change; existing tests pass.

### Phase 1 — OCR observations and arbitration

- preserve pass observations;
- build region hypotheses;
- add script-aware comparison and calibrated scoring hooks;
- add fresh Paddle verification;
- integrate before correction stages.

Exit: measured OCR error improves or review is requested; no silent unverified acceptance.

### Phase 2 — Candidate evaluation framework

- separate hard gates from ranking;
- implement deterministic metrics and missing-evidence behavior;
- run multiple providers from the same source/mask;
- expose all candidate evidence.

Exit: selection is deterministic for fixed inputs/config/seeds and unsafe candidates cannot win.

### Phase 3 — Final residual checkpoint and retry

- run shared Paddle verification on final cleansed pixels;
- add glyph-persistence evidence;
- implement source-based candidate/mask retries;
- enforce pre-Scribe gate.

Exit: seeded residual failures are caught; retry exhaustion becomes review-required.

### Phase 4 — Perspective and reconstruction profile

- validate/rectify existing quads;
- add evidence-bearing reconstruction profile;
- tune candidate generation/evaluation by profile confidence;
- recheck after inverse warp.

Exit: perspective corpus improves without protected-area regression.

### Phase 5 — Learned remover promotion

- pin and provision the DiffSTR adapter;
- run promotion benchmark;
- promote through configuration only if gates pass;
- document rollback.

Exit: learned scene-text removal is a dependable default by measured status, not by name.

## 14. Definition of done

- No Tesseract code path exists.
- No video or temporal inpainting behavior is added.
- Multiple available OCR proposal paths are retained and scored for eligible regions.
- PaddleOCR performs a fresh independent verification pass.
- Unavailable Paddle verification is explicit and cannot masquerade as success.
- Neural regions produce and compare multiple candidates when policy and availability allow.
- Hard rejection gates run before candidate ranking.
- Residual glyphs, repeated texture seams, protected-area changes, and structural breaks have independent evidence.
- Final cleansed pixels are re-verified before Scribe.
- Retries start from the untouched source and are bounded.
- Perspective reconstruction uses the existing quad contract.
- Material routing uses confidence-bearing evidence, not a free-form label.
- DiffSTR is promoted only after benchmark gates; LaMa remains the learned baseline until then.
- Provenance explains every candidate, metric, threshold, rejection, selection, fallback, and review decision.
- Old manifests remain readable and the full regression suite passes.

