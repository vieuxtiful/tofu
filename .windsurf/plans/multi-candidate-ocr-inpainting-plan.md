# Multi-Candidate OCR Assessment & Enhanced Cleanse/Inpainting Plan

## Overview

This plan implements two interrelated capability upgrades for ToFU:

1. **Multi-candidate OCR assessment** — run multiple OCR providers on neural regions, score each candidate by cross-pass agreement, and select the best. Includes independent OCR-based residual glyph detection after cleansing using PaddleOCR.

2. **Enhanced cleanse & inpainting** — run multiple inpainting providers per region, compare candidates with an enhanced quality gate, reject fills with artifacts, and select the best using independent evidence. Incorporates perspective/material-aware reconstruction and bundles learned scene-text removal as a dependable default.

---

## Part 1: Multi-Candidate OCR Assessment

### 1.1 Current State

- `cicerone.py` supports `EasyOCRBackend` and `PaddleOCRBackend` behind an `OCRBackend` ABC.
- Multi-pass detection exists: threshold sweeps for EasyOCR (`run_multipass`), single pass for PaddleOCR.
- `should_paddle_rescue` / `run_paddle_rescue` provides a CJK-specific rescue mechanism — PaddleOCR is only invoked when EasyOCR confidence is low or scene surfaces are uncovered, and only for CJK-dominant scenes.
- `hybrid_audit` exists for `auto`/`hybrid` engine mode but is a late-stage text-level audit, not a per-region candidate scoring system.
- No generalized framework runs all available backends on every neural region, scores them, and selects the best.

### 1.2 Design

#### 1.2.1 OCR Candidate Record

New dataclass in `cicerone.py` or `types.py`:

```python
@dataclass
class OCRCandidate:
    backend: str          # "easyocr", "paddleocr", ...
    text: str
    confidence: float
    bbox: BBox
    polygon: Optional[List[Tuple[int, int]]]
    pass_tag: str         # "standard", "stylized", "hard", "paddle_default", ...
    language: Optional[str]
```

#### 1.2.2 Multi-Backend Detection for Neural Regions

**Where:** New function `detect_multi_candidate()` in `cicerone.py`, called from `detect()` when `multipass=True` and multiple backends are available.

**What it does:**
1. Run the primary backend (EasyOCR by default) with existing multipass threshold sweeps.
2. If PaddleOCR is available, run it on the same asset (full-frame or per-scene-region crops).
3. For each detected region (matched by bbox IoU > 0.3), collect all candidates from all backends/pass-tags.
4. For regions where only one backend produced a detection, keep that candidate directly.

**Neural regions** are defined as: regions routed to the `neural` inpainting strategy by `inpaint_providers.route()` — i.e., textured/complex surfaces where deterministic fills won't work. These are the regions where OCR accuracy matters most (complex backgrounds, stylized text, CJK signage).

#### 1.2.3 Candidate Scoring & Arbitration

New function `score_ocr_candidates()` in `cicerone.py`:

**Scoring signals (weighted):**
- **Backend confidence** (0.30): Each backend's own confidence score, normalized.
- **Cross-backend text agreement** (0.35): String similarity (SequenceMatcher ratio) between candidates from different backends. High agreement = high score. When only one backend detected the region, this term is 0.5 (neutral).
- **Pass stability** (0.20): For multi-pass backends (EasyOCR), whether the same text was detected across threshold sweeps. Stable detections score higher.
- **Scene surface coverage** (0.15): How well the detection's bbox covers its containing scene surface (uses existing `_surface_coverage_frac`). Detections that account for the full surface extent score higher.

**Selection rule:**
- If two backends agree (text similarity > 0.8), select the higher-confidence one.
- If they disagree, select the candidate with the highest composite score.
- Record all candidates and the selection rationale on the `InstText` (new `ocr_provenance` field).

#### 1.2.4 Verification Scan with PaddleOCR

After the primary detection and candidate selection, run a **verification pass** using PaddleOCR on the same regions:

1. For each selected `InstText`, crop the region and run PaddleOCR's `detect_in_regions` on it.
2. Compare the verification read against the selected text.
3. If verification text agrees (similarity > 0.85), boost confidence.
4. If verification text disagrees significantly (similarity < 0.5), flag the region for review and record both readings in `ocr_provenance`.

**Why PaddleOCR for verification:** Higher measured accuracy on dense/vertical CJK (4x recall, 4x transcription accuracy per existing eval notes). Using an independent backend for verification avoids confirmation bias from the same model re-reading its own output.

#### 1.2.5 Integration into `detect()`

Add a new parameter `multi_candidate: bool = True` to `detect()`. When enabled:
1. After all existing detection/refinement passes complete, call `detect_multi_candidate()` on the surviving detections.
2. Run `score_ocr_candidates()` to arbitrate.
3. Run the PaddleOCR verification scan.
4. Store results in `inst.ocr_provenance`.

### 1.3 Files to Modify

| File | Change |
|------|--------|
| `src/tofu/core/types.py` | Add `OCRCandidate` dataclass; add `ocr_provenance` field to `InstText` |
| `src/tofu/layers/cicerone.py` | Add `detect_multi_candidate()`, `score_ocr_candidates()`, `verify_with_paddle()`; integrate into `detect()` |
| `tests/test_cicerone_multicandidate.py` (new) | Test multi-candidate scoring, agreement, verification, fallback |

### 1.4 Testing

- Unit test `score_ocr_candidates()` with synthetic candidates: agreement case, disagreement case, single-backend case.
- Test that verification scan runs when PaddleOCR is available and is skipped gracefully when not.
- Test that `ocr_provenance` is populated with all candidates and selection rationale.
- Regression: existing `test_cicerone*.py` tests must pass unchanged.

---

## Part 2: Independent OCR-Based Residual Glyph Detection

### 2.1 Current State

- `verify.py` already has `_residual_source_text_score()` which re-OCRs the cleansed crop and compares against the original source string.
- This metric uses **EasyOCR** (via `_get_reader`), not PaddleOCR.
- It runs in the **Verify layer** (post-scribe), not in Cleanse (post-cleanse, pre-scribe).
- The penalty is a score multiplier in the QA report, not a trigger for re-inpainting.

### 2.2 Design

#### 2.2.1 Post-Cleanse Residual Check in Cleanse Layer

New function `detect_residual_glyphs()` in `cleanse.py`:

1. After `erase()` completes its inpainting strategies, for each instance that was cleansed:
   - Crop the cleansed region (from the `working` array, before scribe).
   - Run PaddleOCR's `detect_in_regions` on the crop (via `cicerone.PaddleOCRBackend`).
   - If PaddleOCR detects text with confidence > 0.5, compare against the original source text.
   - If similarity > 0.3 (configurable threshold `RESIDUAL_GLYPH_THRESHOLD`), flag as residual.
2. For flagged regions:
   - **Re-inpaint**: Expand the mask by 2px dilation and re-run the inpainting provider (or Telea fallback).
   - **Re-check**: Run the residual check once more on the re-inpainted result.
   - **Max retries**: 2. If still residual after retries, mark `review_required = True` and record evidence.
3. Record residual check results in `inst.repair_provenance["residual_check"]`.

#### 2.2.2 Use PaddleOCR for Residual Detection

- Import `cicerone.PaddleOCRBackend` in `cleanse.py`.
- Use `PaddleOCRBackend.detect_in_regions()` for the residual check.
- Fallback to EasyOCR if PaddleOCR is not available (degraded but functional).
- The existing `verify._residual_source_text_score()` remains as a second line of defense in the QA layer, but should also be updated to prefer PaddleOCR when available.

#### 2.2.3 Integration into `erase()`

Add parameter `residual_check: bool = True` to `erase()`. When enabled:
1. After all fill strategies complete (flat, gradient, neural, telea), run `detect_residual_glyphs()`.
2. For flagged regions, re-inpaint and re-check.
3. Return the cleansed asset with residual check evidence recorded.

### 2.3 Files to Modify

| File | Change |
|------|--------|
| `src/tofu/layers/cleanse.py` | Add `detect_residual_glyphs()`, integrate into `erase()`, add `residual_check` parameter |
| `src/tofu/layers/verify.py` | Update `_residual_source_text_score()` to prefer PaddleOCR when available |
| `src/tofu/core/types.py` | Add `residual_check` dict to `RepairOutcome` or `InstText.repair_provenance` |
| `tests/test_cleanse_residual.py` (new) | Test residual detection, re-inpaint retry, max retry exhaustion, PaddleOCR unavailable fallback |

### 2.4 Testing

- Synthetic test: draw text on an image, "cleanse" with a deliberately incomplete mask, verify residual is detected.
- Test re-inpaint retry: mock `inpaint_providers.repair` to return a clean fill on second call.
- Test max retry exhaustion: mock repair to always leave residual, verify `review_required` is set.
- Test PaddleOCR unavailable: verify graceful fallback to EasyOCR.
- Regression: existing `test_cleanse_*.py` tests must pass.

---

## Part 3: Enhanced Cleanse & Inpainting

### 3.1 Current State

- `inpaint_providers.route()` selects **one** provider per region based on texture/complexity.
- `quality_gate()` scores one candidate per group (seam, outside_delta, residual_edge).
- Only one neural provider runs per group; no multi-candidate comparison.
- No perspective/material-aware reconstruction beyond `BgProfil.texture` (flat/smooth_gradient/textured).
- `SceneRegion` has `texture` and `material` attributes but they're only used for routing, not for conditioning inpainting.
- No dedicated scene-text removal model (DiffSTR is listed as `diffstr_experimental` but not provisioned).

### 3.2 Design

#### 3.2.1 Multi-Candidate Inpainting

New function `repair_multi_candidate()` in `inpaint_providers.py`:

1. For a given group (image + mask), identify all **available and enabled** neural providers.
2. Run each available provider on the same (image, mask) pair.
3. Score each candidate with the enhanced quality gate (§3.2.2).
4. Select the best-scoring candidate that passes the gate.
5. If no candidate passes, fall back to Telea.
6. Record all candidates, scores, and selection rationale.

**Provider selection for multi-candidate:**
- Always include LaMa if available (strong for repeating/global structure).
- Include BrushNet if available (general inpainting, diffusion-based).
- Include DiffSTR if available (dedicated scene-text removal).
- For flat/smooth_gradient regions, the deterministic analytic fill remains the primary strategy; multi-candidate only applies to `neural`-routed regions.

**Integration into `cleanse.erase()`:**
- Replace the single-provider loop in the neural group section with `repair_multi_candidate()`.
- The `candidate_observer` callback fires for each candidate, not just the selected one.

#### 3.2.2 Enhanced Quality Gate

Enhance `quality_gate()` in `inpaint_providers.py` with new rejection signals:

**Existing signals (retained):**
- `seam_score` (0.62 weight) — color gap at mask boundary.
- `outside_delta` (0.28 weight) — pixels outside mask changed.
- `residual_edge_score` (0.10 weight) — edge density persistence inside mask.

**New signals:**

1. **Residual glyph detection (OCR-based)** (new, high-weight):
   - Run PaddleOCR on the candidate's masked region.
   - If any text is detected with confidence > 0.5, the candidate is **auto-rejected** regardless of other scores.
   - This is the strongest signal: visible text remnants are an unconditional failure.
   - Score: `ocr_residual_score = 1.0 - detected_text_confidence` (1.0 = no text found = good).

2. **Texture seam detection** (new):
   - Compare local texture statistics (edge density, chroma variance) in the border ring between the original and the candidate.
   - A large discrepancy indicates the inpainting introduced a visible texture seam.
   - Score: `texture_seam_score = max(0, 1.0 - |edge_density_delta| / threshold)`.

3. **Structural break detection** (new):
   - Detect line/edge discontinuities crossing the mask boundary using Hough line segments or Canny edge continuity.
   - Lines that enter the mask but don't exit (or exit at a different angle) indicate a structural break.
   - Score: `structural_continuity_score` based on the fraction of boundary-crossing edges that are continuous.

**New composite score:**
```
score = 0.40 * seam_score
      + 0.20 * outside_score
      + 0.15 * ocr_residual_score
      + 0.15 * texture_seam_score
      + 0.10 * structural_continuity_score
```

**Auto-reject conditions (override score):**
- OCR detects residual text with confidence > 0.5 → reject.
- `outside_delta` > 0.25 → reject (existing).
- `structural_continuity_score` < 0.3 → reject.

**Pass threshold:** `score >= 0.85` (raised from 0.90 to reflect the more discriminative signals; the OCR auto-reject handles the most dangerous failure mode that was previously only weakly caught).

#### 3.2.3 Perspective/Material-Aware Reconstruction

**Approach: Enhanced routing + pre/post-processing (Option B+C from brainstorming)**

1. **Enhanced routing in `inpaint_providers.route()`:**
   - Use `SceneRegion.material` to select provider-specific parameters:
     - `material == "metal"` → LaMa with prompt guidance for metallic surfaces (BrushNet).
     - `material == "glass"` → LaMa (diffusion tends to hallucinate reflections; LaMa's structure propagation is safer).
     - `material == "fabric"` → BrushNet with texture-aware prompt.
     - `material == "stone"` → LaMa (repeating texture).
   - Use `SceneRegion.perspective` (if available) to adjust mask dilation: perspective-foreshortened text needs wider mask margins to catch skewed anti-aliased edges.

2. **Pre-processing:**
   - For perspective-distorted regions, rectify the crop before inpainting (using the region quad if available), then warp the result back.
   - This gives the inpainting model a cleaner, axis-aligned input.

3. **Post-processing:**
   - After inpainting, apply a texture-consistency check: sample texture statistics from the border ring and verify the fill matches. If not, blend with a Poisson blend (seamless cloning) to harmonize.

**New function `_perspective_aware_repair()` in `cleanse.py`:**
- Checks if the region has a quad (perspective) descriptor.
- If so, rectifies → inpaints → warps back.
- Falls through to standard repair for axis-aligned regions.

#### 3.2.4 Learned Scene-Text Removal as Default

**Approach: Integrate DiffSTR as a first-class provider + improve LaMa config**

1. **DiffSTR integration:**
   - Add `diffstr` as a new provider in `inpainting-providers.json` (runtime: `external_command` or dedicated runner).
   - DiffSTR is specifically trained for scene-text removal — it should be the **preferred** provider for neural regions when available.
   - Update `_preferred_neural_provider()` to prioritize DiffSTR for text-removal tasks.
   - Provider priority order: `diffstr` > `lama` > `brushnet_experimental` (DiffSTR is purpose-built; LaMa is general-purpose but reliable; BrushNet is the diffusion fallback).

2. **LaMa as dependable default:**
   - LaMa remains the default when DiffSTR is not provisioned.
   - Add LaMa-specific prompt/negative-prompt configuration tuned for text removal (already partially done in BrushNet config).
   - Document LaMa as the "dependable default" in provider statuses.

3. **Bundle configuration:**
   - Ship a default `inpainting-providers.json` with LaMa enabled and promoted by default (when the model is present).
   - DiffSTR is opt-in (experimental) until independently provisioned.

### 3.3 Files to Modify

| File | Change |
|------|--------|
| `src/tofu/layers/inpaint_providers.py` | Add `repair_multi_candidate()`, enhance `quality_gate()`, update `route()` for material-aware routing, update `_preferred_neural_provider()` |
| `src/tofu/layers/cleanse.py` | Integrate multi-candidate repair, add `_perspective_aware_repair()`, add residual glyph detection (§2.2) |
| `src/tofu/core/types.py` | Add `material` and `perspective` to `SceneRegion` if not already present; add `InpaintCandidate` dataclass |
| `server/inpainting-providers.example.json` | Add DiffSTR provider config template |
| `tests/test_inpaint_multicandidate.py` (new) | Test multi-candidate selection, enhanced quality gate rejections, perspective-aware repair |
| `tests/test_inpaint_provider_routing.py` | Update for new routing logic |

### 3.4 Testing

- Test `repair_multi_candidate()` with mocked providers: two providers return different candidates, verify best is selected.
- Test enhanced quality gate: synthetic candidate with residual text → auto-rejected.
- Test texture seam detection: synthetic candidate with visible seam → rejected.
- Test structural break detection: synthetic candidate with broken line → rejected.
- Test perspective-aware repair: mock quad descriptor, verify rectify→inpaint→warp pipeline.
- Test DiffSTR routing: when available, DiffSTR is preferred for neural regions.
- Regression: existing `test_inpaint_*.py` and `test_cleanse_*.py` tests must pass.

---

## Implementation Order

1. **Types** (`types.py`) — Add `OCRCandidate`, `InpaintCandidate`, `ocr_provenance` field, any missing `SceneRegion` attributes.
2. **Multi-candidate OCR** (`cicerone.py`) — `detect_multi_candidate()`, `score_ocr_candidates()`, `verify_with_paddle()`, integrate into `detect()`.
3. **Residual glyph detection** (`cleanse.py`) — `detect_residual_glyphs()`, integrate into `erase()`.
4. **Enhanced quality gate** (`inpaint_providers.py`) — Add OCR residual, texture seam, structural break signals.
5. **Multi-candidate inpainting** (`inpaint_providers.py`) — `repair_multi_candidate()`, integrate into `cleanse.erase()`.
6. **Perspective/material-aware repair** (`cleanse.py`, `inpaint_providers.py`) — Enhanced routing, `_perspective_aware_repair()`.
7. **DiffSTR integration** (`inpaint_providers.py`, config) — Provider config, routing priority.
8. **Verify layer update** (`verify.py`) — Prefer PaddleOCR for residual text score.
9. **Tests** — Write all new tests, run full regression.

## Forbidden

- No Tesseract integration (low accuracy).
- No video pipeline or temporal inpainting at this stage.
- No changes to existing test assertions (only additions).

## Risks & Mitigations

| Risk | Mitigation |
|------|-----------|
| PaddleOCR not available in dev/test | All PaddleOCR-dependent features degrade gracefully to EasyOCR or skip |
| Multi-candidate inpainting is slow (N providers × M groups) | Only run multi-candidate for neural-routed regions; deterministic fills remain single-pass |
| Enhanced quality gate too strict → false rejections | Auto-reject only for OCR-residual (high confidence); other signals are scored, not hard-rejected |
| Perspective rectification introduces artifacts | Only rectify when quad confidence is high; fall through to standard repair otherwise |
| DiffSTR not provisioned in most deployments | LaMa remains the dependable default; DiffSTR is strictly opt-in |
