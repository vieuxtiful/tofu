# OCR and Inpainting Enhancement Plan

## Scope

Four focused work streams, no Tesseract, no video/temporal inpainting:

1. Fixture/tolerance regression suite (detection + cleanse + verify)
2. Automatic backend/device capability reporting
3. Hardened low-confidence multilingual OCR arbitration
4. Versioned correction dictionaries and gazetteers

---

## 1. Fixture/Tolerance Regression Suite

### Goal

A dedicated test module that exercises the full pipeline (detection → cleanse → verify) against the 14 existing fixture pairs in `tests/fixtures/` (`.png` + `.gt.json`), measuring accuracy with script-aware tolerance bands and generating a JSON regression report.

### Tolerance Metrics

| Stage | Metric | Tolerance |
|-------|--------|-----------|
| **Detection** | Bbox IoU vs ground truth | ≥ 0.60 (Latin), ≥ 0.50 (CJK/mixed) |
| **Detection** | Text CER vs ground truth | ≤ 0.15 (Latin), ≤ 0.25 (CJK/mixed) |
| **Detection** | Region recall | ≥ 0.80 (all scripts) |
| **Cleanse** | Residual glyph absence | 0 glyph-edge pixels inside mask after inpaint |
| **Cleanse** | Background SSIM (masked region vs surrounding ring) | ≥ 0.70 |
| **Verify** | QA score | ≥ 0.80 (the existing `qa_threshold`) |

### Files to Create

- **`tests/test_fixture_regression.py`** — the regression suite itself
  - `load_all_fixtures()` — parametrized fixture loader iterating `tests/fixtures/*.gt.json`
  - `test_detection_bbox_iou()` — runs `cicerone.detect()` on each fixture, compares detected bboxes to ground truth via IoU
  - `test_detection_text_cer()` — compares detected text to ground truth via `verify.error_rates()` CER
  - `test_detection_region_recall()` — measures how many ground-truth regions were detected at all
  - `test_cleanse_residual_glyphs()` — runs cleanse on each fixture, checks `quality_gate` residual_edge_score and Canny edge density inside mask
  - `test_cleanse_background_ssim()` — SSIM between inpainted region's surrounding ring and the inpainted interior
  - `test_verify_qa_score()` — runs `verify.assess()` on the cleansed manifest, checks QA score ≥ threshold
  - `pytest_generate_regression_report()` — session-scope fixture that writes `tests/eval_out/fixture_regression_report.json` with per-fixture, per-metric results

### Files to Modify

- **`tests/conftest.py`** — add `script_type` fixture parameter (latin / cjk / mixed) derived from `.gt.json` content, and a `regression_tolerance` helper that returns per-script tolerance bands

### Design Notes

- Uses existing `verify.error_rates()` for CER (already script-aware, already tested in `test_error_rates_and_calibration.py`)
- Uses existing `inpaint_providers.quality_gate()` for cleanse quality metrics (residual_edge_score, seam_score, texture_score)
- SSIM via `skimage.metrics.structural_similarity` (already a dependency in the project's requirements)
- Detection tests will use `cicerone.detect()` with `EasyOCRBackend` (skip if `easyocr` not installed, same pattern as existing tests)
- Cleanse tests will use `cleanse.cleanse()` on the detected manifest, then measure quality
- All tests are parametrized over fixtures, so adding a new `.png` + `.gt.json` pair automatically extends coverage
- The JSON report includes: fixture name, script type, per-metric pass/fail, actual values, and tolerance bands — enabling CI regression detection

---

## 2. Automatic Backend/Device Capability Reporting

### Goal

A new unified `/api/capabilities` endpoint that aggregates all backend, device, and resource statuses into one response. Used internally for routing decisions and externally for UI display.

### API Response Shape

```json
{
  "device": {
    "gpu_available": false,
    "gpu_name": null,
    "cuda_version": null,
    "gpu_memory_mb": null,
    "cpu_count": 8
  },
  "ocr": {
    "default_engine": "easyocr",
    "engines": {
      "easyocr": {"available": true, "languages": ["en", "ja", "zh", "ko", ...]},
      "paddleocr": {"available": false, "reason": "isolated venv missing", "languages": ["en", "ch", "japan"]},
      "null": {"available": true}
    }
  },
  "inpainting": {
    "providers": [...],  // reuse existing provider_statuses()
    "multi_candidate": true
  },
  "fonts": {
    "registry_loaded": true,
    "font_count": 42,
    "script_coverage": {"latin": true, "cjk": true, "arabic": false}
  },
  "correction_dicts": {
    "savor": {"version": "1.0.0", "entry_count": 7},
    "wasabi": {"version": "1.0.0", "entry_count": 3},
    "menu": {"version": "1.0.0", "entry_count": 8}
  }
}
```

### Files to Create

- **`src/tofu/utils/capability.py`** — the capability introspection module
  - `device_info()` — GPU/CUDA detection via `torch.cuda.is_available()` (fail-open: returns CPU-only if torch not installed)
  - `ocr_engine_statuses()` — checks EasyOCR (import test), PaddleOCR (`is_available()`), NullBackend (always true)
  - `font_coverage()` — loads `FontRegistry` from the server's font directory, reports script coverage
  - `correction_dict_versions()` — reads version metadata from the external JSON dictionaries (see section 4)
  - `all_capabilities()` — aggregates everything into the response shape above

### Files to Modify

- **`server/main.py`** — add `@app.get("/api/capabilities")` endpoint calling `capability.all_capabilities()`
- **`server/main.py`** — use `capability.ocr_engine_statuses()` in the detect stream's initial event so the frontend knows engine availability before detection starts

### Design Notes

- GPU detection uses `torch.cuda.is_available()` and `torch.cuda.get_device_name()` — fail-open to CPU-only if torch is not installed
- OCR engine checks mirror the existing `_engine_from_env()` / `get_backend()` logic in `cicerone.py`
- Inpainting providers reuse the existing `inpaint_providers.provider_statuses()` — no duplication
- Font coverage reuses `FontRegistry` from `tofu.layers.fonts` — checks which Unicode script ranges have at least one font
- Correction dict versions come from the external JSON files' `version` field (see section 4)
- The endpoint is read-only, no side effects, safe to call at any time
- Internal use: `cicerone.detect()` can consult `ocr_engine_statuses()` to skip PaddleOCR rescue when Paddle is unavailable (already done ad-hoc, now centralized)

---

## 3. Hardened Low-Confidence Multilingual OCR Arbitration

### Goal

Refine and generalize the existing arbitration mechanisms (`hybrid_audit`, `skim_audit`, `assess_multi_candidate_ocr`) with better scoring, configurable thresholds, and broader script coverage beyond the current CJK/risky-read focus.

### Current State

- `hybrid_audit` — PaddleOCR adjudicates risky CJK EasyOCR reads (terminal dash artifacts)
- `skim_audit` — PaddleOCR as cross-engine second opinion for script-less reads that `skim` couldn't judge
- `assess_multi_candidate_ocr` — attaches Paddle verification to risky reads, scoring on primary confidence + similarity + verification confidence
- `skim.needs_arbitration()` — nominates reads with too many characters for their bbox
- `ScriptDetector` — classifies text into script families via Unicode ranges
- `_disambiguate_ja_zh` — Japanese vs Chinese disambiguation via kana presence

### Enhancements

#### 3a. Generalized Arbitration Scoring (`cicerone.py`)

- **Refactor `assess_multi_candidate_ocr()` scoring** into a configurable `ArbitrationScoring` dataclass:
  - `primary_confidence_weight`: float = 0.4
  - `similarity_weight`: float = 0.35
  - `verification_confidence_weight`: float = 0.25
  - `auto_accept_threshold`: float = 0.85
  - `flag_for_review_threshold`: float = 0.60
  - Currently hardcoded; making it configurable enables tuning per deployment
- **Add `arbitration_decision()` function** that returns one of `accept`, `reject`, `flag_review` based on the composite score, with a reason string for the audit trail

#### 3b. Enhanced Script Detection (`cicerone.py`)

- **Extend `ScriptDetector`** with Korean (Hangul) and Arabic script ranges — currently only han/hiragana/katakana/latin
- **Add `script_confidence()` method** — returns a float indicating how much of the text falls into the dominant script (a mixed-script read with 2 han chars and 5 latin chars should weight the latin evidence higher)
- **Use `script_confidence()` in `needs_arbitration()`** — a low-confidence read with mixed scripts should be arbitrated differently than a pure-script read

#### 3c. Broader `skim_audit()` Coverage (`cicerone.py`)

- **Extend `skim_audit()` to consider `skim.needs_arbitration()` nominations** — currently it only handles script-less reads; reads that `needs_arbitration()` flags (too dense for their bbox) should also get a PaddleOCR second opinion
- **Add density-aware arbitration**: when `char_density` is very low (< 4 px/char), require PaddleOCR to read the SAME number of characters (not just "any text") before accepting — a dense read that Paddle reads as fewer characters is evidence of over-segmentation, not a wrong read

#### 3d. Arbitration Audit Trail (`cicerone.py`)

- **Standardize arbitration results** into an `ArbitrationRecord` dataclass:
  - `inst_id`, `original_text`, `candidate_text`, `decision` (accept/reject/flag_review), `composite_score`, `factors` (dict of individual scores), `reason`
  - Attached to `inst.recognition_history` (already used by `hybrid_audit`)
- This makes all arbitration decisions reviewable in the same format regardless of which mechanism triggered them

### Files to Modify

- **`src/tofu/core/types.py`** — add `ArbitrationScoring` and `ArbitrationRecord` dataclasses
- **`src/tofu/layers/cicerone.py`** — refactor scoring, extend `ScriptDetector`, broaden `skim_audit()`, add `arbitration_decision()`
- **`src/tofu/layers/skim.py`** — no changes needed (already provides `needs_arbitration()` and `char_density()`)

### Tests

- **`tests/test_hybrid_arbitration.py`** — extend with:
  - Korean script detection test
  - Arabic script detection test
  - Mixed-script confidence weighting test
  - Configurable scoring threshold test (auto-accept at 0.85, flag at 0.60, reject below)
  - Density-aware arbitration test (dense read where Paddle reads fewer chars → reject)
- **`tests/test_multi_candidate_assessment.py`** — extend with:
  - `ArbitrationScoring` configuration test
  - `ArbitrationRecord` audit trail test
  - `arbitration_decision()` boundary tests

---

## 4. Versioned Correction Dictionaries and Gazetteers

### Goal

Move the hardcoded dictionaries in `savor.py`, `wasabi.py`, and `menu.py` to external JSON files with version fields, loaded at runtime. This enables non-code updates, reproducible audit trails, and version reporting in the capability endpoint.

### External JSON File Structure

**`src/tofu/layers/data/savor.json`**
```json
{
  "version": "1.0.0",
  "description": "CRNN+CTC glyph confusion pairs (letter <-> digit)",
  "confusable_glyphs": {
    "5": "S", "S": "5",
    "0": "O", "O": "0",
    "1": "I", "I": "1",
    "8": "B", "B": "8",
    "2": "Z", "Z": "2",
    "6": "G", "G": "6"
  },
  "dakuten_map": {
    "カ": ["ガ", null],
    "キ": ["ギ", null],
    ...
  },
  "dakuten_langs": ["ja"]
}
```

**`src/tofu/layers/data/wasabi.json`**
```json
{
  "version": "1.0.0",
  "description": "Simplified-Chinese to Japanese shinjitai glyph normalization",
  "simplified_to_japanese": {
    "剧": "劇",
    "烧": "焼",
    "岛": "島"
  }
}
```

**`src/tofu/layers/data/menu.json`**
```json
{
  "version": "1.0.0",
  "description": "Known place/establishment names and signage words for gazetteer correction",
  "known_places": [
    ["歌舞伎町一番街", "ja"],
    ["劇場通り", "ja"],
    ...
  ],
  "known_signage": [
    ["歓迎", "ja"],
    ["入口", "ja"],
    ...
  ],
  "thresholds": {
    "confidence_floor": 0.6,
    "similarity_floor": 0.5,
    "whole_string_min_candidate_len": 3,
    "substring_similarity_floor": 0.5,
    "substring_string_tier_min_len": 4,
    "substring_string_tier_max_diffs": 1,
    "corroboration_min_siblings": 2
  }
}
```

### Files to Create

- **`src/tofu/layers/data/savor.json`** — extracted from `savor.py` constants
- **`src/tofu/layers/data/wasabi.json`** — extracted from `wasabi.py` constants
- **`src/tofu/layers/data/menu.json`** — extracted from `menu.py` constants
- **`src/tofu/layers/data/__init__.py`** — shared data-loading utility:
  - `load_dict(name: str) -> dict` — loads `data/{name}.json`, caches the result, returns the parsed dict
  - `dict_version(name: str) -> str` — returns the `version` field from the loaded JSON
  - `dict_entry_count(name: str) -> int` — returns the number of entries in the primary collection

### Files to Modify

- **`src/tofu/layers/savor.py`** — replace hardcoded `CONFUSABLE_GLYPHS`, `DAKUTEN_MAP`, `DAKUTEN_LANGS` with loads from `data/savor.json` via `load_dict("savor")`. Keep the module-level constants as references to the loaded values for backward compatibility (all existing code that does `savor.CONFUSABLE_GLYPHS` continues to work).
- **`src/tofu/layers/wasabi.py`** — replace hardcoded `SIMPLIFIED_TO_JAPANESE` with load from `data/wasabi.json`. Same backward-compatibility pattern.
- **`src/tofu/layers/menu.py`** — replace hardcoded `KNOWN_PLACES`, `KNOWN_SIGNAGE`, and threshold constants with loads from `data/menu.json`. Same backward-compatibility pattern.
- **`src/tofu/utils/capability.py`** — use `dict_version()` and `dict_entry_count()` to populate the `correction_dicts` section of the capability report.

### Design Notes

- JSON files are loaded once at module import time and cached — no per-detection file IO
- The module-level constants (`CONFUSABLE_GLYPHS`, `SIMPLIFIED_TO_JAPANESE`, `KNOWN_PLACES`, etc.) remain as references to the loaded data, so all existing imports and tests continue to work unchanged
- Thresholds in `menu.json` are loaded into the same module-level constant names, so existing code referencing `menu.CONFIDENCE_FLOOR` etc. continues to work
- Version field enables audit trail: when a detection manifest is saved, the correction dict versions can be recorded for reproducibility
- Adding new gazetteer entries or glyph pairs is now a data-only change (edit JSON, no Python code change)
- The `data/` directory is inside the package, so it ships with installs and is found via `importlib.resources` or `Path(__file__).parent / "data"`

### Tests

- **`tests/test_savor.py`** — add:
  - `test_dict_version_present()` — verify `savor.json` has a `version` field
  - `test_dict_loads_correctly()` — verify loaded data matches expected structure
  - `test_backward_compatible_constants()` — verify `savor.CONFUSABLE_GLYPHS` still works
- **`tests/test_wasabi.py`** — add same pattern
- **`tests/test_menu.py`** — add same pattern, plus:
  - `test_thresholds_loaded_from_json()` — verify `menu.CONFIDENCE_FLOOR` etc. match JSON values
- **`tests/test_dictionary_versioning.py`** (new) — integration tests:
  - `test_all_dicts_have_versions()` — all three JSON files have valid semver version fields
  - `test_dict_entry_counts()` — entry counts are non-zero and match expected
  - `test_capability_report_includes_dict_versions()` — `capability.correction_dict_versions()` returns correct versions

---

## Implementation Order

1. **Section 4** (versioned dictionaries) — foundational; other sections reference dict versions
2. **Section 2** (capability reporting) — depends on dict versions from section 4
3. **Section 3** (arbitration hardening) — independent of sections 2 and 4
4. **Section 1** (fixture regression suite) — depends on all prior work being stable; runs last as validation

## Testing Strategy

- All new tests run under the existing `pytest` infrastructure
- No existing tests should be weakened or deleted
- New tests that require OCR engines or inpainting providers skip gracefully when dependencies are unavailable (same pattern as `test_savor.py`'s `pytest.skip`)
- The fixture regression suite generates a JSON report artifact in `tests/eval_out/` for CI inspection
- CI coverage should not decrease

## Backward Compatibility

- All existing module-level constants remain accessible at their current import paths
- All existing function signatures remain unchanged (new parameters are keyword-only with defaults)
- All existing API endpoints remain unchanged (new `/api/capabilities` is additive)
- No existing test should fail as a result of these changes
