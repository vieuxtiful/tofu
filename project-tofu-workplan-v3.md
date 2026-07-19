# ToFU Workplan v3 — Layer Maturity → High-Accuracy, Seamless UI/UX

*Drafted 2026-07-18 from a full audit of `src/tofu/`, `server/`, and `frontend/src/`.*

---

## Part 1 — Current-state assessment (evidence per layer)

### 1. ToFU (pre-flight validation) — `src/tofu/layers/tofu.py`
**Works:** script-support via static map + real FontRegistry glyph coverage; expansion
feasibility (ToFU_005) with actual-translation measurement; frontend Preflight panel
renders expansion-fit bars.

**Gaps (evidence):**
- Glyph-segmentation score is a hard-coded constant (0.65 complex / 0.85 otherwise,
  `tofu.py:291-311`) — it is not derived from the asset, so ToFU_003 fires identically
  for every Japanese image regardless of content.
- Render-quality context is never supplied: `/api/validate` (`server/main.py:352`)
  passes only `{"font": ...}`; `font_px` and `effects` are always absent, so the
  deterministic score is effectively a constant too.
- Validation checks only the request-level target language. Per-region
  `target_language` overrides (`types.py:120`) — a shipped feature in the Region
  table — are never validated: a region overridden to Hindi passes a Spanish
  pre-flight.
- **Nothing guards render time.** `scribe._get_font` (`scribe.py:51`) silently falls
  back to `arial.ttf`/default bitmap when the selected face fails — the exact
  "tofu blocks" failure the layer is named for can still reach the output image
  with a passing pre-flight.

### 2. Cicerone (detection & recognition) — `src/tofu/layers/cicerone.py`
**Works:** the most mature layer. Multipass CRAFT, scene-constrained filtering,
vertical CJK column merge, garbage-rescue auto-probe, language-adaptive second pass,
coarse-to-fine zoom, second-look re-recognition, ja/zh/ko disambiguation, per-region
`detected_language` in the manifest and UI. Swappable EasyOCR/PaddleOCR backends.

**Gaps (evidence):**
- **Zero typography output.** `CharactText` (font_style/size/positioning/effects)
  is never populated anywhere. No bold, italic, weight, serif/sans, or size detection
  exists. The Region table already has a "det." font chip UI slot
  (`RegionTable.tsx:366-373`) that can never render because `characteristics` is
  always null. This is the single largest gap against goal 2.
- Region orientation is discarded: detection polygons are collapsed to axis-aligned
  bboxes; no skew/rotation angle is computed, so scribe can never re-render slanted
  signage correctly.
- easyocr is **not installed in the dev environment**, so every recognition-dependent
  behavior no-ops locally and none of the CJK machinery can be regression-tested
  where development happens.

### 3. Scene (context & style) — `src/tofu/layers/scene.py`
**Works:** pre-pass (Canny contours + MSER → panel/bordered_region/text_cluster/
surface) constrains detection; enrichment (Otsu + GrabCut text/bg color, texture
flat/textured, containing-region label) exists and never overwrites user values.

**Gaps (evidence):**
- **Enrichment never runs at capture time.** `/api/detect` and `/api/detect/stream`
  call only `scene.analyze_regions()` (pre-pass); `scene.analyze()` enrichment runs
  only inside `/api/render` (`pipeline.py:199`). At capture, instances carry no
  `style_profile`/`background_profile` — the hover-tooltip goal has **no data**.
- The capture canvas has no metadata tooltip at all; the only tooltip is the
  translate-preview one showing `target_text` (`BBoxCanvas.tsx:411-415`).
- `BgProfil.gradients`/`patterns` are declared but never populated; texture is a
  single std-dev threshold.

### 4. Cleanse (erasure & inpainting) — `src/tofu/layers/cleanse.py`
**Works:** OpenCV Telea inpainting with polygon+bbox mask; PIL median-fill fallback;
DNT respected.

**Gaps (evidence):**
- **The mask always includes the full padded bbox rectangle** — `cleanse.py:84-90`
  does `fillPoly(...)` then *also* `rectangle(...)`, so the polygon's precision is
  discarded and the inpainter must hallucinate the entire box interior. With
  `inpaint_radius=3` on large boxes this is exactly the smearing/"bbox artifact"
  failure described. Root cause of goal 4.
- One inpainting strategy regardless of background: `BgProfil` hints
  (flat vs textured, dominant color) are computed by Scene but never consulted.
- No stroke-level masking: the text pixels vs. background pixels segmentation that
  Scene's GrabCut already computes is not shared with Cleanse.

### 5. Scribe (text regeneration) — `src/tofu/layers/scribe.py`
**Works:** binary-search fit, h/v alignment, stroke, tracking/tsume,
italic shear, underline, sub/superscript, rotation param, DPI/EXIF/ICC preservation.

**Gaps (evidence):**
- **No line wrapping** — `_fit_font` measures the whole string as one line; long
  translations shrink to unreadable sizes instead of wrapping to the bbox aspect.
- `font_weight` is stored on StyleProfil but `_get_font` ignores it — bold/light
  face selection never happens even when the registry knows the family's weights.
- `RenderParams.rotation` exists but nothing ever sets it from detected geometry.
- No vertical writing mode for CJK columns (cicerone merges columns; scribe can
  only render them horizontally).
- `StyleProfil.shadow`/`effects` are dead fields; no glyph-coverage fallback
  (ties to ToFU gap).

### 6. Verify (QA) — `src/tofu/layers/verify.py`
**Works:** OCR round-trip legibility (0.7), ring-SSIM background preservation (0.3),
ink-presence fallback, degradation contract, per-region scores surfaced as chips in
the render result.

**Gaps (evidence):**
- **No residual-source-text check**: nothing verifies the original text was actually
  removed — a failed erase under a successful re-render passes.
- Untranslated/DNT regions score NEUTRAL 1.0 and *inflate* the gate; there is no
  coverage metric ("all Cicerone regions addressed") even though the goal names it.
- No Verify step in the frontend: `Stepper` is Upload→Capture→Translate→Render; the
  QA report renders as a static badge + chips with no inspection UI, no overlay,
  no toast-guided remediation, no re-render loop.

### 7. Memory — `src/tofu/layers/memory.py`
QA-gated record builder, no persistence, no matching. Appropriately a stub;
sequenced last per the stated plan.

### Cross-cutting
- No automated test suite; `scripts/eval_detect.py` covers detection only — no
  harness for cleanse/scribe/verify quality.
- SQLite project layer (`server/db.py`) is healthy: snapshots, events, language lock.

---

## Part 2 — Step-wise plan

Methods are restricted to accepted, citable CV/typography techniques (per the
technical-paper constraint). Citations inline.

### Phase 0 — Foundation: environment parity + measurement *(2–3 days)*
The prerequisite for "high accuracy thresholds" is the ability to measure them.
1. Install easyocr (+ torch CPU) into the dev venv; smoke-test `/api/detect` on
   `images/` assets; document the GPU flag path.
2. Build `scripts/eval_render.py` (sibling of `eval_detect.py`): given asset +
   manifest + translations, run cleanse→scribe→verify, write side-by-side overlay
   PNG + JSON (per-region OCR round-trip, ring-SSIM, residual-text). This becomes
   the before/after harness for Phases 3–5.
3. Assemble a small fixture set (5–8 images: flat-bg sign, textured wall, dense CJK
   street scene incl. `japan-street.jpeg`, latin storefront, stylized/italic text)
   with hand ground truth. Add pytest scaffolding for pure-logic units
   (column merge, expansion math, language ID, mask building).
   
**Exit criteria:** eval_detect + eval_render produce baseline JSON reports committed
for every fixture.

### Phase 1 — Cicerone typography + capture-time enrichment *(Weeks 1–2)*
Goal 2 (font/appearance detection → manifest) and the data half of goal 3.
1. **Shared stroke-mask utility** `utils/imaging.py: text_mask(img, bbox, polygon)`
   — extract Scene's Otsu+GrabCut segmentation (Otsu 1979; GrabCut, Rother et al.
   2004) into a reusable function returning the binary glyph mask. Cicerone
   (typography), Cleanse (Phase 3), and Verify (Phase 5) all consume it.
2. **New `typography.py` analysis pass** (called from `build_manifest` /
   `scene.analyze`), per region, all classical + citable:
   - *Weight*: stroke-width via distance transform on the glyph mask (Stroke Width
     Transform family — Epshtein, Ofek & Wexler, CVPR 2010); stroke-width/x-height
     ratio thresholds → light/regular/bold.
   - *Slant/italic*: shear-search maximizing vertical projection-profile variance
     (standard slant estimation — Vinciarelli & Luettin, PRL 2001); |angle| > ~8°
     → italic, and the angle itself feeds Scribe's shear.
   - *Size*: cap-height/x-height from mask row-profile (replaces the 0.75·bbox
     heuristic).
   - *Orientation*: `cv2.minAreaRect` on the detection polygon → baseline rotation
     stored on the instance (consumed by Scribe in Phase 4).
   - *Serif/family (best-effort)*: render the recognized text in the top-N registry
     faces at the estimated size/weight and score with SSIM/NCC template matching —
     the render-and-compare protocol from visual font recognition (cf. DeepFont,
     Wang et al., ACM MM 2015, as the deep upgrade path behind the same interface).
     Emit family + confidence; below threshold, emit only serif/sans class.
3. **Populate the manifest**: `CharactText.font_style/size`, `StyleProfil.font_weight/
   italic/font_size`, rotation. Manifest (de)serialization + interchange exports
   updated; PUT/PATCH endpoints already pass these through.
4. **Run `scene.analyze()` enrichment at detect time** — at the end of both
   `/api/detect` and `/api/detect/stream` (new SSE stage `"enrich"`), so
   style/background/typography data exists at capture, not only at render.
5. **Frontend**: the existing "det." chip (`RegionTable.tsx:366`) lights up; add
   weight/italic badges and a detected-size hint in the Render style panel
   ("detected: bold italic ~42px — apply").

**Exit criteria:** on the fixture set, weight detection ≥85% and italic ≥85%
accuracy vs. hand labels; manifest JSON after Capture contains populated
characteristics + profiles; CJK fixtures keep current language-ID accuracy
(no regression in eval_detect).

### Phase 2 — Scene metadata UX: capture tooltips + richer profiles *(Week 2–3)*
UI half of goal 3.
1. **Capture-tab hover tooltip** on BBoxCanvas bboxes: Style (color swatch + hex,
   weight, italic, est. size) and Background (semantic label, dominant color
   swatch, texture), plus detected language + confidence. Same data drives an
   expanded row detail in the Region table. (All fields exist after Phase 1.)
2. **Background profile depth**, still classical: gradient detection via
   least-squares linear fit of the border-ring luminance (report as
   `gradients: ["linear 12°"]`); texture classification upgraded from one std
   threshold to Laplacian variance + GLCM contrast/homogeneity (Haralick 1973) →
   `flat | smooth_gradient | textured | patterned`. These labels select the
   Cleanse strategy in Phase 3 — this is the Scene↔Cleanse tandem the goals
   describe.
3. **Text/non-text pre-pass hardening**: filter MSER components by stroke-width
   variance (the classical MSER+SWT cascade — Neumann & Matas 2012 / Epshtein
   2010) before clustering into `text_cluster` regions, cutting false surfaces
   that currently admit low-confidence junk detections through the 0.30 gate.

**Exit criteria:** hovering any bbox in Capture shows Style/Background tooltip with
live data; scene pre-pass false-surface count drops on fixtures without recall loss
(eval_detect region counts stable).

### Phase 3 — Cleanse overhaul: stroke-mask, background-adaptive inpainting *(Weeks 3–5)*
Goal 4; largest quality lever for the final image.
1. **Stroke-level masks**: build the inpaint mask from the glyph mask (Phase 1
   utility) dilated 2–3 px — the standard mask protocol in scene-text-removal
   literature (EnsNet, Zhang et al. AAAI 2019; EraseNet, Liu et al. TIP 2020) —
   with bbox-rectangle fill demoted to a fallback when segmentation fails.
   Delete the unconditional `cv2.rectangle` at `cleanse.py:88`.
2. **Background-adaptive strategy selection** from `BgProfil`:
   - `flat` → direct fill with border-ring median + 1–2 px feather (cheapest,
     artifact-free on signs/panels);
   - `smooth_gradient` → fill with the Phase-2 linear gradient model + feather;
   - `textured/patterned` → PatchMatch-style exemplar fill (Barnes et al.,
     SIGGRAPH 2009 — the Photoshop content-aware-fill algorithm; via
     `cv2.xphoto.inpaint` FSR or a vendored PatchMatch), falling back to
     Telea/Navier-Stokes (Telea 2004; Bertalmio et al. 2001) with radius scaled
     to stroke width.
3. **`InpaintBackend` adapter** (mirrors OCRBackend/SceneBackend): `classical`
   (above) default; optional `lama` (LaMa, Suvorov et al., WACV 2022) as the
   opt-in deep backend for hard textures — same pattern as SAM, checkpoint via
   PipelineCfg.
4. **Cleanse/Scribe refactor decision (recommended: shared core, two layers).**
   Introduce `layers/compositor.py` owning image loading, masks, and
   alpha-compositing; Cleanse and Scribe become thin stages over it. This gives
   the intended unification (one place where erase+render coherence lives —
   e.g., re-rendering can blend against the *pre-feather* background) without
   breaking the 7-layer pipeline contract, LayerMode toggles, or the paper's
   architecture narrative.

**Exit criteria:** on eval_render fixtures, ring-SSIM ≥0.90 mean (from baseline),
zero visible rectangle seams on flat/gradient fixtures at 1:1 zoom (manual
checklist), residual-text metric (built in Phase 0 harness) ≤0.1 similarity on all
erased regions.

### Phase 4 — Scribe fidelity + ToFU render-time guard *(Weeks 5–6)*
Goals 5 and the missing half of goal 1.
1. **Line wrapping**: greedy word-wrap (space-delimited scripts) / character-wrap
   (CJK) targeting the bbox aspect ratio inside `_fit_font`'s search; respects
   explicit `leading`.
2. **Weight/italic face selection**: resolve `font_weight` + `italic` through
   FontRegistry family weights (`families_with_weights` already exists) instead of
   ignoring them; detected slant angle (Phase 1) drives the shear when no italic
   face exists.
3. **Rotation & vertical text**: apply Phase-1 `minAreaRect` rotation via existing
   `RenderParams.rotation`; add vertical stacked rendering for regions cicerone
   merged as CJK columns (per-glyph vertical layout with tsume).
4. **Shadow/effects**: render `StyleProfil.shadow` (offset/blur/color) via a
   blurred offset alpha layer.
5. **ToFU render guard (closes goal 1)**: before drawing each region, check
   codepoint coverage of the resolved face via FontRegistry; on misses, swap to
   the top-ranked covering font, log a `tofu` warning into pipeline logs, and
   flag the region in the response (`glyph_fallback: true`) so the UI can toast
   it. Additionally run `ToFU.validate` per distinct effective target language
   (region overrides included) during `/api/render`, with real `font_px` and
   effects context finally supplied.

**Exit criteria:** OCR round-trip mean ≥0.85 on fixtures (from baseline); German
1.35× expansion fixture wraps to 2 lines instead of shrinking below legibility;
a deliberately-wrong-font render produces a fallback + warning, never tofu boxes.

### Phase 5 — Verify expansion + QA Inspector frontend *(Weeks 6–8)*
Goal 6.
1. **New metrics** in `verify.assess`:
   - *Residual source text*: OCR the erased region in the output, compare to the
     source string — high similarity = failed erase (the recognition-based
     protocol from the text-removal literature).
   - *Coverage*: `regions_total / translated / dnt / untranslated / rendered /
     fallback_font` in `QAReport.progress`; untranslated non-DNT regions become a
     scored deduction, not NEUTRAL, with a distinct recommendation.
   - *Style consistency*: color ΔE (CIE76) between detected source text color and
     rendered text color; size ratio check. Populates the "style consistency
     against the original" requirement.
2. **`/api/render/stream`** (SSE, mirrors detect/stream): per-layer progress
   events (tofu → scene → cleanse → scribe → verify) + final payload, enabling
   real progress toasts instead of one spinner.
3. **Frontend QA Inspector — new Stepper step 4 "Verify"** (piloting the
   configured-but-unbuilt final step):
   - Result canvas with per-region score overlay (green/amber/red), hover shows
     metric breakdown (round-trip text read back, SSIM, residual, style ΔE);
   - Source↔localized slider or synced-zoom compare;
   - `recommendations[]` rendered as the toast-like guidance stream during render
     (from SSE) and as a persistent checklist after;
   - Per-region "re-render" action: adjust style panel → re-render only that
     region (server: render accepts optional `region_ids` subset);
   - Double-confirmation banner: coverage summary ("9/9 regions addressed — 7
     rendered, 2 DNT") gating a final "Approve" that stores the QA sign-off
     event in project history.

**Exit criteria:** a failed erase or an unreadable render is visibly flagged on the
overlay with an actionable recommendation; QA gate reflects coverage (an untranslated
region can no longer inflate the score); render progress streams live.

### Phase 6 — Memory MVP *(Weeks 8–10, after baselines above hold)*
Goal 7, sequenced last as specified.
1. **Persistence**: SQLite tables (`tm_records`: source_text, normalized_text,
   source/target lang, style fingerprint, crop pHash, target_text, qa_score,
   asset/project refs; crop thumbnails on disk). Perceptual hash for visual
   identity (pHash/DCT — Zauner 2010).
2. **Matching**: exact (normalized text + lang pair) → fuzzy text (normalized
   edit-distance ≥0.85) → visual (pHash Hamming distance) with combined
   confidence; style fingerprint (weight/italic/color bucket) as a tiebreaker.
3. **Pipeline integration**: post-Cicerone TM lookup populates `target_text`
   suggestions with provenance (`from_memory: score`); Verify-approved runs write
   back (gate already implemented).
4. **Frontend**: suggestion chips in the Translate table ("TM 92% — apply"),
   project-level TM browser panel, cross-asset "seen before" indicator on capture.

**Exit criteria:** re-processing a near-duplicate asset pre-fills matching regions
with approved translations at ≥95% precision on a constructed duplicate-pair
fixture set.

---

## Part 3 — Timeline

Assumes one developer + AI pair, ~full-time. Buffer built into each phase; hard
integration checkpoints at the end of Phases 1, 3, and 5.

| Phase | Scope | Dates (2026) | Duration |
|---|---|---|---|
| 0 | Env parity, eval_render harness, fixtures, baselines | Mon Jul 20 – Wed Jul 22 | 3 d |
| 1 | Cicerone typography, capture-time enrichment, manifest/UI | Thu Jul 23 – Wed Aug 5 | 2 wk |
| 2 | Scene tooltips, gradient/texture profiles, MSER+SWT filter | Thu Aug 6 – Wed Aug 12 | 1 wk |
| **CP-1** | Integration check: capture UX demo on fixtures | Thu Aug 13 | — |
| 3 | Cleanse overhaul, adaptive inpainting, compositor refactor | Fri Aug 14 – Fri Aug 28 | 2 wk |
| **CP-2** | Erase-quality review vs. baselines | Mon Aug 31 | — |
| 4 | Scribe wrapping/rotation/vertical/weights, ToFU render guard | Tue Sep 1 – Fri Sep 11 | 1.5 wk |
| 5 | Verify metrics, render SSE, QA Inspector step | Mon Sep 14 – Fri Sep 25 | 2 wk |
| **CP-3** | End-to-end QA demo: upload → verified render | Mon Sep 28 | — |
| 6 | Memory MVP (store, match, suggest, UI) | Tue Sep 29 – Fri Oct 9 | 2 wk |
| — | Hardening, paper-alignment pass, docs | Mon Oct 12 – Fri Oct 16 | 1 wk |

**Dependency spine:** Phase 1's stroke-mask utility and typography fields feed
Phases 2 (tooltips), 3 (masks), 4 (style application), and 5 (style consistency) —
which is why Cicerone/Scene work is front-loaded even though Cleanse artifacts are
the most visible defect today.

## Part 3b — Phase 0 baseline findings (measured 2026-07-18)

Harness: `scripts/eval_detect.py` + `scripts/eval_render.py` (identity-mode
render: target = source, so scores isolate erase/re-render fidelity).
Fixtures: `tests/fixtures/` (synthetic, exact GT) + 2 real street scenes (no GT yet).

| Fixture | Detect F1 | QA overall | ring-SSIM | OCR round-trip | residual max |
|---|---|---|---|---|---|
| flat-sign | 1.00 | 1.00 | 0.999 | 1.00 | 0.00 |
| gradient-banner | 0.80 | 0.90 | 0.956 | 0.88 | 0.00 |
| textured-wall | 1.00 | 1.00 | 0.985 | 1.00 | 0.00 |
| stylized-italic | 0.44 | 0.76 | **0.655** | 0.80 | 0.00 |
| expansion-en | 1.00 | 1.00 | 0.998 | 1.00 | 0.00 |
| cjk-vertical | **0.00** | n/a (0 regions) | — | — | — |
| gemini-street (no GT) | — | **0.43** | 0.930 | **0.21** | **1.00** |
| japan-street (no GT) | — | **0.28** | 0.926 | 0.00 | 0.00 |

Confirmed, quantified gaps (each maps to a planned phase):
1. **cjk-vertical detects 0 regions.** The en-charset pass reads the column as
   garbage (`'2'`, `'+ 527'`); the auto-probe fails because isolated
   single-kanji crops recognize at conf ~0.08 (< 0.2 evidence bar); even a
   ja-hinted reader misses the vertical column (CRAFT vertical weakness) while
   reading horizontal ようこそ at conf 1.0. → Phase 1 (probe evidence tuning,
   PaddleOCR vertical evaluation).
   **RESOLVED 2026-07-18 (Phase 1b): scene-surface probe.** Uncovered scene
   surfaces are probed per-panel with candidate CJK readers (per-panel reads
   score conf 1.0 where full-frame vertical detection reads 0.08); the winner
   supplies both the adaptive-repass langset and authoritative detections that
   merge_vertical_columns reassembles. cjk-vertical F1 0.00 → **1.00**, edit
   distance 0.0, src ja; all other fixtures byte-identical (health gate skips
   healthy scenes). japan-street remains unrescued — its one scene surface is
   frame-sized (area-capped); needs Phase 2 scene recall and/or PaddleOCR
   (cp313 wheel confirmed available: paddlepaddle 3.3.1).
2. **stylized-italic ring-SSIM 0.655** on a *clean synthetic* — the
   full-bbox-rectangle inpaint mask disturbing surroundings, exactly the bbox
   artifact. → Phase 3.
3. **gemini-street residual max 1.00** — at least one region where the source
   text fully survived the erase. → Phase 3 (masks) + Phase 5 (residual metric
   into verify).
4. **gemini-street OCR round-trip 0.21 / japan-street 0.00** — scribe's
   fallback fonts (arial/segoe) cannot draw hangul/kana; rendered CJK is
   unreadable. → Phase 4 ToFU render guard + registry-driven font resolution.
5. Real-scene GT annotation is outstanding (street scenes report no
   detect-F1); annotate during Phase 1 CJK work.

## Part 3c — CP-1 results (measured 2026-07-19)

Street-scene partial GT annotated by crop verification (major legible signage
only; recall-focused): `images/gemini-street.gt.json` (18 regions),
`japan-street.gt.json` (8 regions). PaddleOCR evaluated standalone under an
**isolated** `.venv-paddle` (paddlepaddle 3.3.1 + paddleocr 3.7.0/PP-OCRv5,
`scripts/eval_paddle.py`) — installing paddle into the app venv is forbidden:
it force-replaces numpy/opencv (one attempt corrupted numpy mid-flight and was
rolled back; app venv now pins numpy 2.3.5, all tests green).
Windows-CPU quirk: paddle 3.3 oneDNN/PIR crash — fixed with
`FLAGS_use_mkldnn=0` + `enable_mkldnn=False`.

| Scene (recall / mean norm-ED) | EasyOCR + surface probe | PaddleOCR PP-OCRv5 |
|---|---|---|
| cjk-vertical | 1.00 / 0.00 (~60 s with probes) | 1.00 / 0.00 in **3.8 s** |
| gemini-street | 0.111 / 0.83 | **0.444 / 0.19** (korean; 60 dets, 22 s) |
| japan-street | 0.00 / — | 0.125 (ja; low-res signage) |

Conclusions:
1. **PaddleOCR is the engine for dense/vertical CJK scenes** — 4× recall and
   4× transcription accuracy on gemini-street; native vertical handling makes
   the surface-probe rescue unnecessary on scenes it covers. EasyOCR stays the
   default for latin/simple scenes (zero extra deps).
2. Language arbitration remains cicerone's job: the japan model on the same
   korean image scores ED 0.917 vs korean's 0.19 — engine choice does not
   replace per-language recognition + script arbitration.
3. **Integration decision needed (Phase 2/3): paddleocr cannot live in the app
   venv** (numpy/opencv pin conflict). Recommended: subprocess bridge behind
   the existing OCRBackend adapter (`.venv-paddle` worker, JSON over stdio),
   replacing the current in-process PaddleOCRBackend that targets the dead
   2.x API. japan-street needs a 2× upscale pass or better scene recall
   regardless of engine.

## Part 3d — Phase 2 scene pre-pass hardening (measured 2026-07-19)

Two fixes to `ClassicalCVBackend`, both regression-tested (72 unit tests +
full eval_detect sweep, byte-identical F1 on every synthetic fixture):

1. **Frame-sized region filter** (`max_region_frac=0.85`): both street photos
   previously returned exactly ONE scene region — the frame itself — because
   contour analysis finds the outer image boundary as a "bordered_region", and
   the largest-first containment dedup then swallowed every real surface
   inside it. Regions covering more of the frame than this fraction are
   dropped before dedup.
2. **MSER cluster single-linkage snowball (real bug, not a tuning issue)**:
   the original merge test grew its own catchment tolerance with the
   accumulating cluster's size (`(w+mw)//2 + merge_dist`), so a cluster's
   reach expanded with every absorption — on dense signage this collapsed
   hundreds of stroke-like components into one near-frame blob. Fixed with
   (a) an SWT-style stroke-width coefficient-of-variation gate on MSER
   components before clustering (Epshtein 2010 / Neumann & Matas 2012 —
   text strokes are near-uniform width, blobs are not) and (b) replacing the
   growing-tolerance merge with union-find over a FIXED pairwise gap that
   additionally **refuses any merge whose resulting envelope would exceed
   12% of the frame area** — single-linkage clustering chains transitively
   regardless of how the pairwise distance is defined, so only a hard
   envelope-growth cap stops the snowball. Verified directly against
   `images/gemini-street.png` in `tests/test_scene_regions.py`.

| Scene | scene surfaces before → after | detect regions | detect recall |
|---|---|---|---|
| gemini-street | 1 → 16 | 14 → 14 | 0.111 → 0.111 (unchanged) |
| japan-street | 1 → 12 | 1 → 2 | 0.00 → 0.00 (unchanged) |

Scene recall improved substantially with zero synthetic-fixture regression,
but street-scene OCR recall did not move — confirms the CP-1 finding
independently: japan-street's signage is too small/distant for EasyOCR's
recognizer regardless of how many candidate surfaces the scene pre-pass now
finds. The remaining lever is the PaddleOCR bridge, landed next.

## Part 3e — PaddleOCR subprocess bridge (landed 2026-07-19)

`PaddleOCRBackend` in cicerone.py rewritten from an in-process reader
(targeting paddleocr's dead 2.x `.ocr()` API, would have crashed on any
install) to a subprocess bridge: `scripts/paddle_worker.py` runs under the
**isolated** `.venv-paddle` interpreter — never imported into the app
process, per the CP-1 numpy/opencv-conflict finding — communicating over a
JSON-over-stdin / JSON-file-out protocol (never stdout, which PaddleOCR's
own logging pollutes). `PaddleOCRBackend.is_available()` replaces every
`try: import paddleocr / except ImportError` call site (cicerone.py x3,
server/main.py x2); `TOFU_PADDLE_VENV` env var overrides the venv location
for non-default deployments. In-memory assets (PIL/ndarray) are written to
a temp PNG before crossing the venv boundary — no numpy objects are ever
pickled across the process split.

Verified: 12 unit tests against the bridge's path resolution and
graceful-degradation (venv missing → empty results, never an exception) run
unconditionally; 4 live round-trip tests (self-skip without `.venv-paddle`)
confirm `detect()`, `detect_in_regions()`, and — critically — the *full*
`cicerone.build_manifest()` pipeline (column merge, script ID, hallucination
pruning) all work correctly against PaddleOCR's `RawDetection` output, not
just EasyOCR's. Live end-to-end through the real FastAPI server via
`/api/detect/stream?engine=paddleocr`: cjk-vertical resolves both regions
correctly (居酒屋, ようこそ) with `src_lang: ja` and full Phase-1 typography
enrichment attached.

**Bug found and fixed during verification**: `_engine_name()` in
server/main.py only recognized `EasyOCRBackend`, so a successful PaddleOCR
detection was reported to the frontend as `engine: "null"` — the exact
signal the UI uses to show "OCR engine unavailable" and block capture.
Fixed to recognize `PaddleOCRBackend` too; `/api/detect` (the non-streaming
endpoint) was also hardcoded to EasyOCR regardless of `OCR_ENGINE`, now
matches the SSE endpoint's engine-selection logic for consistency.

Not yet done: perspective rectification for rotated crops in
`detect_in_regions` (v1 relies on PaddleOCR's own angle classifier, which
CP-1 showed already outperforms EasyOCR without it); a v2 worker can add
`cv2.getPerspectiveTransform` if crop-level accuracy proves to need it.

## Part 4 — Risks & mitigations
- **easyocr/torch on the dev host** (currently absent): Phase 0 gate; if
  installation is blocked, pin PaddleOCR as the dev default via `OCR_ENGINE` and
  keep EasyOCR for CI/server.
- **Font family recognition accuracy** (hardest CV task in scope): scoped as
  best-effort with confidence gating; weight/italic/size (the manifest fields the
  UI needs) do not depend on it.
- **PatchMatch availability** (`cv2.xphoto` is in opencv-contrib): fall back to
  Telea with scaled radius; LaMa stays opt-in so no deep dependency is required.
- **CPU render latency** as passes accumulate: SSE progress (Phase 5) addresses
  perception; per-region re-render limits recompute; reader singletons already
  amortize OCR init.
- **Scope creep in the QA Inspector**: pilot = overlay + compare + recommendations
  + approve; defer annotation/commenting tools.
