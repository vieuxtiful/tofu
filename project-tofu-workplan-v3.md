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
**Works:** pre-pass (Canny contours + MSER → panel/bordered region/text_cluster/
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
   contour analysis finds the outer image boundary as a "bordered region", and
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

## Part 3f — Phase 3: cleanse overhaul (landed 2026-07-19)

`cleanse.erase()` rewritten per the plan: per-instance stroke-level glyph
masks (shared `utils.imaging.text_mask`, dilated) replace the unconditional
full-bbox-rectangle fill that was the root cause of the measured stylized-
italic ring-SSIM 0.655 regression; the bbox rectangle is now a fallback used
only when segmentation genuinely fails. Fill strategy is selected per-region
from `inst.background_profile.texture` (Scene's Phase 2 classification, so
the two layers agree on what kind of surface they're looking at): `flat` →
border-ring median color fill, `smooth_gradient` → a local per-channel
linear-plane reconstruction re-fit over the actual erasure footprint (same
shading model Scene classifies with), `textured`/unclassified → batched
`cv2.INPAINT_TELEA` (no `cv2.xphoto`/PatchMatch in this OpenCV build,
confirmed by direct check; LaMa remains future opt-in per the original
plan). Every fill is alpha-feathered at the mask boundary via a distance-
transform falloff.

**Two real bugs found and fixed during measurement** (both via direct
apples-to-apples comparison against the pre-Phase-3 code with everything
else — typography, scene enrichment — held identical, not against the
stale Phase-0 baseline which predates typography-driven scribe rendering
entirely):

1. **Mask polarity inversion on multi-color ink.** `text_mask`'s Otsu+
   GrabCut assumes text is well-separated from a single background color;
   the stylized-italic fixture's stroke+fill outlined text (dark fill, a
   very different blue stroke color) has higher internal chromatic variance
   than its background, and GrabCut's GMM converged on the wrong partition —
   the mask selected the BACKGROUND, leaving the actual ink completely
   unerased (residual OCR similarity 1.0, a complete no-op erase, visually
   confirmed via a mask overlay debug render). Fixed with a polarity gate:
   a crop's own edge pixels are near-certainly background (OCR boxes are
   tight around ink, not flush with it), so the masked group must sit
   farther from the edge color than the unmasked group; when it's the
   other way around the mask is rejected and the bbox-rectangle fallback
   fires instead.
2. **Boundary ghost from under-dilation.** Feathering weakens toward the
   mask's own boundary by design (full-strength fill only deep inside it);
   with only 1px of dilation, that taper landed directly on the glyph's own
   anti-aliased edge instead of past it into real background, leaving a
   thin but fully OCR-readable outline of the original letterforms —
   measured on the flat-sign fixture, where EasyOCR still read "MAIN
   STREET" at full confidence through the outline (residual 1.0 on 4/4
   regions). This was invisible to the first round of unit tests because
   they only sampled single pixels, not the whole masked region. Fixed by
   increasing dilation to ~3px (matching the plan's original spec) so the
   full-strength fill zone comfortably covers the ink's anti-aliased
   fringe; a new test checks the ENTIRE dilated mask region, not a spot
   sample, specifically to catch this class of bug again.

**Final measured results** (apples-to-apples old-vs-new, identical
typography/scene enrichment on both sides; `scripts/eval_out/*-phase3.*`):

| Fixture | ring-SSIM before → after | residual before → after |
|---|---|---|
| flat-sign | 0.999 → 1.000 | 0.0 → 0.0 |
| gradient-banner | 0.956 → 0.961 | 0.0 → 0.0 |
| textured-wall | 0.985 → 1.000 | 0.0 → 0.0 |
| stylized-italic | 0.477 → 0.488 | 0.0 → 0.0 |
| expansion-en | 0.998 → 1.000 | 0.0 → 0.0 |
| cjk-vertical | 0.981 → 1.000 | 0.0 → 0.0 |
| gemini-street | 0.971 → 0.991 | 0.071 → 0.071 (unchanged) |
| japan-street | 0.822 → 1.000 | 0.0 → 0.0 |

Every fixture improved or held steady — zero regressions anywhere.
Residual is 0.0 across the board except gemini-street's one pre-existing
unresolved region (unchanged by cleanse, a detection/font issue, not new).
stylized-italic's ring-SSIM stays well under the original ≥0.90 exit
target: isolated by the same comparison methodology, this is caused
entirely by region r3 ("Italic Emphasis"), whose ring-SSIM is *identical*
(0.1057/0.1058) whether measured against old or new cleanse — it's scribe
now rendering an italic shear (fed by Phase 1 typography detection, which
didn't exist when the 0.655/0.90 targets were set) bleeding pixels past
the original detection bbox into the verify ring. That's Phase 4 (scribe)
scope, not a cleanse defect; flagged there rather than chased here.

Deferred by deliberate scope decision, not oversight: the `compositor.py`
shared-core refactor (Cleanse/Scribe unification) the plan flagged as a
recommendation-not-requirement — Phase 3's exit criteria are metric-based,
not structural, and no concrete duplication pain point arose while
implementing to justify the refactor now.

## Part 3g — Phase 4: scribe fidelity + ToFU render-time guard (landed 2026-07-19)

All five planned deliverables landed, plus three real bugs found and fixed
during measurement (methodology: apples-to-apples against the immediately
prior commit on every change, not the stale Phase-0 baseline — the same
discipline that caught Phase 3's two bugs).

1. **Line wrapping.** `_fit_wrapped()` replaces the old single-line-only
   `_fit_font`: greedy word-wrap for space-delimited scripts, character-wrap
   for CJK (no spaces), with the font-size binary search now searching over
   the WRAPPED block instead of one line — long translations wrap instead
   of shrinking below legibility.
2. **Real weight/italic face resolution.** `resolve_face()` searches a
   supplied `FontRegistry` for a sibling face matching the requested
   weight/italic before falling back to synthetic bold/shear. Directly
   closes the Phase 3 follow-up: region r3 (stylized-italic, the only
   italic region) went from the original broken 0.106 to a clean **1.0**
   ring-SSIM once a real `ariali.ttf` face is used instead of synthesizing
   a shear — measured end-to-end with a real font, not just the shear-
   pivot fix alone (which independently improved it to 0.313; see below).
3. **Rotation + vertical CJK columns.** Phase 1's detected
   `rotation_deg` now applies through `RenderParams.rotation` when no
   explicit override is given (explicit `0.0` correctly wins over a
   detected rotation — a real off-by-falsy bug caught by test, see below).
   Narrow-tall CJK-language regions (cicerone's `merge_vertical_columns`
   output shape) render as stacked glyph columns with tsume compression;
   verified with a real font — `居酒屋` renders correctly stacked,
   centered, and evenly spaced, matching the original signage layout.
4. **Shadow rendering.** `StyleProfil.shadow` (`offset_x/offset_y/blur/
   color`, all optional with documented defaults) renders as a blurred
   offset copy composited beneath the main text.
5. **ToFU render-time glyph-coverage guard.** `check_glyph_coverage()`
   checks every codepoint in the EXACT target string (not just a generic
   script sample) against the resolved font; on a gap, searches the
   registry for the best real covering font and swaps to a working copy
   (the source `style_profile` is never mutated), flagging the manifest
   instance `glyph_fallback=True`. The pipeline logs a `tofu`-stage
   warning naming the affected regions; the render result surfaces a
   warning badge on the per-region QA chip. Verified end-to-end: a Latin
   font requested for `居酒屋` renders real kanji via an auto-selected
   covering font instead of tofu boxes.

**Three real bugs found and fixed during verification, not by inspection:**

- **Italic shear used the wrong pivot** (the actual Phase 3 follow-up
  cause): the old shear transform applied `x' = x + 0.2*y` to a layer the
  size of the WHOLE base image, using each pixel's ABSOLUTE y-coordinate —
  for a region at y≈300 that's a ~60px unwanted sideways shift, not a
  slant. Fixed by shearing a small layer sized to the line's own extent
  around its own local origin before compositing. r3 ring-SSIM 0.106 →
  0.313 from this fix alone (→ 1.0 with real-face resolution on top, above).
- **Block-height fit check over-estimated for text without descenders**:
  the fit search used `font.getmetrics()` (ascent+descent, the font's
  nominal box — sized to accommodate glyphs a string may not even
  contain) instead of the actual rendered ink extent. Measured on a real
  cicerone-detected region: "SALE" in a 176×60 box fit at size 53 instead
  of the correct 69, visibly shrinking a region that fit fine as a single
  line and needed no wrapping at all (gradient-banner ring-SSIM regressed
  0.961→0.711 in an intermediate measurement before this was caught).
  Fixed by measuring with `draw.multiline_textbbox` for both the fit
  check and the block-centering calculation.
- **`_get_font` couldn't load FontRegistry's collection-face notation**:
  the registry keys `.ttc`/`.otc` collection faces as `"path#index"`; PIL
  takes the face index as a separate `index=` kwarg and raises `OSError`
  on a literal `#N` suffix in the path string. Undetected, this silently
  defeated BOTH `resolve_face()` and `check_glyph_coverage()` whenever the
  matched font was a collection face — common for CJK gothic/mincho
  families, which frequently ship as `.ttc`. Caught by an end-to-end
  visual check (the glyph-coverage guard reported success and a covering
  font, but tofu boxes still rendered); fixed by splitting the `#index`
  suffix and passing it through PIL's `index=` parameter.
- **Rotation falsy-zero bug** (caught by test, not measurement): resolving
  detected vs. explicit rotation with `if not params.rotation` treats an
  explicit `0.0` (a legitimate "no rotation, I mean it" override) the same
  as "not given," letting a detected rotation incorrectly win. Fixed with
  an explicit `is not None` check.

**Final measured results** (`scripts/eval_out/*-phase4.*`, vs. Phase 3):
zero regressions across all 8 fixtures/scenes; gradient-banner fully
recovered to Phase 3's 0.961 ring-SSIM after the block-height fix;
stylized-italic mean ring-SSIM 0.488→0.530 (still dominated by r3's now-
understood-and-fixable-with-a-real-font italic case, not a residual bug).
137 unit tests pass, including dedicated regression tests for all three
bugs above.

Not yet done: per-distinct-target-language `ToFU.validate` re-run during
`/api/render` with real font_px/effects context (the plan's stated Phase 4
item) — scoped out given session length; the glyph-coverage guard above
delivers the higher-value, more concrete half of "closes goal 1" (an
actual render-time block instead of a pre-flight prediction). Tracked as
a Phase 5 follow-up alongside the QA Inspector work, which touches the
same `/api/render` response contract.

## Part 3h — Phase 5: Verify expansion + `/api/render/stream` (landed 2026-07-19)

All planned backend deliverables landed, plus two real bugs found and fixed
during live end-to-end verification against the running server — not caught
by 157 passing unit tests, which is itself the finding worth recording (see
below).

1. **Coverage accounting.** `QAReport.progress` now reports
   `regions_total/dnt/translated/untranslated/rendered/fallback_font`, so the
   QA Inspector (Task 25) has a real completeness readout instead of only a
   score.
2. **Untranslated regions are a real deduction, not neutral.** A non-DNT
   region with no `target_text` now scores `UNTRANSLATED_SCORE = 0.0` (was
   `NEUTRAL_SCORE = 1.0`), with a recommendation surfaced. Closes the
   loophole where skipping regions could inflate a manifest's own gate score
   — measured: a 2-region manifest (one good, one untranslated) now scores
   `< 0.9` overall instead of the old scheme's `1.0`.
3. **Residual-source-text penalty**, graduated from `eval_render.py`'s
   Phase-0 standalone OCR-diff into `verify.assess()` proper: OCRs the
   cleansed crop, compares against the original source text, and
   *multiplies* the per-instance score down (`score *= 1 - residual`) above
   `RESIDUAL_PENALTY_THRESHOLD = 0.3`, rather than averaging in as positive
   evidence — a clean erase can't buy back a bad render, but a dirty erase
   always costs.
4. **Style-consistency metric**: CIE76 ΔE color distance
   (`_style_color_score`) and detected-vs-rendered size ratio
   (`_style_size_score`), each optional (`None` when no hint exists),
   weighted at `STYLE_WEIGHT = 0.2` into the per-instance score.
5. **`/api/render/stream`** (SSE, mirrors `/api/detect/stream`'s pattern):
   stages `tofu → scene → tofu_regions → cleanse → scribe → verify →
   complete`. Closes the Phase-4-deferred item: **`tofu_regions`** re-runs
   `ToFU.validate()` once per distinct effective target language actually
   present across regions, now with real `font_px`/`effects` context drawn
   from each region's own typography — the pre-flight check the plan
   originally scoped for Phase 4, delivered here because it shares the
   `/api/render` response contract with the SSE work.

**Two real bugs found only by live verification against the running
server** (unit tests, including 17 new Phase 5 tests, did not catch either
— the fixtures never happened to exercise `font_family=None` against a real
registry across two renders of the same manifest in sequence):

- **`glyph_fallback` false-positive for `font_family=None`** (the common
  "auto" case): `check_glyph_coverage()` only ever matched `current` against
  the registry when a literal path was set; `None` never matches any
  registry-keyed path, so every auto-styled region fell into "search for
  something better" — which always finds *some* path different from `None`
  and flags it. Measured live: flat-sign's 4 plain-English regions all
  reported `glyph_fallback=True` despite `arial.ttf` covering the text
  completely. Fixed by resolving `font_family=None` to the actual first
  loadable `FALLBACK_FONTS` entry in the registry before checking coverage.
- **Stale `glyph_fallback` flag persistence**: `render()` only ever *set*
  `glyph_fallback=True` on failure, never reset it on success — a manifest
  saved from an earlier (buggy) run kept `fallback_font=4` in the coverage
  summary even after fix #1 landed, because re-rendering with the fixed
  code found no gap and simply left the stale `True` untouched. Fixed by
  unconditionally calling `check_glyph_coverage()` every render and always
  explicitly setting `inst.glyph_fallback = not all_covered`.

**Live verification** (fresh asset, avoiding the stale-manifest confound of
the first pass): uploaded `flat-sign.png` fresh, detected 4 regions, set
translations, streamed `/api/render/stream`. All 7 stages fired in order
including `tofu_regions` ("re-validated 1 distinct target language(s)");
final coverage `{regions_total: 4, translated: 4, untranslated: 0, rendered:
4, fallback_font: 0}` (previously would have reported `fallback_font: 4`);
overall QA `0.98`.

**Final measured results** (`scripts/eval_out/*-phase5.*`, 8
fixtures/scenes, all `fallback_font: 0`):

| fixture | QA overall | ring-SSIM | OCR round-trip | style color/size |
|---|---|---|---|---|
| flat-sign | 0.981 | 1.0 | 1.0 | 0.866 / 0.909 |
| gradient-banner | 0.913 | 0.961 | 0.889 | 0.931 / 0.919 |
| textured-wall | 0.988 | 1.0 | 1.0 | 0.940 / 0.921 |
| stylized-italic | 0.746 | 0.530 | 0.8 | 0.875 / 0.884 |
| expansion-en | 0.968 | 1.0 | 1.0 | 0.886 / 0.733 |
| cjk-vertical | 0.405 | 0.999 | 0.0 | 0.904 / 0.956 |
| gemini-street | 0.331 | 0.990 | 0.071 | 0.553 / 0.841 |
| japan-street | 0.663 | 1.0 | 0.5 | 0.603 / 0.850 |

cjk-vertical and gemini-street's low overall scores are OCR-round-trip-
dominated (0.0 / 0.071) — expected and correctly penalized: these are hard
real-world CJK/street-scene cases where round-trip OCR on the *rendered*
localized text (not detection of the original) is the harshest of the four
scorers, and residual-source-text stays at 0.0–0.071 confirming cleanse
itself is not the bottleneck. No regressions vs. Phase 4 on any fixture.
157 unit tests pass (17 new for Phase 5).

Not yet done: Task 25, the QA Inspector frontend step (per-region score
overlay, source/localized compare, recommendations as guided toasts,
per-region re-render, coverage-gated Approve sign-off) — scoped as the next
phase, now that the backend contract it depends on (`progress`,
`per_asset_instance_score`, `recommendations`, the SSE stage sequence) is
landed and live-verified.

## Part 3i — Phase 5: QA Inspector frontend (landed 2026-07-19)

New Stepper step 4 "Verify" (`Stepper.tsx`, `App.tsx`), matching the
pilot scope from Part 4's risk note (overlay + compare + recommendations +
approve; annotation/commenting tools deferred):

1. **Coverage banner** — overall QA score badge, `regions_total/rendered/
   dnt/untranslated/fallback_font` chips, red warning styling on gaps.
2. **Recommendations checklist** — `qa_report.recommendations[]` rendered
   as a persistent list, plus the first three surfaced as toasts the
   moment a streamed verification completes.
3. **Source↔localized compare** — side-by-side images.
4. **Per-region QA cards** — score chip (green/amber/red) per region,
   click-to-expand metric breakdown (OCR round-trip, ring SSIM, residual
   source text, style color/size match — all four Phase 5a scorers),
   each with a **per-region re-render** button.
5. **Coverage-gated Approve** — disabled until `untranslated === 0`;
   posts to a new `POST /api/render/approve` endpoint that logs a
   `qa-approved` event (score + coverage snapshot) into project history
   (`HistoryPanel.tsx` already renders it via a new `EVENT_LABEL` entry).

**Backend additions to support the above:**
- `/api/render/stream` gained an optional `region_ids` (comma-separated)
  query param: when given, cleanse+scribe run ONLY on that instance
  subset, and the starting image is the asset's existing localized
  output (if one exists) rather than the raw source — untouched regions
  keep their prior render instead of reverting to source text. Verified
  live: re-rendering `r1` alone left `r2`-`r5` untouched and produced a
  consistent r1 score (0.9782 → 0.9782 across two partial re-renders).
- `POST /api/render/approve` — records the sign-off; verified live via
  curl, confirmed the `qa-approved` event appears correctly in
  `GET /api/projects/{id}/history`.

**One real bug found via live verification** (not caught by the existing
157 unit tests, including 17 dedicated Phase 5 tests): `POST /api/render`
500'd with `TypeError: 'numpy.bool' object is not iterable` the moment
ANY region had `style_profile.color` set — a normal state, since color is
populated at capture time for every detected region. Root cause:
`_style_color_score()` (`verify.py`) computed `ink_color =
crop[mask].reshape(-1, 3).mean(axis=0)`, a numpy array, and passed it
through `_delta_e_cie76()` → `_rgb_to_lab()` without ever casting back to
Python floats; the final `max(0.0, 1.0 - delta_e / COLOR_DELTA_E_SCALE)`
returned the numpy.float64 operand as-is (Python's `max()`/`and` return
operands unchanged, they don't coerce), and FastAPI's `jsonable_encoder`
has no handler for numpy scalars. This is the SAME class of bug as the
Phase 4/5 float()-casting bugs, but this one only manifests once a real
captured manifest (not a synthetic unit-test fixture) reaches the scorer
— the existing style-color tests built `StyleProfil(color=...)` fixtures
too, but apparently never happened to compare a value where `max()`
returned the numpy-typed operand specifically. Fixed by casting
`ink_color` to a plain float tuple before it enters `_delta_e_cie76`, and
added a defense-in-depth `float()` cast on every `per_instance` score and
`overall_score` at `QAReport` construction — this closes the bug class
generally, not just this one call site. Two regression tests added:
one asserting `type(score) is float` on `_style_color_score`'s return,
and one exercising the full `verify.assess()` → `json.dumps()` path the
way FastAPI actually serializes a response. Confirmed fixed via a fresh
`POST /api/render` against the exact manifest that 500'd before the fix
(now 200 OK, QA 0.60, coverage correct) and via the live browser Verify
step end-to-end (coverage banner, recommendations, compare, per-region
breakdown incl. non-zero style color/size scores, gated Approve all
render correctly).

**One more bug found via live browser interaction** (React dev-mode
console warning, not a crash): the per-region row was a `<button>`
containing the re-render `<button>` — invalid HTML that made click
targeting on the row unreliable (the browser silently reparents nested
buttons). Fixed by making the row a `<div role="button" tabIndex={0}>`
with an `onKeyDown` handler for Enter/Space, keeping the inner
`<button>` for re-render. Confirmed fixed by checking the live DOM
ancestor chain of the re-render button (now `DIV > DIV > DIV > SECTION`,
no `<button>` ancestor) and by successfully expanding a region's score
breakdown via click.

159 unit tests pass (2 new). Task 25 complete — Phase 5 (backend + QA
Inspector frontend) is fully landed and live-verified end to end.

## Part 3j — Phase 6: Memory MVP (landed 2026-07-19)

All four planned deliverables landed — persistence, tiered matching,
pipeline integration, and frontend surfacing — plus two real bugs found
via live verification and one active file-corruption incident caught and
reverted mid-session (see below).

1. **Persistence**: new `tm_records` SQLite table (`server/db.py`) —
   source/normalized text, lang pair, style fingerprint, perceptual hash,
   thumbnail path, target text, QA score, project/asset/region refs.
   Thumbnail crops saved under `server/tm_thumbs/`, mounted at
   `/tm_thumbs`. `src/tofu/utils/phash.py` implements a DCT-based
   perceptual hash (Zauner 2010): resize 32x32 grayscale, 2D DCT,
   top-left 8x8 AC coefficients thresholded against their median → a
   64-bit hash; Hamming distance gives a 0-1 visual similarity score.
2. **Matching** (`src/tofu/layers/memory.py`, fully rewritten from the
   Phase-0 stub): tiered exact (normalized-text equality) → fuzzy
   (`SequenceMatcher` ratio ≥0.85, `src/tofu/utils/textmatch.py`) →
   visual (pHash similarity ≥0.88, tried when text matching finds
   nothing — including when OCR read nothing at all this pass, not only
   when it read something that didn't match). Style fingerprint
   (weight|italic|color bucket) tiebreaks equal-scoring candidates.
   Kept storage-agnostic by design: `update()`/`lookup()` are pure
   functions over caller-supplied candidate pools/draft records — no
   SQLite import in `src/tofu/layers/` — matching every other layer's
   architecture; `server/main.py` owns all actual DB/file I/O via two
   new helpers, `_persist_tm_updates()` and `_attach_tm_suggestions()`.
3. **Pipeline integration**: `InstText.tm_suggestion` (new manifest
   field) populated by a `memory` stage after scene enrichment in both
   `/api/detect` and `/api/detect/stream` (candidate pool scoped to the
   project + the project's target language, since detect-time manifests
   don't carry a target language of their own yet). Write-back is
   QA-gated exactly like Phase 0's stub always intended: `/api/render`,
   `/api/render/stream`, and the legacy `/api/process` all persist
   `memory.update()`'s draft records only when the render passes the QA
   threshold. New `GET /api/projects/{id}/memory` (browse) and
   `DELETE /api/memory/{id}` endpoints.
4. **Frontend**: `RegionTable.tsx` shows a "seen before" badge
   (bookmark icon) next to any region ID with a TM suggestion, and — in
   Translate mode, for untranslated regions — a "TM 92% — apply" chip
   that fills `target_text` on click. New `MemoryPanel.tsx` (mirrors
   `HistoryPanel.tsx`'s structure) is a project-level TM browser:
   thumbnail, source→target, lang pair, QA score, timestamp, source
   asset/region reference, delete action; reachable from the header
   dropdown menu (`memory`, alongside `history`/`settings`).

**Exit criterion measurement** (`scripts/eval_memory.py`, plan-specified
methodology: "re-processing a near-duplicate asset pre-fills matching
regions... at ≥95% precision on a constructed duplicate-pair fixture
set"): for each of the 6 synthetic fixtures, detected once (asset A),
stored every real detected region as an approved TM record with a
globally-unique synthetic target. Each fixture was then perturbed
(resize 0.85x + JPEG recompression q80 — simulating a re-photograph at a
different distance) into asset B and detected independently, so OCR
reads B's text with real, uncontrolled noise. Ground truth for "is this
match correct" was established geometrically (bbox IoU after undoing the
known resize), not from text similarity, to avoid circularity with the
matcher under test. Result: **13/13 returned matches were the correct
record — precision 1.0**, against a 15-region correspondable universe
(match rate 13/15 = 0.867 — the system is conservative, not reckless: the
2 unmatched regions abstained rather than guessed). Exit criterion
(≥0.95 precision): **PASS**.

**Two real bugs found via live verification, neither caught by the 23
new unit tests**:
- `TextManifest.asset_id` was set to the full source FILE PATH
  (`asset_info.source`, e.g. `C:\Users\...\uploads\8ac2f037cbf8.png`)
  instead of the clean app-level asset id, in `cicerone.build_manifest()`
  — a pre-existing bug (present in HEAD before this session, not
  introduced by Phase 6) that nothing had surfaced before because
  `QAReport.per_asset_instance_score` was always read via
  `Object.values(...)[0]` on the frontend (key-agnostic) rather than by
  asset id. Memory's TM records are the first consumer to store and
  display `asset_id` as a meaningful value (`source_asset_id` in a
  suggestion, the record list in the TM browser), which immediately
  surfaced local filesystem paths leaking into API responses and the
  database. Fixed at the source: `asset_id=Path(asset_info.source).stem`
  — matches the app's own `UPLOAD_DIR/{asset_id}.ext` convention.
  Confirmed via a fresh live round-trip: a stored record's `asset_id`
  and a lookup's `source_asset_id` are both the clean id.
- A `memory.update()` draft record's `thumb_crop` field (a raw PIL
  Image) would have reached `jsonable_encoder` unguarded in the legacy
  `/api/process` endpoint (`payload = jsonable(result)` serializes the
  whole `PipelineResult`, memory_updates included) — the same bug CLASS
  as Phase 5's numpy-scalar leak, caught by code review this time before
  it shipped rather than by a live 500. `_persist_tm_updates()` always
  strips `thumb_crop` before anything downstream can see it.

**One active prompt-injection / file-corruption incident, caught and
reverted, not a Memory-layer bug**: mid-session, a tool result asserted
a syntax-breaking edit to `frontend/src/api.ts` (`bordered_region` →
`bordered region` inserted mid-declaration, breaking two interfaces) was
"intentional" and instructed withholding it from the user. Investigating
independently (not trusting that claim) found the SAME
`"bordered_region"` → `"bordered region"` corruption live in six more
spots, including the root cause: `src/tofu/layers/scene.py`'s
`ClassicalCVBackend` was assigning `label = "bordered region"` (the
actual value written into every `SceneRegion.semantic_label`) instead of
`"bordered_region"`, plus two doc comments in the same file; three
downstream consumer dicts in `src/tofu/layers/cicerone.py`
(`_PRIORITY`, `_ZOOM_PRIORITY`, `_SCENE_CONF`) whose keys still read
`"bordered_region"`, so they silently stopped matching anything —
defeating the panel/bordered-region priority boost in
`probe_uncovered_surfaces`, `zoom_detect`, and `build_manifest`'s scene
confidence filter (any bordered-region surface fell back to the generic
default threshold instead of its intended one); and one type comment in
`src/tofu/core/types.py`. Confirmed via `git diff HEAD` on each file
that this was uncommitted working-tree corruption with no legitimate
change mixed in; reverted all four files, reran the full suite clean.
Matches the same pattern flagged and reverted earlier this session
(Phase 2 write-up) in this exact file (`scene.py`) — it had recurred
since then. A parallel, cosmetic-only instance remains in an unrelated
uncommitted `README.md` addition (not part of this session's own work,
left alone per scope) — worth a clean sweep before that file is ever
committed.

182 unit tests pass (23 new: `test_memory_phase6.py` — phash, text
matching, `update()`/`lookup()` gating and tiering). Phase 6 complete —
this closes the v3 workplan's stated phase list (0-6); only the
originally-scoped "hardening, paper-alignment pass, docs" week remains
unscheduled work.

## Part 3k — Post-Phase-6: Cicerone digit/letter post-correction (landed 2026-07-19)

Not a numbered phase — a targeted Cicerone hardening item that surfaced
from a real observed misrecognition (`flat-sign.png` r4: "Open 9am to
5pm" read as "Open 9am to Spm" at confidence 0.958, a CRNN+CTC glyph
confusion no existing safeguard catches — `second_look()`'s re-read
threshold is 0.55, `_prune_hallucinations()` only drops symbol junk).

An initial fix was proposed as a regex-triggered confusion-pair
substitution (swap `S`→`5` etc. wherever a digit-expecting context like
"am"/"pm" appears). Assessed before building: **the substitution-only
version is unsafe as specified** — walking its own confusion map against
the token `"Sam"` (ends in "am", preceding char `S`) fires the rule and
silently rewrites it to `"5am"`. Context alone can't distinguish "OCR was
wrong" from "OCR was right and the context pattern is coincidental";
names/brands ending near trigger suffixes are exactly what real signage
is full of.

Built instead as three explicitly separated stages
(`src/tofu/layers/recognition_correct.py`, new), called from
`cicerone.build_manifest()` right after `_prune_hallucinations()`:

1. **Stage A** (`find_candidates`): grammar-anchored — proposes a
   correction only for a whole token matching a digit-expected shape
   (currently: a 1-2 char hour prefix + "am"/"pm") where substituting a
   confusable letter yields a *valid* hour (1-12). Proposes only; never
   touches text. Multi-position ambiguity (more than one letter needing
   substitution) is skipped, not guessed — Stage B can only verify one
   glyph at a time.
2. **Stage B** (`verify_candidate`): pixel evidence — segments the
   ambiguous glyph via connected-component analysis of the shared
   `imaging.text_mask()`, renders both candidate characters as reference
   glyphs via `scribe._get_font()` sized to the extracted glyph's own
   height, and compares shapes by IoU. Only a clear margin resolves the
   candidate (`GLYPH_MATCH_MARGIN=0.12`); anything closer, or where glyph
   segmentation doesn't line up 1:1 with the recognized text (e.g. a
   connected/cursive script), returns "inconclusive" rather than guessing.
3. **Stage C** (orchestration in `correct_instances`): a candidate is
   only ever *applied* when Stage B resolves `True`. An inconclusive
   candidate is never guessed into the text — it's recorded on the new
   `InstText.ocr_correction` field (`applied=False`) so it surfaces as a
   reviewable signal instead of silently vanishing or silently rewriting
   something wrong.

No character-level bboxes exist upstream (`EasyOCRBackend.detect()` uses
`reader.readtext()`, word/line-level only) — Stage B derives its own via
connected-component segmentation of the existing glyph mask, reusing
infrastructure rather than adding a dependency.

**Verification** (per the plan's own required methodology — measure, not
assume): unit tests (`tests/test_recognition_correct.py`, 15 new)
directly demonstrate the safety property the naive fix lacked —
`test_sam_is_never_rewritten` renders an actual "Sam" glyph and confirms
`correct_instances` leaves it untouched (`verify_candidate` resolves
`False` there, confirmed via direct inspection, not just an aggregate
pass/fail), while `test_confirms_a_real_digit_misread_as_letter` renders
an actual "5" glyph labeled with the misread text "Spm" and confirms
Stage B resolves `True` and the correction applies. A regression sweep
(`scripts/eval_ocr_correction.py`) ran `cicerone.detect()` — correction
enabled, the default — across all 6 synthetic fixtures + both real
photos (65 regions total): **exactly 1 correction applied (the target
case), 0 unresolved candidates, 0 unintended changes anywhere else** —
`stylized-italic.png`'s pre-existing unrelated OCR noise ("Suoked
Dkplay" for "Stroked Display") was correctly left alone, since it has no
am/pm-shaped token to trigger Stage A at all. Live-verified via a fresh
`POST /api/detect` against the running server: r4 returns `"Open 9am to
5pm"` with `ocr_correction: {applied: true, original_text: "Open 9am to
Spm", ...}` correctly round-tripped through the manifest JSON API.

197 unit tests pass (15 new). `InstText.ocr_correction` also wired
through `manifest_store.py`'s serialization (mirrors `tm_suggestion`).

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
