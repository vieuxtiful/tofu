# ToFU — Cross-Layer Architecture: A Scientific Exposition

**vieuxtiful — 2026-07-26**

> **Revision note.** §2.1 (ToFU) and §2.2 (Cicerone) were audited against the
> implementation on 2026-07-26. Several claims in the first draft described
> behaviour the code did not have on its primary path, and several stated gaps
> turned out to be either wrong or already closed. Both sections have been
> rewritten against verified evidence, and the numbers they cite are
> reproducible with `scripts/eval_detect.py` and `scripts/eval_tofu.py`.
> Sections 2.3–2.12 are carried over from the first draft and have **not**
> been re-audited to the same standard; treat their completeness scores as
> provisional.

---

## 1. SYSTEM OVERVIEW

ToFU (Text-over-Frame Unification) is a context-aware visual text localization
pipeline. It detects text in an image, erases it, and re-renders it in another
language while preserving the scene's visual context. The pipeline operates as
a fixed-stage sequence orchestrated by `TofuPipeline`
(`src/tofu/core/pipeline.py`), threading a single `TextManifest` through every
layer.

### Pipeline Stage Order (v2)

```
tofu (pre-flight) → scene PRE-PASS → cicerone → tofu re-validation →
scene enrichment → cleanse → scribe → garnish → verify → memory
```

Scene runs twice: `analyze_regions()` before Cicerone to constrain detection to
candidate surfaces, and `analyze()` after to enrich each detected instance with
style and background profiles. ToFU likewise re-validates once real text
regions exist.

### Core Data Contract

Every layer exchanges a `TextManifest` (`src/tofu/core/types.py:242`) and
nothing else:

- `InstText` instances (per-region: bounding box, polygon mask, OCR text, style
  profile, background profile, confidence, correction provenance, recognition
  history, garnish overrides)
- `SceneRegion` list (candidate text-bearing surfaces)
- `SemanticTextUnit` list (Basil's cross-region linguistic units)
- `src_lang` — the scene's own language, voted from per-region evidence by
  `cicerone.taste_the_room()` and set inside `build_manifest`

### Execution Modes

Each layer supports three modes (`LayerMode`): **AUTO** (fully algorithmic),
**MANUAL** (user-authored, layer bypassed), **HYBRID** (algorithmic with pause
checkpoints).

### Dependency Philosophy

Heavy backends are optional and resolved at call time, never at import.
PaddleOCR and LaMa run out-of-process under isolated interpreters. scikit-image,
uharfbuzz and freetype-py degrade to reduced-quality paths when absent.
Importing `tofu` pulls in none of them.

---

## 2. LAYER-BY-LAYER EXPOSITION

### 2.1 ToFU — Pre-Flight Validation (Layer 0)

**File**: `src/tofu/layers/tofu.py` (888 lines)

**Purpose**: ToFU is the pipeline's gatekeeper. Before any heavy processing, it
verifies that the target language can be rendered without "tofu" (the Unicode
term for missing-glyph boxes — the exact failure the project is named to
prevent). It checks script support, glyph coverage, source-side segmentation
feasibility, text-expansion fit and render quality, producing a `VldtnReport`
with issues, insights and suggested actions. It runs twice: once before
detection and once after Cicerone, with real region geometry.

**Implementations**:

- **fontTools** (`ttLib.TTFont`, `TTCollection`) via `FontRegistry`
  (`src/tofu/layers/fonts.py`): discovers installed fonts, parses cmap tables,
  scores per-script glyph coverage.
- **Static fallback map** — when no registry is available, `SUPPORTED_SCRIPTS`
  maps ISO 15924 codes to `ScrptSpprt.FULL/PARTIAL/UNSUPPORTED`. This is a
  genuinely different, weaker check, and the layer says so.
- **stdlib `unicodedata`** — UAX #11 East Asian Width for script-normalized
  width estimation when font metrics are unavailable.

**Mechanism**: ToFU maps target language → ISO 15924 script → font coverage
percentage. `FontRegistry` extracts `getBestCmap()` and compares against
`LANG_REQUIRED` character sets. Coverage ≥ 99.5% = FULL, ≥ 80% = PARTIAL, else
UNSUPPORTED. For expansion feasibility (`ToFU_005`) it predicts target width
from `avg_advance_em × font_px × character_count × expansion_ratio`; when a real
translation exists it measures that text directly.

**Which validator you get matters.** Both the server's `/api/validate`
endpoint (`server/main.py`) and `TofuPipeline` now construct ToFU with a real
`FontRegistry` — the pipeline receives the caller's existing registry by
injection (`ToFU(font_registry=…)`) rather than paying for a second font scan.
Before this was wired, the pipeline stage called a module-level singleton built
with no font library, so every pipeline-driven run silently validated against
the static script map while the caller was already holding 409 discovered
fonts. `TofuPipeline._tofu_context()` supplies the requested font, the smallest
region's `font_px` and the union of active effects, which is what makes
ToFU_004 and ToFU_005 measurable rather than assumed.

**Completeness Score**: **80%**

**Robust**:

- Script support mapping covers **63** languages with correct ISO 15924
  assignments
- Expansion factors: 26 languages named individually from
  W3C/localisation-vendor guidance, with **per-script fallbacks** for the
  remaining 37 so an unlisted language inherits a plausible figure instead of
  silently claiming English-equivalent width (`expansion_factor()`)
- Font registry provides true glyph-level coverage scoring on **both** the
  server preflight endpoint and the pipeline stage
- Expansion feasibility distinguishes predictive (blocking) from
  actual-translation (non-blocking) cases
- Script-normalized width estimation via UAX #11: a CJK glyph counts as one em,
  a Latin letter as half, combining marks as zero. The previous raw character
  count scored 出口 → "Exit" as a 2× overflow when the two occupy identical
  width
- Glyph-segmentation feasibility is scored from **measured source geometry**
  (median box height, small-text tail fraction) rather than from a per-language
  constant, and raises no issue at all pre-detection — a guess made before
  examining a pixel is not evidence
- Both warn gates calibrated from a measured distribution
  (`scripts/eval_tofu.py`, nine fixtures × twelve target languages) rather than
  chosen by hand
- Transform clipping warnings (ToFU_007) for skew/scale/offset
- Font-evidence insights bridge Cicerone's visual font matching to the preflight
  UX without auto-selecting fonts
- Semantic substitution insights surface Basil's cross-region units at the gate

**Calibration evidence** (`scripts/eval_out/tofu-calibration-*.json`):

| Check | Before | After | Note |
|---|---|---|---|
| ToFU_003 | 54/108 runs | 12/108 runs | was a pure function of target language, identical for all 9 assets; now fires on one asset across all languages |
| ToFU_004 | 0/108 runs | 24/108 runs | was unreachable without ToFU_001 having already errored; 22 of 24 now fire independently |
| ToFU_005 | 59/108 runs | 59/108 runs | count unchanged; **severity** now tracks the evidence behind it |

ToFU_003 now fires on japan-street (median box 18px, min 8px) — the fixture with
the worst measured OCR quality of the nine (garbage fraction 0.118, recall
0.571). ToFU_004 fires on exactly the two fixtures containing 8px text
(japan-street, gemini-street) and stays quiet on the six synthetic fixtures and
on china-street.

**Severity has to follow the evidence.** Populating `src_lang` correctly made
ToFU_005's predictive branch fire for real on CJK sources — ja→en is a 1.67×
expansion, not the 1.0× an unset `src_lang` had been implying. That is a true
and useful planning signal, but with no font selected the computed `fit` *is*
the language-pair ratio: a single table lookup, identical for every region in
the project regardless of its box. Emitting a per-region blocking ERROR from a
number containing no per-region information failed every CJK→European run at
preflight, before the user had entered a single translation. The blocking
severity is now conditional on the fit having been *measured* — from font
metrics or from an actual translation — and the ratio-only estimate says so in
its own message. This is a case where fixing one defect (§T3) exposed a second
that the first had been masking.

**Incomplete**:

- `_predict_render_quality()` remains a stub returning 0.75; the operative score
  (`_deterministic_render_score`) is a weighted formula, not the learned
  de-rendering model cited (Shimoda et al., 2021). It is now at least
  *reachable* and evidence-fed, but it is not the cited method
- The segmentation score is a geometric proxy, not a segmentation model. It
  tracks source text size, which is one cause of segmentation difficulty rather
  than all of them — gemini-street has the worst detection recall (0.333) yet
  scores 0.885, because its problem is scene density, not glyph size
- Measured stroke ratio is available on every enriched region but is
  deliberately **not** a term in the segmentation score: across these nine
  fixtures it did not separate them (the thinnest strokes, gemini-street at
  0.050, sit with the worst recall while china-street's 0.070 has the best). A
  term the evidence does not support would be noise
- No video asset validation (pipeline returns FAILED for video)
- No font subfamily resolution for italic/light variants in the coverage check
- Issue codes skip ToFU_006; the gap is historical, and codes are append-only
  because they appear in stored manifests and published eval reports

---

### 2.2 Cicerone — Text Detection & Localization (Layer 1)

**File**: `src/tofu/layers/cicerone.py` (3379 lines — the largest layer)

**Purpose**: Cicerone is the pipeline's eye. It finds every text instance and
produces the `TextManifest` all downstream layers consume: bounding boxes,
polygon masks, OCR text, confidence scores, per-instance language, and the
scene's overall source language. It is a multi-pass system: initial detection,
language-adaptive re-detection, zoom rescue, vertical-stack re-splitting,
PaddleOCR rescue, confidence-gated re-reads, hallucination pruning, hybrid
arbitration, and post-recognition correction (Savor, Wasabi, Menu).

**Implementations**:

- **EasyOCR** (default) — CRAFT detector + CRNN recognizer with CTC decoding.
  CRAFT produces 4-point polygons from character-region and affinity-region
  score maps; CRNN decodes each cropped region character-by-character.
- **PaddleOCR** (optional, out-of-process) — DBNet detector + SVTR-family
  recognizer, under an isolated `.venv-paddle` interpreter because
  paddlepaddle force-replaces numpy/opencv on install. Measured 4× recall and
  4× transcription accuracy over EasyOCR on dense/vertical CJK scenes.
- **NullBackend** — empty results when no engine is installed.
- **Pillow** — pre-resize of oversized images (`max_dim=2560`) and bicubic
  upscale of undersized ones (`min_upscale_dim=850`).
- **OpenCV** — perspective rectification of polygon crops before re-recognition
  (EasyOCR path only; the Paddle worker does not yet apply polygons).

**PaddleOCR model versions are per-language, not global.** Verified against
`.venv-paddle` (paddleocr 3.7.0 / paddle 3.3.1): `ch`, `chinese_cht`, `en`,
`japan` and the Latin set resolve to `PP-OCRv6_medium_det` +
`PP-OCRv6_medium_rec`; everything else (korean, arabic, cyrillic, …) falls back
to PP-OCRv5 weights. Quoting a single "PP-OCRvN" for the adapter is wrong in
both directions. That `japan` and `ch` resolve to the *same* PP-OCRv6
checkpoint is precisely the shared-vocabulary defect Wasabi (§2.7) exists to
clean up after.

**Engine selection.** `OCR_ENGINE` defaults to `auto`. Under `auto`/`hybrid`,
`hybrid_audit()` consults PaddleOCR opportunistically whenever it is available —
so Paddle participates in the default configuration, not only when explicitly
selected. `OCR_ENGINE=paddleocr` makes it the primary detector instead.

**Completeness Score**: **82%**

**Robust**:

- Multi-pass detection: initial → language-adaptive re-detection → zoom rescue →
  vertical-stack re-split → Paddle rescue → confidence-gated second look →
  hallucination pruning
- **Two-phase scene filtering**: a deliberately lenient geometric
  survive-for-rescue gate *before* language rescue, then a label-adaptive
  confidence prune with a high-containment geometry bypass. Pruning by
  confidence before rescue discards exactly the candidates rescue exists to
  save — measured on japan-street's banner, a correct ja-charset read scored
  0.015, *below* the wrong-charset garbage at 0.087
- `merge_vertical_columns()` unifies stacked CJK character boxes into column
  detections; `_split_tall_detections()` / `_segment_vertical_bands()` provide
  the complementary direction, re-segmenting over-merged vertical stacks by
  ink-gap band analysis
- `probe_uncovered_surfaces()` and `run_paddle_rescue()` with `_subdivide_bbox`
  tiling and box-completeness retries
- **Rule-based language identification with multi-reader probing**:
  `ScriptDetector` (Unicode-range classification), `guess_latin_language`
  (stopword + diacritic scoring with an evidence-breadth guard),
  `_auto_probe_language` over candidate language sets, a kana-priority Japanese
  override, and `_disambiguate_ja_zh` with explicit evidence and confidence
  floors
- `taste_the_room()` votes the scene's source language, area-weighted so a
  storefront sign outranks incidental Latin fragments that out-*count* it
- `_compose_crop_text()` joins all detections in a re-recognized crop in reading
  order; `_reading_order_detections()` does baseline-tolerant line clustering
- Full recognition provenance: `recognition_history` records **rejected**
  candidates, not only applied corrections
- PaddleOCR out-of-process isolation with fallback to NullBackend
- Small-source upscale path for sub-850px street photography

**Incomplete**:

- Detector orientation handling is on but coarse: the Paddle worker enables
  `use_textline_orientation`, now forwardable from the backend rather than
  hardcoded, but there is no vertical-native detection path — EasyOCR's CRAFT
  link stage groups horizontally only, so vertical CJK still depends on the
  merge/split mitigations above
- Language identification is rule-based, not a learned LID model
- No video frame processing (`temporal_span`, `track_id`, `frame_index` exist
  but are unused)
- No device auto-detection: `gpu` is a real constructor parameter on both
  backends and is threaded through the adaptive and rescue paths, but nothing
  ever sets it from the environment, so runs are CPU-bound in practice
- Reading order for mixed horizontal+vertical signage is best-effort
- `hybrid_audit` adjudicates trailing-punctuation artifacts only. Its gate
  previously also admitted anything under 0.72 confidence, but acceptance
  required a punctuation signal, so those regions could only ever be logged as
  rejected. General low-confidence CJK arbitration remains unimplemented and
  needs its own evidence test rather than borrowing this one
- `_no_terminal_dash_ink`, the evidence test behind that arbitration, accepts
  any wide-short component in the right 28% of a crop. On a two-kanji crop it
  fires on the glyph's own horizontal strokes (measured: a 12×4 component at
  the crop's top edge, and a 4×1 speck *inside* the second glyph's bounding
  box), which blocks a correction the independent engine got right at 0.969
  confidence. A terminal dash is separated from the last glyph and vertically
  centred; neither property is checked

**Integrity note.** Until 2026-07-26 `hybrid_audit` contained a branch keyed on
a specific asset id, region id and literal before/after strings, OR'd into its
acceptance condition. It has been removed. Measurement showed it was already
unreachable — it required region `r1`, and re-detection assigns that text `r2`
(see §4.1 on positional ids) — so removal was verified as an exact behavioural
no-op. No result in this document depends on it.

**Detection quality is unchanged by any of the above.** The three-image sweep
before and after the §2.2 changes is identical on every headline metric:

| Asset | Regions | Garbage | Source | Recall | Mean norm. ED |
|---|---|---|---|---|---|
| japan-street | 17 → 17 | 0.118 → 0.118 | ja → ja | 0.571 → 0.571 | 0.396 → 0.396 |
| china-street | 11 → 11 | 0.000 → 0.000 | zh-cn → zh-cn | 0.875 → 0.875 | 0.067 → 0.067 |
| gemini-street | 27 → 27 | 0.185 → 0.185 | ko → ko | 0.333 → 0.333 | 0.267 → 0.267 |

(`scripts/eval_out/*-pre-hardcode-removal.json` vs `*-post-w1w2w3.json`.)

---

> **The sections below (2.3–2.12 and §3) are carried over from the first draft
> and have _not_ been re-audited against the implementation.** §2.1 and §2.2
> above show what that audit changed where it was performed — claims that were
> true of one code path but not the primary one, gaps that were already closed,
> and gaps that were understated. Expect similar corrections here. Treat every
> completeness score below as provisional.

### 2.3 Scene — Semantic Context & Style Analysis (Layer 2)

**File**: `src/tofu/layers/scene.py` (908 lines)

**Purpose**: Scene has two jobs. `analyze_regions()` runs before Cicerone to
detect candidate text-bearing surfaces and constrain detection, suppressing
false positives. `analyze()` runs after to enrich each detected instance with a
`StyleProfil` (text colour, font attributes) and `BgProfil` (background colour,
texture classification, containing region) for Cleanse and Scribe.

**Implementations**: OpenCV `ClassicalCVBackend` — Canny + contour analysis
(`findContours`, `approxPolyDP`) for structural surfaces; MSER for text-like
blobs; a stroke-width-transform coefficient-of-variation gate (Epshtein et al.,
CVPR 2010); BT.601 luminance statistics for background classification. Optional
`SAMBackend` (Segment Anything, class-agnostic masks, opt-in). `NullSceneBackend`
when OpenCV is absent.

**Mechanism**: Two complementary detectors run in parallel. Contour analysis
(Canny → dilation → `findContours` → `approxPolyDP`) finds near-quadrilateral
panels, with a Gaussian-blur rescue pass when the default saturates in dense
scenes. MSER detects stable connected components; each passes an SWT uniformity
gate (CoV < 0.65), and survivors merge via union-find with an envelope-area
growth cap. For enrichment, text colour is sampled from the glyph mask
(`imaging.text_mask`) and background colour from the border ring. Texture
classification ("flat" / "smooth_gradient" / "textured") drives Cleanse.

**Completeness Score**: **75%** (provisional)

**Robust**: dual-detector approach with SWT filtering; rescue pass for
edge-saturated scenes; MSER cluster growth cap; orientation-aware
deduplication; texture-classification agreement with Cleanse; GarnishProfile
inheritance from the containing region; user-set style values never overwritten.

**Incomplete**: SAM integration is stub-like; no learned surface classifier
(labels are heuristic); style estimation limited to colour plus basic font
attributes; no perspective-aware surface geometry; coarse 3-category texture
classification; no video temporal analysis.

---

### 2.4 Cleanse — Text Erasure & Background Reconstruction (Layer 3)

**File**: `src/tofu/layers/cleanse.py` (666 lines) +
`src/tofu/layers/inpaint_providers.py` (458 lines)

**Purpose**: Cleanse removes source text and reconstructs the background
underneath, using stroke-level glyph masks rather than whole bounding boxes, and
selecting a fill strategy from Scene's texture classification.

**Implementations**: OpenCV — Telea inpainting (Telea, 2004) for textured
surfaces, `dilate` for mask growth, `distanceTransform` for feathered alpha,
GrabCut via `imaging.text_mask`. NumPy — per-channel linear plane reconstruction
for gradients, border-ring median fill for flat surfaces. scikit-image — Sauvola
local thresholding. Optional out-of-process LaMa and BrushNet, confidence-gated.

**Mechanism**: The glyph mask comes from Otsu (even illumination) or Sauvola
(uneven), refined by GrabCut and dilated ~3px for anti-aliased edges. A polarity
sanity check compares masked-region colour to the crop edge and falls back to
the bbox rectangle if the mask selected background instead of ink. Strategy:
flat → border-ring median; smooth_gradient → per-channel plane fit; textured →
Telea with radius scaled to stroke width. Every fill is alpha-feathered via
distance-transform falloff. Neural providers route through `inpaint_providers.py`
with confidence gates, review flags and `inst.repair_provenance`.

**Completeness Score**: **80%** (provisional)

**Robust**: stroke-level masking; three deterministic strategies matched to
texture; polarity check for GrabCut partition failures; feathered blending
throughout; neural provider routing with full provenance; group-based repair for
nearby regions; bbox fallback so erasure is never silently skipped; polygon
clipping preserves detector/user geometry at ≥78% ink retention.

**Incomplete**: no temporal consistency for video; LaMa/BrushNet are
configuration-dependent; GrabCut can still fail on multi-colour ink (mitigated,
not solved); no learned scene-text removal (DiffSTR, EnsNet); Telea degrades on
large complex regions; no self-assessment of reconstruction quality.

---

### 2.5 Scribe — Style-Aware Text Regeneration (Layer 4)

**File**: `src/tofu/layers/scribe.py` (1545 lines)

**Purpose**: Scribe renders target-language text onto the cleansed asset,
matching Scene's captured styling. It is deliberately time-ignorant, consuming a
static `RenderParams` per region.

**Implementations**: Pillow — `ImageFont.truetype`, `ImageDraw.text`,
`Image.alpha_composite`, with a joint binary search over font size and word
wrap. OpenCV — `cv2.remap` for arc/warp transforms with LANCZOS4. arabic-reshaper
+ python-bidi — contextual joining forms and UAX #9 reordering for Arabic and
Hebrew. Optional uharfbuzz + freetype-py via `knead.py` — HarfBuzz shaping and
FreeType rasterization for Brahmic scripts where Pillow's BASIC layout engine
produces incorrect conjuncts and mark positioning.

**Mechanism**: Per region, Scribe renders `inst.target_text` wrapped to the
bounding box, binary-searching wrap configuration and font size together. Text
is rendered on a transparent layer, optionally transformed (skew, scale, arc
warp, offset) with local affine transforms and adaptive supersampling, then
alpha-composited. RTL text passes through arabic-reshaper then python-bidi. For
Brahmic scripts, `knead.py` shapes via HarfBuzz (GSUB conjuncts, GPOS marks) and
rasterizes glyph IDs through FreeType.

**Completeness Score**: **83%** (provisional)

**Robust**: joint font-size + wrap search; local rather than global affine
transforms; adaptive supersampling (4× per target pixel, capped at 18MP); RTL
pipeline; HarfBuzz shaping for Brahmic; CJK vertical (tategaki) rendering; real
italic/weight faces preferred over synthesis; glyph-coverage checking with
best-covering-font swap; named warp presets; font file cache; `.ttc` face index
handling.

**Incomplete**: Pillow's Windows wheel lacks Raqm, so no OpenType
kerning/ligatures for Latin/Cyrillic/Greek/CJK; Indic and Thai/Khmer correctness
depends on optional uharfbuzz+freetype-py; no bidi itemization (mixed
Arabic+Latin shapes in one direction); arc warp is a lightweight baseline, not
perspective reconstruction; no video temporal rendering; no learned style
transfer (SRNet, STEFANN); underline/strikethrough unimplemented; no OpenType
feature control.

---

### 2.6 Basil — Semantic Registration & Target-Span Substitution

**File**: `src/tofu/layers/basil.py` (1250 lines)

**Purpose**: Basil bridges OCR's geometric facts (immutable region IDs and boxes
that anchor Cleanse and Scribe) and translation's linguistic reality (sentences
and named entities spanning several visual regions). It registers nearby source
regions as reading units, retains both source visual order and target semantic
order, aligns user-supplied target phrases back to region anchors, and fails
open when evidence is insufficient.

**Implementations**: Pure Python — a typology table with WALS-derived features
(adjective order, genitive order, designator position, prenominal class) for 20+
languages; role vocabulary partitioned by language; script classification via
Unicode ranges; reading-order reconstruction. Optional offline Stanza for
dependency parsing, NER and cross-lingual alignment, gated on
`TOFU_BASIL_STANZA_DIR` and never auto-downloading.

**Mechanism**: Three phases. Registration groups nearby regions into
`SemanticTextUnit` objects by spatial proximity and script-appropriate reading
order. Pairing compares source and target typological features to decide whether
cross-region rearrangement is linguistically `possible`, `unnecessary`, or
`unknown`. Substitution aligns semantic blocks to visual cubes when a user
supplies a target phrase, recording target-region order while preserving every
captured box. It is advisory at ToFU's gate and never changes geometry.

**Completeness Score**: **72%** (provisional)

**Robust**: WALS-derived typology for 20+ languages; script-aware joining; RTL
and vertical CJK reading order; three-valued pairing verdict that fails open;
compact fr→it glossary as evidence rather than an MT system; never renumbers or
moves Cicerone regions; substitution provenance recorded.

**Incomplete**: Stanza optional and offline-only; glossary covers only fr→it; no
automatic target generation; OCR repair for fragmented gazetteer entities is
proposed, never auto-applied; typology table hand-maintained; no bidirectional
mixed-script handling; limited to place-name/street-sign domains.

> Note: `pairing()` reads `manifest.src_lang`, which until 2026-07-26 was set by
> only one server endpoint and was `None` on every pipeline-driven run. It is
> now populated in `build_manifest` (§2.2), so Basil's verdicts have real input
> on those paths for the first time. Its behaviour there has not been re-measured.

---

### 2.7 Wasabi — CJK Variant Normalization

**File**: `src/tofu/layers/wasabi.py` (80 lines)

**Purpose**: Wasabi corrects a specific, confirmed defect in PaddleOCR's shared
Japanese/Chinese recognition model. `lang="japan"` resolves to the same
checkpoint used for Chinese — verified in §2.2 as `PP-OCRv6_medium_rec` for both
— whose CTC vocabulary contains simplified-Chinese-only and Japanese shinjitai
forms as separate valid tokens with no language conditioning. A Japanese read can
therefore confidently emit a Chinese-only character (measured: 劇場通り → 剧場通,
焼肉 → 烧肉).

**Implementations**: Pure Python — a curated, growable `SIMPLIFIED_TO_JAPANESE`
dictionary.

**Mechanism**: `season()` iterates every ja-labeled instance in the final
manifest, replacing mapped characters with their shinjitai equivalents and
recording the correction on `inst.ocr_correction` in the same shape Savor and
Menu use.

**Completeness Score**: **65%** (provisional)

**Robust**: deliberately not a full Unihan variant table — most simplified forms
match shinjitai, so only genuinely divergent pairs are included; full provenance;
runs after Savor and Menu so it corrects whatever reached the final manifest;
add-only discipline.

**Incomplete**: only 3 confirmed pairs (剧→劇, 烧→焼, 岛→島); no detection of
when PaddleOCR is not the engine (harmlessly no-ops); no traditional-Chinese →
shinjitai normalization; no Korean Hanja; growth is manual.

---

### 2.8 Menu — Gazetteer-Assisted Place-Name Correction

**File**: `src/tofu/layers/menu.py` (455 lines)

**Purpose**: Menu recovers low-confidence reads of real, well-known place names
against a small gazetteer. Whole-string place names cannot be pixel-verified the
way Savor verifies single glyphs, so Menu uses a stricter gate: only touch reads
the recognizer was already unsure about, and only apply strong fuzzy matches.

**Implementations**: Pure Python — `difflib.SequenceMatcher`, positional (zip)
diffing, a curated `KNOWN_PLACES` gazetteer and `KNOWN_SIGNAGE` vocabulary.
Optionally delegates to Savor's `chew_swaps` for pixel verification.

**Mechanism**: Two mutually exclusive courses. Substring: when a gazetteer name
aligns inside a strictly-longer read, Menu slides an exact-length window and
diffs positionally; a long single-diff span (≥4 chars, ≤1 diff) applies on
string evidence alone, a weaker span only when Savor confirms every differing
glyph, and a corroboration tier allows a pixel-inconclusive single-diff span when
≥2 independently-confirmed names co-occur. Whole-string: fuzzy match gated by
`CONFIDENCE_FLOOR=0.6` and `SIMILARITY_FLOOR=0.5`.

**Completeness Score**: **70%** (provisional)

**Robust**: two-tier approach with mutual exclusivity; positional diffing
disambiguates alignments SequenceMatcher cannot; pixel verification for weak
string evidence; document-level corroboration; full provenance;
language-partitioned gazetteer; greedy non-overlap resolution.

**Incomplete**: gazetteer is small (8 place names, 13 signage words) and manual;
Japanese only; no growth from OpenStreetMap/GeoNames; no matching beyond
SequenceMatcher ratio; fixed confidence floor; substring path rejects spaced text.

---

### 2.9 Garnish — Post-Scribe Scene Integration

**File**: `src/tofu/layers/garnish.py` (248 lines)

**Purpose**: Garnish applies deterministic, source-derived edge treatment to
newly rendered text pixels only. The cleansed base is never modified, so
neural/manual repair pixels stay reversible and QA can assert outside-mask
preservation.

**Implementations**: Pillow — `GaussianBlur`, `alpha_composite`. OpenCV —
`distanceTransform` for coverage feathering, `filter2D` for angled motion blur,
`erode`/`dilate`, `fillPoly`, `remap`. NumPy — seeded noise, gamma correction,
edge weighting.

**Mechanism**: Garnish diffs the scribed asset against the cleansed base to
isolate new text pixels, then applies edge blur, distance-transform edge
smoothing, erosion/dilation, seeded grain weighted toward glyph edges, gamma
shift, and motion-blur smudge along a real rotated path — each alpha-masked to
the text coverage area. The "engrain" pass blends text RGB toward the underlying
surface in the soft edge band so edges look embedded rather than haloed.

**Completeness Score**: **68%** (provisional)

**Robust**: only newly rendered pixels are modified; source-confidence-scaled
intensity; motion blur along real rotated paths; engrain pass; per-region
polygon-scoped sub-regions; deterministic seeded noise; Scribe's alpha coverage
preferred over the capture box.

**Incomplete**: no learned weathering model; no perspective-aware treatment;
limited treatment vocabulary; no colour-temperature matching; no shadow casting;
sub-region definition is manual; no video temporal consistency.

---

### 2.10 Memory — Visual Translation Memory (Layer 6)

**File**: `src/tofu/layers/memory.py` (196 lines)

**Purpose**: Memory stores accepted (source region, target render, style, QA
score) tuples for reuse. It is storage-agnostic by design — `update()` returns
draft records and `lookup()` matches against a caller-supplied pool; the server
owns the SQLite writes.

**Implementations**: OpenCV — DCT perceptual hash (pHash, Zauner 2010) via
`utils/phash.py`: 32×32 grayscale, 2D DCT, top-left 8×8 block thresholded
against the median. `difflib.SequenceMatcher` via `utils/textmatch.py`. Pillow
for thumbnail crops.

**Mechanism**: After a run passes the QA gate, each translated non-DNT instance
above threshold becomes a draft record (source text, normalized text, target
text, style fingerprint, pHash, QA score, thumbnail). Retrieval is three-tier:
EXACT normalized match, FUZZY at ratio ≥ 0.85, then VISUAL at pHash similarity
≥ 0.88. Tied candidates prefer a matching style fingerprint.

**Completeness Score**: **74%** (provisional)

**Robust**: three-tier matching with graceful degradation; QA gating enforced in
both pipeline and `update()`; per-instance gate as well as overall; style
fingerprint tiebreaker; never imports server/db.py; pHash robust to
crop/scale/compression; DNT and untranslated regions excluded.

**Incomplete**: no storage backend in-module; no incremental index; no semantic
similarity; 64-bit pHash may collide on similar signs; visual matching ignores
colour/style; no hit-rate analytics; no cross-language TM.

---

### 2.11 Savor — Glyph-Confusion Repair

**File**: `src/tofu/layers/savor.py` (659 lines)

**Purpose**: Savor is Cicerone's quality-control taste tester. CRNN+CTC decodes
character-by-character with no language model, so a locally ambiguous glyph
(5/S, 0/O, 1/I) can be read wrong with high confidence — the model is confident
about the shape, not about semantic plausibility.

**Implementations**: Pillow renders both candidate characters as reference
glyphs at matching size; OpenCV performs connected-component analysis of the
glyph mask; NumPy computes IoU between rendered candidates and observed ink.

**Mechanism**: Three deliberately separate courses. SNIFF (`sniff_out`) is a
cheap context test that only proposes. CHEW (`chew_on`) segments the ambiguous
glyph's own pixels, renders both candidates, and compares by IoU — only a clear
margin (`BITE_MARGIN=0.12`) settles it. SWALLOW OR SPIT (`taste`) rewrites text
only when chew confirms. A second course handles Japanese dakuten/handakuten
confusion.

**Completeness Score**: **73%** (provisional)

**Robust**: context proposes, pixels verify, orchestrator decides; pixel-level
shape comparison; dakuten course; inconclusive results recorded with
`applied=False` for review; small-glyph bicubic rescue; bidirectional confusable
table; full provenance.

**Incomplete**: only one context grammar (time tokens); dakuten course is
Japanese-only; IoU only, no SSIM or Hu moments; uses Scribe's fallback font
rather than the detected source face; single ambiguous glyph per token;
connected-component analysis fails on touching glyphs.

---

### 2.12 Verify — Quality Assessment (Layer 5)

**File**: `src/tofu/layers/verify.py` (684 lines)

**Purpose**: Verify scores the localized asset per instance, producing a
`QAReport`. The pipeline gates success and Memory storage on
`overall_score ≥ PipelineCfg.qa_threshold`.

**Implementations**: EasyOCR for OCR round-trip legibility and residual
source-text detection (reusing Cicerone's backend, no new dependency); NumPy for
SSIM (Wang et al., 2004, eq. 13), CIE76 ΔE, Sobel gradients and ink presence;
OpenCV via `imaging.text_mask`; `difflib.SequenceMatcher` for text similarity.

**Mechanism**: Five metrics from the scene-text-editing literature: OCR
round-trip legibility (SRNet/Wu et al. 2019 protocol); background reconstruction
SSIM over a border ring *outside* each bbox (pixels the pipeline had no licence
to change); ink presence as an EasyOCR-free fallback; residual source text,
which multiplies the score down rather than averaging in; and style consistency
via CIE76 ΔE plus rendered-height ratio. Garnish edge similarity (Sobel
orientation histograms) and texture match are also computed. Untranslated
non-DNT regions score 0.0 rather than neutral, preventing incomplete
localizations from inflating their own gate score.

**Completeness Score**: **80%** (provisional)

**Robust**: five complementary cited metrics; missing metrics skipped, never
failed; residual-source penalty multiplies down; coverage accounting prevents
score inflation; full progress breakdown on `QAReport.progress`; CIE76 ΔE;
Sobel-histogram garnish similarity; outside-mask preservation check; per-instance
scoring.

**Incomplete**: single-window SSIM rather than the windowed mean-map variant;
OCR round-trip uses the same engine that did the detection, so it is not fully
independent; no learned quality model; style consistency limited to colour and
height; no perceptual metric (LPIPS, FID); garnish scoring inconclusive on small
crops; no video temporal assessment.

---

## 3. AUXILIARY MODULES

*(carried over; not re-audited)*

**3.1 Typography** (`layers/typography.py`, 209 lines) — visual font attributes
(weight, slant, size, rotation) from pixels via citable classical methods:
stroke-width analysis (Epshtein et al., CVPR 2010), shear-search de-slanting
(Vinciarelli & Luettin, 2001), `cv2.minAreaRect`. Every estimator degrades to
None on weak evidence. **70%** — robust on clean crops; weak on decorative
faces; no family identification.

**3.2 Font Matching** (`layers/font_matching.py`, 367 lines) — evidence-gated
visual font identification: renders the recognized string in installed faces,
normalizes at measured cap height, then combines silhouette Dice, symmetric
Chamfer, projection-profile and natural-width evidence. **65%** — a
deterministic analogue of visual-font retrieval; no DeepFont-style embedding;
commercial catalog integration is stub-like.

**3.3 Font Registry** (`layers/fonts.py`, 275+ lines) — fontTools discovery,
cached per-font per-script coverage, per-script ranking, and (added 2026-07-26)
`pantry()`, the single font-directory resolver shared by the server and both
eval harnesses. **85%** — robust cmap scoring; no variable-font or OpenType
feature detection.

**3.4 Knead** (`layers/knead.py`, 405 lines) — HarfBuzz shaping + FreeType
rasterization seam for Brahmic scripts, degrading to Pillow. **78%** — correct
for Devanagari/Bengali/Tamil; no Tibetan, no bidi itemization, no OpenType
feature control.

**3.5 Inpaint Providers** (`layers/inpaint_providers.py`, 458 lines) —
confidence-gated repair-provider routing for Cleanse; LaMa and BrushNet
out-of-process. **72%** — routing and provenance robust; availability is
configuration-dependent; no automatic provider selection.

**3.6 Imaging Utils** (`utils/imaging.py`, 206 lines) — shared asset loading and
`text_mask()`, the widest-fanout function in the package (Scene, Cleanse,
Typography, Savor, Font Matching, Verify). Otsu for even illumination, Sauvola
for uneven, GrabCut refinement. **82%** — robust adaptive binarization; weak on
severely degraded images and multi-colour ink.

**3.7 pHash** (`utils/phash.py`, 68 lines) — DCT perceptual hash for visual TM
matching. **85%** — textbook implementation; limited by 64-bit resolution.

**3.8 TextMatch** (`utils/textmatch.py`, 32 lines) — normalization plus fuzzy
similarity. **90%** — simple, correct, well-scoped.

---

## 4. CROSS-CUTTING ARCHITECTURAL PRINCIPLES

### 4.1 Region Anchors — Immutable Downstream, Positional Across Captures

Cicerone's region IDs and bounding boxes are durable anchors for Cleanse,
Scribe, Verify and Memory. Basil can route a target semantic block into a
different existing cube before Scribe renders it, but never renumbers or moves
regions. This separation between capture geometry and linguistic semantics is
the project's core architectural insight.

The immutability is **downstream of capture, not across captures**. IDs are
positional (`id=f"r{order + 1}"`), and `build_manifest` runs up to five times
within a single `detect()` call as each refinement pass revises the detection
set. A re-scan therefore renumbers every region: a translation keyed to `r7`
rebinds to whatever text now sorts seventh. This is not hypothetical — it is
what rendered the fixture-keyed branch described in §2.2 unreachable.

### 4.2 Confidence-Gated Evidence

Every correction layer (Savor, Menu, Wasabi) records provenance on
`inst.ocr_correction` with `{applied, original_text, corrected_text, reason}`.
Cleanse records `inst.repair_provenance` with provider, confidence and review
flags. Font matching records `inst.font_match` with candidates and scores.

`inst.recognition_history` (appended via `_history()`) is the strongest link in
this chain: it records every arbitration attempt including the **rejected**
ones, with the candidate text, both confidences, and the specific evidence that
failed. A reviewer can see that an independent engine read 御嶽 at 0.969 against
a primary 御獄- at 0.682, and that the correction was withheld because the
crop-mask test did not clear.

Preflight issues carry `region_ids` alongside `region_id`: a finding that holds
for a whole (language, font, size, effects) context is emitted once with the
regions it covers, rather than once per region. Measured on an 18-region
Japanese asset with an Arabic target, where the target script has no covering
font installed:

| | Issues surfaced | Validate calls |
|---|---|---|
| per-instance (before) | 25 — 18 identical `ToFU_001` errors + 7 `ToFU_004` | 18 |
| per-context (after) | 2 — one `ToFU_001` covering 18 regions, one `ToFU_004` covering 7 | 15 |

The eighteen copies were never eighteen findings. They were one fact about a
missing font, repeated until the genuinely region-scoped finding underneath was
invisible.

### 4.3 Graceful Degradation

Every heavy dependency (Pillow, OpenCV, NumPy, EasyOCR, PaddleOCR, LaMa,
uharfbuzz, freetype-py, scikit-image, Stanza, Segment Anything) is optional and
resolved at call time. Missing dependencies reduce quality but never crash the
pipeline. The `NullBackend` pattern keeps the contract in bare dev environments.

Degradation is only honest if it is *visible*. ToFU's static-map fallback is a
weaker check than registry-backed coverage, and the layer now documents that
distinction rather than presenting both as the same validation.

### 4.4 Scene/Cleanse Agreement

Scene's texture classification ("flat" / "smooth_gradient" / "textured") drives
Cleanse's strategy selection, so the reconstruction method matches the surface.

### 4.5 Shared Glyph Mask

`imaging.text_mask()` is the single binarization function that Scene, Typography,
Cleanse, Savor, Font Matching and Verify all bottom out in. Its quality
propagates to six layers at once — which is also why ToFU's segmentation
feasibility check is a *source*-side question (§2.1).

---

## 5. IMPLEMENTATION COMPLETENESS SUMMARY

| Layer | Score | Primary Libraries | Key Gap |
|---|---|---|---|
| ToFU | 80% | fontTools, unicodedata | Learned render-quality model; segmentation score is a geometric proxy |
| Cicerone | 82% | EasyOCR, PaddleOCR, Pillow, OpenCV | Vertical-native detection, learned LID, video |
| Scene | 75%\* | OpenCV, (Segment Anything) | Learned surface classifier |
| Cleanse | 80%\* | OpenCV, NumPy, (LaMa, BrushNet) | Temporal consistency, learned STR |
| Scribe | 83%\* | Pillow, OpenCV, (uharfbuzz, freetype-py) | Raqm/OpenType features, video |
| Basil | 72%\* | Pure Python, (Stanza) | General MT, auto target generation |
| Wasabi | 65%\* | Pure Python | Small curated table |
| Menu | 70%\* | Pure Python, (Savor pixels) | Small gazetteer, Japanese-only |
| Garnish | 68%\* | Pillow, OpenCV, NumPy | Learned weathering, perspective-aware |
| Memory | 74%\* | OpenCV (pHash), difflib | No semantic matching, no storage |
| Savor | 73%\* | Pillow, OpenCV, NumPy | Limited context grammars |
| Verify | 80%\* | EasyOCR, NumPy, OpenCV | No learned QA model, single-window SSIM |

\* provisional — not re-audited against the implementation.

**Aggregate System Completeness: ~76%**

The static image pipeline is functional end-to-end. The video pipeline is
entirely unimplemented (data structures exist, no temporal processing). The
system's greatest strength is its evidence-gated, provenance-tracked correction
chain — including its record of what it declined to do. Its greatest gap is the
absence of learned models for render-quality prediction, surface classification
and holistic quality assessment.

The Cicerone score deliberately does **not** absorb the fixture-hardcode finding
in §2.2: that was a research-integrity defect rather than a completeness one,
and it is recorded there rather than amortised into a number.

---

## 6. REPRODUCING THE MEASUREMENTS

```bash
.venv/Scripts/python scripts/eval_detect.py images/japan-street.jpeg --tag <label>
```

Detection quality against ground truth (`images/*.gt.json`), written to
`scripts/eval_out/`. Repeat for `china-street.png` and `gemini-street.png`; any
change to `cicerone.py` requires the full three-image sweep.

```bash
.venv/Scripts/python scripts/eval_tofu.py --tag <label>
```

Pre-flight score distributions across nine fixtures × twelve target languages,
using `TofuPipeline`'s own context builder so the harness cannot drift from the
thing it measures. Manifests are cached under
`scripts/eval_out/tofu_manifests/`; pass `--refresh` to re-detect.

---

*End of document.*
