<div style="text" align="center">
  <img src="images/middlebury-seal.png" width="8%">
  <img src="images/middlebury-institute-seal.png" width="7%" hspace="30" style="margin-left: 40 px;"> 
</div>

---

<div style="image" align="center">
  <img src="images/tofu-wht-alt.png" width="38%">
</div>

<div align="center">
ToFU (text-over-frame unification)
</div>

---

## Summary

ToFU is a seven-layer visual translation pipeline that detects, erases, and re-renders text in images while preserving the original scene's visual context. Given a source image and a target language, ToFU runs text detection and recognition (Cicerone), semantic surface analysis (Scene), source-text erasure and background inpainting (Cleanse), style-matched target-language rendering (Scribe), and quality verification (Verify) — with a visual translation memory (Memory) that accumulates approved localizations for reuse on future assets. Three auxiliary modules — Savor, Menu, and Wasabi — operate as post-recognition quality-control passes inside Cicerone, correcting glyph-level confusions, recovering known place names, and normalizing CJK character variants. The pipeline is engine-agnostic (EasyOCR or PaddleOCR), runs entirely on CPU, and gates output on measurable accuracy thresholds.

---

## Running the stack

**Backend** (Python 3.13 venv — required; easyocr's dependency tree is not yet reliable on 3.14):

```
py -3.13 -m venv .venv
.venv\Scripts\python -m pip install -r server\requirements.txt
cd server
set PYTHONUTF8=1
..\.venv\Scripts\python -m uvicorn main:app --reload --port 8000
```

Notes:
- `PYTHONUTF8=1` is required on Windows — EasyOCR's model-download progress bar prints characters that crash a cp1252 console.
- First detection downloads OCR models (~100 MB) to `~/.EasyOCR` and takes ~20 s; warm detections run in a few seconds on CPU.

## OCR backends

Cicerone now supports two detection/recognition engines behind the same `OCRBackend` interface:

- **EasyOCR** (default): CRAFT detector + CRNN recognizer; mature for stylized scene text.
- **PaddleOCR** (opt-in): DBNet detector + SVTR_LCNet recognizer; generally stronger on rotated, curved, dense, and CJK street signs.

Switch the active engine per request with the `engine` query param (`/api/detect/stream?engine=paddleocr`) or via the `OCR_ENGINE` environment variable (`easyocr` or `paddleocr`). PaddleOCR is listed in `server/requirements.txt` and `src/tofu/requirements.txt` but is only imported when used, so EasyOCR-only installs keep working.

## Architecture: pipeline layers

The pipeline runs in a fixed stage order: **ToFU** (pre-flight validation) → **Scene** (candidate surface detection) → **Cicerone** (multi-pass text detection + recognition, scene-filtered) → **ToFU** (re-validation with detected text) → **Scene** (per-instance style/background enrichment) → **Cleanse** (source-text erasure) → **Scribe** (target-language rendering) → **Verify** (quality scoring) → **Memory** (translation-memory storage). Scene runs twice — once before Cicerone to constrain detection, once after to enrich each detected instance with style and background data.

### ToFU — pre-flight validation (`src/tofu/layers/tofu.py`)

ToFU is the pipeline's gatekeeper. Before any heavy processing begins, it checks whether the target language can actually be rendered: which Unicode script the language uses, whether the font library covers that script's glyphs, and whether the detected text will fit in its bounding box after translation. A real `FontRegistry` (backed by `fonttools`) probes the system font directory for actual codepoint coverage rather than relying on a static lookup table — so a target language is only marked `FULL` when a covering font genuinely exists. Per-script support distinguishes Simplified Han (`Hans`), Traditional Han (`Hant`), Japanese (`Jpan`), and Korean (`Kore`) rather than collapsing all CJK under one generic label.

ToFU also runs a text-expansion feasibility check: using industry-standard horizontal expansion factors (e.g., German expands ~35% relative to English, Japanese contracts to ~60%), it predicts whether a translation will overflow its source bounding box. Regions that exceed a 1.05× ratio receive a warning; those above 1.35× are flagged as infeasible and blocked before Cleanse/Scribe run. When a `TextManifest` is supplied (e.g., after Cicerone detection or from frontend-supplied regions), ToFU re-validates with real per-region data rather than heuristics alone.

### Cicerone — text detection & recognition (`src/tofu/layers/cicerone.py`)

Cicerone is the pipeline's eye. It finds every text instance in the asset and produces the `TextManifest` that all downstream layers consume: bounding boxes, segmentation polygons, OCR text, confidence scores, and per-region language identity. Detection is multi-pass — an initial full-frame scan, then adaptive re-detection on under-covered regions, then zoom passes into scene-identified surfaces at 2× resolution. A scene-filter stage uses the candidate surfaces from Scene's pre-pass to suppress false positives: detections inside panels and text clusters survive at lower confidence thresholds (text is very likely there), while detections in open areas must clear a higher bar.

Two OCR engines are supported behind the same `OCRBackend` interface:

- **EasyOCR** (default): CRAFT detector + CRNN recognizer. Mature for stylized and curved scene text; returns 4-point polygons mapped directly to `InstText`.
- **PaddleOCR** (opt-in): DBNet detector + SVTR_LCNet recognizer. Generally stronger on rotated, curved, dense, and CJK street signs — measured at 4× recall and 4× transcription accuracy over EasyOCR on dense vertical CJK scenes. Runs out-of-process via `scripts/paddle_worker.py` under an isolated `.venv-paddle` interpreter (paddlepaddle force-replaces the app venv's numpy/opencv on install, so it can never be imported in-process).

Switch the active engine per request with the `engine` query param (`/api/detect/stream?engine=paddleocr`) or via the `OCR_ENGINE` environment variable (`easyocr` or `paddleocr`). PaddleOCR is listed in `server/requirements.txt` and `src/tofu/requirements.txt` but is only imported when used, so EasyOCR-only installs keep working.

Cicerone also handles CJK-specific challenges: vertical column merge (`merge_vertical_columns`) unifies x-aligned, width-matched, tightly-stacked character boxes into single column detections; language arbitration ranks candidate readers by corroborating-region count and confidence rather than confidence alone (preventing a confidently-wrong reader from hijacking the scene); and a two-phase scene filter lets a detection survive the confidence floor with its best recognition rather than its first. A `_split_tall_detections` pass re-segments over-tall vertical CJK signs into per-character bands and re-recognizes each, feeding results through the column reassembly.

### Scene — semantic context & style analysis (`src/tofu/layers/scene.py`)

Scene has two jobs. First, a pre-pass (`analyze_regions`) detects candidate text-bearing surfaces — signs, panels, bordered regions — *before* text detection runs. Cicerone uses these to constrain detection and suppress false positives. The default `ClassicalCVBackend` combines two complementary detectors:

- **Contour analysis** (Canny edges + `approxPolyDP`): finds structural surfaces — panels, bordered regions, signs. An adaptive rescue pass retries with higher thresholds and Gaussian blur when the default pass finds zero real candidates (recovering large banners in visually cluttered scenes where edge detection saturates).
- **MSER** (Maximally Stable Extremal Regions): finds text-like blobs — individual characters and short strings that contour analysis misses. Blobs are spatially clustered into bounding boxes labeled `"text_cluster"`.

Surface deduplication is orientation-aware: a small horizontal English label sitting on a large vertical CJK sign is not discarded as a duplicate of the sign, even though it's >85% contained within it — the orientation bucket (wide/tall/square) must also match.

Second, an enrichment pass (`analyze`) fills per-instance `StyleProfil` (estimated text color, font weight, italic, size) and `BgProfil` (background color, texture classification, containing surface label) used by Cleanse for faithful inpainting and by Scribe for style-matched re-rendering. GrabCut-refined color estimation separates text from background pixels more accurately than Otsu-only thresholding, especially on textured or gradient surfaces. Existing user-set values are never overwritten.

### Cleanse — text erasure & inpainting (`src/tofu/layers/cleanse.py`)

Cleanse removes source text from the asset and reconstructs the background underneath. The erasure mask is the glyph's own stroke shape (Otsu + GrabCut, dilated a few pixels for anti-aliased edges) rather than the whole bounding box rectangle — the previous implementation also filled the full padded bbox unconditionally, which forced the inpainter to hallucinate the entire box interior and produced the leftover-artifact failure mode documented in the project workplan. The bbox rectangle is now a fallback used only when segmentation genuinely fails.

The inpainting strategy is selected from Scene's `BgProfil.texture` classification, so Cleanse and Scene agree on what kind of surface they're looking at:

- **Flat** surfaces (solid panels/signs): border-ring median color fill — fast and exact, since there's no texture to lose.
- **Smooth gradients**: per-channel linear plane reconstruction fit over the border ring (the same shading model Scene fits for classification, re-fit here over the actual erasure footprint).
- **Textured or unclassified**: OpenCV content-aware inpainting (Telea 2004), batched into one pass with radius scaled to the detected stroke width.

Every fill is alpha-feathered at the mask boundary (distance-transform falloff) so no strategy leaves a hard seam. Excluded regions (user-removed from the workspace) are still erased — only their rendering is skipped, unlike DNT regions which are left untouched entirely.

### Scribe — style-aware text regeneration (`src/tofu/layers/scribe.py`)

Scribe renders target-language text onto the cleansed asset, matching the source styling captured by Scene. Per region, it renders `inst.target_text` (untranslated and excluded regions are skipped), wrapped to fit the bounding box (greedy word-wrap for space-delimited scripts, character-wrap for CJK), and sized by a binary search over both the wrap and the font size together. Text is colored and faded per `StyleProfil` + opacity, optionally rotated, and alpha-composited onto the asset.

Font resolution anchors a sibling-face search on the fallback font that would otherwise be used: when `font_family` is `None` but `font_weight` or `italic` is explicitly requested (the common case, since typography detection sets weight independently of any explicit font pick), Scribe finds a real bold or italic sibling face in the same family rather than silently rendering plain Arial. Vertical CJK column rendering is supported for narrow-and-tall regions in vertical-writing languages — characters are stacked top-to-bottom, each rendered upright.

Italic is rendered on a layer sized to the text's own extent and sheared around its own local origin before compositing — not the whole base-image-sized layer sheared by each pixel's absolute y-coordinate, which was a real bug that bled a region's ink far outside its detection bbox.

### Verify — quality verification (`src/tofu/layers/verify.py`)

Verify scores the localized asset per instance and gates the pipeline on `overall_score >= qa_threshold`. Five metrics, all standard in the scene-text editing literature:

1. **OCR round-trip legibility** — re-recognize the rendered target text and compare against the intended string. If the OCR that found the source text cannot read the rendered target, the render failed for the same reasons a human reader would struggle.
2. **Background reconstruction (SSIM)** — structural similarity between source and localized asset over a border ring just outside each bbox. Cleanse/Scribe must not disturb pixels beyond the region.
3. **Ink presence** — when EasyOCR is unavailable, a rendered region must still show evidence of rendering: pixel change against the source inside the bbox plus glyph edge energy.
4. **Residual source text** — re-recognize the cleansed (pre-Scribe) crop and compare against the original source string. A high similarity means the source text survived the erase — a double-exposure defect that multiplies the instance's score down rather than averaging in.
5. **Style consistency** — CIE76 color ΔE between detected source text color and mean ink color sampled from the rendered crop, plus a rendered-height ratio against the detected source size.

Coverage accounting is enforced: untranslated non-DNT regions receive a real deduction rather than a neutral score, so an incomplete localization can no longer inflate its own gate by skipping regions. The `QAReport.progress` dict carries the full regions_total/dnt/excluded/translated/untranslated/fallback_font breakdown for the frontend to render as a coverage summary.

### Memory — visual translation memory (`src/tofu/layers/memory.py`)

Memory stores accepted (source region, target render, style, QA score) tuples for reuse: identical or near-identical source text regions in future assets can be localized from memory instead of re-running the pipeline. Matching uses a three-tier cascade — exact text match, fuzzy text similarity, then visual pHash fingerprint — with a style fingerprint (weight/italic/color) as a tiebreaker between equally-scored candidates.

QA gating is enforced here as well as in the pipeline: a visual TM poisoned with bad localizations is worse than no TM, so `update()` refuses to store results below the threshold regardless of caller. The module is storage-agnostic by design — `update()` returns draft records (plus a ready-to-save thumbnail crop) and `lookup()` matches against a caller-supplied candidate pool. `server/main.py` owns the actual SQLite writes and thumbnail-file saves, keeping this module pure and testable with plain Python objects.

### Savor — post-recognition glyph correction (`src/tofu/layers/savor.py`)

Savor is Cicerone's quality-control taste tester. Before recognized text leaves the detection pipeline, Savor samples every suspicious bite and only swallows a correction when the evidence backs it up. It runs alongside Cicerone's other post-recognition passes (second_look, prune_hallucinations, zoom, adaptive) as one more QC step on the same raw output.

Four courses, deliberately kept separate so nothing gets corrected on a guess:

1. **Sniff** (`sniff_out`): a cheap context check — is there a whole token shaped like something a digit was expected in (e.g., an hour before "am"/"pm"), where swapping a confusable glyph's digit counterpart would make it valid? Sniff only proposes candidates; it never rewrites anything.
2. **Chew** (`chew_on`): the actual verification. Segments the ambiguous glyph's own pixels out of the region (connected-component analysis), renders both candidate characters as reference glyphs at matching size, and compares shapes. Only a clear margin settles it — a coin-flip-close comparison is left inconclusive.
3. **Swallow or spit** (`taste`): a correction is only applied when `chew_on()` confirms the digit. Confirmed-letter candidates are rejected; inconclusive ones are recorded on `InstText.ocr_correction` with `applied=False` for human review.
4. **Dakuten** (`sniff_dakuten`): detects when Japanese kana readings are missing their dakuten/handakuten marks by matching against Menu's gazetteer of known real place names. If toggling the differing positions' marks produces a known name at sufficient fuzzy-similarity, the correction is proposed. Every differing character must be a known dakuten relation — a single non-dakuten diff aborts the entire proposal.

### Wasabi — CJK glyph normalization (`src/tofu/layers/wasabi.py`)

PaddleOCR's Japanese language selector doesn't receive a Japanese-specific recognition model — it resolves to the same shared PP-OCRv6 model used for Chinese and English, whose CTC vocabulary contains both simplified-Chinese-only and correct Japanese shinjitai glyph forms as separate valid output tokens with no language conditioning. A `ja`-labeled read can confidently emit a Chinese-only character (measured: `劇場通り` read back as `剧場通`, `焼肉` as `烧肉`). Wasabi corrects this after the fact: `season()` swaps any known simplified-Chinese-only glyph for its Japanese shinjitai equivalent across every ja-labeled instance. The correction table is deliberately curated rather than a full Unihan variant table — most simplified forms actually match Japanese shinjitai (both diverged independently from the same traditional form and landed on the same simplification), so only genuinely observed divergent pairs belong here.

### Menu — gazetteer-assisted place-name recovery (`src/tofu/layers/menu.py`)

A low-confidence OCR read of a real, well-known place or establishment sign (a train-station gate, a named street, a chain storefront) can often be recovered by checking it against a small list of known names, even when the pixels alone weren't legible enough. Menu uses a stricter gate than Savor's pixel-verified glyph swaps: it only touches reads the recognizer was already unsure about (confidence below 0.6), and only applies a candidate that's a strong fuzzy match (similarity ≥ 0.5). The gazetteer is seeded for the project's dense-CJK-signage test scenes but is designed to grow with whatever real signage future assets turn up — not a fixed answer key for one image. Menu also serves as the reference gazetteer for Savor's dakuten course.

## Evaluation and accuracy work

The `scripts/eval_detect.py` harness now reports detection **precision/recall/F1** and **normalized edit distance** when a ground-truth file is supplied:

```bash
# create a ground-truth file next to the image: my-image.gt.json
.venv\Scripts\python scripts\eval_detect.py images/my-image.png --tag baseline --ground-truth images/my-image.gt.json

# compare PaddleOCR
.venv\Scripts\python scripts\eval_detect.py images/my-image.png --tag paddle --engine paddleocr --ground-truth images/my-image.gt.json
```

Ground-truth JSON format:

```json
{
  "regions": [
    {"bbox": [x, y, width, height], "text": "correct text"}
  ]
}
```

## Manual bounding-box snap

When a user draws a rough box on the capture screen, the UI now calls `POST /api/detect/refine` with the crop and snaps the box to the highest-confidence detected text polygon. The endpoint is also usable directly:

```bash
curl -X POST "http://localhost:8000/api/detect/refine" \
  -H "Content-Type: application/json" \
  -d '{"asset_id": "...", "bbox": {"x": 100, "y": 200, "width": 300, "height": 80}}'
```

**Frontend** (proxies `/api` to port 8000):

```
cd frontend
npm install
npm run dev
```

Open http://localhost:5173.

---

<div align="center">

<img src="images/miis-logo-rev.svg" width="35%" />

Made with ❤️
</div>
