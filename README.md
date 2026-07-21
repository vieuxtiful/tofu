<div style="text" align="center">
  <img src="middlebury-seal.png" width="8%">
  <img src="middlebury-institute-seal.png" width="7%" hspace="30" style="margin-left: 40 px;"> 
</div>

---

<div style="image" align="center">
  <img src="tofu-wht.svg" width="38%">
</div>

<div align="center">
ToFU (text-over-frame unification)
</div>

---

<div align="center">

<img src="miis-logo-rev.svg" width="35%" />

Made with ❤️
</div>

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

## Scene detection

The scene pre-pass (`src/tofu/layers/scene.py`) runs before text detection and provides candidate text-bearing surfaces to Cicerone. The default `ClassicalCVBackend` now combines two complementary detectors:

- **Contour analysis** (Canny edges + `approxPolyDP`): finds structural surfaces — panels, bordered regions, signs.
- **MSER** (Maximally Stable Extremal Regions): finds text-like blobs — individual characters and short strings that contour analysis misses. Blobs are spatially clustered into bounding boxes labeled `"text_cluster"`.

Scene regions feed into Cicerone's detection pipeline in three ways:

1. **Adaptive confidence filtering**: detections inside `"panel"` and `"text_cluster"` regions survive at confidence ≥ 0.30 (text is very likely there); `"bordered_region"` regions use 0.40; everything else uses 0.50.
2. **Zoom detection priority**: `"text_cluster"` and `"panel"` regions are zoomed into first at 2× resolution, recovering small/dense text that full-frame detection misses.
3. **Style enrichment**: GrabCut-refined color estimation separates text from background pixels more accurately than Otsu-only thresholding, especially on textured or gradient surfaces.

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
