## 🍢 paddle_worker — out-of-process PaddleOCR runner
## vieuxtiful
"""
Standalone worker invoked as a subprocess under the ISOLATED .venv-paddle
interpreter (never the app venv — paddlepaddle force-replaces numpy/opencv
on install, see CP-1 findings). No `tofu` package imports here; this file
must run with nothing beyond stdlib + paddleocr on sys.path.

Protocol: the caller writes a single JSON request to this process's
stdin, then reads the result from the `out_path` file the request
names — NOT from stdout, because PaddleOCR prints its own log/progress
lines (colored ANSI, model-cache notices) to stdout, which would corrupt
a stdout-as-payload channel. The worker always writes something to
out_path, success or failure, so the caller has one place to look
regardless of how this process exits.

Request (stdin, one JSON object):
  {
    "op": "detect" | "detect_regions",
    "image_path": "<absolute path to an existing image file>",
    "lang": "<paddleocr lang code, e.g. 'en' | 'korean' | 'japan' | 'ch'>",
    "out_path": "<absolute path this worker must write its JSON result to>",
    "regions": [[x, y, w, h], ...],   # detect_regions only
    "pad": 4                          # detect_regions only, default 4
  }

Result (written to out_path):
  {"ok": true, "detections": [{"polygon": [[x,y],...], "text": str, "confidence": float}, ...]}
    -- for op="detect"
  {"ok": true, "per_region": [[{"polygon":...,"text":...,"confidence":...}, ...], ...]}
    -- for op="detect_regions", one list per input region, full-image coords
  {"ok": false, "error": "<message>"}
    -- on any failure
"""

import json
import os
import sys

# paddle 3.3 windows-cpu bug: oneDNN PIR instruction crashes with
# "ConvertPirAttribute2RuntimeAttribute not support" — must be set before
# paddle/paddleocr import
os.environ.setdefault("FLAGS_use_mkldnn", "0")

_readers = {}  # lang -> PaddleOCR instance, reused across detect_regions calls


def _get_reader(lang: str):
    if lang not in _readers:
        from paddleocr import PaddleOCR
        _readers[lang] = PaddleOCR(
            lang=lang,
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
            use_textline_orientation=True,
            enable_mkldnn=False,
        )
    return _readers[lang]


def _run_predict(reader, image) -> list:
    """run PaddleOCR predict() on a path or ndarray; return
    [(polygon, text, confidence), ...] in the image's own coordinate space."""
    results = reader.predict(image)
    out = []
    for res in results:
        data = res if isinstance(res, dict) else res.json.get("res", res.json)
        polys = data.get("rec_polys", data.get("dt_polys", []))
        texts = data.get("rec_texts", [])
        scores = data.get("rec_scores", [])
        for poly, text, score in zip(polys, texts, scores):
            pts = [[float(p[0]), float(p[1])] for p in poly]
            out.append((pts, str(text), float(score)))
    return out


def op_detect(req: dict) -> dict:
    reader = _get_reader(req["lang"])
    dets = _run_predict(reader, req["image_path"])
    return {
        "ok": True,
        "detections": [
            {"polygon": poly, "text": text, "confidence": conf}
            for poly, text, conf in dets
        ],
    }


def op_detect_regions(req: dict) -> dict:
    import numpy as np
    from PIL import Image

    reader = _get_reader(req["lang"])
    img = np.asarray(Image.open(req["image_path"]).convert("RGB"))
    h, w = img.shape[:2]
    pad = int(req.get("pad", 4))
    per_region = []
    for x, y, rw, rh in req["regions"]:
        x0, y0 = max(0, x - pad), max(0, y - pad)
        x1, y1 = min(w, x + rw + pad), min(h, y + rh + pad)
        if x1 - x0 < 3 or y1 - y0 < 3:
            per_region.append([])
            continue
        crop = img[y0:y1, x0:x1]
        dets = _run_predict(reader, crop)
        per_region.append([
            {
                "polygon": [[px + x0, py + y0] for px, py in poly],
                "text": text,
                "confidence": conf,
            }
            for poly, text, conf in dets
        ])
    return {"ok": True, "per_region": per_region}


def main() -> None:
    raw = sys.stdin.read()
    out_path = None
    try:
        req = json.loads(raw)
        out_path = req.get("out_path")
        op = req.get("op")
        if op == "detect":
            result = op_detect(req)
        elif op == "detect_regions":
            result = op_detect_regions(req)
        else:
            result = {"ok": False, "error": f"unknown op '{op}'"}
    except Exception as exc:  # noqa: BLE001 — must always produce a result file
        result = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}

    if out_path:
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False)
    else:
        # no out_path to write to (malformed request) — last resort, stdout
        print(json.dumps(result, ensure_ascii=False))
    sys.exit(0 if result.get("ok") else 1)


if __name__ == "__main__":
    main()
