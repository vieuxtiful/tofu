## 🍢 eval_paddle — PaddleOCR 3.x standalone evaluation
## vieuxtiful
"""
Runs PaddleOCR (PP-OCRv5, paddleocr>=3.x predict API) on an image and
scores it against the same .gt.json ground truth eval_detect uses —
WITHOUT importing the tofu package, so it runs under the isolated
paddle venv (.venv-paddle) and never touches the app venv's pinned
numpy/opencv.

The point of this evaluation (CP-1): PP-OCRv5's DB detector + angle
classifier natively handles rotated/vertical CJK text that CRAFT
fragments — measure whether that materially beats the EasyOCR +
scene-surface-probe path on the street scenes before committing to a
paddleocr 3.x adapter rewrite in cicerone.

usage (from repo root):
  .venv-paddle/Scripts/python scripts/eval_paddle.py [image] [--tag label] [--lang korean|japan|ch|en]
outputs to scripts/eval_out/{stem}-{tag}.paddle.json
"""

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def _iou(a, b) -> float:
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    ix, iy = max(ax, bx), max(ay, by)
    ax2, ay2 = min(ax + aw, bx + bw), min(ay + ah, by + bh)
    if ax2 <= ix or ay2 <= iy:
        return 0.0
    inter = (ax2 - ix) * (ay2 - iy)
    union = aw * ah + bw * bh - inter
    return inter / union if union > 0 else 0.0


def _ed(a: str, b: str) -> int:
    a, b = a or "", b or ""
    if a == b:
        return 0
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(cur[-1] + 1, prev[j] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def _norm_ed(a: str, b: str) -> float:
    return _ed(a, b) / max(len(a or ""), len(b or ""), 1)


def run_paddle(image_path: Path, lang: str):
    """PP-OCRv5 via the 3.x predict API; returns [(bbox, text, conf)]."""
    import os
    # paddle 3.3 windows-cpu bug: oneDNN PIR instruction crashes with
    # "ConvertPirAttribute2RuntimeAttribute not support" — disable mkldnn
    os.environ.setdefault("FLAGS_use_mkldnn", "0")
    from paddleocr import PaddleOCR
    ocr = PaddleOCR(
        lang=lang,
        use_doc_orientation_classify=False,
        use_doc_unwarping=False,
        use_textline_orientation=True,  # vertical/rotated line handling
        enable_mkldnn=False,
    )
    t0 = time.time()
    results = ocr.predict(str(image_path))
    elapsed = time.time() - t0
    dets = []
    for res in results:
        data = res if isinstance(res, dict) else res.json.get("res", res.json)
        polys = data.get("rec_polys", data.get("dt_polys", []))
        texts = data.get("rec_texts", [])
        scores = data.get("rec_scores", [])
        for poly, text, score in zip(polys, texts, scores):
            xs = [int(p[0]) for p in poly]
            ys = [int(p[1]) for p in poly]
            dets.append((
                [min(xs), min(ys), max(xs) - min(xs), max(ys) - min(ys)],
                str(text), float(score),
            ))
    return dets, elapsed


def evaluate(dets, gt_path: Path):
    data = json.loads(gt_path.read_text(encoding="utf-8"))
    gt = data.get("regions", [])
    partial = bool(data.get("partial", False))
    matched = set()
    matches = []
    for bbox, text, conf in dets:
        best_iou, best_gi = 0.0, -1
        for gi, g in enumerate(gt):
            if gi in matched:
                continue
            iou = _iou(bbox, g["bbox"])
            if iou > best_iou and iou >= 0.5:
                best_iou, best_gi = iou, gi
        if best_gi >= 0:
            matched.add(best_gi)
            gt_text = gt[best_gi].get("text")
            matches.append({
                "iou": round(best_iou, 3), "pred_text": text,
                "gt_text": gt_text,
                "norm_ed": round(_norm_ed(text, gt_text), 3) if gt_text else None,
            })
    recall = len(matched) / len(gt) if gt else None
    scored = [m["norm_ed"] for m in matches if m["norm_ed"] is not None]
    return {
        "gt_regions": len(gt), "partial": partial,
        "matched": len(matched),
        "recall": round(recall, 3) if recall is not None else None,
        "precision": (
            None if partial
            else round(len(matched) / len(dets), 3) if dets else 0.0
        ),
        "mean_norm_ed": round(sum(scored) / len(scored), 3) if scored else None,
        "matches": matches,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("image")
    ap.add_argument("--tag", default="paddle")
    ap.add_argument("--lang", default="korean",
                    help="paddleocr lang code: korean | japan | ch | en")
    args = ap.parse_args()

    image_path = Path(args.image)
    out_dir = ROOT / "scripts" / "eval_out"
    out_dir.mkdir(parents=True, exist_ok=True)

    dets, elapsed = run_paddle(image_path, args.lang)

    gt_path = image_path.parent / f"{image_path.stem}.gt.json"
    metrics = evaluate(dets, gt_path) if gt_path.exists() else None

    report = {
        "image": str(image_path), "tag": args.tag, "engine": f"paddleocr/{args.lang}",
        "detections": len(dets),
        "timing_s": round(elapsed, 1),
        "metrics": metrics,
        "regions": [
            {"bbox": b, "text": t, "conf": round(c, 3)} for b, t, c in dets
        ],
    }
    out = out_dir / f"{image_path.stem}-{args.tag}.paddle.json"
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"== {image_path.stem} [{args.tag}] paddleocr/{args.lang} ==")
    print(f"detections: {len(dets)}  in {elapsed:.1f}s")
    if metrics:
        print(f"recall: {metrics['recall']}  (matched {metrics['matched']}/{metrics['gt_regions']})")
        print(f"mean norm edit distance: {metrics['mean_norm_ed']}")
    print(f"report: {out}")


if __name__ == "__main__":
    main()
