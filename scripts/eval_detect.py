## 🍢 eval_detect — detection quality harness
## vieuxtiful
"""
Measures cicerone detection quality on a real image and writes:
  - an annotated overlay PNG (boxes + recognized text + conf + language)
  - a JSON report (region count, per-region details, garbage fraction,
    inferred source language, timing)

Run before/after any detection tuning; compare the reports.

usage (from repo root):
  .venv/Scripts/python scripts/eval_detect.py [image] [--tag label] [--no-probe] [--ground-truth path]
defaults to images/gemini-street.png; outputs to scripts/eval_out/.
"""

import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tofu.layers import cicerone, scene  # noqa: E402
from tofu.layers.cicerone import ScriptDetector  # noqa: E402


def infer_src_lang(manifest) -> str:
    """area-weighted dominant language (mirrors server logic)."""
    votes = Counter()
    for i in manifest.instances:
        if not i.detected_language:
            continue
        area = (
            i.bounding_box.width * i.bounding_box.height
            if i.bounding_box is not None else 1
        )
        votes[i.detected_language] += max(1, area)
    return max(votes, key=lambda k: votes[k]) if votes else "en"


def _iou(a, b) -> float:
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    ix = max(ax, bx)
    iy = max(ay, by)
    ax2 = min(ax + aw, bx + bw)
    ay2 = min(ay + ah, by + bh)
    if ax2 <= ix or ay2 <= iy:
        return 0.0
    inter = (ax2 - ix) * (ay2 - iy)
    a_area = aw * ah
    b_area = bw * bh
    return inter / (a_area + b_area - inter) if (a_area + b_area - inter) > 0 else 0.0


def _ed(s1: str, s2: str) -> int:
    a, b = s1 or "", s2 or ""
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cost = 0 if ca == cb else 1
            cur.append(min(cur[-1] + 1, prev[j] + 1, prev[j - 1] + cost))
        prev = cur
    return prev[-1]


def _norm_ed(s1: str, s2: str) -> float:
    denom = max(len(s1 or ""), len(s2 or ""), 1)
    return _ed(s1, s2) / denom


def evaluate(image_path: Path, manifest, gt_path: Path | None):
    """compute detection and transcription metrics against ground truth."""
    gt = []
    if gt_path is None:
        auto = image_path.parent / f"{image_path.stem}.gt.json"
        if auto.exists():
            gt_path = auto
    if gt_path and gt_path.exists():
        gt = json.loads(gt_path.read_text(encoding="utf-8")).get("regions", [])

    preds = [
        {
            "bbox": [
                i.bounding_box.x, i.bounding_box.y,
                i.bounding_box.width, i.bounding_box.height,
            ],
            "text": i.text or "",
            "conf": i.confidence or 0.0,
        }
        for i in manifest.instances if i.bounding_box is not None
    ]

    def _metrics_for_threshold(threshold: float):
        filtered = [p for p in preds if p["conf"] >= threshold]
        matched_gt_t = set()
        matched_pred_t = set()
        for pi, p in enumerate(filtered):
            best_iou = 0.0
            best_gi = -1
            for gi, g in enumerate(gt):
                if gi in matched_gt_t:
                    continue
                iou = _iou(p["bbox"], g["bbox"])
                if iou > best_iou and iou >= 0.5:
                    best_iou = iou
                    best_gi = gi
            if best_gi >= 0:
                matched_gt_t.add(best_gi)
                matched_pred_t.add(pi)
        tp = len(matched_gt_t)
        fp = len(filtered) - len(matched_pred_t)
        fn = len(gt) - len(matched_gt_t)
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
        return precision, recall, f1

    thresholds = [round(t, 2) for t in [0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]]
    threshold_scores = [
        (t, _metrics_for_threshold(t))
        for t in thresholds
    ]
    best_t, (best_p, best_r, best_f1) = max(threshold_scores, key=lambda x: x[1][2])

    matched_gt = set()
    matched_pred = set()
    matches = []
    for pi, p in enumerate(preds):
        best_iou = 0.0
        best_gi = -1
        for gi, g in enumerate(gt):
            if gi in matched_gt:
                continue
            iou = _iou(p["bbox"], g["bbox"])
            if iou > best_iou and iou >= 0.5:
                best_iou = iou
                best_gi = gi
        if best_gi >= 0:
            matched_gt.add(best_gi)
            matched_pred.add(pi)
            matches.append({
                "pred_idx": pi, "gt_idx": best_gi, "iou": round(best_iou, 3),
                "pred_text": p["text"], "gt_text": gt[best_gi].get("text", ""),
                "norm_ed": round(_norm_ed(p["text"], gt[best_gi].get("text", "")), 3),
            })

    tp = len(matched_gt)
    fp = len(preds) - len(matched_pred)
    fn = len(gt) - len(matched_gt)
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    mean_ed = sum(m["norm_ed"] for m in matches) / len(matches) if matches else 0.0

    return {
        "ground_truth_path": str(gt_path) if gt_path and gt_path.exists() else None,
        "gt_regions": len(gt),
        "tp": tp, "fp": fp, "fn": fn,
        "precision": round(precision, 3),
        "recall": round(recall, 3),
        "f1": round(f1, 3),
        "mean_norm_ed": round(mean_ed, 3),
        "matches": matches,
        "threshold_sweep": [
            {"threshold": t, "precision": round(p, 3), "recall": round(r, 3), "f1": round(f, 3)}
            for t, (p, r, f) in threshold_scores
        ],
        "best_threshold": round(best_t, 2),
        "best_threshold_f1": round(best_f1, 3),
    }


def annotate(image_path: Path, manifest, out_path: Path) -> None:
    from PIL import Image, ImageDraw, ImageFont
    img = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("malgun.ttf", 18)  # hangul+latin capable
    except Exception:
        font = ImageFont.load_default()
    for inst in manifest.instances:
        b = inst.bounding_box
        conf = inst.confidence or 0.0
        color = (34, 197, 94) if conf >= 0.6 else (245, 158, 11) if conf >= 0.3 else (239, 68, 68)
        draw.rectangle([b.x, b.y, b.x + b.width, b.y + b.height], outline=color, width=3)
        label = f"{inst.id} {inst.detected_language or '?'} {conf:.2f} {inst.text or ''}"[:48]
        ty = max(0, b.y - 22)
        tw = draw.textlength(label, font=font)
        draw.rectangle([b.x, ty, b.x + tw + 6, ty + 22], fill=(0, 0, 0))
        draw.text((b.x + 3, ty + 2), label, fill=color, font=font)
    img.save(out_path)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("image", nargs="?", default=str(ROOT / "images" / "gemini-street.png"))
    ap.add_argument("--tag", default="run", help="label for output filenames")
    ap.add_argument("--no-probe", action="store_true", help="disable language auto-probe")
    ap.add_argument("--no-scene", action="store_true", help="skip scene pre-pass")
    ap.add_argument("--ground-truth", default=None, help="path to ground-truth JSON (default: image.gt.json)")
    ap.add_argument("--engine", default="easyocr", choices=("easyocr", "paddleocr"), help="OCR backend to use")
    args = ap.parse_args()

    import os
    os.environ["OCR_ENGINE"] = args.engine

    image_path = Path(args.image)
    out_dir = ROOT / "scripts" / "eval_out"
    out_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    scene_regions = [] if args.no_scene else scene.analyze_regions(str(image_path))
    t_scene = time.time() - t0

    detect_kwargs = {}
    if args.no_probe:
        detect_kwargs["identify_languages"] = False
    t1 = time.time()
    manifest = cicerone.detect(
        str(image_path), scene_regions=scene_regions, **detect_kwargs
    )
    t_detect = time.time() - t1

    # cache the manifest so eval_render.py can reuse it without re-detecting
    from tofu.utils.manifest_store import _manifest_to_dict
    (out_dir / f"{image_path.stem}.manifest.json").write_text(
        json.dumps(_manifest_to_dict(manifest), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    sd = ScriptDetector()
    regions = []
    garbage = 0
    lang_counts: Counter = Counter()
    for i in manifest.instances:
        script = sd.detect_script(i.text or "")
        if script is None:
            garbage += 1
        lang_counts[i.detected_language or "?"] += 1
        b = i.bounding_box
        regions.append({
            "id": i.id, "text": i.text, "conf": round(i.confidence or 0, 3),
            "lang": i.detected_language, "script": script,
            "bbox": [b.x, b.y, b.width, b.height],
        })

    gt_path = Path(args.ground_truth) if args.ground_truth else None
    metrics = evaluate(image_path, manifest, gt_path)

    report = {
        "image": str(image_path),
        "tag": args.tag,
        "region_count": manifest.total_regions,
        "scene_surfaces": len(scene_regions),
        "garbage_fraction": round(garbage / max(1, manifest.total_regions), 3),
        "lang_distribution": dict(lang_counts),
        "inferred_src_lang": infer_src_lang(manifest),
        "timing_s": {"scene": round(t_scene, 1), "detect": round(t_detect, 1)},
        "metrics": metrics,
        "regions": regions,
    }

    stem = image_path.stem
    report_path = out_dir / f"{stem}-{args.tag}.json"
    overlay_path = out_dir / f"{stem}-{args.tag}.png"
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    annotate(image_path, manifest, overlay_path)

    print(f"== {stem} [{args.tag}] ==")
    print(f"regions: {report['region_count']}  |  surfaces: {report['scene_surfaces']}")
    print(f"garbage fraction: {report['garbage_fraction']}")
    print(f"lang distribution: {report['lang_distribution']}")
    print(f"inferred src_lang: {report['inferred_src_lang']}")
    print(f"precision: {metrics['precision']}  recall: {metrics['recall']}  f1: {metrics['f1']}")
    print(f"mean norm edit distance: {metrics['mean_norm_ed']}")
    print(f"best threshold (F1): {metrics.get('best_threshold')}  f1: {metrics.get('best_threshold_f1')}")
    print(f"timing: scene {report['timing_s']['scene']}s, detect {report['timing_s']['detect']}s")
    print(f"report:  {report_path}")
    print(f"overlay: {overlay_path}")


if __name__ == "__main__":
    main()
