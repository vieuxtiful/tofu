## 🍢 eval_detector_evidence — why a region was missed, not just that it was
## vieuxtiful
"""
Failure ATTRIBUTION for text detection, as opposed to failure measurement.

eval_detect scores output geometry against ground truth and sorts every GT
region into matched / near-miss / undetected. That partition is real and it
separated two failure systems that recall alone had merged -- gemini-street
loses regions it has already located (9 near-misses), japan-street loses
regions it never proposed at all (0 near-misses, 3 undetected). But it
cannot say WHY either happened, and the two candidate remedies cost very
different amounts:

  * no detector evidence at this scale  -> tiles, a different detector, or
                                           domain fine-tuning
  * evidence present, below proposal    -> score calibration or a targeted
                                           high-resolution retry
  * evidence present, wrongly grouped   -> polygonization / split-merge
                                           rules, and no amount of
                                           resolution will help

Choosing between those from an outcome metric is guessing. This harness
reads the detector's own score maps at the GT's own coordinates, before any
threshold or component assembly, and answers the question directly.

WHAT IT MEASURES, per ground-truth region:

  peak_region     max CRAFT character-region score inside the GT box.
                  The single most informative field here: it distinguishes
                  "the detector is blind to this" from "the detector saw
                  this and the pipeline discarded it".
  peak_affinity   max CRAFT affinity score inside the GT box. Says whether
                  the characters were LINKABLE, separately from whether
                  they were FOUND -- link_threshold being inert in the
                  pipeline does not prove the affinity map is healthy.
  raw_best_iou    best IoU from getDetBoxes run at the loosest thresholds
                  the pipeline ever uses, before ToFU's merging, scene
                  filtering, pruning and column assembly. Separates
                  candidate-generation failure from later rejection.
  raw_overlaps    how many raw proposals touch the GT box at all. One is a
                  clean candidate; several is fragmentation; zero with a
                  high peak_region means the evidence was absorbed into a
                  neighbour.
  crop_conf       recognizer confidence on the GT crop itself, rectified
                  and upscaled the way the pipeline would. Text that cannot
                  be read from a perfect crop is not worth spending
                  detection work on, and it is worth knowing which GT
                  entries are in that state before optimizing against them.

The score maps are obtained by replicating easyocr.detection.test_net's
preprocessing exactly -- resize_aspect_ratio, normalizeMeanVariance, the
same net -- rather than approximating it, because a diagnostic that reads a
different tensor than the pipeline consumes is worse than no diagnostic.
CRAFT emits its maps at half the resized resolution, so a GT box maps onto
them by `image_coord * ratio / 2` (see craft_utils.adjustResultCoordinates,
which inverts exactly this with ratio_net=2).

usage (from repo root):
  .venv/Scripts/python scripts/eval_detector_evidence.py
  .venv/Scripts/python scripts/eval_detector_evidence.py images/japan-street.jpeg --lang ja
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

## CRAFT's character-region map is a probability surface. low_text is the
## threshold at which a pixel is admitted to a character component at all,
## and the pipeline's loosest rung (PASS_THRESHOLDS[-1]) sets it to 0.2 --
## so anything below that was never going to form a proposal however the
## rest of the pipeline is tuned.
##
## The background floor is deliberately well under it: CRAFT's map is not
## clean zero on textured surfaces, and calling 0.05 of stone grain
## "activation" would classify every miss as weak-but-present and point all
## the work at calibration.
NO_ACTIVATION = 0.10
PROPOSAL_FLOOR = 0.20

## The loosest thresholds the pipeline ever runs (PASS_THRESHOLDS[-1]),
## used for the raw-proposal pass so "before filtering" means before ToFU's
## stages rather than before CRAFT's own.
RAW_TEXT_THRESHOLD = 0.3
RAW_LINK_THRESHOLD = 0.2
RAW_LOW_TEXT = 0.2

## Dilation ring around the GT box, as a fraction of its own size. CRAFT's
## response to a glyph is not perfectly registered to a human-drawn box, and
## a peak one pixel outside a tight annotation is evidence, not absence.
GT_DILATE = 0.15


def _iou(a, b) -> float:
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    ix, iy = max(ax, bx), max(ay, by)
    x2, y2 = min(ax + aw, bx + bw), min(ay + ah, by + bh)
    if x2 <= ix or y2 <= iy:
        return 0.0
    inter = (x2 - ix) * (y2 - iy)
    return inter / (aw * ah + bw * bh - inter)


def score_maps(image_path: Path, languages, mag_ratio=1.0, canvas_size=2560):
    """CRAFT's raw region and affinity maps, plus the image->map scale.

    Replicates easyocr.detection.test_net rather than calling it, because
    test_net computes these and returns only the boxes derived from them.
    """
    import torch
    from easyocr import imgproc

    import easyocr

    reader = easyocr.Reader(list(languages), gpu=False)
    img = imgproc.loadImage(str(image_path))
    resized, ratio, _ = imgproc.resize_aspect_ratio(
        img, canvas_size, interpolation=__import__("cv2").INTER_LINEAR, mag_ratio=mag_ratio,
    )
    x = np.transpose(imgproc.normalizeMeanVariance(resized), (2, 0, 1))
    x = torch.from_numpy(np.array([x])).to("cpu")
    with torch.no_grad():
        y, _ = reader.detector(x)
    out = y[0]
    region = out[:, :, 0].cpu().data.numpy()
    affinity = out[:, :, 1].cpu().data.numpy()
    # image coord -> score-map coord; adjustResultCoordinates inverts this
    return region, affinity, ratio / 2.0, reader


def raw_proposals(region, affinity, scale):
    """getDetBoxes at the loosest thresholds the pipeline ever uses."""
    from easyocr.craft_utils import getDetBoxes

    boxes, _, _ = getDetBoxes(
        region, affinity, RAW_TEXT_THRESHOLD, RAW_LINK_THRESHOLD, RAW_LOW_TEXT, False, False,
    )
    out = []
    for b in boxes:
        pts = np.array(b) / scale  # back to image coordinates
        xs, ys = pts[:, 0], pts[:, 1]
        out.append([float(xs.min()), float(ys.min()),
                    float(xs.max() - xs.min()), float(ys.max() - ys.min())])
    return out


def patch(m, bbox, scale):
    """The score-map window a GT box projects onto, with its dilation ring."""
    x, y, w, h = bbox
    dx, dy = w * GT_DILATE, h * GT_DILATE
    x0 = int(max(0, (x - dx) * scale))
    y0 = int(max(0, (y - dy) * scale))
    x1 = int(min(m.shape[1], (x + w + dx) * scale))
    y1 = int(min(m.shape[0], (y + h + dy) * scale))
    if x1 <= x0 or y1 <= y0:
        return None
    return m[y0:y1, x0:x1]


def classify(peak_region, raw_best, raw_hits) -> str:
    """The decision this harness exists to enable."""
    if peak_region < NO_ACTIVATION:
        return "no-activation"          # tiles / a different detector / fine-tune
    if peak_region < PROPOSAL_FLOOR:
        return "weak-activation"        # calibration or targeted retry
    if raw_hits == 0:
        return "absorbed"               # evidence taken by a neighbour
    if raw_hits > 1:
        return "fragmented"             # grouping / polygonization
    if raw_best < 0.5:
        return "boundary"               # one candidate, wrong extent
    return "proposed-then-lost"         # good candidate, later stage dropped it


def crop_legibility(backend, image_path, bbox, gt_text):
    """Can this text be read at all, given a PERFECT box?

    The point of the field is to stop detection work being spent on targets
    that are not legible at the available image information. Two things had
    to change before it measured that.

    First, the read goes through EasyOCRBackend.detect_in_regions -- the
    pipeline's own crop path, with its upscale to MIN_CROP_HEIGHT -- and not
    a bare readtext on a padded slice. Measured on japan-street, the naive
    version returned nothing at all for 劇場通り and a mangled 闘露岐町一雷 for
    歌舞伎町一番街, a region the full pipeline transcribes exactly. A diagnostic
    that is weaker than the thing it is diagnosing reports the pipeline's
    strengths as the image's failures.

    Second, the score is edit distance to the ground truth, not recognizer
    confidence. EasyOCR's CJK confidences sit near zero even on correct
    reads -- 歌舞伎町一番街 above comes back at 0.0 while being most of the way
    right -- so confidence cannot separate "unreadable" from "read fine, CJK
    scores low". Distance to the known answer can.

    Returns (normalized_edit_distance, text) or (None, None).
    """
    try:
        from tofu.core.types import BBox

        x, y, w, h = [int(v) for v in bbox]
        found = backend.detect_in_regions(str(image_path), [BBox(x=x, y=y, width=w, height=h)])
        detections = found[0] if found else []
        if not detections:
            return 1.0, ""
        # the crop path returns fragments in reading order; the pipeline
        # joins them, so compare the joined text for the same reason
        text = "".join((d.text or "") for d in detections)
        if not gt_text:
            return None, text
        return round(_norm_ed(text, gt_text), 3), text
    except Exception:
        return None, None


def _norm_ed(a: str, b: str) -> float:
    """Levenshtein distance normalized by the longer string."""
    a, b = a or "", b or ""
    if not a and not b:
        return 0.0
    if not a or not b:
        return 1.0
    previous = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        current = [i]
        for j, cb in enumerate(b, 1):
            current.append(min(previous[j] + 1, current[j - 1] + 1,
                               previous[j - 1] + (ca != cb)))
        previous = current
    return previous[-1] / max(len(a), len(b))


def report(image_path: Path, languages, gt_path: Path, mag_ratio=1.0):
    from tofu.layers.cicerone import EasyOCRBackend

    gt = json.loads(gt_path.read_text(encoding="utf-8"))["regions"]
    region, affinity, scale, _reader = score_maps(image_path, languages, mag_ratio)
    proposals = raw_proposals(region, affinity, scale)
    background = float(np.median(region))
    backend = EasyOCRBackend(languages=tuple(languages), gpu=False)

    rows = []
    for g in gt:
        b = g["bbox"]
        pr, pa = patch(region, b, scale), patch(affinity, b, scale)
        peak_r = float(pr.max()) if pr is not None and pr.size else 0.0
        peak_a = float(pa.max()) if pa is not None and pa.size else 0.0
        overlaps = [p for p in proposals if _iou(b, p) > 0.05]
        best = max((_iou(b, p) for p in proposals), default=0.0)
        crop_ned, crop_text = crop_legibility(backend, image_path, b, g.get("text"))
        rows.append({
            "text": g.get("text"),
            "bbox": b,
            "height_px": b[3],
            "peak_region": round(peak_r, 3),
            "peak_affinity": round(peak_a, 3),
            "raw_best_iou": round(best, 3),
            "raw_overlaps": len(overlaps),
            "crop_ned": crop_ned,
            "crop_text": crop_text,
            "verdict": classify(peak_r, best, len(overlaps)),
        })
    return {
        "image": str(image_path),
        "languages": list(languages),
        "mag_ratio": mag_ratio,
        "score_map_background_median": round(background, 4),
        "raw_proposal_count": len(proposals),
        "regions": rows,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("image", nargs="?", default=None)
    ap.add_argument("--lang", default=None)
    ap.add_argument("--mag-ratio", type=float, default=1.0)
    ap.add_argument("--out", type=Path, default=ROOT / "scripts" / "eval_out")
    args = ap.parse_args()

    from tofu.layers.cicerone import expand_langset

    targets = (
        [(Path(args.image), args.lang or "en")] if args.image else
        [(ROOT / "images" / "gemini-street.png", "ko"),
         (ROOT / "images" / "japan-street.jpeg", "ja"),
         (ROOT / "images" / "la-bastille-1789.jpeg", "fr")]
    )

    args.out.mkdir(parents=True, exist_ok=True)
    for image_path, lang in targets:
        gt_path = image_path.with_suffix("").with_suffix(".gt.json")
        if not gt_path.exists():
            gt_path = image_path.parent / f"{image_path.stem}.gt.json"
        if not gt_path.exists():
            print(f"no ground truth for {image_path.name}", file=sys.stderr)
            continue
        data = report(image_path, expand_langset([lang]), gt_path, args.mag_ratio)

        print(f"\n### {image_path.name}  (lang={lang}, mag={args.mag_ratio})")
        print(f"    score-map background median {data['score_map_background_median']}, "
              f"{data['raw_proposal_count']} raw proposals")
        print(f"    {'text':<14}{'h_px':>6}{'peakR':>7}{'peakA':>7}{'rawIoU':>8}"
              f"{'n':>3}{'cropNED':>9}  verdict")
        for r in data["regions"]:
            label = (str(r["text"])[:12] if r["text"] else "—")
            print(f"    {label:<14}{r['height_px']:>6}{r['peak_region']:>7.3f}"
                  f"{r['peak_affinity']:>7.3f}{r['raw_best_iou']:>8.3f}"
                  f"{r['raw_overlaps']:>3}{str(r['crop_ned']):>9}  {r['verdict']}")

        tally = {}
        for r in data["regions"]:
            tally[r["verdict"]] = tally.get(r["verdict"], 0) + 1
        print(f"    -> {tally}")

        path = args.out / f"{image_path.stem}.evidence.json"
        path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
