## 🍢 eval_overinclusion — WHERE a box became too big, not just that it is
## vieuxtiful
"""
Attribution for the over-inclusion defect, run before any tuning.

The measurement that started this: across the three hard fixtures, the
near-miss regions are not clipped, they are BLOATED. Median detected-area
over ground-truth-area is 1.75 for CRAFT, with 11 of 17 oversized. That
reverses the obvious remedy -- expanding boundaries (PaddleOCR's
unclip_ratio, a looser merge gate) makes the dominant failure worse, not
better -- and it is why the DB benchmark's premise did not survive contact.

But "the box is too big" does not say WHO made it too big, and the three
possible answers need completely different work:

  raw CRAFT already bloated      One erroneous link in the binary component
                                 makes minAreaRect enclose a neighbour
                                 before any ToFU code runs. No merge-gate
                                 tuning can reach this; it needs a different
                                 construction from the heatmaps.
  raw reasonable, pipeline grew  A specific later merge/prune step did it,
                                 and that step can be tightened or replaced.
  raw fragmented, merge overshot The pieces were right and the join was
                                 wrong -- a split-versus-merge decision
                                 problem, not a threshold problem.

CRAFT builds a word box by thresholding the region and affinity maps,
labelling connected components, and fitting minAreaRect to each. That last
step is lossy in exactly the direction observed: a component that wanders
into an adjacent sign produces a rectangle covering both, however tight the
heatmap itself was.

WHAT IS RECORDED per ground-truth region:

  raw_rect / raw_area_ratio    the best minAreaRect CRAFT produced at this
                               location, before ToFU touched it, and how
                               oversized it already was
  raw_components               how many raw proposals overlap this GT box --
                               more than one means CRAFT fragmented it and
                               anything downstream had a join to get wrong
  final_area_ratio             the same ratio after the full pipeline
  growth                       final ratio over raw ratio: above 1, ToFU
                               made it worse; at 1 it inherited the problem
  peak_region / peak_affinity  the heatmap evidence, to separate a genuine
                               link from a threshold artefact

NOT recorded, and stated rather than faked: merge ancestry. InstText carries
recognition_history but merge parents are not presently threaded onto the
survivor, so "which stage joined these" is inferred from the geometry here
rather than read. That is the next instrumentation gap if the second class
turns out to be the populous one.

usage (from repo root):
  .venv/Scripts/python scripts/eval_overinclusion.py
  .venv/Scripts/python scripts/eval_overinclusion.py --tag shipped
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

## Area ratios. A box within +-30% of ground truth is doing its job -- human
## annotation is not tighter than that -- so the bands only fire outside it.
BLOATED = 1.30
CLIPPED = 0.70

## How much of the GT a raw proposal must touch before it counts as one of
## the components covering it. Low on purpose: a fragment covering a fifth of
## a column is still evidence CRAFT split that column.
COMPONENT_TOUCH = 0.05

FIXTURES = (
    ("gemini-street", "ko"),
    ("japan-street", "ja"),
    ("la-bastille-1789", "fr"),
    ("china-street", "zh-cn"),
    ("russian-billboard", "ru"),
)


def _iou(a, b) -> float:
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    ix, iy = max(ax, bx), max(ay, by)
    x2, y2 = min(ax + aw, bx + bw), min(ay + ah, by + bh)
    if x2 <= ix or y2 <= iy:
        return 0.0
    inter = (x2 - ix) * (y2 - iy)
    return inter / (aw * ah + bw * bh - inter)


def _covered(gt, box) -> float:
    """Fraction of the GT box covered by `box` -- not IoU.

    A raw fragment can cover a slice of a tall column and score near-zero
    IoU because the fragment is tiny; coverage sees it, IoU does not, and
    the question here is how many pieces CRAFT broke the region into.
    """
    gx, gy, gw, gh = gt
    bx, by, bw, bh = box
    ix, iy = max(gx, bx), max(gy, by)
    x2, y2 = min(gx + gw, bx + bw), min(gy + gh, by + bh)
    if x2 <= ix or y2 <= iy:
        return 0.0
    return ((x2 - ix) * (y2 - iy)) / max(1.0, gw * gh)


def classify(raw_ratio, final_ratio, components, raw_iou) -> str:
    """Which of the three remedies this region needs."""
    if raw_ratio is None:
        return "no-raw-proposal"
    if components > 1 and final_ratio is not None and final_ratio > BLOATED:
        # CRAFT broke it up and something downstream glued the pieces
        # together too generously
        return "fragmented-then-overmerged"
    if raw_ratio > BLOATED:
        # already wrong before ToFU saw it
        return "raw-craft-bloated"
    if final_ratio is not None and final_ratio > BLOATED:
        return "pipeline-bloated"
    if raw_ratio < CLIPPED:
        return "raw-craft-clipped"
    return "geometry-ok-elsewhere"


def report(name: str, lang: str, tag: str):
    from eval_detector_evidence import patch, raw_proposals, score_maps

    from tofu.layers.cicerone import expand_langset

    image = next(
        (ROOT / "images" / f"{name}{ext}")
        for ext in (".png", ".jpeg", ".jpg")
        if (ROOT / "images" / f"{name}{ext}").exists()
    )
    gt = json.loads((ROOT / "images" / f"{name}.gt.json").read_text(encoding="utf-8"))["regions"]
    final_path = ROOT / "scripts" / "eval_out" / f"{name}-{tag}-{name}.json"
    if not final_path.exists():
        print(f"  no {tag} run for {name}; run eval_detect first", file=sys.stderr)
        return []
    final = json.loads(final_path.read_text(encoding="utf-8"))
    finals = []
    for r in final.get("regions", []):
        b = r.get("bbox")
        finals.append([b["x"], b["y"], b["width"], b["height"]] if isinstance(b, dict) else b)

    region_map, affinity_map, scale, _ = score_maps(image, expand_langset([lang]))
    raws = raw_proposals(region_map, affinity_map, scale)

    rows = []
    for g in gt:
        b = g["bbox"]
        gt_area = max(1.0, b[2] * b[3])

        raw_best = max(raws, key=lambda p: _iou(b, p), default=None)
        raw_iou = _iou(b, raw_best) if raw_best else 0.0
        raw_ratio = (raw_best[2] * raw_best[3] / gt_area) if raw_best and raw_iou > 0.05 else None
        components = sum(1 for p in raws if _covered(b, p) >= COMPONENT_TOUCH)

        fin_best = max(finals, key=lambda p: _iou(b, p), default=None)
        fin_iou = _iou(b, fin_best) if fin_best else 0.0
        fin_ratio = (fin_best[2] * fin_best[3] / gt_area) if fin_best and fin_iou > 0.05 else None

        pr = patch(region_map, b, scale)
        pa = patch(affinity_map, b, scale)
        rows.append({
            "fixture": name, "text": g.get("text"), "gt_bbox": b,
            "raw_iou": round(raw_iou, 3),
            "raw_area_ratio": round(raw_ratio, 2) if raw_ratio else None,
            "raw_components": components,
            "final_iou": round(fin_iou, 3),
            "final_area_ratio": round(fin_ratio, 2) if fin_ratio else None,
            "growth": (round(fin_ratio / raw_ratio, 2)
                       if raw_ratio and fin_ratio else None),
            "peak_region": round(float(pr.max()), 3) if pr is not None and pr.size else 0.0,
            "peak_affinity": round(float(pa.max()), 3) if pa is not None and pa.size else 0.0,
            "verdict": classify(raw_ratio, fin_ratio, components, raw_iou),
        })
    return rows


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tag", default="shipped", help="eval_detect tag holding the final geometry")
    ap.add_argument("--all-regions", action="store_true",
                    help="report matched regions too, not only the near-misses")
    args = ap.parse_args()

    every = []
    for name, lang in FIXTURES:
        rows = report(name, lang, args.tag)
        if not rows:
            continue
        every.extend(rows)
        shown = rows if args.all_regions else [
            r for r in rows if 0.1 <= r["final_iou"] < 0.5
        ]
        if not shown:
            continue
        print(f"\n### {name}")
        print(f"    {'text':<14}{'rawIoU':>7}{'rawA':>6}{'comp':>5}"
              f"{'finIoU':>8}{'finA':>6}{'grow':>6}{'pkR':>6}   verdict")
        for r in shown:
            label = str(r["text"])[:12] if r["text"] else "—"
            f = lambda v, w=6, p=2: (" " * (w - 1) + "-") if v is None else f"{v:>{w}.{p}f}"  # noqa: E731
            print(f"    {label:<14}{r['raw_iou']:>7.3f}{f(r['raw_area_ratio'])}"
                  f"{r['raw_components']:>5}{r['final_iou']:>8.3f}{f(r['final_area_ratio'])}"
                  f"{f(r['growth'])}{r['peak_region']:>6.2f}   {r['verdict']}")

    near = [r for r in every if 0.1 <= r["final_iou"] < 0.5]
    tally = {}
    for r in near:
        tally[r["verdict"]] = tally.get(r["verdict"], 0) + 1
    print(f"\n{'=' * 72}\nNEAR-MISS ATTRIBUTION  ({len(near)} regions)")
    for k, v in sorted(tally.items(), key=lambda kv: -kv[1]):
        print(f"    {v:>3}  {k}")
    grew = [r for r in near if r["growth"] and r["growth"] > 1.2]
    print(f"\n    of those, {len(grew)} grew >20% between raw CRAFT and final output")

    out = ROOT / "scripts" / "eval_out" / "overinclusion.json"
    out.write_text(json.dumps(every, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
