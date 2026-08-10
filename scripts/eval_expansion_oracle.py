## 🍢 eval_expansion_oracle — is there anything to gain by growing a box?
## vieuxtiful
"""
Oracle ceiling for the under-coverage cohort, measured before any selector
is built.

THE COHORT. Two error groups turned out to be one. gemini-street's CJK
columns come out of CRAFT at 0.34-0.55 of their ground-truth area and the
pipeline preserves that faithfully. japan-street's and gemini-street's
"truncations" -- 劇場通 for 劇場通り, 東南 for 東南荘, 주 for 맥주 -- were filed as
recognition failures until the geometry said otherwise:

    region    GT bbox              detected bbox        missing ink
    劇場通り    [53,335,20,61]       [53,334,20,49]       BELOW
    맥주       [1168,5,77,117]      [1175,9,48,109]      BELOW

The り is twelve pixels below where the crop ends. The recognizer read what
it was given, correctly. These are extent failures wearing a recognition
costume, and they belong to the same cohort as the columns.

WHAT THIS MEASURES, and what it deliberately does not. It asks one question:
if a perfect chooser picked the best box from a small family of vertical
expansions, how much would mean_best_iou and matched-GT move? That is an
ORACLE -- it uses ground truth to select, which no runtime policy can do.

Its only purpose is to decide whether expansion carries information worth
building a selector for. If the oracle barely moves, no selector can help
and the cohort needs a different remedy. Measuring this first is the same
gate that collapsed the language-model track from "a whole corrector" to
"one year on one plaque".

TWO CEILINGS, and the gap between them is the interesting number:

  geometric     every expansion allowed. The absolute upper bound: what a
                perfect chooser could reach with no evidence at all.
  score-guided  only expansions whose added band carries CRAFT region-map
                mass above a floor. This is what a real policy could see,
                since the heatmap is available at runtime and the ground
                truth is not.

A geometric ceiling far above the score-guided one means the detector's own
belief does not support the boxes that would help -- expansion would be
guessing, and the cohort needs stronger evidence than the score map offers.

RESULT -- MEASURED NO-GO. Run against the shipped baseline, grid above,
BAND_FLOOR 0.05:

    ceiling                mean IoU   matched
    current (shipped)         0.512     19/33
    geometric oracle          0.569     24/33
    score-guided oracle       0.544     19/33

A perfect chooser with no evidence gains five regions. A perfect chooser
restricted to what the heatmap supports gains NONE. Every point of the
score-guided IoU improvement is regions moving within their bucket, not
across the 0.5 line.

The five geometrically-recoverable regions are all gemini-street columns:

    居酒屋   now 0.335   geometric 0.543 ('both', 0.45)   guided 0.335
    ラーメン  now 0.283   geometric 0.581 ('both', 0.80)   guided 0.340
    고깃집   now 0.423   geometric 0.507 ('both', 0.10)   guided 0.457
    대박식당  now 0.478   geometric 0.520 ('up',   0.10)   guided 0.478

For every one, CRAFT's region map carries no mass in the band that would
reach the correct extent -- at a floor of 0.05, a quarter of the proposal
threshold. The detector does not merely decline to call that ink a
character; it does not believe there is ink there. **That is a detector
belief failure, not a post-processing or selector failure**, and no
selector can create evidence the model did not produce.

Where expansion DOES work, the two ceilings nearly coincide -- the heatmap
supports exactly the boxes that help:

    劇場通り      0.774 -> 0.958   ('down', 0.30)
    华联店3F      0.507 -> 0.865   ('down', 0.80)
    歌舞伎町一番街  0.865 -> 0.952   ('down', 0.10)
    上海明牌       0.511 -> 0.600   ('up',   0.20)
    茂昌眼镜公司    0.805 -> 0.885   ('up',   0.10)

But every one of those is ALREADY matched. Score-guided matched is 4/4 on
japan-street and 7/7 on china-street, unchanged. Expansion makes correct
boxes tighter; it does not convert failures.

NON-GOALS established by this measurement:
  * no global expansion policy;
  * no Family B runtime selector -- its recall premise is falsified;
  * no claim that tighter crops improve NED. 劇場通り at 0.958 would hand
    the recognizer the り that Step 0 found twelve pixels below the crop,
    and that is a HYPOTHESIS until recognition is re-run on the alternate
    crop and clean-crop NED compared.

A future reader seeing "+5 geometric-oracle matches" should read the
score-guided column before restarting this project.

usage (from repo root):
  .venv/Scripts/python scripts/eval_expansion_oracle.py
  .venv/Scripts/python scripts/eval_expansion_oracle.py --tag shipped
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

## Fractions of the detected box's own height to try, in each direction.
## Expressed relative to the box rather than in pixels because a column of
## four characters and a column of ten need different absolute growth for
## the same "one more character" -- and the thing being recovered is
## usually exactly one character.
STEPS = (0.10, 0.20, 0.30, 0.45, 0.60, 0.80, 1.00)

## Directions an expansion may grow. Vertical only: the cohort's missing ink
## is below (and occasionally above) the crop, and horizontal growth on a
## column reaches into the neighbouring column rather than into more of this
## one.
DIRECTIONS = ("down", "up", "both")

## Mean CRAFT region-map score the ADDED band must carry before a
## score-guided expansion is allowed. Well under the 0.2 proposal floor:
## the question is not "is this a character" -- the detector already
## declined to call it one -- but "is there ink here at all".
BAND_FLOOR = 0.05

FIXTURES = (("gemini-street", "ko"), ("japan-street", "ja"), ("china-street", "zh-cn"))


def _iou(a, b) -> float:
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    ix, iy = max(ax, bx), max(ay, by)
    x2, y2 = min(ax + aw, bx + bw), min(ay + ah, by + bh)
    if x2 <= ix or y2 <= iy:
        return 0.0
    inter = (x2 - ix) * (y2 - iy)
    return inter / (aw * ah + bw * bh - inter)


def expansions(box, image_h):
    """The small candidate family for one box: itself plus vertical growth."""
    x, y, w, h = box
    out = [("none", 0.0, [x, y, w, h])]
    for step in STEPS:
        grow = max(1.0, h * step)
        for direction in DIRECTIONS:
            if direction == "down":
                ny, nh = y, h + grow
            elif direction == "up":
                ny, nh = max(0.0, y - grow), h + min(y, grow)
            else:
                ny, nh = max(0.0, y - grow), h + min(y, grow) + grow
            nh = min(nh, image_h - ny)
            if nh <= h:
                continue
            out.append((direction, step, [x, ny, w, nh]))
    return out


def band_score(region_map, box, base, scale) -> float:
    """Mean CRAFT region-map value in the strip an expansion ADDS."""
    x, y, w, h = box
    bx, by, bw, bh = base
    # the added band is whatever of `box` lies outside `base` vertically
    bands = []
    if y < by:
        bands.append((y, by))
    if y + h > by + bh:
        bands.append((by + bh, y + h))
    values = []
    for y0, y1 in bands:
        r0 = int(max(0, y0 * scale))
        r1 = int(min(region_map.shape[0], y1 * scale))
        c0 = int(max(0, x * scale))
        c1 = int(min(region_map.shape[1], (x + w) * scale))
        if r1 > r0 and c1 > c0:
            values.append(region_map[r0:r1, c0:c1])
    if not values:
        return 0.0
    return float(np.concatenate([v.ravel() for v in values]).mean())


def run(name: str, lang: str, tag: str):
    from eval_detector_evidence import score_maps

    from tofu.layers.cicerone import expand_langset

    image = next(
        (ROOT / "images" / f"{name}{ext}")
        for ext in (".png", ".jpeg", ".jpg")
        if (ROOT / "images" / f"{name}{ext}").exists()
    )
    gt = json.loads((ROOT / "images" / f"{name}.gt.json").read_text(encoding="utf-8"))["regions"]
    report = json.loads(
        (ROOT / "scripts" / "eval_out" / f"{name}-{tag}-{name}.json").read_text(encoding="utf-8")
    )
    det = []
    for r in report.get("regions", []):
        b = r.get("bbox")
        det.append([b["x"], b["y"], b["width"], b["height"]] if isinstance(b, dict) else b)

    region_map, _affinity, scale, _ = score_maps(image, expand_langset([lang]))
    image_h = region_map.shape[0] / scale

    rows = []
    for g in gt:
        gb = g["bbox"]
        base = max(det, key=lambda d: _iou(gb, d), default=None)
        if base is None:
            continue
        current = _iou(gb, base)
        best_geo, best_guided = current, current
        pick = None
        for direction, step, cand in expansions(base, image_h):
            score = _iou(gb, cand)
            if score > best_geo:
                best_geo = score
                pick = (direction, step)
            if score > best_guided and band_score(region_map, cand, base, scale) >= BAND_FLOOR:
                best_guided = score
        rows.append({
            "text": g.get("text"), "current": round(current, 3),
            "oracle_geometric": round(best_geo, 3),
            "oracle_guided": round(best_guided, 3),
            "pick": pick,
            "undercovered": (base[2] * base[3]) < (gb[2] * gb[3]),
        })
    return rows


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tag", default="shipped")
    args = ap.parse_args()

    everything = []
    for name, lang in FIXTURES:
        rows = run(name, lang, args.tag)
        everything.extend(rows)
        cur = sum(r["current"] for r in rows) / len(rows)
        geo = sum(r["oracle_geometric"] for r in rows) / len(rows)
        gui = sum(r["oracle_guided"] for r in rows) / len(rows)
        print(f"\n### {name}   ({len(rows)} GT regions)")
        print(f"    {'text':<14}{'now':>7}{'geo':>7}{'guided':>8}   gain   pick")
        for r in rows:
            gain = r["oracle_guided"] - r["current"]
            flag = f"  +{gain:.3f}" if gain > 0.001 else ""
            if gain > 0.001 or r["current"] < 0.5:
                print(f"    {str(r['text'])[:12] if r['text'] else '—':<14}"
                      f"{r['current']:>7.3f}{r['oracle_geometric']:>7.3f}"
                      f"{r['oracle_guided']:>8.3f}{flag:>9}   {r['pick'] or ''}")
        print(f"    mean IoU  now {cur:.3f}   geometric {geo:.3f}   score-guided {gui:.3f}")
        print(f"    matched   now {sum(1 for r in rows if r['current']>=0.5)}"
              f"   geometric {sum(1 for r in rows if r['oracle_geometric']>=0.5)}"
              f"   score-guided {sum(1 for r in rows if r['oracle_guided']>=0.5)}")

    n = len(everything)
    print(f"\n{'='*64}\nCOHORT TOTAL  ({n} regions across {len(FIXTURES)} fixtures)")
    for label, key in (("now", "current"), ("geometric ceiling", "oracle_geometric"),
                       ("score-guided ceiling", "oracle_guided")):
        mean = sum(r[key] for r in everything) / n
        matched = sum(1 for r in everything if r[key] >= 0.5)
        print(f"    {label:<22} mean IoU {mean:.3f}   matched {matched}/{n}")
    out = ROOT / "scripts" / "eval_out" / "expansion_oracle.json"
    out.write_text(json.dumps(everything, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
