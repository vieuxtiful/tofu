## 🍢 eval_memory — Phase 6 exit criterion: duplicate-pair TM match precision
"""
plan's stated exit criterion: "re-processing a near-duplicate asset
pre-fills matching regions with approved translations at >=95% precision
on a constructed duplicate-pair fixture set."

methodology: for each fixture, detect once (asset A), store every
detected region as an approved TM record with a globally-unique
synthetic target_text. perturb the SAME fixture (resize + JPEG
recompression, simulating a re-photograph) into asset B and detect it
independently -- OCR reads B's text slightly differently, exercising the
exact/fuzzy/visual tiers for real instead of on hand-built dicts.

ground truth for "is this match correct" is established GEOMETRICALLY
(bbox IoU after undoing the known resize), not by re-deriving it from
text similarity -- that would be circular with the matcher under test.
"""
import io
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from PIL import Image

from tofu.core.types import AssetInfo, AssetType, BBox, QAReport, TextManifest
from tofu.layers import cicerone, memory

ROOT = Path(__file__).resolve().parent.parent
FIXTURES = [
    ROOT / "tests/fixtures/flat-sign.png",
    ROOT / "tests/fixtures/gradient-banner.png",
    ROOT / "tests/fixtures/textured-wall.png",
    ROOT / "tests/fixtures/stylized-italic.png",
    ROOT / "tests/fixtures/expansion-en.png",
    ROOT / "tests/fixtures/cjk-vertical.png",
]
OUT_DIR = ROOT / "scripts/eval_out"
SCALE = 0.85       # near-duplicate: re-shot at a different distance/crop
JPEG_Q = 80        # + recompression artifacts
IOU_MATCH = 0.3    # bbox correspondence threshold for ground truth


def detect(path) -> TextManifest:
    info = AssetInfo(asset_type=AssetType.IMAGE, frame_count=1, source=str(path))
    return cicerone.detect(str(path), info)


def iou(a: BBox, b: BBox) -> float:
    ax0, ay0, ax1, ay1 = a.x, a.y, a.x + a.width, a.y + a.height
    bx0, by0, bx1, by1 = b.x, b.y, b.x + b.width, b.y + b.height
    ix0, iy0 = max(ax0, bx0), max(ay0, by0)
    ix1, iy1 = min(ax1, bx1), min(ay1, by1)
    if ix1 <= ix0 or iy1 <= iy0:
        return 0.0
    inter = (ix1 - ix0) * (iy1 - iy0)
    union = a.width * a.height + b.width * b.height - inter
    return inter / union if union > 0 else 0.0


def main():
    t_start = time.time()
    candidates = []
    gt_by_fixture = {}

    print("== building TM from asset A (original fixtures) ==")
    for fx in FIXTURES:
        manifest_a = detect(fx)
        fixture_gt = []
        instances_with_text = [i for i in manifest_a.instances if i.text]
        for inst in instances_with_text:
            inst.target_text = f"T::{fx.stem}::{inst.id}"
            fixture_gt.append((inst.bounding_box, inst.target_text))
        if instances_with_text:
            qa = QAReport(
                overall_score=1.0,
                per_asset_instance_score={manifest_a.asset_id: {i.id: 1.0 for i in instances_with_text}},
            )
            drafts = memory.update(manifest_a, "es", None, str(fx), qa, qa_threshold=0.5)
            candidates.extend(drafts)
        gt_by_fixture[fx.stem] = fixture_gt
        print(f"  {fx.stem}: {len(instances_with_text)} region(s) -> {len(fixture_gt)} TM record(s)")

    print(f"total candidate pool: {len(candidates)} records\n")
    print("== asset B: perturbed near-duplicates (resize {}x + JPEG q{}) ==".format(SCALE, JPEG_Q))

    total_detected = 0
    total_with_correspondence = 0
    predicted = 0
    correct = 0
    per_fixture_rows = []

    for fx in FIXTURES:
        img = Image.open(fx).convert("RGB")
        w, h = img.size
        resized = img.resize((max(1, int(w * SCALE)), max(1, int(h * SCALE))))
        buf = io.BytesIO()
        resized.save(buf, format="JPEG", quality=JPEG_Q)
        buf.seek(0)
        perturbed = Image.open(buf).convert("RGB")
        tmp_path = OUT_DIR / f"_memeval_{fx.stem}.jpg"
        perturbed.save(tmp_path)

        manifest_b = detect(tmp_path)
        fixture_gt = gt_by_fixture[fx.stem]
        fx_detected = fx_corresponded = fx_predicted = fx_correct = 0

        for inst in manifest_b.instances:
            if not inst.text:
                continue
            total_detected += 1
            fx_detected += 1
            b = inst.bounding_box
            orig_box = BBox(
                x=int(b.x / SCALE), y=int(b.y / SCALE),
                width=int(b.width / SCALE), height=int(b.height / SCALE),
            )
            best_iou, best_target = 0.0, None
            for gt_box, gt_target in fixture_gt:
                score = iou(orig_box, gt_box)
                if score > best_iou:
                    best_iou, best_target = score, gt_target
            if best_iou < IOU_MATCH:
                continue  # spurious/unmatched detection -- not part of the "duplicate region" universe
            total_with_correspondence += 1
            fx_corresponded += 1

            single = TextManifest(
                asset_id="asset-b", total_regions=1, instances=[inst],
                src_lang="en", targ_lang="es",
            )
            matches = memory.lookup(single, str(tmp_path), "es", candidates)
            if inst.id in matches:
                predicted += 1
                fx_predicted += 1
                if matches[inst.id]["target_text"] == best_target:
                    correct += 1
                    fx_correct += 1

        fx_precision = fx_correct / fx_predicted if fx_predicted else None
        per_fixture_rows.append((fx.stem, fx_detected, fx_corresponded, fx_predicted, fx_correct, fx_precision))
        print(f"  {fx.stem}: detected={fx_detected} corresponded={fx_corresponded} "
              f"predicted={fx_predicted} correct={fx_correct} "
              f"precision={fx_precision:.3f}" if fx_precision is not None
              else f"  {fx.stem}: detected={fx_detected} corresponded={fx_corresponded} predicted=0")

    overall_precision = correct / predicted if predicted else None
    recall_of_correspondence = predicted / total_with_correspondence if total_with_correspondence else None

    print("\n== summary ==")
    print(f"regions detected in B: {total_detected}")
    print(f"regions with geometric correspondence to a stored A record: {total_with_correspondence}")
    print(f"regions with a TM match returned: {predicted}")
    print(f"matches that were the CORRECT record: {correct}")
    print(f"PRECISION (correct / predicted): {overall_precision}")
    print(f"match rate (predicted / correspondable): {recall_of_correspondence}")
    print(f"exit criterion (>=0.95 precision): "
          f"{'PASS' if overall_precision is not None and overall_precision >= 0.95 else 'FAIL'}")
    print(f"elapsed: {time.time() - t_start:.1f}s")

    report = {
        "candidates_stored": len(candidates),
        "per_fixture": [
            {"fixture": f, "detected": d, "corresponded": c, "predicted": p, "correct": k,
             "precision": pr}
            for f, d, c, p, k, pr in per_fixture_rows
        ],
        "total_detected": total_detected,
        "total_with_correspondence": total_with_correspondence,
        "predicted": predicted,
        "correct": correct,
        "overall_precision": overall_precision,
        "match_rate": recall_of_correspondence,
    }
    out_path = OUT_DIR / "memory_duplicate_pair_eval.json"
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nreport: {out_path}")


if __name__ == "__main__":
    main()
