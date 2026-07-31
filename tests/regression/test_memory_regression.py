"""Regression test for eval_memory: TM duplicate-pair match precision must not drift.

Runs the duplicate-pair TM matching methodology on each fixture and
asserts per-fixture precision and overall precision stay within
baseline tolerances.
"""
from __future__ import annotations

import io
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from tolerances import compare  # noqa: E402

FIXTURES_DIR = ROOT / "tests" / "fixtures"
SCALE = 0.85
JPEG_Q = 80
IOU_MATCH = 0.3


def _fixture_names() -> list[str]:
    names = []
    for gt in sorted(FIXTURES_DIR.glob("*.gt.json")):
        # NOT gt.stem: Path only strips the LAST suffix, so "flat-sign.gt.json"
        # yields "flat-sign.gt" and every image lookup below missed, skipping
        # the whole suite silently.
        names.append(gt.name.removesuffix(".gt.json"))
    return names


def _memory_metrics(fixture: str) -> dict[str, float]:
    """Run duplicate-pair TM matching for one fixture and extract metrics."""
    from PIL import Image
    from tofu.core.types import AssetInfo, AssetType, BBox, QAReport, TextManifest
    from tofu.layers import cicerone, memory

    fx = FIXTURES_DIR / f"{fixture}.png"
    if not fx.exists():
        pytest.skip(f"fixture image missing: {fx}")

    info_a = AssetInfo(asset_type=AssetType.IMAGE, frame_count=1, source=str(fx))
    manifest_a = cicerone.detect(str(fx), info_a)

    instances_with_text = [i for i in manifest_a.instances if i.text]
    fixture_gt = []
    for inst in instances_with_text:
        inst.target_text = f"T::{fx.stem}::{inst.id}"
        fixture_gt.append((inst.bounding_box, inst.target_text))

    if instances_with_text:
        qa = QAReport(
            overall_score=1.0,
            per_asset_instance_score={manifest_a.asset_id: {i.id: 1.0 for i in instances_with_text}},
        )
        candidates = memory.update(manifest_a, "es", None, str(fx), qa, qa_threshold=0.5)
    else:
        candidates = []

    img = Image.open(fx).convert("RGB")
    w, h = img.size
    resized = img.resize((max(1, int(w * SCALE)), max(1, int(h * SCALE))))
    buf = io.BytesIO()
    resized.save(buf, format="JPEG", quality=JPEG_Q)
    buf.seek(0)
    perturbed = Image.open(buf).convert("RGB")
    tmp_path = ROOT / "scripts" / "eval_out" / f"_memeval_{fx.stem}.jpg"
    tmp_path.parent.mkdir(parents=True, exist_ok=True)
    perturbed.save(tmp_path)

    info_b = AssetInfo(asset_type=AssetType.IMAGE, frame_count=1, source=str(tmp_path))
    manifest_b = cicerone.detect(str(tmp_path), info_b)

    predicted = 0
    correct = 0
    for inst in manifest_b.instances:
        if not inst.text:
            continue
        b = inst.bounding_box
        orig_box = BBox(
            x=int(b.x / SCALE), y=int(b.y / SCALE),
            width=int(b.width / SCALE), height=int(b.height / SCALE),
        )
        best_iou, best_target = 0.0, None
        for gt_box, gt_target in fixture_gt:
            from eval_memory import iou
            score = iou(orig_box, gt_box)
            if score > best_iou:
                best_iou, best_target = score, gt_target
        if best_iou < IOU_MATCH:
            continue
        single = TextManifest(
            asset_id="asset-b", total_regions=1, instances=[inst],
            src_lang="en", targ_lang="es",
        )
        matches = memory.lookup(single, str(tmp_path), "es", candidates)
        if inst.id in matches:
            predicted += 1
            if matches[inst.id]["target_text"] == best_target:
                correct += 1

    precision = correct / predicted if predicted else None
    metrics: dict[str, float] = {}
    if precision is not None:
        metrics["precision"] = round(precision, 4)
    metrics["predicted"] = float(predicted)
    metrics["correct"] = float(correct)
    return metrics


@pytest.mark.parametrize("fixture", _fixture_names())
def test_memory_regression(fixture: str) -> None:
    """TM duplicate-pair precision must stay within baseline tolerances."""
    metrics = _memory_metrics(fixture)
    violations = compare("memory", fixture, metrics)
    if violations:
        msgs = "\n".join(v.message() for v in violations)
        pytest.fail(f"Memory regression detected:\n{msgs}")
