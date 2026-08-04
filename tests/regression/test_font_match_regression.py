"""Regression test for eval_font_match: font retrieval accuracy must not drift.

Uses the serif-vs-sans fixture (which has exact ground truth for the
font face) and asserts the rank of the correct face stays within
baseline tolerances.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from tolerances import compare  # noqa: E402


def _font_match_metrics() -> dict[str, float]:
    """Run font matching on serif-vs-sans and extract rank metrics."""
    from tofu.core.types import BBox, InstText, TextManifest
    from tofu.layers import font_matching, scene
    from tofu.layers.fonts import FontRegistry, faces_of, pantry

    fixture = ROOT / "tests" / "fixtures" / "serif-vs-sans.png"
    gt_path = ROOT / "tests" / "fixtures" / "serif-vs-sans.gt.json"
    if not fixture.exists():
        pytest.skip(f"fixture missing: {fixture}")

    import json
    gt = json.loads(gt_path.read_text(encoding="utf-8"))
    font_dir = pantry()
    if not font_dir:
        pytest.skip("no font directory found")
    registry = FontRegistry(font_dir)
    faces = faces_of(registry)

    # Build the manifest FROM the ground truth rather than detecting.
    #
    # This called load_manifest(fixture) with the PNG's path, and
    # load_manifest takes (store_dir, asset_id) and reads a PERSISTED
    # manifest -- so it raised TypeError on every run and the harness has
    # never measured anything. The unused BBox/InstText/TextManifest
    # imports above are what it was reaching for.
    #
    # Ground truth, not detection, because the loop below indexes
    # gt["regions"][i] against manifest.instances[i]. Detection returns a
    # different number of regions in a different order as recognition
    # changes, which would silently mis-pair each region with another
    # region's expected face -- and this harness exists to measure font
    # retrieval, not detection.
    instances = [
        InstText(
            id=f"r{i + 1}",
            bounding_box=BBox(x=region["bbox"][0], y=region["bbox"][1],
                              width=region["bbox"][2], height=region["bbox"][3]),
            text=region.get("text"), confidence=1.0, reading_order=i,
        )
        for i, region in enumerate(gt["regions"])
    ]
    manifest = TextManifest(
        asset_id="serif-vs-sans", total_regions=len(instances),
        instances=instances, img_dim=None, scene_regions=[],
    )
    try:
        manifest = scene.analyze(str(fixture), manifest)
    except Exception:
        pass  # typography enrichment is best-effort, as it is in the pipeline

    # local_match, not `font_matching.match` -- that name has never existed,
    # and the bare `except Exception: continue` around it turned the
    # AttributeError into an empty metrics dict, which the test then read as
    # "nothing to measure" and skipped. Two bugs hiding each other.
    from PIL import Image
    import numpy as np
    img = np.asarray(Image.open(fixture).convert("RGB"))

    metrics: dict[str, float] = {}
    for i, inst in enumerate(manifest.instances):
        style = gt["regions"][i].get("style", {}) if i < len(gt["regions"]) else {}
        gt_font_file = style.get("font_file")
        if not gt_font_file:
            continue
        result = font_matching.local_match(img, inst, registry)
        if not result:
            continue
        rank = None
        for idx, candidate in enumerate(result.get("candidates") or [], 1):
            path = candidate.get("font_path") or ""
            if gt_font_file.casefold() in path.casefold():
                rank = idx
                break
        # rank is the position of the TRUE face in the ranked candidate list,
        # so 1 is perfect and lower is better. An absent face is recorded as
        # a miss rather than skipped: dropping it would let the metric
        # improve by retrieving fewer faces.
        metrics[f"rank_{inst.id}"] = float(rank if rank is not None else
                                           len(result.get("candidates") or []) + 1)
    return metrics


def test_font_match_regression() -> None:
    """Font matching ranks must stay within baseline tolerances."""
    metrics = _font_match_metrics()
    if not metrics:
        pytest.skip("no font matching metrics produced")
    violations = compare("font_match", "serif-vs-sans", metrics)
    if violations:
        msgs = "\n".join(v.message() for v in violations)
        pytest.fail(f"Font matching regression detected:\n{msgs}")
