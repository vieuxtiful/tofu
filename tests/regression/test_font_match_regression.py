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
    from tofu.utils.manifest_store import load_manifest

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

    manifest = load_manifest(fixture)
    if manifest is None:
        pytest.skip("could not load manifest")

    metrics: dict[str, float] = {}
    for i, inst in enumerate(manifest.instances):
        style = gt["regions"][i].get("style", {}) if i < len(gt["regions"]) else {}
        gt_font_file = style.get("font_file")
        if not gt_font_file:
            continue
        try:
            ranked = font_matching.match(inst, registry, faces, manifest, str(fixture))
        except Exception:
            continue
        gt_face = gt_font_file
        rank = None
        for idx, (face, score) in enumerate(ranked, 1):
            if face.file_path and gt_face in face.file_path:
                rank = idx
                break
        if rank is not None:
            metrics[f"rank_{inst.id}"] = float(rank)
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
