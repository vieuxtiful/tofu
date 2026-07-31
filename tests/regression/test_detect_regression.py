"""Regression test for eval_detect: detection quality must not drift.

Runs cicerone.detect on every fixture with a .gt.json and asserts that
precision, recall, and mean normalized edit distance stay within the
tolerances recorded in baseline_metrics.json.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from tolerances import compare, ToleranceViolation  # noqa: E402

FIXTURES_DIR = ROOT / "tests" / "fixtures"


def _fixture_names() -> list[str]:
    names = []
    for gt in sorted(FIXTURES_DIR.glob("*.gt.json")):
        names.append(gt.stem)
    return names


def _detect_metrics(fixture: str) -> dict[str, float]:
    """Run detection and extract the headline metrics."""
    from tofu.layers import cicerone, scene
    from eval_detect import evaluate as detect_evaluate, _font_registry

    image_path = FIXTURES_DIR / f"{fixture}.png"
    gt_path = FIXTURES_DIR / f"{fixture}.gt.json"
    if not image_path.exists():
        pytest.skip(f"fixture image missing: {image_path}")

    try:
        scene_regions = scene.analyze_regions(str(image_path))
    except Exception:
        scene_regions = []
    manifest = cicerone.detect(
        str(image_path), scene_regions=scene_regions,
        font_registry=_font_registry(),
    )
    metrics = detect_evaluate(image_path, manifest, gt_path, scene_regions)
    result: dict[str, float] = {}
    if metrics.get("precision") is not None:
        result["precision"] = float(metrics["precision"])
    if metrics.get("recall") is not None:
        result["recall"] = float(metrics["recall"])
    if metrics.get("f1") is not None:
        result["f1"] = float(metrics["f1"])
    result["mean_norm_ed"] = float(metrics["mean_norm_ed"])
    return result


@pytest.mark.parametrize("fixture", _fixture_names())
def test_detect_regression(fixture: str) -> None:
    """Detection metrics must stay within baseline tolerances."""
    metrics = _detect_metrics(fixture)
    violations = compare("detect", fixture, metrics)
    if violations:
        msgs = "\n".join(v.message() for v in violations)
        pytest.fail(f"Detection regression detected:\n{msgs}")
