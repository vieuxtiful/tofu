"""Regression test for eval_tofu: preflight score distributions must not drift.

Runs the ToFU preflight validator on each fixture across a spread of
target languages and asserts glyph segmentation and render quality
scores stay within baseline tolerances.
"""
from __future__ import annotations

import json
import statistics
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from tolerances import compare  # noqa: E402

FIXTURES_DIR = ROOT / "tests" / "fixtures"
MANIFEST_CACHE = ROOT / "scripts" / "eval_out" / "tofu_manifests"

DEFAULT_LANGS = ("en", "de", "ja", "ar", "hi", "th")


def _fixture_names() -> list[str]:
    names = []
    for gt in sorted(FIXTURES_DIR.glob("*.gt.json")):
        names.append(gt.stem)
    return names


def _tofu_metrics(fixture: str) -> dict[str, float]:
    """Run ToFU preflight on one fixture and extract score metrics."""
    from tofu.core.pipeline import TofuPipeline
    from tofu.layers import cicerone, scene
    from tofu.layers.fonts import FontRegistry, pantry
    from tofu.layers.tofu import ToFU, lang_to_script
    from tofu.utils.manifest_store import _dict_to_manifest, _manifest_to_dict

    image_path = FIXTURES_DIR / f"{fixture}.png"
    if not image_path.exists():
        pytest.skip(f"fixture image missing: {image_path}")

    cached = MANIFEST_CACHE / f"{fixture}.json"
    if cached.exists():
        manifest = _dict_to_manifest(json.loads(cached.read_text(encoding="utf-8")))
    else:
        try:
            surfaces = scene.analyze_regions(str(image_path))
        except Exception:
            surfaces = []
        manifest = cicerone.detect(str(image_path), scene_regions=surfaces)
        try:
            manifest = scene.analyze(str(image_path), manifest)
        except Exception:
            pass

    font_dir = pantry()
    registry = FontRegistry(font_dir) if font_dir else None
    validator = ToFU(font_library_path=font_dir)

    all_glyph: list[float] = []
    all_render: list[float] = []

    for lang in DEFAULT_LANGS:
        script = lang_to_script.get(lang)
        font = None
        if registry and script:
            ranked = registry.recommend(script, lang, limit=1)
            if ranked and ranked[0][1] > 0:
                font = ranked[0][0]
        context = TofuPipeline._tofu_context(font, manifest)
        report = validator.validate(str(image_path), lang, context, manifest)
        if report.glyph_segmentation_score is not None:
            all_glyph.append(report.glyph_segmentation_score)
        if report.render_quality_score is not None:
            all_render.append(report.render_quality_score)

    metrics: dict[str, float] = {}
    if all_glyph:
        metrics["glyph_median"] = round(statistics.median(all_glyph), 4)
        metrics["glyph_min"] = round(min(all_glyph), 4)
    if all_render:
        metrics["render_median"] = round(statistics.median(all_render), 4)
        metrics["render_min"] = round(min(all_render), 4)
    return metrics


@pytest.mark.parametrize("fixture", _fixture_names())
def test_tofu_regression(fixture: str) -> None:
    """ToFU preflight scores must stay within baseline tolerances."""
    metrics = _tofu_metrics(fixture)
    if not metrics:
        pytest.skip(f"no ToFU metrics produced for {fixture}")
    violations = compare("tofu", fixture, metrics)
    if violations:
        msgs = "\n".join(v.message() for v in violations)
        pytest.fail(f"ToFU regression detected:\n{msgs}")
