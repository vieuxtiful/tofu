"""Regression test for eval_render: render pipeline quality must not drift.

Runs the full render pipeline (cleanse → scribe → verify) on each
fixture and asserts QA scores, residual text similarity, and ring SSIM
stay within baseline tolerances.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from tolerances import compare  # noqa: E402

FIXTURES_DIR = ROOT / "tests" / "fixtures"
MANIFEST_CACHE = ROOT / "scripts" / "eval_out"


def _fixture_names() -> list[str]:
    names = []
    for gt in sorted(FIXTURES_DIR.glob("*.gt.json")):
        names.append(gt.stem)
    return names


def _render_metrics(fixture: str) -> dict[str, float]:
    """Run the render pipeline on one fixture and extract headline metrics."""
    from tofu.layers import cicerone, cleanse, scribe, verify
    from tofu.layers.fonts import FontRegistry, pantry
    from tofu.utils.manifest_store import _manifest_to_dict, _dict_to_manifest

    image_path = FIXTURES_DIR / f"{fixture}.png"
    if not image_path.exists():
        pytest.skip(f"fixture image missing: {image_path}")

    cached = MANIFEST_CACHE / f"{fixture}.manifest.json"
    if cached.exists():
        manifest = _dict_to_manifest(json.loads(cached.read_text(encoding="utf-8")))
    else:
        try:
            from tofu.layers import scene
            scene_regions = scene.analyze_regions(str(image_path))
        except Exception:
            scene_regions = []
        manifest = cicerone.detect(str(image_path), scene_regions=scene_regions)

    for inst in manifest.instances:
        if inst.text and not inst.target_text:
            inst.target_text = inst.text

    font_dir = pantry()
    registry = FontRegistry(font_dir) if font_dir else None
    targ_lang = manifest.src_lang or "en"

    cleansed = cleanse.erase(image_path, manifest)
    localized = scribe.render(cleansed, manifest, targ_lang, font_registry=registry)
    qa = verify.assess(localized, manifest, str(image_path), cleansed_asset=cleansed)

    metrics: dict[str, float] = {}
    if qa.overall_score is not None:
        metrics["qa_overall"] = round(float(qa.overall_score), 4)
    ring_ssim = qa.metrics.get("ring_ssim", {})
    if ring_ssim:
        metrics["mean_ring_ssim"] = round(sum(ring_ssim.values()) / len(ring_ssim), 4)
    ocr_rt = qa.metrics.get("ocr_roundtrip", {})
    if ocr_rt:
        metrics["mean_ocr_roundtrip"] = round(sum(ocr_rt.values()) / len(ocr_rt), 4)
    residual = qa.metrics.get("residual_text", {})
    if residual:
        metrics["mean_residual_similarity"] = round(sum(residual.values()) / len(residual), 4)
        metrics["max_residual_similarity"] = round(max(residual.values()), 4)
    return metrics


@pytest.mark.parametrize("fixture", _fixture_names())
def test_render_regression(fixture: str) -> None:
    """Render pipeline metrics must stay within baseline tolerances."""
    metrics = _render_metrics(fixture)
    if not metrics:
        pytest.skip(f"no render metrics produced for {fixture}")
    violations = compare("render", fixture, metrics)
    if violations:
        msgs = "\n".join(v.message() for v in violations)
        pytest.fail(f"Render regression detected:\n{msgs}")
