"""Regression test for eval_savor: OCR correction must not drift.

Runs detection (with savor enabled) on each fixture and asserts the
applied and unresolved correction counts stay within baseline
tolerances.
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

FIXTURES_DIR = ROOT / "tests" / "fixtures"


def _fixture_names() -> list[str]:
    names = []
    for gt in sorted(FIXTURES_DIR.glob("*.gt.json")):
        # NOT gt.stem: Path only strips the LAST suffix, so "flat-sign.gt.json"
        # yields "flat-sign.gt" and every image lookup below missed, skipping
        # the whole suite silently.
        names.append(gt.name.removesuffix(".gt.json"))
    return names


def _savor_metrics(fixture: str) -> dict[str, float]:
    """Run detection with savor and extract correction metrics."""
    from tofu.core.types import AssetInfo, AssetType
    from tofu.layers import cicerone

    image_path = FIXTURES_DIR / f"{fixture}.png"
    if not image_path.exists():
        pytest.skip(f"fixture image missing: {image_path}")

    info = AssetInfo(asset_type=AssetType.IMAGE, frame_count=1, source=str(image_path))
    manifest = cicerone.detect(str(image_path), info)

    applied = 0
    unresolved = 0
    for inst in manifest.instances:
        correction = inst.ocr_correction or {}
        verdict = correction.get("verdict")
        if verdict == "applied":
            applied += 1
        elif verdict == "unresolved":
            unresolved += 1

    return {
        "applied": float(applied),
        "unresolved": float(unresolved),
    }


@pytest.mark.parametrize("fixture", _fixture_names())
def test_savor_regression(fixture: str) -> None:
    """Savor correction counts must stay within baseline tolerances."""
    metrics = _savor_metrics(fixture)
    violations = compare("savor", fixture, metrics)
    if violations:
        msgs = "\n".join(v.message() for v in violations)
        pytest.fail(f"Savor regression detected:\n{msgs}")
