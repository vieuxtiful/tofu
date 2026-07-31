"""Regression test for eval_cleanse_providers: inpainting quality must not drift.

Uses the same synthetic cases as the eval harness (flat, gradient,
textured, repeating, outlined) and asserts mask_mae, outside_mae, and
mask_recall stay within baseline tolerances.
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


_CLEANSE_CASES = ("flat", "gradient", "textured", "repeating", "outlined")


def _cleanse_metrics(case_name: str) -> dict[str, float]:
    """Run one synthetic cleanse case and extract headline metrics."""
    import numpy as np
    from eval_cleanse_providers import _case
    from tofu.layers import cleanse

    clean, source, manifest = _case(case_name)
    before = np.asarray(source)
    b = manifest.instances[0].bounding_box
    import cv2
    full, _ = cleanse._select_region_mask(
        np, cv2, before, b, before.shape[0], before.shape[1],
        manifest.instances[0].segmentation_mask.polygon, None,
    )
    output = cleanse.erase(source, manifest).convert("RGB")
    after = np.asarray(output)
    clean_arr = np.asarray(clean)
    mask_mae = float(np.abs(after[full].astype(float) - clean_arr[full]).mean())
    outside_mae = float(np.abs(after[~full].astype(float) - before[~full]).mean())
    truth_mask = np.any(before != clean_arr, axis=2)
    mask_recall = int((full & truth_mask).sum()) / max(1, int(truth_mask.sum()))
    return {
        "mask_mae": round(mask_mae, 4),
        "outside_mae": round(outside_mae, 6),
        "mask_recall": round(mask_recall, 5),
    }


@pytest.mark.parametrize("case_name", _CLEANSE_CASES)
def test_cleanse_regression(case_name: str) -> None:
    """Cleanse provider metrics must stay within baseline tolerances."""
    metrics = _cleanse_metrics(case_name)
    violations = compare("cleanse_providers", case_name, metrics)
    if violations:
        msgs = "\n".join(v.message() for v in violations)
        pytest.fail(f"Cleanse regression detected:\n{msgs}")
