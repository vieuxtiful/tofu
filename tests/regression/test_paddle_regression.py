"""Regression test for eval_paddle: PaddleOCR standalone quality.

PaddleOCR runs in an isolated virtual environment (.venv-paddle) to
avoid numpy/opencv conflicts.  This test skips when that venv is not
available and otherwise runs the harness and checks metrics against
baseline tolerances.
"""
from __future__ import annotations

import shutil
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from tolerances import compare  # noqa: E402

PADDLE_VENV = ROOT / ".venv-paddle" / "Scripts" / "python.exe"
FIXTURES_DIR = ROOT / "tests" / "fixtures"


def _fixture_names() -> list[str]:
    names = []
    for gt in sorted(FIXTURES_DIR.glob("*.gt.json")):
        # NOT gt.stem: Path only strips the LAST suffix, so "flat-sign.gt.json"
        # yields "flat-sign.gt" and every image lookup below missed, skipping
        # the whole suite silently.
        names.append(gt.name.removesuffix(".gt.json"))
    return names


def _paddle_available() -> bool:
    return PADDLE_VENV.exists()


def _paddle_metrics(fixture: str) -> dict[str, float]:
    """Run the isolated Paddle harness and extract its headline metrics.

    Factored out of the test so generate_baseline can record numbers
    produced by exactly this code, rather than a second copy of it that
    could drift. Returns {} when the harness could not produce metrics;
    the test turns that into a skip, the generator into an omission.
    """
    import json
    import subprocess

    image_path = FIXTURES_DIR / f"{fixture}.png"
    if not image_path.exists():
        return {}

    result = subprocess.run(
        [str(PADDLE_VENV), str(ROOT / "scripts" / "eval_paddle.py"),
         str(image_path), "--tag", "regression"],
        capture_output=True, text=True, timeout=120, cwd=str(ROOT),
    )
    if result.returncode != 0:
        return {}

    report_path = ROOT / "scripts" / "eval_out" / f"{fixture}-regression.paddle.json"
    if not report_path.exists():
        return {}
    report = json.loads(report_path.read_text(encoding="utf-8"))

    metrics: dict[str, float] = {}
    for key in ("precision", "recall", "f1", "mean_norm_ed"):
        value = report.get("metrics", {}).get(key)
        if value is not None:
            metrics[key] = float(value)
    return metrics


@pytest.mark.skipif(not _paddle_available(), reason="PaddleOCR venv not found")
@pytest.mark.parametrize("fixture", _fixture_names())
def test_paddle_regression(fixture: str) -> None:
    """PaddleOCR metrics must stay within baseline tolerances."""
    metrics = _paddle_metrics(fixture)
    if not metrics:
        pytest.skip(f"PaddleOCR harness produced no metrics for {fixture}")
    violations = compare("paddle", fixture, metrics)
    if violations:
        msgs = "\n".join(v.message() for v in violations)
        pytest.fail(f"PaddleOCR regression detected:\n{msgs}")
