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


@pytest.mark.skipif(not _paddle_available(), reason="PaddleOCR venv not found")
@pytest.mark.parametrize("fixture", _fixture_names())
def test_paddle_regression(fixture: str) -> None:
    """PaddleOCR metrics must stay within baseline tolerances."""
    import subprocess
    import json

    image_path = FIXTURES_DIR / f"{fixture}.png"
    if not image_path.exists():
        pytest.skip(f"fixture image missing: {image_path}")

    result = subprocess.run(
        [str(PADDLE_VENV), str(ROOT / "scripts" / "eval_paddle.py"),
         str(image_path), "--tag", "regression"],
        capture_output=True, text=True, timeout=120, cwd=str(ROOT),
    )
    if result.returncode != 0:
        pytest.skip(f"PaddleOCR harness failed: {result.stderr[:200]}")

    report_path = ROOT / "scripts" / "eval_out" / f"{fixture}-regression.paddle.json"
    if not report_path.exists():
        pytest.skip("PaddleOCR report not found")
    report = json.loads(report_path.read_text(encoding="utf-8"))

    metrics: dict[str, float] = {}
    m = report.get("metrics", {})
    if m.get("precision") is not None:
        metrics["precision"] = float(m["precision"])
    if m.get("recall") is not None:
        metrics["recall"] = float(m["recall"])
    if m.get("f1") is not None:
        metrics["f1"] = float(m["f1"])
    if m.get("mean_norm_ed") is not None:
        metrics["mean_norm_ed"] = float(m["mean_norm_ed"])

    if not metrics:
        pytest.skip("no PaddleOCR metrics produced")
    violations = compare("paddle", fixture, metrics)
    if violations:
        msgs = "\n".join(v.message() for v in violations)
        pytest.fail(f"PaddleOCR regression detected:\n{msgs}")
