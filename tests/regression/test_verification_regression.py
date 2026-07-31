"""Regression test for eval_verification_corpus: verify contract must not drift.

Runs the deterministic verification corpus (no model-backed OCR by
default) and asserts pass rates stay within baseline tolerances.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from tolerances import compare  # noqa: E402

SPEC_PATH = ROOT / "scripts" / "verification_corpus.json"


def _verification_metrics() -> dict[str, float]:
    """Run the verification corpus and extract headline metrics."""
    from eval_verification_corpus import load_spec, run_case

    spec = load_spec(SPEC_PATH)
    output_dir = ROOT / "scripts" / "eval_out" / "verification-regression"
    output_dir.mkdir(parents=True, exist_ok=True)

    total = 0
    matched = 0
    per_wave: dict[str, int] = {}
    per_wave_matched: dict[str, int] = {}

    for case in spec["cases"]:
        wave = case.get("wave", "default")
        result = run_case(case, output_dir, real_ocr=False)
        total += 1
        per_wave[wave] = per_wave.get(wave, 0) + 1
        if result["matched"]:
            matched += 1
            per_wave_matched[wave] = per_wave_matched.get(wave, 0) + 1

    pass_rate = matched / total if total else 0.0
    metrics: dict[str, float] = {"pass_rate": round(pass_rate, 4)}
    for wave in per_wave:
        w_rate = per_wave_matched.get(wave, 0) / per_wave[wave] if per_wave[wave] else 0.0
        metrics[f"pass_rate_{wave}"] = round(w_rate, 4)
    return metrics


def test_verification_regression() -> None:
    """Verification corpus pass rates must stay within baseline tolerances."""
    if not SPEC_PATH.exists():
        pytest.skip(f"verification corpus spec missing: {SPEC_PATH}")
    metrics = _verification_metrics()
    violations = compare("verification_corpus", "overall", metrics)
    if violations:
        msgs = "\n".join(v.message() for v in violations)
        pytest.fail(f"Verification regression detected:\n{msgs}")
