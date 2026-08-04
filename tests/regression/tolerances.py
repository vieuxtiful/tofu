"""Tolerance loader and validator for regression tests.

Converts the project's 8 eval harnesses from report generators into a
regression suite: a checked-in JSON file stores expected metrics per
fixture per harness, and ``compare`` fails when a measured metric drifts
outside its tolerance band.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

ROOT = Path(__file__).resolve().parents[2]
BASELINE_PATH = ROOT / "tests" / "fixtures" / "regression" / "baseline_metrics.json"


## Metrics where a SMALLER number is the better result.  Recording these as
## "at_least" -- which is what every generated entry used to be -- inverts the
## guard they exist to provide: textured-wall's mean_norm_ed baselined at 0.056
## would FAIL if recognition improved to 0.0, and PASS if it degraded to 0.9.
## Anything unrecognised keeps the higher-is-better default, so a new
## score-like metric behaves as before and a new error-like one is a one-line
## addition here.
## `*_residual_similarity` is how much of the SOURCE text still shows after
## erasure -- eval_render prints it as "lower = cleaner erase". Left on the
## higher-is-better default, a render that erased perfectly (0.0) would fail
## against a baseline of 0.5, and one that stopped erasing would pass.
LOWER_IS_BETTER_SUFFIXES = ("_mae", "_ed", "_error", "_err", "_distance", "_loss",
                            "_residual_similarity")
LOWER_IS_BETTER_NAMES = frozenset({"unresolved"})
## Metric families named by PREFIX rather than suffix.  `rank_r1` is the
## position of the true font face in a ranked candidate list -- 1 is a
## perfect retrieval and 5 is a bad one -- so the higher-is-better default
## would invert that guard exactly the way the textured-wall note above
## describes: a rank degrading 1 -> 5 would PASS.
LOWER_IS_BETTER_PREFIXES = ("rank_",)


def direction_for(metric: str) -> str:
    """Which way is 'better' for this metric name."""
    name = metric.casefold()
    if (name in LOWER_IS_BETTER_NAMES
            or name.endswith(LOWER_IS_BETTER_SUFFIXES)
            or name.startswith(LOWER_IS_BETTER_PREFIXES)):
        return "at_most"
    return "at_least"


@dataclass(frozen=True)
class Tolerance:
    metric: str
    expected: float
    tolerance: float
    direction: str = "at_least"  # at_least | at_most | within

    def check(self, value: float) -> bool:
        if self.direction == "at_least":
            return value >= self.expected - self.tolerance
        if self.direction == "at_most":
            return value <= self.expected + self.tolerance
        # within
        return abs(value - self.expected) <= self.tolerance


@dataclass(frozen=True)
class ToleranceViolation:
    harness: str
    fixture: str
    metric: str
    expected: float
    actual: float
    tolerance: float
    direction: str

    def message(self) -> str:
        return (
            f"[{self.harness}/{self.fixture}] {self.metric}: "
            f"expected {self.expected} ({self.direction} ±{self.tolerance}), "
            f"got {self.actual}"
        )


def load_baseline(harness: Optional[str] = None) -> Dict[str, Any]:
    if not BASELINE_PATH.exists():
        return {} if harness else {}
    data = json.loads(BASELINE_PATH.read_text(encoding="utf-8"))
    if harness is None:
        return data
    return data.get("harnesses", {}).get(harness, {})


def _extract_tolerances(harness: str, fixture: str, raw: Dict[str, Any]) -> List[Tolerance]:
    tolerances: List[Tolerance] = []
    for metric_name, spec in raw.items():
        if not isinstance(spec, dict):
            continue
        if "expected" not in spec:
            continue
        tolerances.append(Tolerance(
            metric=metric_name,
            expected=float(spec["expected"]),
            tolerance=float(spec.get("tolerance", 0.0)),
            direction=spec.get("direction", "at_least"),
        ))
    return tolerances


def tolerances_for(harness: str, fixture: str) -> List[Tolerance]:
    """Return the tolerance entries for one harness+fixture pair."""
    data = load_baseline(harness)
    fixture_data = data.get(fixture)
    if not isinstance(fixture_data, dict):
        return []
    return _extract_tolerances(harness, fixture, fixture_data)


def compare(harness: str, fixture: str, results: Dict[str, float]) -> List[ToleranceViolation]:
    """Compare measured results against the baseline.

    ``results`` maps metric names to measured values.  Metrics in the
    baseline that are absent from ``results`` are skipped (the harness may
    not produce every metric for every fixture, e.g. precision on partial
    ground truth).
    """
    violations: List[ToleranceViolation] = []
    for tol in tolerances_for(harness, fixture):
        if tol.metric not in results:
            continue
        value = float(results[tol.metric])
        if not tol.check(value):
            violations.append(ToleranceViolation(
                harness=harness, fixture=fixture, metric=tol.metric,
                expected=tol.expected, actual=value,
                tolerance=tol.tolerance, direction=tol.direction,
            ))
    return violations


def compare_all(harness: str, results_by_fixture: Dict[str, Dict[str, float]]) -> List[ToleranceViolation]:
    """Compare results for multiple fixtures at once."""
    violations: List[ToleranceViolation] = []
    for fixture, metrics in results_by_fixture.items():
        violations.extend(compare(harness, fixture, metrics))
    return violations


def update_baseline(harness: str, results_by_fixture: Dict[str, Dict[str, float]]) -> None:
    """Rewrite the baseline with measured values.

    Preserves existing entries for other harnesses.  New metrics are
    added with a default tolerance of 0 (exact match); the caller is
    expected to widen tolerances manually after reviewing the initial
    baseline.
    """
    data = load_baseline()
    if not data:
        data = {"schema": 1, "harnesses": {}}
    harnesses = data.setdefault("harnesses", {})
    existing = harnesses.get(harness, {})
    for fixture, metrics in results_by_fixture.items():
        fixture_entry = existing.get(fixture, {})
        for metric, value in metrics.items():
            existing_metric = fixture_entry.get(metric)
            # Tolerance is a POLICY knob the docstring invites you to widen by
            # hand, so a re-generation preserves it.  Direction is not: it is a
            # property of what the metric measures, so it is always refreshed.
            tolerance = (
                float(existing_metric.get("tolerance", 0.0))
                if isinstance(existing_metric, dict) else 0.0
            )
            fixture_entry[metric] = {
                "expected": round(float(value), 4),
                "tolerance": tolerance,
                "direction": direction_for(metric),
            }
        existing[fixture] = fixture_entry
    harnesses[harness] = existing
    BASELINE_PATH.parent.mkdir(parents=True, exist_ok=True)
    # json.dumps has no `encoding` parameter (it went away in Python 3), so
    # every write raised TypeError and was swallowed by the generator's
    # per-harness `except Exception` -- which is why the baseline file has
    # sat with an empty `harnesses` object and guarded nothing.
    BASELINE_PATH.write_text(json.dumps(data, indent=2), encoding="utf-8")
