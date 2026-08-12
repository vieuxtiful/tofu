"""Calibration and decision policy for learned glyph-match evidence."""

from __future__ import annotations

import json
import math
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from tofu.layers import decant, temper

MIN_CALIBRATION_SAMPLES = 30
MIN_ACCEPTED_SAMPLES = 5
TARGET_GREEN_PRECISION = 0.98


def _sigmoid(value: float) -> float:
    if value >= 0:
        return 1.0 / (1.0 + math.exp(-value))
    exp_value = math.exp(value)
    return exp_value / (1.0 + exp_value)


@dataclass
class MarginCalibration:
    slope: float = 1.0
    intercept: float = 0.0
    samples: int = 0
    positives: int = 0
    negatives: int = 0
    fitted: bool = False
    acceptance_threshold: float | None = None
    target_precision: float = TARGET_GREEN_PRECISION
    ece_before: float | None = None
    ece_after: float | None = None
    brier_before: float | None = None
    brier_after: float | None = None
    revision: str = "proof-margin-unfitted"
    notes: list[str] = field(default_factory=list)

    def apply(self, margin: float | None) -> float | None:
        if margin is None:
            return None
        return round(_sigmoid(self.slope * float(margin) + self.intercept), 6)

    def to_dict(self) -> dict[str, Any]:
        return {"schema": 1, "method": "platt_margin", **asdict(self)}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> MarginCalibration:
        names = cls.__dataclass_fields__
        return cls(**{key: value for key, value in data.items() if key in names})


def fit(
    margins: Sequence[float],
    outcomes: Sequence[bool],
    *,
    target_precision: float = TARGET_GREEN_PRECISION,
    revision: str = "proof-margin-v1",
) -> MarginCalibration:
    pairs = [(float(margin), bool(outcome)) for margin, outcome in zip(margins, outcomes, strict=False)]
    result = MarginCalibration(
        samples=len(pairs),
        positives=sum(outcome for _, outcome in pairs),
        negatives=sum(not outcome for _, outcome in pairs),
        target_precision=target_precision,
        revision=revision,
    )
    if not pairs:
        result.notes.append("no labelled matches; calibration remains unavailable")
        return result

    raw = [_sigmoid(margin) for margin, _ in pairs]
    labels = [outcome for _, outcome in pairs]
    result.ece_before = temper.expected_calibration_error(raw, labels)
    result.brier_before = temper.brier_score(raw, labels)
    if len(pairs) < MIN_CALIBRATION_SAMPLES:
        result.notes.append(
            f"only {len(pairs)} labelled match(es); need {MIN_CALIBRATION_SAMPLES}"
        )
        return result
    if result.positives == 0 or result.negatives == 0:
        result.notes.append("both correct and incorrect matches are required")
        return result

    slope, intercept = 1.0, 0.0
    learning_rate = 0.1
    for _ in range(2000):
        grad_slope = 0.0
        grad_intercept = 0.0
        for margin, outcome in pairs:
            error = _sigmoid(slope * margin + intercept) - float(outcome)
            grad_slope += error * margin
            grad_intercept += error
        slope -= learning_rate * grad_slope / len(pairs)
        intercept -= learning_rate * grad_intercept / len(pairs)
    result.slope = round(slope, 8)
    result.intercept = round(intercept, 8)
    result.fitted = True

    calibrated = [result.apply(margin) for margin, _ in pairs]
    result.ece_after = temper.expected_calibration_error(calibrated, labels)
    result.brier_after = temper.brier_score(calibrated, labels)
    if result.slope <= 0:
        result.notes.append(
            "fitted slope is non-positive; larger margins do not predict correctness"
        )
        return result
    result.acceptance_threshold = _precision_threshold(
        calibrated, labels, target_precision
    )
    if result.acceptance_threshold is None:
        result.notes.append(
            f"no threshold accepts {MIN_ACCEPTED_SAMPLES} samples at "
            f"precision {target_precision:.3f}"
        )
    return result


def _precision_threshold(
    probabilities: Sequence[float], outcomes: Sequence[bool], target: float
) -> float | None:
    for threshold in sorted(set(probabilities)):
        accepted = [outcome for probability, outcome in zip(probabilities, outcomes, strict=False)
                    if probability >= threshold]
        if len(accepted) >= MIN_ACCEPTED_SAMPLES and sum(accepted) / len(accepted) >= target:
            return round(float(threshold), 6)
    return None


def decide(
    margin: float | None,
    survival: dict[str, Any] | None,
    calibration: MarginCalibration,
) -> dict[str, Any]:
    state = (survival or {}).get("state", decant.UNKNOWN)
    probability = calibration.apply(margin) if calibration.fitted else None
    decision = "review_required"
    reasons = []
    if state in {decant.ABSENT, decant.WEAK}:
        decision = "unresolvable"
        reasons.append(f"evidence_survival_{state}")
    elif state != decant.PRESENT:
        reasons.append(f"evidence_survival_{state}")
    elif not calibration.fitted:
        reasons.append("margin_calibration_unavailable")
    elif calibration.acceptance_threshold is None:
        reasons.append("acceptance_threshold_unavailable")
    elif probability is not None and probability >= calibration.acceptance_threshold:
        decision = "accepted"
    else:
        reasons.append("calibrated_probability_below_acceptance")
    return {
        "schema": 1,
        "decision": decision,
        "raw_margin": margin,
        "calibrated_probability": probability,
        "evidence_survival": state,
        "calibration_revision": calibration.revision,
        "acceptance_threshold": calibration.acceptance_threshold,
        "reasons": reasons,
    }


def save(calibration: MarginCalibration, path: Any) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(calibration.to_dict(), indent=2) + "\n", encoding="utf-8")


def load(path: Any) -> MarginCalibration:
    try:
        target = Path(path)
        if not target.is_file():
            return MarginCalibration(notes=["no calibration artifact"])
        return MarginCalibration.from_dict(json.loads(target.read_text(encoding="utf-8")))
    except Exception as exc:
        return MarginCalibration(notes=[f"calibration unreadable ({type(exc).__name__})"])
