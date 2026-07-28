## 🍢 Temper - ToFU confidence calibration
## vieuxtiful
"""
tempering brings something to the state where it behaves predictably --
chocolate that sets with a snap, eggs that thicken instead of scrambling.
The machine-learning term for the same idea is the same word: temperature
scaling, which brings a model's confidence into agreement with how often it
is actually right.

The problem this solves: a verifier that says "90% confident" should be
correct about nine times in ten.  Nothing guarantees that.  A score can rank
regions perfectly and still be systematically overconfident, and a reviewer
who trusts the number then skips regions that needed them.  Ranking quality
and calibration are different properties and need different measurements.

  ECE      Expected Calibration Error.  Bin predictions by confidence, and
           in each bin compare mean confidence against observed accuracy;
           ECE is the sample-weighted mean of those gaps.  Guo, Pleiss, Sun
           & Weinberger, "On Calibration of Modern Neural Networks",
           ICML 2017.

  Brier    Mean squared error between the predicted probability and the
           outcome.  Brier, "Verification of Forecasts Expressed in Terms of
           Probability", Monthly Weather Review 78(1), 1950.  Unlike ECE it
           is a proper scoring rule: it rewards being both calibrated AND
           discriminative, so the two are reported together.

  fitting  Temperature scaling (Guo et al. again), a single parameter fitted
           by minimising negative log-likelihood.  One parameter is
           deliberate -- it cannot reorder anything, so it can improve
           calibration without touching which regions rank as worst, and it
           cannot overfit a corpus of a few dozen regions the way isotonic
           regression would.

BOTH metrics are corpus-level by construction: they compare a SET of
confidences against a SET of outcomes.  Neither is defined for a single
region -- a one-sample ECE has one bin whose gap is just the error, and a
one-sample Brier is a squared residual.  So the harness measures them over
the corpus and fits the map; Verify only ever applies the fitted map.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

DEFAULT_BINS = 10
_EPS = 1e-6


def _clamp(p: float) -> float:
    """Keep probabilities off 0 and 1 so logit and log stay finite."""
    return min(1.0 - _EPS, max(_EPS, float(p)))


def _logit(p: float) -> float:
    p = _clamp(p)
    return math.log(p / (1.0 - p))


def _sigmoid(x: float) -> float:
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    exp_x = math.exp(x)          # avoids overflow for large negative x
    return exp_x / (1.0 + exp_x)


def expected_calibration_error(
    confidences: Sequence[float],
    outcomes: Sequence[bool],
    bins: int = DEFAULT_BINS,
) -> Optional[float]:
    """Binned ECE (Guo et al. 2017), in 0-1. Lower is better.

    Empty bins contribute nothing, and the weights are bin populations, so a
    bin holding one region cannot outvote a bin holding forty.
    """
    pairs = [(_clamp(c), bool(o)) for c, o in zip(confidences, outcomes) if c is not None]
    if not pairs:
        return None
    total = len(pairs)
    error = 0.0
    for index in range(bins):
        low = index / bins
        high = (index + 1) / bins
        # upper-closed on the last bin so a confidence of exactly 1.0 lands
        members = [
            (c, o) for c, o in pairs
            if (low < c <= high) or (index == 0 and c <= high)
        ]
        if not members:
            continue
        mean_confidence = sum(c for c, _ in members) / len(members)
        accuracy = sum(1 for _, o in members if o) / len(members)
        error += (len(members) / total) * abs(accuracy - mean_confidence)
    return round(error, 6)


def brier_score(
    confidences: Sequence[float], outcomes: Sequence[bool],
) -> Optional[float]:
    """Mean squared error of the probability forecast (Brier 1950), 0-1."""
    pairs = [(_clamp(c), bool(o)) for c, o in zip(confidences, outcomes) if c is not None]
    if not pairs:
        return None
    return round(sum((c - (1.0 if o else 0.0)) ** 2 for c, o in pairs) / len(pairs), 6)


def _nll(confidences: Sequence[float], outcomes: Sequence[bool], temperature: float) -> float:
    """Negative log-likelihood of the outcomes under a tempered forecast."""
    total = 0.0
    for confidence, outcome in zip(confidences, outcomes):
        p = _clamp(_sigmoid(_logit(confidence) / temperature))
        total -= math.log(p) if outcome else math.log(1.0 - p)
    return total


@dataclass
class Calibration:
    """A fitted temperature plus the evidence that justified it."""
    temperature: float = 1.0
    samples: int = 0
    ece_before: Optional[float] = None
    ece_after: Optional[float] = None
    brier_before: Optional[float] = None
    brier_after: Optional[float] = None
    bins: int = DEFAULT_BINS
    fitted_at: Optional[str] = None
    notes: List[str] = field(default_factory=list)

    def apply(self, confidence: Optional[float]) -> Optional[float]:
        """Temper one confidence. Identity at T=1, so an unfitted map is safe."""
        if confidence is None:
            return None
        if abs(self.temperature - 1.0) < 1e-9:
            return float(confidence)
        return round(_sigmoid(_logit(confidence) / self.temperature), 6)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema": 1, "method": "temperature_scaling",
            "temperature": self.temperature, "samples": self.samples,
            "ece_before": self.ece_before, "ece_after": self.ece_after,
            "brier_before": self.brier_before, "brier_after": self.brier_after,
            "bins": self.bins, "fitted_at": self.fitted_at, "notes": self.notes,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Calibration":
        return cls(
            temperature=float(data.get("temperature", 1.0)),
            samples=int(data.get("samples", 0)),
            ece_before=data.get("ece_before"), ece_after=data.get("ece_after"),
            brier_before=data.get("brier_before"), brier_after=data.get("brier_after"),
            bins=int(data.get("bins", DEFAULT_BINS)),
            fitted_at=data.get("fitted_at"), notes=list(data.get("notes") or []),
        )


#: Below this many labelled regions a fitted temperature is noise, so the
#: map stays the identity and says why.  Guo et al. fit on validation sets
#: of thousands; a demo corpus is three orders of magnitude smaller, and a
#: temperature fitted on a handful of regions would encode their accidents.
MIN_FIT_SAMPLES = 30


def fit(
    confidences: Sequence[float],
    outcomes: Sequence[bool],
    bins: int = DEFAULT_BINS,
    fitted_at: Optional[str] = None,
) -> Calibration:
    """Fit a temperature by minimising NLL, and measure what it achieved.

    Returns the identity map when the corpus is too small or the outcomes are
    all one class -- with no failures observed there is no evidence that any
    confidence was ever wrong, and nothing to calibrate against.
    """
    pairs = [
        (float(c), bool(o)) for c, o in zip(confidences, outcomes) if c is not None
    ]
    calibration = Calibration(bins=bins, samples=len(pairs), fitted_at=fitted_at)
    if not pairs:
        calibration.notes.append("no labelled regions; map is the identity")
        return calibration

    raw = [c for c, _ in pairs]
    labels = [o for _, o in pairs]
    calibration.ece_before = expected_calibration_error(raw, labels, bins)
    calibration.brier_before = brier_score(raw, labels)

    if len(pairs) < MIN_FIT_SAMPLES:
        calibration.notes.append(
            f"only {len(pairs)} labelled region(s); need {MIN_FIT_SAMPLES} to fit a "
            "temperature, so the map is the identity and the metrics above are "
            "descriptive only"
        )
        calibration.ece_after, calibration.brier_after = (
            calibration.ece_before, calibration.brier_before
        )
        return calibration
    if len(set(labels)) < 2:
        calibration.notes.append(
            "outcomes are all one class; nothing to calibrate against"
        )
        calibration.ece_after, calibration.brier_after = (
            calibration.ece_before, calibration.brier_before
        )
        return calibration

    # Coarse-to-fine scan rather than gradient descent: one bounded parameter,
    # a convex-enough objective, and no scipy dependency in this layer.
    best = 1.0
    low, high = 0.05, 10.0
    for _ in range(6):
        step = (high - low) / 40.0
        candidates = [low + step * i for i in range(41)]
        best = min(candidates, key=lambda t: _nll(raw, labels, t))
        low, high = max(0.05, best - step), min(10.0, best + step)
    calibration.temperature = round(best, 6)

    tempered = [calibration.apply(c) for c in raw]
    calibration.ece_after = expected_calibration_error(tempered, labels, bins)
    calibration.brier_after = brier_score(tempered, labels)
    if (calibration.ece_after or 0) > (calibration.ece_before or 0):
        # Temperature scaling minimises NLL, which does not guarantee a lower
        # BINNED ece; when it lands worse, keep the honest identity.
        calibration.notes.append(
            f"fitted T={calibration.temperature} raised binned ECE "
            f"({calibration.ece_before} -> {calibration.ece_after}); kept identity"
        )
        calibration.temperature = 1.0
        calibration.ece_after, calibration.brier_after = (
            calibration.ece_before, calibration.brier_before
        )
    return calibration


def load(path: Any) -> Calibration:
    """Load a fitted map, falling back to the identity when absent.

    Absence is the normal state before a corpus run, so it must never raise:
    an uncalibrated verifier reports raw confidence, which is exactly what it
    did before this module existed.
    """
    try:
        target = Path(path)
        if not target.is_file():
            return Calibration(notes=["no calibration artifact; using identity"])
        return Calibration.from_dict(json.loads(target.read_text(encoding="utf-8")))
    except Exception as exc:
        return Calibration(notes=[f"calibration unreadable ({type(exc).__name__}); using identity"])


def save(calibration: Calibration, path: Any) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(calibration.to_dict(), indent=2) + "\n", encoding="utf-8"
    )
