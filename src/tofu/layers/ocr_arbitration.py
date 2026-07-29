"""Conservative, engine-independent OCR candidate arbitration.

This module deliberately contains no OCR backend imports.  It converts each
backend's raw confidence through its own versioned monotonic calibration curve
before making a decision, and exposes every gate as structured evidence.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field
from enum import Enum
from typing import Mapping, Optional, Sequence


class DecisionKind(str, Enum):
    ACCEPT = "accept"
    AGREE = "agree"
    REVIEW = "review"
    REJECT = "reject"


class ReasonCode(str, Enum):
    CONSENSUS_TEXT = "consensus_text"
    ALTERNATE_ACCEPTED = "alternate_accepted"
    ALTERNATE_EMPTY = "alternate_empty"
    SAME_BACKEND = "same_backend"
    UNCALIBRATED_ENGINE = "uncalibrated_engine"
    RISKY_ALTERNATE = "risky_alternate"
    SCRIPT_INCOMPATIBLE = "script_incompatible"
    LANGUAGE_INCOMPATIBLE = "language_incompatible"
    INSUFFICIENT_INK_SUPPORT = "insufficient_ink_support"
    INSUFFICIENT_GEOMETRY_SUPPORT = "insufficient_geometry_support"
    BELOW_ABSOLUTE_FLOOR = "below_absolute_floor"
    INSUFFICIENT_CALIBRATED_MARGIN = "insufficient_calibrated_margin"
    CONFLICTING_TEXT = "conflicting_text"


@dataclass(frozen=True)
class OCRCandidate:
    """One OCR backend's result and the evidence needed to audit it."""

    engine: str
    engine_version: str
    text: str
    raw_confidence: float
    languages: tuple[str, ...] = ()
    script: Optional[str] = None
    language_distribution: Mapping[str, float] = field(default_factory=dict)
    risk_flags: tuple[str, ...] = ()
    provenance: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not 0.0 <= self.raw_confidence <= 1.0:
            raise ValueError("raw_confidence must be within [0, 1]")
        if any(value < 0.0 or value > 1.0 for value in self.language_distribution.values()):
            raise ValueError("language probabilities must be within [0, 1]")


@dataclass(frozen=True)
class CalibrationCurve:
    """A versioned monotonic piecewise-linear confidence calibration."""

    engine: str
    version: str
    points: tuple[tuple[float, float], ...]

    def __post_init__(self) -> None:
        if len(self.points) < 2:
            raise ValueError("calibration curve needs at least two points")
        previous_raw = previous_calibrated = -1.0
        for raw, calibrated in self.points:
            if not 0.0 <= raw <= 1.0 or not 0.0 <= calibrated <= 1.0:
                raise ValueError("calibration points must be within [0, 1]")
            if raw <= previous_raw:
                raise ValueError("raw calibration coordinates must strictly increase")
            if calibrated < previous_calibrated:
                raise ValueError("calibrated confidence must be monotonic")
            previous_raw, previous_calibrated = raw, calibrated

    def calibrate(self, raw_confidence: float) -> float:
        if not 0.0 <= raw_confidence <= 1.0:
            raise ValueError("raw_confidence must be within [0, 1]")
        if raw_confidence <= self.points[0][0]:
            return self.points[0][1]
        for (x0, y0), (x1, y1) in zip(self.points, self.points[1:]):
            if raw_confidence <= x1:
                ratio = (raw_confidence - x0) / (x1 - x0)
                return y0 + ratio * (y1 - y0)
        return self.points[-1][1]


@dataclass(frozen=True)
class CalibrationRegistry:
    version: str
    curves: Mapping[str, CalibrationCurve]

    def curve_for(self, candidate: OCRCandidate) -> Optional[CalibrationCurve]:
        return self.curves.get(candidate.engine.casefold())


# Conservative initial mappings.  They are intentionally versioned so held-out
# corpus calibration can replace them without changing arbitration semantics.
DEFAULT_CALIBRATION = CalibrationRegistry(
    version="ocr-confidence-1.0.0",
    curves={
        "easyocr": CalibrationCurve(
            engine="easyocr",
            version="1.0.0",
            points=((0.0, 0.0), (0.4, 0.28), (0.7, 0.62), (0.9, 0.86), (1.0, 0.98)),
        ),
        "paddleocr": CalibrationCurve(
            engine="paddleocr",
            version="1.0.0",
            points=((0.0, 0.0), (0.4, 0.24), (0.7, 0.58), (0.9, 0.84), (1.0, 0.97)),
        ),
    },
)


@dataclass(frozen=True)
class ArbitrationSignals:
    """Evidence independent of backend confidence."""

    ink_support: float
    geometry_support: float
    script_compatible: Optional[bool] = None
    language_compatible: Optional[bool] = None

    def __post_init__(self) -> None:
        if not 0.0 <= self.ink_support <= 1.0:
            raise ValueError("ink_support must be within [0, 1]")
        if not 0.0 <= self.geometry_support <= 1.0:
            raise ValueError("geometry_support must be within [0, 1]")


@dataclass(frozen=True)
class ArbitrationPolicy:
    version: str = "conservative-1.0.0"
    absolute_floor: float = 0.78
    calibrated_margin: float = 0.12
    minimum_ink_support: float = 0.55
    minimum_geometry_support: float = 0.65
    language_probability_floor: float = 0.35


@dataclass(frozen=True)
class CandidateEvidence:
    engine: str
    engine_version: str
    text: str
    raw_confidence: float
    calibrated_confidence: Optional[float]
    calibration_version: Optional[str]
    script: Optional[str]
    languages: tuple[str, ...]
    risk_flags: tuple[str, ...]
    provenance: Mapping[str, object]


@dataclass(frozen=True)
class ArbitrationDecision:
    kind: DecisionKind
    reason_codes: tuple[ReasonCode, ...]
    selected: str
    primary: CandidateEvidence
    alternate: CandidateEvidence
    policy_version: str
    calibration_registry_version: str
    signals: ArbitrationSignals


_NON_WORD = re.compile(r"[\W_]+", re.UNICODE)


def _normalized_text(text: str) -> str:
    return _NON_WORD.sub("", unicodedata.normalize("NFKC", text).casefold())


def _candidate_evidence(
    candidate: OCRCandidate, registry: CalibrationRegistry
) -> CandidateEvidence:
    curve = registry.curve_for(candidate)
    return CandidateEvidence(
        engine=candidate.engine,
        engine_version=candidate.engine_version,
        text=candidate.text,
        raw_confidence=candidate.raw_confidence,
        calibrated_confidence=curve.calibrate(candidate.raw_confidence) if curve else None,
        calibration_version=curve.version if curve else None,
        script=candidate.script,
        languages=candidate.languages,
        risk_flags=candidate.risk_flags,
        provenance=dict(candidate.provenance),
    )


def _script_compatible(
    primary: OCRCandidate, alternate: OCRCandidate, override: Optional[bool]
) -> bool:
    if override is not None:
        return override
    return not primary.script or not alternate.script or primary.script == alternate.script


def _probable_languages(candidate: OCRCandidate, floor: float) -> set[str]:
    probable = {
        language.casefold()
        for language, probability in candidate.language_distribution.items()
        if probability >= floor
    }
    return probable or {language.casefold() for language in candidate.languages}


def _language_compatible(
    primary: OCRCandidate,
    alternate: OCRCandidate,
    override: Optional[bool],
    floor: float,
) -> bool:
    if override is not None:
        return override
    primary_languages = _probable_languages(primary, floor)
    alternate_languages = _probable_languages(alternate, floor)
    return not primary_languages or not alternate_languages or bool(
        primary_languages & alternate_languages
    )


def _decision(
    kind: DecisionKind,
    reasons: Sequence[ReasonCode],
    selected: str,
    primary: CandidateEvidence,
    alternate: CandidateEvidence,
    policy: ArbitrationPolicy,
    registry: CalibrationRegistry,
    signals: ArbitrationSignals,
) -> ArbitrationDecision:
    return ArbitrationDecision(
        kind=kind,
        reason_codes=tuple(reasons),
        selected=selected,
        primary=primary,
        alternate=alternate,
        policy_version=policy.version,
        calibration_registry_version=registry.version,
        signals=signals,
    )


def arbitrate(
    primary: OCRCandidate,
    alternate: OCRCandidate,
    signals: ArbitrationSignals,
    *,
    policy: ArbitrationPolicy = ArbitrationPolicy(),
    calibration: CalibrationRegistry = DEFAULT_CALIBRATION,
) -> ArbitrationDecision:
    """Choose conservatively between two OCR candidates.

    ``accept`` is the only decision that replaces the primary. ``agree`` keeps
    the primary while recording independent consensus. ``review`` exposes a
    plausible conflict to a human. ``reject`` means the alternate failed a
    hard safety gate.
    """
    primary_evidence = _candidate_evidence(primary, calibration)
    alternate_evidence = _candidate_evidence(alternate, calibration)

    # Silence is never evidence that visible text should be removed.
    if not _normalized_text(alternate.text):
        return _decision(
            DecisionKind.REJECT,
            [ReasonCode.ALTERNATE_EMPTY],
            "primary",
            primary_evidence,
            alternate_evidence,
            policy,
            calibration,
            signals,
        )
    if primary.engine.casefold() == alternate.engine.casefold():
        return _decision(
            DecisionKind.REJECT,
            [ReasonCode.SAME_BACKEND],
            "primary",
            primary_evidence,
            alternate_evidence,
            policy,
            calibration,
            signals,
        )
    if (
        primary_evidence.calibrated_confidence is None
        or alternate_evidence.calibrated_confidence is None
    ):
        return _decision(
            DecisionKind.REVIEW,
            [ReasonCode.UNCALIBRATED_ENGINE],
            "primary",
            primary_evidence,
            alternate_evidence,
            policy,
            calibration,
            signals,
        )
    if _normalized_text(primary.text) == _normalized_text(alternate.text):
        return _decision(
            DecisionKind.AGREE,
            [ReasonCode.CONSENSUS_TEXT],
            "primary",
            primary_evidence,
            alternate_evidence,
            policy,
            calibration,
            signals,
        )
    if alternate.risk_flags:
        return _decision(
            DecisionKind.REJECT,
            [ReasonCode.RISKY_ALTERNATE],
            "primary",
            primary_evidence,
            alternate_evidence,
            policy,
            calibration,
            signals,
        )

    hard_failures: list[ReasonCode] = []
    if not _script_compatible(primary, alternate, signals.script_compatible):
        hard_failures.append(ReasonCode.SCRIPT_INCOMPATIBLE)
    if not _language_compatible(
        primary,
        alternate,
        signals.language_compatible,
        policy.language_probability_floor,
    ):
        hard_failures.append(ReasonCode.LANGUAGE_INCOMPATIBLE)
    if hard_failures:
        return _decision(
            DecisionKind.REJECT,
            hard_failures,
            "primary",
            primary_evidence,
            alternate_evidence,
            policy,
            calibration,
            signals,
        )

    review_reasons: list[ReasonCode] = []
    if signals.ink_support < policy.minimum_ink_support:
        review_reasons.append(ReasonCode.INSUFFICIENT_INK_SUPPORT)
    if signals.geometry_support < policy.minimum_geometry_support:
        review_reasons.append(ReasonCode.INSUFFICIENT_GEOMETRY_SUPPORT)
    if alternate_evidence.calibrated_confidence < policy.absolute_floor:
        review_reasons.append(ReasonCode.BELOW_ABSOLUTE_FLOOR)
    calibrated_delta = (
        alternate_evidence.calibrated_confidence - primary_evidence.calibrated_confidence
    )
    if calibrated_delta < policy.calibrated_margin:
        review_reasons.append(ReasonCode.INSUFFICIENT_CALIBRATED_MARGIN)
    if review_reasons:
        review_reasons.append(ReasonCode.CONFLICTING_TEXT)
        return _decision(
            DecisionKind.REVIEW,
            review_reasons,
            "primary",
            primary_evidence,
            alternate_evidence,
            policy,
            calibration,
            signals,
        )

    return _decision(
        DecisionKind.ACCEPT,
        [ReasonCode.ALTERNATE_ACCEPTED],
        "alternate",
        primary_evidence,
        alternate_evidence,
        policy,
        calibration,
        signals,
    )
