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
from typing import Any, Dict, List, Mapping, Optional, Sequence

from tofu.core.types import BBox, OCRObservation, OCRHypothesisDecision, Polygon


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
    # Keep case and combining marks intact.  They are semantic OCR evidence:
    # ``RÉPUBLIQUE`` vs ``Republique`` must not be laundered into an
    # "independent consensus" merely because both strings normalize to the
    # same lower-case token.  Whitespace/punctuation remain non-semantic.
    return _NON_WORD.sub("", unicodedata.normalize("NFKC", text))


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


# ---------------------------------------------------------------------------
# Region hypothesis formation and multi-candidate scoring
# ---------------------------------------------------------------------------

_IOU_THRESHOLD = 0.3
_CONTAINMENT_THRESHOLD = 0.6
_ADJACENT_GAP_PX = 15

# Calibrated weights for transcription hypothesis scoring
HYPOTHESIS_WEIGHTS: Dict[str, float] = {
    "cross_backend": 0.30,
    "confidence": 0.20,
    "stability": 0.20,
    "geometry": 0.15,
    "language": 0.10,
    "glyph": 0.05,
}


def _bbox_iou(a: BBox, b: BBox) -> float:
    ax0, ay0 = a.x, a.y
    ax1, ay1 = a.x + a.width, a.y + a.height
    bx0, by0 = b.x, b.y
    bx1, by1 = b.x + b.width, b.y + b.height
    ix0, iy0 = max(ax0, bx0), max(ay0, by0)
    ix1, iy1 = min(ax1, bx1), min(ay1, by1)
    if ix1 <= ix0 or iy1 <= iy0:
        return 0.0
    inter = (ix1 - ix0) * (iy1 - iy0)
    union = a.width * a.height + b.width * b.height - inter
    return inter / union if union > 0 else 0.0


def _bbox_containment(inner: BBox, outer: BBox) -> float:
    ix0, iy0 = max(inner.x, outer.x), max(inner.y, outer.y)
    ix1 = min(inner.x + inner.width, outer.x + outer.width)
    iy1 = min(inner.y + inner.height, outer.y + outer.height)
    if ix1 <= ix0 or iy1 <= iy0:
        return 0.0
    inter = (ix1 - ix0) * (iy1 - iy0)
    inner_area = inner.width * inner.height
    return inter / inner_area if inner_area > 0 else 0.0


def _are_adjacent(a: BBox, b: BBox, gap: int = _ADJACENT_GAP_PX) -> bool:
    ax0, ay0, ax1, ay1 = a.x, a.y, a.x + a.width, a.y + a.height
    bx0, by0, bx1, by1 = b.x, b.y, b.x + b.width, b.y + b.height
    horizontal_gap = max(0, max(bx0 - ax1, ax0 - bx1))
    vertical_gap = max(0, max(by0 - ay1, ay0 - by1))
    return horizontal_gap <= gap and vertical_gap <= gap


def _orientation_match(a: BBox, b: BBox) -> bool:
    a_vert = a.height > a.width * 1.5
    b_vert = b.height > b.width * 1.5
    return a_vert == b_vert


@dataclass
class RegionHypothesis:
    """A cluster of OCR observations that likely correspond to one text region."""
    hypothesis_id: str
    member_ids: List[str]
    union_bbox: BBox
    observations: List[OCRObservation]


def form_region_hypotheses(
    observations: Sequence[OCRObservation],
    *,
    iou_threshold: float = _IOU_THRESHOLD,
    containment_threshold: float = _CONTAINMENT_THRESHOLD,
    adjacent_gap: int = _ADJACENT_GAP_PX,
    max_proposals: int = 8,
) -> List[RegionHypothesis]:
    """Cluster raw OCR observations into region hypotheses.

    Two observations join the same hypothesis when their bboxes overlap
    (IoU >= threshold), one contains the other (containment >= threshold),
    or they are adjacent with matching orientation.  Union-Find is used so
    transitive merges propagate correctly.
    """
    if not observations:
        return []
    n = len(observations)
    parent = list(range(n))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(x: int, y: int) -> None:
        rx, ry = find(x), find(y)
        if rx != ry:
            parent[rx] = ry

    for i in range(n):
        for j in range(i + 1, n):
            oi, oj = observations[i], observations[j]
            iou = _bbox_iou(oi.bbox, oj.bbox)
            cont_ij = _bbox_containment(oi.bbox, oj.bbox)
            cont_ji = _bbox_containment(oj.bbox, oi.bbox)
            adjacent = _are_adjacent(oi.bbox, oj.bbox, adjacent_gap)
            orient_ok = _orientation_match(oi.bbox, oj.bbox)
            if (
                iou >= iou_threshold
                or cont_ij >= containment_threshold
                or cont_ji >= containment_threshold
                or (adjacent and orient_ok)
            ):
                union(i, j)

    clusters: Dict[int, List[int]] = {}
    for i in range(n):
        root = find(i)
        clusters.setdefault(root, []).append(i)

    hypotheses: List[RegionHypothesis] = []
    for idx, (root, members) in enumerate(sorted(clusters.items())):
        if len(hypotheses) >= max_proposals:
            break
        member_obs = [observations[i] for i in members]
        xs = [o.bbox.x for o in member_obs]
        ys = [o.bbox.y for o in member_obs]
        x1s = [o.bbox.x + o.bbox.width for o in member_obs]
        y1s = [o.bbox.y + o.bbox.height for o in member_obs]
        union_bbox = BBox(
            x=min(xs), y=min(ys),
            width=max(x1s) - min(xs),
            height=max(y1s) - min(ys),
        )
        hypotheses.append(RegionHypothesis(
            hypothesis_id=f"rh-{idx + 1}",
            member_ids=[o.observation_id for o in member_obs],
            union_bbox=union_bbox,
            observations=member_obs,
        ))
    return hypotheses


def _calibrate_observation(
    obs: OCRObservation, registry: CalibrationRegistry
) -> Optional[float]:
    curve = registry.curves.get(obs.backend.casefold())
    if curve is None:
        return None
    return curve.calibrate(obs.raw_confidence)


def _cross_backend_bonus(observations: List[OCRObservation]) -> float:
    backends = {o.backend.casefold() for o in observations if not o.error}
    if len(backends) <= 1:
        return 0.0
    return min(1.0, (len(backends) - 1) * 0.5)


def _stability_score(observations: List[OCRObservation]) -> float:
    texts = [_normalized_text(o.text) for o in observations if not o.error and o.text]
    if not texts:
        return 0.0
    from collections import Counter
    counts = Counter(texts)
    most_common_count = counts.most_common(1)[0][1]
    return most_common_count / len(texts)


def _geometry_score(observations: List[OCRObservation]) -> float:
    if len(observations) <= 1:
        return 0.5
    ious = []
    for i in range(len(observations)):
        for j in range(i + 1, len(observations)):
            ious.append(_bbox_iou(observations[i].bbox, observations[j].bbox))
    if not ious:
        return 0.5
    return sum(ious) / len(ious)


def _language_consistency(observations: List[OCRObservation]) -> float:
    langs = {o.language_hint for o in observations if o.language_hint}
    if not langs:
        return 0.5
    return 1.0 if len(langs) == 1 else 0.3


def _glyph_evidence(observations: List[OCRObservation]) -> float:
    scripts = {o.detected_script for o in observations if o.detected_script}
    if not scripts:
        return 0.5
    return 1.0 if len(scripts) == 1 else 0.3


def score_hypothesis(
    hypothesis: RegionHypothesis,
    *,
    calibration: CalibrationRegistry = DEFAULT_CALIBRATION,
    weights: Dict[str, float] = HYPOTHESIS_WEIGHTS,
) -> Dict[str, Any]:
    """Score a region hypothesis and select the best observation.

    Returns a dict with:
    - selected_observation_id
    - selected_text
    - transcription_score (weighted sum)
    - geometry_score
    - score_breakdown (per-signal values)
    - reason_codes
    """
    obs = hypothesis.observations
    valid = [o for o in obs if not o.error]
    if not valid:
        return {
            "selected_observation_id": None,
            "selected_text": None,
            "transcription_score": 0.0,
            "geometry_score": 0.0,
            "score_breakdown": {k: None for k in weights},
            "reason_codes": ["all_errored"],
        }

    cross = _cross_backend_bonus(valid)
    stability = _stability_score(valid)
    geometry = _geometry_score(valid)
    language = _language_consistency(valid)
    glyph = _glyph_evidence(valid)

    calibrated_confs = []
    for o in valid:
        c = _calibrate_observation(o, calibration)
        o.calibrated_confidence = c
        if c is not None:
            calibrated_confs.append((c, o))

    if calibrated_confs:
        best_calibrated = max(calibrated_confs, key=lambda pair: pair[0])
        confidence_signal = best_calibrated[0]
        selected = best_calibrated[1]
    else:
        confidence_signal = max(o.raw_confidence for o in valid)
        selected = max(valid, key=lambda o: o.raw_confidence)

    available_weights = {
        "cross_backend": cross,
        "confidence": confidence_signal,
        "stability": stability,
        "geometry": geometry,
        "language": language,
        "glyph": glyph,
    }

    # Renormalize weights when some signals are missing
    active_weights = {k: v for k, v in weights.items() if available_weights.get(k) is not None}
    total_weight = sum(active_weights.values())
    if total_weight <= 0:
        total_weight = 1.0

    score_breakdown: Dict[str, Optional[float]] = {}
    weighted_sum = 0.0
    for signal_name, weight in weights.items():
        value = available_weights.get(signal_name)
        if value is not None and signal_name in active_weights:
            normalized_weight = active_weights[signal_name] / total_weight
            weighted_sum += value * normalized_weight
            score_breakdown[signal_name] = round(value, 4)
        else:
            score_breakdown[signal_name] = None

    reason_codes: List[str] = []
    if cross > 0:
        reason_codes.append("cross_backend_consensus")
    if stability >= 0.8:
        reason_codes.append("text_stable")
    if geometry >= 0.5:
        reason_codes.append("geometry_aligned")
    if not reason_codes:
        reason_codes.append("single_observation")

    return {
        "selected_observation_id": selected.observation_id,
        "selected_text": selected.text,
        "transcription_score": round(weighted_sum, 4),
        "geometry_score": round(geometry, 4),
        "score_breakdown": score_breakdown,
        "reason_codes": reason_codes,
        "selected_observation": selected,
    }


def build_hypothesis_decision(
    hypothesis: RegionHypothesis,
    verification_state: str,
    verification_observation_id: Optional[str],
    *,
    calibration: CalibrationRegistry = DEFAULT_CALIBRATION,
    weights: Dict[str, float] = HYPOTHESIS_WEIGHTS,
    policy_revision: str = "ocr-v1",
    auto_accept_threshold: float = 0.75,
    require_verification_for_accept: bool = True,
) -> OCRHypothesisDecision:
    """Build an OCRHypothesisDecision from a scored hypothesis + verification."""
    scored = score_hypothesis(hypothesis, calibration=calibration, weights=weights)
    selected_id = scored["selected_observation_id"]
    selected_text = scored["selected_text"]
    trans_score = scored["transcription_score"]
    geom_score = scored["geometry_score"]

    auto_accepted = trans_score >= auto_accept_threshold
    if require_verification_for_accept:
        if verification_state in ("unavailable", "error", "no_text"):
            auto_accepted = False
        elif verification_state == "disagree":
            auto_accepted = False

    review_required = not auto_accepted

    reason_codes = list(scored["reason_codes"])
    if verification_state == "agree":
        reason_codes.append("verification_agree")
    elif verification_state == "disagree":
        reason_codes.append("verification_disagree")
    elif verification_state == "unavailable":
        reason_codes.append("verification_unavailable")
    elif verification_state == "no_text":
        reason_codes.append("verification_no_text")
    elif verification_state == "error":
        reason_codes.append("verification_error")

    return OCRHypothesisDecision(
        region_id=hypothesis.hypothesis_id,
        member_observation_ids=list(hypothesis.member_ids),
        selected_observation_id=selected_id,
        selected_text=selected_text,
        geometry_score=geom_score,
        transcription_score=trans_score,
        verification_state=verification_state,
        verification_observation_id=verification_observation_id,
        auto_accepted=auto_accepted,
        review_required=review_required,
        reason_codes=reason_codes,
        score_breakdown=scored["score_breakdown"],
        policy_revision=policy_revision,
    )
