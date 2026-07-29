import pytest

from tofu.layers.ocr_arbitration import (
    ArbitrationPolicy,
    ArbitrationSignals,
    CalibrationCurve,
    CalibrationRegistry,
    DecisionKind,
    OCRCandidate,
    ReasonCode,
    arbitrate,
)


def _candidate(
    engine,
    text,
    confidence,
    *,
    script="Latin",
    languages=("fr",),
    risks=(),
):
    return OCRCandidate(
        engine=engine,
        engine_version="test",
        text=text,
        raw_confidence=confidence,
        script=script,
        languages=languages,
        risk_flags=risks,
    )


def _signals(**overrides):
    values = {"ink_support": 0.9, "geometry_support": 0.9}
    values.update(overrides)
    return ArbitrationSignals(**values)


def test_curve_is_monotonic_and_interpolates():
    curve = CalibrationCurve("test", "1.0.0", ((0.0, 0.0), (0.5, 0.2), (1.0, 1.0)))
    values = [curve.calibrate(raw / 20) for raw in range(21)]

    assert values == sorted(values)
    assert curve.calibrate(0.75) == pytest.approx(0.6)


@pytest.mark.parametrize(
    "points",
    [
        ((0.0, 0.0), (0.0, 0.5)),
        ((0.0, 0.5), (1.0, 0.4)),
        ((0.0, 0.0), (1.1, 1.0)),
    ],
)
def test_curve_rejects_invalid_or_nonmonotonic_points(points):
    with pytest.raises(ValueError):
        CalibrationCurve("test", "1.0.0", points)


def test_accepts_only_calibrated_independent_well_supported_alternate():
    decision = arbitrate(
        _candidate("easyocr", "Republique", 0.45),
        _candidate("paddleocr", "République", 0.98),
        _signals(),
    )

    assert decision.kind is DecisionKind.ACCEPT
    assert decision.selected == "alternate"
    assert decision.reason_codes == (ReasonCode.ALTERNATE_ACCEPTED,)
    assert decision.primary.calibration_version == "1.0.0"
    assert decision.alternate.calibrated_confidence > decision.primary.calibrated_confidence


def test_raw_confidences_are_not_compared_without_per_engine_calibration():
    registry = CalibrationRegistry(
        version="test",
        curves={
            "easyocr": CalibrationCurve("easyocr", "a", ((0.0, 0.0), (1.0, 0.9))),
            # Higher raw confidence deliberately calibrates much lower.
            "paddleocr": CalibrationCurve("paddleocr", "b", ((0.0, 0.0), (1.0, 0.5))),
        },
    )
    decision = arbitrate(
        _candidate("easyocr", "primary", 0.70),
        _candidate("paddleocr", "alternate", 0.99),
        _signals(),
        calibration=registry,
    )

    assert decision.kind is DecisionKind.REVIEW
    assert ReasonCode.BELOW_ABSOLUTE_FLOOR in decision.reason_codes
    assert ReasonCode.INSUFFICIENT_CALIBRATED_MARGIN in decision.reason_codes


def test_same_normalized_text_records_agreement_without_replacement():
    decision = arbitrate(
        _candidate("easyocr", "RÉPUBLIQUE!", 0.8),
        _candidate("paddleocr", "république", 0.8),
        _signals(),
    )

    assert decision.kind is DecisionKind.AGREE
    assert decision.selected == "primary"


def test_silence_can_never_remove_primary_text():
    decision = arbitrate(
        _candidate("easyocr", "Visible", 0.2),
        _candidate("paddleocr", " \n", 1.0),
        _signals(),
    )

    assert decision.kind is DecisionKind.REJECT
    assert decision.reason_codes == (ReasonCode.ALTERNATE_EMPTY,)
    assert decision.selected == "primary"


def test_same_backend_is_not_independent_evidence():
    decision = arbitrate(
        _candidate("easyocr", "one", 0.1),
        _candidate("EASYOCR", "two", 1.0),
        _signals(),
    )

    assert decision.kind is DecisionKind.REJECT
    assert ReasonCode.SAME_BACKEND in decision.reason_codes


def test_unknown_calibration_requires_review():
    decision = arbitrate(
        _candidate("easyocr", "one", 0.1),
        _candidate("new-engine", "two", 1.0),
        _signals(),
    )

    assert decision.kind is DecisionKind.REVIEW
    assert ReasonCode.UNCALIBRATED_ENGINE in decision.reason_codes


def test_risk_flags_block_auto_replacement():
    decision = arbitrate(
        _candidate("easyocr", "one", 0.1),
        _candidate("paddleocr", "two", 1.0, risks=("proper-name-rewrite",)),
        _signals(),
    )

    assert decision.kind is DecisionKind.REJECT
    assert ReasonCode.RISKY_ALTERNATE in decision.reason_codes


@pytest.mark.parametrize(
    ("alternate_kwargs", "signal_kwargs", "reason"),
    [
        ({"script": "Cyrillic"}, {}, ReasonCode.SCRIPT_INCOMPATIBLE),
        ({"languages": ("pl",)}, {}, ReasonCode.LANGUAGE_INCOMPATIBLE),
        ({}, {"script_compatible": False}, ReasonCode.SCRIPT_INCOMPATIBLE),
        ({}, {"language_compatible": False}, ReasonCode.LANGUAGE_INCOMPATIBLE),
    ],
)
def test_script_and_language_incompatibility_reject(
    alternate_kwargs, signal_kwargs, reason
):
    primary = _candidate("easyocr", "one", 0.1)
    alternate = _candidate("paddleocr", "two", 1.0, **alternate_kwargs)

    decision = arbitrate(primary, alternate, _signals(**signal_kwargs))

    assert decision.kind is DecisionKind.REJECT
    assert reason in decision.reason_codes


@pytest.mark.parametrize(
    ("signals", "reason"),
    [
        (_signals(ink_support=0.2), ReasonCode.INSUFFICIENT_INK_SUPPORT),
        (_signals(geometry_support=0.2), ReasonCode.INSUFFICIENT_GEOMETRY_SUPPORT),
    ],
)
def test_missing_positive_visual_evidence_requires_review(signals, reason):
    decision = arbitrate(
        _candidate("easyocr", "one", 0.1),
        _candidate("paddleocr", "two", 1.0),
        signals,
    )

    assert decision.kind is DecisionKind.REVIEW
    assert reason in decision.reason_codes


def test_absolute_floor_and_margin_are_independent_gates():
    policy = ArbitrationPolicy(absolute_floor=0.9, calibrated_margin=0.5)
    decision = arbitrate(
        _candidate("easyocr", "one", 0.8),
        _candidate("paddleocr", "two", 0.9),
        _signals(),
        policy=policy,
    )

    assert decision.kind is DecisionKind.REVIEW
    assert ReasonCode.BELOW_ABSOLUTE_FLOOR in decision.reason_codes
    assert ReasonCode.INSUFFICIENT_CALIBRATED_MARGIN in decision.reason_codes


def test_language_distributions_take_precedence_over_declared_reader_languages():
    primary = OCRCandidate(
        engine="easyocr",
        engine_version="test",
        text="one",
        raw_confidence=0.1,
        languages=("en", "fr"),
        language_distribution={"fr": 0.9, "en": 0.1},
        script="Latin",
    )
    alternate = OCRCandidate(
        engine="paddleocr",
        engine_version="test",
        text="dwa",
        raw_confidence=1.0,
        languages=("en", "fr"),
        language_distribution={"pl": 0.9, "en": 0.1},
        script="Latin",
    )

    decision = arbitrate(primary, alternate, _signals())

    assert decision.kind is DecisionKind.REJECT
    assert ReasonCode.LANGUAGE_INCOMPATIBLE in decision.reason_codes


def test_decision_contains_policy_calibration_and_raw_evidence():
    primary = OCRCandidate(
        engine="easyocr",
        engine_version="test",
        text="one",
        raw_confidence=0.2,
        languages=("fr",),
        script="Latin",
        provenance={"pass": 2, "threshold": 0.5},
    )
    decision = arbitrate(
        primary,
        _candidate("paddleocr", "two", 1.0),
        _signals(),
    )

    assert decision.policy_version == "conservative-1.0.0"
    assert decision.calibration_registry_version == "ocr-confidence-1.0.0"
    assert decision.primary.raw_confidence == 0.2
    assert decision.alternate.raw_confidence == 1.0
    assert decision.primary.provenance == {"pass": 2, "threshold": 0.5}
