"""Accuracy regressions at the Cicerone/arbitration/persistence boundary."""

from tofu.core.types import (
    BBox,
    InstText,
    OCRAssessmentPolicy,
    SceneRegion,
    TextManifest,
)
from tofu.layers.cicerone import assess_multi_candidate_ocr
from tofu.layers.ocr_arbitration import (
    ArbitrationSignals,
    CalibrationCurve,
    CalibrationRegistry,
    DecisionKind,
    OCRCandidate,
    ReasonCode,
    arbitrate,
)
from tofu.layers.ocr_verification import OCRVerificationResult, PaddleRegionVerifier
from tofu.utils.manifest_store import _dict_to_manifest, _manifest_to_dict


def candidate(
    engine,
    text,
    confidence,
    *,
    script="Latin",
    languages=("fr",),
    distribution=None,
    risks=(),
    provenance=None,
):
    return OCRCandidate(
        engine=engine,
        engine_version="fixture-1",
        text=text,
        raw_confidence=confidence,
        script=script,
        languages=languages,
        language_distribution=distribution or {},
        risk_flags=risks,
        provenance=provenance or {},
    )


SAFE_VISUALS = ArbitrationSignals(ink_support=0.92, geometry_support=0.94)


def test_cross_engine_safety_matrix_minimizes_false_positive_replacements():
    primary = candidate("easyocr", "RUE", 0.42)
    cases = [
        (
            candidate("paddleocr", "RUE", 0.91),
            SAFE_VISUALS,
            DecisionKind.AGREE,
            "primary",
            ReasonCode.CONSENSUS_TEXT,
        ),
        (
            candidate("paddleocr", "AVENUE", 0.65),
            SAFE_VISUALS,
            DecisionKind.REVIEW,
            "primary",
            ReasonCode.BELOW_ABSOLUTE_FLOOR,
        ),
        (
            candidate("paddleocr", "RÃ‰PUBLIQUE", 0.99),
            SAFE_VISUALS,
            DecisionKind.ACCEPT,
            "alternate",
            ReasonCode.ALTERNATE_ACCEPTED,
        ),
        (
            candidate("paddleocr", "Ð£Ð›Ð˜Ð¦Ð", 0.99, script="Cyrillic"),
            SAFE_VISUALS,
            DecisionKind.REJECT,
            "primary",
            ReasonCode.SCRIPT_INCOMPATIBLE,
        ),
        (
            candidate("paddleocr", "", 1.0),
            SAFE_VISUALS,
            DecisionKind.REJECT,
            "primary",
            ReasonCode.ALTERNATE_EMPTY,
        ),
        (
            candidate("paddleocr", "7", 0.99, risks=("isolated-symbol",)),
            SAFE_VISUALS,
            DecisionKind.REJECT,
            "primary",
            ReasonCode.RISKY_ALTERNATE,
        ),
    ]

    for alternate, signals, kind, selected, reason in cases:
        decision = arbitrate(primary, alternate, signals)
        assert decision.kind is kind
        assert decision.selected == selected
        assert reason in decision.reason_codes


def test_higher_raw_alternate_abstains_when_engine_calibration_is_worse():
    calibration = CalibrationRegistry(
        version="held-out-fixture",
        curves={
            "easyocr": CalibrationCurve(
                "easyocr", "fixture-a", ((0.0, 0.0), (1.0, 0.95))
            ),
            "paddleocr": CalibrationCurve(
                "paddleocr", "fixture-b", ((0.0, 0.0), (1.0, 0.45))
            ),
        },
    )

    decision = arbitrate(
        candidate("easyocr", "PRIMARY", 0.72),
        candidate("paddleocr", "ALTERNATE", 0.99),
        SAFE_VISUALS,
        calibration=calibration,
    )

    assert decision.kind is DecisionKind.REVIEW
    assert decision.selected == "primary"
    assert decision.alternate.raw_confidence > decision.primary.raw_confidence
    assert (
        decision.alternate.calibrated_confidence
        < decision.primary.calibrated_confidence
    )


def test_candidate_provenance_survives_decision_and_manifest_round_trip():
    alternate = candidate(
        "paddleocr",
        "RÃ‰PUBLIQUE",
        0.99,
        provenance={
            "pass": 2,
            "model": "fixture-model",
            "crop": {"x": 4, "y": 5, "width": 80, "height": 20},
        },
    )
    decision = arbitrate(
        candidate("easyocr", "REPUBLIQUE", 0.30),
        alternate,
        SAFE_VISUALS,
    )
    assert decision.alternate.provenance == alternate.provenance

    inst = InstText("r1", BBox(4, 5, 80, 20), text=decision.alternate.text)
    inst.ocr_provenance = {
        "selected": decision.selected,
        "decision": decision.kind.value,
        "reason_codes": [reason.value for reason in decision.reason_codes],
        "alternate": {
            "engine": decision.alternate.engine,
            "raw_confidence": decision.alternate.raw_confidence,
            "provenance": dict(decision.alternate.provenance),
        },
    }
    manifest = TextManifest("asset", 1, [inst])

    restored = _dict_to_manifest(_manifest_to_dict(manifest)).instances[0]

    assert restored.ocr_provenance == inst.ocr_provenance
    assert restored.ocr_provenance["alternate"]["provenance"]["pass"] == 2


def test_risk_based_assessment_only_spends_alternate_reads_on_risky_regions(
    monkeypatch,
):
    clean = InstText("clean", BBox(0, 0, 100, 20), text="CLEAR", confidence=0.95)
    weak = InstText("weak", BBox(0, 30, 100, 20), text="FAINT", confidence=0.40)
    textured = InstText(
        "textured", BBox(0, 60, 100, 20), text="WALL", confidence=0.93
    )
    vertical = InstText(
        "vertical", BBox(120, 0, 15, 80), text="VERT", confidence=0.93
    )
    surface = SceneRegion(
        BBox(0, 55, 110, 35), "surface", 0.9, texture="textured"
    )
    observed = {}

    def verify(_self, _asset, regions, expected_texts, language=None):
        observed["texts"] = list(expected_texts)
        return [
            OCRVerificationResult(
                "agree", text=text, confidence=0.9, similarity=1.0
            )
            for text in expected_texts
        ]

    monkeypatch.setattr(PaddleRegionVerifier, "verify_regions", verify)

    verified = assess_multi_candidate_ocr(
        "asset",
        [clean, weak, textured, vertical],
        [surface],
        OCRAssessmentPolicy(mode="risk_based"),
    )

    assert observed["texts"] == ["FAINT", "WALL", "VERT"]
    # Consensus is verified evidence but does not count as a source-text
    # mutation; the return value deliberately reports replacements only.
    assert verified == 0
    assert clean.ocr_provenance is None
    assert all(
        inst.ocr_provenance and not inst.ocr_provenance["review_required"]
        for inst in (weak, textured, vertical)
    )


def test_assessment_applies_only_a_high_margin_visually_supported_correction(
    monkeypatch,
):
    inst = InstText(
        "r1", BBox(0, 0, 80, 20), text="REPUBLIQUE", confidence=0.20,
        detected_language="fr",
    )

    def verify(_self, _asset, regions, expected_texts, language=None):
        return [OCRVerificationResult(
            "disagree",
            text="RÃ‰PUBLIQUE",
            confidence=1.0,
            similarity=0.9,
            detections=[{
                "text": "RÃ‰PUBLIQUE",
                "confidence": 1.0,
                "polygon": [[0, 0], [80, 0], [80, 20], [0, 20]],
            }],
        )]

    monkeypatch.setattr(PaddleRegionVerifier, "verify_regions", verify)
    monkeypatch.setattr(
        "tofu.layers.cicerone._has_ink_support", lambda asset, bbox: True
    )

    changed = assess_multi_candidate_ocr(
        "asset", [inst], policy=OCRAssessmentPolicy(mode="risk_based")
    )

    assert changed == 1
    assert inst.text == "RÃ‰PUBLIQUE"
    assert inst.ocr_provenance["decision"] == "accept"
    assert inst.ocr_provenance["selected_backend"] == "alternate"
    assert inst.ocr_correction["applied"] is True


def test_assessment_no_text_alternate_never_deletes_primary(monkeypatch):
    inst = InstText(
        "r1", BBox(0, 0, 80, 20), text="VISIBLE", confidence=0.20,
        detected_language="en",
    )
    monkeypatch.setattr(
        PaddleRegionVerifier,
        "verify_regions",
        lambda *_args, **_kwargs: [
            OCRVerificationResult("no_text", text="", confidence=1.0)
        ],
    )
    monkeypatch.setattr(
        "tofu.layers.cicerone._has_ink_support", lambda asset, bbox: True
    )

    changed = assess_multi_candidate_ocr("asset", [inst])

    assert changed == 0
    assert inst.text == "VISIBLE"
    assert inst.ocr_provenance["decision"] == "reject"
    assert "alternate_empty" in inst.ocr_provenance["reason_codes"]
