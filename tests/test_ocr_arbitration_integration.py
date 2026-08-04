"""Accuracy regressions at the Cicerone/arbitration/persistence boundary."""

from tofu.core.types import (
    BBox,
    InstText,
    OCRAssessmentPolicy,
    SceneRegion,
    TextManifest,
)
from tofu.layers.cicerone import assess_multi_candidate_ocr, _promote_verifier_confusion
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
    # The budget rations VERIFIER round trips -- the clean region costs none,
    # which is what the assertion above pins. It does not ration weighing the
    # readings the pipeline already produced: that is pure Python over
    # recognition_history, so the clean region is graded too, and carries
    # `verified: false` to keep "corroborated by a second engine" distinct
    # from "consistent with itself".
    assert clean.ocr_provenance["hypothesis"]["verified"] is False
    assert "review_required" not in clean.ocr_provenance
    assert all(
        inst.ocr_provenance and not inst.ocr_provenance["review_required"]
        for inst in (weak, textured, vertical)
    )
    assert all(
        inst.ocr_provenance["hypothesis"]["verified"] is True
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


def test_exhaustive_mode_lifts_the_proposal_budget(monkeypatch):
    """'exhaustive' has to mean no budget, not a different label on one.

    The mode used to append a risk flag and then truncate anyway, so the
    ninth risky region recorded "risk-based verification budget exhausted"
    under a policy that had asked for the opposite. Nine regions because
    max_region_proposals defaults to eight -- the ninth is the one that can
    tell the two modes apart.
    """
    def risky(index):
        return InstText(
            f"r{index}", BBox(0, index * 25, 80, 20), text=f"WORD{index}",
            confidence=0.20, detected_language="en", reading_order=index,
        )

    monkeypatch.setattr(
        PaddleRegionVerifier,
        "verify_regions",
        lambda _self, _asset, regions, expected_texts, language=None: [
            OCRVerificationResult("agree", text=text, confidence=0.9, similarity=1.0)
            for text in expected_texts
        ],
    )

    def budget_exhausted(inst):
        return any(
            entry.get("reason") == "risk-based verification budget exhausted"
            for entry in (inst.recognition_history or [])
        )

    rationed = [risky(i) for i in range(9)]
    assess_multi_candidate_ocr(
        "asset", rationed, policy=OCRAssessmentPolicy(mode="risk_based")
    )
    assert sum(budget_exhausted(inst) for inst in rationed) == 1

    unrationed = [risky(i) for i in range(9)]
    assess_multi_candidate_ocr(
        "asset", unrationed, policy=OCRAssessmentPolicy(mode="exhaustive")
    )
    assert not any(budget_exhausted(inst) for inst in unrationed)
    assert all(
        inst.ocr_provenance["hypothesis"]["verified"] is True
        for inst in unrationed
    )


class TestVerifierConfusionPromotion:
    """Arbitration's reading reaches the manifest when the difference is a glyph.

    The pairwise decision withholds acceptance whenever the independent
    verifier disagrees with the primary, on the reasoning that a
    disagreement is usually two engines failing differently. That same flag
    fires when the verifier is simply right. Measured on decolonisons r2:
    primary "nos rues 4", verifier "nos rues!", arbitration selects the
    verifier's observation, geometry 1.0, glyph 1.0 -- and the region
    shipped "4".
    """

    def _record(self, **overrides):
        record = {
            "selected_observation_id": "r2-verifier",
            "selected_text": "nos rues!",
            "geometry_score": 1.0,
            "score_breakdown": {"glyph": 1.0},
            "agrees_with_pairwise": False,
            "reason_codes": ["verification_disagree"],
        }
        record.update(overrides)
        return record

    def _inst(self, text="nos rues 4"):
        return InstText("r2", BBox(0, 0, 100, 40), text=text, detected_language="fr")

    def test_a_confused_glyph_is_promoted(self):
        inst = self._inst()
        assert _promote_verifier_confusion(inst, self._record(), "nos rues 4") is True
        assert inst.text == "nos rues!"
        assert inst.ocr_correction["applied"] is True
        assert inst.ocr_correction["original_text"] == "nos rues 4"
        assert inst.recognition_history[-1]["stage"] == "hypothesis_promotion"

    def test_whitespace_alone_never_blocks_the_comparison(self):
        """The mark the recogniser set tight is the difference at issue.

        "nos rues 4" and "nos rues!" differ in length as strings; compared
        without whitespace they differ in exactly one glyph. Spacing is a
        separate, locale-aware pass.
        """
        inst = self._inst()
        assert _promote_verifier_confusion(inst, self._record(), "nos rues 4") is True

    def test_a_different_word_is_never_promoted(self):
        inst = self._inst("nos rues")
        record = self._record(selected_text="vos rues")
        assert _promote_verifier_confusion(inst, record, "nos rues") is False
        assert inst.text == "nos rues"

    def test_an_added_character_is_never_promoted(self):
        inst = self._inst("nos rues")
        record = self._record(selected_text="nos rues!!")
        assert _promote_verifier_confusion(inst, record, "nos rues") is False

    def test_imperfect_geometry_is_never_promoted(self):
        # the two readings have to describe the same ink in the same place
        inst = self._inst()
        record = self._record(geometry_score=0.5)
        assert _promote_verifier_confusion(inst, record, "nos rues 4") is False

    def test_imperfect_glyph_evidence_is_never_promoted(self):
        inst = self._inst()
        record = self._record(score_breakdown={"glyph": 0.5})
        assert _promote_verifier_confusion(inst, record, "nos rues 4") is False

    def test_a_reading_the_primary_engine_produced_is_never_promoted(self):
        """It has to be a SECOND engine, not a re-reading of the same one."""
        inst = self._inst()
        record = self._record(selected_observation_id="r2-primary")
        assert _promote_verifier_confusion(inst, record, "nos rues 4") is False

    def test_agreement_is_nothing_to_promote(self):
        inst = self._inst()
        record = self._record(agrees_with_pairwise=True)
        assert _promote_verifier_confusion(inst, record, "nos rues 4") is False

    def test_a_missing_hypothesis_is_handled(self):
        assert _promote_verifier_confusion(self._inst(), None, "nos rues 4") is False

    def test_the_disagreement_flag_is_cleared_once_promoted(self):
        """The record was written against the text the pairwise pass chose.

        Leaving the flag set would have the workspace mark this region as a
        disagreement forever, against a reading it no longer ships.
        """
        inst = self._inst()
        record = self._record()
        assert _promote_verifier_confusion(inst, record, "nos rues 4") is True
        assert record["agrees_with_pairwise"] is True
        assert record["promoted"] is True
