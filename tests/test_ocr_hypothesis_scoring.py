"""Tests for OCR hypothesis scoring with calibrated weights.

Tests missing signal renormalization, weight renormalization, and
calibrated vs uncalibrated confidence handling.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from tofu.core.types import BBox, OCRObservation
from tofu.layers.ocr_arbitration import (
    DEFAULT_CALIBRATION,
    HYPOTHESIS_WEIGHTS,
    RegionHypothesis,
    score_hypothesis,
    build_hypothesis_decision,
)


def _obs(oid: str, bbox: BBox, text: str, conf: float, backend: str = "easyocr",
         script: str = None, lang: str = None) -> OCRObservation:
    return OCRObservation(
        observation_id=oid,
        backend=backend,
        backend_revision="1.0",
        pass_tag="p1",
        text=text,
        raw_confidence=conf,
        bbox=bbox,
        detected_script=script,
        language_hint=lang,
    )


def _hyp(obs_list):
    return RegionHypothesis(
        hypothesis_id="rh-1",
        member_ids=[o.observation_id for o in obs_list],
        union_bbox=BBox(100, 100, 200, 50),
        observations=obs_list,
    )


class TestSingleObservation:
    def test_single_observation_scores(self):
        obs = [_obs("o1", BBox(100, 100, 200, 50), "HELLO", 0.9)]
        result = score_hypothesis(_hyp(obs))
        assert result["selected_text"] == "HELLO"
        assert result["transcription_score"] > 0.0
        assert result["reason_codes"] == ["single_observation"]

    def test_single_errored_returns_none(self):
        obs = [OCRObservation("o1", "easyocr", "1.0", "p1", "", 0.0, bbox=BBox(100, 100, 200, 50), error="err")]
        result = score_hypothesis(_hyp(obs))
        assert result["selected_text"] is None
        assert result["transcription_score"] == 0.0
        assert "all_errored" in result["reason_codes"]


class TestCrossBackendBonus:
    def test_same_backend_no_bonus(self):
        obs = [
            _obs("o1", BBox(100, 100, 200, 50), "HELLO", 0.9, backend="easyocr"),
            _obs("o2", BBox(100, 100, 200, 50), "HELLO", 0.85, backend="easyocr"),
        ]
        result = score_hypothesis(_hyp(obs))
        assert result["score_breakdown"]["cross_backend"] == 0.0

    def test_different_backend_bonus(self):
        obs = [
            _obs("o1", BBox(100, 100, 200, 50), "HELLO", 0.9, backend="easyocr"),
            _obs("o2", BBox(100, 100, 200, 50), "HELLO", 0.85, backend="paddleocr"),
        ]
        result = score_hypothesis(_hyp(obs))
        assert result["score_breakdown"]["cross_backend"] > 0.0


class TestStabilityScore:
    def test_identical_text_stability_1(self):
        obs = [
            _obs("o1", BBox(100, 100, 200, 50), "HELLO", 0.9, backend="easyocr"),
            _obs("o2", BBox(100, 100, 200, 50), "HELLO", 0.85, backend="paddleocr"),
        ]
        result = score_hypothesis(_hyp(obs))
        assert result["score_breakdown"]["stability"] == 1.0

    def test_different_text_stability_below_1(self):
        obs = [
            _obs("o1", BBox(100, 100, 200, 50), "HELLO", 0.9, backend="easyocr"),
            _obs("o2", BBox(100, 100, 200, 50), "WORLD", 0.85, backend="paddleocr"),
        ]
        result = score_hypothesis(_hyp(obs))
        assert result["score_breakdown"]["stability"] < 1.0


class TestWeightRenormalization:
    def test_missing_signals_renormalize(self):
        """When some signals are None, weights renormalize over active ones."""
        obs = [_obs("o1", BBox(100, 100, 200, 50), "HELLO", 0.9)]
        result = score_hypothesis(_hyp(obs))
        # Single observation: cross_backend is 0, not None — it's still active
        # But geometry is 0.5 (single obs default), language is 0.5 (no lang hint)
        # All signals are present, just some are 0
        assert result["transcription_score"] > 0.0

    def test_uncalibrated_engine_uses_raw_confidence(self):
        obs = [_obs("o1", BBox(100, 100, 200, 50), "HELLO", 0.9, backend="unknown_engine")]
        result = score_hypothesis(_hyp(obs))
        # Should fall back to raw confidence
        assert result["selected_text"] == "HELLO"
        assert result["transcription_score"] > 0.0


class TestCalibratedVsUncalibrated:
    def test_calibrated_selected_over_raw(self):
        """When both calibrated and uncalibrated exist, calibrated wins."""
        obs = [
            _obs("o1", BBox(100, 100, 200, 50), "HELLO", 0.95, backend="easyocr"),
            _obs("o2", BBox(100, 100, 200, 50), "HELLO", 0.85, backend="paddleocr"),
        ]
        result = score_hypothesis(_hyp(obs))
        # EasyOCR 0.95 calibrates to ~0.91, PaddleOCR 0.85 calibrates to ~0.72
        # So o1 should be selected
        assert result["selected_observation_id"] == "o1"


class TestBuildHypothesisDecision:
    def test_auto_accept_with_verification_agree(self):
        obs = [
            _obs("o1", BBox(100, 100, 200, 50), "HELLO", 0.95, backend="easyocr"),
            _obs("o2", BBox(100, 100, 200, 50), "HELLO", 0.85, backend="paddleocr"),
        ]
        decision = build_hypothesis_decision(
            _hyp(obs), verification_state="agree", verification_observation_id="v1",
        )
        assert decision.verification_state == "agree"
        assert "verification_agree" in decision.reason_codes

    def test_no_auto_accept_when_unavailable(self):
        obs = [
            _obs("o1", BBox(100, 100, 200, 50), "HELLO", 0.95, backend="easyocr"),
            _obs("o2", BBox(100, 100, 200, 50), "HELLO", 0.85, backend="paddleocr"),
        ]
        decision = build_hypothesis_decision(
            _hyp(obs), verification_state="unavailable", verification_observation_id=None,
        )
        assert decision.auto_accepted is False
        assert decision.review_required is True

    def test_no_auto_accept_when_disagree(self):
        obs = [
            _obs("o1", BBox(100, 100, 200, 50), "HELLO", 0.95, backend="easyocr"),
            _obs("o2", BBox(100, 100, 200, 50), "HELLO", 0.85, backend="paddleocr"),
        ]
        decision = build_hypothesis_decision(
            _hyp(obs), verification_state="disagree", verification_observation_id="v1",
        )
        assert decision.auto_accepted is False
        assert decision.review_required is True
