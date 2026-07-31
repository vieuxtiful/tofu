"""Tests for OCR verification states: agree, disagree, no_text, unavailable, error.

Tests that PaddleRegionVerifier produces the correct state for each
scenario, and that the states propagate correctly through
build_hypothesis_decision.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from tofu.layers.ocr_verification import (
    OCRVerificationResult,
    PaddleRegionVerifier,
    RegionRequest,
    normalize_for_ocr_agreement,
    text_similarity,
)


class TestTextSimilarity:
    def test_identical_text(self):
        assert text_similarity("HELLO", "HELLO") == 1.0

    def test_case_insensitive_non_cjk(self):
        assert text_similarity("Hello", "HELLO") == 1.0

    def test_case_sensitive_cjk(self):
        # CJK should not casefold
        sim = text_similarity("ＨＥＬＬＯ", "hello", language="ja")
        assert sim < 1.0

    def test_empty_both(self):
        assert text_similarity("", "") == 1.0

    def test_empty_one(self):
        assert text_similarity("HELLO", "") == 0.0

    def test_partial_match(self):
        sim = text_similarity("HELLO WORLD", "HELLO")
        assert 0.0 < sim < 1.0


class TestNormalize:
    def test_whitespace_collapsed(self):
        assert normalize_for_ocr_agreement("  hello   world  ") == "hello world"

    def test_casefold_non_cjk(self):
        assert normalize_for_ocr_agreement("HELLO", "en") == "hello"

    def test_no_casefold_cjk(self):
        # Case carries no meaning in Japanese script, but the Latin runs inside
        # Japanese text keep theirs -- a trademark read as "SONY" must not be
        # folded to "sony" here.  (This used to assert on a FULLWIDTH string,
        # which NFKC folds to ASCII before the casefold branch is ever reached,
        # so it tested compatibility folding rather than the property it names.)
        assert normalize_for_ocr_agreement("HELLO", "ja") == "HELLO"
        assert normalize_for_ocr_agreement("HELLO", "zh-cn") == "HELLO"

    def test_compatibility_forms_fold_for_every_language(self):
        # Deliberate, and shared with verify._normalize and ocr_arbitration:
        # this is a comparison key for AGREEMENT, never stored text.  Two
        # engines reading one glyph as fullwidth and halfwidth agree about the
        # content, and folding is what stops that becoming a false disagreement.
        assert normalize_for_ocr_agreement("ＨＥＬＬＯ", "ja") == "HELLO"
        assert normalize_for_ocr_agreement("ﾊﾛｰ", "ja") == "ハロー"
        assert normalize_for_ocr_agreement("１２３", "ja") == "123"


class TestVerificationStates:
    def test_agree_state(self):
        result = OCRVerificationResult(
            state="agree", text="HELLO", confidence=0.9, similarity=0.95,
        )
        assert result.state == "agree"
        evidence = result.evidence()
        assert evidence["state"] == "agree"
        assert evidence["similarity"] == 0.95

    def test_disagree_state(self):
        result = OCRVerificationResult(
            state="disagree", text="WORLD", confidence=0.7, similarity=0.2,
        )
        assert result.state == "disagree"

    def test_no_text_state(self):
        result = OCRVerificationResult(state="no_text", similarity=0.0)
        assert result.state == "no_text"
        assert result.text == ""

    def test_unavailable_state(self):
        result = OCRVerificationResult(state="unavailable")
        assert result.state == "unavailable"
        assert result.confidence == 0.0

    def test_error_state(self):
        result = OCRVerificationResult(state="error", error="ImportError: paddleocr")
        assert result.state == "error"
        assert "ImportError" in result.error


class TestPaddleRegionVerifier:
    def test_unavailable_returns_unavailable(self):
        """When PaddleOCR is not installed, all regions return unavailable."""
        if PaddleRegionVerifier.available():
            pytest.skip("PaddleOCR is installed; skipping unavailable test")
        verifier = PaddleRegionVerifier()
        import numpy as np
        crop = np.zeros((100, 200, 3), dtype=np.uint8)
        from tofu.core.types import BBox
        results = verifier.verify_regions(
            crop, [BBox(0, 0, 200, 100)], expected_texts=["TEST"],
        )
        assert len(results) == 1
        assert results[0].state == "unavailable"


class TestRegionRequest:
    def test_region_request_fields(self):
        from tofu.core.types import BBox
        req = RegionRequest(
            crop=None,
            bbox=BBox(0, 0, 100, 50),
            polygon=[(0, 0), (100, 0), (100, 50), (0, 50)],
            language_hint="en",
            source_text="HELLO",
        )
        assert req.bbox.width == 100
        assert req.source_text == "HELLO"
        assert req.language_hint == "en"
