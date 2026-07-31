"""Tests for OCR observation clustering into region hypotheses.

Tests overlap merging, containment, adjacent-word separation, and
orientation mismatch — the geometric clustering that runs before
arbitration.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from tofu.core.types import BBox, OCRObservation
from tofu.layers.ocr_arbitration import form_region_hypotheses, RegionHypothesis


def _obs(oid: str, bbox: BBox, text: str = "TEST", backend: str = "easyocr") -> OCRObservation:
    return OCRObservation(
        observation_id=oid,
        backend=backend,
        backend_revision="1.0",
        pass_tag="p1",
        text=text,
        raw_confidence=0.8,
        bbox=bbox,
    )


class TestOverlapClustering:
    def test_high_iou_observations_merge(self):
        """Two observations with IoU >= 0.3 should join the same hypothesis."""
        obs = [
            _obs("o1", BBox(100, 100, 200, 50)),
            _obs("o2", BBox(110, 100, 200, 50)),
        ]
        hyps = form_region_hypotheses(obs)
        assert len(hyps) == 1
        assert len(hyps[0].member_ids) == 2

    def test_non_overlapping_observations_separate(self):
        """Two distant observations should form separate hypotheses."""
        obs = [
            _obs("o1", BBox(100, 100, 100, 40)),
            _obs("o2", BBox(500, 400, 100, 40)),
        ]
        hyps = form_region_hypotheses(obs)
        assert len(hyps) == 2

    def test_transitive_merge(self):
        """A overlaps B, B overlaps C → all three in one hypothesis."""
        obs = [
            _obs("o1", BBox(100, 100, 200, 50)),
            _obs("o2", BBox(200, 100, 200, 50)),
            _obs("o3", BBox(300, 100, 200, 50)),
        ]
        hyps = form_region_hypotheses(obs)
        assert len(hyps) == 1
        assert len(hyps[0].member_ids) == 3


class TestContainment:
    def test_contained_observations_merge(self):
        """One observation fully inside another should merge."""
        obs = [
            _obs("o1", BBox(100, 100, 300, 100)),
            _obs("o2", BBox(120, 110, 100, 40)),
        ]
        hyps = form_region_hypotheses(obs)
        assert len(hyps) == 1


class TestAdjacentWordSeparation:
    def test_adjacent_same_orientation_merge(self):
        """Adjacent boxes with same orientation and small gap should merge."""
        obs = [
            _obs("o1", BBox(100, 100, 100, 40)),
            _obs("o2", BBox(210, 100, 100, 40)),
        ]
        hyps = form_region_hypotheses(obs)
        assert len(hyps) == 1

    def test_distant_words_separate(self):
        """Boxes far apart should not merge even if same orientation."""
        obs = [
            _obs("o1", BBox(100, 100, 100, 40)),
            _obs("o2", BBox(500, 100, 100, 40)),
        ]
        hyps = form_region_hypotheses(obs)
        assert len(hyps) == 2


class TestOrientationMismatch:
    def test_adjacent_different_orientation_separate(self):
        """Adjacent boxes with different orientation should not merge."""
        obs = [
            _obs("o1", BBox(100, 100, 200, 40)),   # horizontal
            _obs("o2", BBox(310, 50, 40, 200)),     # vertical
        ]
        hyps = form_region_hypotheses(obs)
        assert len(hyps) == 2


class TestEmptyAndMax:
    def test_empty_observations(self):
        hyps = form_region_hypotheses([])
        assert hyps == []

    def test_max_proposals_limit(self):
        """Should cap at max_proposals hypotheses."""
        obs = [_obs(f"o{i}", BBox(i * 500, i * 500, 50, 50)) for i in range(20)]
        hyps = form_region_hypotheses(obs, max_proposals=5)
        assert len(hyps) == 5

    def test_errored_observations_still_clustered(self):
        """Errored observations still participate in geometry clustering."""
        obs = [
            OCRObservation("o1", "easyocr", "1.0", "p1", "", 0.0, bbox=BBox(100, 100, 200, 50), error="timeout"),
            _obs("o2", BBox(110, 100, 200, 50), backend="paddleocr"),
        ]
        hyps = form_region_hypotheses(obs)
        assert len(hyps) == 1
