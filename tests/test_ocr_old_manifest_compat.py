"""Tests for backward compatibility: old manifests without new OCR fields.

Old manifests that lack ocr_provenance, OCRObservation, and
OCRHypothesisDecision fields must still load and process correctly.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from tofu.core.types import BBox, InstText, TextManifest
from tofu.utils.manifest_store import _manifest_to_dict, _dict_to_manifest


def _make_old_manifest() -> TextManifest:
    """Create a manifest with no ocr_provenance or new OCR fields."""
    return TextManifest(
        asset_id="test-asset",
        total_regions=2,
        instances=[
            InstText(
                id="r1",
                bounding_box=BBox(100, 100, 200, 50),
                text="HELLO",
                confidence=0.9,
            ),
            InstText(
                id="r2",
                bounding_box=BBox(100, 200, 200, 50),
                text="WORLD",
                confidence=0.85,
            ),
        ],
        src_lang="en",
    )


class TestOldManifestRoundTrip:
    def test_old_manifest_serializes(self):
        manifest = _make_old_manifest()
        d = _manifest_to_dict(manifest)
        assert "instances" in d
        for inst in d["instances"]:
            # ocr_provenance may be None but should not crash
            assert inst.get("ocr_provenance") is None

    def test_old_manifest_deserializes(self):
        manifest = _make_old_manifest()
        d = _manifest_to_dict(manifest)
        restored = _dict_to_manifest(d)
        assert restored.total_regions == 2
        assert restored.instances[0].text == "HELLO"
        assert restored.instances[0].ocr_provenance is None

    def test_old_manifest_ocr_provenance_none(self):
        """Old manifests have ocr_provenance=None and that's fine."""
        manifest = _make_old_manifest()
        for inst in manifest.instances:
            assert inst.ocr_provenance is None
            assert inst.recognition_history is None

    def test_manifest_with_ocr_provenance_roundtrips(self):
        """New manifests with ocr_provenance data round-trip correctly."""
        manifest = _make_old_manifest()
        manifest.instances[0].ocr_provenance = {
            "hypothesis_id": "rh-1",
            "selected_text": "HELLO",
            "verification_state": "agree",
            "auto_accepted": True,
            "observations": [
                {"observation_id": "o1", "backend": "easyocr", "text": "HELLO"},
            ],
        }
        d = _manifest_to_dict(manifest)
        restored = _dict_to_manifest(d)
        assert restored.instances[0].ocr_provenance is not None
        assert restored.instances[0].ocr_provenance["hypothesis_id"] == "rh-1"
        assert restored.instances[0].ocr_provenance["selected_text"] == "HELLO"

    def test_json_roundtrip_preserves_none(self):
        """JSON serialization preserves None for absent new fields."""
        manifest = _make_old_manifest()
        d = _manifest_to_dict(manifest)
        json_str = json.dumps(d, ensure_ascii=False)
        restored_d = json.loads(json_str)
        restored = _dict_to_manifest(restored_d)
        assert restored.instances[0].ocr_provenance is None
        assert restored.instances[1].ocr_provenance is None
