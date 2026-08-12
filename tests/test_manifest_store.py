## 🍢 manifest persistence round-trip
import pytest

from tofu.core.types import (
    BBox, BgProfil, CharactText, InstText, Mask, SceneRegion, StyleProfil,
    TextManifest,
)
from tofu.utils.manifest_store import (
    load_manifest, save_manifest, _dict_to_manifest, _manifest_to_dict,
)


def full_manifest() -> TextManifest:
    inst = InstText(
        id="r1",
        bounding_box=BBox(x=10, y=20, width=100, height=40),
        adjusted_bbox=BBox(x=18, y=13, width=100, height=40),
        segmentation_mask=Mask(polygon=[(10, 20), (110, 20), (110, 60), (10, 60)],
                               confidence=0.87),
        text="居酒屋", target_text="izakaya", language="ja",
        confidence=0.87, detected_language="ja", reading_order=0,
        dnt=False, target_language="en",
        style_profile=StyleProfil(font_family="msgothic.ttc", font_weight="Bold",
                                  color="#ffffff", italic=True, font_size=42,
                                  stroke_color="#000000", stroke_width=2.0,
                                  tsume=0.3, underline_offset=3.5,
                                  underline_width=1.5),
        background_profile=BgProfil(semantic_label="panel", texture="flat",
                                    dominant_color="#b42828", material="painted sign / panel"),
        characteristics=CharactText(font_style="gothic-bold", size=42),
        repair_provenance={
            "requested_provider": "lama", "executed_provider": "telea_fallback",
            "confidence": .3, "review_required": True,
        },
        ocr_quality={
            "state": "review_required", "reasons": ["component_count_mismatch"],
            "estimated_glyph_height": 7.0, "component_surplus": 1,
        },
        font_match={
            "schema": 1, "provider": "local_glyph_retrieval", "status": "review",
            "confidence": .71, "margin": .04, "source_text": "å±…é…’å±‹",
            "candidates": [{"family": "Example", "available": True, "license": "installed", "score": .72}],
            "recommended_substitute": {"font_path": "example.ttf", "family": "Example", "score": .72},
        },
        material_evidence={
            "schema": 1, "material_class": "painted_panel",
            "decision_eligible": False,
        },
    )
    return TextManifest(
        asset_id="asset-1", total_regions=1, instances=[inst],
        src_lang="ja", targ_lang="en", img_dim=(960, 640),
        asset_class="sign",
        asset_classification={
            "asset_class": "sign", "confidence": .9, "source": "cicerone_layout",
        },
        scene_regions=[SceneRegion(
            bbox=BBox(x=0, y=0, width=200, height=300),
            semantic_label="panel", confidence=0.9,
            background_color="#b42828", border_detected=True,
            material="painted sign / panel",
            material_evidence={
                "schema": 1, "material_class": "painted_panel",
                "decision_eligible": False,
            },
            polygon=[(0, 0), (200, 0), (200, 300), (0, 300)],
        )],
    )


class TestRoundTrip:
    def test_dict_round_trip_preserves_everything(self):
        m = full_manifest()
        m2 = _dict_to_manifest(_manifest_to_dict(m))
        i, i2 = m.instances[0], m2.instances[0]
        assert i2.text == i.text and i2.target_text == i.target_text
        assert i2.style_profile.tsume == 0.3
        assert i2.style_profile.italic is True
        assert i2.style_profile.underline_offset == 3.5
        assert i2.style_profile.underline_width == 1.5
        assert i2.background_profile.semantic_label == "panel"
        assert i2.background_profile.material == "painted sign / panel"
        assert i2.characteristics.font_style == "gothic-bold"
        assert i2.segmentation_mask.polygon == i.segmentation_mask.polygon
        assert i2.repair_provenance == i.repair_provenance
        assert i2.ocr_quality == i.ocr_quality
        assert i2.font_match == i.font_match
        assert i2.material_evidence == i.material_evidence
        assert i2.bounding_box == BBox(x=10, y=20, width=100, height=40)
        assert i2.adjusted_bbox == BBox(x=18, y=13, width=100, height=40)
        assert m2.img_dim == (960, 640)
        assert m2.scene_regions[0].semantic_label == "panel"
        assert m2.scene_regions[0].material == "painted sign / panel"
        assert m2.scene_regions[0].material_evidence["decision_eligible"] is False
        assert m2.scene_regions[0].polygon[0] == (0, 0)
        assert m2.asset_class == "sign"
        assert m2.asset_classification["confidence"] == .9

    def test_disk_round_trip(self, tmp_path):
        m = full_manifest()
        save_manifest(tmp_path, "asset-1", m)
        m2 = load_manifest(tmp_path, "asset-1")
        assert m2 is not None
        assert m2.instances[0].style_profile.font_weight == "Bold"

    def test_missing_file_returns_none(self, tmp_path):
        assert load_manifest(tmp_path, "nope") is None

    def test_suite_guard_refuses_an_invalid_direct_save(self, tmp_path):
        damaged = TextManifest(
            asset_id="bad", total_regions=1,
            instances=[InstText(id="r4", bounding_box=BBox(0, 0, 10, 10))],
        )
        with pytest.raises(AssertionError, match="not contiguous"):
            save_manifest(tmp_path, "bad", damaged)
        assert not (tmp_path / "bad.manifest.json").exists()
