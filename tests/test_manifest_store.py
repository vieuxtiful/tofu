## 🍢 manifest persistence round-trip
import json

import pytest
from server.config import ConfigError, load_settings

from tofu.core.types import (
    BBox,
    BgProfil,
    CharactText,
    InstText,
    Mask,
    SceneRegion,
    StyleProfil,
    TextManifest,
)
from tofu.core.vision2 import (
    ArtifactRevision,
    CandidateLedger,
    CandidateProposal,
    FusionEvidence,
    GlyphCandidateEvidence,
    GlyphMatchEvidence,
    Vision2Config,
    Vision2Decision,
    Vision2State,
)
from tofu.layers import decant
from tofu.layers.fusion import (
    FEATURE_SCHEMA,
    assess_manifest,
    evaluate,
    load_calibration,
)
from tofu.utils.manifest_store import (
    SERVER_OWNED_INSTANCE_FIELDS,
    _dict_to_manifest,
    _manifest_to_dict,
    load_manifest,
    merge_server_owned,
    save_manifest,
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
        surface_observation={
            "surface_id": "surface-abc", "observation_revision": "scene-material-observation-v1",
            "material_class": "painted_panel", "calibration_status": "unfitted",
            "substrate_trust": "unmeasured", "missing_features": ["glyph_excluded_substrate"],
            "decision_weight": 0.0,
        },
        lineage_candidate_id="node-r1",
        glyph_match_evidence=GlyphMatchEvidence(
            schema="glyph-match-v1", decision=Vision2Decision.SHADOW,
            candidates=[GlyphCandidateEvidence(
                text="居酒屋", rank=1, support=.91, source="ground_truth",
                lineage_candidate_id="node-r1",
            )],
            evidence_survival_state="present", supported_domain=True,
            revisions=ArtifactRevision(
                feature_schema="features-v1", model="encoder-v1",
                calibration="calibration-v1", corpus="corpus-v1",
                candidate_pool="pool-v1",
            ),
            lineage_candidate_id="node-r1",
        ),
        fusion_evidence=FusionEvidence(
            schema="fusion-v1", decision=Vision2Decision.REVIEW_REQUIRED,
            reason_codes=["shadow_only"], lineage_candidate_id="node-r1",
        ),
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
            surface_observation={
                "surface_id": "surface-abc", "observation_revision": "scene-material-observation-v1",
                "material_class": "painted_panel", "calibration_status": "unfitted",
                "substrate_trust": "unmeasured", "missing_features": ["glyph_excluded_substrate"],
                "decision_weight": 0.0,
            },
            polygon=[(0, 0), (200, 0), (200, 300), (0, 300)],
        )],
        vision2_candidate_ledger=CandidateLedger(
            schema="vision2-candidate-ledger-v1", state=Vision2State.SHADOW,
            proposals=[CandidateProposal(
                candidate_id="node-r1", source_stage="raw_craft",
                geometry=(10, 20, 100, 40), text="居酒屋", confidence=.87,
            )],
            total_candidates=1, retained_candidates=1, max_candidates=256,
            source_counts={"raw_craft": 1}, suppression_counts={"active": 1},
        ),
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
        assert i2.surface_observation == i.surface_observation
        assert i2.glyph_match_evidence == i.glyph_match_evidence
        assert i2.fusion_evidence == i.fusion_evidence
        assert i2.bounding_box == BBox(x=10, y=20, width=100, height=40)
        assert i2.adjusted_bbox == BBox(x=18, y=13, width=100, height=40)
        assert m2.img_dim == (960, 640)
        assert m2.scene_regions[0].semantic_label == "panel"
        assert m2.scene_regions[0].material == "painted sign / panel"
        assert m2.scene_regions[0].material_evidence["decision_eligible"] is False
        assert m2.scene_regions[0].surface_observation == m.scene_regions[0].surface_observation
        assert m2.scene_regions[0].polygon[0] == (0, 0)
        assert m2.asset_class == "sign"
        assert m2.asset_classification["confidence"] == .9
        assert m2.vision2_candidate_ledger == m.vision2_candidate_ledger

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


class TestVision2Contracts:
    def test_default_configuration_is_inert(self):
        config = Vision2Config()
        assert config.state is Vision2State.OFF
        assert config.enabled is False
        assert config.may_surface_recommendations is False

    def test_server_configuration_defaults_off(self, monkeypatch):
        monkeypatch.delenv("TOFU_VISION2_STATE", raising=False)
        assert load_settings().vision2.state is Vision2State.OFF

    def test_server_configuration_accepts_shadow(self, monkeypatch):
        monkeypatch.setenv("TOFU_VISION2_STATE", "shadow")
        assert load_settings().vision2.state is Vision2State.SHADOW

    def test_server_configuration_rejects_unknown_state(self, monkeypatch):
        monkeypatch.setenv("TOFU_VISION2_STATE", "green")
        with pytest.raises(ConfigError, match="TOFU_VISION2_STATE"):
            load_settings()

    def test_unknown_decision_fails_closed_on_load(self):
        data = _manifest_to_dict(full_manifest())
        data["instances"][0]["glyph_match_evidence"]["decision"] = "future_green"
        restored = _dict_to_manifest(data)
        evidence = restored.instances[0].glyph_match_evidence
        assert evidence is not None
        assert evidence.decision is Vision2Decision.NOT_EVALUATED

    def test_artifact_revision_mismatches_are_explicit(self):
        actual = ArtifactRevision(
            feature_schema="features-v2", model="model-v2", calibration="cal-v1",
            corpus="corpus-v1", candidate_pool="pool-v2",
        )
        expected = ArtifactRevision(
            feature_schema="features-v1", model="model-v1", calibration="cal-v1",
            corpus="corpus-v1", candidate_pool="pool-v1",
        )
        assert actual.mismatches(expected) == ["feature_schema", "model", "candidate_pool"]

    @pytest.mark.parametrize("field_name", SERVER_OWNED_INSTANCE_FIELDS)
    def test_client_omission_preserves_every_server_owned_instance_field(self, field_name):
        manifest = full_manifest()
        stored_value = _manifest_to_dict(manifest)["instances"][0][field_name]
        incoming = {
            "asset_id": manifest.asset_id,
            "instances": [{
                "id": "r1",
                "bounding_box": {"x": 10, "y": 20, "width": 100, "height": 40},
                "text": "居酒屋",
            }],
        }
        merged = merge_server_owned(incoming, manifest)
        assert merged["instances"][0][field_name] == stored_value

    def test_instance_ownership_is_joined_by_id_not_array_position(self):
        first = InstText(
            id="r1", bounding_box=BBox(0, 0, 10, 10),
            review_features={"owner": "r1"},
        )
        second = InstText(
            id="r2", bounding_box=BBox(20, 0, 10, 10),
            review_features={"owner": "r2"},
        )
        stored = TextManifest(asset_id="a1", total_regions=2, instances=[first, second])
        incoming = {
            "asset_id": "a1",
            "instances": [
                {"id": "r2", "bounding_box": {"x": 20, "y": 0, "width": 10, "height": 10}},
                {"id": "r1", "bounding_box": {"x": 0, "y": 0, "width": 10, "height": 10}},
            ],
        }
        merged = merge_server_owned(incoming, stored)
        assert [item["review_features"]["owner"] for item in merged["instances"]] == ["r2", "r1"]


class TestVision2EvidenceSurvival:
    def test_strict_missing_segmentation_is_unknown_not_absent(self):
        inst = InstText(
            id="r1", bounding_box=BBox(0, 0, 60, 30), text="SORTIE", confidence=.9,
        )
        verdict = decant.survival(inst, strict=True)
        assert verdict["state"] == "unknown"
        assert verdict["reasons"] == ["segmentation_not_measured"]

    def test_no_detector_activation_remains_absent(self):
        inst = InstText(
            id="r1", bounding_box=BBox(0, 0, 60, 30), text="SORTIE", confidence=.9,
            segmentation_mask=Mask(polygon=[(0, 0), (60, 0), (60, 30), (0, 30)], confidence=.9),
            review_features={"craft_region": .02},
        )
        assert decant.survival(inst, strict=True)["state"] == "absent"

    def test_small_glyphs_are_weak(self):
        inst = InstText(
            id="r1", bounding_box=BBox(0, 0, 60, 30), text="SORTIE", confidence=.9,
            segmentation_mask=Mask(polygon=[(0, 0), (60, 0), (60, 30), (0, 30)], confidence=.9),
            ocr_quality={"estimated_glyph_height": 7.0},
        )
        verdict = decant.survival(inst, strict=True)
        assert verdict["state"] == "weak"
        assert "glyph_height_below_legible" in verdict["reasons"]

    def test_component_stability_is_measured_without_becoming_a_score(self):
        inst = InstText(
            id="r1", bounding_box=BBox(0, 0, 60, 30), text="SORTIE", confidence=.9,
            segmentation_mask=Mask(polygon=[(0, 0), (60, 0), (60, 30), (0, 30)], confidence=.8),
            ocr_quality={"raw_component_count": 6, "base_component_count": 5},
        )
        verdict = decant.survival(inst, strict=True)
        assert verdict["measured"]["component_stability"] == pytest.approx(5 / 6, abs=1e-4)
        assert "score" not in verdict and "probability" not in verdict

    def test_survival_contract_is_non_decision_bearing(self):
        inst = InstText(
            id="r1", bounding_box=BBox(0, 0, 60, 30), text="SORTIE", confidence=.9,
            segmentation_mask=Mask(polygon=[(0, 0), (60, 0), (60, 30), (0, 30)], confidence=.9),
        )
        verdict = decant.survival(inst, strict=True)
        assert verdict["policy_revision"] == decant.POLICY_REVISION
        assert verdict["decision_eligible"] is False


def _fusion_instance(survival="present", margin=.2):
    return InstText(
        id="r1", bounding_box=BBox(0, 0, 20, 20), lineage_candidate_id="incumbent",
        evidence_survival={"state": survival, "reasons": [], "measured": {},
                           "schema": 1, "policy_revision": "test",
                           "decision_eligible": False},
        glyph_match_evidence=GlyphMatchEvidence(
            schema="proof-runtime-v1", decision=Vision2Decision.SHADOW,
            candidates=[GlyphCandidateEvidence("A", 1, .9, "test", margin=margin),
                        GlyphCandidateEvidence("B", 2, .4, "test")],
            supported_domain=True,
            revisions=ArtifactRevision(feature_schema="proof-runtime-v1",
                                       model="model-1", candidate_pool="pool-1"),
        ),
    )


def _fusion_artifact(tmp_path, **updates):
    data = {
        "schema": "vision2-fusion-calibration-v1",
        "retrieval_feature_schema": "proof-runtime-v1",
        "revision": {"feature_schema": FEATURE_SCHEMA,
                     "model": "model-1", "calibration": "cal-1",
                     "corpus": "corpus-1", "candidate_pool": "pool-1"},
        "coefficients": {"retrieval_support": 8.0}, "intercept": -2.0,
        "recommendation_threshold": .9, "rejection_threshold": .1,
        "minimum_margin": .1, "maximum_contradiction": .25,
    }
    data.update(updates)
    path = tmp_path / "fusion.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    return path


class TestFusionDecisionContract:
    def test_shadow_and_survival_gates_precede_recommendation(self, tmp_path):
        calibration, error = load_calibration(str(_fusion_artifact(tmp_path)))
        shadow = evaluate(_fusion_instance(), Vision2State.SHADOW, calibration, error)
        assert shadow.decision is Vision2Decision.SHADOW
        assert shadow.calibrated_probability is None
        for state in ("weak", "absent"):
            result = evaluate(_fusion_instance(state), Vision2State.AUTO_GT_RECOMMEND,
                              calibration, error)
            assert result.decision is Vision2Decision.UNRESOLVABLE
        for state in ("partial", "unknown"):
            result = evaluate(_fusion_instance(state), Vision2State.AUTO_GT_RECOMMEND,
                              calibration, error)
            assert result.decision is Vision2Decision.REVIEW_REQUIRED

    def test_exact_artifact_recommends_without_mutating_source(self, tmp_path):
        calibration, error = load_calibration(str(_fusion_artifact(tmp_path)))
        inst = _fusion_instance()
        inst.text = "incumbent OCR"
        result = evaluate(inst, Vision2State.AUTO_GT_RECOMMEND, calibration, error)
        assert result.decision is Vision2Decision.RECOMMENDED
        assert result.calibrated_probability >= .9
        assert inst.text == "incumbent OCR"

    def test_mismatch_missing_feature_and_contradiction_fail_closed(self, tmp_path):
        calibration, error = load_calibration(str(_fusion_artifact(tmp_path)))
        inst = _fusion_instance()
        inst.glyph_match_evidence.revisions.model = "wrong"
        result = evaluate(inst, Vision2State.AUTO_GT_RECOMMEND, calibration, error)
        assert "artifact_mismatch:model" in result.reason_codes
        calibration, error = load_calibration(str(_fusion_artifact(
            tmp_path, coefficients={"component_alignment_cost": 1.0})))
        result = evaluate(_fusion_instance(), Vision2State.AUTO_GT_RECOMMEND,
                          calibration, error)
        assert "feature_missing:component_alignment_cost" in result.reason_codes
        calibration, error = load_calibration(str(_fusion_artifact(tmp_path)))
        inst = _fusion_instance()
        inst.glyph_match_evidence.candidates[0].contradiction = .9
        result = evaluate(inst, Vision2State.AUTO_GT_RECOMMEND, calibration, error)
        assert "contradiction_above_calibrated_maximum" in result.reason_codes

    def test_reconsideration_is_at_most_one_and_provenance_only(self, tmp_path):
        inst = _fusion_instance()
        inst.text = "incumbent"
        manifest = TextManifest(
            asset_id="a", total_regions=1, instances=[inst],
            vision2_candidate_ledger=CandidateLedger(
                schema="v1", state=Vision2State.AUTO_GT_RECOMMEND,
                proposals=[CandidateProposal(candidate_id="retained",
                    source_stage="raw", geometry=(1, 1, 10, 10),
                    glyph_match_evidence=_fusion_instance().glyph_match_evidence)]))
        config = Vision2Config(state=Vision2State.AUTO_GT_RECOMMEND,
                               fusion_calibration_path=str(_fusion_artifact(tmp_path)))
        assess_manifest(manifest, config)
        assess_manifest(manifest, config)
        assert inst.fusion_evidence.reconsideration_count == 1
        assert inst.fusion_evidence.reconsideration_candidate_id == "retained"
        assert inst.text == "incumbent"
