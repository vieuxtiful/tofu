import numpy as np

from tofu.core.types import BBox, BgProfil, InstText, ReconstructionProfile, SceneRegion
from tofu.layers import inpaint_providers
from tofu.layers.ocr_verification import (
    OCRVerificationResult,
    PaddleRegionVerifier,
    normalize_for_ocr_agreement,
    text_similarity,
)


def test_ocr_normalization_is_unicode_and_script_aware():
    assert normalize_for_ocr_agreement("Ａ  B", "en") == "a b"
    assert text_similarity("東京 駅", "東京 駅", "ja") == 1.0
    assert text_similarity("SOURCE", "OTHER", "en") < 0.5


def test_paddle_unavailable_is_explicit(monkeypatch):
    monkeypatch.setattr(PaddleRegionVerifier, "available", staticmethod(lambda: False))
    result = PaddleRegionVerifier().verify_regions(
        np.zeros((10, 10, 3), dtype=np.uint8),
        [BBox(0, 0, 10, 10)],
        ["source"],
    )[0]
    assert result.state == "unavailable"
    assert result.evidence()["backend"] == "paddleocr"


def test_hard_rejection_cannot_be_compensated_by_rank_score():
    source = np.full((24, 24, 3), 125, dtype=np.uint8)
    mask = np.zeros((24, 24), dtype=bool)
    mask[8:16, 8:16] = True
    candidate = source.copy()
    candidate[0:4, 0:4] = 0
    passed, evidence = inpaint_providers.quality_gate(source, candidate, mask)
    assert passed is False
    assert "protected_area_changed" in evidence["hard_rejections"]


def test_multi_candidate_selects_highest_ranked_promoted_provider(monkeypatch):
    source = np.full((20, 20, 3), 100, dtype=np.uint8)
    mask = np.zeros((20, 20), dtype=bool)
    mask[6:14, 6:14] = True
    specs = {
        "lama": inpaint_providers.ProviderSpec("lama", "fixture", True, True, True, "ready", {"revision": "a"}),
        "diffstr_experimental": inpaint_providers.ProviderSpec(
            "diffstr_experimental", "fixture", True, True, True, "ready", {"revision": "b"}
        ),
        "brushnet_experimental": inpaint_providers.ProviderSpec(
            "brushnet_experimental", "fixture", False, False, False, "disabled", {}
        ),
    }
    monkeypatch.setattr(inpaint_providers, "provider_spec", lambda provider_id: specs[provider_id])
    monkeypatch.setattr(
        inpaint_providers,
        "repair",
        lambda provider_id, *_: inpaint_providers.RepairOutcome(provider_id, source.copy(), 1),
    )
    scores = {"diffstr_experimental": 0.86, "lama": 0.94}
    calls = []

    def gate(_source, candidate, _mask):
        provider = "diffstr_experimental" if len(calls) == 0 else "lama"
        calls.append(provider)
        return True, {"passed": True, "score": scores[provider], "hard_rejections": []}

    monkeypatch.setattr(inpaint_providers, "quality_gate", gate)
    result = inpaint_providers.repair_multi_candidate(source, mask)
    assert result.selected is not None
    assert result.selected.provider == "lama"
    assert sum(bool(item["selected"]) for item in result.candidates) == 1


def test_high_confidence_structural_material_prefers_lama(monkeypatch):
    available = {
        provider_id: inpaint_providers.ProviderSpec(
            provider_id, "fixture", True, True, True, "ready", {}
        )
        for provider_id in inpaint_providers.NEURAL_PROVIDER_IDS
    }
    monkeypatch.setattr(inpaint_providers, "provider_spec", lambda provider_id: available[provider_id])
    inst = InstText(
        "r1", BBox(0, 0, 20, 10),
        background_profile=BgProfil(texture="textured"),
        reconstruction_profile=ReconstructionProfile(
            material_class="masonry", material_confidence=.9
        ),
    )
    surface = SceneRegion(BBox(0, 0, 20, 10), "surface", 1.0, texture="textured")
    route = inpaint_providers.route(
        inst, surface,
        np.full((20, 20, 3), 100, dtype=np.uint8),
        np.ones((20, 20), dtype=bool),
    )
    assert route.provider == "lama"
    assert "material=masonry" in route.reason
