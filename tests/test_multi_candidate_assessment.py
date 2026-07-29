import numpy as np

from tofu.core.types import (
    BBox, BgProfil, InstText, ReconstructionProfile, SceneRegion, StyleProfil,
)
from tofu.layers import cleanse, inpaint_providers
from tofu.layers import cicerone
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


def _verification(text, confidence, bbox):
    return OCRVerificationResult(
        state="disagree",
        text=text,
        confidence=confidence,
        similarity=0.25,
        detections=[{
            "text": text,
            "confidence": confidence,
            "polygon": [
                [bbox.x, bbox.y],
                [bbox.x + bbox.width, bbox.y],
                [bbox.x + bbox.width, bbox.y + bbox.height],
                [bbox.x, bbox.y + bbox.height],
            ],
        }],
    )


def test_general_arbitration_accepts_only_a_strong_supported_alternate(monkeypatch):
    bbox = BBox(4, 4, 80, 24)
    inst = InstText(
        "r1", bbox, text="RFPUBLIQUE", confidence=.20,
        detected_language="fr",
        recognition_history=[{
            "stage": "detection_pass", "engine": "easyocr", "pass": 1,
            "candidate_text": "RFPUBLIQUE", "candidate_confidence": .20,
            "selected": True,
        }],
    )
    monkeypatch.setattr(cicerone, "_has_ink_support", lambda *_: True)
    monkeypatch.setattr(
        PaddleRegionVerifier,
        "verify_regions",
        lambda self, asset, regions, expected_texts, language=None: [
            _verification("RÃ‰PUBLIQUE", .99, regions[0])
        ],
    )

    changed = cicerone.assess_multi_candidate_ocr(
        np.zeros((40, 100, 3), dtype=np.uint8), [inst],
    )

    assert changed == 1
    assert inst.text == "RÃ‰PUBLIQUE"
    assert inst.ocr_provenance["decision"] == "accept"
    assert inst.ocr_provenance["calibration_registry_version"]
    assert len(inst.ocr_provenance["observations"]) == 3
    assert inst.recognition_history[-1]["accepted"] is True


def test_general_arbitration_never_removes_on_verifier_silence(monkeypatch):
    bbox = BBox(4, 4, 80, 24)
    inst = InstText(
        "r1", bbox, text="AVENUE", confidence=.30, detected_language="fr",
    )
    monkeypatch.setattr(cicerone, "_has_ink_support", lambda *_: True)
    monkeypatch.setattr(
        PaddleRegionVerifier,
        "verify_regions",
        lambda self, asset, regions, expected_texts, language=None: [
            OCRVerificationResult("no_text", text="", confidence=0.0, similarity=0.0)
        ],
    )

    changed = cicerone.assess_multi_candidate_ocr(
        np.zeros((40, 100, 3), dtype=np.uint8), [inst],
    )

    assert changed == 0
    assert inst.text == "AVENUE"
    assert inst.ocr_provenance["decision"] == "reject"
    assert inst.ocr_provenance["review_required"] is True
    assert "alternate_empty" in inst.ocr_provenance["reason_codes"]


def test_general_arbitration_batches_by_language(monkeypatch):
    instances = [
        InstText("r1", BBox(0, 0, 40, 20), text="RUE", confidence=.4, detected_language="fr"),
        InstText("r2", BBox(0, 30, 40, 20), text="CALLE", confidence=.4, detected_language="es"),
        InstText("r3", BBox(0, 60, 40, 20), text="PLACE", confidence=.4, detected_language="fr"),
    ]
    calls = []

    def verify(self, asset, regions, expected_texts, language=None):
        calls.append((tuple(self.languages), len(regions), language))
        return [
            _verification(text, .4, bbox)
            for text, bbox in zip(expected_texts, regions)
        ]

    monkeypatch.setattr(PaddleRegionVerifier, "verify_regions", verify)
    monkeypatch.setattr(cicerone, "_has_ink_support", lambda *_: True)

    cicerone.assess_multi_candidate_ocr(
        np.zeros((100, 100, 3), dtype=np.uint8), instances,
    )

    assert sorted(calls) == [(("es",), 1, "es"), (("fr",), 2, "fr")]


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


def test_crossing_line_break_is_hard_rejected():
    import cv2

    source = np.full((48, 64, 3), 240, dtype=np.uint8)
    cv2.line(source, (3, 24), (60, 24), (15, 15, 15), 3)
    mask = np.zeros((48, 64), dtype=bool)
    mask[16:33, 24:41] = True
    candidate = source.copy()
    candidate[mask] = 240
    structural = inpaint_providers._structural_continuity_score(
        cv2, np,
        cv2.cvtColor(source, cv2.COLOR_RGB2GRAY),
        cv2.cvtColor(candidate, cv2.COLOR_RGB2GRAY),
        mask,
    )
    assert structural is not None and structural < .30
    passed, evidence = inpaint_providers.quality_gate(source, candidate, mask)
    assert passed is False
    assert "structural_break" in evidence["hard_rejections"]


def test_perspective_context_rectifies_and_restores_only_mask():
    import cv2

    yy, xx = np.mgrid[0:60, 0:80]
    source = np.stack((xx, yy, np.full_like(xx, 100)), axis=-1).astype(np.uint8)
    mask = np.zeros((60, 80), dtype=bool)
    mask[18:39, 20:58] = True
    inst = InstText(
        "r1", BBox(18, 15, 44, 30),
        style_profile=StyleProfil(transform={
            "quad": [[.08, .12], [.92, 0], [1, .86], [0, 1]],
        }),
    )
    rectified, rectified_mask, restore, evidence = cleanse._perspective_repair_context(
        cv2, np, source, mask, inst
    )
    assert evidence["applied"] is True
    assert rectified.shape[:2] == rectified_mask.shape
    assert restore is not None
    generated = np.full_like(rectified, 177)
    restored = restore(generated)
    assert restored.shape == source.shape
    assert np.array_equal(restored[~mask], source[~mask])
    assert np.any(restored[mask] != source[mask])


def test_degenerate_perspective_quad_falls_back():
    import cv2

    source = np.zeros((20, 20, 3), dtype=np.uint8)
    mask = np.zeros((20, 20), dtype=bool)
    mask[5:15, 5:15] = True
    inst = InstText(
        "r1", BBox(5, 5, 10, 10),
        style_profile=StyleProfil(transform={
            "quad": [[0, 0], [0, 0], [0, 0], [0, 0]],
        }),
    )
    repaired_source, repaired_mask, restore, evidence = cleanse._perspective_repair_context(
        cv2, np, source, mask, inst
    )
    assert evidence["applied"] is False
    assert restore is None
    assert repaired_source is source
    assert repaired_mask is mask
