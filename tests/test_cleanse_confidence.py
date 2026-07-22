"""Accuracy gates for mask evidence, grouping, and cache provenance."""
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest
from PIL import Image

from tofu.core.types import BBox, BgProfil, InstText, TextManifest
from tofu.layers import cleanse, inpaint_providers


def test_surface_polygon_guides_context_without_clipping_observed_text(monkeypatch):
    image = np.full((30, 40, 3), 100, dtype=np.uint8)
    observed = np.zeros((30, 40), dtype=bool)
    observed[10:16, 12:24] = True
    monkeypatch.setattr(cleanse, "_region_mask", lambda *_args: observed.copy())
    selected, evidence = cleanse._select_region_mask(
        np, cv2, image, BBox(10, 8, 18, 12), 30, 40,
        None, [(0, 0), (18, 0), (18, 29), (0, 29)],
    )
    assert np.array_equal(selected, observed)
    assert evidence["surface_mask_containment"] < 1
    assert evidence["surface_polygon_used_for_context"] is True


def test_instance_polygon_only_constrains_when_it_preserves_mask_evidence(monkeypatch):
    image = np.full((30, 40, 3), 100, dtype=np.uint8)
    observed = np.zeros((30, 40), dtype=bool)
    observed[10:16, 12:24] = True
    monkeypatch.setattr(cleanse, "_region_mask", lambda *_args: observed.copy())
    selected, evidence = cleanse._select_region_mask(
        np, cv2, image, BBox(10, 8, 18, 12), 30, 40,
        [(12, 10), (15, 10), (15, 15), (12, 15)], None,
    )
    # A too-tight detector polygon would erase half the observed glyph.  It
    # must fail open instead of creating an OCR-visible source-text remnant.
    assert np.array_equal(selected, observed)
    assert evidence["instance_polygon_applied"] is False


def test_nearby_compatible_neural_regions_share_one_group():
    surface = object()
    first = np.zeros((40, 70), dtype=bool); first[12:18, 10:20] = True
    second = np.zeros((40, 70), dtype=bool); second[12:18, 27:36] = True
    far = np.zeros((40, 70), dtype=bool); far[12:18, 54:63] = True
    plans = [
        {"provider": "lama", "surface": surface, "mask": first},
        {"provider": "lama", "surface": surface, "mask": second},
        {"provider": "lama", "surface": surface, "mask": far},
    ]
    groups = cleanse._neural_groups(np, cv2, plans)
    assert sorted(len(group) for group in groups) == [1, 2]


def test_cleanse_cache_restores_matching_provenance_only(tmp_path, monkeypatch):
    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root / "server"))
    import main

    asset = tmp_path / "asset.png"
    Image.new("RGB", (24, 18), "#336699").save(asset)
    monkeypatch.setattr(main, "OUTPUT_DIR", tmp_path)
    monkeypatch.setattr(main, "_asset_path", lambda _asset_id: asset)
    manifest = TextManifest("asset", 1, [InstText("r1", BBox(2, 2, 8, 8))])
    first = main._cleansed_base("asset", manifest)
    assert first.size == (24, 18)
    assert manifest.instances[0].repair_provenance is not None
    key = main._cleanse_cache_key("asset", manifest)
    sidecar = tmp_path / "cleanse-cache" / f"asset-{key}.json"
    payload = json.loads(sidecar.read_text(encoding="utf-8"))
    assert payload["schema"] == "cleanse-cache-v4"
    assert "r1" in payload["repair_provenance"]
    manifest.instances[0].repair_provenance = None
    main._cleansed_base("asset", manifest)
    assert manifest.instances[0].repair_provenance == payload["repair_provenance"]["r1"]


def test_unpromoted_neural_candidate_is_observed_but_never_silently_accepted(monkeypatch):
    image = np.full((40, 64, 3), 130, dtype=np.uint8)
    image[14:26, 18:46] = 20
    inst = InstText("r1", BBox(16, 12, 32, 16), background_profile=BgProfil(texture="textured"))
    manifest = TextManifest("asset", 1, [inst])
    observed = []
    monkeypatch.setattr(inpaint_providers, "route", lambda *_args: inpaint_providers.RepairRoute(
        "lama", "neural", .7, False, True, "fixture neural candidate",
    ))
    monkeypatch.setattr(inpaint_providers, "provider_spec", lambda _name: inpaint_providers.ProviderSpec(
        "lama", "fixture", True, False, True, "ready", {},
    ))
    monkeypatch.setattr(inpaint_providers, "repair", lambda _provider, source, mask: inpaint_providers.RepairOutcome(
        "lama", np.where(mask[..., None], 130, source).astype(np.uint8), 3,
    ))
    monkeypatch.setattr(inpaint_providers, "quality_gate", lambda *_args: (True, {"passed": True, "score": 1.0}))
    cleanse.erase(Image.fromarray(image), manifest, candidate_observer=lambda info, candidate, mask: observed.append((info, candidate, mask)) or {"id": "c1", "url": "/outputs/c1.png"})
    repair = inst.repair_provenance
    assert len(observed) == 1
    assert repair["executed_provider"] == "telea_fallback"
    assert repair["candidates"][0]["accepted"] is False
    assert repair["candidates"][0]["evidence"]["artifact"]["id"] == "c1"


def test_candidate_overlay_can_only_be_applied_as_an_undoable_patch(tmp_path, monkeypatch):
    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root / "server"))
    import main

    asset = tmp_path / "asset.png"
    Image.new("RGB", (32, 24), "#526b7a").save(asset)
    monkeypatch.setattr(main, "OUTPUT_DIR", tmp_path)
    monkeypatch.setattr(main, "_asset_path", lambda _asset_id: asset)
    candidate = np.full((24, 32, 3), 100, dtype=np.uint8)
    mask = np.zeros((24, 32), dtype=bool); mask[6:14, 9:21] = True
    artifact = main._candidate_observer("asset", "plan-key")(
        {"group_key": "repair-fixture", "group_ids": ["r1"], "provider": "lama",
         "decision": "unpromoted_provider", "execution": {"ok": True}, "quality_gate": {"passed": True}},
        candidate, mask,
    )
    result = main.apply_repair_candidate("asset", artifact["id"])
    assert result["patches"][-1]["candidate_id"] == artifact["id"]
    patch = Image.open(tmp_path / result["patches"][-1]["file"]).convert("RGBA")
    assert patch.getbbox() is not None
    retried = main.apply_repair_candidate("asset", artifact["id"])
    assert retried["already_applied"] is True
    assert len(retried["patches"]) == 1
    with pytest.raises(main.HTTPException) as stale:
        main.apply_repair_candidate("asset", artifact["id"], main.CandidateApplyRequest(cache_key="other-preview"))
    assert stale.value.status_code == 409


def test_manual_treatment_uses_the_preview_manifest_snapshot(tmp_path, monkeypatch):
    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root / "server"))
    import main

    asset = tmp_path / "asset.png"
    Image.new("RGB", (48, 32), "#526b7a").save(asset)
    monkeypatch.setattr(main, "OUTPUT_DIR", tmp_path)
    monkeypatch.setattr(main, "_asset_path", lambda _asset_id: asset)
    seen = []
    monkeypatch.setattr(main, "_cleansed_base", lambda _asset_id, manifest: seen.append(manifest) or Image.open(asset).convert("RGBA"))
    snapshot = {
        "asset_id": "asset", "total_regions": 1, "src_lang": "ja", "targ_lang": "en",
        "img_dim": [48, 32], "scene_regions": [], "asset_type": "image", "frame_count": 1,
        "fps": None, "duration": None, "prcssng_time": None,
        "instances": [{"id": "r1", "bounding_box": {"x": 7, "y": 5, "width": 20, "height": 10},
                       "text": "x", "target_text": "y", "confidence": 1, "detected_language": "ja",
                       "reading_order": 0, "dnt": False, "target_language": "en"}],
    }
    result = main.inpaint(main.InpaintRequest(
        asset_id="asset", points=[[10, 10], [18, 10]], radius=4, manifest=snapshot,
    ))
    assert result["patches"]
    assert seen and seen[0].instances[0].bounding_box.x == 7


def test_treatment_restore_keeps_patch_pixels_for_unified_redo(tmp_path, monkeypatch):
    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root / "server"))
    import main

    asset = tmp_path / "asset.png"
    Image.new("RGB", (32, 24), "#526b7a").save(asset)
    monkeypatch.setattr(main, "OUTPUT_DIR", tmp_path)
    monkeypatch.setattr(main, "_asset_path", lambda _asset_id: asset)
    patches = [
        {"id": "p1", "file": "p1.png", "bbox": {"x": 2, "y": 2, "width": 4, "height": 4}},
        {"id": "p2", "file": "p2.png", "bbox": {"x": 12, "y": 3, "width": 4, "height": 4}},
    ]
    Image.new("RGBA", (4, 4), "#ff0000").save(tmp_path / "p1.png")
    Image.new("RGBA", (4, 4), "#00ff00").save(tmp_path / "p2.png")
    main._patch_index("asset").write_text(json.dumps(patches), encoding="utf-8")
    main._archive_patches("asset", patches)
    main.undo_inpaint("asset", "p2")
    restored = main.restore_treatment("asset", main.TreatmentRestoreRequest(patch_ids=["p1", "p2"]))
    assert [patch["id"] for patch in restored["patches"]] == ["p1", "p2"]
    assert (tmp_path / "p2.png").is_file()


def test_localized_baseline_is_first_render_entry_not_latest_manual_edit(tmp_path, monkeypatch):
    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root / "server"))
    import main

    asset = tmp_path / "asset.png"
    Image.new("RGB", (32, 24), "#526b7a").save(asset)
    monkeypatch.setattr(main, "OUTPUT_DIR", tmp_path)
    monkeypatch.setattr(main, "_asset_path", lambda _asset_id: asset)
    source = {
        "asset_id": "asset", "total_regions": 1, "src_lang": "en", "targ_lang": "fr",
        "img_dim": [32, 24], "scene_regions": [], "asset_type": "image", "frame_count": 1,
        "fps": None, "duration": None, "prcssng_time": None,
        "instances": [{"id": "r1", "bounding_box": {"x": 2, "y": 2, "width": 20, "height": 10},
                       "text": "old", "target_text": "nouveau", "confidence": 1, "detected_language": "en",
                       "reading_order": 0, "dnt": False, "target_language": "fr"}],
    }
    first = main.capture_localized_baseline("asset", main.LocalizedBaselineRequest(manifest=source, patch_ids=[]))
    edited = json.loads(json.dumps(source)); edited["instances"][0]["target_text"] = "edited"
    second = main.capture_localized_baseline("asset", main.LocalizedBaselineRequest(manifest=edited, patch_ids=["p1"]))
    assert first["manifest"]["instances"][0]["target_text"] == "nouveau"
    assert second["manifest"]["instances"][0]["target_text"] == "nouveau"
