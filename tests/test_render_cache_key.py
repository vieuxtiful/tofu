"""Shared preview/final cache identity tests."""
import sys
import json
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "server"))

import main  # noqa: E402
from tofu.core.types import TextManifest  # noqa: E402


def test_cleanse_cache_key_changes_when_provider_policy_changes(tmp_path, monkeypatch):
    asset = tmp_path / "asset.png"
    asset.write_bytes(b"fixture")
    monkeypatch.setattr(main, "_asset_path", lambda _asset_id: asset)
    manifest = TextManifest("asset", 0, [])
    monkeypatch.setattr(main.inpaint_providers, "provider_statuses", lambda: [{"id": "lama", "promoted": False}])
    before = main._cleanse_cache_key("asset", manifest)
    monkeypatch.setattr(main.inpaint_providers, "provider_statuses", lambda: [{"id": "lama", "promoted": True}])
    after = main._cleanse_cache_key("asset", manifest)
    assert before != after


def test_shared_composed_base_is_pixel_stable_for_preview_and_final(tmp_path, monkeypatch):
    asset = tmp_path / "asset.png"
    Image.new("RGB", (20, 16), "#547c9e").save(asset)
    monkeypatch.setattr(main, "OUTPUT_DIR", tmp_path)
    monkeypatch.setattr(main, "_asset_path", lambda _asset_id: asset)
    manifest = TextManifest("asset", 0, [])
    patch_name = "asset.patch-p1.png"
    patch = Image.new("RGBA", (5, 4), (220, 40, 20, 255))
    patch.save(tmp_path / patch_name)
    (tmp_path / "asset.patches.json").write_text(json.dumps([
        {"id": "p1", "file": patch_name, "bbox": {"x": 4, "y": 5, "width": 5, "height": 4}},
    ]), encoding="utf-8")
    main._flatten_patches("asset")
    preview_base = main._composite_patches("asset", main._cleansed_base("asset", manifest))
    final_base = main._composite_patches("asset", main._cleansed_base("asset", manifest))
    assert np.array_equal(np.asarray(preview_base), np.asarray(final_base))
