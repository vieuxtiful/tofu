"""Regression coverage for Garnish's non-destructive Verify scores."""

import dataclasses
import json

import numpy as np

from tofu.core.types import BBox, GarnishProfile, InstText, SceneRegion, TextManifest
from tofu.layers import verify


BOX = BBox(20, 15, 40, 25)


def _image():
    return np.full((60, 90, 3), 70, dtype=np.uint8)


def _with_text(image):
    output = image.copy()
    output[22:33, 28:52] = 230
    return output


def _mask(image, bbox, refine=True):
    """Stable synthetic glyph mask; production continues to use imaging.text_mask."""
    crop = image[bbox.y:bbox.y + bbox.height, bbox.x:bbox.x + bbox.width]
    return crop[..., 0] > 150


def _manifest(profile=None):
    inst = InstText(id="r1", bounding_box=BOX, text="OLD", target_text="NEW")
    region = SceneRegion(BBox(0, 0, 90, 60), "sign", .9, garnish_profile=profile)
    return TextManifest(asset_id="garnish-verify", total_regions=1, img_dim=(90, 60),
                        instances=[inst], scene_regions=[region])


def test_edge_similarity_distinguishes_matching_and_blurred_treatment(monkeypatch):
    monkeypatch.setattr(verify, "_text_mask", _mask)
    source = _with_text(_image())
    matching = source.copy()
    # A deliberately soft replacement retains the same coarse glyph shape
    # but not the source's physical edge energy.
    blurred = _image()
    blurred[21:34, 27:53] = 150
    blurred[22:33, 28:52] = 190
    assert verify._garnish_edge_similarity_score(np, source, matching, _manifest().instances[0]) > 0.99
    assert verify._garnish_edge_similarity_score(np, source, blurred, _manifest().instances[0]) < 0.9


def test_texture_match_detects_ring_regression():
    source = _with_text(_image())
    localized = source.copy()
    assert verify._garnish_texture_match_score(np, source, localized, _manifest().instances[0]) > 0.99

    noisy = localized.copy()
    rng = np.random.default_rng(7)
    noisy[3:14, 8:73] = rng.integers(0, 255, (11, 65, 3), dtype=np.uint8)
    assert verify._garnish_texture_match_score(np, source, noisy, _manifest().instances[0]) < 0.9


def test_outside_mask_preservation_detects_bleed(monkeypatch):
    monkeypatch.setattr(verify, "_text_mask", _mask)
    cleansed = _image()
    localized = _with_text(cleansed)
    inst = _manifest().instances[0]
    assert verify._outside_mask_preservation_score(np, cleansed, localized, inst) == 1.0

    bled = localized.copy()
    bled[16:21, 21:27] = 130  # inside bbox, safely outside synthetic glyph mask
    assert verify._outside_mask_preservation_score(np, cleansed, bled, inst) < 1.0


def test_assess_is_backward_compatible_without_an_active_profile(monkeypatch):
    monkeypatch.setattr(verify, "_ocr_roundtrip_score", lambda *args: None)
    monkeypatch.setattr(verify, "_residual_source_text_score", lambda *args: None)
    source = _with_text(_image())
    report = verify.assess(source, _manifest(GarnishProfile()), source, _image())
    assert report.metrics["garnish_edge"] == {}
    assert report.metrics["garnish_texture"] == {}
    assert report.metrics["outside_mask"] == {}
    assert "garnish" not in report.metrics["scorer"]


def test_garnish_metrics_are_json_safe_python_floats(monkeypatch):
    monkeypatch.setattr(verify, "_text_mask", _mask)
    monkeypatch.setattr(verify, "_ocr_roundtrip_score", lambda *args: None)
    monkeypatch.setattr(verify, "_residual_source_text_score", lambda *args: None)
    cleansed = _image()
    localized = _with_text(cleansed)
    report = verify.assess(
        localized,
        _manifest(GarnishProfile(edge_blur_px=1.0)),
        localized,
        cleansed,
    )
    for values in (report.metrics["garnish_edge"], report.metrics["garnish_texture"], report.metrics["outside_mask"]):
        assert all(type(value) is float for value in values.values())
    json.dumps(dataclasses.asdict(report))

