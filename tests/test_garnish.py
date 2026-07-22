from PIL import Image, ImageDraw
import cv2
import numpy as np

from tofu.core.types import BBox, GarnishProfile, GarnishRegion, InstText, SceneRegion, TextManifest
from tofu.layers import garnish, scene
from tofu.utils.manifest_store import _dict_to_manifest, _manifest_to_dict


def _manifest(profile):
    return TextManifest(asset_id="g", total_regions=1, img_dim=(80, 50), instances=[
        InstText(id="r1", bounding_box=BBox(20, 12, 35, 20), text="old", target_text="new"),
    ], scene_regions=[SceneRegion(BBox(0, 0, 80, 50), "panel", .9, garnish_profile=profile)])


def test_garnish_identity_and_outside_pixels_preserved():
    base = Image.new("RGB", (80, 50), "#64748b")
    scribed = base.copy(); ImageDraw.Draw(scribed).rectangle((25, 16, 45, 25), fill="#f8fafc")
    manifest = _manifest(GarnishProfile())
    assert list(garnish.apply(scribed, manifest, base).getdata()) == list(scribed.getdata())

    manifest.scene_regions[0].garnish_profile = GarnishProfile(edge_blur_px=1, grain_strength=.1, source_confidence=.8)
    out = garnish.apply(scribed, manifest, base)
    assert out.getpixel((2, 2)) == scribed.getpixel((2, 2))
    assert out.getpixel((30, 18)) != base.getpixel((30, 18))


def test_garnish_profile_manifest_round_trip():
    manifest = _manifest(GarnishProfile(edge_blur_px=1.2, grain_strength=.2, source_confidence=.7))
    manifest.instances[0].garnish_override = GarnishProfile(gamma_shift=1.2)
    manifest.instances[0].garnish_enabled = False
    restored = _dict_to_manifest(_manifest_to_dict(manifest))
    assert restored.scene_regions[0].garnish_profile.edge_blur_px == 1.2
    assert restored.instances[0].garnish_override.gamma_shift == 1.2
    assert restored.instances[0].garnish_enabled is False


def test_disabled_garnish_bypasses_profile_without_losing_override():
    base = Image.new("RGB", (80, 50), "#64748b")
    scribed = base.copy(); ImageDraw.Draw(scribed).rectangle((25, 16, 45, 25), fill="#f8fafc")
    manifest = _manifest(GarnishProfile(edge_blur_px=1.5, grain_strength=.2))
    manifest.instances[0].garnish_override = GarnishProfile(edge_blur_px=2.0, grain_strength=.3)
    manifest.instances[0].garnish_enabled = False
    assert list(garnish.apply(scribed, manifest, base).getdata()) == list(scribed.getdata())
    assert manifest.instances[0].garnish_override.edge_blur_px == 2.0


def test_garnish_sub_region_is_persisted_and_clips_effect_to_its_polygon():
    base = Image.new("RGB", (80, 50), "#64748b")
    scribed = base.copy(); ImageDraw.Draw(scribed).rectangle((25, 16, 45, 25), fill="#b4b4b4")
    manifest = _manifest(GarnishProfile())
    manifest.instances[0].garnish_regions = [GarnishRegion(
        id="r1-g1", polygon=[(20, 12), (35, 12), (35, 32), (20, 32)],
        profile=GarnishProfile(gamma_shift=1.5), source="manual",
    )]
    manifest.instances[0].garnish_scope = "per_region"
    restored = _dict_to_manifest(_manifest_to_dict(manifest))
    assert restored.instances[0].garnish_regions[0].id == "r1-g1"
    output = garnish.apply(scribed, restored, base)
    # The selected left half is treated; the right-hand glyph pixels retain
    # their original Scribe raster despite belonging to the same instance.
    assert output.getpixel((28, 18)) != scribed.getpixel((28, 18))
    assert output.getpixel((42, 18)) == scribed.getpixel((42, 18))


def test_whole_selection_scope_ignores_saved_sub_region_masks():
    base = Image.new("RGB", (80, 50), "#64748b")
    scribed = base.copy(); ImageDraw.Draw(scribed).rectangle((25, 16, 45, 25), fill="#b4b4b4")
    manifest = _manifest(GarnishProfile())
    inst = manifest.instances[0]
    inst.garnish_scope = "whole_selection"
    inst.garnish_override = GarnishProfile(gamma_shift=1.5)
    inst.garnish_regions = [GarnishRegion(id="r1-g1", polygon=[(20, 12), (35, 12), (35, 32), (20, 32)])]
    output = garnish.apply(scribed, manifest, base)
    assert output.getpixel((28, 18)) != scribed.getpixel((28, 18))
    assert output.getpixel((42, 18)) != scribed.getpixel((42, 18))


def test_smudge_angle_uses_a_rotated_motion_path():
    alpha = np.zeros((41, 41), dtype=np.uint8)
    alpha[20, 20] = 255
    horizontal = garnish._motion_blur_alpha(cv2, np, alpha, 1.0, 0)
    vertical = garnish._motion_blur_alpha(cv2, np, alpha, 1.0, 90)
    hys, hxs = np.nonzero(horizontal)
    vys, vxs = np.nonzero(vertical)
    assert np.ptp(hxs) > np.ptp(hys)
    assert np.ptp(vys) > np.ptp(vxs)


def test_flat_scene_has_no_automatic_garnish_profile():
    flat = np.full((40, 60, 3), 127, dtype=np.uint8)
    profile = scene._analyze_garnish_profile(flat)
    assert profile.edge_blur_px == 0
    assert profile.grain_strength == 0
    assert profile.smudge_strength == 0
