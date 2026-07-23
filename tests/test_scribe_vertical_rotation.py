## 🍢 scribe: vertical CJK column rendering + detected-rotation propagation
import numpy as np
import pytest
from PIL import Image, ImageFont

from tofu.core.types import (
    BBox, CharactText, InstText, RenderParams, StyleProfil, TextManifest,
)
from tofu.layers import scribe

CJK_FONT_CANDIDATES = ("msgothic.ttc", "meiryo.ttc", "YuGothM.ttc", "malgun.ttf")


def _find_cjk_font():
    for cand in CJK_FONT_CANDIDATES:
        try:
            ImageFont.truetype(cand, 12)
            return cand
        except Exception:
            continue
    return None


CJK_FONT = _find_cjk_font()
needs_cjk_font = pytest.mark.skipif(
    CJK_FONT is None, reason="no CJK-capable font found on this host"
)


class TestShouldRenderVertical:
    def test_tall_narrow_cjk_region_is_vertical(self):
        bbox = BBox(x=0, y=0, width=90, height=260)
        assert scribe._should_render_vertical(bbox, "ja", "居酒屋") is True

    def test_wide_region_is_not_vertical(self):
        bbox = BBox(x=0, y=0, width=300, height=90)
        assert scribe._should_render_vertical(bbox, "ja", "ようこそ") is False

    def test_non_cjk_language_is_never_vertical(self):
        bbox = BBox(x=0, y=0, width=90, height=260)
        assert scribe._should_render_vertical(bbox, "en", "TALL") is False

    def test_single_character_is_not_vertical(self):
        # one glyph gives no reliable orientation signal
        bbox = BBox(x=0, y=0, width=90, height=260)
        assert scribe._should_render_vertical(bbox, "ja", "酒") is False

    def test_no_language_is_not_vertical(self):
        bbox = BBox(x=0, y=0, width=90, height=260)
        assert scribe._should_render_vertical(bbox, None, "居酒屋") is False


@needs_cjk_font
class TestVerticalRendering:
    def _render(self, text, bbox, tsume=0.0, target_language="ja"):
        asset = Image.new("RGB", (960, 640), (238, 234, 228))
        inst = InstText(
            id="r1", bounding_box=bbox, text="x", target_text=text,
            target_language=target_language,
            style_profile=StyleProfil(font_family=CJK_FONT, tsume=tsume),
        )
        manifest = TextManifest(asset_id="a", total_regions=1, instances=[inst])
        return np.asarray(scribe.render(asset, manifest, "ja"))

    def test_three_characters_produce_three_ink_bands(self):
        bbox = BBox(x=148, y=90, width=92, height=262)
        out = self._render("居酒屋", bbox)
        region = out[bbox.y:bbox.y + bbox.height, bbox.x:bbox.x + bbox.width]
        dark_rows = np.where((region < 200).any(axis=(1, 2)))[0]
        assert len(dark_rows) > 0
        gaps = np.diff(dark_rows)
        band_breaks = (gaps > 5).sum()
        assert band_breaks == 2  # 3 bands => 2 gaps between them

    def test_ink_stays_within_bbox_width(self):
        bbox = BBox(x=148, y=90, width=92, height=262)
        out = self._render("居酒屋", bbox)
        region = out[bbox.y:bbox.y + bbox.height, bbox.x - 20:bbox.x + bbox.width + 20]
        dark_cols = np.where((region < 200).any(axis=(0, 2)))[0]
        # columns are offset by the -20 crop margin; must fall within
        # [20, 20+bbox.width] (i.e. inside the real bbox), not spill
        # into the added margin on either side
        assert dark_cols.min() >= 15
        assert dark_cols.max() <= bbox.width + 25

    def test_tsume_compresses_block_height(self):
        bbox = BBox(x=100, y=50, width=100, height=400)
        loose = self._render("居酒屋", bbox, tsume=0.0)
        tight = self._render("居酒屋", bbox, tsume=0.8)
        band = slice(bbox.y, bbox.y + bbox.height)
        loose_rows = np.where((loose[band, bbox.x:bbox.x + bbox.width] < 200).any(axis=(1, 2)))[0]
        tight_rows = np.where((tight[band, bbox.x:bbox.x + bbox.width] < 200).any(axis=(1, 2)))[0]
        assert (tight_rows.max() - tight_rows.min()) < (loose_rows.max() - loose_rows.min())


class TestRotationPropagation:
    """detected typography rotation (Phase 1) applies when no explicit
    RenderParams.rotation override is given; an explicit override always
    wins."""

    def _manifest_with_rotation(self, detected_deg, explicit_rotation=None):
        bbox = BBox(x=50, y=50, width=200, height=60)
        inst = InstText(
            id="r1", bounding_box=bbox, text="x", target_text="TILT",
            style_profile=StyleProfil(color="#000000"),
            characteristics=CharactText(positioning={"rotation_deg": detected_deg}),
        )
        manifest = TextManifest(asset_id="a", total_regions=1, instances=[inst])
        render_params = None
        if explicit_rotation is not None:
            render_params = {"r1": RenderParams(position=bbox, style=inst.style_profile,
                                                 rotation=explicit_rotation)}
        return manifest, render_params

    def test_detected_rotation_applied_when_no_override(self):
        asset = Image.new("RGB", (400, 300), (255, 255, 255))
        manifest, _ = self._manifest_with_rotation(15.0)
        upright_asset = Image.new("RGB", (400, 300), (255, 255, 255))
        upright_manifest, _ = self._manifest_with_rotation(0.0)

        out_rotated = np.asarray(scribe.render(asset, manifest, "en"))
        out_upright = np.asarray(scribe.render(upright_asset, upright_manifest, "en"))
        # the two renders must differ -- rotation actually changed pixels
        assert not np.array_equal(out_rotated, out_upright)

    def test_explicit_override_wins_over_detected(self):
        asset = Image.new("RGB", (400, 300), (255, 255, 255))
        manifest, render_params = self._manifest_with_rotation(15.0, explicit_rotation=0.0)
        out = np.asarray(scribe.render(asset, manifest, "en", render_params=render_params))
        asset2 = Image.new("RGB", (400, 300), (255, 255, 255))
        upright_manifest, _ = self._manifest_with_rotation(0.0)
        out_upright = np.asarray(scribe.render(asset2, upright_manifest, "en"))
        # explicit rotation=0.0 overrides the detected 15deg -> matches upright
        assert np.array_equal(out, out_upright)

    def test_editor_transform_rotation_overrides_detected_rotation(self):
        asset = Image.new("RGB", (400, 300), (255, 255, 255))
        manifest, _ = self._manifest_with_rotation(15.0)
        manifest.instances[0].style_profile.transform = {"rotation": 0.0}
        out = np.asarray(scribe.render(asset, manifest, "en"))
        upright_asset = Image.new("RGB", (400, 300), (255, 255, 255))
        upright_manifest, _ = self._manifest_with_rotation(0.0)
        upright = np.asarray(scribe.render(upright_asset, upright_manifest, "en"))
        assert np.array_equal(out, upright)


class TestShadow:
    def _render_with_shadow(self, shadow):
        asset = Image.new("RGB", (300, 150), (255, 255, 255))
        bbox = BBox(x=30, y=30, width=240, height=90)
        inst = InstText(
            id="r1", bounding_box=bbox, text="x", target_text="SHADOW",
            style_profile=StyleProfil(color="#000000", shadow=shadow),
        )
        manifest = TextManifest(asset_id="a", total_regions=1, instances=[inst])
        return np.asarray(scribe.render(asset, manifest, "en")), bbox

    def test_shadow_adds_ink_offset_from_main_text(self):
        out, bbox = self._render_with_shadow({"offset_x": 6, "offset_y": 6, "blur": 1})
        no_shadow, _ = self._render_with_shadow(None)
        # a shadowed render spreads ink over a wider (offset + blurred)
        # footprint than the plain glyphs alone, so total darkness over
        # the region (with margin, to catch the offset spill) must be
        # strictly greater with a shadow than without
        m = 10
        y0, y1 = bbox.y - m, bbox.y + bbox.height + m
        x0, x1 = bbox.x - m, bbox.x + bbox.width + m
        probe = out[y0:y1, x0:x1]
        probe_plain = no_shadow[y0:y1, x0:x1]
        assert probe.mean() < probe_plain.mean()

    def test_no_shadow_key_uses_defaults_without_raising(self):
        out, _ = self._render_with_shadow({})  # empty dict -> DEFAULT_SHADOW
        assert out is not None

    def test_none_shadow_renders_normally(self):
        out, bbox = self._render_with_shadow(None)
        region = out[bbox.y:bbox.y + bbox.height, bbox.x:bbox.x + bbox.width]
        assert (region < 250).any()  # text still renders without a shadow spec
