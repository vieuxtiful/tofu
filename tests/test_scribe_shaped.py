## 🍢 scribe: complex-script rendering routed through HarfBuzz (knead)
from pathlib import Path

import numpy as np
import pytest
from PIL import Image, ImageFont

from conftest import DEVANAGARI_FONT, LATIN_FONT, THAI_FONT
from tofu.core.types import BBox, InstText, Mask, StyleProfil, TextManifest
from tofu.layers import knead, scribe

NIRMALA = DEVANAGARI_FONT
LEELAWADEE = THAI_FONT
ARIAL = LATIN_FONT

HINDI = "हिन्दी"
KSHA = "क्ष"
THAI = "ภาษาไทย"

needs_shaping = pytest.mark.skipif(not knead.available(), reason="shaping libraries absent")
needs_deva = pytest.mark.skipif(NIRMALA is None, reason="no Devanagari font on this system")
needs_thai = pytest.mark.skipif(THAI_FONT is None, reason="no Thai font on this system")
needs_arial = pytest.mark.skipif(ARIAL is None, reason="no Latin font on this system")

REGION = (8, 8, 392, 72)


def manifest(target, lang, font_path=None, **style):
    inst = InstText(
        id="r1", bounding_box=BBox(x=8, y=8, width=384, height=64),
        segmentation_mask=Mask(
            polygon=[(8, 8), (392, 8), (392, 72), (8, 72)], confidence=0.9),
        text="source", target_text=target, target_language=lang,
        style_profile=StyleProfil(
            color="#101010", font_family=str(font_path) if font_path else None,
            font_size=34, **style),
    )
    return TextManifest(asset_id="a", total_regions=1, instances=[inst], src_lang="en")


def render(target, lang, font_path=None, size=(400, 80), **style):
    asset = Image.new("RGB", size, (255, 255, 255))
    out = scribe.render(asset, manifest(target, lang, font_path, **style), lang)
    return out.convert("RGB")


def render_unshaped(monkeypatch, target, lang, font_path=None, **style):
    """Same render with eligibility forced off -- the Pillow path."""
    monkeypatch.setattr(scribe, "shaping_script", lambda lang: None)
    return render(target, lang, font_path, **style)


def ink_mask(img):
    return np.asarray(img.convert("L")) < 200


def ink_bounds(img):
    m = ink_mask(img)
    cols = np.where(m.any(axis=0))[0]
    return (cols.min(), cols.max()) if cols.size else None


class TestEligibility:
    def test_indic_and_sea_scripts_are_shaped(self):
        for lang in ("hi", "mr", "bn", "pa", "gu", "ta", "te", "kn", "ml",
                     "si", "th", "km", "lo", "my"):
            assert scribe.shaping_script(lang) is not None, lang

    def test_latin_cyrillic_greek_cjk_hangul_are_not_shaped(self):
        """The whole safety property: these keep the Pillow path, so no
        existing render re-flows."""
        for lang in ("en", "de", "fr", "vi", "ru", "uk", "bg", "el",
                     "ja", "zh-cn", "zh-tw", "ko"):
            assert scribe.shaping_script(lang) is None, lang

    def test_rtl_is_not_shaped(self):
        """Arabic/Hebrew stay on press_joins + serving_order, which already
        render correctly and handle mixed-direction text."""
        for lang in ("ar", "fa", "he"):
            assert scribe.shaping_script(lang) is None, lang

    def test_unknown_language_is_not_shaped(self):
        assert scribe.shaping_script(None) is None
        assert scribe.shaping_script("xx-unmapped") is None

    def test_kill_switch_disables_shaping(self, monkeypatch):
        monkeypatch.setenv("TOFU_SHAPING", "0")
        assert scribe.shaping_script("hi") is None


@needs_shaping
@needs_deva
class TestDevanagari:
    def test_render_differs_from_the_pillow_path(self, monkeypatch):
        shaped = render(HINDI, "hi", NIRMALA)
        unshaped = render_unshaped(monkeypatch, HINDI, "hi", NIRMALA)
        assert not np.array_equal(np.asarray(shaped), np.asarray(unshaped))

    def test_conjuncts_reduce_the_glyph_count(self):
        font = ImageFont.truetype(str(NIRMALA), 34)
        assert len(knead.knead_run(HINDI, font, script="Deva")) < len(HINDI)

    def test_ksha_shapes_to_a_single_conjunct(self):
        font = ImageFont.truetype(str(NIRMALA), 34)
        assert len(knead.knead_run(KSHA, font, script="Deva")) == 1

    def test_shaped_render_produces_ink(self):
        x0, y0, x1, y1 = REGION
        assert ink_mask(render(HINDI, "hi", NIRMALA))[y0:y1, x0:x1].any()

    def test_kill_switch_reproduces_the_pillow_render(self, monkeypatch):
        baseline = render_unshaped(monkeypatch, HINDI, "hi", NIRMALA)
        monkeypatch.undo()
        monkeypatch.setenv("TOFU_SHAPING", "0")
        assert np.array_equal(np.asarray(render(HINDI, "hi", NIRMALA)),
                              np.asarray(baseline))


@needs_shaping
@needs_thai
class TestThai:
    def test_renders_ink_without_raising(self):
        x0, y0, x1, y1 = REGION
        assert ink_mask(render(THAI, "th", LEELAWADEE))[y0:y1, x0:x1].any()

    def test_render_differs_from_the_pillow_path(self, monkeypatch):
        shaped = render(THAI, "th", LEELAWADEE)
        unshaped = render_unshaped(monkeypatch, THAI, "th", LEELAWADEE)
        assert not np.array_equal(np.asarray(shaped), np.asarray(unshaped))


@needs_shaping
@needs_deva
class TestMeasureDrawInvariant:
    """The property the fitter depends on: what scribe measures is what it
    draws. Measuring with shaping and drawing without would overflow the box.
    """

    def test_measured_width_matches_drawn_ink_width(self):
        font = ImageFont.truetype(str(NIRMALA), 34)
        run = scribe._shaped_run(HINDI, font, "Deva")
        measured, _l, _t, _b = scribe._shaped_line_box(run, font)

        img = render(HINDI, "hi", NIRMALA)
        bounds = ink_bounds(img)
        assert bounds is not None
        drawn = bounds[1] - bounds[0] + 1
        # within a couple of px: measurement is float, the raster is snapped
        assert drawn == pytest.approx(measured, abs=3)

    def test_shaped_text_stays_inside_its_bbox(self):
        img = render(HINDI + " " + KSHA, "hi", NIRMALA)
        bounds = ink_bounds(img)
        assert bounds is not None
        assert bounds[0] >= 0 and bounds[1] < 400


@needs_shaping
@needs_arial
class TestNoRegressionForUnshapedScripts:
    """Byte-identical output for every script that keeps the Pillow path.
    A failure here means eligibility leaked."""

    @pytest.mark.parametrize("text,lang", [
        ("HELLO WORLD", "en"),
        ("Willkommen", "de"),
        ("ПРИВЕТ МИР", "ru"),
        ("ΚΑΛΗΜΕΡΑ", "el"),
        ("こんにちは", "ja"),
        ("你好世界", "zh-cn"),
        ("안녕하세요", "ko"),
    ])
    def test_render_is_byte_identical_with_shaping_stubbed(self, monkeypatch, text, lang):
        baseline = np.asarray(render(text, lang))
        monkeypatch.setattr(scribe, "shaping_script", lambda lang: None)
        assert np.array_equal(np.asarray(render(text, lang)), baseline)

    def test_rtl_still_uses_the_reshaper_path(self, monkeypatch):
        """Arabic must be untouched by this change -- it is served by
        press_joins/serving_order, not by shaping."""
        baseline = np.asarray(render("مرحبا بالعالم", "ar"))
        monkeypatch.setattr(scribe, "shaping_script", lambda lang: None)
        assert np.array_equal(np.asarray(render("مرحبا بالعالم", "ar")), baseline)


@needs_shaping
@needs_deva
class TestLetterSpacing:
    def test_tracking_widens_a_shaped_line(self):
        plain = render(HINDI, "hi", NIRMALA)
        spaced = render(HINDI, "hi", NIRMALA, tracking=6)
        pb, sb = ink_bounds(plain), ink_bounds(spaced)
        assert sb[1] - sb[0] > pb[1] - pb[0]

    def test_tracking_does_not_break_the_conjunct(self):
        """Spacing lands between clusters, so क्ष stays a single glyph."""
        font = ImageFont.truetype(str(NIRMALA), 34)
        run = scribe._shaped_run(KSHA, font, "Deva", tracking=8)
        assert len(run) == 1
