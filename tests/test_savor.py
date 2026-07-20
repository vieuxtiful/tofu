## 🍢 Savor: sniff_out (course 1), chew_on (course 2), taste (courses 1-3)
import numpy as np
import pytest
from PIL import Image, ImageDraw, ImageFont

from tofu.core.types import BBox, InstText
from tofu.layers import savor


def _font(size):
    try:
        return ImageFont.truetype("arial.ttf", size)
    except Exception:
        pytest.skip("arial.ttf not available on this host")


def render_spaced_text(text, font_size=36, gap=10, margin=20):
    """draw each character with a guaranteed pixel gap between it and its
    neighbors, so connected-component segmentation can't merge glyphs
    regardless of the font's natural kerning -- keeps chew_on() tests
    focused on glyph-shape comparison, not segmentation robustness."""
    font = _font(font_size)
    widths = []
    for ch in text:
        bbox = font.getbbox(ch)
        widths.append(bbox[2] - bbox[0] if ch != " " else font_size // 2)
    total_w = sum(widths) + gap * (len(text) - 1) + margin * 2
    total_h = font_size * 2 + margin * 2
    img = Image.new("RGB", (total_w, total_h), (235, 235, 230))
    draw = ImageDraw.Draw(img)
    x = margin
    y = margin
    for ch, w in zip(text, widths):
        draw.text((x, y), ch, font=font, fill=(20, 20, 20))
        x += w + gap
    bbox = BBox(x=0, y=0, width=total_w, height=total_h)
    return img, bbox


def make_inst(text, bbox, id="r1"):
    return InstText(id=id, bounding_box=bbox, text=text, confidence=0.95)


class TestSniffOut:
    """course 1: context-only smell test. proposes Morsels; never rewrites."""

    def test_finds_the_motivating_case(self):
        morsels = savor.sniff_out("Open 9am to Spm")
        assert len(morsels) == 1
        m = morsels[0]
        assert m.orig_char == "S" and m.digit_char == "5"
        assert m.corrected_token == "5pm"
        assert m.token_text == "Spm"

    def test_proposes_but_does_not_reject_sam_by_smell_alone(self):
        """sniff_out is context-only and CANNOT distinguish "Sam" (a
        name) from "Spm" (a misread digit) -- this is the exact gap the
        naive regex-substitution plan had. it still proposes a Morsel
        here; safety comes from chew_on() refusing to confirm it (see
        TestTaste.test_sam_is_never_swallowed below)."""
        morsels = savor.sniff_out("Sam")
        assert len(morsels) == 1
        assert morsels[0].orig_char == "S" and morsels[0].digit_char == "5"

    def test_no_morsel_without_am_pm_context(self):
        assert savor.sniff_out("$8 Special") == []
        assert savor.sniff_out("Big Ben") == []

    def test_already_correct_digit_yields_no_morsel(self):
        assert savor.sniff_out("Open 9am to 5pm") == []
        assert savor.sniff_out("12pm") == []

    def test_invalid_hour_after_substitution_is_rejected(self):
        # 'B' -> '8' would give "18pm", not a valid 1-12 hour
        assert savor.sniff_out("1Bpm") == []

    def test_multi_position_ambiguity_is_skipped_not_guessed(self):
        # both characters are confusable letters -- more than one bite
        # can verify at once, so no Morsel is proposed
        assert savor.sniff_out("BBpm") == []

    def test_two_char_prefix_single_ambiguous_position(self):
        morsels = savor.sniff_out("1Zpm")
        assert len(morsels) == 1
        assert morsels[0].orig_char == "Z" and morsels[0].digit_char == "2"
        assert morsels[0].corrected_token == "12pm"


class TestChewOn:
    """course 2: the actual bite -- real pixel evidence decides."""

    def test_confirms_a_real_digit_misread_as_letter(self):
        # pixels show the TRUE glyph '5' (what a "5pm" sign really looks
        # like); the manifest's recognized text is the misread "Spm"
        img, bbox = render_spaced_text("5pm")
        inst = make_inst("Spm", bbox)
        morsel = savor.sniff_out("Spm")[0]
        verdict = savor.chew_on(img, inst, morsel)
        assert verdict is True  # swallow

    def test_rejects_when_pixels_show_the_original_letter(self):
        img, bbox = render_spaced_text("Sam")
        inst = make_inst("Sam", bbox)
        morsel = savor.sniff_out("Sam")[0]
        verdict = savor.chew_on(img, inst, morsel)
        assert verdict is not True  # spit out (False) or still chewing (None) -- never swallowed

    def test_none_on_degenerate_crop(self):
        inst = make_inst("Spm", BBox(x=0, y=0, width=1, height=1))
        morsel = savor.sniff_out("Spm")[0]
        img = Image.new("RGB", (50, 50), (240, 240, 240))
        assert savor.chew_on(img, inst, morsel) is None


class TestTaste:
    """courses 1-3 end to end -- the actual public API."""

    def test_fixes_the_motivating_case_end_to_end(self):
        img, bbox = render_spaced_text("5pm")
        inst = make_inst("Spm", bbox)
        n = savor.taste(img, [inst])
        assert n == 1
        assert inst.text == "5pm"
        assert inst.ocr_correction is not None
        assert inst.ocr_correction["applied"] is True
        assert inst.ocr_correction["original_text"] == "Spm"

    def test_sam_is_never_swallowed(self):
        """the concrete safety regression: a naive context-only fix
        rewrites 'Sam' to '5am'. this must not happen once pixel
        evidence (an actual rendered 'S', not a '5') is checked."""
        img, bbox = render_spaced_text("Sam")
        inst = make_inst("Sam", bbox)
        n = savor.taste(img, [inst])
        assert n == 0
        assert inst.text == "Sam"

    def test_no_morsel_leaves_ocr_correction_unset(self):
        img, bbox = render_spaced_text("Big Ben")
        inst = make_inst("Big Ben", bbox)
        savor.taste(img, [inst])
        assert inst.text == "Big Ben"
        assert inst.ocr_correction is None

    def test_empty_text_is_a_no_op(self):
        inst = make_inst(None, BBox(x=0, y=0, width=10, height=10))
        assert savor.taste(Image.new("RGB", (50, 50)), [inst]) == 0

    def test_already_correct_time_is_untouched(self):
        img, bbox = render_spaced_text("5pm")
        inst = make_inst("5pm", bbox)
        n = savor.taste(img, [inst])
        assert n == 0
        assert inst.text == "5pm"
        assert inst.ocr_correction is None
