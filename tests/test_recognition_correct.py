## 🍢 recognition_correct: Stage A grammar, Stage B glyph verification, Stage C safety
import numpy as np
import pytest
from PIL import Image, ImageDraw, ImageFont

from tofu.core.types import BBox, InstText
from tofu.layers import recognition_correct as rc


def _font(size):
    try:
        return ImageFont.truetype("arial.ttf", size)
    except Exception:
        pytest.skip("arial.ttf not available on this host")


def render_spaced_text(text, font_size=36, gap=10, margin=20):
    """draw each character with a guaranteed pixel gap between it and its
    neighbors, so connected-component segmentation can't merge glyphs
    regardless of the font's natural kerning -- keeps Stage B tests
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


class TestFindCandidates:
    def test_finds_the_motivating_case(self):
        cands = rc.find_candidates("Open 9am to Spm")
        assert len(cands) == 1
        c = cands[0]
        assert c.orig_char == "S" and c.digit_char == "5"
        assert c.corrected_token == "5pm"
        assert c.token_text == "Spm"

    def test_proposes_but_does_not_reject_sam_at_stage_a(self):
        """Stage A is context-only and CANNOT distinguish "Sam" (a name)
        from "Spm" (a misread digit) -- this is the exact gap the naive
        regex-substitution plan had. Stage A alone still proposes a
        candidate here; safety comes from Stage B refusing to confirm it
        (see TestCorrectInstances.test_sam_is_never_rewritten below)."""
        cands = rc.find_candidates("Sam")
        assert len(cands) == 1
        assert cands[0].orig_char == "S" and cands[0].digit_char == "5"

    def test_no_candidate_without_am_pm_context(self):
        assert rc.find_candidates("$8 Special") == []
        assert rc.find_candidates("Big Ben") == []

    def test_already_correct_digit_yields_no_candidate(self):
        assert rc.find_candidates("Open 9am to 5pm") == []
        assert rc.find_candidates("12pm") == []

    def test_invalid_hour_after_substitution_is_rejected(self):
        # 'B' -> '8' would give "18pm", not a valid 1-12 hour
        assert rc.find_candidates("1Bpm") == []

    def test_multi_position_ambiguity_is_skipped_not_guessed(self):
        # both characters are confusable letters -- more than Stage B
        # can verify as a single glyph, so no candidate is proposed
        assert rc.find_candidates("BBpm") == []

    def test_two_char_prefix_single_ambiguous_position(self):
        cands = rc.find_candidates("1Zpm")
        assert len(cands) == 1
        assert cands[0].orig_char == "Z" and cands[0].digit_char == "2"
        assert cands[0].corrected_token == "12pm"


class TestVerifyCandidate:
    def test_confirms_a_real_digit_misread_as_letter(self):
        # pixels show the TRUE glyph '5' (what a "5pm" sign really looks
        # like); the manifest's recognized text is the misread "Spm"
        img, bbox = render_spaced_text("5pm")
        inst = make_inst("Spm", bbox)
        cand = rc.find_candidates("Spm")[0]
        verdict = rc.verify_candidate(img, inst, cand)
        assert verdict is True

    def test_rejects_when_pixels_show_the_original_letter(self):
        img, bbox = render_spaced_text("Sam")
        inst = make_inst("Sam", bbox)
        cand = rc.find_candidates("Sam")[0]
        verdict = rc.verify_candidate(img, inst, cand)
        assert verdict is not True  # False or None -- either way, not confirmed

    def test_none_on_degenerate_crop(self):
        inst = make_inst("Spm", BBox(x=0, y=0, width=1, height=1))
        cand = rc.find_candidates("Spm")[0]
        img = Image.new("RGB", (50, 50), (240, 240, 240))
        assert rc.verify_candidate(img, inst, cand) is None


class TestCorrectInstances:
    def test_fixes_the_motivating_case_end_to_end(self):
        img, bbox = render_spaced_text("5pm")
        inst = make_inst("Spm", bbox)
        n = rc.correct_instances(img, [inst])
        assert n == 1
        assert inst.text == "5pm"
        assert inst.ocr_correction is not None
        assert inst.ocr_correction["applied"] is True
        assert inst.ocr_correction["original_text"] == "Spm"

    def test_sam_is_never_rewritten(self):
        """the concrete safety regression: a naive context-only fix
        rewrites 'Sam' to '5am'. this must not happen once pixel
        evidence (an actual rendered 'S', not a '5') is checked."""
        img, bbox = render_spaced_text("Sam")
        inst = make_inst("Sam", bbox)
        n = rc.correct_instances(img, [inst])
        assert n == 0
        assert inst.text == "Sam"

    def test_no_candidate_leaves_ocr_correction_unset(self):
        img, bbox = render_spaced_text("Big Ben")
        inst = make_inst("Big Ben", bbox)
        rc.correct_instances(img, [inst])
        assert inst.text == "Big Ben"
        assert inst.ocr_correction is None

    def test_empty_text_is_a_no_op(self):
        inst = make_inst(None, BBox(x=0, y=0, width=10, height=10))
        assert rc.correct_instances(Image.new("RGB", (50, 50)), [inst]) == 0

    def test_already_correct_time_is_untouched(self):
        img, bbox = render_spaced_text("5pm")
        inst = make_inst("5pm", bbox)
        n = rc.correct_instances(img, [inst])
        assert n == 0
        assert inst.text == "5pm"
        assert inst.ocr_correction is None
