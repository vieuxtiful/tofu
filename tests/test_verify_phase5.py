## 🍢 verify: coverage accounting, residual-text penalty, style consistency
import numpy as np
from PIL import Image

from tofu.core.types import (
    BBox, BgProfil, CharactText, InstText, Mask, StyleProfil, TextManifest,
)
from tofu.layers import verify


def make_asset(w=300, h=200, color=(240, 240, 240)):
    return Image.new("RGB", (w, h), color)


def make_manifest(instances, targ_lang="en"):
    return TextManifest(asset_id="a", total_regions=len(instances),
                        instances=instances, src_lang="en", targ_lang=targ_lang)


def text_inst(id="r1", x=50, y=50, w=120, h=40, text="HELLO", target=None,
              dnt=False, style_profile=None, characteristics=None):
    return InstText(
        id=id, bounding_box=BBox(x=x, y=y, width=w, height=h),
        segmentation_mask=Mask(
            polygon=[(x, y), (x + w, y), (x + w, y + h), (x, y + h)],
            confidence=0.9,
        ),
        text=text, target_text=target, dnt=dnt,
        style_profile=style_profile, characteristics=characteristics,
    )


class TestCoverageAccounting:
    def test_progress_reports_full_breakdown(self):
        asset = make_asset()
        instances = [
            text_inst(id="r1", target="HI"),
            text_inst(id="r2", x=200, target=None),  # untranslated
            text_inst(id="r3", x=50, y=150, w=40, h=20, dnt=True),
        ]
        qa = verify.assess(asset, make_manifest(instances), asset)
        p = qa.progress
        assert p["regions_total"] == 3
        assert p["dnt"] == 1
        assert p["translated"] == 1
        assert p["untranslated"] == 1
        assert p["rendered"] == 1
        assert p["fallback_font"] == 0

    def test_fallback_font_counted(self):
        asset = make_asset()
        inst = text_inst(target="HI")
        inst.glyph_fallback = True
        qa = verify.assess(asset, make_manifest([inst]), asset)
        assert qa.progress["fallback_font"] == 1


class TestUntranslatedDeduction:
    def test_untranslated_is_a_real_deduction_not_neutral(self):
        asset = make_asset()
        inst = text_inst(target=None)
        qa = verify.assess(asset, make_manifest([inst]), asset)
        assert qa.per_asset_instance_score["a"]["r1"] == verify.UNTRANSLATED_SCORE
        assert qa.per_asset_instance_score["a"]["r1"] != verify.NEUTRAL_SCORE

    def test_untranslated_region_gets_a_recommendation(self):
        asset = make_asset()
        inst = text_inst(target=None)
        qa = verify.assess(asset, make_manifest([inst]), asset)
        assert any("untranslated" in r.lower() for r in qa.recommendations)

    def test_untranslated_drags_down_overall_score(self):
        """the actual bug being fixed: an incomplete localization must
        not be able to inflate its own gate score by skipping regions."""
        asset = make_asset()
        instances = [
            text_inst(id="r1", target="HI"),      # will score well (DNT-adjacent baseline)
            text_inst(id="r2", x=200, target=None),  # untranslated -> 0.0
        ]
        qa = verify.assess(asset, make_manifest(instances), asset)
        # with one perfect-ish region and one at 0.0, overall must be
        # meaningfully below what an all-NEUTRAL treatment would give (1.0)
        assert qa.overall_score < 0.9

    def test_dnt_still_neutral(self):
        asset = make_asset()
        inst = text_inst(dnt=True, target=None)
        qa = verify.assess(asset, make_manifest([inst]), asset)
        assert qa.per_asset_instance_score["a"]["r1"] == verify.NEUTRAL_SCORE


class TestResidualSourceText:
    def _rendered_pair(self, leave_source_visible: bool):
        """a 'cleansed' crop that either still shows the source text
        (simulated: draw source-looking text) or is genuinely blank."""
        from PIL import ImageDraw, ImageFont
        base = make_asset()
        if leave_source_visible:
            draw = ImageDraw.Draw(base)
            try:
                font = ImageFont.truetype("arial.ttf", 28)
            except Exception:
                font = ImageFont.load_default()
            draw.text((55, 55), "HELLO", font=font, fill=(0, 0, 0))
        return base

    def test_residual_score_none_without_cleansed_asset(self):
        asset = make_asset()
        inst = text_inst(target="HI")
        qa = verify.assess(asset, make_manifest([inst]), asset, cleansed_asset=None)
        assert qa.metrics["residual_text"] == {}

    def test_clean_erase_no_penalty_applied(self):
        asset = make_asset()
        cleansed = make_asset()  # blank -- nothing left to OCR
        inst = text_inst(target="HELLO")
        qa_with = verify.assess(asset, make_manifest([inst]), asset, cleansed_asset=cleansed)
        qa_without = verify.assess(asset, make_manifest([inst]), asset, cleansed_asset=None)
        # a blank cleansed crop should not depress the score relative to
        # not having the metric at all (no residual detected -> no penalty)
        assert qa_with.per_asset_instance_score["a"]["r1"] == \
            qa_without.per_asset_instance_score["a"]["r1"]


class TestStyleColorScore:
    def test_matching_color_scores_high(self):
        # ink INSET within the bbox (background margin on all sides) --
        # text_mask's Otsu split needs both classes present in the crop;
        # an ink block that exactly fills the bbox has no background
        # pixels to separate against and correctly returns no mask
        img = np.full((200, 300, 3), 240, dtype=np.uint8)
        img[60:80, 60:160] = (20, 20, 20)  # dark "ink" block, margin all sides
        localized = Image.fromarray(img)
        inst = text_inst(target="HELLO", style_profile=StyleProfil(color="#141414"))
        score = verify._style_color_score(np, np.asarray(localized), inst)
        assert score is not None and score > 0.9

    def test_mismatched_color_scores_low(self):
        img = np.full((200, 300, 3), 240, dtype=np.uint8)
        img[60:80, 60:160] = (20, 20, 200)  # blue ink, margin all sides
        localized = Image.fromarray(img)
        inst = text_inst(target="HELLO", style_profile=StyleProfil(color="#ff0000"))  # expects red
        score = verify._style_color_score(np, np.asarray(localized), inst)
        assert score is not None and score < 0.5

    def test_no_color_hint_returns_none(self):
        img = np.full((200, 300, 3), 240, dtype=np.uint8)
        inst = text_inst(target="HELLO", style_profile=StyleProfil())
        assert verify._style_color_score(np, img, inst) is None


class TestStyleSizeScore:
    def test_matching_size_scores_high(self):
        img = np.full((200, 300, 3), 240, dtype=np.uint8)
        img[60:80, 60:160] = (20, 20, 20)  # 20px-tall ink block, margin all sides
        inst = text_inst(target="HELLO", characteristics=CharactText(size=20))
        score = verify._style_size_score(np, img, inst)
        assert score is not None and score > 0.85

    def test_mismatched_size_scores_low(self):
        img = np.full((200, 300, 3), 240, dtype=np.uint8)
        img[50:60, 50:170] = (20, 20, 20)  # 10px-tall ink vs detected 80px
        inst = text_inst(target="HELLO", characteristics=CharactText(size=80))
        score = verify._style_size_score(np, img, inst)
        assert score is not None and score < 0.3

    def test_no_detected_size_returns_none(self):
        img = np.full((200, 300, 3), 240, dtype=np.uint8)
        inst = text_inst(target="HELLO", characteristics=None)
        assert verify._style_size_score(np, img, inst) is None


class TestColorMath:
    def test_identical_colors_zero_delta_e(self):
        assert verify._delta_e_cie76((10, 20, 30), (10, 20, 30)) == 0.0

    def test_black_vs_white_large_delta_e(self):
        de = verify._delta_e_cie76((0, 0, 0), (255, 255, 255))
        assert de > 50  # CIE76 black-white is ~100

    def test_parse_hex_color_variants(self):
        assert verify._parse_hex_color("#fff") == (255, 255, 255)
        assert verify._parse_hex_color("#ff0000") == (255, 0, 0)
        assert verify._parse_hex_color(None) is None
        assert verify._parse_hex_color("not-a-color") is None
