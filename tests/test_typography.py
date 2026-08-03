## 🍢 typography estimation against fixture ground truth
import json

import numpy as np
import pytest
from PIL import Image, ImageDraw, ImageFont

from conftest import FIXTURES
from tofu.core.types import BBox
from tofu.layers.typography import TypographyProfile, analyze_region
from tofu.utils.imaging import text_mask


def fixture_regions(name: str):
    img = np.asarray(Image.open(FIXTURES / f"{name}.png").convert("RGB"))
    gt = json.loads((FIXTURES / f"{name}.gt.json").read_text(encoding="utf-8"))
    return img, gt["regions"]


def padded_bbox(r, pad=4):
    x, y, w, h = r["bbox"]
    return BBox(x=x - pad, y=y - pad, width=w + 2 * pad, height=h + 2 * pad)


class TestTextMask:
    def test_mask_covers_glyphs_not_background(self):
        img, regions = fixture_regions("flat-sign")
        mask = text_mask(img, padded_bbox(regions[0]))
        assert mask is not None
        frac = mask.sum() / mask.size
        assert 0.02 < frac < 0.6  # strokes are the minority class

    def test_degenerate_bbox_returns_none(self):
        img, _ = fixture_regions("flat-sign")
        assert text_mask(img, BBox(x=0, y=0, width=2, height=2)) is None

    def test_uniform_crop_returns_none(self):
        img = np.full((100, 100, 3), 128, dtype=np.uint8)
        assert text_mask(img, BBox(x=10, y=10, width=60, height=60)) is None


class TestWeightAcrossTypefaces:
    """serif-vs-sans is the fixture that catches a typeface-confounded weight.

    stylized-italic sets everything in one family, so a weight estimator can
    pass it while being wrong about what it is measuring. serif-vs-sans sets
    Times against Arial, regular against bold, and mixes strings that have
    descenders with strings that do not -- which is what exposed the
    estimator dividing stroke width by the full ink extent: 'Handgloves' in
    Times BOLD measured 0.1001 against 'SANS-NOM' in Arial REGULAR at
    0.1139, and every bold region on the fixture was called regular.
    """

    @pytest.fixture(scope="class")
    @classmethod
    def analyzed(cls):
        img, regions = fixture_regions("serif-vs-sans")
        return [(r, analyze_region(img, padded_bbox(r))) for r in regions]

    def test_every_region_is_analyzable(self, analyzed):
        assert all(p is not None for _, p in analyzed)

    def test_weight_matches_ground_truth(self, analyzed):
        wrong = [
            f"{r['text']!r} ({r['style']['font_file']}): "
            f"{p.weight} != {r['style']['weight']} at ratio {p.stroke_ratio}"
            for r, p in analyzed if p.weight != r["style"]["weight"]
        ]
        assert not wrong, "\n".join(wrong)

    def test_a_descender_does_not_make_lettering_look_lighter(self, analyzed):
        """The bug this fixture caught, pinned directly.

        'Handgloves' has an ascender and a descender; 'SANS-NOM' has
        neither. Measuring against total ink extent made the first look
        thinner than the second even when it was the bolder of the two.
        """
        by_text = {r["text"]: (r, p) for r, p in analyzed}
        bold_desc = by_text["Handgloves"][1]
        regular_caps = [
            p for r, p in analyzed
            if r["text"] == "SANS-NOM" and r["style"]["weight"] == "regular"
        ]
        assert regular_caps
        assert bold_desc.stroke_ratio > max(p.stroke_ratio for p in regular_caps)


class TestWeightAndItalic:
    """stylized-italic fixture carries exact style ground truth."""

    @pytest.fixture(scope="class")
    @classmethod
    def analyzed(cls):
        img, regions = fixture_regions("stylized-italic")
        return [
            (r, analyze_region(img, padded_bbox(r)))
            for r in regions
        ]

    def test_all_regions_analyzable(self, analyzed):
        assert all(p is not None for _, p in analyzed)

    def test_weight_matches_ground_truth(self, analyzed):
        for r, p in analyzed:
            expected = r["style"]["weight"]
            assert p.weight == expected, f"{r['text']}: {p.weight} != {expected}"

    def test_italic_matches_ground_truth(self, analyzed):
        for r, p in analyzed:
            assert p.italic == r["style"]["italic"], r["text"]

    def test_axis_aligned_text_has_zero_rotation(self, analyzed):
        for r, p in analyzed:
            assert p.rotation_deg == 0.0, r["text"]

    def test_font_px_within_tolerance(self, analyzed):
        for r, p in analyzed:
            gt_h = r["bbox"][3]
            assert p.font_px is not None
            assert abs(p.font_px - gt_h) <= gt_h * 0.35, r["text"]


class TestRotationVsItalic:
    def _rotated_sample(self, deg: float):
        base = Image.new("RGB", (400, 160), (240, 240, 240))
        d = ImageDraw.Draw(base)
        try:
            f = ImageFont.truetype("arialbd.ttf", 40)
        except Exception:
            f = ImageFont.load_default()
        d.text((60, 55), "ROTATED", font=f, fill=(20, 20, 20))
        rot = base.rotate(-deg, resample=Image.BICUBIC, fillcolor=(240, 240, 240))
        return np.asarray(rot)

    def test_rotation_detected(self):
        p = analyze_region(self._rotated_sample(8), BBox(x=30, y=20, width=340, height=120))
        assert p.rotation_deg == pytest.approx(8.0, abs=2.0)

    def test_rotated_text_is_not_italic(self):
        # baseline lean must not masquerade as italic
        p = analyze_region(self._rotated_sample(8), BBox(x=30, y=20, width=340, height=120))
        assert p.italic is False


class TestLabel:
    def test_bold_italic_label(self):
        assert TypographyProfile(weight="bold", italic=True).label() == "bold italic"

    def test_regular_label(self):
        assert TypographyProfile(weight="regular", italic=False).label() == "regular"

    def test_empty_profile_has_no_label(self):
        assert TypographyProfile().label() is None
