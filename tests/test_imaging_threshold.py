## 🍢 text_mask binarization: global Otsu vs local Sauvola selection
import numpy as np
import pytest
from PIL import Image, ImageDraw, ImageFont

import cv2

from tofu.core.types import BBox
from tofu.utils.imaging import (
    UNEVEN_SOAK_FLOOR,
    _nigari_mask,
    _uneven_soak,
    text_mask,
)


def lettered_crop(gradient: bool, text: str = "TOFU RAMEN"):
    """A text crop plus its ground-truth glyph mask, optionally lit unevenly.

    The gradient is a left-to-right falloff to 30% brightness — a lighting
    ramp of the kind that makes a single global threshold impossible, and
    which real street signage produces routinely.
    """
    width, height = 320, 90
    canvas = Image.new("L", (width, height), 235)
    draw = ImageDraw.Draw(canvas)
    try:
        font = ImageFont.truetype("arial.ttf", 48)
    except OSError:
        font = ImageFont.load_default()
    draw.text((12, 16), text, font=font, fill=25)
    ink = np.asarray(canvas) < 128

    lit = np.asarray(canvas).astype(np.float32)
    if gradient:
        lit = lit * np.linspace(1.0, 0.30, width, dtype=np.float32)[None, :]
    lit = np.clip(lit, 0, 255).astype(np.uint8)
    return np.dstack([lit] * 3), ink


def iou(mask, truth) -> float:
    union = np.logical_or(mask, truth).sum()
    return float(np.logical_and(mask, truth).sum()) / float(union) if union else 0.0


def full_bbox(rgb) -> BBox:
    return BBox(x=0, y=0, width=rgb.shape[1], height=rgb.shape[0])


class TestUnevenSoak:
    def test_flat_lighting_scores_low_despite_strong_ink(self):
        # the measure must track ILLUMINATION, not ink contrast: blurring
        # away the glyphs leaves a constant on an evenly lit crop
        rgb, _ = lettered_crop(gradient=False)
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        assert _uneven_soak(gray, cv2) < UNEVEN_SOAK_FLOOR

    def test_lighting_gradient_scores_high(self):
        rgb, _ = lettered_crop(gradient=True)
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        assert _uneven_soak(gray, cv2) > UNEVEN_SOAK_FLOOR

    def test_uniform_crop_scores_near_zero(self):
        gray = np.full((80, 80), 128, dtype=np.uint8)
        assert _uneven_soak(gray, cv2) == pytest.approx(0.0, abs=1.0)


class TestSelection:
    def test_gradient_crop_beats_otsu_substantially(self):
        """The whole point: Otsu swallows the shaded end of the ramp as ink."""
        rgb, truth = lettered_crop(gradient=True)
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        _, otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        otsu_fg = otsu > 0
        n = int(otsu_fg.sum())
        otsu_fg = otsu_fg if n <= otsu_fg.size - n else ~otsu_fg

        hybrid = text_mask(rgb, full_bbox(rgb), refine=False)
        assert hybrid is not None
        assert iou(hybrid, truth) > 0.8
        assert iou(hybrid, truth) > iou(otsu_fg, truth) * 2

    def test_flat_crop_keeps_otsu_quality(self):
        """Sauvola's window noise must not be paid where Otsu already wins."""
        rgb, truth = lettered_crop(gradient=False)
        hybrid = text_mask(rgb, full_bbox(rgb), refine=False)
        assert hybrid is not None
        assert iou(hybrid, truth) > 0.95
        # and it should genuinely be the Otsu answer, not Sauvola coincidentally
        assert iou(hybrid, truth) > iou(_nigari_mask(
            cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY), np), truth)


class TestContractPreserved:
    def test_still_returns_minority_class(self):
        rgb, _ = lettered_crop(gradient=True)
        mask = text_mask(rgb, full_bbox(rgb), refine=False)
        assert mask is not None
        assert mask.sum() < mask.size / 2

    def test_uniform_crop_still_returns_none(self):
        flat = np.full((100, 100, 3), 128, dtype=np.uint8)
        assert text_mask(flat, BBox(x=10, y=10, width=60, height=60)) is None

    def test_degenerate_bbox_still_returns_none(self):
        rgb, _ = lettered_crop(gradient=True)
        assert text_mask(rgb, BBox(x=0, y=0, width=2, height=2)) is None

    def test_mask_shape_matches_crop(self):
        rgb, _ = lettered_crop(gradient=True)
        mask = text_mask(rgb, BBox(x=5, y=5, width=100, height=40), refine=False)
        assert mask is not None
        assert mask.shape == (40, 100)


class TestSauvolaDegradation:
    def test_tiny_crop_does_not_raise(self):
        """Window sizing must stay odd and inside the crop on small regions."""
        rgb, _ = lettered_crop(gradient=True)
        for side in (4, 5, 7, 12):
            mask = text_mask(rgb, BBox(x=0, y=0, width=side, height=side), refine=False)
            assert mask is None or mask.shape == (side, side)

    def test_falls_back_to_otsu_when_skimage_absent(self, monkeypatch):
        """Dependency-soft contract: no scikit-image must mean Otsu, not None."""
        import builtins

        real_import = builtins.__import__

        def no_skimage(name, *args, **kwargs):
            if name.startswith("skimage"):
                raise ImportError("simulated: scikit-image unavailable")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", no_skimage)
        rgb, _ = lettered_crop(gradient=True)
        assert _nigari_mask(cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY), np) is None
        mask = text_mask(rgb, full_bbox(rgb), refine=False)
        assert mask is not None  # degraded to Otsu rather than failing
