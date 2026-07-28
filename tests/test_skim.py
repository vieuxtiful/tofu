"""skim — the stroke-thickness plausibility gate.

Synthetic crops rather than photographs: the property under test is
"strokes this thin relative to their own height are not lettering", and a
drawn bar states that property exactly, with no OCR in the loop. The
measured photographic values these thresholds came from are recorded in
skim.py's module docstring.
"""

import numpy as np
import pytest

from tofu.core.types import BBox, InstText
from tofu.layers import skim, typography


def _bar_image(stroke_px: int, height: int = 60, width: int = 40):
    """White field with one dark vertical bar of a known width."""
    img = np.full((height + 20, width + 20, 3), 255, dtype=np.uint8)
    x0 = 10 + (width - stroke_px) // 2
    img[10:10 + height, x0:x0 + stroke_px] = 20
    return img


def _box(height: int = 60, width: int = 40) -> BBox:
    return BBox(x=10, y=10, width=width, height=height)


def _inst(text: str, conf: float) -> InstText:
    return InstText(id="r1", bounding_box=_box(), text=text, confidence=conf)


class TestStrokeMeasurement:
    def test_a_hairline_measures_below_the_light_floor(self):
        ratio = skim.stroke_ratio(_bar_image(2), _box())
        assert ratio is not None
        assert ratio < typography.LIGHT_RATIO

    def test_a_real_stroke_measures_above_the_light_floor(self):
        ratio = skim.stroke_ratio(_bar_image(10), _box())
        assert ratio is not None
        assert ratio > typography.LIGHT_RATIO

    def test_the_floor_is_typographys_own_constant_not_a_second_copy(self):
        # if these ever diverge, the weight classifier and this layer would
        # disagree about what counts as lettering
        assert skim.stroke_floor() == typography.LIGHT_RATIO


class TestScumVerdict:
    def test_a_short_weak_hairline_read_is_scum(self):
        verdict, reason = skim.is_scum(_inst("7", 0.447), _bar_image(2))
        assert verdict is True
        assert "LIGHT_RATIO" in reason

    def test_a_short_weak_read_with_real_strokes_survives(self):
        verdict, _ = skim.is_scum(_inst("7", 0.447), _bar_image(10))
        assert verdict is False

    @pytest.mark.parametrize("text", ["777", "est_ 1962", "ABCD"])
    def test_a_longer_read_is_never_examined_however_thin(self, text):
        # "est. 1962" on the textured-wall fixture is REAL and measures
        # 0.0637, below the floor. The length gate is the only thing
        # standing between this layer and deleting it.
        verdict, _ = skim.is_scum(_inst(text, 0.447), _bar_image(2))
        assert verdict is False

    def test_a_confident_read_is_never_examined_however_thin(self):
        verdict, _ = skim.is_scum(_inst("7", 0.95), _bar_image(2))
        assert verdict is False

    def test_an_empty_read_is_left_to_the_existing_prune(self):
        verdict, _ = skim.is_scum(_inst("   ", 0.4), _bar_image(2))
        assert verdict is False


class TestFailsOpen:
    """A precision pass must never cost recall when it cannot measure."""

    def test_an_unloadable_asset_keeps_the_region(self):
        verdict, _ = skim.is_scum(_inst("7", 0.4), object())
        assert verdict is False

    def test_a_missing_bbox_keeps_the_region(self):
        inst = InstText(id="r1", bounding_box=None, text="7", confidence=0.4)
        assert skim.is_scum(inst, _bar_image(2))[0] is False

    def test_an_unmeasurable_crop_keeps_the_region(self):
        # a flat field has no separable ink at all, so the ratio is None
        flat = np.full((80, 60, 3), 128, dtype=np.uint8)
        verdict, _ = skim.is_scum(_inst("7", 0.4), flat)
        assert verdict is False


class TestScriptBearingReadsAreNeverOffered:
    """The caller contract, enforced where cicerone applies it.

    Regression guard with a measured cost: skim's floor comes from Latin
    typography, and CJK glyphs are built from many thin strokes inside a
    dense square.  Run against every read rather than only script-less
    ones, it removed the real Hangul regions 윗 / 줄 / 꿀식 / 주 from
    gemini-street and took recall from 0.333 to 0.278.
    """

    def _pruned(self, text):
        from tofu.layers.cicerone import _prune_hallucinations
        inst = InstText(id="r1", bounding_box=_box(), text=text, confidence=0.447)
        return _prune_hallucinations([inst], _bar_image(2))

    @pytest.mark.parametrize("text", ["줄", "주", "歌", "A"])
    def test_a_thin_script_bearing_read_survives(self, text):
        assert len(self._pruned(text)) == 1

    def test_a_thin_script_less_digit_read_is_still_lifted(self):
        # the phantom this layer exists for: a digit carries no script
        # evidence, so nothing contradicts the stroke measurement
        assert self._pruned("7") == []


class TestSkimReturnsRatherThanMutates:
    def test_the_caller_owns_the_manifest(self):
        good = _inst("AVENUE", 0.99)
        bad = _inst("7", 0.447)
        lifted = skim.skim([good, bad], _bar_image(2))
        assert [inst for inst, _ in lifted] == [bad]
        # nothing on the inputs was changed by the decision itself
        assert bad.text == "7" and bad.confidence == 0.447
