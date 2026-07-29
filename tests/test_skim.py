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


class TestCharDensity:
    def test_a_vertical_column_is_measured_down_its_long_axis(self):
        # a vertical CJK column packs characters down the height; dividing
        # its narrow width by the count would condemn every one of them
        tall = InstText(id="r1", bounding_box=BBox(x=0, y=0, width=20, height=200),
                        text="ABCD", confidence=0.9)
        assert skim.char_density(tall) == 50.0
        assert skim.needs_arbitration(tall) is False

    def test_a_box_too_thin_to_draw_its_characters_is_nominated(self):
        # measured on japan-street: "2932" inside an 11x21 box
        thin = InstText(id="r1", bounding_box=BBox(x=0, y=0, width=11, height=21),
                        text="2932", confidence=0.656)
        assert skim.char_density(thin) < skim.MIN_PX_PER_CHAR
        assert skim.needs_arbitration(thin) is True

    @pytest.mark.parametrize("w,h,text", [(25, 15, "3F"), (19, 11, "2F"), (155, 51, "AVENUE")])
    def test_measured_real_reads_are_never_nominated(self, w, h, text):
        inst = InstText(id="r1", bounding_box=BBox(x=0, y=0, width=w, height=h),
                        text=text, confidence=0.5)
        assert skim.needs_arbitration(inst) is False


class TestSkimAudit:
    """The cross-engine veto. Paddle is monkeypatched: its inference is not
    run-to-run deterministic, so a live call has no place in this suite."""

    def _run(self, instances, paddle_returns, available=True):
        from unittest.mock import patch
        from tofu.layers import cicerone

        def fake_regions(self, asset, bboxes, **kw):
            return [paddle_returns for _ in bboxes]

        with patch.object(cicerone.PaddleOCRBackend, "is_available", staticmethod(lambda: available)), \
             patch.object(cicerone.PaddleOCRBackend, "__init__", lambda self, **kw: None), \
             patch.object(cicerone.PaddleOCRBackend, "detect_in_regions", fake_regions):
            return cicerone.skim_audit("asset.png", instances)

    def _thin(self):
        return InstText(id="r1", bounding_box=BBox(x=0, y=0, width=11, height=21),
                        text="2932", confidence=0.656)

    def test_silence_plus_a_thin_box_removes_the_read(self):
        assert self._run([self._thin()], []) == []

    def test_a_second_engine_reading_text_vetoes_the_removal(self):
        from tofu.layers.cicerone import RawDetection
        seen = [RawDetection(polygon=[(0, 0), (11, 0), (11, 21), (0, 21)],
                             text="2932", confidence=0.9, language="en")]
        assert len(self._run([self._thin()], seen)) == 1

    def test_silence_alone_never_removes_a_normally_proportioned_read(self):
        # not nominated by density, so Paddle is never even asked about it
        fat = InstText(id="r1", bounding_box=BBox(x=0, y=0, width=200, height=40),
                       text="OPEN", confidence=0.5)
        assert len(self._run([fat], [])) == 1

    def test_a_script_bearing_read_is_never_a_candidate(self):
        han = InstText(id="r1", bounding_box=BBox(x=0, y=0, width=11, height=21),
                       text="歌舞伎町", confidence=0.6)
        assert len(self._run([han], [])) == 1

    def test_an_unavailable_paddle_changes_nothing(self):
        assert len(self._run([self._thin()], [], available=False)) == 1

    def test_survivors_are_renumbered_densely(self):
        keep = InstText(id="r1", bounding_box=BBox(x=0, y=0, width=200, height=40),
                        text="OPEN", confidence=0.9)
        out = self._run([keep, self._thin()], [])
        assert [i.id for i in out] == ["r1"]
        assert out[0].reading_order == 0


class TestSkimReturnsRatherThanMutates:
    def test_the_caller_owns_the_manifest(self):
        good = _inst("AVENUE", 0.99)
        bad = _inst("7", 0.447)
        lifted = skim.skim([good, bad], _bar_image(2))
        assert [inst for inst, _ in lifted] == [bad]
        # nothing on the inputs was changed by the decision itself
        assert bad.text == "7" and bad.confidence == 0.447
