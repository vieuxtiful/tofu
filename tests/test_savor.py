## 🍢 Savor: sniff_out (course 1), chew_on (course 2), taste (courses 1-3)
from pathlib import Path

import cv2
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


def make_inst(text, bbox, id="r1", confidence=0.95, detected_language=None):
    return InstText(id=id, bounding_box=bbox, text=text, confidence=confidence,
                     detected_language=detected_language)


def _font_registry():
    """real FontRegistry over the Windows system font directory -- skips
    on hosts without it (mirrors _font()'s arial.ttf skip pattern)."""
    from tofu.layers.fonts import FontRegistry
    font_dir = "C:/Windows/Fonts"
    if not Path(font_dir).exists():
        pytest.skip("no system font directory available on this host")
    fr = FontRegistry(font_dir)
    if not getattr(fr, "_fonts", None):
        pytest.skip("FontRegistry found no usable fonts on this host")
    return fr


def _ja_font(size):
    for name in ("BIZ-UDGothicB.ttc", "msgothic.ttc", "meiryo.ttc", "YuGothM.ttc"):
        try:
            return ImageFont.truetype(name, size)
        except Exception:
            continue
    pytest.skip("no Japanese-capable font available on this host")


def render_vertical_text(text, font_size=36, gap=16, margin=20):
    """draw each character stacked top-to-bottom with a guaranteed pixel
    gap, mirroring render_spaced_text()'s horizontal layout but for
    vertical CJK columns (see _is_vertical_instance).

    a dakuten/handakuten mark is typically rendered as its own
    disconnected blob a couple pixels from its base glyph's body (a
    real photo's blur/antialiasing would normally fuse the two) -- a
    light morphological close bridges that same small gap here without
    touching the character-to-character gap (much larger, via `gap`),
    so _plate_up segments exactly one component per character."""
    font = _ja_font(font_size)
    heights = []
    for ch in text:
        bbox = font.getbbox(ch)
        heights.append(bbox[3] - bbox[1])
    total_h = sum(heights) + gap * (len(text) - 1) + margin * 2
    total_w = font_size * 2 + margin * 2
    img = Image.new("L", (total_w, total_h), 235)
    draw = ImageDraw.Draw(img)
    x = margin
    y = margin
    for ch, h in zip(text, heights):
        draw.text((x, y), ch, font=font, fill=20)
        y += h + gap
    arr = np.array(img)
    _, otsu = cv2.threshold(arr, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    fg = (otsu == 0).astype(np.uint8) * 255  # text is dark (minority) on light background
    closed = cv2.morphologyEx(fg, cv2.MORPH_CLOSE, np.ones((4, 4), np.uint8))
    out = np.where(closed > 0, 20, 235).astype(np.uint8)
    img2 = Image.fromarray(np.stack([out] * 3, axis=-1))
    bbox = BBox(x=0, y=0, width=total_w, height=total_h)
    return img2, bbox


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


class TestReferenceBiteFontFix:
    """regression guard for the font-resolution bug found live: without
    a font_registry, _reference_bite's Latin-only fallback chain has no
    Japanese glyph and silently renders identical "missing glyph"
    placeholder boxes for every kana -- IoU 1.0 between all of them."""

    def test_font_registry_resolves_distinguishable_ja_glyphs(self):
        fr = _font_registry()
        ha = savor._reference_bite(np, "ハ", 30, font_registry=fr, lang="ja")
        ba = savor._reference_bite(np, "バ", 30, font_registry=fr, lang="ja")
        if ha is None or ba is None:
            pytest.skip("no Japanese-capable font resolved on this host")
        assert ha.shape != ba.shape  # dakuten mark visibly extends バ's bbox over ハ's

    def test_no_font_registry_keeps_digit_course_behavior_unchanged(self):
        # this is exactly how chew_on() still calls _reference_bite for
        # the digit/letter course -- must remain untouched. without a
        # font_registry, a CJK glyph still "renders" (a missing-glyph
        # placeholder box, not None) -- the bug was that this placeholder
        # is IDENTICAL regardless of which character was requested,
        # which is what made pre-fix pairwise IoU always 1.0.
        five = savor._reference_bite(np, "5", 30)
        assert five is not None  # Latin digits ARE covered by FALLBACK_FONTS
        ha = savor._reference_bite(np, "ハ", 30)
        ba = savor._reference_bite(np, "バ", 30)
        assert ha is not None and ba is not None
        assert np.array_equal(ha, ba)  # same placeholder box -- no real CJK coverage without a registry


class TestSniffDakuten:
    """course 4's smell test: only fires when toggling dakuten/
    handakuten marks would turn the text into a KNOWN real name."""

    def test_no_trigger_for_non_japanese_lang(self, monkeypatch):
        monkeypatch.setattr("tofu.layers.menu.KNOWN_PLACES", [("バンダイ", "ja")])
        assert savor.sniff_dakuten("ハンタイ", "en") == []
        assert savor.sniff_dakuten("ハンタイ", None) == []

    def test_no_trigger_when_diff_is_not_a_dakuten_relation(self, monkeypatch):
        # "ハンタイ" vs "ハンライ" differs at a position that isn't a
        # dakuten toggle (タ vs ラ) -- must not guess
        monkeypatch.setattr("tofu.layers.menu.KNOWN_PLACES", [("ハンライ", "ja")])
        assert savor.sniff_dakuten("ハンタイ", "ja") == []

    def test_two_position_bandai_case(self, monkeypatch):
        # the real motivating case: ハ->バ AND タ->ダ, not a single
        # character -- regression guard for the initially-assumed
        # single-position diff, which was wrong
        monkeypatch.setattr("tofu.layers.menu.KNOWN_PLACES", [("バンダイ", "ja")])
        morsels = savor.sniff_dakuten("ハンタイ", "ja")
        assert len(morsels) == 1
        assert morsels[0].positions == [(0, "ハ", "バ"), (2, "タ", "ダ")]
        assert morsels[0].candidate_word == "バンダイ"

    def test_already_correct_text_yields_no_morsel(self, monkeypatch):
        monkeypatch.setattr("tofu.layers.menu.KNOWN_PLACES", [("バンダイ", "ja")])
        assert savor.sniff_dakuten("バンダイ", "ja") == []

    def test_no_gazetteer_match_yields_no_morsel(self, monkeypatch):
        monkeypatch.setattr("tofu.layers.menu.KNOWN_PLACES", [("バンダイ", "ja")])
        assert savor.sniff_dakuten("xyz123", "ja") == []


class TestPlateUpVertical:
    """_plate_up's reading-order sort: left-to-right by default (the
    digit/letter course's only need), top-to-bottom when vertical=True
    (stacked CJK columns)."""

    def _mask_with_blobs(self, boxes):
        m = np.zeros((200, 200), dtype=bool)
        for x0, y0, x1, y1 in boxes:
            m[y0:y1, x0:x1] = True
        return m

    def test_default_sorts_left_to_right(self):
        import cv2
        # three blobs placed out of x-order
        mask = self._mask_with_blobs([(100, 10, 120, 30), (10, 10, 30, 30), (50, 10, 70, 30)])
        plated = savor._plate_up(np, cv2, mask)
        xs = [b[0] for b in plated]
        assert xs == sorted(xs)

    def test_vertical_sorts_top_to_bottom(self):
        import cv2
        # three blobs at the same x, placed out of y-order
        mask = self._mask_with_blobs([(10, 100, 30, 120), (10, 10, 30, 30), (10, 50, 30, 70)])
        plated = savor._plate_up(np, cv2, mask, vertical=True)
        ys = [b[1] for b in plated]
        assert ys == sorted(ys)


class TestChewDakuten:
    """course 4's bite: pixel evidence decides whether to trust the
    gazetteer-suggested dakuten toggle, position by position.

    uses a synthetic カキタク/ガキダク pair rather than the real バンダイ
    case -- ン renders as two visually disconnected strokes even in a
    clean vector font, breaking the 1:1 glyph-count assumption this
    (and the pre-existing digit course's chew_on) relies on; that's a
    test-fixture-fidelity problem, not a mechanism problem, and is
    already covered structurally by TestSniffDakuten's exact バンダイ
    position assertions (pure string logic, no pixel rendering needed
    there). カキタク has the same diff-same-diff-same shape (positions
    0 and 2 differ) and 0.5 gazetteer similarity as the real case, with
    every character a clean single connected blob."""

    def test_confirms_real_dakuten_marks(self, monkeypatch):
        fr = _font_registry()
        img, bbox = render_vertical_text("ガキダク")
        inst = make_inst("カキタク", bbox, detected_language="ja")
        monkeypatch.setattr("tofu.layers.menu.KNOWN_PLACES", [("ガキダク", "ja")])
        morsel = savor.sniff_dakuten("カキタク", "ja")[0]
        verdicts = savor.chew_dakuten(img, inst, morsel, font_registry=fr)
        assert verdicts.get(0) is True
        assert verdicts.get(2) is True

    def test_rejects_when_pixels_show_no_marks(self, monkeypatch):
        fr = _font_registry()
        img, bbox = render_vertical_text("カキタク")  # true pixels: marks genuinely absent
        inst = make_inst("カキタク", bbox, detected_language="ja")
        monkeypatch.setattr("tofu.layers.menu.KNOWN_PLACES", [("ガキダク", "ja")])
        morsel = savor.sniff_dakuten("カキタク", "ja")[0]
        verdicts = savor.chew_dakuten(img, inst, morsel, font_registry=fr)
        assert verdicts.get(0) is not True
        assert verdicts.get(2) is not True


class TestTasteDakutenEndToEnd:
    """course 4 through the public taste() entry point -- same
    カキタク/ガキダク pair as TestChewDakuten (see that class's
    docstring), exercised at a high (0.846-like) confidence that
    already defeats menu.py's confidence-gated gazetteer check --
    proving this course fires precisely where menu.py's can't."""

    def test_fixes_the_dakuten_case_end_to_end(self, monkeypatch):
        fr = _font_registry()
        img, bbox = render_vertical_text("ガキダク")
        inst = make_inst("カキタク", bbox, confidence=0.846, detected_language="ja")
        monkeypatch.setattr("tofu.layers.menu.KNOWN_PLACES", [("ガキダク", "ja")])
        n = savor.taste(img, [inst], font_registry=fr)
        assert n == 1
        assert inst.text == "ガキダク"
        assert inst.ocr_correction is not None
        assert inst.ocr_correction["applied"] is True

    def test_without_font_registry_dakuten_course_is_a_no_op(self, monkeypatch):
        # fail-open: omitting font_registry (every pre-existing caller's
        # call shape) must never crash and must never apply a dakuten
        # correction, since _reference_bite can't render CJK without it
        img, bbox = render_vertical_text("ガキダク") if Path("C:/Windows/Fonts").exists() else (
            Image.new("RGB", (50, 200), (240, 240, 240)), BBox(x=0, y=0, width=50, height=200)
        )
        inst = make_inst("カキタク", bbox, confidence=0.846, detected_language="ja")
        monkeypatch.setattr("tofu.layers.menu.KNOWN_PLACES", [("ガキダク", "ja")])
        n = savor.taste(img, [inst])  # no font_registry
        assert n == 0
        assert inst.text == "カキタク"


class TestChewSwaps:
    """the generic per-position swap verifier both the dakuten course
    and menu's substring gazetteer share (chew_dakuten is a thin
    wrapper over it -- the dakuten tests above double as its coverage
    for the multi-position path)."""

    def test_confirms_arbitrary_non_dakuten_swap(self):
        # a swap outside DAKUTEN_MAP entirely: true pixels カ, proposal
        # asks "is this really カ, or the recognized タ?" -- pixels must
        # favor カ (the proposed char). three chars so the column is
        # decisively taller-than-wide (VERTICAL_ASPECT_MIN) and glyphs
        # plate up in top-to-bottom order.
        fr = _font_registry()
        img, bbox = render_vertical_text("カキク")
        inst = make_inst("タキク", bbox, detected_language="ja")
        verdicts = savor.chew_swaps(img, inst, [(0, "タ", "カ")], "ja", font_registry=fr)
        assert verdicts.get(0) is True

    def test_degenerate_crop_returns_empty(self):
        inst = make_inst("カキ", BBox(x=0, y=0, width=1, height=1), detected_language="ja")
        img = Image.new("RGB", (50, 50), (240, 240, 240))
        assert savor.chew_swaps(img, inst, [(0, "タ", "カ")], "ja") == {}


class TestChewSwapsSmallGlyphUpscale:
    """chew_swaps' small-glyph rescue: below GLYPH_UPSCALE_MIN_PX per
    glyph, the crop is upscaled before masking -- street-photo-scale
    glyphs (~13-18px) under-segment at native resolution and previously
    came back wholly unverifiable."""

    def _render_small_vertical(self, text, font_size=12, gap=5, margin=6):
        font = _ja_font(font_size)
        heights = []
        for ch in text:
            bbox = font.getbbox(ch)
            heights.append(bbox[3] - bbox[1])
        total_h = sum(heights) + gap * (len(text) - 1) + margin * 2
        total_w = font_size * 2 + margin * 2
        img = Image.new("RGB", (total_w, total_h), (235, 235, 230))
        draw = ImageDraw.Draw(img)
        y = margin
        for ch, h in zip(text, heights):
            draw.text((margin, y), ch, font=font, fill=(20, 20, 20))
            y += h + gap
        return img, BBox(x=0, y=0, width=total_w, height=total_h)

    def test_small_glyphs_verifiable_via_upscale(self):
        fr = _font_registry()
        img, bbox = self._render_small_vertical("ガキダク")
        # per-glyph extent must actually be under the upscale threshold,
        # or this test isn't exercising the rescue path at all
        assert bbox.height / 4 < savor.GLYPH_UPSCALE_MIN_PX
        inst = make_inst("カキタク", bbox, detected_language="ja")
        verdicts = savor.chew_swaps(
            img, inst, [(0, "カ", "ガ"), (2, "タ", "ダ")], "ja", font_registry=fr
        )
        assert verdicts.get(0) is True
        assert verdicts.get(2) is True
