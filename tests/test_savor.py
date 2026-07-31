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


class TestOcrQuality:
    def _plate(self, components, clusters):
        return np.ones((30, 80), dtype=bool), components, clusters

    def test_reliable_geometry_records_observability_evidence(self):
        inst = make_inst("AB", BBox(0, 0, 80, 30), detected_language="fr")
        components = [
            {"box": (5, 3, 20, 25), "area": 120},
            {"box": (30, 3, 45, 25), "area": 120},
        ]
        quality = savor.assess_ocr_quality(
            np.full((30, 80, 3), 200, dtype=np.uint8), inst,
            self._plate(components, [item["box"] for item in components]), engine=object(),
        )
        assert quality["state"] == "reliable"
        assert quality["source_dimensions"] == {"width": 80, "height": 30}
        assert quality["estimated_glyph_height"] == 22.0

    def test_component_surplus_without_reread_is_review_only(self):
        inst = make_inst("J", BBox(0, 0, 80, 30), detected_language="fr")
        components = [
            {"box": (5, 3, 18, 25), "area": 100},
            {"box": (25, 3, 38, 25), "area": 100},
        ]
        quality = savor.assess_ocr_quality(
            np.full((30, 80, 3), 200, dtype=np.uint8), inst,
            self._plate(components, [item["box"] for item in components]), engine=None,
        )
        assert quality["state"] == "review_required"
        assert {"component_count_mismatch", "ambiguous_segmentation", "engine_unavailable"} <= set(quality["reasons"])

    def test_high_confidence_unlocalizable_component_anomaly_stays_reliable(self):
        inst = make_inst("QUAI", BBox(0, 0, 100, 30), confidence=.999, detected_language="fr")
        # A detached Q-tail-like fragment makes two geometric groups, which
        # cannot align with one recognized word.  It is telemetry, not a
        # basis to send a certain correct word into review.
        components = [
            {"box": (0, 3, 10, 25), "area": 100},
            {"box": (24, 3, 39, 25), "area": 120},
            {"box": (43, 3, 58, 25), "area": 120},
            {"box": (62, 3, 77, 25), "area": 120},
            {"box": (81, 3, 96, 25), "area": 120},
        ]
        quality = savor.assess_ocr_quality(
            np.full((30, 100, 3), 200, dtype=np.uint8), inst,
            self._plate(components, [item["box"] for item in components]), engine=object(),
        )
        assert quality["state"] == "reliable"
        assert quality["reasons"] == ["component_count_mismatch"]
        assert quality["segmentation_evidence"]["state"] == "surplus_unlocalized"

    def test_already_all_caps_skips_case_review(self, monkeypatch):
        inst = make_inst("QUAI", BBox(0, 0, 80, 30), confidence=.999, detected_language="fr")
        components = [
            {"box": (5, 3, 20, 25), "area": 120},
            {"box": (25, 3, 40, 25), "area": 120},
            {"box": (45, 3, 60, 25), "area": 120},
            {"box": (65, 3, 79, 25), "area": 120},
        ]
        plate = self._plate(components, [item["box"] for item in components])
        monkeypatch.setattr(savor, "_plate_clusters", lambda *_args: plate)
        monkeypatch.setattr(savor, "chew_case", lambda *_args, **_kwargs: pytest.fail("case course must be skipped"))
        savor.taste(np.full((30, 80, 3), 200, dtype=np.uint8), [inst], engine=object())
        assert inst.ocr_correction is None
        assert inst.ocr_quality["state"] == "reliable"

    def test_sub_mark_resolution_is_never_reported_reliable(self):
        inst = make_inst("A", BBox(0, 0, 20, 8), detected_language="fr")
        components = [{"box": (2, 2, 8, 4), "area": 12}]
        quality = savor.assess_ocr_quality(
            np.full((8, 20, 3), 200, dtype=np.uint8), inst,
            self._plate(components, [item["box"] for item in components]), engine=object(),
        )
        assert quality["state"] == "unresolvable"
        assert "below_mark_resolution" in quality["reasons"]


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


class TestCaseSignature:
    def test_uniform_cap_clusters_repair_mixed_case_claim(self):
        # R e p u b l i q u e claimed, but every observed glyph spans the
        # same robust cap band.  The decision uses the full signature, not
        # merely one tall character.
        clusters = [(i * 10, 0, i * 10 + 7, 20) for i in range(10)]
        candidate, evidence = savor.case_signature_verdict("Republique", clusters)
        assert candidate == "REPUBLIQUE"
        assert evidence["state"] == "upper_confirmed"

    def test_mixed_height_title_case_is_review_only(self):
        clusters = [
            (0, 0, 7, 20),   # R cap
            (10, 6, 17, 20), # u x-height
            (20, 6, 27, 20), # e x-height
        ]
        candidate, evidence = savor.case_signature_verdict("Rue", clusters)
        assert candidate is None
        assert evidence["state"] == "review"

    def test_insufficient_case_evidence_never_repairs(self):
        candidate, evidence = savor.case_signature_verdict("Io", [(0, 0, 6, 16), (8, 0, 14, 16)])
        assert candidate is None
        assert evidence["state"] == "unresolvable"


def test_correction_ledger_preserves_earliest_ocr_text_for_cofiring_courses():
    inst = make_inst("Republique", BBox(0, 0, 100, 20))
    savor._record_correction(
        inst, applied=True, original_text="Republique", corrected_text="REPUBLIQUE",
        reason="case", course="case_signature",
    )
    savor._record_correction(
        inst, applied=True, original_text="REPUBLIQUE", corrected_text="RÉPUBLIQUE",
        reason="accent", course="latin_diacritic",
    )
    assert inst.ocr_correction["original_text"] == "Republique"
    assert inst.ocr_correction["corrected_text"] == "RÉPUBLIQUE"
    assert [step["course"] for step in inst.ocr_correction["steps"]] == [
        "case_signature", "latin_diacritic",
    ]


class TestDetachedMarkScan:
    def test_excess_mark_reconciles_to_one_claimed_glyph_and_is_visible(self):
        raw = [
            {"box": (0, 6, 8, 20), "area": 60},  # e body
            {"box": (2, 0, 5, 4), "area": 12},   # acute
        ]
        scan = savor._detached_mark_scan(raw, "e")
        assert scan is not None
        assert savor._mark_zone_verdict(scan, 0) is True

    def test_tittle_is_expected_but_diaeresis_has_excess_mark(self):
        raw_i = [
            {"box": (0, 12, 8, 42), "area": 100}, {"box": (1, 4, 3, 9), "area": 12},
        ]
        assert savor._mark_zone_verdict(savor._detached_mark_scan(raw_i, "i"), 0) is False
        raw_iumlaut = raw_i + [{"box": (5, 4, 7, 9), "area": 12}]
        assert savor._mark_zone_verdict(savor._detached_mark_scan(raw_iumlaut, "i"), 0) is True

    def test_unreconciled_components_are_unresolvable(self):
        raw = [{"box": (0, 6, 8, 20), "area": 60}, {"box": (20, 6, 28, 20), "area": 60}]
        assert savor._detached_mark_scan(raw, "e") is None

    def test_short_accented_cap_word_uses_base_bands_not_mark_top(self):
        # ÉCOLE-like geometry: the acute sits above the first cap.  Removing
        # it from band statistics keeps all five base glyphs tall.
        raw = [
            {"box": (0, 4, 8, 24), "area": 110}, {"box": (2, 0, 5, 3), "area": 12},
            {"box": (10, 4, 18, 24), "area": 110}, {"box": (20, 4, 28, 24), "area": 110},
            {"box": (30, 4, 38, 24), "area": 110}, {"box": (40, 4, 48, 24), "area": 110},
        ]
        scan = savor._detached_mark_scan(raw, "École")
        assert scan is not None
        candidate, _evidence = savor.case_signature_verdict("École", scan["base_components"])
        assert candidate == "ÉCOLE"


class TestLatinDiacriticCourse:
    def test_stored_french_form_preserves_observed_uppercase_base(self):
        proposals = savor._latin_diacritic_proposals("REPUBLIQUE", "fr")
        assert len(proposals) == 1
        assert savor._compose_marks("REPUBLIQUE", proposals[0]["positions"]) == "RÉPUBLIQUE"

    def test_language_scope_never_consults_french_forms_for_english(self):
        assert savor._latin_diacritic_proposals("cafe", "en") == []


## 🍢 Savor: sniff_clumps / chew_clump (course 0, the amuse-bouche)

# The production plate this course exists for: the r2 region of
# images/avenue-de-la-république.jpeg, whose recognizer emitted a single "J"
# where the enamel carries "l" and "a".  Boxes are mask-local, measured from
# the real crop, and the acute over the E is the 15th raw component.
AVENUE_R2_BBOX = BBox(x=120, y=171, width=214, height=57)
AVENUE_R2_COMPONENTS = [
    {"box": (5, 11, 14, 46), "area": 189},    # d
    {"box": (16, 24, 25, 46), "area": 137},   # e
    {"box": (41, 11, 44, 46), "area": 104},   # l  ) the clump the recognizer
    {"box": (47, 24, 55, 46), "area": 142},   # a  ) collapsed into one "J"
    {"box": (71, 11, 83, 46), "area": 321},   # R
    {"box": (87, 11, 97, 46), "area": 224},   # E
    {"box": (91, 7, 95, 10), "area": 10},     # acute, detached above the E
    {"box": (100, 11, 113, 46), "area": 246}, # P
    {"box": (116, 11, 128, 46), "area": 302}, # U
    {"box": (131, 11, 143, 46), "area": 329}, # B
    {"box": (146, 11, 156, 46), "area": 162}, # L
    {"box": (158, 12, 163, 46), "area": 152}, # I
    {"box": (166, 12, 178, 50), "area": 309}, # Q
    {"box": (181, 12, 193, 47), "area": 287}, # U
    {"box": (196, 12, 206, 47), "area": 204}, # E
]

AVENUE_ASSET = Path("images/avenue-de-la-république.jpeg")


def _avenue_plate(mask=None):
    """A plate view over the real r2 geometry; only raw components matter to
    sniff_clumps, so the mask/cluster slots stay unset unless a test needs
    pixels."""
    return (mask, AVENUE_R2_COMPONENTS, None)


class StubEngine:
    """Minimal OCRBackend surface chew_clump touches: one re-read per span."""

    def __init__(self, text, confidence=0.5):
        self.text, self.confidence, self.calls = text, confidence, []

    def detect_in_regions(self, asset, regions, pad=4, polygons=None):
        from tofu.layers.cicerone import RawDetection
        self.calls.append((regions[0], pad))
        if self.text is None:
            return [[]]
        box = regions[0]
        polygon = [(box.x, box.y), (box.x + box.width, box.y),
                   (box.x + box.width, box.y + box.height), (box.x, box.y + box.height)]
        return [[RawDetection(polygon=polygon, text=self.text, confidence=self.confidence)]]


class TestWordGroups:
    def test_line_splits_at_gap_outliers_not_at_ordinary_tracking(self):
        bases, _detached = savor._separate_marks(AVENUE_R2_COMPONENTS)
        assert [len(group) for group in savor._word_groups(bases)] == [2, 2, 10]

    def test_a_single_glyph_line_is_one_word(self):
        assert len(savor._word_groups([{"box": (0, 0, 5, 20), "area": 50}])) == 1

    def test_tightly_tracked_line_never_splits_on_sub_pixel_gaps(self):
        # Glyphs 1px apart: the ratio alone would call a 2.5px gap a word
        # break, which is exactly what WORD_GAP_MIN_PX exists to stop.
        bases = [{"box": (index * 9, 0, index * 9 + 8, 20), "area": 80} for index in range(6)]
        assert len(savor._word_groups(bases)) == 1


class TestSniffClumps:
    def test_production_clump_is_localized_and_named_from_a_stored_phrase(self):
        morsels, evidence = savor.sniff_clumps("de J REPUBLIQUE", _avenue_plate(), "fr")
        assert evidence["surplus"] == 1
        assert evidence["observed_word_glyphs"] == [2, 2, 10]
        assert evidence["claimed_word_glyphs"] == [2, 1, 10]
        assert len(morsels) == 1
        morsel = morsels[0]
        assert (morsel.token_index, morsel.token_text, morsel.token_start) == (1, "J", 3)
        assert morsel.recovered == "la" and morsel.phrase == "de la"
        # The plate supplies the cardinality, and the boxes handed on are the
        # real separate components -- not a re-split of one merged blob.
        assert morsel.glyph_boxes == [(41, 11, 44, 46), (47, 24, 55, 46)]

    def test_reconciled_line_proposes_nothing(self):
        morsels, evidence = savor.sniff_clumps("de la REPUBLIQUE", _avenue_plate(), "fr")
        assert morsels == [] and evidence["state"] == "reconciled"

    def test_a_language_the_resource_does_not_cover_is_out_of_scope(self):
        # Not merely "no phrase matched": the one-glyph-one-component premise
        # is not asserted for scripts this course was never calibrated on, so
        # it must not raise a surplus alarm on them either.
        for language in ("en", "ja", None):
            morsels, evidence = savor.sniff_clumps("de J REPUBLIQUE", _avenue_plate(), language)
            assert morsels == []
            assert evidence["reason"] == "language_out_of_course_scope"
            assert "surplus" not in evidence

    def test_unanchored_slot_is_never_recovered(self):
        # Same geometry, but no stored phrase's anchor matches the neighbour,
        # so nothing may be proposed for the disputed slot.
        morsels, evidence = savor.sniff_clumps("xy J REPUBLIQUE", _avenue_plate(), "fr")
        assert morsels == [] and evidence["reason"] == "no_stored_phrase_supports_the_slot"

    def test_surplus_spread_across_two_tokens_is_not_localizable(self):
        morsels, evidence = savor.sniff_clumps("de J REPUBLIQU", _avenue_plate(), "fr")
        assert morsels == []
        assert evidence["reason"] == "surplus_not_isolated_to_one_token"

    def test_multi_character_clump_is_out_of_scope_for_v1(self):
        morsels, evidence = savor.sniff_clumps("de JK REPUBLIQU", _avenue_plate(), "fr")
        assert morsels == []
        assert evidence["reason"] == "clump_is_not_a_single_emitted_character"

    def test_vertical_regions_are_out_of_scope(self):
        morsels, evidence = savor.sniff_clumps(
            "de J REPUBLIQUE", _avenue_plate(), "fr", vertical=True
        )
        assert morsels == [] and evidence["reason"] == "no_horizontal_plate"

    def test_all_cap_anchors_case_the_recovered_token(self):
        morsels, _evidence = savor.sniff_clumps("DE J REPUBLIQUE", _avenue_plate(), "fr")
        assert [morsel.recovered for morsel in morsels] == ["LA"]


class TestChewClump:
    def _real_plate(self):
        if not AVENUE_ASSET.exists():
            pytest.skip("production asset not present on this host")
        from tofu.utils.imaging import load_rgb, text_mask
        mask = text_mask(load_rgb(str(AVENUE_ASSET)), AVENUE_R2_BBOX, refine=True)
        if mask is None:
            pytest.skip("text_mask unavailable on this host")
        return str(AVENUE_ASSET), _avenue_plate(mask)

    def _morsel(self, plate):
        return savor.sniff_clumps("de J REPUBLIQUE", plate, "fr")[0][0]

    def test_production_case_splits_when_shape_and_reread_agree(self):
        asset, plate = self._real_plate()
        inst = make_inst("de J REPUBLIQUE", AVENUE_R2_BBOX, id="r2", detected_language="fr")
        # "Io" is the measured real re-read of this crop: right count, and an
        # I/l homoglyph confusion on identity.
        engine = StubEngine("Io")
        verdict, evidence = savor.chew_clump(asset, inst, self._morsel(plate), plate, engine=engine)
        assert verdict is True and evidence["state"] == "split_confirmed"
        # The split beats the merged reading; identity did NOT come from the
        # re-read, which disagreed with it.
        assert min(evidence["split_scores"]) >= evidence["merged_score"] + savor.BITE_MARGIN
        assert evidence["reread_glyphs"] == 2
        assert evidence["reread_matches_identity"] is False
        # The re-read really did look at the clump alone, not the whole region.
        span, _pad = engine.calls[0]
        assert span.width < AVENUE_R2_BBOX.width // 4

    def test_no_engine_can_detect_but_never_swallow(self):
        asset, plate = self._real_plate()
        inst = make_inst("de J REPUBLIQUE", AVENUE_R2_BBOX, id="r2", detected_language="fr")
        verdict, evidence = savor.chew_clump(asset, inst, self._morsel(plate), plate, engine=None)
        assert verdict is None and evidence["state"] == "reread_unavailable"

    def test_reread_disagreeing_on_count_blocks_the_split(self):
        asset, plate = self._real_plate()
        inst = make_inst("de J REPUBLIQUE", AVENUE_R2_BBOX, id="r2", detected_language="fr")
        verdict, evidence = savor.chew_clump(
            asset, inst, self._morsel(plate), plate, engine=StubEngine("J")
        )
        assert verdict is None and evidence["state"] == "reread_disagrees_on_glyph_count"

    def test_an_illegible_reread_never_stands_in_for_one(self):
        asset, plate = self._real_plate()
        inst = make_inst("de J REPUBLIQUE", AVENUE_R2_BBOX, id="r2", detected_language="fr")
        verdict, _evidence = savor.chew_clump(
            asset, inst, self._morsel(plate), plate, engine=StubEngine(None)
        )
        assert verdict is None


class TestTasteClumpCourse:
    def test_production_region_recovers_the_article_then_the_accent(self):
        if not AVENUE_ASSET.exists():
            pytest.skip("production asset not present on this host")
        inst = make_inst("de J REPUBLIQUE", AVENUE_R2_BBOX, id="r2",
                         confidence=0.866, detected_language="fr")
        swallowed = savor.taste(str(AVENUE_ASSET), [inst], engine=StubEngine("Io"))
        # Segmentation first, then the mark course it had been blocking.
        assert inst.text == "de la RÉPUBLIQUE"
        assert swallowed == 2
        assert [step["course"] for step in inst.ocr_correction["steps"]] == [
            "phrase_segmentation", "latin_diacritic",
        ]
        assert inst.ocr_correction["original_text"] == "de J REPUBLIQUE"
        assert inst.ocr_correction["applied"] is True

    def test_unresolved_surplus_is_surfaced_for_review_not_silently_dropped(self):
        if not AVENUE_ASSET.exists():
            pytest.skip("production asset not present on this host")
        inst = make_inst("de J REPUBLIQUE", AVENUE_R2_BBOX, id="r2",
                         confidence=0.866, detected_language="fr")
        savor.taste(str(AVENUE_ASSET), [inst], engine=None)
        assert inst.text == "de J REPUBLIQUE"
        step = inst.ocr_correction["steps"][0]
        assert step["course"] == "phrase_segmentation" and step["applied"] is False
        assert step["clump_evidence"]["state"] == "reread_unavailable"


class TestMarkZoneAddressing:
    def test_mark_zone_is_addressed_by_glyph_not_by_text_offset(self):
        # In "de la REPUBLIQUE" the acute belongs to the E at GLYPH index 5,
        # while the proposal addresses TEXT offset 7.  Conflating the two
        # inspects the U's zone instead and reports a mark that is not there.
        assert savor._glyph_index("de la REPUBLIQUE", 7) == 5
        inst = make_inst("de la REPUBLIQUE", AVENUE_R2_BBOX, id="r2", detected_language="fr")
        proposal = savor._latin_diacritic_proposals(inst.text, "fr")[0]
        assert proposal["positions"] == [(7, "́")]
        assert savor.chew_accents(inst, _avenue_plate(), proposal["positions"]) == {7: True}
        assert savor._compose_marks(inst.text, proposal["positions"]) == "de la RÉPUBLIQUE"

    def test_a_glyph_without_its_mark_is_still_reported_absent(self):
        stripped = [item for item in AVENUE_R2_COMPONENTS if item["area"] != 10]
        inst = make_inst("de la REPUBLIQUE", AVENUE_R2_BBOX, id="r2", detected_language="fr")
        proposal = savor._latin_diacritic_proposals(inst.text, "fr")[0]
        assert savor.chew_accents(inst, (None, stripped, None), proposal["positions"]) == {7: False}
