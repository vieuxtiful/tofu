## 🍢 scribe: right-to-left joining (arabic-reshaper) and visual order (bidi)
import numpy as np
import pytest
from PIL import Image, ImageFont

from tofu.core.types import BBox, InstText, Mask, TextManifest
from tofu.layers import scribe

# "hello world" / "peace world" -- short, and every letter of the Arabic
# phrase is a joining letter, so reshaping visibly changes the glyph run.
ARABIC = "مرحبا بالعالم"
HEBREW = "שלום עולם"
PRESENTATION_FORMS = range(0xFE70, 0xFF00)


def arabic_font_available() -> bool:
    try:
        ImageFont.truetype("arial.ttf", 20)
    except OSError:
        return False
    return True


needs_font = pytest.mark.skipif(
    not arabic_font_available(), reason="no Arabic-capable system font"
)


def presentation_form_count(s: str) -> int:
    return sum(1 for ch in s if ord(ch) in PRESENTATION_FORMS)


def make_manifest(target: str, lang: str):
    inst = InstText(
        id="r1", bounding_box=BBox(x=10, y=10, width=280, height=60),
        segmentation_mask=Mask(
            polygon=[(10, 10), (290, 10), (290, 70), (10, 70)], confidence=0.9,
        ),
        text="source", target_text=target, target_language=lang,
    )
    return TextManifest(asset_id="a", total_regions=1, instances=[inst], src_lang="en")


def ink_components(img, bbox) -> int:
    """Connected ink blobs inside a region. Cursive joining physically
    connects letters, so correct shaping yields strictly fewer blobs."""
    import cv2
    arr = np.asarray(img.convert("L"))
    crop = arr[bbox[1]:bbox[3], bbox[0]:bbox[2]]
    n, _ = cv2.connectedComponents((crop < 128).astype(np.uint8))
    return n - 1


class TestScriptRouting:
    def test_rtl_languages_detected(self):
        assert scribe.is_rtl_lang("ar")
        assert scribe.is_rtl_lang("fa")
        assert scribe.is_rtl_lang("he")

    def test_ltr_languages_not_flagged(self):
        for lang in ("en", "de", "ru", "ja", "zh-cn", "ko", "hi", "th"):
            assert not scribe.is_rtl_lang(lang), lang

    def test_unknown_and_empty_language_are_not_rtl(self):
        assert not scribe.is_rtl_lang(None)
        assert not scribe.is_rtl_lang("")
        assert not scribe.is_rtl_lang("xx-unmapped")


class TestPressJoins:
    def test_arabic_becomes_joining_forms(self):
        assert presentation_form_count(ARABIC) == 0
        joined = scribe.press_joins(ARABIC, "ar")
        assert presentation_form_count(joined) > 0

    def test_persian_also_reshaped(self):
        assert presentation_form_count(scribe.press_joins(ARABIC, "fa")) > 0

    def test_hebrew_untouched_since_it_does_not_join(self):
        assert scribe.press_joins(HEBREW, "he") == HEBREW

    def test_latin_untouched(self):
        assert scribe.press_joins("HELLO WORLD", "en") == "HELLO WORLD"

    def test_empty_text_safe(self):
        assert scribe.press_joins("", "ar") == ""

    def test_missing_reshaper_passes_text_through(self, monkeypatch):
        """Dependency-soft: absent arabic-reshaper must degrade, not raise."""
        import builtins
        real_import = builtins.__import__

        def no_reshaper(name, *args, **kwargs):
            if name.startswith("arabic_reshaper"):
                raise ImportError("simulated: arabic-reshaper unavailable")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", no_reshaper)
        assert scribe.press_joins(ARABIC, "ar") == ARABIC


class TestServingOrder:
    def test_rtl_line_is_reordered(self):
        assert scribe.serving_order(HEBREW, "he") != HEBREW

    def test_reordering_is_a_permutation_not_a_rewrite(self):
        out = scribe.serving_order(HEBREW, "he")
        assert sorted(out) == sorted(HEBREW)

    def test_ltr_line_untouched(self):
        assert scribe.serving_order("HELLO WORLD", "en") == "HELLO WORLD"

    def test_base_direction_pinned_to_rtl(self):
        """A caption opening with a Latin brand must still lay out RTL-base.

        Auto-detected base direction keys off the first strong character,
        which would put the brand on the wrong edge.
        """
        mixed = "ACME " + HEBREW
        out = scribe.serving_order(mixed, "he")
        assert out.strip().startswith(HEBREW[-1])  # Hebrew run leads
        assert out.strip().endswith("ACME")

    def test_missing_bidi_passes_line_through(self, monkeypatch):
        import builtins
        real_import = builtins.__import__

        def no_bidi(name, *args, **kwargs):
            if name.startswith("bidi"):
                raise ImportError("simulated: python-bidi unavailable")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", no_bidi)
        assert scribe.serving_order(HEBREW, "he") == HEBREW


@needs_font
class TestRenderEndToEnd:
    REGION = (10, 10, 290, 70)

    def render(self, target, lang):
        asset = Image.new("RGB", (300, 80), (255, 255, 255))
        return scribe.render(asset, make_manifest(target, lang), lang)

    def test_arabic_letters_are_joined_in_the_rendered_output(self, monkeypatch):
        """The payoff: joined cursive ink forms fewer blobs than isolated
        letters. Compares a real render against the same render with both
        RTL passes disabled."""
        shaped = self.render(ARABIC, "ar")
        monkeypatch.setattr(scribe, "press_joins", lambda t, l: t)
        monkeypatch.setattr(scribe, "serving_order", lambda t, l: t)
        unshaped = self.render(ARABIC, "ar")

        assert ink_components(shaped, self.REGION) < ink_components(unshaped, self.REGION)

    def test_rendered_arabic_differs_from_unshaped(self, monkeypatch):
        shaped = np.asarray(self.render(ARABIC, "ar").convert("L"))
        monkeypatch.setattr(scribe, "press_joins", lambda t, l: t)
        monkeypatch.setattr(scribe, "serving_order", lambda t, l: t)
        unshaped = np.asarray(self.render(ARABIC, "ar").convert("L"))
        assert not np.array_equal(shaped, unshaped)

    def test_hebrew_renders_ink(self):
        out = np.asarray(self.render(HEBREW, "he").convert("L"))
        x0, y0, x1, y1 = self.REGION
        assert (out[y0:y1, x0:x1] < 200).any()

    def test_latin_render_is_unaffected_by_the_rtl_path(self, monkeypatch):
        """LTR output must be byte-identical with the RTL passes stubbed
        out -- the feature has to be inert for every other language."""
        baseline = np.asarray(self.render("HELLO WORLD", "en").convert("L"))
        monkeypatch.setattr(scribe, "press_joins", lambda t, l: t)
        monkeypatch.setattr(scribe, "serving_order", lambda t, l: t)
        stubbed = np.asarray(self.render("HELLO WORLD", "en").convert("L"))
        assert np.array_equal(baseline, stubbed)
