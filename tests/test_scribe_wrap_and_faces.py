## 🍢 scribe: line wrapping, italic shear-pivot fix, font-face resolution
import numpy as np
from PIL import Image, ImageDraw

from tofu.core.types import BBox, InstText, Mask, StyleProfil, TextManifest
from tofu.layers import scribe
from tofu.layers.fonts import FontCoverage, FontRegistry


def make_registry(entries):
    """FontRegistry populated directly (bypassing file discovery) with
    (path, family, subfamily, weight_class) tuples."""
    reg = FontRegistry()
    for path, family, subfamily, weight_class in entries:
        reg._fonts[path] = FontCoverage(
            font_path=path, family=family, subfamily=subfamily,
            weight_class=weight_class,
        )
    return reg


ARIAL_FAMILY = [
    ("arial.ttf", "Arial", "Regular", 400),
    ("arialbd.ttf", "Arial", "Bold", 700),
    ("ariali.ttf", "Arial", "Italic", 400),
    ("arialbi.ttf", "Arial", "Bold Italic", 700),
]


class TestResolveFace:
    def test_no_registry_passes_through_unchanged(self):
        path, synth_italic = scribe.resolve_face(None, "arial.ttf", "bold", True)
        assert path == "arial.ttf" and synth_italic is True

    def test_unknown_font_passes_through_unchanged(self):
        reg = make_registry(ARIAL_FAMILY)
        path, synth_italic = scribe.resolve_face(reg, "unknown/font.ttf", "bold", False)
        assert path == "unknown/font.ttf" and synth_italic is False

    def test_finds_real_bold_italic_sibling(self):
        reg = make_registry(ARIAL_FAMILY)
        path, synth_italic = scribe.resolve_face(reg, "arial.ttf", "bold", True)
        assert path == "arialbi.ttf"
        assert synth_italic is False  # real face found; no shear needed

    def test_finds_real_italic_sibling_regular_weight(self):
        reg = make_registry(ARIAL_FAMILY)
        path, synth_italic = scribe.resolve_face(reg, "arial.ttf", None, True)
        assert path == "ariali.ttf"
        assert synth_italic is False

    def test_already_correct_face_returned_as_is(self):
        reg = make_registry(ARIAL_FAMILY)
        path, synth_italic = scribe.resolve_face(reg, "arialbd.ttf", "bold", False)
        assert path == "arialbd.ttf" and synth_italic is False

    def test_missing_sibling_falls_back_to_synthetic(self):
        # a family with no italic face at all: light-only
        reg = make_registry([("thin.ttf", "Thin Family", "Thin", 200)])
        path, synth_italic = scribe.resolve_face(reg, "thin.ttf", None, True)
        assert path == "thin.ttf"
        assert synth_italic is True  # no real italic sibling; must synthesize

    def test_no_font_family_no_weight_or_italic_stays_true_auto(self):
        # nothing requested: font_family=None passes straight through --
        # there's nothing to resolve towards, so "auto" stays "auto"
        reg = make_registry(ARIAL_FAMILY)
        path, synth_italic = scribe.resolve_face(reg, None, None, False)
        assert path is None and synth_italic is False

    def test_no_font_family_but_bold_italic_requested_anchors_on_fallback(self):
        # "auto" family left as None, but weight/italic IS requested --
        # e.g. typography detected bold source text and the region was
        # never given an explicit font pick. must anchor the sibling
        # search on the FALLBACK_FONTS default (arial.ttf) and find a
        # real bold-italic sibling, not silently drop the request the
        # way font_family=None used to be treated unconditionally.
        reg = make_registry(ARIAL_FAMILY)
        path, synth_italic = scribe.resolve_face(reg, None, "bold", True)
        assert path == "arialbi.ttf"
        assert synth_italic is False

    def test_no_font_family_weight_requested_no_fallback_installed(self):
        # weight/italic requested but the registry has none of the
        # FALLBACK_FONTS names at all -- nothing to anchor on, passes
        # through unchanged rather than erroring
        reg = make_registry([("thin.ttf", "Thin Family", "Thin", 200)])
        path, synth_italic = scribe.resolve_face(reg, None, "bold", False)
        assert path is None and synth_italic is False


class TestLineWrapping:
    def _bbox_asset(self, w=300, h=200):
        return Image.new("RGB", (w, h), (255, 255, 255))

    def test_long_text_wraps_to_multiple_lines(self):
        asset = self._bbox_asset(200, 200)
        bbox = BBox(x=20, y=20, width=160, height=160)
        inst = InstText(
            id="r1", bounding_box=bbox,
            text="x", target_text="a somewhat long sentence that must wrap",
        )
        manifest = TextManifest(asset_id="a", total_regions=1, instances=[inst])
        out = scribe.render(asset, manifest, "en")
        out_np = np.asarray(out)
        # ink should appear at multiple distinct row bands, not one dense line
        region = out_np[bbox.y:bbox.y + bbox.height, bbox.x:bbox.x + bbox.width]
        dark_rows = np.where((region < 200).any(axis=(1, 2)))[0]
        assert len(dark_rows) > 0
        # at least two separated bands of ink rows (multi-line)
        gaps = np.diff(dark_rows)
        assert (gaps > 3).sum() >= 1

    def test_cjk_text_wraps_without_spaces(self):
        from tofu.layers.cleanse import _region_mask  # noqa: F401 (import smoke)
        asset = self._bbox_asset(150, 150)
        bbox = BBox(x=10, y=10, width=120, height=120)
        cjk_text = "" .join(["これはテスト用の日本語の文章です"])
        inst = InstText(id="r1", bounding_box=bbox, text="x", target_text=cjk_text)
        manifest = TextManifest(asset_id="a", total_regions=1, instances=[inst])
        # must not raise, and must produce visible ink
        out = scribe.render(asset, manifest, "ja")
        out_np = np.asarray(out)
        region = out_np[bbox.y:bbox.y + bbox.height, bbox.x:bbox.x + bbox.width]
        assert (region < 200).any()

    def test_wrap_lines_word_boundary(self):
        img = Image.new("RGBA", (10, 10))
        draw = ImageDraw.Draw(img)
        font = scribe._get_font(None, 16)
        lines = scribe._wrap_lines(draw, "one two three four", font, max_width=60)
        assert len(lines) > 1
        assert "".join(lines).replace(" ", "") == "onetwothreefour"

    def test_wrap_lines_char_wrap_no_spaces(self):
        img = Image.new("RGBA", (10, 10))
        draw = ImageDraw.Draw(img)
        font = scribe._get_font(None, 16)
        lines = scribe._wrap_lines(draw, "abcdefghij", font, max_width=30)
        assert len(lines) > 1
        assert "".join(lines) == "abcdefghij"

    def test_no_descender_text_fits_as_tightly_as_single_line_fit(self):
        """regression: the fit check must use the ACTUAL rendered ink
        extent, not font.getmetrics()'s nominal ascent+descent — that
        nominal box accounts for glyphs the string may not even contain
        (e.g. descenders), which over-estimated height for all-caps/no-
        descender text and picked an unnecessarily smaller font (measured
        on a real cicerone-detected region: "SALE" in a 176x60 box fit at
        size 53 instead of the correct 69, visibly shrinking a region
        that fit fine as a single line and needed no wrapping at all)."""
        draw = ImageDraw.Draw(Image.new("RGBA", (1, 1)))
        bbox = BBox(x=0, y=0, width=176, height=60)
        font, lines, _ = scribe._fit_wrapped(draw, "SALE", bbox, None, None, None)
        assert lines == ["SALE"]  # must not wrap short text that fits

        # reconstruct the pre-regression single-line-only fit (checks the
        # ACTUAL textbbox extent directly, with no line-wrap machinery at
        # all) as the ground truth for "how large should this fit":
        lo, hi, best_size = 6, bbox.height * 2, 6
        while lo <= hi:
            mid = (lo + hi) // 2
            f = scribe._get_font(None, mid)
            l, t, r, b = draw.textbbox((0, 0), "SALE", font=f)
            if (r - l) <= bbox.width and (b - t) <= bbox.height:
                best_size, lo = mid, mid + 1
            else:
                hi = mid - 1
        # the wrapped fit must match the reference fit exactly -- any gap
        # here means the block-height estimate is over-conservative again
        assert font.size == best_size


class TestItalicShearPivot:
    """regression: the shear must be bounded to the line's own height,
    not the text's absolute position in the image (the actual Phase 3
    finding — a region at y~300 shifted ~60px sideways under the old
    global-layer shear)."""

    def test_italic_shift_bounded_regardless_of_vertical_position(self):
        asset = Image.new("RGB", (500, 500), (255, 255, 255))
        results = {}
        for y in (20, 400):  # near top vs far down the image
            bbox = BBox(x=50, y=y, width=200, height=50)
            inst = InstText(
                id="r1", bounding_box=bbox, text="x", target_text="Slant",
                style_profile=StyleProfil(italic=True, color="#000000"),
            )
            manifest = TextManifest(asset_id="a", total_regions=1, instances=[inst])
            out = np.asarray(scribe.render(asset, manifest, "en"))
            # leftmost dark column within a generous margin around the bbox
            region = out[max(0, y - 10):y + 70, 0:500]
            dark_cols = np.where((region < 200).any(axis=(0, 2)))[0]
            assert len(dark_cols) > 0
            results[y] = dark_cols.min()
        # the leftmost ink column must land in roughly the same place
        # regardless of the region's vertical position — under the old
        # bug it would differ by tens of pixels (proportional to y)
        assert abs(results[20] - results[400]) < 15
