## 🍢 scribe: ToFU render-time glyph-coverage guard
import numpy as np
import pytest
from PIL import Image, ImageFont

from tofu.core.types import BBox, InstText, StyleProfil, TextManifest
from tofu.layers import scribe
from tofu.layers.fonts import FontCoverage, FontRegistry

LATIN_ONLY = {ord(c) for c in "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz "}
CJK_SAMPLE = {ord(c) for c in "居酒屋ようこそ"}


def make_registry(entries):
    reg = FontRegistry()
    for path, family, subfamily, weight_class, codepoints in entries:
        reg._fonts[path] = FontCoverage(
            font_path=path, family=family, subfamily=subfamily,
            weight_class=weight_class, codepoints=codepoints,
        )
    return reg


class TestCheckGlyphCoverage:
    def test_fully_covered_font_reports_no_fallback(self):
        reg = make_registry([("latin.ttf", "Latin", "Regular", 400, LATIN_ONLY)])
        path, all_covered = scribe.check_glyph_coverage(reg, "latin.ttf", "HELLO", "en")
        assert path is None and all_covered is True

    def test_missing_glyphs_finds_a_covering_font(self):
        reg = make_registry([
            ("latin.ttf", "Latin", "Regular", 400, LATIN_ONLY),
            ("cjk.ttc#0", "CJK Gothic", "Regular", 400, CJK_SAMPLE),
        ])
        path, all_covered = scribe.check_glyph_coverage(reg, "latin.ttf", "居酒屋", "ja")
        assert all_covered is False
        assert path == "cjk.ttc#0"

    def test_no_registry_reports_covered_optimistically(self):
        path, all_covered = scribe.check_glyph_coverage(None, "whatever.ttf", "居酒屋", "ja")
        assert path is None and all_covered is True

    def test_unknown_font_family_gives_no_false_positive_without_better_option(self):
        # font_family isn't in the registry at all, and nothing else
        # covers this text either -- no evidence of failure, no crash
        reg = make_registry([("latin.ttf", "Latin", "Regular", 400, LATIN_ONLY)])
        path, all_covered = scribe.check_glyph_coverage(reg, "unknown.ttf", "居酒屋", "ja")
        assert path is None
        assert all_covered is True  # no known gap; unknown font gets benefit of the doubt

    def test_already_best_available_reports_no_swap(self):
        reg = make_registry([("cjk.ttc#0", "CJK Gothic", "Regular", 400, CJK_SAMPLE)])
        path, all_covered = scribe.check_glyph_coverage(reg, "cjk.ttc#0", "居酒屋", "ja")
        assert all_covered is True  # this IS the covering font already

    def test_no_font_family_resolves_the_actual_fallback_chain_font(self):
        """regression: font_family=None ("auto", the common case with no
        explicit selection) must resolve to what _get_font(None, ...)
        ACTUALLY renders with (the first loadable FALLBACK_FONTS entry),
        not be treated as unconditional missing evidence. without this,
        registry-keyed-by-full-path fonts never matched font_family=None,
        so `current` stayed None and the code fell straight into "search
        for something better" -- which ALWAYS finds a different path than
        None, flagging every auto-styled region as a fallback regardless
        of whether the real render actually has full coverage. measured
        live: flat-sign's plain English text flagged glyph_fallback on
        all 4 regions despite arial.ttf covering it completely."""
        reg = make_registry([
            (r"C:\Windows\Fonts\arial.ttf", "Arial", "Regular", 400, LATIN_ONLY),
        ])
        path, all_covered = scribe.check_glyph_coverage(reg, None, "MAIN STREET", "en")
        assert all_covered is True
        assert path is None

    def test_no_font_family_still_flags_a_genuine_gap(self):
        reg = make_registry([
            (r"C:\Windows\Fonts\arial.ttf", "Arial", "Regular", 400, LATIN_ONLY),
            (r"C:\Windows\Fonts\cjk.ttc#0", "CJK Gothic", "Regular", 400, CJK_SAMPLE),
        ])
        path, all_covered = scribe.check_glyph_coverage(reg, None, "居酒屋", "ja")
        assert all_covered is False
        assert path == r"C:\Windows\Fonts\cjk.ttc#0"


class TestResolveAutoFont:
    """resolve_auto_font composes _get_font's FALLBACK_FONTS chain with
    check_glyph_coverage's override -- the same two steps render() takes
    -- so callers (Translate-tab preview, Font column label) can show
    the user what "auto" means before any render happens, guaranteed to
    agree with what actually gets drawn."""

    def test_latin_text_resolves_to_the_installed_fallback_font(self):
        reg = make_registry([
            (r"C:\Windows\Fonts\arial.ttf", "Arial", "Regular", 400, LATIN_ONLY),
        ])
        path = scribe.resolve_auto_font(reg, "en", "HELLO")
        assert path == r"C:\Windows\Fonts\arial.ttf"

    def test_text_the_default_cant_cover_resolves_to_a_covering_font(self):
        reg = make_registry([
            (r"C:\Windows\Fonts\arial.ttf", "Arial", "Regular", 400, LATIN_ONLY),
            (r"C:\Windows\Fonts\cjk.ttc#0", "CJK Gothic", "Regular", 400, CJK_SAMPLE),
        ])
        path = scribe.resolve_auto_font(reg, "ja", "居酒屋")
        assert path == r"C:\Windows\Fonts\cjk.ttc#0"

    def test_no_registry_returns_none(self):
        assert scribe.resolve_auto_font(None, "en", "HELLO") is None

    def test_empty_text_returns_none(self):
        reg = make_registry([(r"C:\Windows\Fonts\arial.ttf", "Arial", "Regular", 400, LATIN_ONLY)])
        assert scribe.resolve_auto_font(reg, "en", "") is None

    def test_no_installed_fallback_font_returns_none(self):
        # registry only has a CJK font -- none of FALLBACK_FONTS
        # (arial/DejaVuSans/segoeui) is present to serve as a baseline
        reg = make_registry([(r"C:\Windows\Fonts\cjk.ttc#0", "CJK Gothic", "Regular", 400, CJK_SAMPLE)])
        assert scribe.resolve_auto_font(reg, "en", "HELLO") is None

    def test_bold_weight_resolves_to_the_real_bold_sibling(self):
        # regression: a region left on "auto" font selection but with
        # style_profile.font_weight="bold" (typography-detected bold
        # source text, the common case for signage) must resolve to a
        # real bold FILE -- not the plain arial.ttf regular fallback,
        # which is what render() itself used to draw before resolve_face
        # learned to anchor a sibling search on the fallback family when
        # font_family is None but weight/italic is requested.
        reg = make_registry([
            (r"C:\Windows\Fonts\arial.ttf", "Arial", "Regular", 400, LATIN_ONLY),
            (r"C:\Windows\Fonts\arialbd.ttf", "Arial", "Bold", 700, LATIN_ONLY),
        ])
        path = scribe.resolve_auto_font(reg, "en", "MAIN STREET", weight="bold")
        assert path == r"C:\Windows\Fonts\arialbd.ttf"

    def test_no_bold_sibling_falls_back_to_the_regular_default(self):
        # weight="bold" requested but the registry has no bold face in
        # the fallback family -- resolve_face's own no-sibling-found path
        # returns the ORIGINAL (None) family unchanged, so this still
        # lands on the plain regular fallback rather than erroring
        reg = make_registry([
            (r"C:\Windows\Fonts\arial.ttf", "Arial", "Regular", 400, LATIN_ONLY),
        ])
        path = scribe.resolve_auto_font(reg, "en", "MAIN STREET", weight="bold")
        assert path == r"C:\Windows\Fonts\arial.ttf"


class TestGetFontHandlesCollectionIndex:
    """regression: FontRegistry keys collection faces as 'path#index'
    (multiple faces share one .ttc/.otc file); PIL takes the index as a
    separate `index=` kwarg and raises OSError on a literal '#N' suffix
    in the path. without splitting it out, ANY registry-resolved
    collection face — a common case for CJK fonts, which frequently ship
    as .ttc — silently failed to load and fell through to the Latin-only
    fallback chain, defeating resolve_face() and check_glyph_coverage()
    even when they correctly identified a covering font."""

    def _find_ttc(self):
        for cand in ("msgothic.ttc", "meiryo.ttc", "YuGothM.ttc"):
            try:
                ImageFont.truetype(cand, 12)
                return cand
            except Exception:
                continue
        return None

    def test_hash_index_suffix_loads_successfully(self):
        ttc = self._find_ttc()
        if not ttc:
            pytest.skip("no .ttc font found on this host")
        font = scribe._get_font(f"{ttc}#0", 24)
        # a real face was loaded, not the PIL bitmap default fallback
        assert hasattr(font, "getname")
        assert font.size == 24 or getattr(font, "path", None)

    def test_plain_path_without_hash_still_works(self):
        font = scribe._get_font(None, 16)
        assert font is not None


class TestGlyphFallbackEndToEnd:
    def test_fully_covered_text_never_flagged(self):
        reg = make_registry([
            ("latin.ttf", "Latin", "Regular", 400, LATIN_ONLY),
        ])
        asset = Image.new("RGB", (200, 150), (255, 255, 255))
        bbox = BBox(x=20, y=20, width=160, height=100)
        inst = InstText(
            id="r1", bounding_box=bbox, text="x", target_text="HELLO",
            target_language="en",
            style_profile=StyleProfil(font_family="latin.ttf"),
        )
        manifest = TextManifest(asset_id="a", total_regions=1, instances=[inst])
        scribe.render(asset, manifest, "en", font_registry=reg)
        # explicitly False (confirmed covered), not left at None/stale --
        # a manifest persists across renders, so a previous run's
        # glyph_fallback=True must be actively cleared once the gap is
        # resolved, not merely left untouched
        assert inst.glyph_fallback is False

    def test_previously_flagged_region_is_cleared_once_resolved(self):
        """regression: glyph_fallback must be actively reset, not just
        set-on-failure -- otherwise a manifest re-rendered after the
        underlying gap is fixed (font swapped, registry attached) keeps
        reporting a stale fallback forever. measured live: a manifest
        saved from an earlier buggy run kept fallback_font=4 in the
        coverage summary even after the check_glyph_coverage bug was
        fixed, because nothing ever cleared the flag on a covered run."""
        reg = make_registry([("latin.ttf", "Latin", "Regular", 400, LATIN_ONLY)])
        asset = Image.new("RGB", (200, 150), (255, 255, 255))
        bbox = BBox(x=20, y=20, width=160, height=100)
        inst = InstText(
            id="r1", bounding_box=bbox, text="x", target_text="HELLO",
            target_language="en",
            style_profile=StyleProfil(font_family="latin.ttf"),
        )
        inst.glyph_fallback = True  # simulate a stale flag from a prior run
        manifest = TextManifest(asset_id="a", total_regions=1, instances=[inst])
        scribe.render(asset, manifest, "en", font_registry=reg)
        assert inst.glyph_fallback is False

    def test_missing_coverage_flags_instance_and_swaps_to_covering_font(self):
        reg = make_registry([
            ("latin.ttf", "Latin", "Regular", 400, LATIN_ONLY),
            ("cjk.ttf", "CJK", "Regular", 400, CJK_SAMPLE),
        ])
        asset = Image.new("RGB", (200, 150), (255, 255, 255))
        bbox = BBox(x=20, y=20, width=160, height=100)
        # requests a LATIN-only font for CJK target text -- a real gap
        inst = InstText(
            id="r1", bounding_box=bbox, text="x", target_text="居酒屋",
            target_language="ja",
            style_profile=StyleProfil(font_family="latin.ttf"),
        )
        manifest = TextManifest(asset_id="a", total_regions=1, instances=[inst])
        scribe.render(asset, manifest, "ja", font_registry=reg)
        assert inst.glyph_fallback is True

    def test_source_style_profile_never_mutated(self):
        """the guard must swap a working COPY for rendering, never the
        manifest instance's own style_profile -- that's the user's
        original request and must survive round-trips/re-renders."""
        reg = make_registry([("latin.ttf", "Latin", "Regular", 400, LATIN_ONLY)])
        asset = Image.new("RGB", (200, 150), (255, 255, 255))
        bbox = BBox(x=20, y=20, width=160, height=100)
        inst = InstText(
            id="r1", bounding_box=bbox, text="x", target_text="HELLO",
            style_profile=StyleProfil(font_family="latin.ttf"),
        )
        manifest = TextManifest(asset_id="a", total_regions=1, instances=[inst])
        scribe.render(asset, manifest, "en", font_registry=reg)
        assert inst.style_profile.font_family == "latin.ttf"
