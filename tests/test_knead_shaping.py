## 🍢 knead: HarfBuzz shaping + FreeType rasterization
from pathlib import Path

import numpy as np
import pytest
from PIL import ImageFont

from conftest import DEVANAGARI_FONT, LATIN_FONT
from tofu.layers import knead

ARIAL = LATIN_FONT      # arial.ttf on Windows, DejaVuSans/Liberation on CI
NIRMALA = DEVANAGARI_FONT

HINDI = "हिन्दी"
KSHA = "क्ष"
ARABIC = "مرحبا"

needs_shaping = pytest.mark.skipif(not knead.available(), reason="uharfbuzz/freetype-py absent")
needs_arial = pytest.mark.skipif(ARIAL is None, reason="no Latin font on this system")
needs_deva = pytest.mark.skipif(NIRMALA is None, reason="no Devanagari font on this system")


def arial(size=48):
    return ImageFont.truetype(str(ARIAL), size)


def nirmala(size=48):
    return ImageFont.truetype(str(NIRMALA), size, index=0)


class TestAvailability:
    def test_available_is_a_bool(self):
        assert isinstance(knead.available(), bool)


@needs_arial
class TestFaceIdentity:
    def test_resolves_path_index_and_size(self):
        path, index, size = knead.face_identity(arial(37))
        assert Path(path).is_file()
        assert index == 0 and size == 37

    def test_bare_name_resolves_through_pillow(self):
        """scribe's fallback chain is bare filenames FreeType cannot find;
        piggybacking on Pillow's lookup is the whole point of this helper."""
        ident = knead.face_identity(ImageFont.truetype(ARIAL.name, 20))
        assert ident is not None and Path(ident[0]).is_file()

    def test_default_bitmap_font_rejected(self):
        """load_default() carries a BytesIO, not a real file -- the caller
        must stay on Pillow rather than crash inside FreeType."""
        assert knead.face_identity(ImageFont.load_default()) is None

    def test_non_font_object_rejected(self):
        assert knead.face_identity(object()) is None
        assert knead.face_identity(None) is None


@needs_shaping
@needs_arial
class TestLatinKerning:
    def test_shaping_applies_kerning_pillow_misses(self):
        """The Latin payoff. Pillow's BASIC engine applies no GPOS at all,
        so its advance matches shaping with `kern` explicitly disabled."""
        font = arial()
        shaped = knead.run_width("AVATAR Wave To", font)
        unkerned = knead.run_width("AVATAR Wave To", font, features={"kern": False})
        pillow = font.getlength("AVATAR Wave To")

        assert shaped < unkerned                      # kerning tightened it
        assert unkerned == pytest.approx(pillow, abs=1.0)  # Pillow == no kerning
        assert shaped < pillow - 5                    # and the gap is material

    def test_kerned_pair_is_tighter_than_sum_of_parts(self):
        font = arial()
        av = knead.run_width("AV", font)
        a = knead.run_width("A", font)
        v = knead.run_width("V", font)
        assert av < a + v

    def test_run_width_matches_run_advance(self):
        font = arial()
        run = knead.knead_run("Hamburgefonstiv", font)
        assert knead.run_width("Hamburgefonstiv", font) == pytest.approx(run.advance)

    def test_empty_text_returns_none(self):
        assert knead.knead_run("", arial()) is None


@needs_shaping
@needs_deva
class TestComplexScript:
    def test_devanagari_forms_conjuncts(self):
        """6 codepoints collapse to fewer glyphs -- GSUB doing work Pillow
        cannot do at all."""
        run = knead.knead_run(HINDI, nirmala())
        assert len(run) < len(HINDI)

    def test_ksha_is_a_single_conjunct_glyph(self):
        run = knead.knead_run(KSHA, nirmala())
        assert len(run) == 1

    def test_no_notdef_glyphs_for_a_covering_face(self):
        run = knead.knead_run(HINDI, nirmala())
        assert all(g.gid != 0 for g in run.glyphs)

    def test_uncovered_face_yields_notdef(self):
        """Arial has no Devanagari; shaping must still succeed structurally
        so the caller's glyph-coverage guard is what catches it."""
        run = knead.knead_run(HINDI, arial())
        assert run is not None and any(g.gid == 0 for g in run.glyphs)


@needs_shaping
@needs_arial
class TestDirection:
    def test_arabic_detected_as_rtl(self):
        run = knead.knead_run(ARABIC, arial())
        assert run.direction == "rtl"

    def test_latin_detected_as_ltr(self):
        assert knead.knead_run("HELLO", arial()).direction == "ltr"

    def test_script_hint_forces_rtl_direction(self):
        run = knead.knead_run("123", arial(), script="Arab")
        assert run.direction == "rtl"

    def test_explicit_direction_overrides_hint(self):
        run = knead.knead_run(ARABIC, arial(), direction="ltr")
        assert run.direction == "ltr"


@needs_shaping
@needs_arial
class TestBakeRun:
    def bake(self, text="Hamburg", stroke=0, stroke_fill=None, size=(300, 90)):
        font = arial()
        run = knead.knead_run(text, font)
        return knead.bake_run(
            size, run, (10, 60), font, (0, 0, 0, 255),
            stroke_width=stroke, stroke_fill=stroke_fill,
        )

    def test_produces_ink(self):
        img = self.bake()
        assert img is not None
        assert np.asarray(img)[..., 3].max() > 0

    def test_layer_matches_requested_canvas_size(self):
        assert self.bake(size=(240, 70)).size == (240, 70)

    def test_stroke_widens_the_ink(self):
        plain = np.asarray(self.bake())[..., 3] > 0
        stroked = np.asarray(self.bake(stroke=3, stroke_fill=(255, 0, 0, 255)))[..., 3] > 0
        assert stroked.sum() > plain.sum()

    def test_fill_colour_is_honoured(self):
        font = arial()
        run = knead.knead_run("A", font)
        img = knead.bake_run((120, 90), run, (10, 60), font, (12, 200, 90, 255))
        arr = np.asarray(img)
        opaque = arr[..., 3] > 200
        assert opaque.any()
        assert arr[opaque][:, :3].mean(axis=0) == pytest.approx([12, 200, 90], abs=2)

    def test_glyphs_stay_inside_a_tight_canvas(self):
        """Clipping must not raise when a run overruns its layer."""
        font = arial()
        run = knead.knead_run("WWWWWWWWWWWWWWWW", font)
        assert knead.bake_run((40, 30), run, (5, 20), font, (0, 0, 0, 255)) is not None

    def test_none_run_returns_none(self):
        assert knead.bake_run((50, 50), None, (0, 0), arial(), (0, 0, 0, 255)) is None


@needs_shaping
@needs_arial
class TestInkBox:
    def test_ink_box_is_narrower_than_advance_for_side_bearings(self):
        font = arial()
        run = knead.knead_run("III", font)
        left, top, right, bottom = knead.run_ink_box(run, font)
        assert right - left < run.advance
        assert bottom > top

    def test_whitespace_only_run_has_no_ink(self):
        font = arial()
        run = knead.knead_run("   ", font)
        assert knead.run_ink_box(run, font) is None


@needs_shaping
@needs_arial
class TestProofRun:
    def test_tracking_widens_the_advance(self):
        font = arial()
        run = knead.knead_run("HELLO", font)
        spaced = knead.proof_run(run, 4.0)
        # 5 clusters -> 4 boundaries
        assert spaced.advance == pytest.approx(run.advance + 4 * 4.0)

    def test_zero_tracking_is_a_noop(self):
        font = arial()
        run = knead.knead_run("HELLO", font)
        assert knead.proof_run(run, 0).advance == pytest.approx(run.advance)

    def test_none_run_passes_through(self):
        assert knead.proof_run(None, 5.0) is None

    def test_glyph_count_and_ids_unchanged(self):
        font = arial()
        run = knead.knead_run("HELLO", font)
        spaced = knead.proof_run(run, 3.0)
        assert [g.gid for g in spaced.glyphs] == [g.gid for g in run.glyphs]

    def test_negative_tracking_tightens(self):
        font = arial()
        run = knead.knead_run("HELLO", font)
        assert knead.proof_run(run, -2.0).advance < run.advance


@needs_shaping
@needs_deva
class TestProofRunClusters:
    def test_spacing_lands_between_clusters_not_inside_them(self):
        """The reason this is cluster-aware. क्ष is ONE cluster of glyphs;
        spacing inside it would prise the conjunct apart."""
        font = nirmala()
        run = knead.knead_run(KSHA, font)
        spaced = knead.proof_run(run, 10.0)
        # single cluster -> no interior boundary -> advance unchanged
        assert spaced.advance == pytest.approx(run.advance)

    def test_multi_cluster_devanagari_gains_only_boundary_spacing(self):
        font = nirmala()
        run = knead.knead_run(HINDI, font)
        clusters = len({g.cluster for g in run.glyphs})
        spaced = knead.proof_run(run, 5.0)
        assert spaced.advance == pytest.approx(run.advance + (clusters - 1) * 5.0)
        assert len(spaced) == len(run)  # no glyphs added or dropped


class TestDegradation:
    def test_missing_libraries_disable_shaping(self, monkeypatch):
        """Every entry point must return None, never raise, so scribe can
        keep its Pillow path."""
        import builtins
        real_import = builtins.__import__

        def no_shaping(name, *args, **kwargs):
            if name.startswith(("uharfbuzz", "freetype")):
                raise ImportError("simulated: shaping libraries unavailable")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", no_shaping)
        assert knead.available() is False
        if ARIAL is not None:
            assert knead.knead_run("AV", arial()) is None
            assert knead.run_width("AV", arial()) is None
