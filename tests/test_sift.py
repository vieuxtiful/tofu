## 🍢 sift — typographic categorisation for the Font Manager
"""
sift() answers a BROWSING question ("is this a serif?"), which is a
different question from the one font_matching answers ("which installed
face did this sign use?"). The second must be reached from pixels; the
first is allowed to read what the foundry declared. These tests cover the
classifier and, at the end, the boundary between the two.
"""

import dataclasses

import pytest

from conftest import LATIN_FONT, system_font
from tofu.layers import sift as sift_mod
from tofu.layers.fonts import FontCoverage, FontRegistry
from tofu.layers.sift import CATEGORIES, DISPLAY, MONO, SANS, SERIF, UNKNOWN, sift


@pytest.fixture(autouse=True)
def _clear_cache():
    """the module memoizes per face path; a stale verdict from one test
    would silently satisfy the next"""
    sift_mod.clear_cache()
    yield
    sift_mod.clear_cache()


# -- name fallback -------------------------------------------------------
# Reached only when the face cannot be opened, so a path that does not
# exist on disk is exactly the right way to exercise it.

@pytest.mark.parametrize(
    "family,expected",
    [
        ("Courier New", MONO),
        ("JetBrains Mono", MONO),
        ("Cascadia Code", MONO),
        ("Segoe UI Symbol Display", DISPLAY),
        ("Brush Script MT", DISPLAY),
        ("Noto Serif", SERIF),
        ("EB Garamond", SERIF),
        ("Noto Sans", SANS),
        ("Franklin Gothic", SANS),
    ],
)
def test_name_fallback_reads_declarative_tokens(family, expected):
    assert sift("C:/nowhere/does-not-exist.ttf", family) == expected


def test_name_fallback_prefers_mono_over_sans():
    # "IBM Plex Sans Mono" is a monospace face first; someone filtering for
    # mono wants it and someone filtering for sans does not
    assert sift("C:/nowhere/missing.ttf", "IBM Plex Sans Mono") == MONO


def test_unclassifiable_face_is_unknown_not_guessed():
    assert sift("C:/nowhere/missing.ttf", "Zapfino Whatever") == UNKNOWN


def test_falls_back_to_filename_when_no_family_given():
    assert sift("C:/nowhere/SomeMonoFace.ttf") == MONO


# -- real faces ----------------------------------------------------------

def test_real_latin_face_classifies():
    if LATIN_FONT is None:
        pytest.skip("no Latin face on this machine")
    assert sift(str(LATIN_FONT)) in CATEGORIES


def test_known_serif_and_sans_faces_disagree():
    serif = system_font("georgia.ttf", "Georgia.ttf", "DejaVuSerif.ttf",
                        "LiberationSerif-Regular.ttf")
    sans = system_font("arial.ttf", "Arial.ttf", "DejaVuSans.ttf",
                       "LiberationSans-Regular.ttf")
    if serif is None or sans is None:
        pytest.skip("need both a serif and a sans face installed")
    assert sift(str(serif)) == SERIF
    assert sift(str(sans)) == SANS


def test_known_monospace_face():
    mono = system_font("consola.ttf", "cour.ttf", "DejaVuSansMono.ttf",
                       "LiberationMono-Regular.ttf")
    if mono is None:
        pytest.skip("no monospace face on this machine")
    assert sift(str(mono)) == MONO


def test_collection_face_suffix_round_trips():
    """`file.ttc#3` is the registry's convention for one face inside a
    collection; sift has to index into the collection rather than choke."""
    ttc = system_font("cambria.ttc", "Cambria.ttc", "msgothic.ttc")
    if ttc is None:
        pytest.skip("no font collection on this machine")
    assert sift(f"{ttc}#0") in CATEGORIES


def test_result_is_memoized():
    assert sift("C:/nowhere/missing.ttf", "Noto Serif") == SERIF
    # the family argument is ignored on a cache hit -- the key is the path,
    # and one path is one face
    assert sift("C:/nowhere/missing.ttf", "Noto Sans") == SERIF
    sift_mod.clear_cache()
    assert sift("C:/nowhere/missing.ttf", "Noto Sans") == SANS


# -- the boundary --------------------------------------------------------

def test_font_coverage_carries_no_category_or_panose():
    """The classifier deliberately lives off the dataclass font_matching
    already holds. If a `category`/`panose` field ever lands on
    FontCoverage, "just peek at the metadata" becomes one attribute access
    away from a matcher whose entire job is to decide from pixels.
    """
    names = {f.name for f in dataclasses.fields(FontCoverage)}
    assert not {n for n in names if "panose" in n or "categ" in n or "serif" in n}


def test_font_matching_does_not_import_sift():
    import tofu.layers.font_matching as fm

    src = open(fm.__file__, encoding="utf-8").read()
    assert "sift" not in src


def test_registry_does_not_depend_on_sift():
    """sift imports nothing from fonts.py and fonts.py imports nothing from
    sift.py, so the dependency runs one way only: server -> both."""
    import tofu.layers.fonts as fonts_mod

    src = open(fonts_mod.__file__, encoding="utf-8").read()
    assert "sift" not in src


# -- italic, which the registry now answers once for everybody -----------

def test_families_report_italic_from_subfamily():
    registry = FontRegistry()
    glyphs = {ord(ch) for ch in "MURS "}
    registry._fonts = {
        "a.ttf": FontCoverage("a.ttf", "Fake Sans", "Regular", 400, glyphs),
        "b.ttf": FontCoverage("b.ttf", "Fake Sans", "Bold Italic", 700, glyphs),
        "c.ttf": FontCoverage("c.ttf", "Fake Sans", "Oblique", 400, glyphs),
    }
    weights = registry.families_with_weights("Latin", None, limit=None)[0]["weights"]
    by_sub = {w["subfamily"]: w["italic"] for w in weights}
    assert by_sub == {"Regular": False, "Bold Italic": True, "Oblique": True}
