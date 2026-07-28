## 🍢 the render honours the face the bouquet agreed on
"""
For as long as font matching has existed, the Translate preview drew a
region's matched face while /api/render drew the generic auto fallback:
a localiser approved one typeface and shipped another, and the frontend
ladder carried a whole ``useMatch: false`` mode just to describe the
discrepancy.  ``_matched_faces_applied`` closes it from the render side.

What must stay true is the boundary underneath.  Font matching is still
evidence a human accepts -- these tests exist to prove the render borrows
the face and gives it straight back, so nothing is silently written into a
manifest that only ever holds explicit choices.
"""

import copy
import dataclasses
import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "server"))

from conftest import system_font
from tofu.core.types import BBox, InstText, StyleProfil, TextManifest
from tofu.layers import scribe

main = pytest.importorskip("main")


def _inst(rid="r1", *, font_match=None, style_profile=None, target="MURS"):
    return InstText(
        id=rid, bounding_box=BBox(x=20, y=20, width=200, height=60),
        text="MURS", target_text=target,
        style_profile=style_profile, font_match=font_match,
    )


def _match(path, status="matched"):
    return {
        "schema": 1, "status": status, "provider": "local_glyph_retrieval",
        "candidates": [],
        "recommended_substitute": {
            "font_path": path, "family": "Fake", "subfamily": "Regular", "score": 0.9,
        },
    }


def _manifest(*instances):
    return TextManifest(asset_id="a", total_regions=len(instances),
                        instances=list(instances), src_lang="fr")


MATCHED = r"C:\Windows\Fonts\somematch.ttf"
CHOSEN = r"C:\Windows\Fonts\thehumanschoice.ttf"


# -- the borrow ----------------------------------------------------------

def test_an_auto_region_renders_with_the_matched_face():
    inst = _inst(font_match=_match(MATCHED), style_profile=StyleProfil(color="#000000"))
    manifest = _manifest(inst)
    with main._matched_faces_applied(manifest) as borrowed:
        assert inst.style_profile.font_family == MATCHED
        assert borrowed == ["r1"]
    assert inst.style_profile.font_family is None


def test_the_style_profile_is_handed_back_exactly():
    """Not merely 'font_family is None again' -- the whole manifest has to
    compare equal, or something else leaked out of the render."""
    inst = _inst(font_match=_match(MATCHED), style_profile=StyleProfil(color="#123456"))
    manifest = _manifest(inst)
    before = copy.deepcopy(manifest)
    with main._matched_faces_applied(manifest):
        pass
    assert dataclasses.asdict(manifest) == dataclasses.asdict(before)


def test_a_region_with_no_style_profile_gets_none_back_not_an_empty_one():
    """A StyleProfil() left behind would look like a user having opened the
    style panel, which changes what other layers infer about the region."""
    inst = _inst(font_match=_match(MATCHED), style_profile=None)
    manifest = _manifest(inst)
    with main._matched_faces_applied(manifest):
        assert inst.style_profile is not None
        assert inst.style_profile.font_family == MATCHED
    assert inst.style_profile is None


def test_the_face_is_returned_even_when_the_render_raises():
    inst = _inst(font_match=_match(MATCHED), style_profile=StyleProfil())
    manifest = _manifest(inst)
    with pytest.raises(RuntimeError):
        with main._matched_faces_applied(manifest):
            raise RuntimeError("scribe blew up")
    assert inst.style_profile.font_family is None


# -- what it must not touch ----------------------------------------------

def test_an_explicit_pick_outranks_the_match():
    """Rung 1 beats rung 2 in the render for the same reason it does in the
    ladder: an inference never overrides a human."""
    inst = _inst(font_match=_match(MATCHED),
                 style_profile=StyleProfil(font_family=CHOSEN))
    manifest = _manifest(inst)
    with main._matched_faces_applied(manifest) as borrowed:
        assert inst.style_profile.font_family == CHOSEN
        assert borrowed == []
    assert inst.style_profile.font_family == CHOSEN


def test_an_unavailable_match_is_not_a_recommendation():
    inst = _inst(font_match=_match(MATCHED, status="unavailable"),
                 style_profile=StyleProfil())
    manifest = _manifest(inst)
    with main._matched_faces_applied(manifest) as borrowed:
        assert inst.style_profile.font_family is None
        assert borrowed == []


def test_a_region_with_no_evidence_is_left_on_auto():
    inst = _inst(font_match=None, style_profile=StyleProfil())
    manifest = _manifest(inst)
    with main._matched_faces_applied(manifest) as borrowed:
        assert inst.style_profile.font_family is None
        assert borrowed == []


def test_a_match_carrying_no_path_is_ignored():
    """A contextual reference (the French enamel card) has a family but no
    installed file behind it; it cannot be rendered with."""
    evidence = _match(MATCHED)
    evidence["recommended_substitute"] = {"family": "Plaak", "font_path": None}
    inst = _inst(font_match=evidence, style_profile=StyleProfil())
    manifest = _manifest(inst)
    with main._matched_faces_applied(manifest) as borrowed:
        assert inst.style_profile.font_family is None
        assert borrowed == []


def test_only_the_matched_regions_are_borrowed_from():
    matched = _inst("r1", font_match=_match(MATCHED), style_profile=StyleProfil())
    plain = _inst("r2", font_match=None, style_profile=StyleProfil())
    picked = _inst("r3", font_match=_match(MATCHED),
                   style_profile=StyleProfil(font_family=CHOSEN))
    manifest = _manifest(matched, plain, picked)
    with main._matched_faces_applied(manifest) as borrowed:
        assert borrowed == ["r1"]


# -- and it actually reaches the pixels -----------------------------------

def test_the_render_draws_the_matched_face():
    """The point of the whole exercise: two renders of one string, one
    reaching the face through style_profile and one through font_match,
    have to produce the same image -- and both differ from the fallback."""
    face = system_font("georgia.ttf", "Georgia.ttf", "DejaVuSerif.ttf",
                       "LiberationSerif-Regular.ttf")
    if face is None:
        pytest.skip("no distinctive serif face installed")
    face = str(face)

    def render(manifest):
        asset = Image.new("RGB", (260, 100), (255, 255, 255))
        with main._matched_faces_applied(manifest):
            return np.asarray(scribe.render(asset, manifest, "fr").convert("RGB"))

    via_match = render(_manifest(
        _inst(font_match=_match(face), style_profile=StyleProfil(color="#000000"))))
    via_pick = render(_manifest(
        _inst(style_profile=StyleProfil(color="#000000", font_family=face))))
    via_auto = render(_manifest(
        _inst(style_profile=StyleProfil(color="#000000"))))

    assert np.array_equal(via_match, via_pick), "the match must render as the pick does"
    assert via_match.min() < 250, "nothing was drawn at all"
    if not np.array_equal(via_pick, via_auto):
        # only meaningful when the platform's auto fallback is a different
        # face from the one under test
        assert not np.array_equal(via_match, via_auto)


def test_the_manifest_survives_a_real_render_unchanged():
    inst = _inst(font_match=_match(MATCHED), style_profile=StyleProfil(color="#000000"))
    manifest = _manifest(inst)
    before = copy.deepcopy(manifest)
    asset = Image.new("RGB", (260, 100), (255, 255, 255))
    with main._matched_faces_applied(manifest):
        scribe.render(asset, manifest, "fr")
    # scribe may legitimately set glyph_fallback on the instance; the style
    # profile itself must come back byte-for-byte
    assert manifest.instances[0].style_profile.font_family is None
    assert (dataclasses.asdict(manifest.instances[0].style_profile)
            == dataclasses.asdict(before.instances[0].style_profile))
