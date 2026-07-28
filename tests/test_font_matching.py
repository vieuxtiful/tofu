"""Visual font retrieval remains evidence only; it never mutates the style."""

from pathlib import Path

import pytest
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from conftest import system_font

from tofu.core.types import BBox, CharactText, InstText, SceneRegion, StyleProfil, TextManifest
from tofu.layers.basil import bouquet
from tofu.layers.font_matching import (
    GlyphProfile, _eligible_faces, _glyph_profile, _relative_agreement, _visual_score,
    _weight_target,
    agree_on_face, external_catalog_match, local_match,
)
from tofu.layers.fonts import FontCoverage, FontRegistry


def _font_path(name: str) -> str | None:
    candidates = [
        Path("C:/Windows/Fonts") / name,
        Path("/usr/share/fonts/truetype/dejavu") / name,
        Path("/Library/Fonts") / name,
    ]
    for candidate in candidates:
        if candidate.exists():
            return str(candidate)
    return None


def _registry(exact: str, other: str) -> FontRegistry:
    glyphs = {ord(ch) for ch in "MURS "}
    registry = FontRegistry()
    registry._fonts = {
        exact: FontCoverage(exact, "Exact Sans", "Bold", 700, glyphs),
        other: FontCoverage(other, "Different Serif", "Bold", 700, glyphs),
    }
    return registry


def test_local_glyph_match_ranks_exact_face_without_changing_user_style():
    exact = _font_path("arialbd.ttf") or _font_path("DejaVuSans-Bold.ttf")
    other = _font_path("timesbd.ttf") or _font_path("DejaVuSerif-Bold.ttf")
    if not exact or not other:
        pytest.skip("two known system font faces are required for glyph retrieval test")

    font = ImageFont.truetype(exact, 54)
    image = Image.new("RGB", (300, 120), "white")
    draw = ImageDraw.Draw(image)
    box = draw.textbbox((0, 0), "MURS", font=font)
    draw.text((16 - box[0], 24 - box[1]), "MURS", font=font, fill="black")
    inst = InstText(id="r4", bounding_box=BBox(10, 18, box[2] - box[0] + 12, box[3] - box[1] + 12), text="MURS")

    result = local_match(np.asarray(image), inst, _registry(exact, other))

    assert result is not None
    assert result["candidates"][0]["family"] == "Exact Sans"
    assert result["recommended_substitute"]["font_path"] == exact
    assert inst.style_profile is None  # retrieval must never silently select a font


def test_unconfigured_catalog_never_sends_a_crop(monkeypatch):
    monkeypatch.delenv("TOFU_WHATFONTIS_API_KEY", raising=False)
    inst = InstText(id="r1", bounding_box=BBox(0, 0, 20, 20), text="Rue", font_match={"candidates": []})
    manifest = TextManifest(asset_id="a", total_regions=1, instances=[inst])

    assert external_catalog_match("not-opened.png", manifest, None) == 0
    assert inst.font_match["external_provider"]["enabled"] is False


class TestBouquet:
    """One sign, one hand: which regions may share a typeface.

    Deliberately narrower than a semantic unit -- a headline and its own
    fine print belong to one phrase but not to one face.
    """

    def _sign_line(self, rid, x, y, w, h, text, color="#ffffff", weight=None, italic=None):
        return InstText(
            id=rid, bounding_box=BBox(x, y, w, h), text=text,
            style_profile=StyleProfil(color=color, font_weight=weight, italic=italic),
        )

    def _manifest(self, *instances):
        return TextManifest(asset_id="a", total_regions=len(instances), instances=list(instances))

    def test_two_lines_of_one_plaque_are_one_bouquet(self):
        # the la-rue-sans-nom geometry: stacked lines, same ink, same weight
        manifest = self._manifest(
            self._sign_line("r1", 300, 250, 314, 100, "La rue"),
            self._sign_line("r2", 240, 400, 543, 110, "SANS-NOM"),
        )
        assert bouquet(manifest) == [{"id": "c1", "region_ids": ["r1", "r2"]}]

    def test_a_lone_region_is_not_a_bouquet(self):
        manifest = self._manifest(self._sign_line("r1", 0, 0, 100, 40, "Rue"))
        assert bouquet(manifest) == []

    def test_different_ink_colour_is_a_different_hand(self):
        manifest = self._manifest(
            self._sign_line("r1", 300, 250, 314, 100, "La rue", color="#ffffff"),
            self._sign_line("r2", 240, 400, 543, 110, "SANS-NOM", color="#101010"),
        )
        assert bouquet(manifest) == []

    def test_different_weight_is_a_different_hand(self):
        manifest = self._manifest(
            self._sign_line("r1", 300, 250, 314, 100, "La rue", weight="bold"),
            self._sign_line("r2", 240, 400, 543, 110, "SANS-NOM", weight="light"),
        )
        assert bouquet(manifest) == []

    def test_italic_never_shares_a_hand_with_upright(self):
        manifest = self._manifest(
            self._sign_line("r1", 300, 250, 314, 100, "La rue", italic=True),
            self._sign_line("r2", 240, 400, 543, 110, "SANS-NOM", italic=False),
        )
        assert bouquet(manifest) == []

    def test_signage_across_a_street_is_not_one_hand(self):
        # same ink and size, but nowhere near each other
        manifest = self._manifest(
            self._sign_line("r1", 0, 0, 200, 60, "Rue"),
            self._sign_line("r2", 3000, 2000, 200, 60, "Place"),
        )
        assert bouquet(manifest) == []

    def test_excluded_regions_take_no_part(self):
        excluded = self._sign_line("r2", 240, 400, 543, 110, "SANS-NOM")
        excluded.excluded = True
        manifest = self._manifest(self._sign_line("r1", 300, 250, 314, 100, "La rue"), excluded)
        assert bouquet(manifest) == []

    def test_a_users_font_pick_cannot_split_the_signs_evidence(self):
        """The regression this whole class exists to protect.

        Capture measured all three lines of the rue-vieux plaque as
        'regular'.  A localiser then picked a bold face for two of them.
        Weight bucketing used to read the PICK before the measurement, so
        the third line dropped into a different bucket, ``_same_hand``
        failed on every pair crossing that line, and one sign produced two
        bouquets that went on to recommend different faces.  What the user
        chooses afterwards cannot change how the sign was printed.
        """
        lines = [
            self._sign_line("r1", 228, 109, 235, 71, "Rue des"),
            self._sign_line("r2", 281, 206, 175, 72, "MURS", weight="Bold"),
            self._sign_line("r3", 89, 209, 176, 69, "VIEUX", weight="Bold"),
        ]
        for line in lines:
            line.characteristics = CharactText(font_style="regular")
        assert bouquet(self._manifest(*lines)) == [
            {"id": "c1", "region_ids": ["r1", "r2", "r3"]}
        ]

    def test_measured_weight_still_separates_two_real_hands(self):
        """The gate above must not become a no-op: a line capture actually
        measured as bold does not share a hand with one measured light."""
        heavy = self._sign_line("r1", 300, 250, 314, 100, "La rue")
        heavy.characteristics = CharactText(font_style="bold")
        faint = self._sign_line("r2", 240, 400, 543, 110, "SANS-NOM")
        faint.characteristics = CharactText(font_style="light")
        assert bouquet(self._manifest(heavy, faint)) == []


class TestBouquetPanels:
    """A confident bordered sign answers the proximity question by itself.

    ``_nearby`` allows a vertical gap of 1.6x glyph height.  A large plaque
    with generously leaded lines exceeds that while still being one sign in
    one face, so scene evidence gets its own route into the bouquet -- the
    same reasoning ``bunch()`` applies to panels.  Only proximity is waived:
    every typographic gate still has to pass.
    """

    def _line(self, rid, x, y, w, h, text, lang="fr", color="#ffffff"):
        return InstText(
            id=rid, bounding_box=BBox(x, y, w, h), text=text,
            detected_language=lang,
            style_profile=StyleProfil(color=color),
            characteristics=CharactText(font_style="regular"),
        )

    def _manifest(self, *instances, panel=None, confidence=0.9, label="bordered_region"):
        regions = []
        if panel is not None:
            regions.append(SceneRegion(bbox=panel, semantic_label=label, confidence=confidence))
        return TextManifest(
            asset_id="a", total_regions=len(instances),
            instances=list(instances), scene_regions=regions,
        )

    # far enough apart that _nearby fails: 400px gap on 60px glyphs
    FAR_A = ("r1", 100, 100, 300, 60, "GRAND")
    FAR_B = ("r2", 100, 560, 300, 60, "HOTEL")
    PANEL = BBox(50, 50, 500, 700)

    def test_a_confident_panel_binds_lines_too_far_apart_to_be_nearby(self):
        manifest = self._manifest(
            self._line(*self.FAR_A), self._line(*self.FAR_B), panel=self.PANEL,
        )
        assert bouquet(manifest) == [{"id": "c1", "region_ids": ["r1", "r2"]}]

    def test_without_a_panel_the_same_geometry_stays_apart(self):
        """Proves the pre-pass is what did the work, not a loosened gate."""
        manifest = self._manifest(self._line(*self.FAR_A), self._line(*self.FAR_B))
        assert bouquet(manifest) == []

    def test_a_panel_does_not_bind_a_headline_to_its_fine_print(self):
        """The case ``_same_hand``'s docstring names: one phrase, two faces.
        Proximity is waived inside a panel; glyph height is not."""
        manifest = self._manifest(
            self._line("r1", 100, 100, 400, 120, "GRAND"),
            self._line("r2", 100, 560, 200, 20, "since 1897"),
            panel=self.PANEL,
        )
        assert bouquet(manifest) == []

    def test_a_panel_does_not_bind_a_headline_to_its_romanisation(self):
        """Two scripts on one sign rarely share a typeface.  The language
        gate lives on this permissive path precisely because it is the one
        that can reach across a whole plaque."""
        manifest = self._manifest(
            self._line("r1", 100, 100, 300, 60, "大博食堂", lang="ja"),
            self._line("r2", 100, 560, 300, 60, "Daihaku", lang="en"),
            panel=self.PANEL,
        )
        assert bouquet(manifest) == []

    def test_a_panel_does_not_bind_across_different_ink(self):
        manifest = self._manifest(
            self._line(*self.FAR_A, color="#ffffff"),
            self._line(*self.FAR_B, color="#101010"),
            panel=self.PANEL,
        )
        assert bouquet(manifest) == []

    def test_an_unconfident_panel_says_nothing(self):
        manifest = self._manifest(
            self._line(*self.FAR_A), self._line(*self.FAR_B),
            panel=self.PANEL, confidence=0.2,
        )
        assert bouquet(manifest) == []

    def test_a_text_cluster_is_not_a_sign(self):
        """Only bordered things speak for a shared physical surface;
        ``text_cluster`` is an implementation grouping."""
        manifest = self._manifest(
            self._line(*self.FAR_A), self._line(*self.FAR_B),
            panel=self.PANEL, label="text_cluster",
        )
        assert bouquet(manifest) == []

    def test_regions_outside_the_panel_are_not_bound_by_it(self):
        outside = self._line("r2", 100, 1400, 300, 60, "HOTEL")
        manifest = self._manifest(self._line(*self.FAR_A), outside, panel=self.PANEL)
        assert bouquet(manifest) == []

    def test_the_pre_pass_never_breaks_a_pairwise_grouping(self):
        """Union-find only ever merges, so adding scene evidence cannot
        take away a bouquet the pairwise pass would have made."""
        adjacent = (
            self._line("r1", 300, 250, 314, 100, "La rue"),
            self._line("r2", 240, 400, 543, 110, "SANS-NOM"),
        )
        assert bouquet(self._manifest(*adjacent)) == [{"id": "c1", "region_ids": ["r1", "r2"]}]
        with_panel = self._manifest(*adjacent, panel=BBox(0, 0, 2000, 2000), label="text_cluster")
        assert bouquet(with_panel) == [{"id": "c1", "region_ids": ["r1", "r2"]}]


class TestCohortConsensus:
    """A per-region winner is one small sample's opinion; a sign has one face."""

    def _members(self):
        # scores taken from the real la-rue-sans-nom measurement: Centaur is
        # r1's favourite and r2's WORST, which is exactly what maximin rejects
        def inst(rid, w, h, text, candidates, top):
            return InstText(
                id=rid, bounding_box=BBox(0, 0, w, h), text=text,
                font_match={
                    "candidates": candidates,
                    "recommended_substitute": top,
                },
            )
        serif = {"family": "Centaur", "subfamily": "Regular", "font_path": "centaur.ttf", "score": 0.784}
        sans = {"family": "Franklin Gothic", "subfamily": "Regular", "font_path": "franklin.ttf", "score": 0.813}
        return [
            inst("r1", 314, 100, "La rue", [serif, sans], dict(serif)),
            inst("r2", 543, 110, "SANS-NOM", [sans, serif], dict(sans)),
        ]

    def _run(self, monkeypatch, scores):
        """Drive the consensus with a stubbed scoring kernel.

        `**kwargs` on the score stub is load-bearing: agree_on_face passes
        the bouquet's pooled typographic reading through `source_profile`,
        and this test is about the consensus POLICY, not the kernel.
        """
        import tofu.layers.font_matching as fm

        monkeypatch.setattr(fm, "_source_mask", lambda img, inst: np.ones((20, 40), dtype=bool))
        monkeypatch.setattr(fm, "_render_mask", lambda path, text, height: (path, text))
        monkeypatch.setattr(fm, "_glyph_profile", lambda mask: fm.GlyphProfile(1.0, 0.5, 1.0, 0.5))
        monkeypatch.setattr(fm, "_visual_score", lambda src, cand, **kw: (scores[cand[0]][cand[1]], {}))
        members = self._members()
        manifest = TextManifest(asset_id="a", total_regions=2, instances=members)
        cohorts = [{"id": "c1", "region_ids": ["r1", "r2"]}]
        agree_on_face(None, manifest, cohorts, FontRegistry())
        return members

    def test_rejects_the_face_that_is_best_for_one_member_and_worst_for_the_other(self, monkeypatch):
        scores = {
            "centaur.ttf": {"La rue": 0.784, "SANS-NOM": 0.463},
            "franklin.ttf": {"La rue": 0.713, "SANS-NOM": 0.813},
        }
        r1, r2 = self._run(monkeypatch, scores)
        # both regions end on ONE face -- the sign cannot have two
        assert r1.font_match["recommended_substitute"]["font_path"] == "franklin.ttf"
        assert r2.font_match["recommended_substitute"]["font_path"] == "franklin.ttf"

    def test_preserves_each_region_own_favourite_for_audit(self, monkeypatch):
        scores = {
            "centaur.ttf": {"La rue": 0.784, "SANS-NOM": 0.463},
            "franklin.ttf": {"La rue": 0.713, "SANS-NOM": 0.813},
        }
        r1, _ = self._run(monkeypatch, scores)
        assert r1.font_match["region_substitute"]["family"] == "Centaur"

    def test_records_which_regions_were_overruled_and_by_how_much(self, monkeypatch):
        scores = {
            "centaur.ttf": {"La rue": 0.784, "SANS-NOM": 0.463},
            "franklin.ttf": {"La rue": 0.713, "SANS-NOM": 0.813},
        }
        r1, _ = self._run(monkeypatch, scores)
        dissent = {d["region_id"]: d for d in r1.font_match["cohort"]["dissent"]}
        assert dissent["r1"]["preferred"] == "Centaur"
        assert dissent["r1"]["cohort_score"] == pytest.approx(0.713, abs=1e-3)
        assert "r2" not in dissent  # r2 already preferred the winner

    def test_a_face_that_suits_everyone_wins_outright_with_no_dissent(self, monkeypatch):
        scores = {
            "centaur.ttf": {"La rue": 0.90, "SANS-NOM": 0.90},
            "franklin.ttf": {"La rue": 0.50, "SANS-NOM": 0.50},
        }
        r1, r2 = self._run(monkeypatch, scores)
        assert r1.font_match["recommended_substitute"]["font_path"] == "centaur.ttf"
        assert r1.font_match["cohort"]["agreement"] == pytest.approx(1.0)

    def test_never_writes_a_font_into_the_style_profile(self, monkeypatch):
        scores = {
            "centaur.ttf": {"La rue": 0.784, "SANS-NOM": 0.463},
            "franklin.ttf": {"La rue": 0.713, "SANS-NOM": 0.813},
        }
        r1, r2 = self._run(monkeypatch, scores)
        # the invariant this whole module is built on still holds
        assert r1.style_profile is None and r2.style_profile is None

    def test_no_cohort_leaves_evidence_untouched(self, monkeypatch):
        import tofu.layers.font_matching as fm

        members = self._members()
        manifest = TextManifest(asset_id="a", total_regions=2, instances=members)
        assert agree_on_face(None, manifest, [], FontRegistry()) == 0
        assert members[0].font_match["recommended_substitute"]["family"] == "Centaur"
        assert "cohort" not in members[0].font_match


def _rendered_mask(font_path, text, size=128):
    """Tight glyph mask for `text` set in `font_path` -- the same thing
    text_mask() extracts from a photograph, minus the photograph.

    128px because stroke contrast needs the resolution: Times New Roman's
    hairlines quantise to its stem width below roughly 90px of ink, and the
    ratio then reads a flat 1.0 that is indistinguishable from a genuine
    monoline.  Real captures clear this easily -- the la-rue-sans-nom
    plaque's lettering is 85px of ink at a heavier weight.
    """
    font = ImageFont.truetype(str(font_path), size)
    probe = Image.new("L", (8, 8), 0)
    box = ImageDraw.Draw(probe).textbbox((0, 0), text, font=font)
    canvas = Image.new("L", (box[2] - box[0] + 24, box[3] - box[1] + 24), 0)
    ImageDraw.Draw(canvas).text((12 - box[0], 12 - box[1]), text, font=font, fill=255)
    arr = np.asarray(canvas) > 96
    ys, xs = np.nonzero(arr)
    return arr[ys.min():ys.max() + 1, xs.min():xs.max() + 1]


class TestTypographicFeatures:
    """The kernel has to be able to see a serif.

    Asserted comparatively rather than against absolute thresholds: the
    numbers are string-dependent (the same plaque reads 1.54 on its
    lowercase line and 2.11 on its capitals), so only the ORDERING between
    a serif and a sans set in the same string is meaningful.
    """

    @pytest.fixture(scope="class")
    @classmethod
    def faces(cls):
        serif = system_font("times.ttf", "Times New Roman.ttf", "DejaVuSerif.ttf")
        sans = system_font("arial.ttf", "Arial.ttf", "DejaVuSans.ttf")
        if not serif or not sans:
            pytest.skip("a serif and a sans system face are required")
        return serif, sans

    @pytest.mark.parametrize("text", ["La rue", "SANS-NOM", "Handgloves"])
    def test_serif_structure_is_higher_for_a_serif_than_a_sans(self, faces, text):
        serif, sans = faces
        serif_profile = _glyph_profile(_rendered_mask(serif, text))
        sans_profile = _glyph_profile(_rendered_mask(sans, text))
        if serif_profile.serif is None or sans_profile.serif is None:
            pytest.skip("scikit-image unavailable; the feature fails open")
        assert serif_profile.serif > sans_profile.serif, text

    @pytest.mark.parametrize("text", ["La rue", "Handgloves"])
    def test_stroke_contrast_is_higher_for_a_serif_than_a_sans(self, faces, text):
        """Lowercase only, and deliberately so -- see the all-caps test
        below, which records where this feature stops working."""
        serif, sans = faces
        serif_profile = _glyph_profile(_rendered_mask(serif, text))
        sans_profile = _glyph_profile(_rendered_mask(sans, text))
        if serif_profile.contrast is None or sans_profile.contrast is None:
            pytest.skip("scikit-image unavailable; the feature fails open")
        assert serif_profile.contrast > sans_profile.contrast, text

    def test_a_short_all_capital_string_is_the_weak_case(self, faces):
        """A known limitation, asserted so it stays known.

        S, A, N, O and M give the measurements almost nothing to work with:
        no hairline for contrast to compare a stem against, and eight
        near-identical stem terminals for the serif count.  Measured on
        Times against Arial, BOTH agreements sit above 0.85 on "SANS-NOM"
        while a lowercase string separates them an order of magnitude
        better.  This is why the cohort exists -- a sign's lowercase line
        is what identifies its face, and its capitals ride along.
        """
        serif, sans = faces

        def separation(text):
            serif_profile = _glyph_profile(_rendered_mask(serif, text))
            sans_profile = _glyph_profile(_rendered_mask(sans, text))
            gaps = [
                _relative_agreement(serif_profile.contrast, sans_profile.contrast),
                _relative_agreement(serif_profile.serif, sans_profile.serif),
            ]
            # lower agreement = better separation between the two classes
            return min(g for g in gaps if g is not None)

        assert separation("SANS-NOM") > 0.8
        assert separation("Handgloves") < separation("SANS-NOM")

    def test_a_serif_source_scores_its_own_face_over_a_sans(self, faces):
        """The regression this whole change exists for: silhouette alone
        ranked a grotesk above the true serif on the la-rue-sans-nom
        plaque, 0.8130 against 0.4634."""
        serif, sans = faces
        source = _rendered_mask(serif, "Handgloves")
        serif_score, _ = _visual_score(source, _rendered_mask(serif, "Handgloves"))
        sans_score, _ = _visual_score(source, _rendered_mask(sans, "Handgloves"))
        assert serif_score > sans_score

    def test_the_typographic_terms_are_what_makes_the_difference(self, faces):
        """Guards the assertion above against passing for some unrelated
        reason: the new evidence has to be doing the work."""
        serif, sans = faces
        source = _rendered_mask(serif, "Handgloves")
        right, wrong = _rendered_mask(serif, "Handgloves"), _rendered_mask(sans, "Handgloves")
        full_gap = _visual_score(source, right)[0] - _visual_score(source, wrong)[0]
        silhouette_gap = (_visual_score(source, right, typographic=False)[0]
                          - _visual_score(source, wrong, typographic=False)[0])
        assert full_gap > silhouette_gap

    def test_a_wrong_class_candidate_is_penalised_categorically(self, faces):
        serif, sans = faces
        _, evidence = _visual_score(
            _rendered_mask(serif, "Handgloves"), _rendered_mask(sans, "Handgloves"))
        assert "class_mismatch_penalty" in evidence

    def test_a_right_class_candidate_is_not_penalised(self, faces):
        serif, _ = faces
        _, evidence = _visual_score(
            _rendered_mask(serif, "Handgloves"), _rendered_mask(serif, "Handgloves"))
        assert "class_mismatch_penalty" not in evidence

    def test_the_original_evidence_keys_survive(self, faces):
        """Historical manifests carry these four; they must not be renamed."""
        serif, sans = faces
        _, evidence = _visual_score(
            _rendered_mask(serif, "La rue"), _rendered_mask(sans, "La rue"))
        assert {"dice", "chamfer", "projection", "aspect"} <= set(evidence)

    def test_visual_score_still_returns_a_score_and_a_dict(self, faces):
        serif, _ = faces
        mask = _rendered_mask(serif, "La rue")
        result = _visual_score(mask, mask)
        assert isinstance(result, tuple) and len(result) == 2
        assert isinstance(result[0], float) and isinstance(result[1], dict)

    def test_an_unreadable_mask_fails_open_rather_than_raising(self):
        assert _glyph_profile(np.ones((4, 4), dtype=bool)) == GlyphProfile(None, 0.0, None, 0.0)

    def test_a_supplied_source_profile_overrides_the_measured_one(self, faces):
        """How a bouquet lends its reading to a region that cannot measure.

        The plaque's capitals cannot tell a serif from a sans on their own;
        its lowercase line can, and the two are one sign.
        """
        serif, sans = faces
        source = _rendered_mask(sans, "Handgloves")
        candidate = _rendered_mask(sans, "Handgloves")
        own, _ = _visual_score(source, candidate)
        lent, _ = _visual_score(
            source, candidate,
            source_profile=_glyph_profile(_rendered_mask(serif, "Handgloves")))
        assert lent < own, "a serif reading must make a sans candidate score worse"


class TestCandidatePool:
    """A face that cannot be reached is worse than one scored and rejected."""

    def _latin_registry(self):
        anchor = system_font("arial.ttf", "Arial.ttf", "DejaVuSans.ttf")
        if anchor is None:
            pytest.skip("no system font directory")
        registry = FontRegistry()
        registry.discover(str(anchor.parent))
        if len(registry._fonts) < 40:
            pytest.skip("too few installed faces to exercise the cap")
        return registry

    def _regular_inst(self, ratio=0.1179):
        return InstText(
            id="r1", bounding_box=BBox(0, 0, 314, 100), text="La rue",
            style_profile=StyleProfil(font_weight=None),
            characteristics=CharactText(size=81, positioning={"stroke_ratio": ratio}),
        )

    def test_weight_target_falls_back_to_the_measured_stroke_ratio(self):
        """typography stores font_weight only for heavy/bold/light, so a
        regular reading arrives as None and used to leave every face in the
        library equidistant."""
        assert _weight_target(self._regular_inst()) == 400

    def test_a_measured_bold_ratio_targets_a_bold_face(self):
        assert _weight_target(self._regular_inst(ratio=0.16)) == 700

    def test_an_explicit_label_still_wins_over_the_ratio(self):
        inst = self._regular_inst()
        inst.style_profile = StyleProfil(font_weight="bold")
        assert _weight_target(inst) == 700

    def test_no_evidence_at_all_still_yields_no_target(self):
        assert _weight_target(InstText(id="r1", bounding_box=BBox(0, 0, 10, 10), text="x")) is None

    def test_the_pool_is_no_longer_ordered_by_family_name(self):
        """Measured: it used to run 'Agency FB' through 'Gill Sans' and
        never reach Times New Roman, Palatino, Perpetua, Rockwell or
        Sylfaen -- 134 of 219 installed families were unreachable."""
        registry = self._latin_registry()
        families = [f.family for f in _eligible_faces(registry, "La rue", self._regular_inst())]
        assert families
        assert families != sorted(families)
