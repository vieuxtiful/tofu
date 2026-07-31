## 🍢 tofu pre-flight validation units
from tofu.core.types import (
    BBox, InstText, ScrptSpprt, TextManifest, VldtnSeverity,
)
from tofu.layers.tofu import (
    EXPANSION_FAIL, EXPANSION_WARN, ToFU, lang_to_script, validate,
)


def manifest_with(text: str, width: int, target: str | None = None,
                  src: str = "en") -> TextManifest:
    inst = InstText(
        id="r1",
        bounding_box=BBox(x=0, y=0, width=width, height=40),
        text=text, target_text=target,
    )
    return TextManifest(asset_id="a", total_regions=1, instances=[inst], src_lang=src)


class TestScriptSupport:
    def test_unknown_language_fails(self):
        report = validate(None, "xx-unknown")
        assert not report.passed
        assert any(i.code == "ToFU_000" for i in report.issues)

    def test_known_language_passes(self):
        report = validate(None, "es")
        assert report.passed

    def test_partial_script_warns_not_fails(self):
        # thai is PARTIAL in the static map (no registry configured in unit env)
        t = ToFU()  # force static-map path regardless of host fonts
        report = t.validate(None, "th")
        assert report.passed
        assert any(i.code == "ToFU_002" for i in report.issues)

    def test_every_mapped_language_has_iso_script(self):
        assert all(len(s) == 4 for s in lang_to_script.values())


class TestExpansionFeasibility:
    def _tofu(self):
        return ToFU()  # no registry: char-count / ratio paths

    def test_de_expansion_predicts_overflow_warning(self):
        # en → de is 1.35x: exactly at FAIL boundary; ratio path gives fit=1.35
        report = self._tofu().validate(None, "de", text_manifest=manifest_with("hello world", 200))
        fit = report.expansion_fit["r1"]
        assert fit > EXPANSION_WARN
        assert any(i.code == "ToFU_005" for i in report.issues)

    def test_zh_contraction_fits(self):
        report = self._tofu().validate(None, "zh-cn", text_manifest=manifest_with("hello world", 200))
        assert report.expansion_fit["r1"] < 1.0
        assert not any(i.code == "ToFU_005" for i in report.issues)

    def test_actual_translation_measured_not_predicted(self):
        # user shortened the translation: char-count proxy, no blocking error
        report = self._tofu().validate(
            None, "de", text_manifest=manifest_with("hello world", 200, target="kurz")
        )
        assert report.expansion_fit["r1"] < 1.0
        assert report.passed

    def test_actual_long_translation_warns_never_errors(self):
        # scribe shrinks to fit, so an ACTUAL overlong translation must be
        # a warning (renders small), not a blocking error
        long_target = "x" * 50
        report = self._tofu().validate(
            None, "de", text_manifest=manifest_with("hi", 200, target=long_target)
        )
        issues = [i for i in report.issues if i.code == "ToFU_005"]
        assert issues and all(i.severity == VldtnSeverity.WARNING for i in issues)
        assert report.passed

    def test_regions_without_text_skipped(self):
        m = manifest_with("", 200)
        report = self._tofu().validate(None, "de", text_manifest=m)
        assert report.expansion_fit == {}


class TestStaticSupportMap:
    def test_override_map_respected(self):
        t = ToFU(supported_scripts={"Latn": ScrptSpprt.UNSUPPORTED})
        report = t.validate(None, "en")
        assert not report.passed
        assert any(i.code == "ToFU_001" for i in report.issues)


class TestVisualFontEvidence:
    def test_font_match_becomes_reviewable_preflight_insight(self):
        manifest = manifest_with("MURS", 160)
        manifest.instances[0].font_match = {
            "provider": "local_glyph_retrieval", "status": "review",
            "confidence": 0.61,
            "recommended_substitute": {
                "family": "Arial Narrow", "subfamily": "Bold",
                "font_path": "C:/Windows/Fonts/arialn.ttf", "score": 0.79,
            },
            "candidates": [],
        }

        report = ToFU().validate(None, "it", text_manifest=manifest)

        insight = next(item for item in report.insights if item.kind == "font_substitute")
        assert insight.region_id == "r1"
        assert insight.severity == "review"
        assert insight.confidence == 0.61
        assert insight.family == "Arial Narrow"
        # ToFU must expose advice, never make an implicit style selection.
        assert manifest.instances[0].style_profile is None

    def test_licensed_style_reference_is_deduplicated_across_regions(self):
        manifest = manifest_with("Rue", 120)
        other = InstText(id="r2", bounding_box=BBox(x=0, y=45, width=140, height=40), text="MURS")
        manifest.instances.append(other)
        manifest.total_regions = 2
        reference = {
            "family": "Plaak", "license": "commercial", "available": False,
            "source": "contextual_style_reference", "foundry": "205TF",
            "url": "https://www.205.tf/Plaak", "reason": "Reference only.",
        }
        for inst in manifest.instances:
            inst.font_match = {"status": "review", "candidates": [reference]}

        report = ToFU().validate(None, "it", text_manifest=manifest)

        references = [item for item in report.insights if item.kind == "style_reference"]
        assert len(references) == 1
        assert references[0].region_ids == ["r1", "r2"]
        assert references[0].license == "commercial"


class TestPipelineWiring:
    """The pipeline's ToFU stage used to call the module-level singleton,
    which is constructed with no font library — so a caller already
    holding a real FontRegistry still got static-map validation, and the
    request font never reached ToFU_005's measurement path at all."""

    def _pipeline(self, registry=None):
        from tofu.core.pipeline import TofuPipeline
        return TofuPipeline(font_registry=registry)

    def test_injected_registry_reaches_the_validator(self):
        from tofu.layers.fonts import FontRegistry
        registry = FontRegistry()
        assert self._pipeline(registry)._validator.font_registry is registry

    def test_registry_injection_avoids_a_second_font_scan(self):
        # ToFU must ACCEPT a built registry, not rediscover from a path:
        # discovery walks the whole font directory with fontTools.
        from tofu.layers.fonts import FontRegistry
        registry = FontRegistry()
        assert ToFU(font_registry=registry).font_registry is registry

    def test_request_font_is_threaded_into_context(self):
        ctx = self._pipeline()._tofu_context("C:/Windows/Fonts/arial.ttf", None)
        assert ctx == {"font": "C:/Windows/Fonts/arial.ttf"}

    def test_context_carries_smallest_region_size_and_effects(self):
        from tofu.core.types import CharactText, StyleProfil
        manifest = manifest_with("hello", 200)
        manifest.instances[0].characteristics = CharactText(size=22)
        manifest.instances[0].style_profile = StyleProfil(shadow={"blur": 2})
        small = InstText(
            id="r2", bounding_box=BBox(x=0, y=50, width=80, height=12),
            text="fine print", characteristics=CharactText(size=7),
        )
        manifest.instances.append(small)

        ctx = self._pipeline()._tofu_context(None, manifest)

        # legibility is governed by the text most likely to fail
        assert ctx["font_px"] == 7
        assert ctx["effects"] == ["shadow"]

    def test_context_is_none_when_there_is_nothing_to_say(self):
        assert self._pipeline()._tofu_context(None, None) is None


class TestScriptNormalizedExpansion:
    """A character count is not a width. The old proxy compared
    len(target)/len(source), which treats a CJK glyph and a latin letter
    as the same size — so every CJK→latin translation inherited a
    spurious overflow warning."""

    def _tofu(self):
        return ToFU()  # no registry: forces the no-metrics proxy

    def test_equal_width_cjk_to_latin_does_not_warn(self):
        # 出口 is two full-em glyphs; "Exit" is four half-em ones.
        # Same width — the old count proxy called this a 2x overflow.
        report = self._tofu().validate(
            None, "en", text_manifest=manifest_with("出口", 200, target="Exit", src="ja")
        )
        assert report.expansion_fit["r1"] == 1.0
        assert not any(i.code == "ToFU_005" for i in report.issues)

    def test_genuinely_longer_translation_still_warns(self):
        # the fix must not silence real overflow: 焼肉 -> "Viande grillée"
        # is a true ~3.5x, and scribe will have to shrink the face
        report = self._tofu().validate(
            None, "fr",
            text_manifest=manifest_with("焼肉", 200, target="Viande grillée", src="ja"),
        )
        assert report.expansion_fit["r1"] > EXPANSION_FAIL
        assert any(i.code == "ToFU_005" for i in report.issues)

    def test_combining_marks_do_not_add_width(self):
        # é as base + U+0301 must measure the same as precomposed é
        from tofu.layers.tofu import measure_in_ems
        assert measure_in_ems("e\u0301") == measure_in_ems("\u00e9")

    def test_fullwidth_latin_counts_as_full_em(self):
        from tofu.layers.tofu import measure_in_ems
        assert measure_in_ems("ＡＢ") == 2.0    # U+FF21/FF22, UAX #11 "F"
        assert measure_in_ems("AB") == 1.0


class TestGlyphSegmentationEvidence:
    """ToFU_003 used to be a restatement of its own input: the score took
    exactly two values across nine fixtures x twelve languages, identical
    for every image, and fired on precisely the complex-script target
    languages. It also asked about the TARGET language, though glyph
    segmentation is done to the SOURCE ink."""

    def _small_text(self, src="en"):
        insts = [
            InstText(id=f"r{n}", bounding_box=BBox(x=0, y=n * 12, width=60, height=9),
                     text="tiny")
            for n in range(6)
        ]
        return TextManifest(asset_id="a", total_regions=len(insts),
                            instances=insts, src_lang=src)

    def _large_text(self, src="en"):
        insts = [
            InstText(id=f"r{n}", bounding_box=BBox(x=0, y=n * 80, width=300, height=60),
                     text="BIG")
            for n in range(4)
        ]
        return TextManifest(asset_id="a", total_regions=len(insts),
                            instances=insts, src_lang=src)

    def test_no_manifest_raises_no_segmentation_issue(self):
        # nothing has been looked at yet; a guess made before examining a
        # single pixel is not evidence
        report = ToFU().validate(None, "ja")
        assert not any(i.code == "ToFU_003" for i in report.issues)
        assert report.glyph_segmentation_score is not None

    def test_small_source_text_warns(self):
        report = ToFU().validate(None, "en", text_manifest=self._small_text())
        assert report.glyph_segmentation_score < 0.6
        assert any(i.code == "ToFU_003" for i in report.issues)

    def test_large_source_text_does_not_warn(self):
        report = ToFU().validate(None, "ja", text_manifest=self._large_text())
        assert report.glyph_segmentation_score >= 0.6
        assert not any(i.code == "ToFU_003" for i in report.issues)

    def test_score_follows_source_script_not_target(self):
        # identical geometry, identical target: only the SOURCE differs,
        # and only the source is what gets segmented
        latin = ToFU().validate(None, "en", text_manifest=self._large_text(src="en"))
        cjk = ToFU().validate(None, "en", text_manifest=self._large_text(src="ja"))
        assert cjk.glyph_segmentation_score < latin.glyph_segmentation_score

    def test_target_language_alone_does_not_move_the_score(self):
        manifest = self._large_text(src="en")
        scores = {
            lang: ToFU().validate(None, lang, text_manifest=manifest).glyph_segmentation_score
            for lang in ("en", "ja", "ar", "th", "de")
        }
        assert len(set(scores.values())) == 1, scores

    def test_healthy_median_with_a_small_text_tail_scores_lower(self):
        # a scene can have a comfortable median and still be full of
        # unreadable signage -- the tail term is what catches that
        big = [InstText(id=f"b{n}", bounding_box=BBox(x=0, y=n * 80, width=300, height=60), text="BIG")
               for n in range(4)]
        tail = [InstText(id=f"t{n}", bounding_box=BBox(x=0, y=500 + n * 12, width=40, height=9), text="s")
                for n in range(3)]
        clean = TextManifest(asset_id="a", total_regions=4, instances=big, src_lang="en")
        mixed = TextManifest(asset_id="a", total_regions=7, instances=big + tail, src_lang="en")

        assert (ToFU().validate(None, "en", text_manifest=mixed).glyph_segmentation_score
                < ToFU().validate(None, "en", text_manifest=clean).glyph_segmentation_score)


class TestExpansionFactorCoverage:
    """EXPANSION_FACTORS names 31 of the 68 languages in lang_to_script.
    The other 37 fell through to a silent 1.0 — asserting that Khmer,
    Amharic and Malayalam are all exactly as wide as English."""

    def test_every_mapped_language_resolves_a_factor(self):
        from tofu.layers.tofu import expansion_factor
        # every english variant (en, en-US, en-GB, …) shares the 1.00
        # baseline; every other mapped language must resolve a non-1.0
        # factor, named or via its script default, so nothing silently
        # equals english.
        english_baseline = {lang for lang in lang_to_script
                            if lang.split("-")[0].lower() == "en"}
        unresolved = [
            lang for lang in lang_to_script
            if expansion_factor(lang) == 1.0 and lang not in english_baseline
        ]
        assert unresolved == [], unresolved

    def test_named_language_beats_its_script_default(self):
        from tofu.layers.tofu import EXPANSION_FACTORS, expansion_factor
        assert expansion_factor("de") == EXPANSION_FACTORS["de"] == 1.35

    def test_unnamed_language_inherits_its_script(self):
        from tofu.layers.tofu import SCRIPT_EXPANSION_FACTORS, expansion_factor
        # km -> Khmr; never listed individually
        assert expansion_factor("km") == SCRIPT_EXPANSION_FACTORS["Khmr"]

    def test_unknown_language_is_neutral(self):
        from tofu.layers.tofu import expansion_factor
        assert expansion_factor("xx-unknown") == 1.0
        assert expansion_factor(None) == 1.0


class TestExpansionBlockingRequiresEvidence:
    """Populating `src_lang` correctly made ToFU_005's predictive branch
    fire for real on CJK sources (ja→en is 1.67x, not the 1.0x an unset
    src_lang implied). That is a true and useful planning signal — but
    with no font selected, `fit` IS the language-pair ratio: one table
    lookup, identical for every region. A number carrying no per-region
    evidence must not fail a run per region, least of all before the user
    has entered a single translation."""

    def _cjk_manifest(self):
        insts = [
            InstText(id=f"r{n}", bounding_box=BBox(x=0, y=n * 40, width=90, height=30),
                     text="焼肉")
            for n in range(3)
        ]
        return TextManifest(asset_id="a", total_regions=3, instances=insts, src_lang="ja")

    def test_ratio_only_estimate_warns_but_does_not_block(self):
        report = ToFU().validate(None, "en", text_manifest=self._cjk_manifest())
        issues = [i for i in report.issues if i.code == "ToFU_005"]
        assert issues, "the expansion signal must still be surfaced"
        assert all(i.severity == VldtnSeverity.WARNING for i in issues)
        assert report.passed

    def test_ratio_only_message_states_its_own_weakness(self):
        report = ToFU().validate(None, "en", text_manifest=self._cjk_manifest())
        issue = next(i for i in report.issues if i.code == "ToFU_005")
        assert "expansion ratio alone" in issue.message

    def test_translated_region_still_only_warns(self):
        m = self._cjk_manifest()
        m.instances[0].target_text = "Grilled meat restaurant, second floor"
        report = ToFU().validate(None, "en", text_manifest=m)
        assert report.passed
