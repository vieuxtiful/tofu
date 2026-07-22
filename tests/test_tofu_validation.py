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
