## 🍢 VTM v1.0 — reference implementation against the published conformance vectors
import json

import pytest

from conftest import ROOT
from tofu.core.types import BBox, InstText, Mask, StyleProfil, TextManifest
from tofu.utils import vtm

SPEC_DIR = ROOT / "spec"
CONFORMANCE = SPEC_DIR / "conformance"
INDEX = json.loads((CONFORMANCE / "index.json").read_text(encoding="utf-8"))


def load(relative: str):
    return json.loads((CONFORMANCE / relative).read_text(encoding="utf-8"))


def sample_manifest(**manifest_kwargs):
    inst = InstText(
        id="r1",
        bounding_box=BBox(x=10, y=20, width=120, height=40),
        segmentation_mask=Mask(
            polygon=[(10, 20), (130, 20), (130, 60), (10, 60)], confidence=0.93),
        text="焼肉", target_text="Grilled Meat", language="ja",
        confidence=0.93, reading_order=1,
        style_profile=StyleProfil(
            font_family="arial.ttf", font_size=28, color="#1a1a1a",
            font_weight="bold", align_h="center", tsume=0.2,
            stroke_color="#ffffff", stroke_width=2.0),
    )
    kwargs = dict(
        asset_id="street-01", total_regions=1, instances=[inst],
        src_lang="ja", targ_lang="en", img_dim=(1024, 768),
    )
    kwargs.update(manifest_kwargs)
    return TextManifest(**kwargs)


class TestPublishedArtifacts:
    def test_spec_and_schema_are_published(self):
        assert (SPEC_DIR / "vtm-1.0.md").is_file()
        assert (SPEC_DIR / "vtm-1.0.schema.json").is_file()

    def test_schema_is_valid_json_and_declares_draft(self):
        """Assert the contract, not the prose. The title is documentation and
        may be reworded; the draft declaration is what tooling depends on."""
        schema = json.loads((SPEC_DIR / "vtm-1.0.schema.json").read_text(encoding="utf-8"))
        assert schema["$schema"].startswith("https://json-schema.org/draft/")
        assert "visual translation memory" in schema["title"].lower()
        assert schema["type"] == "object"

    def test_schema_and_implementation_agree_on_required_keys(self):
        """The spec and the code must not drift. Both state what is REQUIRED;
        this asserts they say the same thing."""
        schema = json.loads((SPEC_DIR / "vtm-1.0.schema.json").read_text(encoding="utf-8"))
        assert set(schema["required"]) == set(vtm.REQUIRED_TOP_LEVEL)
        assert set(schema["$defs"]["entry"]["required"]) == set(vtm.REQUIRED_ENTRY)

    def test_version_constant_matches_the_spec_filename(self):
        assert vtm.VTM_VERSION == "1.0"
        assert vtm.VTM_NAMESPACE == "urn:vieuxtiful:vtm:1.0"


@pytest.mark.parametrize("case", INDEX["valid"], ids=lambda c: c["file"])
class TestValidVectors:
    def test_reader_accepts(self, case):
        problems = vtm.validate(load(case["file"]))
        assert problems == [], f"{case['file']} should validate: {problems}"

    def test_reader_can_import(self, case):
        instances = vtm.import_vtm(load(case["file"]))
        assert isinstance(instances, list)


@pytest.mark.parametrize("case", INDEX["invalid"], ids=lambda c: c["file"])
class TestInvalidVectors:
    def test_reader_rejects(self, case):
        assert vtm.validate(load(case["file"])), f"{case['file']} should have been rejected"

    def test_rejected_for_the_right_reason(self, case):
        """Failing by accident is not conformance -- the message has to name
        the actual defect."""
        problems = " ".join(vtm.validate(load(case["file"])))
        assert case["expect"] in problems, (
            f"{case['file']}: expected a problem mentioning {case['expect']!r}, got {problems!r}")

    def test_import_raises_rather_than_silently_accepting(self, case):
        with pytest.raises(ValueError):
            vtm.import_vtm(load(case["file"]))


class TestExport:
    def test_export_validates_against_its_own_spec(self):
        assert vtm.validate(vtm.export_vtm(sample_manifest())) == []

    def test_languages_are_normalised_to_bcp47(self):
        doc = vtm.export_vtm(sample_manifest())
        assert doc["source_language"] == "ja-JP"
        assert doc["target_language"] == "en-US"

    def test_asset_dimensions_are_required(self):
        """The spec's central requirement: geometry without a resolution is
        uninterpretable, so exporting it is refused rather than emitted."""
        with pytest.raises(ValueError, match="dimensions"):
            vtm.export_vtm(sample_manifest(img_dim=None))

    def test_explicit_image_size_overrides_manifest(self):
        doc = vtm.export_vtm(sample_manifest(img_dim=None), image_size=(640, 480))
        assert doc["asset"]["width"] == 640 and doc["asset"]["height"] == 480

    def test_generator_version_tracks_the_package(self):
        from tofu import __version__
        assert vtm.export_vtm(sample_manifest())["generator"]["version"] == __version__

    def test_dnt_and_excluded_regions_are_omitted(self):
        m = sample_manifest()
        m.instances[0].dnt = True
        assert vtm.export_vtm(m)["entries"] == []
        m.instances[0].dnt = False
        m.instances[0].excluded = True
        assert vtm.export_vtm(m)["entries"] == []

    def test_untranslated_omitted_by_default_but_optional(self):
        m = sample_manifest()
        m.instances[0].target_text = None
        assert vtm.export_vtm(m)["entries"] == []
        assert len(vtm.export_vtm(m, include_untranslated=True)["entries"]) == 1


class TestStyleSplit:
    def test_portable_style_uses_neutral_names(self):
        style = vtm.export_vtm(sample_manifest())["entries"][0]["style"]
        assert style["weight"] == "bold"      # not font_weight
        assert style["align"] == "center"     # not align_h
        assert style["stroke"] == {"color": "#ffffff", "width": 2.0}

    def test_vendor_specific_fields_go_to_extension(self):
        """tsume is a CJK compression control no other implementation can be
        expected to honour, so it must not sit in the portable style."""
        entry = vtm.export_vtm(sample_manifest())["entries"][0]
        assert "tsume" not in entry["style"]
        assert entry["x-tofu"]["style"]["tsume"] == 0.2

    def test_extension_survives_a_round_trip(self):
        back = vtm.import_vtm(vtm.export_vtm(sample_manifest()))[0]
        assert back.style_profile.tsume == 0.2

    def test_foreign_extension_cannot_inject_attributes(self):
        """An x-tofu block from an untrusted document must not be able to set
        arbitrary attributes on StyleProfil."""
        doc = vtm.export_vtm(sample_manifest())
        doc["entries"][0]["x-tofu"]["style"]["not_a_real_field"] = "evil"
        back = vtm.import_vtm(doc)[0]
        assert not hasattr(back.style_profile, "not_a_real_field")


class TestRoundTrip:
    def test_core_fields_survive(self):
        original = sample_manifest().instances[0]
        back = vtm.import_vtm(vtm.export_vtm(sample_manifest()))[0]
        assert back.id == original.id
        assert back.text == original.text
        assert back.target_text == original.target_text
        assert back.bounding_box == original.bounding_box
        assert back.reading_order == original.reading_order
        assert back.confidence == original.confidence

    def test_style_survives(self):
        back = vtm.import_vtm(vtm.export_vtm(sample_manifest()))[0]
        assert back.style_profile.font_family == "arial.ttf"
        assert back.style_profile.font_size == 28
        assert back.style_profile.font_weight == "bold"
        assert back.style_profile.align_h == "center"
        assert back.style_profile.stroke_width == 2.0

    def test_polygon_is_carried(self):
        geometry = vtm.export_vtm(sample_manifest())["entries"][0]["geometry"]
        assert geometry["polygon"][0] == [10, 20]


class TestValidatorBehaviour:
    def test_reports_every_problem_not_just_the_first(self):
        """A human fixing a hand-edited memory wants the whole list."""
        problems = vtm.validate({"vtm_version": "1.0"})
        assert len(problems) >= 3

    def test_non_object_rejected(self):
        assert vtm.validate([]) == ["document is not a JSON object"]
        assert vtm.validate("nope") == ["document is not a JSON object"]

    def test_bbox_containment_names_the_likely_cause(self):
        doc = load("valid/minimal.vtm.json")
        doc["entries"][0]["geometry"]["bbox"]["width"] = 99999
        assert any("different resolution" in p for p in vtm.validate(doc))
