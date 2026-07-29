import dataclasses
import json

import pytest
from PIL import Image, ImageDraw

from tofu.core.types import BBox, CharactText, InstText, StyleProfil, TextManifest
from tofu.layers import verify
from tofu.layers.cicerone import classify_asset_context
from tofu.layers.verify import build_verification_report
from tofu.layers.fonts import FontCoverage, FontRegistry


@pytest.fixture(autouse=True)
def _disable_real_ocr(monkeypatch):
    """Contract tests inject deterministic OCR and never initialize models."""
    monkeypatch.setattr(verify, "_get_reader", lambda _lang: None)


def _instance(region_id, target="Bonjour", **overrides):
    values = {
        "id": region_id,
        "bounding_box": BBox(x=10, y=20, width=100, height=30),
        "text": "Hello",
        "target_text": target,
        "language": "en",
        "confidence": 0.92,
        "style_profile": StyleProfil(font_family="Inter"),
    }
    values.update(overrides)
    return InstText(**values)


def _rendered(*region_ids):
    image = Image.new("RGB", (200, 100), "white")
    image.text_masks = {region_id: object() for region_id in region_ids}
    return image


def _masked_render(masks):
    image = Image.new("RGB", (200, 100), "white")
    image.text_masks = {}
    for region_id, box in masks.items():
        mask = Image.new("L", image.size, 0)
        ImageDraw.Draw(mask).rectangle(box, fill=255)
        ImageDraw.Draw(image).rectangle(box, fill="black")
        image.text_masks[region_id] = mask
    return image


def _registry(path, codepoints):
    registry = FontRegistry()
    registry._fonts[path] = FontCoverage(
        font_path=path,
        family="Test",
        codepoints={ord(character) for character in codepoints},
    )
    return registry


def test_contract_records_evidence_and_six_component_slots():
    manifest = TextManifest(
        asset_id="asset-1",
        total_regions=1,
        instances=[_instance("r1")],
        src_lang="en",
        targ_lang="fr",
    )

    report = build_verification_report(_rendered("r1"), manifest)

    assert report.project.overall_status == "pass"
    assert report.project.component_scores["coverage"] == 100.0
    assert set(report.project.component_scores) == {
        "coverage",
        "content_integrity",
        "spatial_fit",
        "typographic_intent",
        "script_rendering_validity",
        "contextual_fit",
    }
    region = report.regions[0]
    assert region.inventory_status == "rendered"
    assert region.source_evidence.text == "Hello"
    assert region.source_evidence.bounds == BBox(10, 20, 100, 30)
    assert region.target_evidence.text == "Bonjour"
    assert region.target_evidence.language == "fr"
    # Script evidence is ISO 15924, not an ad-hoc lowercase name: the codes
    # are what lang_to_script already declares (Latn/Hans/Hant/Jpan/Kore), so
    # detected and declared script are finally in the same vocabulary.
    assert region.target_evidence.script == "Latn"
    assert region.target_evidence.scripts == ["Latn"]
    assert region.target_evidence.script_counts == {"Latn": 7}
    assert region.target_evidence.is_mixed_script is False
    assert region.target_evidence.han_variant is None
    assert region.target_evidence.font_family == "Inter"
    assert region.target_evidence.clean_render_present is True
    assert report.run_metadata["stage"] == "post_scribe_pre_garnish"
    json.dumps(dataclasses.asdict(report))


def test_inventory_distinguishes_intentional_omissions_from_gaps():
    manifest = TextManifest(
        asset_id="asset-1",
        total_regions=4,
        instances=[
            _instance("rendered"),
            _instance("untranslated", target=None),
            _instance("dnt", target=None, dnt=True),
            _instance("excluded", target=None, excluded=True),
        ],
        src_lang="en",
        targ_lang="fr",
    )

    report = build_verification_report(_rendered("rendered"), manifest)
    statuses = {region.region_id: region.inventory_status for region in report.regions}

    assert statuses == {
        "rendered": "rendered",
        "untranslated": "untranslated",
        "dnt": "do_not_translate",
        "excluded": "excluded",
    }
    assert report.project.overall_status == "review"
    assert report.project.region_totals["rendered"] == 1
    assert report.project.region_totals["untranslated"] == 1
    assert report.project.region_totals["do_not_translate"] == 1
    assert report.project.region_totals["excluded"] == 1
    assert "intentional_omission" in report.regions[2].flags


def test_inventory_reports_missing_and_extra_rendered_regions():
    manifest = TextManifest(
        asset_id="asset-1",
        total_regions=1,
        instances=[_instance("expected")],
        src_lang="en",
        targ_lang="fr",
    )

    report = build_verification_report(_rendered("unexpected"), manifest)
    statuses = {region.region_id: region.inventory_status for region in report.regions}

    assert statuses == {"expected": "missing", "unexpected": "extra"}
    assert report.project.overall_status == "fail"
    assert report.project.component_scores["coverage"] == 0.0
    assert report.project.summary_flags == ["missing_regions", "extra_rendered_regions"]


def test_phase2_records_exact_glyph_coverage_and_assigned_font():
    inst = _instance("r1", target="Bonjour")
    inst.style_profile.font_family = "complete.ttf"
    manifest = TextManifest(
        asset_id="asset-1", total_regions=1, instances=[inst],
        src_lang="en", targ_lang="fr",
    )

    report = build_verification_report(
        _masked_render({"r1": (20, 30, 80, 45)}),
        manifest,
        _registry("complete.ttf", "Bonjour"),
    )
    region = report.regions[0]

    assert region.checks["glyph_coverage"]["status"] == "pass"
    assert region.checks["glyph_coverage"]["coverage"] == 1.0
    assert region.checks["glyph_coverage"]["missing_codepoints"] == []
    assert region.target_evidence.font_family == "complete.ttf"
    assert region.scores["script_rendering_validity"] == 100.0


def test_phase2_unsupported_glyphs_are_a_critical_failure():
    inst = _instance("r1", target="AΩ")
    inst.style_profile.font_family = "latin.ttf"
    manifest = TextManifest(
        asset_id="asset-1", total_regions=1, instances=[inst],
        src_lang="en", targ_lang="en",
    )

    report = build_verification_report(
        _masked_render({"r1": (20, 30, 80, 45)}),
        manifest,
        _registry("latin.ttf", "A"),
    )
    region = report.regions[0]

    assert region.checks["glyph_coverage"]["status"] == "fail"
    assert region.checks["glyph_coverage"]["missing_codepoints"] == ["U+03A9"]
    assert "unsupported_glyphs" in region.flags
    assert report.project.overall_status == "fail"


def test_phase2_geometry_measures_overflow_and_collision_from_scribe_masks():
    instances = [
        _instance("r1", bounding_box=BBox(10, 20, 50, 30)),
        _instance("r2", bounding_box=BBox(50, 20, 50, 30)),
    ]
    manifest = TextManifest(
        asset_id="asset-1", total_regions=2, instances=instances,
        src_lang="en", targ_lang="fr",
    )

    report = build_verification_report(
        _masked_render({
            "r1": (20, 25, 70, 40),  # extends beyond r1 and overlaps r2 ink
            "r2": (60, 25, 90, 40),
        }),
        manifest,
    )
    first = report.regions[0]

    assert first.checks["geometry"]["overflow_pixels"] > 0
    assert first.checks["geometry"]["collisions"] == ["r2"]
    assert first.scores["spatial_fit"] < 100
    assert "spatial_overflow" in first.flags
    assert "spatial_collision" in first.flags
    assert report.project.component_scores["spatial_fit"] < 100


class _Reader:
    def __init__(self, results):
        self.results = results

    def readtext(self, _crop):
        return self.results


def test_phase3_reocr_records_match_and_confidence(monkeypatch):
    monkeypatch.setattr(
        verify,
        "_get_reader",
        lambda _lang: _Reader([([], "Bonjour", 0.94)]),
    )
    manifest = TextManifest(
        asset_id="asset-1", total_regions=1, instances=[_instance("r1")],
        src_lang="en", targ_lang="fr",
    )

    report = build_verification_report(
        _masked_render({"r1": (20, 30, 80, 45)}), manifest
    )
    evidence = report.regions[0].checks["target_ocr"]

    assert evidence["status"] == "pass"
    assert evidence["recognized"] == "Bonjour"
    assert evidence["confidence"] == 0.94
    assert report.regions[0].scores["content_integrity"] == 100.0
    assert report.project.component_scores["content_integrity"] == 100.0


def test_phase3_normalizes_compatibility_forms_and_punctuation(monkeypatch):
    monkeypatch.setattr(
        verify,
        "_get_reader",
        lambda _lang: _Reader([([], "Hello, 123!", 0.9)]),
    )
    manifest = TextManifest(
        asset_id="asset-1", total_regions=1,
        instances=[_instance("r1", target="Ｈｅｌｌｏ １２３")],
        src_lang="en", targ_lang="en",
    )

    report = build_verification_report(
        _masked_render({"r1": (20, 30, 80, 45)}), manifest
    )

    assert report.regions[0].checks["target_ocr"]["normalized_expected"] == "hello123"
    assert report.regions[0].scores["content_integrity"] == 100.0


def test_phase3_empty_reocr_is_critical_unreadable_output(monkeypatch):
    monkeypatch.setattr(
        verify,
        "_get_reader",
        lambda _lang: _Reader([]),
    )
    manifest = TextManifest(
        asset_id="asset-1", total_regions=1, instances=[_instance("r1")],
        src_lang="en", targ_lang="fr",
    )

    report = build_verification_report(
        _masked_render({"r1": (20, 30, 80, 45)}), manifest
    )

    assert report.regions[0].scores["content_integrity"] == 0.0
    assert "unreadable_output" in report.regions[0].flags
    assert report.project.overall_status == "fail"


def test_phase4_typography_scores_weight_size_alignment_and_contrast():
    inst = _instance(
        "r1",
        bounding_box=BBox(10, 20, 100, 40),
        characteristics=CharactText(size=20),
        style_profile=StyleProfil(
            font_family="regular.ttf", font_weight="regular", align_h="left"
        ),
    )
    manifest = TextManifest(
        asset_id="asset-1", total_regions=1, instances=[inst],
        src_lang="en", targ_lang="fr",
    )

    report = build_verification_report(
        _masked_render({"r1": (12, 28, 72, 47)}),
        manifest,
        _registry("regular.ttf", "Bonjour"),
    )
    typography = report.regions[0].checks["typography"]

    assert typography["weight_score"] == 100.0
    assert typography["size_score"] >= 95.0
    assert typography["alignment_score"] == 100.0
    assert typography["contrast_ratio"] >= 4.5
    assert report.regions[0].scores["typographic_intent"] >= 95.0


def test_phase4_reversed_relative_size_flags_hierarchy():
    instances = [
        _instance(
            "headline", bounding_box=BBox(10, 10, 80, 40),
            characteristics=CharactText(size=40),
        ),
        _instance(
            "caption", bounding_box=BBox(10, 55, 80, 30),
            characteristics=CharactText(size=20),
        ),
    ]
    manifest = TextManifest(
        asset_id="asset-1", total_regions=2, instances=instances,
        src_lang="en", targ_lang="fr",
    )

    report = build_verification_report(
        _masked_render({
            "headline": (20, 20, 70, 29),
            "caption": (20, 57, 70, 81),
        }),
        manifest,
    )

    assert all(
        region.checks["typography"]["hierarchy_score"] == 0.0
        for region in report.regions
    )
    assert all("typography_hierarchy_review" in region.flags for region in report.regions)
    assert report.project.overall_status == "review"


def test_phase4_severe_low_contrast_is_critical_and_caps_region():
    image = Image.new("RGB", (200, 100), "white")
    mask = Image.new("L", image.size, 0)
    ImageDraw.Draw(mask).rectangle((20, 30, 80, 45), fill=255)
    image.text_masks = {"r1": mask}  # white "ink" on white background
    manifest = TextManifest(
        asset_id="asset-1", total_regions=1, instances=[_instance("r1")],
        src_lang="en", targ_lang="fr",
    )

    report = build_verification_report(image, manifest)
    region = report.regions[0]

    assert region.checks["typography"]["contrast_ratio"] == 1.0
    assert "severe_low_contrast" in region.flags
    assert region.checks["critical_rules"]["status"] == "fail"
    assert region.checks["critical_rules"]["score_cap"] == 49.0
    assert report.project.overall_status == "fail"


def test_phase4_dark_outline_prevents_false_low_contrast_failure():
    image = Image.new("RGB", (200, 100), "#e6c8ae")
    mask = Image.new("L", image.size, 0)
    draw = ImageDraw.Draw(image)
    draw.rectangle((20, 30, 80, 45), fill="#ead0b8", outline="#201810", width=3)
    ImageDraw.Draw(mask).rectangle((20, 30, 80, 45), fill=255)
    image.text_masks = {"r1": mask}
    manifest = TextManifest(
        asset_id="asset-1", total_regions=1, instances=[_instance("r1")],
        src_lang="en", targ_lang="fr",
    )

    report = build_verification_report(image, manifest)
    region = report.regions[0]

    assert region.checks["typography"]["contrast_ratio"] >= 4.5
    assert "severe_low_contrast" not in region.flags
    assert region.status != "fail"


def test_phase5_ready_rollup_renormalizes_only_measured_components(monkeypatch):
    monkeypatch.setattr(
        verify,
        "_get_reader",
        lambda _lang: _Reader([([], "Bonjour", 0.98)]),
    )
    manifest = TextManifest(
        asset_id="asset-1", total_regions=1, instances=[_instance("r1")],
        src_lang="en", targ_lang="fr",
    )

    report = build_verification_report(
        _masked_render({"r1": (30, 28, 90, 47)}), manifest
    )
    region = report.regions[0]

    assert region.status == "pass"
    assert region.overall_score == 100.0
    assert report.project.overall_status == "pass"
    assert report.project.overall_score == 100.0
    assert report.project.summary.startswith("Ready:")
    assert report.project.review_order == []
    assert report.visual_flags == []
    assert report.run_metadata["pending_components"] == []
    assert report.run_metadata["component_weights"]["coverage"] == 0.20


def test_phase5_critical_cap_overrides_high_component_average(monkeypatch):
    monkeypatch.setattr(
        verify,
        "_get_reader",
        lambda _lang: _Reader([([], "AΩ", 0.99)]),
    )
    inst = _instance("r1", target="AΩ")
    inst.style_profile.font_family = "latin.ttf"
    manifest = TextManifest(
        asset_id="asset-1", total_regions=1, instances=[inst],
        src_lang="en", targ_lang="en",
    )

    report = build_verification_report(
        _masked_render({"r1": (30, 28, 90, 47)}),
        manifest,
        _registry("latin.ttf", "A"),
    )

    assert report.regions[0].overall_score == 49.0
    assert report.project.overall_score == 49.0
    assert report.project.overall_status == "fail"
    assert report.project.summary.startswith("Blocked:")


def test_phase5_review_order_and_visual_flags_are_overlay_ready():
    failed = _instance("failed")
    review = _instance("review", target=None)
    manifest = TextManifest(
        asset_id="asset-1", total_regions=2, instances=[review, failed],
        src_lang="en", targ_lang="fr",
    )
    image = Image.new("RGB", (200, 100), "white")
    mask = Image.new("L", image.size, 0)
    ImageDraw.Draw(mask).rectangle((20, 30, 80, 45), fill=255)
    image.text_masks = {"failed": mask}  # severe white-on-white contrast

    report = build_verification_report(image, manifest)

    assert report.project.review_order == ["failed", "review"]
    flags = {flag.region_id: flag for flag in report.visual_flags}
    assert flags["failed"].severity == "fail"
    assert flags["failed"].color == "#dc2626"
    assert flags["failed"].bounds == failed.bounding_box
    assert flags["review"].severity == "review"
    assert flags["review"].color == "#f59e0b"
    assert report.regions[0].region_id == "review"  # source order is preserved
    json.dumps(dataclasses.asdict(report))


@pytest.mark.parametrize(
    ("instances", "img_dim", "expected"),
    [
        ([_instance("r1", text="OPEN", bounding_box=BBox(10, 10, 100, 30))], (500, 500), "sign"),
        ([_instance("r1", text="GO", bounding_box=BBox(20, 20, 700, 140))], (1000, 500), "billboard"),
        ([
            _instance("r1", text="Festival", bounding_box=BBox(10, 10, 300, 100)),
            _instance("r2", text="Tonight", bounding_box=BBox(10, 130, 200, 40)),
            _instance("r3", text="8 PM", bounding_box=BBox(10, 190, 100, 30)),
        ], (600, 800), "poster"),
        ([
            _instance("r1", text="500 g"), _instance("r2", text="12%"),
            _instance("r3", text="$4.99"),
        ], (500, 500), "product_label"),
        ([
            _instance(f"r{i}", text=text, bounding_box=BBox(10, 10 + i * 30, 80, 20))
            for i, text in enumerate(("Save", "Cancel", "Open", "Close", "Help"))
        ], (500, 500), "ui_graphic"),
    ],
)
def test_phase6_cicerone_classifies_supported_asset_contexts(instances, img_dim, expected):
    manifest = TextManifest(
        asset_id="asset-1", total_regions=len(instances), instances=instances,
        src_lang="en", targ_lang="fr", img_dim=img_dim,
    )

    classification = classify_asset_context(manifest)

    assert classification["asset_class"] == expected
    assert classification["source"] == "cicerone_layout"
    assert classification["evidence"]["region_count"] == len(instances)
    assert manifest.asset_class == expected


def test_phase6_auto_classification_recomputes_after_manifest_edits():
    manifest = TextManifest(
        asset_id="asset-1", total_regions=1,
        instances=[_instance("sign", text="OPEN")],
        src_lang="en", targ_lang="fr", img_dim=(500, 500),
    )
    assert classify_asset_context(manifest)["asset_class"] == "sign"
    manifest.instances = [
        _instance(f"ui-{i}", text=text, bounding_box=BBox(10, 10 + i * 30, 80, 20))
        for i, text in enumerate(("Save", "Cancel", "Open", "Close", "Help"))
    ]
    manifest.total_regions = len(manifest.instances)

    refreshed = classify_asset_context(manifest)

    assert refreshed["asset_class"] == "ui_graphic"
    assert refreshed["source"] == "cicerone_layout"


@pytest.mark.parametrize(
    ("asset_class", "source", "target", "expected_fragment"),
    [
        ("sign", "OPEN", "This translated sign contains far too many words to scan quickly", "long"),
        ("billboard", "GO", "This billboard translation contains far more than eight separate words", "too long"),
        ("product_label", "Net weight 500 g", "Poids net", "quantities"),
        ("ui_graphic", "Save", "Enregistrer toutes les modifications maintenant", "expansion"),
    ],
)
def test_phase6_context_checks_explain_review_findings(
    asset_class, source, target, expected_fragment
):
    inst = _instance("r1", text=source, target=target)
    manifest = TextManifest(
        asset_id="asset-1", total_regions=1, instances=[inst],
        src_lang="en", targ_lang="fr", img_dim=(500, 500),
        asset_class=asset_class,
    )

    report = build_verification_report(
        _masked_render({"r1": (30, 28, 90, 47)}), manifest
    )
    context = report.regions[0].checks["contextual_fit"]

    assert context["status"] == "review"
    assert expected_fragment in context["explanation"]
    assert report.regions[0].scores["contextual_fit"] < 70
    assert "contextual_fit_review" in report.regions[0].flags
    assert report.project.overall_status == "review"


def test_phase6_poster_reports_changed_headline_hierarchy():
    instances = [
        _instance("headline", bounding_box=BBox(10, 10, 100, 50)),
        _instance("detail", bounding_box=BBox(10, 70, 100, 20)),
    ]
    manifest = TextManifest(
        asset_id="asset-1", total_regions=2, instances=instances,
        src_lang="en", targ_lang="fr", img_dim=(500, 500),
        asset_class="poster",
    )

    report = build_verification_report(
        _masked_render({
            "headline": (30, 20, 90, 29),
            "detail": (30, 72, 90, 89),
        }),
        manifest,
    )

    headline_context = report.regions[0].checks["contextual_fit"]
    assert headline_context["status"] == "review"
    assert "headline hierarchy changed" in headline_context["explanation"]
