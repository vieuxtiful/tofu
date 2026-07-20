## 🍢 memory (Phase 6): phash, text matching, TM store, lookup/update gating
import numpy as np
import pytest
from PIL import Image

from tofu.utils import phash as phash_mod
from tofu.utils import textmatch


class TestPHash:
    def test_identical_images_zero_distance(self):
        img = Image.new("RGB", (100, 60), (200, 50, 50))
        h1 = phash_mod.phash(img)
        h2 = phash_mod.phash(img)
        assert h1 is not None and h1 == h2
        assert phash_mod.hamming_distance(h1, h2) == 0
        assert phash_mod.visual_similarity(h1, h2) == 1.0

    def test_very_different_images_high_distance(self):
        a = np.zeros((100, 100, 3), dtype=np.uint8)
        a[:50, :] = (255, 255, 255)  # top half white, bottom black
        b = np.zeros((100, 100, 3), dtype=np.uint8)
        b[:, :50] = (255, 255, 255)  # left half white, right black
        h1 = phash_mod.phash(Image.fromarray(a))
        h2 = phash_mod.phash(Image.fromarray(b))
        sim = phash_mod.visual_similarity(h1, h2)
        assert sim is not None and sim < 0.9

    def test_resized_crop_still_matches_closely(self):
        """the whole point of a perceptual hash over a raw pixel hash --
        a re-captured sign at a different resolution should still match.
        uses rendered text (anti-aliased edges), not a hard-edged block --
        a sharp binary rectangle is a pathological worst case for DCT
        thresholding (ringing near the edge flips coefficients right at
        the median), not representative of an actual photographed sign."""
        from PIL import ImageDraw, ImageFont
        base = Image.new("RGB", (300, 200), (235, 235, 230))
        draw = ImageDraw.Draw(base)
        try:
            font = ImageFont.truetype("arial.ttf", 48)
        except Exception:
            font = ImageFont.load_default()
        draw.text((30, 70), "MAIN STREET", font=font, fill=(20, 20, 20))
        resized = base.resize((150, 100))
        h1 = phash_mod.phash(base)
        h2 = phash_mod.phash(resized)
        sim = phash_mod.visual_similarity(h1, h2)
        assert sim is not None and sim > 0.9

    def test_degenerate_crop_returns_none(self):
        tiny = Image.new("RGB", (2, 2), (0, 0, 0))
        assert phash_mod.phash(tiny) is None

    def test_hamming_distance_none_on_missing_hash(self):
        assert phash_mod.hamming_distance(None, "abc") is None
        assert phash_mod.hamming_distance("abc", None) is None


class TestTextMatch:
    def test_normalize_strips_punctuation_and_case(self):
        assert textmatch.normalize_text("Main Street!") == textmatch.normalize_text("main   street")

    def test_normalize_empty(self):
        assert textmatch.normalize_text(None) == ""
        assert textmatch.normalize_text("") == ""

    def test_fuzzy_similarity_identical(self):
        assert textmatch.fuzzy_similarity("Open 9am to 5pm", "Open 9am to 5pm") == 1.0

    def test_fuzzy_similarity_near_duplicate_above_threshold(self):
        sim = textmatch.fuzzy_similarity("Open 9am to 5pm", "Open 9am to Spm")
        assert sim >= textmatch.FUZZY_MATCH_THRESHOLD

    def test_fuzzy_similarity_unrelated_below_threshold(self):
        sim = textmatch.fuzzy_similarity("Main Street", "Exit Only")
        assert sim < textmatch.FUZZY_MATCH_THRESHOLD

    def test_fuzzy_similarity_empty_is_zero(self):
        assert textmatch.fuzzy_similarity("", "anything") == 0.0
        assert textmatch.fuzzy_similarity("anything", None) == 0.0


from tofu.core.types import BBox, CharactText, InstText, QAReport, StyleProfil, TextManifest
from tofu.layers import memory


def make_asset(w=300, h=200, color=(240, 240, 240)):
    return Image.new("RGB", (w, h), color)


def make_manifest(instances, asset_id="a", src_lang="en", targ_lang="es"):
    return TextManifest(asset_id=asset_id, total_regions=len(instances),
                        instances=instances, src_lang=src_lang, targ_lang=targ_lang)


def text_inst(id="r1", x=20, y=20, w=120, h=40, text="Main Street", target=None,
              dnt=False, style_profile=None):
    return InstText(
        id=id, bounding_box=BBox(x=x, y=y, width=w, height=h),
        text=text, target_text=target, dnt=dnt, style_profile=style_profile,
    )


class TestMemoryUpdate:
    def test_gated_below_threshold_returns_nothing(self):
        inst = text_inst(target="Calle Principal")
        manifest = make_manifest([inst])
        qa = QAReport(overall_score=0.5, per_asset_instance_score={"a": {"r1": 0.5}})
        assert memory.update(manifest, "es", make_asset(), make_asset(), qa, qa_threshold=0.8) == []

    def test_no_qa_report_returns_nothing(self):
        inst = text_inst(target="Calle Principal")
        manifest = make_manifest([inst])
        assert memory.update(manifest, "es", make_asset(), make_asset(), None) == []

    def test_dnt_and_untranslated_regions_excluded(self):
        instances = [
            text_inst(id="r1", target=None),          # untranslated
            text_inst(id="r2", x=200, dnt=True),        # DNT
        ]
        manifest = make_manifest(instances)
        qa = QAReport(overall_score=0.95,
                      per_asset_instance_score={"a": {"r1": 1.0, "r2": 1.0}})
        assert memory.update(manifest, "es", make_asset(), make_asset(), qa) == []

    def test_approved_region_produces_a_draft_record(self):
        inst = text_inst(target="Calle Principal")
        manifest = make_manifest([inst])
        qa = QAReport(overall_score=0.95, per_asset_instance_score={"a": {"r1": 0.95}})
        drafts = memory.update(manifest, "es", make_asset(), make_asset(), qa)
        assert len(drafts) == 1
        d = drafts[0]
        assert d["source_text"] == "Main Street"
        assert d["target_text"] == "Calle Principal"
        assert d["normalized_text"] == textmatch.normalize_text("Main Street")
        assert d["qa_score"] == 0.95
        assert d["target_lang"] == "es"
        assert d["thumb_crop"] is not None  # cropped from source_asset
        assert d["phash"] is not None

    def test_no_source_asset_still_produces_record_without_phash(self):
        inst = text_inst(target="Calle Principal")
        manifest = make_manifest([inst])
        qa = QAReport(overall_score=0.95, per_asset_instance_score={"a": {"r1": 0.95}})
        drafts = memory.update(manifest, "es", make_asset(), None, qa)
        assert len(drafts) == 1
        assert drafts[0]["phash"] is None
        assert drafts[0]["thumb_crop"] is None


class TestMemoryLookup:
    def test_exact_match(self):
        inst = text_inst(text="Main Street")
        manifest = make_manifest([inst])
        candidates = [{
            "id": 1, "asset_id": "prev-asset", "source_text": "Main Street",
            "normalized_text": textmatch.normalize_text("Main Street"),
            "target_text": "Calle Principal", "phash": None,
            "style_fingerprint": "regular|upright|auto",
        }]
        matches = memory.lookup(manifest, None, "es", candidates)
        assert matches["r1"]["target_text"] == "Calle Principal"
        assert matches["r1"]["method"] == "exact"
        assert matches["r1"]["score"] == 1.0
        assert matches["r1"]["source_asset_id"] == "prev-asset"

    def test_fuzzy_match_on_ocr_noise(self):
        inst = text_inst(text="Open 9am to Spm")  # OCR misread "5" as "S"
        manifest = make_manifest([inst])
        candidates = [{
            "id": 1, "asset_id": "prev-asset", "source_text": "Open 9am to 5pm",
            "normalized_text": textmatch.normalize_text("Open 9am to 5pm"),
            "target_text": "Abierto de 9am a 5pm", "phash": None,
            "style_fingerprint": "regular|upright|auto",
        }]
        matches = memory.lookup(manifest, None, "es", candidates)
        assert matches["r1"]["method"] == "fuzzy"
        assert matches["r1"]["target_text"] == "Abierto de 9am a 5pm"

    def test_unrelated_text_no_match(self):
        inst = text_inst(text="Exit Only")
        manifest = make_manifest([inst])
        candidates = [{
            "id": 1, "asset_id": "prev-asset", "source_text": "Main Street",
            "normalized_text": textmatch.normalize_text("Main Street"),
            "target_text": "Calle Principal", "phash": None,
            "style_fingerprint": "regular|upright|auto",
        }]
        assert "r1" not in memory.lookup(manifest, None, "es", candidates)

    def test_visual_match_when_text_differs_but_image_is_the_same_sign(self):
        from PIL import ImageDraw, ImageFont
        base = Image.new("RGB", (300, 200), (235, 235, 230))
        draw = ImageDraw.Draw(base)
        try:
            font = ImageFont.truetype("arial.ttf", 40)
        except Exception:
            font = ImageFont.load_default()
        draw.text((30, 70), "MAIN STREET", font=font, fill=(20, 20, 20))
        inst = text_inst(text="", x=20, y=60, w=260, h=70)  # OCR found nothing this time
        manifest = make_manifest([inst])
        inst_hash = phash_mod.phash(base.crop((20, 60, 280, 130)))
        candidates = [{
            "id": 1, "asset_id": "prev-asset", "source_text": "MAIN STREET",
            "normalized_text": textmatch.normalize_text("MAIN STREET"),
            "target_text": "CALLE PRINCIPAL", "phash": inst_hash,
            "style_fingerprint": "regular|upright|auto",
        }]
        matches = memory.lookup(manifest, base, "es", candidates)
        # OCR found nothing this pass (inst.text == "") -- exact/fuzzy have
        # no text to work with and fall through to the visual tier, which
        # still recognizes the same physical sign from its crop alone
        assert matches["r1"]["method"] == "visual"
        assert matches["r1"]["target_text"] == "CALLE PRINCIPAL"

    def test_style_fingerprint_tiebreaks_equal_fuzzy_scores(self):
        inst = text_inst(text="Main Street",
                         style_profile=StyleProfil(font_weight="bold"))
        manifest = make_manifest([inst])
        candidates = [
            {"id": 1, "asset_id": "a1", "source_text": "Main Street",
             "normalized_text": textmatch.normalize_text("Main Street"),
             "target_text": "wrong style match", "phash": None,
             "style_fingerprint": "regular|upright|auto"},
            {"id": 2, "asset_id": "a2", "source_text": "Main Street",
             "normalized_text": textmatch.normalize_text("Main Street"),
             "target_text": "right style match", "phash": None,
             "style_fingerprint": "bold|upright|auto"},
        ]
        matches = memory.lookup(manifest, None, "es", candidates)
        assert matches["r1"]["target_text"] == "right style match"

    def test_empty_candidates_no_matches(self):
        inst = text_inst(text="Main Street")
        manifest = make_manifest([inst])
        assert memory.lookup(manifest, None, "es", []) == {}

    def test_dnt_region_never_matched(self):
        inst = text_inst(text="Main Street", dnt=True)
        manifest = make_manifest([inst])
        candidates = [{
            "id": 1, "asset_id": "prev-asset", "source_text": "Main Street",
            "normalized_text": textmatch.normalize_text("Main Street"),
            "target_text": "Calle Principal", "phash": None,
            "style_fingerprint": "regular|upright|auto",
        }]
        assert memory.lookup(manifest, None, "es", candidates) == {}
