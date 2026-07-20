## 🍢 cicerone pure-logic units (no OCR model required)
from unittest.mock import patch

from tofu.core.types import BBox, InstText, SceneRegion
from tofu.layers.cicerone import (
    EasyOCRBackend,
    RawDetection,
    ScriptDetector,
    _auto_probe_language,
    _compose_crop_text,
    _disambiguate_ja_zh,
    _polygon_bbox,
    _prune_hallucinations,
    _segment_vertical_bands,
    _split_tall_detections,
    build_manifest,
    guess_latin_language,
    merge_detections,
    merge_vertical_columns,
    probe_uncovered_surfaces,
    union_prefer_primary,
    expand_langset,
)


def det(x, y, w, h, text="t", conf=0.9, lang=None):
    return RawDetection(
        polygon=[(x, y), (x + w, y), (x + w, y + h), (x, y + h)],
        text=text, confidence=conf, language=lang,
    )


# -- script identification ----------------------------------------------------

class TestScriptDetector:
    sd = ScriptDetector()

    def test_latin(self):
        assert self.sd.detect_script("Main Street") == "latin"

    def test_hangul(self):
        assert self.sd.detect_script("맥주") == "hangul"

    def test_han_only_is_han(self):
        assert self.sd.detect_script("居酒屋") == "han"

    def test_any_kana_classifies_japanese(self):
        # han + kana mix must resolve japanese, not han
        assert self.sd.detect_script("居酒屋ようこそ") == "japanese"

    def test_no_script_chars(self):
        assert self.sd.detect_script("123 !?") is None
        assert self.sd.detect_script("") is None


class TestGuessLatinLanguage:
    def test_spanish_stopwords_win(self):
        assert guess_latin_language(["la calle mayor", "salida de la avenida"]) == "es"

    def test_english_never_returned(self):
        # english is the default; the guesser only reports a BEAT
        assert guess_latin_language(["the main street", "exit open"]) is None

    def test_weak_signal_returns_none(self):
        assert guess_latin_language(["zzz qqq"]) is None


# -- vertical column merge ----------------------------------------------------

class TestMergeVerticalColumns:
    def test_stacked_chars_merge_into_one_column(self):
        # three 64x64 char boxes, x-aligned, 26px gaps (< 0.8 * width)
        chars = [det(100, 100 + i * 90, 64, 64, text=c, conf=0.8)
                 for i, c in enumerate("居酒屋")]
        merged = merge_vertical_columns(chars)
        assert len(merged) == 1
        assert merged[0].text == "居酒屋"
        b = _polygon_bbox(merged[0].polygon)
        assert b.y == 100 and b.height == 2 * 90 + 64

    def test_wide_lines_never_join(self):
        # wide boxes (w > 1.6h) are text LINES, not chars
        lines = [det(100, 100 + i * 50, 300, 40) for i in range(3)]
        assert len(merge_vertical_columns(lines)) == 3

    def test_x_misaligned_chars_stay_separate(self):
        a = det(100, 100, 64, 64)
        b = det(300, 190, 64, 64)  # far right of a's column
        assert len(merge_vertical_columns([a, b])) == 2


# -- crop text composition ----------------------------------------------------

class TestComposeCropText:
    def test_vertical_crop_reads_top_to_bottom(self):
        dets = [det(10, 200, 60, 60, "주", 0.9),
                det(10, 10, 60, 60, "맥", 0.9)]
        composed = _compose_crop_text(dets)
        assert composed.text == "맥주"  # no separator for vertical

    def test_horizontal_crop_reads_left_to_right_with_spaces(self):
        dets = [det(200, 10, 80, 40, "STREET", 0.9),
                det(10, 10, 80, 40, "MAIN", 0.9)]
        assert _compose_crop_text(dets).text == "MAIN STREET"

    def test_low_confidence_noise_dropped(self):
        dets = [det(10, 10, 80, 40, "MAIN", 0.9),
                det(200, 10, 80, 40, "###", 0.05)]
        assert _compose_crop_text(dets).text == "MAIN"

    def test_empty_returns_none(self):
        assert _compose_crop_text([]) is None


# -- cross-pass merging -------------------------------------------------------

class TestMerging:
    def test_merge_keeps_higher_confidence_on_overlap(self):
        base = [det(0, 0, 100, 40, "low", 0.5)]
        extra = [det(2, 2, 100, 40, "high", 0.9)]
        out = merge_detections(base, extra)
        assert len(out) == 1 and out[0].text == "high"

    def test_union_primary_wins_regardless_of_confidence(self):
        # tuned-charset read must beat a confident wrong-charset read
        primary = [det(0, 0, 100, 40, "사", 0.6)]
        secondary = [det(2, 2, 100, 40, "4", 0.95)]
        out = union_prefer_primary(primary, secondary)
        assert len(out) == 1 and out[0].text == "사"

    def test_union_appends_non_overlapping(self):
        primary = [det(0, 0, 100, 40)]
        secondary = [det(500, 500, 100, 40, "far")]
        assert len(union_prefer_primary(primary, secondary)) == 2


# -- scene-surface probe gates ------------------------------------------------

def _inst(x, y, w, h, text, conf):
    return InstText(
        id="r1", bounding_box=BBox(x=x, y=y, width=w, height=h),
        text=text, confidence=conf,
    )


def _surface(x, y, w, h, label="panel"):
    return SceneRegion(
        bbox=BBox(x=x, y=y, width=w, height=h),
        semantic_label=label, confidence=0.8,
    )


class TestSurfaceProbeGates:
    """gates return (None, []) BEFORE any reader init, so these run
    without an OCR model."""

    def test_healthy_latin_scene_skips_probe(self):
        # >=2 confident script-bearing reads: no rescue needed
        instances = [
            _inst(10, 10, 100, 30, "MAIN STREET", 0.95),
            _inst(10, 60, 100, 30, "EXIT 25", 0.9),
        ]
        surfaces = [_surface(200, 200, 80, 80)]
        langset, dets = probe_uncovered_surfaces(
            None, surfaces, instances, EasyOCRBackend.__new__(EasyOCRBackend)
        )
        assert langset is None and dets == []

    def test_no_surfaces_skips_probe(self):
        langset, dets = probe_uncovered_surfaces(
            None, [], [], EasyOCRBackend.__new__(EasyOCRBackend)
        )
        assert langset is None and dets == []

    def test_covered_surfaces_skip_probe(self):
        # the only surface fully contains a healthy detection: nothing
        # uncovered remains, so no probe
        instances = [_inst(20, 20, 60, 20, "맥주", 0.9)]
        surfaces = [_surface(10, 10, 100, 100)]
        langset, dets = probe_uncovered_surfaces(
            None, surfaces, instances, EasyOCRBackend.__new__(EasyOCRBackend)
        )
        assert langset is None and dets == []


# -- language set expansion ---------------------------------------------------

class TestExpandLangset:
    def test_cjk_pairs_with_english_only(self):
        assert expand_langset(["ja"]) == ("ja", "en")

    def test_english_alone(self):
        assert expand_langset(["en"]) == ("en",)

    def test_zh_cn_maps_to_ch_sim(self):
        assert expand_langset(["zh-cn"]) == ("ch_sim", "en")

    def test_empty_defaults_english(self):
        assert expand_langset([]) == ("en",)


# -- evidence-breadth ranking (dense-CJK capture-perf follow-up) --------------
# a single high-confidence-but-implausible read must not outrank a
# correctly-scripted reader whose crops are merely HARDER -- measured on
# japan-street.jpeg, where Korean's hangul beat Japanese's genuinely
# low-but-real kanji reads under the old confidence-mean-only ranking.

def _fake_reader(per_langset_dets):
    """returns a stand-in for EasyOCRBackend's constructor: a function
    matching EasyOCRBackend(languages=..., gpu=...)'s signature that
    hands back a stub instance whose detect_in_regions() returns
    per_langset_dets[languages[0]] (a List[List[RawDetection]], one
    inner list per probed bbox), so _auto_probe_language/
    probe_uncovered_surfaces's REAL ranking logic runs end-to-end
    against canned per-language results, without loading a model."""
    def make(languages, gpu=False):
        fake = EasyOCRBackend.__new__(EasyOCRBackend)
        fake.languages = languages
        fake.gpu = gpu
        results = per_langset_dets.get(languages[0], [])
        fake.detect_in_regions = lambda asset, bboxes: results
        return fake
    return make


class TestEvidenceBreadthRanking:
    def test_auto_probe_language_prefers_broad_weak_over_narrow_confident(self):
        instances = [
            _inst(0, 0, 40, 40, ":::", 0.3),
            _inst(0, 100, 40, 40, ";;;", 0.3),
        ]
        engine = EasyOCRBackend.__new__(EasyOCRBackend)
        engine.languages = ("en",)
        engine.gpu = False
        fake = _fake_reader({
            # japanese: TWO real (if low-confidence) script-bearing hits
            "ja": [
                [det(0, 0, 40, 40, "居", 0.25, lang="ja")],
                [det(0, 0, 40, 40, "屋", 0.22, lang="ja")],
            ],
            # korean: ONE high-confidence hit, nothing in the 2nd region
            # -- old mean-of-top2 scoring (0.3) beat japanese's (0.235)
            "ko": [
                [det(0, 0, 40, 40, "맥", 0.6, lang="ko")],
                [],
            ],
        })
        with patch("tofu.layers.cicerone.EasyOCRBackend", side_effect=fake):
            winners = _auto_probe_language(
                None, instances, engine, probe_regions=8, min_evidence=0.2,
            )
        assert winners and winners[0].languages[0] == "ja"

    def test_probe_uncovered_surfaces_prefers_broad_weak_over_narrow_confident(self):
        surfaces = [_surface(0, 0, 40, 40), _surface(0, 100, 40, 40)]
        engine = EasyOCRBackend.__new__(EasyOCRBackend)
        engine.languages = ("en",)
        engine.gpu = False
        fake = _fake_reader({
            "ja": [
                [det(0, 0, 40, 40, "居", 0.25, lang="ja")],
                [det(0, 0, 40, 40, "屋", 0.22, lang="ja")],
            ],
            "ko": [
                [det(0, 0, 40, 40, "맥", 0.6, lang="ko")],
                [],
            ],
        })
        with patch("tofu.layers.cicerone.EasyOCRBackend", side_effect=fake):
            langset, dets = probe_uncovered_surfaces(
                None, surfaces, [], engine, min_evidence=0.2,
            )
        assert langset is not None and langset[0] == "ja"


# -- two-phase scene_filter (dense-CJK capture-perf follow-up) ---------------
# confidence from the wrong charset, or even the right charset on a hard
# crop, is not a reliable "is this real text" signal -- measured on
# japan-street's banner: 6/7 characters read correctly at confidence
# 0.015, below the filter's own floor. the strict confidence prune must
# apply AFTER language rescue has had a chance to fix the text, not
# before.

class TestTwoPhaseSceneFilter:
    def test_low_confidence_survives_when_rescue_improves_it(self):
        # a detection inside a text_cluster surface at confidence 0.05
        # -- well below the 0.30 floor even for the lenient text_cluster
        # threshold. the OLD single-phase filter drops this before any
        # rescue can run; phase A's geometric-only gate lets it through,
        # and rescue (stubbed here) brings its confidence up enough to
        # survive phase B.
        low_conf = det(10, 10, 50, 20, text=":::", conf=0.05)
        surfaces = [_surface(0, 0, 100, 100, label="text_cluster")]
        engine = EasyOCRBackend.__new__(EasyOCRBackend)
        engine.languages = ("en",)
        engine.gpu = False

        def fake_detect_in_regions(self, asset, bboxes):
            return [[det(10, 10, 50, 20, "居酒屋", 0.35, lang=self.languages[0])]]

        with patch.object(EasyOCRBackend, "detect_in_regions", fake_detect_in_regions):
            m = build_manifest(
                "fake.png", [low_conf], scene_regions=surfaces, engine=engine,
                identify_languages=True, prune_garbage=False,
            )
        assert m.total_regions == 1
        assert m.instances[0].text == "居酒屋"

    def test_low_confidence_still_pruned_when_nothing_rescues_it(self):
        # same low-confidence-in-surface detection, but identify_languages
        # disabled -- no rescue happens, so phase B must still prune it
        # exactly as the old single-phase filter would have. proves the
        # restructuring doesn't just make everything survive.
        low_conf = det(10, 10, 50, 20, text=":::", conf=0.05)
        surfaces = [_surface(0, 0, 100, 100, label="text_cluster")]
        m = build_manifest(
            "fake.png", [low_conf], scene_regions=surfaces,
            identify_languages=False, prune_garbage=False,
        )
        assert m.total_regions == 0

    def test_high_confidence_survives_without_any_surface(self):
        # unchanged behavior: a confident detection needs no surface at
        # all (the existing >=0.5-anywhere bypass, both phases).
        confident = det(500, 500, 50, 20, text="STOP", conf=0.9)
        m = build_manifest(
            "fake.png", [confident], scene_regions=[_surface(0, 0, 10, 10)],
            identify_languages=False, prune_garbage=False,
        )
        assert m.total_regions == 1


# -- ja/zh-cn disambiguation evidence gate (dense-CJK capture-perf) ----------

class TestDisambiguationEvidenceGate:
    def test_sparse_no_kana_evidence_does_not_downgrade_ja(self):
        # a single han-only "ja"-labeled instance and no kana anywhere:
        # too little text survived to trust "no kana" as meaningful --
        # measured live, an explicit correct source_lang="ja" hint still
        # got silently downgraded to zh-cn this way on a 2-3-instance
        # manifest. must NOT relabel below MIN_DISAMBIGUATION_EVIDENCE.
        inst = _inst(0, 0, 40, 40, "屋", 0.5)
        inst.detected_language = "ja"
        _disambiguate_ja_zh([inst])
        assert inst.detected_language == "ja"

    def test_sufficient_no_kana_evidence_downgrades_ja(self):
        # MIN_DISAMBIGUATION_EVIDENCE han-only "ja"-labeled instances,
        # still no kana anywhere: now there's enough real CJK text to
        # trust the absence, so the downgrade fires as intended.
        insts = [_inst(0, i * 50, 40, 40, "屋", 0.5) for i in range(2)]
        for inst in insts:
            inst.detected_language = "ja"
        _disambiguate_ja_zh(insts)
        assert all(i.detected_language == "zh-cn" for i in insts)

    def test_kana_presence_upgrades_regardless_of_count(self):
        # a SINGLE kana-bearing instance is unambiguous evidence of
        # Japanese -- the presence side of the heuristic is trusted
        # immediately, no minimum-count gate applies there.
        kana_inst = _inst(0, 0, 40, 40, "ようこそ", 0.9)
        han_inst = _inst(0, 50, 40, 40, "屋", 0.5)
        han_inst.detected_language = "zh-cn"
        _disambiguate_ja_zh([kana_inst, han_inst])
        assert han_inst.detected_language == "ja"


# -- vertical-stack re-split (dense-CJK capture-perf, china-street) ---------
# CRAFT can over-merge an entire multi-character vertical CJK sign into
# ONE box (measured: 127x603 for a 6-character sign) -- recognizing that
# whole crop as one text line produces garbage regardless of charset.

class TestSegmentVerticalBands:
    def test_square_box_still_floors_at_two_bands(self):
        # the aspect-ratio TRIGGER (is this box worth splitting at all)
        # lives in _split_tall_detections's caller, not here -- this
        # class tests the segmentation geometry in isolation. a square
        # box's aspect-implied char count rounds to 1, but the fallback
        # floors at 2 (a single "band" would just be the whole box back
        # again, which is never useful to return from a split function)
        bbox = BBox(x=0, y=0, width=100, height=100)
        bands = _segment_vertical_bands(None, bbox)
        assert len(bands) == 2
        assert all(b.height == 50 for b in bands)

    def test_tall_box_falls_back_to_equal_division_without_an_image(self):
        # asset=None -> load_rgb returns None -> pure geometric fallback
        bbox = BBox(x=10, y=20, width=50, height=300)
        bands = _segment_vertical_bands(None, bbox)
        assert len(bands) == 6  # round(300/50) == 6
        assert all(b.width == 50 for b in bands)
        assert bands[0].y == 20
        assert bands[-1].y + bands[-1].height == 320

    def test_too_short_for_even_one_band_returns_empty(self):
        bbox = BBox(x=0, y=0, width=200, height=10)
        assert _segment_vertical_bands(None, bbox) == []


class TestSplitTallDetections:
    def test_short_wide_detection_passes_through_unchanged(self):
        line = det(0, 0, 200, 30, text="MAIN STREET", conf=0.9)
        engine = EasyOCRBackend.__new__(EasyOCRBackend)
        engine.languages = ("en",)
        assert _split_tall_detections(None, engine, [line]) is None

    def test_korean_engine_excluded_outright(self):
        # a single Hangul syllable block is visually composed of 2-3
        # jamo sub-glyphs with real internal gaps -- the ink-gap
        # segmenter can't tell that apart from genuine inter-character
        # gaps, so Korean is excluded rather than relying on the
        # confidence-comparison gate alone (measured live: a correctly-
        # read character split into two DIFFERENT wrong ones, both at
        # >99% confidence, beating the original's own low confidence).
        stack = det(0, 0, 40, 240, text="1", conf=0.05)
        engine = EasyOCRBackend.__new__(EasyOCRBackend)
        engine.languages = ("ko", "en")
        assert _split_tall_detections(None, engine, [stack]) is None

    def test_tall_detection_gets_split_and_recomposed(self):
        # a 40x240 box (6:1 aspect) with garbage text, mirroring the
        # over-merged-column shape measured on china-street
        stack = det(0, 0, 40, 240, text="1", conf=0.05)
        engine = EasyOCRBackend.__new__(EasyOCRBackend)
        engine.languages = ("ch_sim",)

        def fake_detect_in_regions(self, asset, bboxes):
            chars = "美珠宝"
            return [
                [RawDetection(
                    polygon=[(b.x, b.y), (b.x + b.width, b.y),
                             (b.x + b.width, b.y + b.height), (b.x, b.y + b.height)],
                    text=chars[i % len(chars)], confidence=0.9, language="zh-cn",
                )]
                for i, b in enumerate(bboxes)
            ]

        with patch.object(EasyOCRBackend, "detect_in_regions", fake_detect_in_regions):
            split = _split_tall_detections(None, engine, [stack])
        assert split is not None
        assert len(split) >= 2
        assert all(d.confidence == 0.9 for d in split)
        assert set(d.text for d in split) <= set("美珠宝")


# -- small-image upscale (dense-CJK capture-perf) ----------------------------

class TestPrepareUpscale:
    def test_small_image_upscales(self, tmp_path):
        from PIL import Image
        p = tmp_path / "small.png"
        Image.new("RGB", (400, 300), (255, 255, 255)).save(p)
        engine = EasyOCRBackend(min_upscale_dim=850, upscale_factor=2.0)
        prepared, scale = engine._prepare(str(p))
        assert scale == (2.0, 2.0)
        assert prepared.shape[1] == 800 and prepared.shape[0] == 600

    def test_normal_sized_image_untouched(self, tmp_path):
        from PIL import Image
        p = tmp_path / "normal.png"
        Image.new("RGB", (960, 640), (255, 255, 255)).save(p)
        engine = EasyOCRBackend(min_upscale_dim=850, upscale_factor=2.0)
        prepared, scale = engine._prepare(str(p))
        assert scale == (1.0, 1.0)
        assert prepared == str(p)

    def test_oversized_image_still_downscales_not_upscales(self, tmp_path):
        from PIL import Image
        p = tmp_path / "big.png"
        Image.new("RGB", (3000, 2000), (255, 255, 255)).save(p)
        engine = EasyOCRBackend(max_dim=2560, min_upscale_dim=850)
        prepared, scale = engine._prepare(str(p))
        assert scale[0] < 1.0 and scale[1] < 1.0


# -- high-containment bypass (bbox-placement follow-up) ----------------------
# even after the two-phase scene_filter fix, a detection can be near-
# perfectly contained in a genuine panel/bordered_region surface and
# still fail every confidence bar, because neon glow/bloom caps
# EasyOCR's own confidence regardless of charset (measured on
# japan-street's banner: 6/7 characters correct at confidence 0.015-0.09,
# 95% contained in its contour region). total containment in a
# deliberately-bounded sign shape should be trusted over an unreliable
# confidence score.

class TestHighContainmentBypass:
    def test_near_total_containment_in_bordered_region_survives_low_confidence(self):
        # far below every _SCENE_CONF bar (0.30-0.50), but >85% contained
        # in a bordered_region surface
        banner = det(193, 125, 240, 40, text="banner text", conf=0.015)
        surfaces = [_surface(191, 125, 242, 39, label="bordered_region")]
        m = build_manifest(
            "fake.png", [banner], scene_regions=surfaces,
            identify_languages=False, prune_garbage=False,
        )
        assert m.total_regions == 1

    def test_partial_containment_in_bordered_region_still_pruned(self):
        # same low confidence, but only ~21% contained (matching the
        # measured japan-street MSER-surface case that motivated phase
        # A/B in the first place) -- must NOT bypass at this containment
        low_conf = det(193, 125, 240, 40, text="banner text", conf=0.015)
        surfaces = [_surface(362, 130, 66, 31, label="bordered_region")]
        m = build_manifest(
            "fake.png", [low_conf], scene_regions=surfaces,
            identify_languages=False, prune_garbage=False,
        )
        assert m.total_regions == 0

    def test_high_containment_in_text_cluster_does_not_bypass(self):
        # the bypass is scoped to panel/bordered_region (deliberately
        # bounded sign shapes) -- a text_cluster (MSER pixel-statistics
        # blob) must still go through the normal confidence bar
        low_conf = det(10, 10, 50, 20, text="x", conf=0.015)
        surfaces = [_surface(0, 0, 100, 100, label="text_cluster")]
        m = build_manifest(
            "fake.png", [low_conf], scene_regions=surfaces,
            identify_languages=False, prune_garbage=False,
        )
        assert m.total_regions == 0


# -- hallucination-pruning size floor (bbox-placement follow-up) ------------

class TestHallucinationSizeFloor:
    def test_confident_tiny_symbol_only_read_is_dropped(self):
        # measured live: a 10x10px box on a decorative emblem logo read
        # "~" at confidence 0.50 -- clears the old confidence-only bar
        # but no legible symbol exists at 10x10px
        inst = InstText(
            id="r1", bounding_box=BBox(x=0, y=0, width=10, height=10),
            text="~", confidence=0.50,
        )
        kept = _prune_hallucinations([inst])
        assert kept == []

    def test_confident_large_symbol_only_read_survives(self):
        # a real signage symbol (e.g. a degree sign, a currency symbol)
        # confidently read at a real-text scale must still survive --
        # the size floor must not become a blanket symbol ban
        inst = InstText(
            id="r1", bounding_box=BBox(x=0, y=0, width=40, height=40),
            text="°C", confidence=0.9,
        )
        kept = _prune_hallucinations([inst])
        assert len(kept) == 1

    def test_low_confidence_tiny_symbol_still_dropped_as_before(self):
        # unchanged existing behavior: low confidence alone already
        # dropped this regardless of size
        inst = InstText(
            id="r1", bounding_box=BBox(x=0, y=0, width=100, height=100),
            text="...", confidence=0.1,
        )
        kept = _prune_hallucinations([inst])
        assert kept == []

    def test_tiny_digit_bearing_read_survives_regardless_of_size(self):
        # digit-only content is always real localizable content (prices,
        # phone numbers, address plates) regardless of box size
        inst = InstText(
            id="r1", bounding_box=BBox(x=0, y=0, width=10, height=10),
            text="5", confidence=0.5,
        )
        kept = _prune_hallucinations([inst])
        assert len(kept) == 1
