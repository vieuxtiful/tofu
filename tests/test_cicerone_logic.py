## 🍢 cicerone pure-logic units (no OCR model required)
from pathlib import Path
from unittest.mock import patch

import pytest

from tofu.core.types import BBox, InstText, SceneRegion
from tofu.layers.cicerone import (
    EasyOCRBackend,
    RawDetection,
    ScriptDetector,
    _auto_probe_language,
    _compose_crop_text,
    _disambiguate_ja_zh,
    _has_ink_support,
    _polygon_bbox,
    _prune_hallucinations,
    _segment_vertical_bands,
    _split_tall_detections,
    build_manifest,
    guess_latin_language,
    label_latin_languages,
    merge_baseline_runs,
    merge_detections,
    merge_vertical_columns,
    iter_multipass,
    run_multipass,
    probe_uncovered_surfaces,
    tag_detection_pass,
    union_prefer_primary,
    expand_langset,
    zoom_detect,
    _dedup_zoom_detections,
    rescue_clipped_edge_glyphs,
)


def det(x, y, w, h, text="t", conf=0.9, lang=None):
    return RawDetection(
        polygon=[(x, y), (x + w, y), (x + w, y + h), (x, y + h)],
        text=text, confidence=conf, language=lang,
    )


def test_canonical_multipass_iterator_drives_stream_and_sync_paths():
    backend = object.__new__(EasyOCRBackend)
    backend.languages = ("en",)
    calls = []

    def detect(_asset, *, text_threshold, low_text):
        calls.append((text_threshold, low_text))
        n = len(calls)
        return [det(n * 20, 0, 10, 10, text=f"p{n}")]

    backend.detect = detect
    emissions = list(iter_multipass(backend, "asset"))

    assert [item[0] for item in emissions] == [1, 1, 2, 2, 3, 3]
    assert [item[3] is None for item in emissions] == [
        True, False, True, False, True, False
    ]
    assert [d.text for d in emissions[-1][3]] == ["p1", "p2", "p3"]

    calls.clear()
    final = run_multipass(backend, "asset")
    assert [d.text for d in final] == ["p1", "p2", "p3"]


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

    def test_single_short_stopword_match_returns_none(self):
        # regression guard: china-street's "ET" (a random OCR fragment)
        # matching French "et" must not, on its own, flip the whole
        # scene to French -- a lone 2-letter coincidence is not
        # independent evidence, however cleanly it happens to match.
        assert guess_latin_language(["ET"]) is None
        assert guess_latin_language(["OPTICAL", "3F", "ET"]) is None

    def test_two_independent_hits_still_win(self):
        assert guess_latin_language(["et", "avec"]) == "fr"

    def test_unaccented_french_street_plate_still_wins(self):
        # EasyOCR's English reader may strip É, but lexical evidence remains.
        assert guess_latin_language(["AVENUE", "de la REPUBLIQUE"]) == "fr"

    def test_marginal_read_with_self_corroborating_evidence_can_vote(self):
        instances = [
            InstText(
                id="r1", bounding_box=BBox(x=0, y=0, width=100, height=20),
                text="AVENUE", confidence=0.99, detected_language="en",
            ),
            InstText(
                id="r2", bounding_box=BBox(x=0, y=30, width=200, height=20),
                text="de Io Republique", confidence=0.492,
                detected_language="en",
            ),
        ]

        assert label_latin_languages(instances) == "fr"
        assert [inst.detected_language for inst in instances] == ["fr", "fr"]


class TestDeclaredSourceLanguageMargin:
    """A declared source is evidence; a competing guess must be DECISIVE.

    Measured on a multilingual medical-device label declared en-US, whose
    product name repeats in five languages: the French/Italian/Spanish lines
    contribute de/del/du and scored es=14 against en=10, which was enough to
    stamp "es" on all 41 latin regions -- "Boston" and "300 Commercial Street"
    included.
    """

    ENLABEL = [
        "enLabel Total Knee Replacement System", "Make Complex Packaging",
        "enLabel totale dU systeme de remplacement du genou",
        "enLabel Insgesamt Knie Ersatz",
        "enLabel soslituzione totale del ginocchio",
        "enLabel sistema de reemplazo Total de rodilla",
        "300 Commercial Street", "Boston", "Simple", "Compliant",
        "Lot", "No.: J5322901", "Serial No.: EN1OOO", "60 of 100",
    ]

    def test_incidental_foreign_tokens_cannot_flip_a_declared_asset(self):
        assert guess_latin_language(self.ENLABEL) == "es"      # undeclared: bare win
        assert guess_latin_language(self.ENLABEL, "en-US") is None

    def test_a_genuinely_foreign_asset_still_resolves(self):
        # the declared language has no evidence at all here, so the guess is
        # decisive by any margin -- a French plate in an en project is still French
        assert guess_latin_language(["AVENUE", "de la REPUBLIQUE"], "en-GB") == "fr"

    def test_non_latin_declaration_does_not_protect(self):
        # a CJK source says nothing about which latin language a stray
        # latin region is in, so the ordinary heuristic applies
        assert guess_latin_language(["la calle mayor", "salida de la avenida"],
                                    "ja-JP") == "es"

    def test_declaration_is_stamped_verbatim_when_it_stands(self):
        instances = [
            InstText(id="r1", bounding_box=BBox(x=0, y=0, width=200, height=20),
                     text=text, confidence=0.95)
            for text in self.ENLABEL
        ]
        # "en-US", never "en" -- every catalog lookup resolves a bare "en" to en-GB
        assert label_latin_languages(instances, "en-US") == "en-US"
        assert {i.detected_language for i in instances} == {"en-US"}


class TestLocalizableSymbols:
    def test_confident_ampersand_survives_the_symbol_junk_floor(self):
        # "Simple & Compliant": EasyOCR reads the "&" at 0.996 in a 12x16 box
        # (192px2, under MIN_SYMBOL_JUNK_AREA), and it was being deleted --
        # dropping a word that becomes "et"/"y"/"und" in the target.
        instances = [
            InstText(id="r1", bounding_box=BBox(x=126, y=188, width=12, height=16),
                     text="&", confidence=0.996),
        ]
        assert [i.text for i in _prune_hallucinations(instances)] == ["&"]

    def test_decorative_marks_are_still_junk(self):
        instances = [
            InstText(id="r1", bounding_box=BBox(x=0, y=0, width=20, height=22),
                     text="‥", confidence=0.96),
            InstText(id="r2", bounding_box=BBox(x=0, y=0, width=10, height=10),
                     text="~", confidence=0.50),
        ]
        assert _prune_hallucinations(instances) == []

    def test_a_low_confidence_symbol_is_still_junk(self):
        # the exemption is for small CONFIDENT punctuation only
        instances = [
            InstText(id="r1", bounding_box=BBox(x=0, y=0, width=12, height=16),
                     text="&", confidence=0.30),
        ]
        assert _prune_hallucinations(instances) == []


class TestDetectionPassProvenance:
    class FakeBackend:
        languages = ("en",)

    def test_merge_keeps_winner_and_losing_pass_evidence(self):
        first = det(0, 0, 100, 20, "REPUBLIQUE", 0.72)
        second = det(0, 0, 100, 20, "RÉPUBLIQUE", 0.91)
        tag_detection_pass(
            [first], engine=self.FakeBackend(), pass_number=1,
            text_threshold=0.7, low_text=0.4,
        )
        tag_detection_pass(
            [second], engine=self.FakeBackend(), pass_number=2,
            text_threshold=0.5, low_text=0.3,
        )

        merged = merge_detections([first], [second])

        assert merged[0].text == "RÉPUBLIQUE"
        assert [(p["pass"], p["selected"]) for p in merged[0].provenance] == [
            (1, False), (2, True),
        ]

    def test_manifest_carries_raw_provenance_into_history(self):
        raw = det(0, 0, 100, 20, "AVENUE", 0.9)
        tag_detection_pass(
            [raw], engine=self.FakeBackend(), pass_number=1,
            text_threshold=0.7, low_text=0.4,
        )

        manifest = build_manifest(
            "fixture.png", [raw], identify_languages=False,
            prune_garbage=False,
        )

        history = manifest.instances[0].recognition_history
        assert history and history[0]["stage"] == "detection_pass"
        assert history[0]["candidate_text"] == "AVENUE"


class TestBaselineAwareReadingOrder:
    def test_words_on_one_visual_baseline_sort_left_to_right(self):
        # rue-vieux: MURS begins three px higher than VIEUX despite both
        # words sharing the same sign line. Raw top-edge ordering produced
        # "Rue des MURS VIEUX"; line grouping must preserve human reading.
        manifest = build_manifest(
            "fixture.png",
            [
                det(228, 109, 235, 71, "Rue des"),
                det(281, 206, 175, 72, "MURS"),
                det(89, 209, 176, 69, "VIEUX"),
            ],
            identify_languages=False, prune_garbage=False,
        )
        assert [inst.text for inst in manifest.instances] == ["Rue des", "VIEUX", "MURS"]
        assert [inst.reading_order for inst in manifest.instances] == [0, 1, 2]


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

    def test_confident_latin_words_are_lines_not_a_column(self):
        # regression guard for russian-billboard: the slogan sets 'Za' at the
        # head of three DIFFERENT lines, roughly in one x slot. Every geometric
        # gate passes -- the boxes are char-like, x-aligned and closely spaced
        # -- so two of them merged into a single 47x77 "column" that swallowed
        # one Za entirely. Confident multi-character latin reads are words.
        za = [det(347, 109, 35, 26, text="Za", conf=1.0),
              det(361, 153, 37, 29, text="Za", conf=1.0)]
        assert len(merge_vertical_columns(za)) == 2

    def test_single_latin_char_is_still_a_cjk_misread(self):
        # the guard must not close the door the merge exists to open: ONE
        # confident latin character is the classic shape of a misread kanji,
        # and those fragments still have to be able to form a column.
        frags = [det(100, 100, 30, 32, text="H", conf=0.9),
                 det(100, 134, 30, 32, text="E", conf=0.9)]
        assert len(merge_vertical_columns(frags)) == 1

    def test_low_confidence_latin_still_merges(self):
        # per-character CJK reads come back as unreliable latin all the time;
        # only a read the recognizer is SURE about counts as a word.
        frags = [det(100, 100, 30, 32, text="ab", conf=0.2),
                 det(100, 134, 30, 32, text="cd", conf=0.2)]
        assert len(merge_vertical_columns(frags)) == 1

    def test_duplicate_tail_fragment_not_concatenated(self):
        # regression guard for china-street's 茂昌眼镜公司镜司 bug: a
        # stray leftover fragment covering the same physical tail
        # glyphs as an already-complete read must not get concatenated
        # onto it as if it were the next characters in sequence.
        complete = det(0, 0, 20, 100, text="茂昌眼镜公司", conf=0.9)
        duplicate_tail = det(0, 80, 20, 30, text="公司", conf=0.6)  # overlaps complete's own tail
        merged = merge_vertical_columns([complete, duplicate_tail])
        assert len(merged) == 1
        assert merged[0].text == "茂昌眼镜公司"

    def test_genuine_next_characters_still_concatenate(self):
        # regression guard against over-tightening: two DIFFERENT,
        # non-overlapping fragments of a genuinely longer sign must
        # still compose normally
        top = det(0, 0, 20, 60, text="劇場", conf=0.9)
        bottom = det(0, 60, 20, 60, text="通り", conf=0.9)
        merged = merge_vertical_columns([top, bottom])
        assert len(merged) == 1
        assert merged[0].text == "劇場通り"


class TestMergeVerticalColumnsColorGate:
    """regression guard for japan-street's バンダイ/焼肉 bug: two
    geometrically column-shaped fragments that pass every existing
    check (x-aligned, comparable width, small gap) but belong to two
    visibly DIFFERENT signs must not be merged when an asset is given
    to sample their actual colors."""

    def test_color_mismatch_blocks_merge_when_asset_given(self):
        import numpy as np
        img = np.zeros((200, 200, 3), dtype=np.uint8)
        img[0:64, 0:64] = [255, 255, 255]
        img[70:134, 0:64] = [0, 0, 0]
        top = det(0, 0, 64, 64, text="A", conf=0.9)
        bottom = det(0, 70, 64, 64, text="B", conf=0.9)
        with patch("tofu.layers.scene._load_rgb", return_value=img):
            merged = merge_vertical_columns([top, bottom], asset="fake")
        assert len(merged) == 2

    def test_similar_color_still_merges_with_asset(self):
        import numpy as np
        img = np.full((200, 200, 3), 128, dtype=np.uint8)
        top = det(0, 0, 64, 64, text="A", conf=0.9)
        bottom = det(0, 70, 64, 64, text="B", conf=0.9)
        with patch("tofu.layers.scene._load_rgb", return_value=img):
            merged = merge_vertical_columns([top, bottom], asset="fake")
        assert len(merged) == 1
        assert merged[0].text == "AB"

    def test_no_asset_preserves_geometry_only_behavior(self):
        # fail-open: without an asset the color gate must never change
        # the pre-existing geometry-only merge behavior
        top = det(0, 0, 64, 64, text="A", conf=0.9)
        bottom = det(0, 70, 64, 64, text="B", conf=0.9)
        merged = merge_vertical_columns([top, bottom])
        assert len(merged) == 1
        assert merged[0].text == "AB"


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


# -- whole-over-fragments at the zoom union (keep_the_loaf) ------------------
# the zoom pass exists to break a coarse box that spans several signs into
# one box per sign, and the fine boxes evict whatever contains them.  On a
# poster that is backwards: the coarse pass read a whole LINE correctly and
# the zoom pass hands back one word of it.  These guard the discriminator
# between the two -- whether the coarse read contains the fine ones as text.

class TestKeepTheLoaf:
    def test_a_correct_line_is_not_evicted_by_its_own_words(self):
        # la-bastille: the coarse pass reads the display line at 0.529 and
        # the zoom pass shatters it into two pieces, either of which was
        # enough to evict it.  Boxes are the ones actually measured.
        fine = [det(64, 12, 132, 63, "IBas", 0.392),
                det(270, 16, 65, 57, "Le", 0.342)]
        coarse = [det(26, 11, 309, 63, "la Bastille", 0.529)]
        out = union_prefer_primary(fine, coarse, keep_the_loaf=True)
        assert [d.text for d in out] == ["la Bastille"]

    def test_off_by_default(self):
        fine = [det(64, 12, 132, 63, "IBas", 0.392)]
        coarse = [det(26, 11, 309, 63, "la Bastille", 0.529)]
        assert [d.text for d in union_prefer_primary(fine, coarse)] == ["IBas"]

    def test_a_merely_overlapping_word_is_not_a_crumb(self):
        # decolonisons' 'esclavagistes' sits 0.754 inside the line it ends,
        # short of the containment bar.  It is not evidence that the line
        # spans two signs, and it is not claimed as part of it either --
        # both stand, and the line is the one that matches ground truth.
        fine = [det(260, 513, 124, 25, "esclavagistes", 1.0),
                det(254, 520, 4, 11, "1", 0.026)]
        coarse = [det(83, 515, 300, 19, "crimes coloniaux et esclavagistes", 0.919)]
        out = union_prefer_primary(fine, coarse, keep_the_loaf=True)
        assert sorted(d.text for d in out) == [
            "crimes coloniaux et esclavagistes", "esclavagistes",
        ]

    def test_a_coarse_box_spanning_several_signs_still_loses(self):
        # china-street: the coarse read of a six-character vertical column is
        # garbage and the per-character zoom reads are the real text.  None
        # of them is a span of it, so the zoom pass keeps its authority.
        fine = [det(408, 559, 64, 61, "娘", 0.689),
                det(407, 501, 63, 58, "夫", 0.53)]
        coarse = [det(410, 504, 61, 232, "矗:", 0.014)]
        out = union_prefer_primary(fine, coarse, keep_the_loaf=True)
        assert sorted(d.text for d in out) == ["夫", "娘"]

    def test_junk_fragments_neither_qualify_nor_veto(self):
        # decolonisons again: the zoom pass emits 'n' at 0.019 and 'sue' at
        # 0.002 across the line the coarse pass read whole at 0.902.  Reads
        # that weak are not evidence that the coarse box spans two signs.
        fine = [det(263, 508, 52, 5, "n", 0.019), det(320, 508, 60, 6, "sue", 0.002)]
        coarse = [det(144, 498, 234, 16, "mémoire des luttes contre", 0.902)]
        out = union_prefer_primary(fine, coarse, keep_the_loaf=True)
        assert [d.text for d in out] == ["mémoire des luttes contre"]

    def test_a_corroborated_loaf_carries_its_crumbs_confidence(self):
        # russian-billboard-2: the whole slogan line reads at 0.454 while the
        # fragments it replaces read at 0.961.  Left at its raw confidence
        # the rescued line is deleted by the scene filter's floor.
        fine = [det(107, 84, 57, 21, "IECTE", 0.961)]
        coarse = [det(78, 74, 220, 35, "BMЕСTЕ С РОССИЕЙ!", 0.454)]
        out = union_prefer_primary(fine, coarse, keep_the_loaf=True)
        assert len(out) == 1
        assert out[0].confidence == pytest.approx(0.961)
        assert out[0].provenance[-1]["rule"] == "keep_the_loaf"
        assert out[0].provenance[-1]["raw_confidence"] == pytest.approx(0.454)

    def test_the_same_text_read_twice_leaves_the_region_to_the_tighter_box(self):
        # china-street's '华 联店' is found by both passes.  The fine box is
        # the tighter one and both reads clear the scene filter, so handing
        # the region back to the coarse pass would only cost IoU.
        fine = [det(356, 242, 125, 41, "华 联店", 0.270)]
        coarse = [det(354, 240, 130, 44, "华 联店", 0.202)]
        out = union_prefer_primary(fine, coarse, keep_the_loaf=True)
        assert out[0].confidence == pytest.approx(0.270)

    def test_a_duplicate_below_the_filters_floor_is_rescued_by_the_coarse_read(self):
        # la-bastille's '7789': the fine read at 0.202 is deleted outright by
        # the scene filter, so deferring to it loses the region entirely.
        fine = [det(334, 76, 85, 64, "7789", 0.202)]
        coarse = [det(332, 72, 90, 74, "7789", 0.772)]
        out = union_prefer_primary(fine, coarse, keep_the_loaf=True)
        assert len(out) == 1
        assert out[0].confidence == pytest.approx(0.772)

    def test_a_cyrillic_fragment_matches_a_half_latin_line_read(self):
        # the span test compares pared skeletons, or a Cyrillic crumb and a
        # part-latin read of the same pixels would look unrelated.
        fine = [det(161, 59, 43, 25, "ЦЕЕ", 1.0)]
        coarse = [det(80, 55, 126, 34, "В БУДУЩЕЕ", 0.928)]
        out = union_prefer_primary(fine, coarse, keep_the_loaf=True)
        assert [d.text for d in out] == ["В БУДУЩЕЕ"]


# -- horizontal line assembly (merge_baseline_runs) -------------------------
# the mirror of merge_vertical_columns. CRAFT links horizontally, but
# width_ths was tightened 0.5 -> 0.3 to stop adjacent SIGNS chaining, which
# leaves the words of one poster line as separate detections.

class _LineStubEngine:
    """Minimal EasyOCRBackend stand-in: one canned re-read per call."""

    def __init__(self, languages=("fr", "en"), reread=None):
        self.languages = tuple(languages)
        self.reread = reread
        self.regions_seen = []

    def detect_in_regions(self, asset, boxes):
        self.regions_seen.extend(boxes)
        if self.reread is None:
            return [[] for _ in boxes]
        text, conf = self.reread
        return [[det(b.x, b.y, b.width, b.height, text, conf)] for b in boxes]


class TestMergeBaselineRuns:
    def test_words_on_one_baseline_become_a_line_and_are_re_read(self):
        # decolonisons' credit line, measured: 'Texte' + 'Naïké Desquesnes'
        # re-read whole as 'Texte Naïké Desquesnes' at 0.907.
        parts = [det(44, 584, 34, 11, "Texte", 1.0),
                 det(86, 582, 124, 15, "Naïké Desquesnes", 0.875)]
        engine = _LineStubEngine(reread=("Texte Naïké Desquesnes", 0.907))
        out = merge_baseline_runs(None, engine, parts)
        assert out is not None
        assert [d.text for d in out] == ["Texte Naïké Desquesnes"]
        assert _polygon_bbox(out[0].polygon).width == 166

    def test_cjk_primaries_are_skipped_entirely(self):
        # no word spaces to reassemble, and chaining adjacent signs is the
        # failure width_ths was tightened to prevent.
        parts = [det(100, 100, 40, 40, "美", 0.9), det(150, 100, 40, 40, "珠", 0.9)]
        for lang in ("ja", "ch_sim", "ch_tra", "ko"):
            engine = _LineStubEngine(languages=(lang, "en"), reread=("美珠", 0.99))
            assert merge_baseline_runs(None, engine, parts) is None

    def test_different_baselines_do_not_join(self):
        # avenue-de-la-republique: 'AVENUE' over 'de la RÉPUBLIQUE' are two
        # LINES, and merging them halved recall when add_margin was raised.
        parts = [det(150, 122, 155, 51, "AVENUE", 1.0),
                 det(120, 171, 214, 57, "de la RÉPUBLIQUE", 0.854)]
        engine = _LineStubEngine(reread=("AVENUE de la RÉPUBLIQUE", 0.99))
        assert merge_baseline_runs(None, engine, parts) is None

    def test_overlapping_boxes_are_a_duplicate_not_a_line(self):
        parts = [det(30, 51, 54, 55, "D", 1.0),
                 det(45, 43, 410, 63, "Décolonisons", 0.989)]
        engine = _LineStubEngine(reread=("D Décolonisons", 0.99))
        assert merge_baseline_runs(None, engine, parts) is None

    def test_a_re_read_weaker_than_its_worst_part_is_refused(self):
        parts = [det(0, 0, 100, 40, "Pour une", 0.9),
                 det(110, 0, 100, 40, "les", 0.8)]
        engine = _LineStubEngine(reread=("Pour une les", 0.7))
        assert merge_baseline_runs(None, engine, parts) is None

    def test_a_re_read_the_scene_filter_would_delete_is_refused(self):
        # la-bastille: 'la Bastille' + 'ETLA' re-read as 'la Babtille BT la'
        # at 0.433, which clears its weakest raw member (0.387) and is then
        # deleted downstream, costing both regions.
        parts = [det(26, 11, 309, 63, "la Bastille", 0.529),
                 det(346, 26, 105, 43, "ETLA", 0.387)]
        engine = _LineStubEngine(reread=("la Babtille BT la", 0.433))
        assert merge_baseline_runs(None, engine, parts) is None

    def test_a_re_read_that_lost_a_word_is_refused(self):
        parts = [det(0, 0, 100, 40, "Pour une", 0.9),
                 det(110, 0, 100, 40, "esclavagistes", 0.9)]
        engine = _LineStubEngine(reread=("Pour une", 0.99))
        assert merge_baseline_runs(None, engine, parts) is None

    def test_a_column_gutter_is_not_a_word_space(self):
        # serif-vs-sans sets the same specimen twice, side by side. Both
        # SANS-NOM boxes sit on one baseline with a 52px gutter against a
        # 76px height (0.68); merging them cost recall 0.833 -> 0.667.
        parts = [det(82, 188, 356, 76, "SANS-NOM", 1.0),
                 det(490, 188, 364, 76, "SANS-NOM", 0.867)]
        engine = _LineStubEngine(reread=("SANS-NOM SANS-NOM", 0.972))
        assert merge_baseline_runs(None, engine, parts) is None

    def test_words_of_one_line_still_join_across_a_normal_space(self):
        # the same fixture's right-hand column, where 'La' and 'rue' are a
        # genuine word pair at 0.21 of box height.
        parts = [det(502, 88, 74, 58, "La", 1.0), det(588, 94, 96, 48, "rue", 1.0)]
        engine = _LineStubEngine(reread=("La rue", 1.0))
        out = merge_baseline_runs(None, engine, parts)
        assert out is not None and [d.text for d in out] == ["La rue"]

    def test_a_lone_detection_is_left_alone(self):
        engine = _LineStubEngine(reread=("anything", 0.99))
        assert merge_baseline_runs(None, engine, [det(0, 0, 100, 40, "solo", 0.9)]) is None


# -- zoom-pass internal dedup (repeat-detect duplicate follow-up) ------------
# overlapping scene surfaces (e.g. an MSER text_cluster and a contour-rescue
# bordered_region both covering the same sign) each get re-detected
# independently in zoom_detect; without an internal dedup the same text is
# returned twice, and union_prefer_primary never catches it since it only
# checks its secondary list against primary, never primary against itself.

class TestZoomDetectDedup:
    def test_overlapping_surfaces_collapse_to_one_detection(self):
        import numpy as np

        img = np.zeros((100, 100, 3), dtype=np.uint8)
        surfaces = [
            SceneRegion(bbox=BBox(x=10, y=10, width=40, height=20),
                        semantic_label="text_cluster", confidence=0.8),
            SceneRegion(bbox=BBox(x=8, y=8, width=44, height=24),
                        semantic_label="bordered_region", confidence=0.8),
        ]
        engine = EasyOCRBackend.__new__(EasyOCRBackend)
        engine.languages = ("en",)
        engine.gpu = False
        # same canned read on the scaled crop regardless of which surface
        # produced it -- mapped back, both land on nearly the same box
        engine.detect = lambda crop, **kw: [det(20, 20, 40, 20, "OPTICAL", 0.9)]

        with patch("tofu.utils.imaging.load_rgb", return_value=img):
            fine = zoom_detect(engine, None, surfaces)

        assert len(fine) == 1
        assert fine[0].text == "OPTICAL"

    def test_overlapping_surfaces_with_different_text_both_kept(self):
        # regression guard: bbox overlap ALONE must not collapse two
        # detections -- distinct characters recovered from different
        # overlapping candidate surfaces are a normal, expected outcome
        # (measured live: naively reusing bbox-only NMS here dropped a
        # genuine Korean-text detection on gemini-street, regressing
        # recall 0.111->0.056). only text that ALSO closely matches
        # should be treated as the same duplicated read.
        import numpy as np

        img = np.zeros((100, 100, 3), dtype=np.uint8)
        surfaces = [
            SceneRegion(bbox=BBox(x=10, y=10, width=40, height=20),
                        semantic_label="text_cluster", confidence=0.8),
            SceneRegion(bbox=BBox(x=8, y=8, width=44, height=24),
                        semantic_label="bordered_region", confidence=0.8),
        ]
        engine = EasyOCRBackend.__new__(EasyOCRBackend)
        engine.languages = ("en",)
        engine.gpu = False
        calls = {"n": 0}
        def fake_detect(crop, **kw):
            calls["n"] += 1
            text = "ABCDE" if calls["n"] == 1 else "저녁"
            return [det(20, 20, 40, 20, text, 0.9)]
        engine.detect = fake_detect

        with patch("tofu.utils.imaging.load_rgb", return_value=img):
            fine = zoom_detect(engine, None, surfaces)

        assert len(fine) == 2
        assert {d.text for d in fine} == {"ABCDE", "저녁"}

    def test_a_letter_sized_piece_of_a_word_is_not_a_second_detection(self):
        # measured on the textured-wall fixture: the scene pre-pass proposed
        # a wide band covering the whole sign AND two small surfaces sitting
        # inside the word on it, so zoom read "BAKERY" once whole and again
        # as the pieces "B" and "RI".  Those pieces overlap the whole read
        # almost totally but score far below the 0.6 similarity bar against
        # it, so the overlap+similarity rule above kept all three and
        # union_prefer_primary promoted them over the coarse pass --
        # precision 0.5 with two false positives, on an asset whose recall
        # was already perfect.  Boxes here are the ones actually measured.
        fine = [
            det(275, 147, 63, 66, "B", 0.414),
            det(278, 142, 313, 73, "BAKERY", 1.0),
            det(473, 144, 80, 69, "RI", 0.938),
            det(355, 281, 158, 39, "est_ 1962", 0.66),
        ]
        kept = _dedup_zoom_detections(fine)
        assert [d.text for d in kept] == ["BAKERY", "est_ 1962"]

    def test_equal_length_reads_never_swallow_each_other(self):
        # the containment rule compares text LENGTH strictly, which is what
        # stops it eliminating both members of a pair: two boxes cannot each
        # be the longer one.  Without that, a fully-overlapping pair of
        # equal-length reads would annihilate and the region would vanish
        # entirely rather than merely lose a duplicate.
        fine = [det(100, 100, 50, 50, "AB", 0.5), det(100, 100, 50, 50, "CD", 0.9)]
        assert len(_dedup_zoom_detections(fine)) == 2

    def test_a_neighbour_clipping_the_edge_of_a_longer_read_survives(self):
        # a short read that merely OVERLAPS a longer one is a neighbour, not
        # a piece of it -- only near-total containment marks a fragment.
        fine = [det(278, 142, 313, 73, "BAKERY", 1.0), det(560, 142, 120, 73, "XY", 0.9)]
        assert {d.text for d in _dedup_zoom_detections(fine)} == {"BAKERY", "XY"}


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


# -- ink-support hallucination gate (Cluster 3) ------------------------------
# a moderately-low-confidence read (script OR not) sitting on pixels with NO
# separable ink at all is a hallucination text_mask catches independently of
# confidence -- it complements, never replaces, the script/digit logic that
# deliberately protects genuine low-confidence-but-real text (e.g. the
# japan-street neon banner read correctly at 0.015 confidence).

_IMAGES = Path(__file__).resolve().parents[1] / "images"


class TestInkSupportGate:
    def _uniform_png(self, tmp_path):
        # a perfectly uniform crop is the ONLY reliable "no ink" signal:
        # Otsu finds no separation, so text_mask returns None. this is what
        # a hallucination over blown-out sky looks like at the pixel level.
        from PIL import Image
        p = tmp_path / "blank.png"
        Image.new("RGB", (120, 120), (200, 200, 200)).save(p)
        return str(p)

    def test_ink_support_false_on_uniform_crop(self, tmp_path):
        pytest.importorskip("cv2")
        blank = self._uniform_png(tmp_path)
        assert _has_ink_support(blank, BBox(x=10, y=10, width=60, height=60)) is False

    def test_ink_support_true_on_real_text_crop(self):
        # real dense signage from the china-street fixture -- genuine ink
        # always Otsu-separates; this must never read as "no ink"
        pytest.importorskip("cv2")
        img = _IMAGES / "china-street.png"
        if not img.exists():
            pytest.skip("china-street fixture not present")
        # 茂昌眼镜公司: tall vertical sign, the densest text region in frame
        assert _has_ink_support(str(img), BBox(x=545, y=13, width=127, height=603)) is True

    def test_ink_support_fails_open_on_bad_asset(self):
        # never block a detection over an unrelated load failure
        assert _has_ink_support("/no/such/file.png", BBox(x=0, y=0, width=10, height=10)) is True

    def test_prune_drops_low_conf_no_ink_detection(self, tmp_path):
        # the r1-class hallucination: a hiragana read at 0.542 over blank
        # pixels. has_script is True, so the script/digit logic KEEPS it --
        # the ink gate is what drops it.
        pytest.importorskip("cv2")
        blank = self._uniform_png(tmp_path)
        inst = InstText(
            id="r1", bounding_box=BBox(x=20, y=20, width=8, height=11),
            text="に", confidence=0.542,
        )
        assert _prune_hallucinations([inst], blank) == []

    def test_prune_keeps_low_conf_real_ink_detection(self):
        # regression guard against over-tightening: a low-confidence read
        # over GENUINE ink (a real fixture crop, not a synthetic blank) must
        # survive -- the japan-street-banner-at-0.015 class of real text the
        # script/digit logic exists to protect.
        pytest.importorskip("cv2")
        img = _IMAGES / "china-street.png"
        if not img.exists():
            pytest.skip("china-street fixture not present")
        inst = InstText(
            id="r1", bounding_box=BBox(x=545, y=13, width=127, height=603),
            text="茂昌眼镜公司", confidence=0.05,
        )
        assert len(_prune_hallucinations([inst], str(img))) == 1

    def test_prune_ink_gate_scoped_to_low_confidence(self, tmp_path):
        # a confident read (>= 0.65) is never subjected to the ink gate,
        # even over blank pixels -- confident reads are trusted outright.
        pytest.importorskip("cv2")
        blank = self._uniform_png(tmp_path)
        inst = InstText(
            id="r1", bounding_box=BBox(x=20, y=20, width=40, height=40),
            text="設計", confidence=0.9,
        )
        assert len(_prune_hallucinations([inst], blank)) == 1

    def test_prune_without_asset_skips_ink_gate(self):
        # asset=None (the older call convention) must behave exactly as
        # before: the ink gate is simply not applied.
        inst = InstText(
            id="r1", bounding_box=BBox(x=0, y=0, width=8, height=11),
            text="に", confidence=0.542,
        )
        assert len(_prune_hallucinations([inst])) == 1


class TestSourceLanguageVote:
    """`src_lang` was read by ToFU (text-expansion ratio), basil.pairing()
    and memory, but only ever WRITTEN by one server endpoint -- so every
    pipeline-driven run predicted expansion against an English source no
    matter what the sign actually said."""

    def test_build_manifest_sets_src_lang(self):
        m = build_manifest(
            "fixture.png",
            [det(0, 0, 200, 60, "焼肉", lang="ja")],
            identify_languages=False, prune_garbage=False,
        )
        assert m.src_lang == "ja"

    def test_vote_is_area_weighted_not_count_weighted(self):
        # one big storefront sign outranks several small latin fragments:
        # the scene is Japanese even though "en" wins on raw count
        m = build_manifest(
            "fixture.png",
            [
                det(0, 0, 400, 200, "歌舞伎町一番街", lang="ja"),
                det(0, 300, 30, 10, "3F", lang="en"),
                det(40, 300, 30, 10, "B1", lang="en"),
                det(80, 300, 30, 10, "XV", lang="en"),
            ],
            identify_languages=False, prune_garbage=False,
        )
        assert m.src_lang == "ja"

    def test_no_language_evidence_leaves_src_lang_unset(self):
        # never invent a source language: None means "unknown", which
        # ToFU treats differently from a confident "en"
        m = build_manifest(
            "fixture.png",
            [det(0, 0, 100, 40, "1234")],
            identify_languages=False, prune_garbage=False,
        )
        assert m.src_lang is None


# -- clipped edge-glyph rescue -------------------------------------------------

class _StubEngine:
    """Returns a canned re-read for every widened crop."""

    def __init__(self, text, conf=0.9):
        self._text, self._conf = text, conf
        self.seen = []

    def detect_in_regions(self, asset, boxes, pad=0, polygons=None):
        self.seen = list(boxes)
        return [[det(b.x, b.y, b.width, b.height, self._text, self._conf)] for b in boxes]


def _edge_inst(text, conf=0.995, box=(493, 111, 39, 39)):
    x, y, w, h = box
    return InstText(id="r1", bounding_box=BBox(x=x, y=y, width=w, height=h),
                    text=text, confidence=conf)


class TestRescueClippedEdgeGlyphs:
    def test_recovers_a_clipped_leading_glyph(self):
        # rue-des-martyrs: the box stops ~2px inside the 'D' and the recognizer
        # reads 'ES' at 0.995 -- confidently, so nothing downstream catches it.
        inst = _edge_inst("ES")
        assert rescue_clipped_edge_glyphs(None, [inst], _StubEngine("DES", 0.98)) == 1
        assert inst.text == "DES"
        assert inst.recognition_history[-1]["stage"] == "edge_rescue"

    def test_recovers_a_clipped_trailing_glyph(self):
        inst = _edge_inst("RUE")
        assert rescue_clipped_edge_glyphs(None, [inst], _StubEngine("RUES", 0.98)) == 1
        assert inst.text == "RUES"

    def test_rejects_an_unrelated_reread(self):
        # a wider crop that reads something else entirely is a different
        # answer, not a recovered glyph -- this is what stops the pass from
        # absorbing a neighbouring region's text.
        inst = _edge_inst("ES")
        assert rescue_clipped_edge_glyphs(None, [inst], _StubEngine("MARTYRS", 0.99)) == 0
        assert inst.text == "ES"

    def test_rejects_an_infix_change(self):
        # the old text must survive WHOLE at one end; 'DESK' contains 'ES'
        # only in the middle, so the original read did not merely lose an edge
        inst = _edge_inst("ES")
        assert rescue_clipped_edge_glyphs(None, [inst], _StubEngine("DESK", 0.99)) == 0
        assert inst.text == "ES"

    def test_rejects_a_much_less_confident_read(self):
        inst = _edge_inst("ES", conf=0.99)
        assert rescue_clipped_edge_glyphs(None, [inst], _StubEngine("DES", 0.40)) == 0
        assert inst.text == "ES"

    def test_widens_the_box_it_probes(self):
        inst = _edge_inst("ES")
        engine = _StubEngine("DES", 0.98)
        rescue_clipped_edge_glyphs(None, [inst], engine)
        probed = engine.seen[0]
        assert probed.x < 493 and probed.width > 39
