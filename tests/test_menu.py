## 🍢 menu: gazetteer-assisted correction (pure-logic, no OCR model required)
from tofu.core.types import BBox, InstText
from tofu.layers.menu import (
    CONFIDENCE_FLOOR,
    browse,
    consult_menu,
    consult_menu_substring,
)


def _inst(text, confidence, lang="ja", id="r1"):
    return InstText(
        id=id, bounding_box=BBox(x=0, y=0, width=10, height=10),
        text=text, confidence=confidence, detected_language=lang,
    )


class TestConsultMenu:
    def test_low_confidence_near_miss_corrects(self):
        # missing one glyph (伎->皮) and a trailing character -- the
        # exact japan-street r2 case that motivated this module
        match = consult_menu("歌舞皮町一番", "ja", 0.237)
        assert match is not None
        assert match.text == "歌舞伎町一番街"

    def test_confident_read_never_overridden(self):
        # even a strong gazetteer match must not touch a confident read
        assert consult_menu("歌舞皮町一番", "ja", CONFIDENCE_FLOOR) is None
        assert consult_menu("歌舞皮町一番", "ja", 0.9) is None

    def test_weak_similarity_no_match(self):
        assert consult_menu("xyz123", "ja", 0.1) is None

    def test_empty_text_no_match(self):
        assert consult_menu("", "ja", 0.1) is None

    def test_language_filter_excludes_cross_language_candidates(self):
        # a low-confidence english read must not match a japanese gazetteer entry
        assert consult_menu("kabukicho", "en", 0.1) is None

    def test_no_language_hint_searches_all_entries(self):
        match = consult_menu("東南装", None, 0.1)
        assert match is not None and match.text == "東南荘"

    def test_marginal_confidence_nonsense_read_corrects(self):
        # regression guard: measured live at 0.511 confidence -- just
        # above the OLD 0.5 floor, which silently exempted this
        # nonsense (not a real Japanese word) read from ever being
        # checked against the gazetteer at all
        match = consult_menu("招你み焼本練", "ja", 0.511)
        assert match is not None
        assert match.text == "お好み焼本陣"


class TestBrowse:
    def test_corrects_matching_instance_in_place(self):
        inst = _inst("歌舞皮町一番", 0.237)
        n = browse([inst])
        assert n == 1
        assert inst.text == "歌舞伎町一番街"
        assert inst.ocr_correction["applied"] is True
        assert inst.ocr_correction["original_text"] == "歌舞皮町一番"

    def test_confident_instance_untouched(self):
        inst = _inst("目", 0.795)
        n = browse([inst])
        assert n == 0
        assert inst.text == "目"
        assert inst.ocr_correction is None

    def test_already_correct_text_not_recorded_as_corrected(self):
        inst = _inst("劇場通り", 0.1)
        n = browse([inst])
        assert n == 0
        assert inst.ocr_correction is None

    def test_two_char_candidate_never_fires_whole_string(self):
        # "下り" is ubiquitous rail/road signage; sharing one char with
        # gazetteer "下島" scores exactly 0.5 -- coin-flip evidence that
        # must never rewrite a whole read. 2-char names are served only
        # by the pixel-verified substring path.
        inst = _inst("下り", 0.55)
        n = browse([inst])
        assert n == 0
        assert inst.text == "下り"
        assert inst.ocr_correction is None


class TestConsultMenuSubstring:
    """composite reads: known names aligned INSIDE a longer text."""

    def test_motivating_composite_proposes_both_misread_spans(self):
        # the real japan-subs directional post: 湯屋/下島/濁河温泉 read
        # as one instance. 下島 is already correct (no proposal); the
        # other two names each align with positional diffs.
        spans = consult_menu_substring("周屋下島周河温泉", "ja")
        assert [(s.start, s.end, s.candidate) for s in spans] == [
            (0, 2, "湯屋"), (4, 8, "濁河温泉"),
        ]
        assert spans[0].diffs == [(0, "周", "湯")]
        assert spans[1].diffs == [(4, "周", "濁")]
        assert spans[1].similarity == 0.75

    def test_misaligned_short_window_excluded_by_positional_gate(self):
        # SequenceMatcher alone can't disambiguate 2-char alignment:
        # "屋下" also scores 0.5 against 湯屋 (屋 matches at the WRONG
        # position). the zip diff (0-of-2 positional matches) kills it.
        spans = consult_menu_substring("周屋下島周河温泉", "ja")
        assert all(not (s.start == 1 and s.candidate == "湯屋") for s in spans)

    def test_equal_length_text_yields_nothing(self):
        # equal-length reads belong to the whole-string and dakuten
        # paths -- substring requires a STRICTLY longer composite
        assert consult_menu_substring("ハンタイ", "ja") == []

    def test_language_filter(self):
        assert consult_menu_substring("周屋下島周河温泉", "en") == []
        assert consult_menu_substring("周屋下島周河温泉", "zh-cn") == []

    def test_spaced_text_yields_nothing(self):
        assert consult_menu_substring("周屋 下島", "ja") == []

    def test_transposition_fails_majority_position_gate(self, monkeypatch):
        # a transposed pair scores well on ratio (common chars) but has
        # ZERO positionally-matching chars beyond the rest -- e.g.
        # candidate アイウ vs window イアウ: ratio 0.667, but 2 diffs of
        # 3 positions fails 2*diffs <= len
        monkeypatch.setattr("tofu.layers.menu.KNOWN_PLACES", [("アイウ", "ja")])
        assert consult_menu_substring("イアウエオ", "ja") == []

    def test_two_occurrences_both_proposed(self, monkeypatch):
        monkeypatch.setattr("tofu.layers.menu.KNOWN_PLACES", [("東南荘", "ja")])
        spans = consult_menu_substring("東南口と東南口", "ja")
        assert [(s.start, s.end) for s in spans] == [(0, 3), (4, 7)]
        assert all(s.candidate == "東南荘" for s in spans)

    def test_overlapping_proposals_resolved_by_similarity(self, monkeypatch):
        # a longer, closer candidate wins the overlap; the shorter,
        # weaker one is dropped rather than double-correcting the span
        monkeypatch.setattr(
            "tofu.layers.menu.KNOWN_PLACES",
            [("アイウ", "ja"), ("アイウエオ", "ja")],
        )
        spans = consult_menu_substring("アイエエオカ", "ja")
        assert len(spans) == 1
        assert spans[0].candidate == "アイウエオ"
        assert spans[0].start == 0 and spans[0].end == 5

    def test_touching_spans_both_survive(self, monkeypatch):
        # half-open overlap semantics: [0:3) and [3:6) touch, don't overlap
        monkeypatch.setattr(
            "tofu.layers.menu.KNOWN_PLACES",
            [("東南荘", "ja"), ("劇場通", "ja")],
        )
        spans = consult_menu_substring("東南口劇場遥り", "ja")
        assert [(s.start, s.end, s.candidate) for s in spans] == [
            (0, 3, "東南荘"), (3, 6, "劇場通"),
        ]


class TestBrowseSubstring:
    """browse()'s substring course: string tier, pixel tier, and the
    whole-string suppression rule."""

    def test_motivating_composite_fully_corrected_without_asset(self):
        # the japan-subs post end to end: 濁河温泉 applies on string
        # evidence (len-4, single diff), 下島 is an exact window, and
        # those two confirmed siblings corroborate the weak 湯屋 span
        # even though no asset is available for pixel verification.
        inst = _inst("周屋下島周河温泉", 0.657)
        n = browse([inst])
        assert n == 1
        assert inst.text == "湯屋下島濁河温泉"
        assert inst.ocr_correction["applied"] is True
        assert "濁河温泉" in inst.ocr_correction["reason"]
        assert "corroborated" in inst.ocr_correction["reason"]

    def test_pixel_tier_applies_when_all_diffs_confirmed(self, monkeypatch):
        monkeypatch.setattr(
            "tofu.layers.savor.chew_swaps",
            lambda asset, inst, positions, lang, font_registry=None: {
                pos: True for pos, _, _ in positions
            },
        )
        inst = _inst("周屋下島周河温泉", 0.657)
        n = browse([inst], asset="fake.png")
        assert n == 1
        assert inst.text == "湯屋下島濁河温泉"  # both spans applied

    def test_pixel_tier_false_discards_span_entirely(self, monkeypatch):
        # pixels actively contradict the weak 湯屋 proposal: no record
        # of it at all -- but the string-tier span still applies
        monkeypatch.setattr(
            "tofu.layers.savor.chew_swaps",
            lambda asset, inst, positions, lang, font_registry=None: {
                pos: False for pos, _, _ in positions
            },
        )
        inst = _inst("周屋下島周河温泉", 0.657)
        browse([inst], asset="fake.png")
        assert inst.text == "周屋下島濁河温泉"
        assert "湯屋" not in inst.ocr_correction["reason"]

    def test_pixel_tier_inconclusive_recorded_unapplied(self, monkeypatch):
        # gazetteer stripped to leave only ONE confirmed sibling (the
        # applied 濁河温泉) -- below CORROBORATION_MIN_SIBLINGS, so the
        # inconclusive 湯屋 span stays a review record, not a rewrite
        monkeypatch.setattr(
            "tofu.layers.menu.KNOWN_PLACES", [("湯屋", "ja"), ("濁河温泉", "ja")],
        )
        monkeypatch.setattr(
            "tofu.layers.savor.chew_swaps",
            lambda asset, inst, positions, lang, font_registry=None: {},
        )
        inst = _inst("周屋下島周河温泉", 0.657)
        browse([inst], asset="fake.png")
        assert inst.text == "周屋下島濁河温泉"  # string tier still applied
        assert "left unchanged, pixel evidence inconclusive" in inst.ocr_correction["reason"]

    def test_substring_alignment_suppresses_whole_string_rewrite(self):
        # regression guard: at 0.55 confidence the OLD whole-string path
        # would fire (fuzzy 0.5 vs 濁河温泉) and collapse the entire
        # 8-char post to one 4-char name, destroying 湯屋 and 下島.
        inst = _inst("周屋下島周河温泉", 0.55)
        browse([inst])
        assert inst.text == "湯屋下島濁河温泉"  # span-corrected, never collapsed


class TestCorroborationTier:
    """document-level context: ≥2 independently-confirmed sibling names
    on the same composite read let a pixel-inconclusive single-diff
    span apply. never over a pixel contradiction; siblings counted once
    (no bootstrap)."""

    def test_fires_with_two_confirmed_siblings(self, monkeypatch):
        monkeypatch.setattr(
            "tofu.layers.savor.chew_swaps",
            lambda asset, inst, positions, lang, font_registry=None: {},
        )
        inst = _inst("周屋下島周河温泉", 0.657)
        n = browse([inst], asset="fake.png")
        assert n == 1
        assert inst.text == "湯屋下島濁河温泉"
        assert "corroborated by 2 co-occurring known names" in inst.ocr_correction["reason"]

    def test_does_not_fire_with_one_sibling(self, monkeypatch):
        # only 濁河温泉 confirms (下島 removed from gazetteer): 1 < 2
        monkeypatch.setattr(
            "tofu.layers.menu.KNOWN_PLACES", [("湯屋", "ja"), ("濁河温泉", "ja")],
        )
        inst = _inst("周屋下島周河温泉", 0.657)
        browse([inst])
        assert inst.text == "周屋下島濁河温泉"
        assert "corroborated" not in inst.ocr_correction["reason"]

    def test_never_overrides_pixel_contradiction(self, monkeypatch):
        # pixels actively say the recognized char is right: the span is
        # discarded BEFORE the corroboration tier, siblings or not
        monkeypatch.setattr(
            "tofu.layers.savor.chew_swaps",
            lambda asset, inst, positions, lang, font_registry=None: {
                pos: False for pos, _, _ in positions
            },
        )
        inst = _inst("周屋下島周河温泉", 0.657)
        browse([inst], asset="fake.png")
        assert inst.text == "周屋下島濁河温泉"
        assert "湯屋" not in inst.ocr_correction["reason"]

    def test_multi_diff_span_never_corroborated(self, monkeypatch):
        # a 2-diff span (r5-style coincidence shape) stays a review item
        # regardless of siblings -- context can't carry two glyph swaps
        monkeypatch.setattr(
            "tofu.layers.menu.KNOWN_PLACES",
            [("東南荘", "ja"), ("劇場通り", "ja"), ("濁河温泉", "ja")],
        )
        # 東南荘 and 劇場通り appear exactly (2 siblings); the 濁河温泉
        # window has 2 diffs (小坂温泉-like coincidence) -> not promoted
        inst = _inst("東南荘の劇場通り小坂温泉", 0.9)
        browse([inst])
        assert "小坂温泉" in (inst.text or "")  # untouched span
        if inst.ocr_correction:
            assert inst.ocr_correction.get("applied") is not True or "濁河温泉" not in str(inst.ocr_correction.get("corrected_text"))

    def test_spurious_proposal_on_correct_read_never_string_applied(self, monkeypatch):
        # 小坂温泉郷 (a CORRECT high-confidence read) coincidentally
        # aligns 濁河温泉 with 2 diffs at 0.5 similarity -- must reach
        # the pixel tier only, and pixels saying False must leave the
        # text untouched with nothing recorded
        monkeypatch.setattr(
            "tofu.layers.savor.chew_swaps",
            lambda asset, inst, positions, lang, font_registry=None: {
                pos: False for pos, _, _ in positions
            },
        )
        inst = _inst("小坂温泉郷", 0.9986)
        n = browse([inst], asset="fake.png")
        assert n == 0
        assert inst.text == "小坂温泉郷"
        assert inst.ocr_correction is None
