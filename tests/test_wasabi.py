## 🍢 wasabi: Japanese/simplified-Chinese glyph normalization (pure-logic)
from tofu.core.types import BBox, InstText
from tofu.layers.wasabi import (
    normalize_japanese_kanji,
    restore_cyrillic_homoglyphs,
    season,
)


def _inst(text, lang="ja", confidence=0.9, id="r1"):
    return InstText(
        id=id, bounding_box=BBox(x=0, y=0, width=10, height=10),
        text=text, confidence=confidence, detected_language=lang,
    )


class TestNormalizeJapaneseKanji:
    def test_known_pair_corrected(self):
        assert normalize_japanese_kanji("剧場通") == "劇場通"
        assert normalize_japanese_kanji("烧肉") == "焼肉"

    def test_island_pair_corrected(self):
        # measured live: japan-subs's "下島" (Shimojima) read back as "下岛"
        assert normalize_japanese_kanji("下岛") == "下島"

    def test_unknown_characters_untouched(self):
        assert normalize_japanese_kanji("歌舞伎町一番街") == "歌舞伎町一番街"

    def test_empty_string(self):
        assert normalize_japanese_kanji("") == ""

    def test_mixed_text_only_known_chars_swapped(self):
        assert normalize_japanese_kanji("烧肉屋") == "焼肉屋"


class TestSeason:
    def test_corrects_ja_instance_regardless_of_confidence(self):
        # a systematic model limitation, not a recognition-uncertainty
        # signal -- even a CONFIDENT wrong glyph gets fixed, unlike
        # menu's confidence-gated gazetteer correction
        inst = _inst("剧場通", confidence=0.96)
        n = season([inst])
        assert n == 1
        assert inst.text == "劇場通"
        assert inst.ocr_correction["applied"] is True
        assert inst.ocr_correction["original_text"] == "剧場通"
        assert inst.ocr_correction["correction_resource"]["data_version"] == "1.0.0"

    def test_non_ja_instance_untouched(self):
        inst = _inst("烧肉", lang="zh-cn")
        n = season([inst])
        assert n == 0
        assert inst.text == "烧肉"
        assert inst.ocr_correction is None

    def test_already_correct_text_not_recorded_as_corrected(self):
        inst = _inst("劇場通り")
        n = season([inst])
        assert n == 0
        assert inst.ocr_correction is None

    def test_falls_back_to_language_field_when_no_detected_language(self):
        inst = InstText(
            id="r1", bounding_box=BBox(x=0, y=0, width=10, height=10),
            text="烧肉", confidence=0.9, detected_language=None, language="ja",
        )
        n = season([inst])
        assert n == 1
        assert inst.text == "焼肉"

    def test_empty_text_skipped(self):
        inst = _inst("")
        assert season([inst]) == 0

    def test_cyrillic_repair_is_not_gated_on_a_language_label(self):
        # the defect makes a Cyrillic region LOOK latin, so by the time this
        # runs the region is routinely labelled 'en'. gating the course on
        # the label would exempt exactly the regions that need it.
        inst = _inst("BMЕСTЕ С РОССИЕЙ!", lang="en")
        assert season([inst]) == 1
        assert inst.text == "ВМЕСТЕ С РОССИЕЙ!"
        assert inst.ocr_correction["course"] == "cyrillic_homoglyph"
        assert inst.ocr_correction["applied"] is True


class TestRestoreCyrillicHomoglyphs:
    def test_latin_lookalikes_inside_a_cyrillic_word_are_restored(self):
        # russian-billboard-2: a joint (ru, en) charset decodes В, М and Т as
        # latin B, M and T, and nothing downstream can tell.
        assert restore_cyrillic_homoglyphs("BMЕСTЕ С РОССИЕЙ!") == "ВМЕСТЕ С РОССИЕЙ!"

    def test_a_wholly_latin_word_survives_a_cyrillic_region(self):
        # russian-billboard's 'Za' -- the war symbol, genuinely latin, set
        # against Cyrillic slogans. its ground truth says a latin read there
        # is CORRECT, so the repair must not reach it.
        assert restore_cyrillic_homoglyphs("Za ПОБЕДУ!") == "Za ПОБЕДУ!"
        assert restore_cyrillic_homoglyphs("Za") == "Za"

    def test_evidence_free_words_are_decided_by_their_region(self):
        # every character of 'ПOБEДY' below is a shared letterform except
        # П, Б and Д -- those three settle the script for the whole word.
        assert restore_cyrillic_homoglyphs("ПOБEДY!") == "ПОБЕДУ!"

    def test_a_region_with_no_unambiguous_cyrillic_is_left_alone(self):
        # 'BMECTE' spelled entirely in latin lookalikes is genuinely
        # undecidable on its own, and guessing would corrupt latin text.
        assert restore_cyrillic_homoglyphs("BMECTE") == "BMECTE"

    def test_latin_and_cjk_regions_untouched(self):
        for text in ("Valentina Ursu (RFE/RL)", "OPTICAL", "MING 上海", "23"):
            assert restore_cyrillic_homoglyphs(text) == text

    def test_correct_cyrillic_passes_through(self):
        assert restore_cyrillic_homoglyphs("В БУДУЩЕЕ") == "В БУДУЩЕЕ"
