## 🍢 wasabi: Japanese/simplified-Chinese glyph normalization (pure-logic)
from tofu.core.types import BBox, InstText
from tofu.layers.wasabi import normalize_japanese_kanji, season


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
