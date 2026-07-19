## 🍢 cicerone pure-logic units (no OCR model required)
from tofu.layers.cicerone import (
    RawDetection,
    ScriptDetector,
    _compose_crop_text,
    _polygon_bbox,
    guess_latin_language,
    merge_detections,
    merge_vertical_columns,
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
