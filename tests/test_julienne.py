## 🍢 julienne — per-character script identity
"""
A string is not "in a script"; its characters are (UAX #24).  The cases in
TestRequirementBreakdown are the exact strings whose previous single-script
verdicts were wrong, and they are asserted character-for-character so the
majority vote can never come back.
"""

import pytest

from tofu.layers import julienne
from tofu.layers.tofu import lang_to_script


class TestRequirementBreakdown:
    """The five strings that exposed the majority-vote bug."""

    def test_kanji_and_hiragana_is_not_purely_han(self):
        a = julienne.analyse("古い壁のとおり", "ja")
        assert a.scripts == ["Hani", "Hira"]
        assert a.counts == {"Hani": 2, "Hira": 5}
        assert a.is_mixed

    def test_all_hiragana_is_hiragana(self):
        a = julienne.analyse("とうきょう", "ja")
        assert a.scripts == ["Hira"] and not a.is_mixed

    def test_katakana_word_including_the_prolonged_sound_mark(self):
        a = julienne.analyse("タワー", "ja")
        assert a.scripts == ["Kana"] and not a.is_mixed
        # U+30FC is the KATAKANA-HIRAGANA prolonged sound mark: Script=Common
        # with Script_Extensions {Hira, Kana}, so it is Katakana only by context
        assert a.per_character[-1] == ("ー", "Kana")

    def test_kanji_and_katakana_is_mixed(self):
        a = julienne.analyse("東京タワー", "ja")
        assert a.scripts == ["Hani", "Kana"]
        assert a.counts == {"Hani": 2, "Kana": 3}
        assert a.is_mixed

    def test_hangul_and_hanja_is_not_han(self):
        """서울 is Hangul; 特別市 is hanja.  The old majority vote called the
        whole string CJK because three of five characters are ideographs."""
        a = julienne.analyse("서울特別市", "ko")
        assert a.scripts == ["Hang", "Hani"]
        assert a.counts == {"Hang": 2, "Hani": 3}
        assert a.is_mixed


class TestContextResolution:
    def test_prolonged_sound_mark_follows_the_run_it_extends(self):
        assert julienne.analyse("ラーメン").per_character[1] == ("ー", "Kana")
        assert julienne.analyse("らーめん").per_character[1] == ("ー", "Hira")

    def test_a_leading_extension_takes_the_following_run(self):
        assert julienne.analyse("ーメン").per_character[0] == ("ー", "Kana")

    @pytest.mark.parametrize("text", ["Rue des Murs", "Rue, des Murs!", "Rue 12 Murs"])
    def test_punctuation_digits_and_spaces_never_make_a_string_mixed(self, text):
        a = julienne.analyse(text, "fr")
        assert a.scripts == ["Latn"] and not a.is_mixed

    def test_a_string_of_only_punctuation_has_no_script(self):
        a = julienne.analyse("!?—", "fr")
        assert a.scripts == [] and a.dominant is None and a.summary == "Zyyy"

    def test_empty_and_none_are_safe(self):
        for value in (None, ""):
            a = julienne.analyse(value, "fr")
            assert a.scripts == [] and not a.is_mixed


class TestCompositeScriptCodes:
    """ISO 15924 Jpan and Kore ARE multi-script by definition."""

    def test_jpan_admits_han_hiragana_and_katakana(self):
        assert julienne.expected_scripts("Jpan") == {"Hani", "Hira", "Kana"}

    def test_kore_admits_hangul_and_han(self):
        assert julienne.expected_scripts("Kore") == {"Hang", "Hani"}

    @pytest.mark.parametrize("lang,text", [
        ("ja", "東京タワー"), ("ja", "古い壁のとおり"), ("ja", "とうきょう"),
        ("ko", "서울特別市"), ("ko", "옛 벽 거리"),
    ])
    def test_correct_japanese_and_korean_are_never_flagged(self, lang, text):
        result = julienne.validate(text, lang, lang_to_script.get(lang))
        assert result["unexpected_scripts"] == []
        assert result["status"] == "pass"

    def test_wrong_script_is_still_caught(self):
        result = julienne.validate("Улица", "fr", lang_to_script.get("fr"))
        assert result["unexpected_scripts"] == ["Cyrl"]
        assert result["status"] == "review"


class TestEmbeddedVersusUntranslated:
    """A foreign script means opposite things depending on what surrounds it.

    Brand names and model numbers legitimately stay in the source script, so
    the presence of Latin in Arabic copy is not a defect.  A region carrying
    NONE of its own declared script, however, was never translated.
    """

    @pytest.mark.parametrize("lang,text", [
        ("ar", "TOFU برو"),        # brand inside Arabic
        ("ja", "iPhoneを使う"),      # brand inside Japanese
        ("zh-cn", "WiFi 密码"),     # token inside Chinese
        ("ko", "KTX 열차"),         # token inside Korean
        ("ru", "Wi-Fi пароль"),    # token inside Russian
    ])
    def test_a_foreign_token_beside_the_declared_script_is_embedded(self, lang, text):
        result = julienne.validate(text, lang, lang_to_script.get(lang))
        assert result["embedded_scripts"] == ["Latn"]
        assert result["unexpected_scripts"] == []
        assert result["status"] == "pass"

    @pytest.mark.parametrize("lang,text,foreign", [
        ("fr", "Библиотека Открыто", "Cyrl"),
        ("ru", "Rue des Murs", "Latn"),
        ("zh-cn", "Open today", "Latn"),
    ])
    def test_none_of_the_declared_script_reads_as_untranslated(self, lang, text, foreign):
        result = julienne.validate(text, lang, lang_to_script.get(lang))
        assert result["unexpected_scripts"] == [foreign]
        assert result["embedded_scripts"] == []
        assert result["status"] == "review"

    def test_the_rule_is_presence_not_character_share(self):
        """Latin spends far more characters per unit of meaning than Han, so
        「iPhoneを使う」 is 67% Latin BY CHARACTER and entirely ordinary
        Japanese by content.  A share threshold flagged it; presence does not."""
        result = julienne.validate("iPhoneを使う", "ja", "Jpan")
        assert result["foreign_share"] > 0.5      # a majority of characters
        assert result["expected_script_present"] is True
        assert result["status"] == "pass"         # and still not a defect


class TestHanVariant:
    """Unicode unifies both Chinese orthographies as Hani, so the variant
    comes from GB 2312 against Big5 rather than any Unicode property."""

    @pytest.mark.parametrize("text,variant", [
        ("旧墙街", "Hans"), ("舊牆街", "Hant"),
        ("台湾", "Hans"), ("臺灣", "Hant"),
    ])
    def test_exclusive_characters_decide_the_variant(self, text, variant):
        assert julienne.analyse(text, "zh-cn").han_variant == variant

    def test_shared_characters_are_honestly_undetermined(self):
        """出 and 口 exist in both orthographies; guessing would be a lie."""
        assert julienne.analyse("出口", "zh-cn").han_variant == "undetermined"

    @pytest.mark.parametrize("lang,declared,text", [
        ("zh-cn", "Hans", "舊牆街"), ("zh-hk", "Hant", "旧墙街"),
    ])
    def test_wrong_orthography_is_a_mismatch(self, lang, declared, text):
        result = julienne.validate(text, lang, declared)
        assert result["han_variant_mismatch"] is True
        assert result["status"] == "review"

    @pytest.mark.parametrize("lang,declared,text", [
        ("zh-cn", "Hans", "旧墙街"), ("zh-hk", "Hant", "舊牆街"),
        ("zh-cn", "Hans", "出口"),
    ])
    def test_right_orthography_passes(self, lang, declared, text):
        assert not julienne.validate(text, lang, declared).get("han_variant_mismatch")

    @pytest.mark.parametrize("lang,text", [("ja", "東京タワー"), ("ko", "서울特別市")])
    def test_the_variant_axis_does_not_apply_to_japanese_or_korean(self, lang, text):
        """東 and 別 are absent from GB 2312 and so look traditional-exclusive.
        In 東京 they are kanji and in 特別市 hanja, where simplified-vs-
        traditional is not a distinction that exists.  Asking anyway would
        brand every Japanese sign Traditional Chinese."""
        assert julienne.analyse(text, lang).han_variant is None
        assert not julienne.validate(text, lang, lang_to_script.get(lang)).get(
            "han_variant_mismatch"
        )


class TestDirection:
    @pytest.mark.parametrize("text,direction", [
        ("Rue des Murs", "ltr"), ("Улица", "ltr"), ("東京タワー", "ltr"),
        ("שלום", "rtl"), ("مرحبا", "rtl"),
    ])
    def test_direction_from_bidi_class(self, text, direction):
        assert julienne.analyse(text).direction == direction
