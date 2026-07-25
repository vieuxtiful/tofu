"""Basil's language-pairing verdict.

Cross-region rearrangement is a property of the language PAIR, not of any
one sign, so the verdict has to be reproducible from declared typology
alone.  These are the three outcomes the layer depends on.
"""

from tofu.layers import basil


def test_logographic_pair_never_needs_plating():
    # ja and zh agree on adjective order, genitive order and designator
    # position alike, so no arrangement of a Japanese sign's regions is
    # ever grammatically forced in Chinese.
    verdict = basil.pairing("ja", "zh-cn")

    assert verdict["verdict"] == "unnecessary"
    assert "preserves modifier-head order" in verdict["reasons"][0]


def test_french_prenominal_class_makes_italian_plating_possible():
    # fr and it agree on every feature value; only French's closed class of
    # prenominal adjectives explains why VIEUX MURS becomes MURI VECCHI.
    verdict = basil.pairing("fr", "it")

    assert verdict["verdict"] == "possible"
    assert any("prenominal adjectives" in reason for reason in verdict["reasons"])


def test_designator_and_genitive_differences_are_each_sufficient():
    assert basil.pairing("fr", "en")["verdict"] == "possible"   # adjective + designator
    assert basil.pairing("ja", "en")["verdict"] == "possible"   # genitive + designator
    # English and German both put the designator after the name and the
    # adjective before it, so "Main Street" -> "Hauptstraße" moves nothing.
    assert basil.pairing("en", "de")["verdict"] == "unnecessary"


def test_unknown_language_fails_open_rather_than_claiming_unnecessary():
    verdict = basil.pairing("fr", "qqq")

    assert verdict["verdict"] == "unknown"
    assert "failing open" in verdict["reasons"][0]


def test_locale_subtags_and_identical_languages_resolve():
    assert basil.pairing("pt-BR", "pt-PT")["verdict"] == "unnecessary"
    assert basil.pairing("zh-cn", "ja")["verdict"] == "unnecessary"
    assert basil.pairing(None, "it")["verdict"] == "unknown"
