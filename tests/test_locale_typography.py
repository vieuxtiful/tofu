"""The space before an exclamation mark is a locale rule, not a language one.

French (France) sets a space before '!', '?' and ';'. French (Canada) sets
those tight and spaces only ':'. Same language, opposite answer, so nothing
here may be decided on the language subtag alone.
"""

from tofu.core.types import BBox, InstText, TextManifest
from tofu.layers.cicerone import _apply_locale_typography
from tofu.utils.locale_typography import apply_punctuation_spacing, resolve_locale


class TestResolveLocale:
    def test_a_region_qualified_tag_is_used_as_given(self):
        assert resolve_locale("fr-CA", "fr-FR") == "fr-CA"

    def test_a_bare_tag_falls_back_to_its_default_region(self):
        # follows the map already in utils/interchange.py
        assert resolve_locale("fr", None) == "fr-FR"

    def test_the_asset_lends_its_region_to_a_bare_region_tag(self):
        """The ordering that matters.

        Detection emits a bare 'fr' on every region, so resolving "the first
        tag that parses" would let that silently outrank an asset the user
        explicitly set to fr-CA -- and then space the marks Canadian French
        sets tight.
        """
        assert resolve_locale("fr", "fr-CA") == "fr-CA"

    def test_the_asset_lends_nothing_across_a_language_boundary(self):
        # a French line on a Canadian-English poster is not fr-CA
        assert resolve_locale("fr", "en-CA") == "fr-FR"

    def test_an_unknown_language_gets_no_locale(self):
        assert resolve_locale("en", None) is None
        assert resolve_locale(None, None) is None


class TestPunctuationSpacing:
    def test_french_france_spaces_the_mark(self):
        assert apply_punctuation_spacing("nos rues!", "fr-FR") == "nos rues !"

    def test_canadian_french_does_not(self):
        assert apply_punctuation_spacing("nos rues!", "fr-CA") == "nos rues!"

    def test_canadian_french_still_spaces_a_colon(self):
        assert apply_punctuation_spacing("Attention: ici", "fr-CA") == "Attention : ici"

    def test_a_locale_with_no_rule_is_left_alone(self):
        assert apply_punctuation_spacing("what!", "en-US") == "what!"
        assert apply_punctuation_spacing("what!", None) == "what!"

    def test_it_is_idempotent(self):
        once = apply_punctuation_spacing("nos rues!", "fr-FR")
        assert apply_punctuation_spacing(once, "fr-FR") == once

    def test_an_existing_space_of_any_kind_is_left_alone(self):
        # a narrow no-break space already there is more correct than the
        # ordinary space this would insert, not less
        assert apply_punctuation_spacing("nos rues !", "fr-FR") == "nos rues !"

    def test_a_run_of_marks_is_spaced_once(self):
        assert apply_punctuation_spacing("Quoi?!", "fr-FR") == "Quoi ?!"

    def test_a_leading_mark_gets_no_space(self):
        # "!" alone is a region, not a sentence
        assert apply_punctuation_spacing("! seul", "fr-FR") == "! seul"

    def test_a_mark_inside_a_token_is_left_alone(self):
        # spacing only the left side would be worse than what was recognised,
        # and closing both sides is general normalisation, not a locale rule
        assert apply_punctuation_spacing("Attention:ici", "fr-FR") == "Attention:ici"

    def test_guillemets_are_spaced_on_the_inside(self):
        assert apply_punctuation_spacing("«bonjour»", "fr-FR") == "« bonjour »"

    def test_a_full_stop_is_not_a_spaced_mark_anywhere(self):
        assert apply_punctuation_spacing("fin.", "fr-FR") == "fin."


class TestManifestPass:
    def _manifest(self, text, region_lang, asset_lang):
        inst = InstText("r1", BBox(0, 0, 100, 40), text=text, detected_language=region_lang)
        return TextManifest(asset_id="a", total_regions=1, instances=[inst],
                            src_lang=asset_lang), inst

    def test_the_region_is_respaced_and_the_change_is_recorded(self):
        manifest, inst = self._manifest("nos rues!", "fr", "fr")
        assert _apply_locale_typography(manifest) == 1
        assert inst.text == "nos rues !"
        entry = inst.recognition_history[-1]
        assert entry["stage"] == "locale_typography"
        assert entry["primary_text"] == "nos rues!"

    def test_a_canadian_asset_leaves_the_region_alone(self):
        manifest, inst = self._manifest("nos rues!", "fr", "fr-CA")
        assert _apply_locale_typography(manifest) == 0
        assert inst.text == "nos rues!"

    def test_an_empty_region_is_skipped(self):
        manifest, inst = self._manifest("   ", "fr", "fr")
        assert _apply_locale_typography(manifest) == 0
