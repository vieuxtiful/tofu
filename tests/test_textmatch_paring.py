## 🍢 textmatch: confusable paring + span matching (pure-logic)
import pytest

from tofu.utils.textmatch import (
    best_span_similarity,
    fuzzy_similarity,
    pare,
    pared_similarity,
)


class TestPare:
    def test_cyrillic_confusables_fold_onto_their_latin_twins(self):
        assert pare("ВМЕСТЕ") == "BMECTE"
        assert pare("РОССИЯ") == "POCCИЯ"  # И and Я have no latin lookalike

    def test_case_is_preserved_because_the_map_depends_on_it(self):
        # Cyrillic В is latin B; Cyrillic в is NOT latin b.  Casefolding
        # before paring would destroy exactly that distinction.
        assert pare("В") == "B"
        assert pare("в") == "в"

    def test_non_confusable_characters_pass_through(self):
        for text in ("Valentina Ursu", "上海明牌", "1789", ""):
            assert pare(text) == text

    def test_a_half_latin_read_pares_to_the_same_skeleton_as_the_truth(self):
        # russian-billboard-2's measured read against its ground truth.
        assert pare("BMЕСTЕ С РОССИЕЙ!") == pare("ВМЕСТЕ С РОССИЕЙ!")


class TestPairedSimilarity:
    def test_script_corruption_stops_hiding_a_match(self):
        read, truth = "BMЕСTЕ С РОССИЕЙ!", "ВМЕСТЕ С РОССИЕЙ!"
        assert fuzzy_similarity(read, truth) < 1.0
        assert pared_similarity(read, truth) == pytest.approx(1.0)


class TestBestSpanSimilarity:
    def test_a_fragment_scores_against_the_part_it_came_from(self):
        # the whole point: plain similarity cannot see a short read as a
        # PIECE of a long one, because the length difference dominates.
        needle, haystack = "esclavagistes", "crimes coloniaux et esclavagistes"
        assert fuzzy_similarity(needle, haystack) < 0.7
        assert best_span_similarity(needle, haystack) == pytest.approx(1.0)

    def test_unrelated_text_scores_nothing(self):
        assert best_span_similarity("娘", "矗") == 0.0

    def test_a_needle_longer_than_the_haystack_is_not_a_span(self):
        assert best_span_similarity("BAKERY", "B") == 0.0

    def test_equal_length_falls_back_to_whole_string_comparison(self):
        assert best_span_similarity("OPTC", "OPTC") == pytest.approx(1.0)

    def test_empty_operands(self):
        assert best_span_similarity("", "anything") == 0.0
        assert best_span_similarity("anything", "") == 0.0
