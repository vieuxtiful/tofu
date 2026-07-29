## 🍢 CER/WER content integrity and ECE/Brier calibration
"""
Content integrity used to be ``SequenceMatcher.ratio()``, whose value scales
with string length: one identical single-character error scored Japanese
0.571 and German 0.952.  CER normalises by reference length, which is why
speech and OCR evaluation have used it for decades.

ECE and Brier are corpus-level by construction and are asserted here on
synthetic populations with known calibration, so the metric itself is
verified rather than the corpus that happens to be checked in.
"""

import math

import pytest

from tofu.layers import temper
from tofu.layers.verify import _edit_distance, error_rates


class TestEditDistance:
    @pytest.mark.parametrize("a,b,expected", [
        ("", "", 0), ("abc", "abc", 0), ("abc", "", 3), ("", "abc", 3),
        ("abc", "abd", 1),          # substitution
        ("abc", "abcd", 1),         # insertion
        ("abcd", "abc", 1),         # deletion
        ("kitten", "sitting", 3),   # the canonical Levenshtein example
    ])
    def test_levenshtein(self, a, b, expected):
        assert _edit_distance(a, b) == expected

    def test_it_works_over_word_tokens_too(self):
        assert _edit_distance(["rue", "des", "murs"], ["rue", "des", "chats"]) == 1


class TestCharacterErrorRate:
    def test_perfect_and_total_failure_bound_the_range(self):
        assert error_rates("東京駅", "東京駅", "ja")["cer"] == 0.0
        assert error_rates("東京駅", "", "ja")["cer"] == 1.0

    def test_cer_is_clamped_when_the_hypothesis_runs_long(self):
        rates = error_rates("Rue", "Rue des Vieux Murs de la Ville", "fr")
        assert rates["cer"] == 1.0  # never above 1, so the score stays in 0-100

    def test_one_error_costs_a_share_of_the_reference_not_a_length_penalty(self):
        """The defect the metric change exists to fix: one wrong character in
        a three-character string is a third of the content, and CER says so.
        SequenceMatcher said 0.571 for Japanese and 0.952 for German."""
        ja = error_rates("東京駅", "東京馬", "ja")["cer"]
        de = error_rates("Strasse der Alten Mauern", "Strasse der Alten Mauerz", "de")["cer"]
        assert ja == pytest.approx(1 / 3, abs=1e-4)
        assert de == pytest.approx(1 / 21, abs=1e-4)

    @pytest.mark.parametrize("lang,ref,hyp", [
        ("zh-cn", "旧墙街", "旧墙衔"), ("zh-hk", "舊牆街", "舊牆衔"),
        ("ja", "東京駅", "東京馬"),
    ])
    def test_equal_length_cjk_references_are_penalised_equally(self, lang, ref, hyp):
        assert error_rates(ref, hyp, lang)["cer"] == pytest.approx(1 / 3, abs=1e-4)

    def test_an_empty_reference_yields_no_rate_rather_than_a_zero(self):
        assert error_rates("", "anything", "fr")["cer"] is None


class TestWordErrorRate:
    @pytest.mark.parametrize("lang,text", [
        ("fr", "Rue des Vieux Murs"), ("de", "Strasse der Alten Mauern"),
        ("ru", "Улица Старых Стен"), ("pl", "Ulica Starych Murow"),
    ])
    def test_space_delimited_scripts_get_a_word_rate(self, lang, text):
        rates = error_rates(text, text, lang)
        assert rates["word_segmentable"] is True and rates["wer"] == 0.0

    @pytest.mark.parametrize("lang,text", [
        ("zh-cn", "旧墙街"), ("zh-hk", "舊牆街"), ("ja", "古い壁の通り"), ("ko", "옛 벽 거리"),
    ])
    def test_unsegmented_scripts_get_no_word_rate(self, lang, text):
        """CJK writing puts no spaces between words, so a WER there would
        measure whichever segmenter was chosen, not the text."""
        rates = error_rates(text, text, lang)
        assert rates["word_segmentable"] is False and rates["wer"] is None

    def test_wer_separates_a_wrong_word_from_scattered_noise(self):
        """The diagnostic CER cannot give: these two have similar CER and
        opposite meanings for a reviewer."""
        one_word = error_rates("Rue des Vieux Murs", "Rue des Vieux Chats", "fr")
        scattered = error_rates("Rue des Vieux Murs", "Rue dos Veiux Murs", "fr")
        assert one_word["wer"] < scattered["wer"]


class TestExpectedCalibrationError:
    def test_a_perfectly_calibrated_population_scores_zero(self):
        # 100 predictions at 0.8 confidence, exactly 80 of them correct
        confidences = [0.8] * 100
        outcomes = [True] * 80 + [False] * 20
        assert temper.expected_calibration_error(confidences, outcomes) == pytest.approx(0.0, abs=1e-6)

    def test_systematic_overconfidence_is_measured(self):
        # claims 0.95, right only half the time -> gap of about 0.45
        confidences = [0.95] * 100
        outcomes = [True] * 50 + [False] * 50
        assert temper.expected_calibration_error(confidences, outcomes) == pytest.approx(0.45, abs=1e-3)

    def test_bins_are_weighted_by_population(self):
        """One region in a badly-calibrated bin must not outvote forty in a
        good one."""
        confidences = [0.9] * 40 + [0.1]
        outcomes = [True] * 36 + [False] * 4 + [True]
        assert temper.expected_calibration_error(confidences, outcomes) < 0.05

    def test_no_samples_yields_none_not_zero(self):
        assert temper.expected_calibration_error([], []) is None


class TestBrierScore:
    def test_perfect_confident_predictions_score_zero(self):
        assert temper.brier_score([1.0, 1.0], [True, True]) == pytest.approx(0.0, abs=1e-5)

    def test_confidently_wrong_scores_one(self):
        assert temper.brier_score([1.0, 1.0], [False, False]) == pytest.approx(1.0, abs=1e-5)

    def test_maximum_uncertainty_scores_a_quarter(self):
        assert temper.brier_score([0.5] * 4, [True, False, True, False]) == pytest.approx(0.25)

    def test_it_rewards_discrimination_where_ece_does_not(self):
        """A forecaster always saying the base rate is perfectly calibrated
        and useless; Brier is a proper scoring rule and says so."""
        always_base = temper.brier_score([0.5] * 4, [True, False, True, False])
        discriminating = temper.brier_score([0.9, 0.1, 0.9, 0.1], [True, False, True, False])
        assert discriminating < always_base


class TestFitting:
    def test_a_small_corpus_keeps_the_identity_and_says_why(self):
        c = temper.fit([0.9] * 5, [True, False, True, True, True])
        assert c.temperature == 1.0
        assert any("labelled region" in note for note in c.notes)
        assert c.ece_before is not None      # still measured, just not fitted

    def test_a_single_outcome_class_cannot_be_calibrated_against(self):
        c = temper.fit([0.7] * 40, [True] * 40)
        assert c.temperature == 1.0
        assert any("one class" in note for note in c.notes)

    def test_overconfidence_is_tempered_and_ece_improves(self):
        confidences = [0.97] * 60 + [0.93] * 60
        outcomes = ([True] * 36 + [False] * 24) + ([True] * 30 + [False] * 30)
        c = temper.fit(confidences, outcomes)
        assert c.samples == 120
        assert c.ece_after <= c.ece_before
        if c.temperature != 1.0:
            assert c.temperature > 1.0      # tempering DOWN an overconfident model

    def test_the_identity_map_never_alters_a_confidence(self):
        c = temper.Calibration()
        for value in (0.0, 0.25, 0.5, 0.99, 1.0):
            assert c.apply(value) == value
        assert c.apply(None) is None

    def test_a_fitted_map_keeps_confidences_inside_the_unit_interval(self):
        c = temper.Calibration(temperature=2.5)
        for value in (0.0, 1e-9, 0.5, 1.0):
            assert 0.0 <= c.apply(value) <= 1.0

    def test_temperature_scaling_cannot_reorder_regions(self):
        """One monotone parameter: it changes how confident the verifier
        sounds, never which region a reviewer should open first."""
        c = temper.Calibration(temperature=3.0)
        raw = [0.1, 0.35, 0.6, 0.85, 0.99]
        mapped = [c.apply(v) for v in raw]
        assert mapped == sorted(mapped)


class TestArtifactRoundTrip:
    def test_save_and_load(self, tmp_path):
        original = temper.fit([0.9] * 40 + [0.4] * 40, [True] * 30 + [False] * 10 + [True] * 15 + [False] * 25)
        path = tmp_path / "calib" / "verification-calibration.json"
        temper.save(original, path)
        assert temper.load(path).temperature == original.temperature

    def test_a_missing_artifact_degrades_to_the_identity(self, tmp_path):
        c = temper.load(tmp_path / "nope.json")
        assert c.temperature == 1.0 and c.apply(0.42) == 0.42

    def test_a_corrupt_artifact_degrades_to_the_identity(self, tmp_path):
        path = tmp_path / "bad.json"
        path.write_text("{not json", encoding="utf-8")
        c = temper.load(path)
        assert c.temperature == 1.0
        assert any("unreadable" in note for note in c.notes)
