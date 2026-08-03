## 🍢 n-gram scoring seam + hypothesis LM signal (no model artifact needed)
import pytest

from tofu.core.types import BBox, OCRObservation
from tofu.layers.language_models import KenLMScoringProvider
from tofu.layers.ocr_arbitration import (
    HYPOTHESIS_WEIGHTS,
    RegionHypothesis,
    _language_model_score,
    score_hypothesis,
)


def obs(oid, text, conf, backend="easyocr", script="Latn"):
    return OCRObservation(
        observation_id=oid, backend=backend, backend_revision="1",
        pass_tag="p", text=text, raw_confidence=conf,
        bbox=BBox(x=0, y=0, width=100, height=20), detected_script=script,
    )


class _StubScorer:
    """Stands in for a side-loaded model: known strings score well."""

    def __init__(self, table):
        self.table = table
        self.seen = []

    def score(self, text, script=None):
        self.seen.append((text, script))
        return self.table.get(text)


class TestKenLMProviderWithoutAModel:
    def test_absent_model_reports_unready_rather_than_failing(self):
        status = KenLMScoringProvider(model_dir="").status()
        assert status["ready"] is False
        assert status["families"] == {"latin": False, "cjk": False}
        assert "TOFU_KENLM_DIR" in status["reason"]

    def test_absent_model_scores_nothing(self):
        assert KenLMScoringProvider(model_dir="").score("MAIN STREET", "Latn") is None

    def test_empty_text_scores_nothing(self):
        assert KenLMScoringProvider(model_dir="").score("   ", "Latn") is None

    def test_scripts_route_to_their_own_model_family(self):
        # CJK is scored character by character and Latin word by word; a
        # single model spanning both would dilute exactly the statistics
        # the ranking depends on.
        assert KenLMScoringProvider._family("Hani") == "cjk"
        assert KenLMScoringProvider._family("Hang") == "cjk"
        assert KenLMScoringProvider._family("Jpan") == "cjk"
        assert KenLMScoringProvider._family("Latn") == "latin"
        assert KenLMScoringProvider._family("Cyrl") == "latin"
        assert KenLMScoringProvider._family(None) == "latin"


class TestLanguageModelSignal:
    def test_no_model_means_no_signal_not_a_zero(self):
        # None is what lets the weight renormalize away. A 0.0 would score
        # every candidate as maximally implausible on every host that has
        # not side-loaded a model.
        assert _language_model_score([obs("a", "MAIN STREET", 0.9)]) is None

    def test_a_plausible_reading_scores_above_an_implausible_one(self):
        scorer = _StubScorer({"MAIN STREET": -1.0, "MAIN STBEET": -5.5})
        good = _language_model_score([obs("a", "MAIN STREET", 0.9)], scorer=scorer)
        bad = _language_model_score([obs("b", "MAIN STBEET", 0.9)], scorer=scorer)
        assert good > bad

    def test_the_best_reading_in_a_hypothesis_is_the_one_that_counts(self):
        scorer = _StubScorer({"MAIN STBEET": -5.5, "MAIN STREET": -1.0})
        both = _language_model_score(
            [obs("a", "MAIN STBEET", 0.9), obs("b", "MAIN STREET", 0.4)], scorer=scorer
        )
        assert both == pytest.approx(_language_model_score(
            [obs("b", "MAIN STREET", 0.4)], scorer=scorer))

    def test_the_signal_is_clamped_to_the_unit_interval(self):
        scorer = _StubScorer({"x": -99.0, "y": 5.0})
        assert _language_model_score([obs("a", "x", 0.9)], scorer=scorer) == 0.0
        assert _language_model_score([obs("b", "y", 0.9)], scorer=scorer) == 1.0

    def test_unscorable_observations_are_skipped_not_counted_as_bad(self):
        scorer = _StubScorer({"KNOWN": -1.0})
        only_known = _language_model_score([obs("a", "KNOWN", 0.9)], scorer=scorer)
        with_unknown = _language_model_score(
            [obs("a", "KNOWN", 0.9), obs("b", "UNSCORABLE", 0.9)], scorer=scorer)
        assert with_unknown == only_known


class TestHypothesisScoringWithTheNewWeight:
    def test_the_weight_is_declared(self):
        assert HYPOTHESIS_WEIGHTS["language_model"] == pytest.approx(0.10)

    def test_scoring_without_a_model_leaves_the_signal_unmeasured(self):
        # "not measured" and "scored zero" must stay distinguishable, which
        # is what the None in score_breakdown carries.
        hypothesis = RegionHypothesis(
            hypothesis_id="h1", member_ids=["a", "b"],
            union_bbox=BBox(x=0, y=0, width=100, height=20),
            observations=[obs("a", "RUE", 0.9), obs("b", "RUE", 0.8)],
        )
        scored = score_hypothesis(hypothesis)
        assert scored["score_breakdown"]["language_model"] is None
        assert scored["selected_text"] == "RUE"
