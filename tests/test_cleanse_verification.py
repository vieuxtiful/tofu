from tofu.layers.cleanse_verification import assess_residual


class Result:
    def __init__(self, state, confidence, similarity=None):
        self.state, self.confidence, self.similarity = state, confidence, similarity
    def evidence(self):
        return {"state": self.state}


def test_residual_source_text_requests_retry_when_available():
    decision = assess_residual(Result("agree", .8, .9), 0, True)
    assert decision.review_required and decision.retry
    assert decision.state == "source_text"
    assert decision.evidence["attempt"] == 0


def test_unavailable_residual_check_requires_review_without_retry():
    decision = assess_residual(Result("unavailable", 0), 1, False)
    assert decision.review_required and not decision.retry
    assert decision.state == "unavailable"
