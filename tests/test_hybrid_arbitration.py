from unittest.mock import patch

from tofu.core.types import BBox, InstText
from tofu.layers.cicerone import RawDetection, hybrid_audit


def _region(text="御獄-", confidence=.68):
    return InstText("r1", BBox(10, 10, 50, 24), text=text,
                    confidence=confidence, detected_language="ja")


class _Paddle:
    PADDLE_TO_TOFU = {"ja": "ja"}
    @staticmethod
    def is_available():
        return True

    def __init__(self, *args, **kwargs):
        pass

    def detect_in_regions(self, asset, boxes):
        return [[RawDetection([(10, 10), (60, 10), (60, 34), (10, 34)], "御嶽", .984, "ja")]]


def test_hybrid_records_and_accepts_evidence_backed_japanese_artifact():
    inst = _region()
    with patch("tofu.layers.cicerone.PaddleOCRBackend", _Paddle), \
         patch("tofu.layers.cicerone._no_terminal_dash_ink", return_value=True):
        assert hybrid_audit("any.jpg", [inst]) == 1
    assert inst.text == "御嶽"
    assert inst.recognition_history and inst.recognition_history[-1]["accepted"] is True
    assert inst.recognition_history[-1]["candidate_text"] == "御嶽"


def test_hybrid_preserves_a_real_latin_hyphen():
    inst = InstText("r1", BBox(10, 10, 50, 24), text="Hida-osaka",
                    confidence=.5, detected_language="en")
    assert hybrid_audit("any.jpg", [inst]) == 0
    assert inst.text == "Hida-osaka"
    assert inst.recognition_history is None


def test_hybrid_fails_open_when_crop_mask_is_ambiguous():
    inst = _region()
    with patch("tofu.layers.cicerone.PaddleOCRBackend", _Paddle), \
         patch("tofu.layers.cicerone._no_terminal_dash_ink", return_value=None):
        assert hybrid_audit("any.jpg", [inst]) == 0
    assert inst.text == "御獄-"
    assert inst.recognition_history[-1]["accepted"] is False
