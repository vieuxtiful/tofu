## 🍢 OCR correction memory: remember what a region SAID, not what it meant
from tofu.core.types import BBox, InstText, TextManifest
from tofu.layers import memory


def inst(rid, text, original=None, history=None, dnt=False, lang="fr"):
    i = InstText(id=rid, bounding_box=BBox(x=0, y=0, width=60, height=20),
                 text=text, confidence=0.9, detected_language=lang, dnt=dnt)
    if original is not None:
        i.ocr_correction = {"steps": [{"course": "savor", "applied": True,
                                       "original_text": original}]}
    if history is not None:
        i.recognition_history = history
    return i


def manifest(instances, asset_id="a1", src="fr"):
    return TextManifest(asset_id=asset_id, total_regions=len(instances),
                        instances=instances, src_lang=src)


class TestRememberCorrections:
    def test_records_a_region_corrected_away_from_the_ocr_read(self):
        m = manifest([inst("r1", "DES", original="ES")])
        drafts = memory.remember_corrections(m)
        assert len(drafts) == 1
        assert drafts[0]["ocr_text"] == "ES"
        assert drafts[0]["corrected_text"] == "DES"
        # keyed on the MISREAD: that is what a future run will produce again
        assert drafts[0]["normalized_ocr_text"] == "es"

    def test_uncorrected_regions_are_not_recorded(self):
        m = manifest([inst("r1", "QUAI", original="QUAI")])
        assert memory.remember_corrections(m) == []

    def test_region_with_no_history_is_not_recorded(self):
        # nothing to compare against is not the same as nothing changed
        m = manifest([inst("r1", "QUAI")])
        assert memory.remember_corrections(m) == []

    def test_falls_back_to_recognition_history(self):
        m = manifest([inst("r1", "nos", history=[{"stage": "edge_rescue", "text": "os"}])])
        drafts = memory.remember_corrections(m)
        assert len(drafts) == 1 and drafts[0]["ocr_text"] == "os"

    def test_earliest_read_wins_over_later_steps(self):
        i = inst("r1", "RÉPUBLIQUE")
        i.ocr_correction = {"steps": [
            {"course": "savor", "applied": True, "original_text": "REPUBUQUE"},
            {"course": "menu", "applied": True, "original_text": "REPUBLIQUE"},
        ]}
        drafts = memory.remember_corrections(manifest([i]))
        assert drafts[0]["ocr_text"] == "REPUBUQUE"


class TestRecallCorrections:
    POOL = [{"id": 7, "asset_id": "a0", "ocr_text": "ES",
             "normalized_ocr_text": "es", "corrected_text": "DES", "phash": None}]

    def test_exact_misread_is_offered(self):
        m = manifest([inst("r1", "ES")])
        out = memory.recall_corrections(m, self.POOL)
        assert out["r1"]["corrected_text"] == "DES"
        assert out["r1"]["method"] == "exact"
        assert out["r1"]["source_asset_id"] == "a0"

    def test_nothing_offered_without_candidates(self):
        assert memory.recall_corrections(manifest([inst("r1", "ES")]), []) == {}

    def test_text_already_correct_is_not_re_proposed(self):
        # the stored correction IS the text in hand; offering it is noise
        m = manifest([inst("r1", "DES")])
        assert memory.recall_corrections(m, self.POOL) == {}

    def test_dnt_regions_are_skipped(self):
        m = manifest([inst("r1", "ES", dnt=True)])
        assert memory.recall_corrections(m, self.POOL) == {}

    def test_recall_never_mutates_the_instance(self):
        i = inst("r1", "ES")
        memory.recall_corrections(manifest([i]), self.POOL)
        assert i.text == "ES"

    def test_unrelated_text_is_not_matched(self):
        m = manifest([inst("r1", "ORFÈVRES")])
        assert memory.recall_corrections(m, self.POOL) == {}
