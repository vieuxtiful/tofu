## 🍢 ordinal/administrative abbreviation review course
from pathlib import Path

import pytest

from tofu.core.types import BBox, InstText
from tofu.layers import ordinal


def inst(text, conf=0.535, box=(239, 162, 75, 47)):
    x, y, w, h = box
    return InstText(id="r1", bounding_box=BBox(x=x, y=y, width=w, height=h),
                    text=text, confidence=conf)


def steps(i):
    return [s for s in ((i.ocr_correction or {}).get("steps") or [])
            if s.get("course") == "ordinal_abbreviation"]


class TestOrdinalProposals:
    def test_shattered_plate_reading_is_proposed(self):
        # quai-des-orfevres as first reported: the italic serifed '1' read as
        # 'f', the word boundary lost, the raised 't' read as '!'.
        i = inst("ferArr !")
        assert ordinal.propose(None, [i], language="fr") == 1
        assert steps(i)[0]["candidate_text"] == "1er Arr\u1d57"

    def test_a_confident_read_can_still_be_incomplete(self):
        # regression guard for a gate I had to remove: once the clipped-glyph
        # rescue recovers the leading digit this region reads '1erArr' at
        # 0.944 -- confident, and still missing the word boundary AND the
        # raised terminal. Confidence describes the glyphs that WERE emitted.
        i = inst("1erArr", conf=0.944)
        assert ordinal.propose(None, [i], language="fr") == 1
        assert steps(i)[0]["candidate_text"] == "1er Arr\u1d57"

    def test_never_mutates_the_text(self):
        i = inst("ferArr !")
        ordinal.propose(None, [i], language="fr")
        assert i.text == "ferArr !"
        assert steps(i)[0]["applied"] is False

    def test_correct_plate_proposes_nothing(self):
        i = inst("1er Arr\u1d57", conf=0.99)
        assert ordinal.propose(None, [i], language="fr") == 0

    def test_unrelated_text_proposes_nothing(self):
        for text in ("QUAI", "DES", "ORF\u00c8VRES", "RUE DES MARTYRS"):
            i = inst(text, conf=0.999)
            assert ordinal.propose(None, [i], language="fr") == 0

    def test_scoped_to_french_projects(self):
        i = inst("ferArr !")
        assert ordinal.propose(None, [i], language="it") == 0
        assert i.ocr_correction is None

    def test_second_looks_rejected_candidate_also_matches(self):
        # second_look offered 'JerArr !' at 0.511 and was right to reject it;
        # the structural course reaches the same reading from either spelling.
        i = inst("JerArr !", conf=0.511)
        assert ordinal.propose(None, [i], language="fr") == 1
        assert steps(i)[0]["candidate_text"] == "1er Arr\u1d57"

    def test_proposal_carries_resource_identity(self):
        i = inst("ferArr !")
        ordinal.propose(None, [i], language="fr")
        resource = steps(i)[0]["ordinal_evidence"]["resource"]
        assert resource["id"] == "ordinal.french-administrative"
        assert resource["data_version"] == "1.0.0"
        assert resource["checksum"].startswith("sha256:")


class TestOrdinalResource:
    def test_resource_is_loadable_and_checksummed(self):
        from tofu.utils.correction_resources import load_correction_resource

        res = load_correction_resource(ordinal.FRENCH_ORDINALS)
        assert res.locale == "fr"
        assert res.kind == "ordinal-abbreviation"
        assert any(e["ordinal"] == "1er" for e in res.entries)
