import json
from pathlib import Path

import pytest

from tofu.layers import menu, wasabi
from tofu.utils.correction_resources import (
    CorrectionResourceError,
    diacritic_entries,
    _payload_checksum,
    gazetteer_entries,
    load_correction_file,
    load_correction_resource,
    phrase_entries,
    variant_pairs,
)


def _resource(tmp_path, entries, kind="phrase-forms", name="phrases.json"):
    data = {
        "id": "test.phrases", "kind": kind, "locale": "mul", "schema_version": "1",
        "data_version": "1.0.0", "checksum": _payload_checksum(entries),
        "provenance": {}, "license": "CC0-1.0", "entries": entries,
    }
    path = tmp_path / name
    path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    return load_correction_file(path)


class TestPhraseResource:
    def test_packaged_french_phrases_expose_tokens_and_folded_anchors(self):
        entries = phrase_entries(load_correction_resource("savor/latin_phrases-1.0.0.json"))
        de_la = next(entry for entry in entries if entry["phrase"] == "de la")
        assert de_la["tokens"] == ("de", "la") and de_la["language"] == "fr"
        # The fold is comparison-only: the stored token keeps its accent.
        a_la = next(entry for entry in entries if entry["phrase"] == "à la")
        assert a_la["tokens"] == ("à", "la") and a_la["folded"] == ("a", "la")

    def test_single_token_phrase_is_rejected_for_having_no_anchor(self, tmp_path):
        with pytest.raises(CorrectionResourceError, match="at least two tokens"):
            phrase_entries(_resource(tmp_path, [{"phrase": "la", "language": "fr"}]))

    def test_duplicate_folded_phrase_is_rejected(self, tmp_path):
        with pytest.raises(CorrectionResourceError, match="duplicate folded phrase"):
            phrase_entries(_resource(tmp_path, [
                {"phrase": "de la", "language": "fr"},
                {"phrase": "DE LÀ", "language": "fr"},
            ]))

    def test_irregular_spacing_is_rejected(self, tmp_path):
        with pytest.raises(CorrectionResourceError, match="single-space separated"):
            phrase_entries(_resource(tmp_path, [{"phrase": "de  la", "language": "fr"}]))

    def test_wrong_kind_is_rejected(self, tmp_path):
        with pytest.raises(CorrectionResourceError, match="expected phrase-forms"):
            phrase_entries(_resource(tmp_path, [{"text": "x", "language": "fr"}], kind="gazetteer"))


def test_packaged_menu_resources_preserve_public_tables():
    places = load_correction_resource("menu/known_places-1.0.0.json")
    signage = load_correction_resource("menu/known_signage-1.0.0.json")

    assert places.data_version == "1.0.0"
    assert places.provenance["method"]
    assert places.checksum.startswith("sha256:")
    assert gazetteer_entries(places) == menu.KNOWN_PLACES
    assert gazetteer_entries(signage) == menu.KNOWN_SIGNAGE


def test_packaged_wasabi_resource_preserves_public_mapping():
    resource = load_correction_resource(
        "wasabi/simplified_to_japanese-1.0.0.json"
    )

    assert resource.resource_id == "wasabi.simplified-to-japanese"
    assert variant_pairs(resource) == wasabi.SIMPLIFIED_TO_JAPANESE


def test_packaged_latin_diacritic_resource_is_audited_stored_data():
    resource = load_correction_resource("savor/latin_diacritics-1.0.0.json")
    assert resource.audit_identity()["id"] == "savor.latin-diacritics"
    entries = diacritic_entries(resource)
    assert {entry["folded"] for entry in entries} >= {"republique", "cafe", "eglise", "theatre"}


def _diacritic_resource(entries):
    return {
        "id": "test.diacritics", "kind": "diacritic-forms", "locale": "mul",
        "schema_version": "1", "data_version": "1.0.0",
        "checksum": _payload_checksum(entries), "provenance": {}, "license": "CC0-1.0",
        "entries": entries,
    }


def test_diacritic_resource_rejects_bad_fold_and_length_and_duplicates(tmp_path):
    for entries, pattern in [
        ([{"folded": "wrong", "text": "Café", "language": "fr"}], "does not match"),
        ([{"folded": "e", "text": "e\u0301", "language": "fr"}], "preserve character length"),
        ([
            {"folded": "cafe", "text": "Café", "language": "fr"},
            {"folded": "cafe", "text": "Café", "language": "fr"},
        ], "duplicate folded"),
    ]:
        path = tmp_path / "invalid-diacritic.json"
        path.write_text(json.dumps(_diacritic_resource(entries)), encoding="utf-8")
        with pytest.raises(CorrectionResourceError, match=pattern):
            diacritic_entries(load_correction_file(path))


def test_resource_rejects_tampered_entries(tmp_path):
    source = Path("src/tofu/resources/corrections/menu/known_places-1.0.0.json")
    data = json.loads(source.read_text(encoding="utf-8"))
    data["entries"][0]["text"] = "tampered"
    path = tmp_path / "tampered.json"
    path.write_text(json.dumps(data), encoding="utf-8")

    with pytest.raises(CorrectionResourceError, match="checksum mismatch"):
        load_correction_file(path)


def test_resource_rejects_unsupported_schema(tmp_path):
    data = {
        "id": "test",
        "kind": "gazetteer",
        "locale": "en",
        "schema_version": "999",
        "data_version": "1.0.0",
        "checksum": "sha256:irrelevant",
        "provenance": {},
        "license": "MIT",
        "entries": [],
    }
    path = tmp_path / "future.json"
    path.write_text(json.dumps(data), encoding="utf-8")

    with pytest.raises(CorrectionResourceError, match="unsupported schema_version"):
        load_correction_file(path)


def test_resource_paths_cannot_escape_package():
    with pytest.raises(CorrectionResourceError, match="safe relative path"):
        load_correction_resource("../outside.json")
