import pytest

from tofu.core.types import BBox, InstText, TextManifest
from tofu.utils.manifest_store import _dict_to_manifest, _manifest_to_dict
from tofu.utils.translation_workflow import (
    append_attempt,
    apply_decision,
    make_tm_attempt,
    source_text_hash,
    validate_translation,
)


def _inst():
    return InstText(
        "r1", BBox(1, 2, 30, 12), text="Total: {count} 20%",
        tm_suggestion={
            "target_text": "Total: {count} 20%",
            "score": 1.0,
            "method": "exact",
            "source_asset_id": "old",
            "record_id": 7,
        },
    )


def test_translation_validator_preserves_placeholders_and_numbers():
    good = validate_translation("Total: {count} 20%", "Total: {count} 20%")
    assert good["eligible"] is True
    bad = validate_translation("Total: {count} 20%", "Total: 25%")
    assert bad["eligible"] is False
    assert {issue["code"] for issue in bad["issues"]} >= {
        "placeholder_mismatch", "number_mismatch",
    }


def test_tm_attempt_is_proposal_with_stale_source_evidence():
    inst = _inst()
    attempt = make_tm_attempt(inst, "fr", manifest_revision="rev-1")
    assert attempt is not None
    assert attempt["provider"] == "tm_lookup"
    assert attempt["source_text_hash"] == source_text_hash(inst.text)
    assert attempt["eligible_auto_accept"] is True
    assert inst.target_text is None


def test_accept_and_edit_have_separate_append_only_history():
    inst = _inst()
    attempt = make_tm_attempt(inst, "fr", manifest_revision="rev-1")
    assert attempt is not None
    append_attempt(inst, attempt)
    accepted = apply_decision(inst, action="accept", attempt=attempt, text=None)
    assert accepted["state"] == "accepted_human"
    assert inst.target_text == attempt["text"]
    edited = apply_decision(inst, action="edit", attempt=attempt, text="Montant : {count} 20%")
    assert edited["state"] == "edited_human"
    assert attempt["state"] == "superseded"
    assert len(inst.translation_history) == 2


def test_source_change_makes_attempt_stale():
    inst = _inst()
    attempt = make_tm_attempt(inst, "fr", manifest_revision="rev-1")
    assert attempt is not None
    inst.text = "Changed"
    with pytest.raises(RuntimeError, match="source text changed"):
        apply_decision(inst, action="accept", attempt=attempt, text=None)


def test_translation_contract_round_trips_in_manifest():
    inst = _inst()
    attempt = make_tm_attempt(inst, "fr", manifest_revision="rev-1")
    assert attempt is not None
    append_attempt(inst, attempt)
    apply_decision(inst, action="accept", attempt=attempt, text=None)
    manifest = TextManifest("asset", 1, [inst])
    restored = _dict_to_manifest(_manifest_to_dict(manifest)).instances[0]
    assert restored.translation_attempts == inst.translation_attempts
    assert restored.translation_decision == inst.translation_decision
    assert restored.translation_history == inst.translation_history

