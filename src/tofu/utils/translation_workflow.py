"""Local-first translation proposal and decision primitives.

Cloud providers intentionally do not live here. This module owns stable
attempt IDs, stale-source evidence, deterministic validators, and the
append-only decision history shared by future providers.
"""

from __future__ import annotations

import hashlib
import re
import time
import uuid
from typing import Any, Dict, Iterable, Optional

from tofu.core.types import InstText


_PLACEHOLDER = re.compile(
    r"(%(?:\d+\$)?[a-zA-Z]|"
    r"\{[^{}\s]+\}|"
    r"\$\{[^{}]+\}|"
    r"\{\{\s*[^{}]+\s*\}\}|"
    r"<\/?[A-Za-z][^>]*>)"
)
_NUMBER = re.compile(r"(?<!\w)[+-]?\d+(?:[.,]\d+)*(?:%|°[CFcf])?(?!\w)")


def source_text_hash(text: Optional[str]) -> str:
    normalized = (text or "").replace("\r\n", "\n")
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def validate_translation(
    source: str,
    target: str,
    *,
    glossary: Optional[Dict[str, str]] = None,
) -> Dict[str, Any]:
    """Return deterministic eligibility evidence; never a quality probability."""
    issues: list[Dict[str, str]] = []
    source_placeholders = sorted(_PLACEHOLDER.findall(source))
    target_placeholders = sorted(_PLACEHOLDER.findall(target))
    if source_placeholders != target_placeholders:
        issues.append({"code": "placeholder_mismatch", "severity": "error"})
    source_numbers = sorted(_NUMBER.findall(source))
    target_numbers = sorted(_NUMBER.findall(target))
    if source_numbers != target_numbers:
        issues.append({"code": "number_mismatch", "severity": "error"})
    if not target.strip():
        issues.append({"code": "empty_target", "severity": "error"})
    if source.strip() and source.strip().casefold() == target.strip().casefold():
        issues.append({"code": "source_identical", "severity": "review"})
    ratio = len(target.strip()) / max(1, len(source.strip()))
    if ratio > 2.5 or ratio < .25:
        issues.append({"code": "expansion_risk", "severity": "review"})
    glossary_misses = []
    for term, required in (glossary or {}).items():
        if term.casefold() in source.casefold() and required.casefold() not in target.casefold():
            glossary_misses.append({"source": term, "required": required})
    if glossary_misses:
        issues.append({"code": "glossary_violation", "severity": "error"})
    return {
        "issues": issues,
        "glossary_misses": glossary_misses,
        "source_placeholders": source_placeholders,
        "target_placeholders": target_placeholders,
        "source_numbers": source_numbers,
        "target_numbers": target_numbers,
        "length_ratio": round(ratio, 4),
        "eligible": not any(issue["severity"] == "error" for issue in issues),
        "review_required": bool(issues),
        "validator_revision": "translation-validators-v1",
    }


def make_tm_attempt(
    inst: InstText,
    target_lang: str,
    *,
    manifest_revision: str,
    glossary_revision: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    suggestion = inst.tm_suggestion
    if not suggestion or not (inst.text or "").strip() or inst.dnt or inst.excluded:
        return None
    target = str(suggestion.get("target_text") or "")
    validation = validate_translation(inst.text or "", target)
    method = str(suggestion.get("method") or "unknown")
    exact_approved = method == "exact" and float(suggestion.get("score") or 0.0) >= 1.0
    review_required = bool(validation["review_required"] or not exact_approved)
    return {
        "attempt_id": uuid.uuid4().hex,
        "provider": "tm_lookup",
        "provider_revision": "visual-tm-v1",
        "created_at": time.time(),
        "source_text_hash": source_text_hash(inst.text),
        "target_lang": target_lang,
        "text": target,
        "raw_score": suggestion.get("score"),
        "raw_score_kind": f"tm_{method}_similarity",
        "tm_record_id": suggestion.get("record_id"),
        "source_asset_id": suggestion.get("source_asset_id"),
        "manifest_revision": manifest_revision,
        "glossary_revision": glossary_revision,
        "validation": validation,
        "state": "review_required" if review_required else "suggested",
        "eligible_auto_accept": bool(exact_approved and validation["eligible"]),
    }


def append_attempt(inst: InstText, attempt: Dict[str, Any]) -> None:
    if any(existing.get("attempt_id") == attempt["attempt_id"] for existing in inst.translation_attempts):
        return
    inst.translation_attempts.append(attempt)


def find_attempt(inst: InstText, attempt_id: str) -> Optional[Dict[str, Any]]:
    return next(
        (attempt for attempt in inst.translation_attempts if attempt.get("attempt_id") == attempt_id),
        None,
    )


def apply_decision(
    inst: InstText,
    *,
    action: str,
    attempt: Optional[Dict[str, Any]],
    text: Optional[str],
    actor: str = "user",
) -> Dict[str, Any]:
    if inst.dnt or inst.excluded:
        raise ValueError("translation decisions cannot target DNT or excluded regions")
    if action not in {"accept", "reject", "edit"}:
        raise ValueError("action must be accept, reject, or edit")
    if attempt is not None and attempt.get("source_text_hash") != source_text_hash(inst.text):
        raise RuntimeError("source text changed after translation proposal")
    if action == "accept":
        if attempt is None:
            raise ValueError("accept requires an attempt")
        chosen = str(attempt.get("text") or "")
        state = "accepted_human"
        attempt["state"] = state
    elif action == "reject":
        if attempt is None:
            raise ValueError("reject requires an attempt")
        chosen = inst.target_text
        state = "rejected"
        attempt["state"] = state
    else:
        if text is None or not text.strip():
            raise ValueError("edit requires non-empty text")
        chosen = text
        state = "edited_human"
        if attempt is not None:
            attempt["state"] = "superseded"
    if action != "reject":
        inst.target_text = chosen
    decision = {
        "state": state,
        "attempt_id": attempt.get("attempt_id") if attempt else None,
        "text": chosen,
        "source_text_hash": source_text_hash(inst.text),
        "actor": actor,
        "decided_at": time.time(),
    }
    inst.translation_decision = decision
    inst.translation_history.append(dict(decision))
    return decision

