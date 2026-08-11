"""Cleanse-only verification decisions, independent of render QA.

The result is intentionally a small typed policy contract.  Cleanse consumes
it while ``verify`` remains the public post-render report facade.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict


@dataclass(frozen=True)
class CleanseVerificationDecision:
    review_required: bool
    state: str
    retry: bool
    evidence: Dict[str, Any]


def assess_residual(result: Any, attempt: int, retry_available: bool) -> CleanseVerificationDecision:
    evidence = dict(result.evidence())
    evidence["attempt"] = attempt
    source_like = result.state == "agree" and result.confidence >= .50 and (result.similarity or 0.0) >= .30
    hallucinated = result.state == "disagree" and result.confidence >= .70

    # `no_text` is the one indeterminate state that IS good news here: the
    # verifier read the cleansed crop and found nothing, which is exactly
    # what a successful erase looks like. Every other no-verdict state means
    # the check did not happen, and a check that did not happen cannot clear
    # a region -- see INDETERMINATE_STATES in ocr_verification.
    unavailable = result.state in {"unavailable", "error", "no_expectation"}

    # A correlated verifier is the same OCR family that proposed the text it
    # is now judging. Its agreement is weak evidence and its silence is
    # weaker still, so a clean result from one does not earn auto-accept.
    #
    # Computed from the families directly rather than read off
    # `result.authority`. That property is state-aware and reports NONE for
    # every indeterminate state -- correct for transcription, wrong here,
    # because `no_text` is precisely the state this check cares about and is
    # a real verdict in a residual context. Asking the state-free helper
    # keeps the two meanings of `no_text` from colliding.
    from tofu.layers.ocr_verification import AUTHORITY_CORRELATED, verification_authority

    correlated = verification_authority(
        getattr(result, "proposal_family", None),
        getattr(result, "family", None),
    ) == AUTHORITY_CORRELATED
    weak_clearance = correlated and not (source_like or hallucinated or unavailable)

    state = "source_text" if source_like else "hallucinated_text" if hallucinated else result.state
    blocking = source_like or hallucinated or unavailable
    return CleanseVerificationDecision(
        review_required=blocking or weak_clearance,
        state=(
            state if blocking
            else "correlated_verifier" if weak_clearance
            else "passed"
        ),
        retry=(source_like or hallucinated) and retry_available,
        evidence=evidence,
    )
