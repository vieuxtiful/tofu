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
    unavailable = result.state in {"unavailable", "error"}
    state = "source_text" if source_like else "hallucinated_text" if hallucinated else result.state
    return CleanseVerificationDecision(
        review_required=source_like or hallucinated or unavailable,
        state=state if (source_like or hallucinated or unavailable) else "passed",
        retry=(source_like or hallucinated) and retry_available,
        evidence=evidence,
    )
