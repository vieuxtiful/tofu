"""Narrow, evidence-bearing OCR verification shared by Cicerone and Cleanse.

Two properties this module has to defend (roadmap P1.9, P1.25).

**Absence of evidence is not agreement.** `unavailable`, `error` and
`no_text` all mean the check did not produce a verdict. They must never
reach a caller looking like a pass. The transcription path already treats
them that way; `INDETERMINATE_STATES` below names the set once so a fourth
such state cannot be added later and quietly default to "fine".

**A proposal cannot verify itself.** If PaddleOCR proposes a transcription
and PaddleOCR is then asked whether that transcription is right, the two
observations share every failure mode that matters -- same training data,
same script priors, same systematic confusions -- so agreement between them
is close to no information. That is not a hypothetical: `cicerone` builds a
`PaddleRegionVerifier` to check regions whose primary read may itself have
come from Paddle. Nothing prevented it, and nothing recorded it.

The fix is `verification_authority()`: verification carries the family it
came from, callers say which family proposed, and same-family agreement is
labelled `correlated` so it can be given less weight than it looks like it
deserves. Labelling rather than forbidding is deliberate -- a correlated
check is still worth more than none, it is just not worth what an
independent one is.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from typing import Any, Dict, List, Optional, Protocol, Sequence

from tofu.core.types import BBox, Polygon

# States that carry no verdict. A caller that treats any of these as
# agreement has converted a missing check into a passing one.
INDETERMINATE_STATES = frozenset({"unavailable", "error", "no_text", "no_expectation"})

# Verification authority levels, most to least trustworthy.
AUTHORITY_INDEPENDENT = "independent"   # different OCR family than the proposal
AUTHORITY_CORRELATED = "correlated"     # same family: shared failure modes
AUTHORITY_NONE = "none"                 # no verdict at all


def is_indeterminate(state: Optional[str]) -> bool:
    """True when the verification produced no verdict either way."""
    return state in INDETERMINATE_STATES or state is None


def verification_authority(
    proposal_family: Optional[str],
    verifier_family: Optional[str],
    state: Optional[str] = None,
) -> str:
    """How much independent weight this verification actually carries.

    An unknown proposal family is treated as CORRELATED rather than
    independent: the pessimistic reading is the safe one, because the cost
    of over-trusting a self-check is an accepted wrong transcription, while
    the cost of under-trusting an independent one is a review.
    """
    if state is not None and is_indeterminate(state):
        return AUTHORITY_NONE
    if not proposal_family or not verifier_family:
        return AUTHORITY_CORRELATED
    return (
        AUTHORITY_INDEPENDENT
        if proposal_family.strip().lower() != verifier_family.strip().lower()
        else AUTHORITY_CORRELATED
    )


@dataclass
class OCRVerificationResult:
    state: str
    text: str = ""
    confidence: float = 0.0
    similarity: Optional[float] = None
    backend: str = "paddleocr"
    detections: List[Dict[str, Any]] = field(default_factory=list)
    error: Optional[str] = None
    ## Which OCR family produced this verification, and which produced the
    ## proposal it judged. Kept as data rather than inferred at the call site
    ## so the correlation survives into the manifest -- a decision whose
    ## independence cannot be reconstructed later is not auditable.
    family: str = "paddle"
    model_revision: Optional[str] = None
    proposal_family: Optional[str] = None

    @property
    def authority(self) -> str:
        return verification_authority(self.proposal_family, self.family, self.state)

    @property
    def is_verdict(self) -> bool:
        """False when this result cannot support an accept/reject decision."""
        return not is_indeterminate(self.state)

    def evidence(self) -> Dict[str, Any]:
        return {
            "state": self.state,
            "backend": self.backend,
            "text": self.text,
            "confidence": round(float(self.confidence), 4),
            "similarity": None if self.similarity is None else round(float(self.similarity), 4),
            "detections": self.detections,
            "error": self.error,
            "family": self.family,
            "model_revision": self.model_revision,
            "proposal_family": self.proposal_family,
            "authority": self.authority,
        }


def normalize_for_ocr_agreement(text: Optional[str], language: Optional[str] = None) -> str:
    value = unicodedata.normalize("NFKC", text or "")
    value = re.sub(r"\s+", " ", value).strip()
    if not (language or "").lower().startswith(("ja", "zh", "ko")):
        value = value.casefold()
    return value


def text_similarity(left: Optional[str], right: Optional[str], language: Optional[str] = None) -> float:
    a = normalize_for_ocr_agreement(left, language)
    b = normalize_for_ocr_agreement(right, language)
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return SequenceMatcher(None, a, b).ratio()


class PaddleRegionVerifier:
    """Fresh Paddle crop reads with explicit unavailable/error states."""

    family = "paddle"

    def __init__(
        self,
        languages: Optional[Sequence[str]] = None,
        gpu: bool = False,
        proposal_family: Optional[str] = None,
    ):
        self.languages = list(languages or [])
        self.gpu = gpu
        # Which family produced the text being checked. Supplied by the
        # caller because only the caller knows -- cleanse verifies against
        # the ORIGINAL source read, cicerone against an arbitrated primary,
        # and those have different provenance.
        self.proposal_family = proposal_family

    @staticmethod
    def available() -> bool:
        from tofu.layers.cicerone import PaddleOCRBackend
        return PaddleOCRBackend.is_available()

    def verify_regions(
        self,
        asset: Any,
        regions: Sequence[BBox],
        expected_texts: Optional[Sequence[Optional[str]]] = None,
        language: Optional[str] = None,
    ) -> List[OCRVerificationResult]:
        expected = list(expected_texts or [None] * len(regions))
        if len(expected) != len(regions):
            raise ValueError("expected_texts must match regions")
        if not self.available():
            return [self._result("unavailable") for _ in regions]
        try:
            from tofu.layers.cicerone import PaddleOCRBackend
            langs = self.languages or ([language] if language else None)
            backend = PaddleOCRBackend(languages=langs, gpu=self.gpu)
            reads = backend.detect_in_regions(asset, list(regions))
        except Exception as exc:
            return [
                self._result("error", error=f"{type(exc).__name__}: {exc}"[:300])
                for _ in regions
            ]
        results: List[OCRVerificationResult] = []
        for index, region in enumerate(regions):
            detections = reads[index] if index < len(reads) else []
            if not detections:
                results.append(self._result("no_text", similarity=0.0))
                continue
            ordered = sorted(detections, key=lambda d: (min(p[1] for p in d.polygon), min(p[0] for p in d.polygon)))
            text = " ".join(d.text.strip() for d in ordered if d.text.strip())
            confidence = max((float(d.confidence or 0.0) for d in detections), default=0.0)
            similarity = text_similarity(expected[index], text, language) if expected[index] else None
            if similarity is None:
                # Nothing was claimed about this region, so nothing was
                # checked. Previously this fell through to "disagree", which
                # reported a contradiction that had never been tested --
                # noise pointing the opposite way from the usual failure,
                # but a false verdict either way.
                state = "no_expectation"
            else:
                state = "agree" if similarity >= 0.85 else "disagree"
            results.append(self._result(
                state,
                text=text,
                confidence=confidence,
                similarity=similarity,
                detections=[{
                    "text": d.text,
                    "confidence": round(float(d.confidence or 0.0), 4),
                    "polygon": [list(point) for point in d.polygon],
                } for d in detections],
            ))
        return results

    def _result(self, state: str, **kw: Any) -> OCRVerificationResult:
        """Stamp every result with its provenance, including the failures.

        Constructing results anywhere but here is how an `unavailable` ends
        up with no family recorded, which in turn makes `authority` read
        CORRELATED when the real answer is "nothing happened".
        """
        return OCRVerificationResult(
            state,
            family=self.family,
            proposal_family=self.proposal_family,
            **kw,
        )


@dataclass
class RegionRequest:
    """One region to verify: crop, geometry, and optional source text."""
    crop: Any  # ndarray
    bbox: BBox
    polygon: Optional[Polygon] = None
    language_hint: Optional[str] = None
    source_text: Optional[str] = None  # for similarity comparison


class OCRRegionVerifier(Protocol):
    """Protocol for fresh, deduplicated OCR region verification."""

    def verify_regions(
        self,
        asset: Any,
        regions: Sequence[RegionRequest],
    ) -> Sequence[OCRVerificationResult]: ...

