"""Narrow, evidence-bearing OCR verification shared by Cicerone and Cleanse."""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from typing import Any, Dict, List, Optional, Protocol, Sequence

from tofu.core.types import BBox, Polygon


@dataclass
class OCRVerificationResult:
    state: str
    text: str = ""
    confidence: float = 0.0
    similarity: Optional[float] = None
    backend: str = "paddleocr"
    detections: List[Dict[str, Any]] = field(default_factory=list)
    error: Optional[str] = None

    def evidence(self) -> Dict[str, Any]:
        return {
            "state": self.state,
            "backend": self.backend,
            "text": self.text,
            "confidence": round(float(self.confidence), 4),
            "similarity": None if self.similarity is None else round(float(self.similarity), 4),
            "detections": self.detections,
            "error": self.error,
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

    def __init__(self, languages: Optional[Sequence[str]] = None, gpu: bool = False):
        self.languages = list(languages or [])
        self.gpu = gpu

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
            return [OCRVerificationResult("unavailable") for _ in regions]
        try:
            from tofu.layers.cicerone import PaddleOCRBackend
            langs = self.languages or ([language] if language else None)
            backend = PaddleOCRBackend(languages=langs, gpu=self.gpu)
            reads = backend.detect_in_regions(asset, list(regions))
        except Exception as exc:
            return [
                OCRVerificationResult("error", error=f"{type(exc).__name__}: {exc}"[:300])
                for _ in regions
            ]
        results: List[OCRVerificationResult] = []
        for index, region in enumerate(regions):
            detections = reads[index] if index < len(reads) else []
            if not detections:
                results.append(OCRVerificationResult("no_text", similarity=0.0))
                continue
            ordered = sorted(detections, key=lambda d: (min(p[1] for p in d.polygon), min(p[0] for p in d.polygon)))
            text = " ".join(d.text.strip() for d in ordered if d.text.strip())
            confidence = max((float(d.confidence or 0.0) for d in detections), default=0.0)
            similarity = text_similarity(expected[index], text, language) if expected[index] else None
            state = "agree" if similarity is not None and similarity >= 0.85 else "disagree"
            results.append(OCRVerificationResult(
                state=state,
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

