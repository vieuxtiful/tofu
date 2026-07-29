"""Optional local language and diacritic model adapters.

These adapters never download weights or make network calls.  They expose a
small, versioned evidence contract so Cicerone can remain conservative when a
host has not provisioned a model.
"""
from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
from typing import Dict, Iterable, Optional, Protocol, Sequence


@dataclass(frozen=True)
class LanguageEvidence:
    language: Optional[str]
    confidence: float
    provider: str
    model_version: Optional[str]
    reason: str


class LanguageModelProvider(Protocol):
    def identify(self, texts: Sequence[str]) -> LanguageEvidence: ...
    def status(self) -> Dict[str, object]: ...


_FASTTEXT_TO_TOFU = {
    "en": "en", "fr": "fr", "es": "es", "pl": "pl", "ro": "ro",
    "vi": "vi", "de": "de", "it": "it", "pt": "pt", "nl": "nl",
    "ru": "ru", "uk": "uk", "ja": "ja", "ko": "ko", "th": "th",
    "zh": "zh-cn",
}


class FastTextLanguageProvider:
    """Side-loaded fastText LID model, usually ``lid.176.ftz``.

    The provider deliberately returns unknown on absent/invalid models and on
    short or ambiguous evidence.  A false English label is worse than no
    label because it constrains later OCR passes to the wrong charset.
    """
    provider_id = "fasttext-lid"

    def __init__(self, model_path: Optional[str] = None, min_confidence: float = .72):
        self.model_path = Path(model_path or os.environ.get("TOFU_FASTTEXT_LID", ""))
        self.min_confidence = min_confidence
        self._model = None
        self._failed: Optional[str] = None

    def _load(self):
        if self._model is not None or self._failed:
            return self._model
        if not self.model_path or not self.model_path.is_file():
            self._failed = "no readable model configured in TOFU_FASTTEXT_LID"
            return None
        try:
            import fasttext  # type: ignore
            self._model = fasttext.load_model(str(self.model_path))
        except Exception as exc:  # optional native extension/model errors
            self._failed = f"model load failed: {type(exc).__name__}"
        return self._model

    def identify(self, texts: Sequence[str]) -> LanguageEvidence:
        text = " ".join(t.strip() for t in texts if t and t.strip())
        if len(text) < 3:
            return LanguageEvidence(None, 0.0, self.provider_id, None, "insufficient text")
        model = self._load()
        if model is None:
            return LanguageEvidence(None, 0.0, self.provider_id, None, self._failed or "unavailable")
        labels, probs = model.predict(text.replace("\n", " "), k=1)
        raw = labels[0].removeprefix("__label__") if labels else ""
        confidence = float(probs[0]) if probs else 0.0
        lang = _FASTTEXT_TO_TOFU.get(raw)
        if lang is None or confidence < self.min_confidence:
            return LanguageEvidence(None, confidence, self.provider_id, self.model_path.name, "ambiguous prediction")
        return LanguageEvidence(lang, confidence, self.provider_id, self.model_path.name, "model prediction")

    def status(self) -> Dict[str, object]:
        ready = bool(self.model_path and self.model_path.is_file())
        return {
            "id": self.provider_id, "available": ready, "ready": ready,
            "version": self.model_path.name if ready else None,
            "reason": None if ready else "no readable model configured in TOFU_FASTTEXT_LID",
        }


class DiacriticRestorationProvider:
    """Versioned optional ONNX seam for conservative accent restoration.

    An artifact is intentionally not bundled.  Until a compatible local model
    is configured, callers get no candidate rather than a fabricated accent.
    The public contract lets a later artifact be enabled without changing OCR
    provenance or review behavior.
    """
    provider_id = "onnx-diacritic-restoration"

    def __init__(self, model_path: Optional[str] = None):
        self.model_path = Path(model_path or os.environ.get("TOFU_DIACRITIC_MODEL", ""))

    def status(self) -> Dict[str, object]:
        ready = bool(self.model_path and self.model_path.is_file())
        return {
            "id": self.provider_id, "available": ready, "ready": ready,
            "version": self.model_path.name if ready else None,
            "reason": None if ready else "no readable model configured in TOFU_DIACRITIC_MODEL",
        }

    def candidates(self, text: str, language: Optional[str]) -> Iterable[str]:
        # A model artifact defines the tokenizer/output vocabulary.  Returning
        # no candidate is the only safe fallback; Cicerone's existing
        # dictionary/OCR arbitration remains responsible for review hints.
        return ()


_default_provider: LanguageModelProvider = FastTextLanguageProvider()


def get_language_provider() -> LanguageModelProvider:
    return _default_provider


def set_language_provider(provider: LanguageModelProvider) -> None:
    global _default_provider
    _default_provider = provider

