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


## Scripts whose text is scored CHARACTER by character rather than word by
## word. CJK is written without word spaces, and its OCR errors are
## character substitutions rather than boundary errors, so a character
## n-gram is both simpler (no segmenter to depend on) and better matched
## to the failure it has to rank. Mixed kana/kanji/latin falls out of the
## same treatment for free -- every codepoint is a token.
_CHARACTER_SCRIPTS = {"Hani", "Hang", "Hira", "Kana", "Jpan", "Hans", "Hant"}


class KenLMScoringProvider:
    """Side-loaded KenLM n-gram model for OCR candidate rescoring.

    A CRNN+CTC recognizer decodes character by character with no notion of
    whether the reading it produced is a plausible string -- savor.py's
    module note says exactly this, and it is why the glyph-confusion
    courses have to reason from pixels alone. An n-gram model is the
    cheapest thing that supplies the missing signal: it cannot read the
    image, but it can say that MAIN STREET is a likelier string than MAIN
    STBEET.

    One engine, one model file per script family, routed by the script
    already detected upstream. A single model spanning Latin and CJK is
    deliberately NOT supported: the vocabularies are disjoint, and mixing
    them dilutes exactly the n-gram statistics the ranking depends on.

    Like every other provider here it never downloads anything and never
    makes a network call. Absent ``TOFU_KENLM_DIR``, an unreadable model,
    or a missing ``kenlm`` package all resolve to "no score", and callers
    treat that as one signal being unavailable rather than as evidence.
    """
    provider_id = "kenlm-ngram"

    def __init__(self, model_dir: Optional[str] = None):
        self.model_dir = Path(model_dir or os.environ.get("TOFU_KENLM_DIR", ""))
        self._models: Dict[str, object] = {}
        self._failed: Dict[str, str] = {}

    @staticmethod
    def _family(script: Optional[str]) -> str:
        return "cjk" if script in _CHARACTER_SCRIPTS else "latin"

    def _load(self, family: str):
        if family in self._models:
            return self._models[family]
        if family in self._failed:
            return None
        path = self.model_dir / f"{family}.klm"
        if not self.model_dir or not path.is_file():
            self._failed[family] = f"no readable model at {path}"
            return None
        try:
            import kenlm  # type: ignore
            model = kenlm.Model(str(path))
        except Exception as exc:  # optional native extension/model errors
            self._failed[family] = f"model load failed: {type(exc).__name__}"
            return None
        self._models[family] = model
        return model

    def score(self, text: str, script: Optional[str] = None) -> Optional[float]:
        """Mean log10 probability per token, or None when unavailable.

        Normalized by token count so a long correct line is not ranked
        below a short one purely for having more tokens to be charged for.
        """
        cleaned = (text or "").strip()
        if not cleaned:
            return None
        family = self._family(script)
        model = self._load(family)
        if model is None:
            return None
        tokens = list(cleaned) if family == "cjk" else cleaned.split()
        if not tokens:
            return None
        try:
            total = model.score(" ".join(tokens), bos=True, eos=True)
        except Exception:
            return None
        return float(total) / len(tokens)

    def status(self) -> Dict[str, object]:
        families = {}
        for family in ("latin", "cjk"):
            path = self.model_dir / f"{family}.klm" if self.model_dir else None
            families[family] = bool(path and path.is_file())
        ready = any(families.values())
        return {
            "id": self.provider_id, "available": ready, "ready": ready,
            "version": str(self.model_dir) if ready else None,
            "families": families,
            "reason": None if ready else "no readable model configured in TOFU_KENLM_DIR",
        }


_default_provider: LanguageModelProvider = FastTextLanguageProvider()
_default_scorer = KenLMScoringProvider()


def get_scoring_provider() -> KenLMScoringProvider:
    return _default_scorer


def set_scoring_provider(provider: KenLMScoringProvider) -> None:
    global _default_scorer
    _default_scorer = provider


def get_language_provider() -> LanguageModelProvider:
    return _default_provider


def set_language_provider(provider: LanguageModelProvider) -> None:
    global _default_provider
    _default_provider = provider

