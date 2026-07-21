## 🍢 translate — pluggable machine-translation seam
## vieuxtiful
"""
Translator backend interface for future MT integration.

The product workflow is TMS round-trip (export empty targets →
translate externally → re-import); no MT provider is wired. This module
exists so an MT provider (DeepL, Google, an LLM, a TM lookup) can slot
in later behind a stable interface without reworking callers:

    backend = get_backend()
    translations = backend.translate(
        [("r1", "MAIN STREET")], src_lang="en", targ_lang="es",
    )   # -> {"r1": "CALLE MAYOR"} (or {} when not configured)
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple


class TranslatorBackend(ABC):
    """swappable machine-translation adapter."""

    name: str = "base"

    @abstractmethod
    def translate(
        self,
        segments: List[Tuple[str, str]],
        src_lang: str,
        targ_lang: str,
    ) -> Dict[str, str]:
        """translate (segment_id, source_text) pairs.

        returns {segment_id: translated_text} for segments it handled;
        untranslated segments are simply absent from the result.
        """


class NullTranslator(TranslatorBackend):
    """no-provider default: translates nothing, so the TMS round-trip
    remains the only translation path until a real backend is set."""

    name = "null"

    def translate(
        self,
        segments: List[Tuple[str, str]],
        src_lang: str,
        targ_lang: str,
    ) -> Dict[str, str]:
        return {}


_default_backend: Optional[TranslatorBackend] = None


def get_backend() -> TranslatorBackend:
    global _default_backend
    if _default_backend is None:
        _default_backend = NullTranslator()
    return _default_backend


def set_backend(backend: TranslatorBackend) -> None:
    global _default_backend
    _default_backend = backend
