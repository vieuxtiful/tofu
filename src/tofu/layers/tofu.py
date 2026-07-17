## 🍢 ToFU
## vieuxtiful
"""
pre-flight validation layer for glyph support and render feasibility.

this module provides the tofu (text-over-frame unification) validation layer,
which ensures that the target language can be rendered correctly by the
system before any heavy processing begins. it checks font availability,
script support, and predicts potential rendering issues.
"""

from typing import Optional, Dict, List, Any
from tofu.core.types import (
    VldtnReport,
    VldtnClass,
    ScrptSpprt,
    VldtnSeverity,
)


# basic [unicode] language mapping (iso 15924)
lang_to_script: Dict[str, str] = {
    "en": "Latn", # english
    "es": "Latn", # spanish
    "fr": "Latn", # french
    "de": "Latn", # german
    "it": "Latn", # italian
    "pt": "Latn", # portuguese
    "ru": "Cyrl", # russian
    "uk": "Cyrl", # ukrainian
    "el": "Grek", # greek
    "ar": "Arab", # arabic
    "fa": "Arab", # persian
    "he": "Hebr", # hebrew
    "ja": "Jpan", # japanese
    "zh-cn": "Hans",   # simplified han
    "zh-sg": "Hans",   # simplified han
    "zh-tw": "Hant",   # traditional han
    "zh-hk": "Hant",   # traditional han
    "zh-mo": "Hant",   # traditional han
    "sr-latn": "Latn", # latin
    "sr-cyrl": "Cyrl", # cyrillic
    "ko": "Kore", # korean
    "hi": "Deva", # hindi
    "th": "Thai", # thai
    "vi": "Latn", # vietnamese (uses latin with diacritics)
    "bn": "Beng", # bengali
    "pa": "Guru", # punjabi
    "gu": "Gujr", # gujarati
    "mr": "Deva", # marathi
    "ta": "Taml", # tamil
    "te": "Telu", # telugu
    "kn": "Knda", # kannada
    "ml": "Mlym", # malayalam
    "si": "Sinh", # sinhala
    "my": "Mymr", # myanmar
    "km": "Khmr", # khmer
    "lo": "Laoo", # lao
    "am": "Ethi", # amharic
    "ti": "Ethi", # tigrinya
    "hy": "Armn", # armenian
    "ka": "Geor", # georgian
    "mn": "Cyrl", # mongolian
    "kk": "Cyrl", # kazakh
    "uz": "Latn", # uzbek
    "az": "Latn", # azeri
    "tr": "Latn", # turkish
    "nl": "Latn", # dutch
    "sv": "Latn", # swedish
    "no": "Latn", # norwegian
    "da": "Latn", # danish
    "fi": "Latn", # finnish
    "is": "Latn", # icelandic
    "pl": "Latn", # polish
    "cs": "Latn", # czech
    "sk": "Latn", # slovak
    "hu": "Latn", # hungarian
    "ro": "Latn", # romanian
    "bg": "Cyrl", # bulgarian
    "sr": "Cyrl", # serbian
    "hr": "Latn", # croatian
    "sl": "Latn", # slovenian
    "et": "Latn", # estonian
    "lv": "Latn", # latvian
    "lt": "Latn" # lithuanian
}

# font lib placeholder – intended to query
# system fonts or a provided font collection.
# for all intents and purposes, unilateral script support 
# is assumed (except a few)
SUPPORTED_SCRIPTS: Dict[str, ScrptSpprt] = {
    "Latn": ScrptSpprt.FULL,      # latin
    "Cyrl": ScrptSpprt.FULL,      # cyrillic
    "Grek": ScrptSpprt.FULL,      # greek
    "Arab": ScrptSpprt.PARTIAL,   # some fonts may lack certain ligatures
    "Hebr": ScrptSpprt.PARTIAL,   # hebrew
    "Hani": ScrptSpprt.FULL,      # hanzi/hanja/hangul
    "Hang": ScrptSpprt.FULL,      # hangul
    "Deva": ScrptSpprt.PARTIAL,   # devanagari
    "Thai": ScrptSpprt.PARTIAL,   # thai
    "Beng": ScrptSpprt.PARTIAL,   # bengali
    "Guru": ScrptSpprt.PARTIAL,   # gurmukhi
    "Gujr": ScrptSpprt.PARTIAL,   # gujarati
    "Taml": ScrptSpprt.PARTIAL,   # tamil
    "Telu": ScrptSpprt.PARTIAL,   # telugu
    "Knda": ScrptSpprt.PARTIAL,   # kannada
    "Mlym": ScrptSpprt.PARTIAL,   # malayalam
    "Sinh": ScrptSpprt.PARTIAL,   # sinhala
    "Mymr": ScrptSpprt.PARTIAL,   # myanmar
    "Khmr": ScrptSpprt.PARTIAL,   # khmer
    "Laoo": ScrptSpprt.PARTIAL,   # lao
    "Ethi": ScrptSpprt.PARTIAL,   # ethiopic / ge'ez
    "Armn": ScrptSpprt.FULL,      # armenian
    "Geor": ScrptSpprt.FULL,      # georgian
}


class ToFU:
    """
    pre-flight validation engine for localisation rendering.

    the tofu layer verifies that the target language can be rendered
    without "tofu" (missing glyphs) and estimates the feasibility of
    the entire pipeline for the given input.
    """

    def __init__(
        self,
        font_library_path: Optional[str] = None,
        supported_scripts: Optional[Dict[str, ScrptSpprt]] = None,
    ):
        """
        initialize the tofu validator.

        args:
            font_library_path: path to a font collection (optional).
            supported_scripts: override the default script support map.
        """
        self.font_library_path = font_library_path
        self._supported_scripts = supported_scripts or SUPPORTED_SCRIPTS.copy()

    def validate(
        self,
        asset: Any,          # could be an image, video, or file path
        targ_lang: str,
        context: Optional[Dict] = None,
    ) -> VldtnReport:
        """
        perform pre-flight validation for the target language.

        this method checks:
        1. script support for the target language.
        2. glyph segmentation feasibility (based on language heuristics).
        3. render quality prediction (using de-rendering parameters).

        args:
            asset: the input asset (image/video) – currently used only for
                   contextual checks; can be none.
            targ_lang: target language code (e.g., 'en', 'ja').
            context: optional additional context (e.g., resolution, style).

        returns:
            vldtnreport containing validation results, issues, and suggestions.
        """
        issues: List[VldtnClass] = []
        suggested_actions: List[str] = []
        script_support: Dict[str, ScrptSpprt] = {}

        # 1. script support check
        script = lang_to_script.get(targ_lang)
        if script is None:
            # unknown language – treat as unsupported
            issues.append(
                VldtnClass(
                    severity=VldtnSeverity.ERROR,
                    code="ToFU_000",
                    message=f"language '{targ_lang}' is not recognized.",
                    suggestion="add language mapping or use a fallback script.",
                )
            )
            suggested_actions.append("define script for language.")
            script_support[targ_lang] = ScrptSpprt.UNSUPPORTED
        else:
            support = self._supported_scripts.get(script, ScrptSpprt.UNSUPPORTED)
            script_support[script] = support
            if support == ScrptSpprt.UNSUPPORTED:
                issues.append(
                    VldtnClass(
                        severity=VldtnSeverity.ERROR,
                        code="ToFU_001",
                        message=f"script '{script}' for language '{targ_lang}' is not supported.",
                        suggestion="install fonts supporting this script or choose a different target language.",
                    )
                )
                suggested_actions.append(f"add font support for script '{script}'.")
            elif support == ScrptSpprt.PARTIAL:
                issues.append(
                    VldtnClass(
                        severity=VldtnSeverity.WARNING,
                        code="ToFU_002",
                        message=f"script '{script}' has only partial font support.",
                        suggestion="rendering may have visual artifacts; manual review recommended.",
                    )
                )
                suggested_actions.append("consider using a more complete font.")

        # 2. glyph segmentation feasibility (language‑based heuristic)
        glyph_score = self._estimate_glyph_segmentation_score(targ_lang, context)
        if glyph_score < 0.7:
            issues.append(
                VldtnClass(
                    severity=VldtnSeverity.WARNING,
                    code="ToFU_003",
                    message="glyph segmentation may be unreliable for this language.",
                    suggestion="manual review or pre‑processing may improve results.",
                )
            )
            suggested_actions.append("manual glyph segmentation review.")

        # 3. render quality prediction (using de‑rendering parameters)
        render_score = self._predict_render_quality(targ_lang, context)
        if render_score < 0.6:
            issues.append(
                VldtnClass(
                    severity=VldtnSeverity.WARNING,
                    code="ToFU_004",
                    message="predicted render quality is below threshold.",
                    suggestion="consider alternative font or rendering strategy.",
                )
            )
            suggested_actions.append("try alternative rendering configuration.")

        # determine overall pass/fail (any error fails)
        passed = all(i.severity != VldtnSeverity.ERROR for i in issues)

        return VldtnReport(
            passed=passed,
            issues=issues,
            scrpt_spprt=script_support,
            glyph_segmentation_score=glyph_score,
            render_quality_score=render_score,
            suggested_actions=suggested_actions,
        )

    def _estimate_glyph_segmentation_score(self, language: str, context: Optional[Dict]) -> float:
        """
        estimate the feasibility of glyph segmentation for the given language.

        this is a heuristic based on language complexity. in a real system,
        it would be derived from a glyph segmentation model or historical data.

        args:
            language: language code.
            context: optional context (e.g., image resolution).

        returns:
            a score between 0 and 1, where 1 means highly feasible.
        """
        # simple heuristic: languages with complex scripts get lower scores.
        complex_scripts = {"Arab", "Hani", "Hang", "Deva", "Thai"}
        script = lang_to_script.get(language)
        if script in complex_scripts:
            return 0.65
        else:
            return 0.85

    def _predict_render_quality(self, language: str, context: Optional[Dict]) -> float:
        """
        predict render quality based on de‑rendering parameters.

        based on shimoda et al., 2021 – de‑rendering stylized texts.
        in practice, this would use a learned model; here we provide a stub.

        args:
            language: language code.
            context: optional context (e.g., font size, background).

        returns:
            a score between 0 and 1.
        """
        # e.g., return a fixed score; in production, this would
        # analyze font metrics and rendering conditions.
        return 0.75


# global default instance (can be reused)
_default_tofu: Optional[ToFU] = None


def get_tofu() -> ToFU:
    """get or create a default tofu instance."""
    global _default_tofu
    if _default_tofu is None:
        _default_tofu = ToFU()
    return _default_tofu


def validate(asset: Any, targ_lang: str, context: Optional[Dict] = None) -> VldtnReport:
    """
    convenience function for pipeline use.

    this function wraps the default tofu instance and is the entry point
    called by the pipeline orchestrator.

    args:
        asset: the input asset (image/video).
        targ_lang: target language code.
        context: optional additional context.

    returns:
        vldtnreport with validation results.
    """
    return get_tofu().validate(asset, targ_lang, context)