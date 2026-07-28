## 🍢 Tofu - pre-flight validation layer for glyph support and render feasibility.
## vieuxtiful
"""
this module provides the tofu (text-over-frame unification) validation layer,
which ensures that the target language can be rendered correctly by the
system before any heavy processing begins. it checks font availability,
script support, and predicts potential rendering issues.
"""

from typing import Optional, Dict, List, Any
import math
import statistics
import unicodedata
from tofu.core.types import (
    VldtnReport,
    VldtnClass,
    ScrptSpprt,
    VldtnSeverity,
    VldtnInsight,
    TextManifest,
)
from tofu.layers.fonts import FontRegistry, HAVE_FONTTOOLS


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

# issue-code registry (the full set this layer can raise):
#   ToFU_000  target language has no script mapping            ERROR
#   ToFU_001  no font covers the target script                 ERROR
#   ToFU_002  target script has only partial font support       WARNING
#   ToFU_003  source glyph segmentation looks unreliable        WARNING
#   ToFU_004  predicted render quality below RENDER_WARN        WARNING
#   ToFU_005  translated text will not fit its region      ERROR/WARNING
#   ToFU_007  transformed text will clip at its box edge        WARNING
# ToFU_006 was never allocated -- the gap is historical, not a missing
# check. codes are append-only: they appear in stored manifests and in
# published eval reports, so renumbering would silently reinterpret them.

# horizontal text-expansion factors relative to an english baseline
# (industry localisation heuristics; W3C/localisation-vendor guidance)
EXPANSION_FACTORS: Dict[str, float] = {
    "en": 1.00,
    "de": 1.35, "fi": 1.30, "hu": 1.25, "es": 1.25, "fr": 1.20,
    "pt": 1.20, "it": 1.20, "pl": 1.20, "nl": 1.20, "ru": 1.15,
    "el": 1.15, "vi": 1.15, "th": 1.15, "sv": 1.10, "tr": 1.10,
    "ar": 1.05, "hi": 1.05, "he": 0.95, "ko": 0.80,
    "ja": 0.60,
    "zh-cn": 0.60, "zh-sg": 0.60, "zh-tw": 0.60, "zh-hk": 0.60, "zh-mo": 0.60,
}

# per-script fallbacks for the languages EXPANSION_FACTORS does not name
# individually. the table above covers 26 of the 63 languages in
# lang_to_script; the other 37 used to fall through to a silent 1.0,
# which asserts "this language is exactly as wide as english" for
# everything from Amharic to Khmer. a script-level figure is still a
# generalisation, but it is a defensible one and it is visible.
SCRIPT_EXPANSION_FACTORS: Dict[str, float] = {
    "Latn": 1.15,   # most non-english latin orthographies run longer
    "Cyrl": 1.15,
    "Grek": 1.15,
    "Arab": 1.05,
    "Hebr": 0.95,
    "Hans": 0.60, "Hant": 0.60, "Hani": 0.60,
    "Jpan": 0.60, "Kore": 0.80, "Hang": 0.80,
    "Deva": 1.05, "Beng": 1.05, "Guru": 1.05, "Gujr": 1.05,
    "Taml": 1.10, "Telu": 1.10, "Knda": 1.10, "Mlym": 1.15,
    "Sinh": 1.10, "Thai": 1.15, "Laoo": 1.15, "Mymr": 1.15, "Khmr": 1.20,
    "Ethi": 1.05, "Armn": 1.10, "Geor": 1.10,
}


def expansion_factor(lang: Optional[str]) -> float:
    """Width factor for a language relative to english.

    Falls back to the language's SCRIPT before falling back to 1.0, so an
    unlisted language inherits a plausible figure instead of silently
    claiming english-equivalent width.
    """
    if not lang:
        return 1.0
    if lang in EXPANSION_FACTORS:
        return EXPANSION_FACTORS[lang]
    return SCRIPT_EXPANSION_FACTORS.get(lang_to_script.get(lang, ""), 1.0)

# fit thresholds: predicted_width / bbox_width
EXPANSION_WARN = 1.05   # likely needs condensing or wrapping
EXPANSION_FAIL = 1.35   # will not fit — block before cleanse/scribe run

# characters that occupy a full em rather than roughly half of one
_FULL_EM_WIDTHS = {"W", "F"}   # UAX #11 Wide, Fullwidth


def measure_in_ems(text: str) -> float:
    """Rough advance width of a string in em units, with no font metrics.

    UAX #11 (East Asian Width) is the standard answer to "how wide is
    this character, approximately": W/F occupy a full em, everything else
    about half. Combining marks add nothing — they stack onto their base
    glyph rather than advancing the pen.

    This exists because a raw character count is not a width. Comparing
    len(target)/len(source) scored 焼肉 → "Viande grillée" as a 7×
    expansion when the honest figure is 3.5×, and — worse — called
    出口 → "Exit" a 2× overflow when the two actually occupy the same
    width. Every CJK→Latin region inherited a spurious warning from that
    arithmetic.

    Deliberately NOT a substitute for real metrics: when a font registry
    and a selected font are available, ToFU_005 measures the actual
    advances instead. This is the fallback for when they are not.
    """
    total = 0.0
    for ch in text or "":
        if unicodedata.combining(ch):
            continue
        total += 1.0 if unicodedata.east_asian_width(ch) in _FULL_EM_WIDTHS else 0.5
    return total

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
    "Hani": ScrptSpprt.FULL,      # generic han fallback
    "Hang": ScrptSpprt.FULL,      # hangul
    "Hans": ScrptSpprt.FULL,      # simplified han — mainland china, singapore
    "Hant": ScrptSpprt.FULL,      # traditional han — taiwan, hong kong, macau
    "Jpan": ScrptSpprt.FULL,      # japanese — kana + kanji
    "Kore": ScrptSpprt.FULL,      # korean — hangul (+ hanja tier)
    "Hira": ScrptSpprt.FULL,      # hiragana
    "Kana": ScrptSpprt.FULL,      # katakana
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


# scripts whose glyphs are dense, connected, or conjunct-forming — the
# properties that make segmenting one glyph from its neighbours in PIXELS
# hard, as distinct from alphabetic scripts with separated letterforms
DENSE_SCRIPTS = {
    "Hani", "Hans", "Hant", "Jpan", "Kore", "Hang", "Hira", "Kana",
    "Arab", "Deva", "Beng", "Guru", "Gujr", "Taml", "Telu", "Knda",
    "Mlym", "Sinh", "Mymr", "Khmr", "Thai",
}

# a text box shorter than this cannot hold a reliably segmentable glyph:
# CRNN recognition accuracy collapses under ~16 px x-height, and a
# detected box runs roughly 1.5-2x x-height
MIN_LEGIBLE_BOX_PX = 16
# median box height at which segmentation stops being size-limited
COMFORTABLE_BOX_PX = 40

# --- warn gates, calibrated rather than guessed --------------------------
# both of these were set by hand and never checked against a distribution.
# scripts/eval_tofu.py measured them over nine fixtures x twelve target
# languages (scripts/eval_out/tofu-calibration-baseline.json):
#
#   - the old segmentation score took exactly TWO values (0.65 / 0.85)
#     across all 108 runs, identical for every asset despite median region
#     heights spanning 18px to 165px. against a 0.7 gate it fired on 54/108
#     runs -- precisely the six complex-script target languages, and
#     nothing about the image. it restated its own input.
#   - the old render score never once fell below its 0.6 gate (0/108).
#     with full glyph coverage that formula floors at 0.70, so the check
#     could not fire without ToFU_001 having already errored.
#
# after re-keying the segmentation score to measured SOURCE geometry and
# threading real typography into the render context, the same sweep gives:
#
#   ToFU_003   54/108 -> 12/108, and now on ONE asset across all twelve
#              target languages rather than on six languages across every
#              asset. that asset is japan-street (median box 18px, min
#              8px), the fixture whose measured OCR quality is genuinely
#              the worst of the nine: garbage fraction 0.118, recall
#              0.571 against ground truth.
#   ToFU_004    0/108 -> 24/108, of which 22 fire INDEPENDENTLY of
#              ToFU_001 (it was previously unreachable without one). it
#              now flags exactly the two fixtures containing 8px text --
#              japan-street and gemini-street -- and stays quiet on the
#              six synthetic fixtures and on china-street.
#
# re-derive with: .venv/Scripts/python scripts/eval_tofu.py --tag <label>
SEGMENTATION_WARN = 0.60
RENDER_WARN = 0.75


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
        font_registry: Optional[FontRegistry] = None,
    ):
        """
        initialize the tofu validator.

        args:
            font_library_path: path to a font collection (optional).
            supported_scripts: override the default script support map.
            font_registry: an ALREADY-BUILT registry to score against.
                discovery walks a whole font directory with fontTools, so
                a process that already has one (the server, the pipeline)
                must be able to hand it over instead of paying for a
                second scan. takes precedence over font_library_path.

        with neither a path nor a registry, validation falls back to the
        static SUPPORTED_SCRIPTS map -- script-level assumptions rather
        than real per-font glyph coverage. that fallback is a legitimate
        degraded mode, but it is NOT the same check, so callers that can
        supply fonts should.
        """
        self.font_library_path = font_library_path
        self._supported_scripts = supported_scripts or SUPPORTED_SCRIPTS.copy()
        if font_registry is not None:
            self.font_registry: Optional[FontRegistry] = font_registry
        else:
            self.font_registry = (
                FontRegistry(font_library_path)
                if font_library_path and HAVE_FONTTOOLS
                else None
            )

    def validate(
        self,
        asset: Any,          # could be an image, video, or file path
        targ_lang: str,
        context: Optional[Dict] = None,
        text_manifest: Optional[TextManifest] = None,
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
            support, coverage_pct, font_used = self._resolve_script_support(
                script, targ_lang, context, suggested_actions
            )
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

        # 2. glyph segmentation feasibility — scored from the SOURCE text
        #    that will actually be segmented, using measured region
        #    geometry when a manifest exists. no manifest means nothing
        #    has been looked at yet, so the score is reported but never
        #    raised as an issue.
        source_lang = (text_manifest.src_lang if text_manifest else None) or targ_lang
        glyph_score = self._estimate_glyph_segmentation_score(
            source_lang, context, text_manifest
        )
        if text_manifest is not None and glyph_score < SEGMENTATION_WARN:
            issues.append(
                VldtnClass(
                    severity=VldtnSeverity.WARNING,
                    code="ToFU_003",
                    message=(
                        "source text is small or fragmented enough that glyph "
                        "segmentation may be unreliable; erasure and style "
                        "matching will be approximate."
                    ),
                    suggestion="manual review or a higher-resolution source may improve results.",
                )
            )
            suggested_actions.append("manual glyph segmentation review.")

        # 3. render quality prediction (deterministic; learned model swaps in later)
        render_score = self._predict_render_quality(targ_lang, context)
        if script is not None:
            render_score = self._deterministic_render_score(
                coverage_pct, context
            )
        if render_score < RENDER_WARN:
            issues.append(
                VldtnClass(
                    severity=VldtnSeverity.WARNING,
                    code="ToFU_004",
                    message="predicted render quality is below threshold.",
                    suggestion="consider alternative font or rendering strategy.",
                )
            )
            suggested_actions.append("try alternative rendering configuration.")

        # 4. text-expansion feasibility (pure font math; requires a manifest —
        #    e.g., re-validation after cicerone, or frontend-supplied regions)
        expansion_fit: Dict[str, float] = {}
        insights: List[VldtnInsight] = []
        if text_manifest is not None:
            expansion_fit = self._check_expansion_feasibility(
                text_manifest, targ_lang, context, issues, suggested_actions
            )
            insights = [
                *self._font_evidence_insights(text_manifest),
                *self._semantic_substitution_insights(text_manifest),
            ]

        # determine overall pass/fail (any error fails)
        passed = all(i.severity != VldtnSeverity.ERROR for i in issues)

        return VldtnReport(
            passed=passed,
            issues=issues,
            scrpt_spprt=script_support,
            glyph_segmentation_score=glyph_score,
            render_quality_score=render_score,
            expansion_fit=expansion_fit,
            suggested_actions=suggested_actions,
            insights=insights,
        )

    @staticmethod
    def _font_evidence_insights(text_manifest: TextManifest) -> List[VldtnInsight]:
        """Translate Cicerone's persisted visual-font evidence into ToFU UX.

        This is intentionally a read-only bridge between layers.  Cicerone
        owns observation and font_matching owns retrieval; ToFU makes the
        evidence useful at the decision gate without promoting a low-score
        candidate, a paid face, or a contextual reference into an automatic
        selection.
        """
        insights: List[VldtnInsight] = []
        style_refs: Dict[tuple, VldtnInsight] = {}
        for inst in text_manifest.instances:
            match = inst.font_match if isinstance(inst.font_match, dict) else None
            if not match:
                continue
            substitute = match.get("recommended_substitute")
            if isinstance(substitute, dict) and substitute.get("font_path"):
                confidence = match.get("confidence")
                try:
                    confidence = float(confidence) if confidence is not None else None
                except (TypeError, ValueError):
                    confidence = None
                score = substitute.get("score")
                try:
                    score = float(score) if score is not None else None
                except (TypeError, ValueError):
                    score = None
                accepted = match.get("status") == "matched"
                family = str(substitute.get("family") or "installed font")
                subfamily = substitute.get("subfamily")
                label = f"{family}{f' {subfamily}' if subfamily else ''}"
                source_text = (inst.text or "this region").strip()
                insights.append(VldtnInsight(
                    key=f"font-substitute:{inst.id}", kind="font_substitute",
                    title="Visual font match" if accepted else "Visual font match needs review",
                    detail=(
                        f"{inst.id} ({source_text!r}) has a {confidence:.0%} local glyph-shape match; "
                        f"{label} is the nearest installed {'match' if accepted else 'substitute'}."
                        if confidence is not None else
                        f"{inst.id} ({source_text!r}) has {label} as its nearest installed substitute."
                    ),
                    severity="info" if accepted else "review", region_id=inst.id,
                    region_ids=[inst.id], confidence=confidence, visual_score=score,
                    family=family, subfamily=str(subfamily) if subfamily else None,
                    font_path=str(substitute["font_path"]), license="installed",
                    source=str(match.get("provider") or "local_glyph_retrieval"),
                ))

            candidates = [*(match.get("candidates") or []), *(match.get("external_candidates") or [])]
            for candidate in candidates:
                if not isinstance(candidate, dict) or candidate.get("license") != "commercial":
                    continue
                family = str(candidate.get("family") or "Licensed font")
                source = str(candidate.get("source") or "catalog")
                ref_key = (family, candidate.get("url"), source)
                existing = style_refs.get(ref_key)
                if existing:
                    existing.region_ids.append(inst.id)
                    continue
                is_reference = source == "contextual_style_reference"
                detail = str(candidate.get("reason") or (
                    "This licensed catalog candidate requires a licence and editor review before use."
                ))
                insight = VldtnInsight(
                    key=f"style-reference:{family.lower().replace(' ', '-')}",
                    kind="style_reference" if is_reference else "licensed_font_candidate",
                    title="Licensed style reference" if is_reference else "Licensed font candidate",
                    detail=detail, severity="warning", region_id=inst.id,
                    region_ids=[inst.id], family=family,
                    subfamily=str(candidate["subfamily"]) if candidate.get("subfamily") else None,
                    license="commercial", foundry=candidate.get("foundry"),
                    url=candidate.get("url"), source=source,
                )
                style_refs[ref_key] = insight
                insights.append(insight)
        return insights

    @staticmethod
    def _semantic_substitution_insights(text_manifest: TextManifest) -> List[VldtnInsight]:
        """Surface Basil's target-order evidence at ToFU's preflight gate.

        This remains advisory: preflight explains that a multi-box source is
        one linguistic unit, but it never fills a target phrase or changes a
        region's geometry.  That keeps the localization architecture honest
        about the distinction between capture anchors and translated syntax.
        """
        insights: List[VldtnInsight] = []
        for unit in text_manifest.semantic_units or []:
            if len(unit.region_ids) < 2:
                continue
            substitution = unit.substitution if isinstance(unit.substitution, dict) else None
            target_order = substitution.get("target_region_order") if substitution else None
            applied = bool(substitution and substitution.get("applied"))
            detail = (
                f"{unit.source_text!r} is registered as {unit.entity_type.replace('_', ' ')} across "
                f"{' → '.join(unit.region_ids)}. "
            )
            if applied and target_order:
                cubes = substitution.get("spatial_anchor_order") or unit.region_ids
                detail += (
                    f"Its semantic blocks are {' → '.join(target_order)} and are plated into visual cubes "
                    f"{' → '.join(cubes)}; Scribe preserves every captured box while Basil may route a "
                    "block to a different cube."
                )
            else:
                detail += "Use Translation substitution to enter a complete target phrase and review its anchor mapping."
            insights.append(VldtnInsight(
                key=f"semantic-substitution:{unit.id}", kind="semantic_substitution",
                title="Semantic translation unit" if applied else "Translation substitution available",
                detail=detail, severity="info" if applied else "review",
                region_id=unit.region_ids[0], region_ids=list(unit.region_ids),
                confidence=unit.confidence, source=unit.analysis_provider,
            ))
        return insights

    @staticmethod
    def _estimate_glyph_segmentation_score(
        language: Optional[str],
        context: Optional[Dict],
        text_manifest: Optional[TextManifest] = None,
    ) -> float:
        """
        estimate how reliably source glyphs can be segmented from pixels.

        this is a SOURCE-side property. glyph segmentation is what
        imaging.text_mask() does to the original ink so that cleanse can
        erase it, savor can compare it and typography can measure it —
        none of which has anything to do with the language being rendered.
        scoring it from targ_lang (as this did) asked the wrong question
        and, being a constant per language, always returned the same
        answer whatever the image looked like.

        with a manifest, the score comes from what was actually measured:

          - median box height, over [10, COMFORTABLE_BOX_PX] px. small
            text is the documented failure mode for binarization and for
            CRNN recognition alike.
          - the small-text tail: the fraction of regions clearing
            MIN_LEGIBLE_BOX_PX. a scene can have a healthy median and
            still be full of unreadable signage.
          - a modest prior for dense/connected/conjunct scripts, which are
            harder to separate per glyph at any size.

        measured stroke ratio is deliberately NOT a term: it is available
        on every enriched region, but across this project's fixtures it
        did not separate them (the thinnest strokes, gemini-street at
        0.050, sit with the WORST recall while china-street's 0.070 has
        the best). a term the evidence does not support is noise.

        without a manifest — the pre-detection call, before a single pixel
        has been examined — this returns the script prior alone, and
        validate() deliberately raises no issue from it. a guess made
        before looking is not evidence.

        returns a score between 0 and 1, where 1 means highly feasible.
        """
        script = lang_to_script.get(language or "")
        dense = script in DENSE_SCRIPTS
        prior = 0.65 if dense else 0.85
        if text_manifest is None:
            return prior

        heights = [
            i.bounding_box.height
            for i in text_manifest.instances
            if i.bounding_box is not None and i.bounding_box.height > 0
        ]
        if not heights:
            return prior

        median_h = statistics.median(heights)
        size_term = max(0.0, min(1.0, (median_h - 10) / (COMFORTABLE_BOX_PX - 10)))
        tail_term = sum(1 for h in heights if h >= MIN_LEGIBLE_BOX_PX) / len(heights)
        script_term = 0.75 if dense else 1.0
        return round(
            0.55 * size_term + 0.25 * tail_term + 0.20 * script_term, 4
        )

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
        # default when no script/coverage information is available;
        # _deterministic_render_score supersedes this when a script resolves.
        return 0.75

    def _resolve_script_support(
        self,
        script: str,
        targ_lang: str,
        context: Optional[Dict],
        suggested_actions: List[str],
    ) -> tuple:
        """resolve script support and true glyph coverage.

        precedence:
          1. user-selected font (context['font']) scored against the exact
             script + language-required characters (e.g., vietnamese diacritics)
          2. best font in the registry (recommendation surfaced to frontend)
          3. static fallback map (no font library configured)

        returns (support, coverage_pct, font_used).
        """
        selected_font = (context or {}).get("font")

        if self.font_registry and self.font_registry.fonts:
            if selected_font and selected_font in self.font_registry.fonts:
                font_used = selected_font
            else:
                ranked = self.font_registry.recommend(script, targ_lang, limit=1)
                if not ranked or ranked[0][1] == 0.0:
                    suggested_actions.append(
                        f"no font in library covers script '{script}'."
                    )
                    return ScrptSpprt.UNSUPPORTED, 0.0, None
                font_used = ranked[0][0]
                suggested_actions.append(f"recommended font: {font_used}")

            support = self.font_registry.classify(font_used, script, targ_lang)
            coverage = self.font_registry.coverage(font_used, script)
            missing = self.font_registry.missing_required(font_used, targ_lang)
            if missing:
                preview = "".join(sorted(missing)[:12])
                suggested_actions.append(
                    f"font '{font_used}' lacks required characters for "
                    f"'{targ_lang}': {preview}…"
                )
            return support, coverage, font_used

        # static fallback — no registry available
        support = self._supported_scripts.get(script, ScrptSpprt.UNSUPPORTED)
        coverage = {
            ScrptSpprt.FULL: 1.0,
            ScrptSpprt.PARTIAL: 0.85,
            ScrptSpprt.UNSUPPORTED: 0.0,
        }[support]
        return support, coverage, None

    def _deterministic_render_score(
        self, coverage_pct: float, context: Optional[Dict]
    ) -> float:
        """deterministic render-quality prediction over inputs that exist now:

        - glyph coverage % of the target script/text          (weight 0.5)
        - target px size vs. minimum legibility               (weight 0.3)
        - effect-stack complexity (shadow/glow/warp count)    (weight 0.2)

        a learned de-rendering model (shimoda et al., 2021) can replace this
        behind the same signature.
        """
        ctx = context or {}

        # size legibility: full credit at >= 16px, zero below 6px
        font_px = ctx.get("font_px")
        if font_px is None:
            size_score = 0.8  # unknown — mildly optimistic
        else:
            size_score = max(0.0, min(1.0, (font_px - 6) / 10))

        # effect complexity: each effect layer costs 15%
        effects = ctx.get("effects") or []
        effect_score = max(0.0, 1.0 - 0.15 * len(effects))

        return round(
            0.5 * coverage_pct + 0.3 * size_score + 0.2 * effect_score, 4
        )

    def _check_expansion_feasibility(
        self,
        text_manifest: TextManifest,
        targ_lang: str,
        context: Optional[Dict],
        issues: List[VldtnClass],
        suggested_actions: List[str],
    ) -> Dict[str, float]:
        """ToFU_005: predict whether translated text will fit each region
        before cleanse/scribe ever run.

        fit = predicted_target_width / bbox_width. when the region already
        carries an actual translation (target_text), the check measures
        THAT text instead of predicting from expansion ratios — a user
        who deliberately shortened a translation must not be blocked by
        a prediction they already resolved. otherwise:
          - with font metrics: len(text) * expansion_ratio * avg_advance_em * font_px
          - without: bbox_width * expansion_ratio (assumes source fills the box)

        returns {region_id: fit_ratio}; fit > 1 means predicted overflow.
        """
        src_lang = text_manifest.src_lang or "en"
        ratio = expansion_factor(targ_lang) / expansion_factor(src_lang)
        font = (context or {}).get("font")
        fits: Dict[str, float] = {}

        for inst in text_manifest.instances:
            bbox = inst.bounding_box
            if not inst.text or bbox is None or bbox.width <= 0:
                continue

            target = inst.target_text
            fit = None
            measured = False   # did anything region-specific inform this fit?
            if self.font_registry and font:
                measure = target or inst.text
                adv_em = self.font_registry.avg_advance_em(font, measure)
                if adv_em > 0:
                    font_px = (
                        (inst.characteristics.size if inst.characteristics else None)
                        or (context or {}).get("font_px")
                        or bbox.height * 0.75  # cap-height heuristic
                    )
                    # actual target: measure it directly; else predict via ratio
                    predicted = len(measure) * (1.0 if target else ratio) * adv_em * font_px
                    fit = predicted / bbox.width
                    measured = True
            if fit is None:
                if target:
                    measured = True
                    # no metrics but an actual translation: compare the two
                    # strings by approximate WIDTH, not character count —
                    # a CJK glyph is one em where a latin letter is half
                    source_em = measure_in_ems(inst.text)
                    fit = (
                        measure_in_ems(target) / source_em if source_em > 0 else 1.0
                    )
                else:
                    # no metrics — assume source text fills its box
                    fit = ratio

            fits[inst.id] = round(fit, 3)
            if fit > EXPANSION_FAIL:
                # with an ACTUAL translation, scribe shrinks the font to
                # fit — the text cannot overflow, only render small. a
                # blocking error is reserved for the predictive
                # (pre-translation) case where planning can still react.
                #
                # ...but only when the prediction actually looked at this
                # region. with neither font metrics nor a translation,
                # `fit` IS the language-pair ratio: one table lookup,
                # identical for every region in the project regardless of
                # its box. that is a real planning signal — a ja→en job
                # predicts 1.67x and the editor should know — but it
                # carries no per-region evidence, so it must not fail a
                # run per region. this surfaced the moment src_lang was
                # populated correctly: every CJK source localized to a
                # european language began failing preflight outright,
                # before the user had entered a single translation.
                blocking = not target and measured
                issues.append(
                    VldtnClass(
                        severity=VldtnSeverity.ERROR if blocking else VldtnSeverity.WARNING,
                        code="ToFU_005",
                        message=(
                            f"translation for region '{inst.id}' will render well below "
                            f"source size ({fit:.0%} of available width; font shrunk to fit)."
                            if target else
                            f"translated text for region '{inst.id}' predicted to "
                            f"overflow its bounding box ({fit:.0%} of available width)."
                            + ("" if measured else
                               f" estimated from the {text_manifest.src_lang or 'en'}→{targ_lang} "
                               "expansion ratio alone; select a font for a per-region measurement.")
                        ),
                        suggestion="shorten translation, reduce font size, or enlarge the region.",
                        region_id=inst.id,
                    )
                )
                suggested_actions.append(
                    f"resolve text expansion overflow in region '{inst.id}'."
                )
            elif fit > EXPANSION_WARN:
                issues.append(
                    VldtnClass(
                        severity=VldtnSeverity.WARNING,
                        code="ToFU_005",
                        message=(
                            f"translated text for region '{inst.id}' may be tight "
                            f"({fit:.0%} of available width)."
                        ),
                        suggestion="consider condensed styling or line wrapping.",
                        region_id=inst.id,
                    )
                )

            # Strict preview/final parity: transformed ink is always clipped
            # to this same captured region.  This intentionally warns rather
            # than silently expanding a cube or permitting neighbour bleed.
            transform = (inst.style_profile.transform if inst.style_profile else None) or {}
            skew_x = float(transform.get("skew_x", 0) or 0)
            skew_y = float(transform.get("skew_y", 0) or 0)
            scale_x = float(transform.get("scale_x", 1) or 1)
            scale_y = float(transform.get("scale_y", 1) or 1)
            offset_x = float(transform.get("offset_x", 0) or 0)
            offset_y = float(transform.get("offset_y", 0) or 0)
            transform_active = any((offset_x, offset_y, skew_x, skew_y, abs(scale_x - 1), abs(scale_y - 1)))
            if transform_active:
                shear_x = abs(math.tan(math.radians(max(-45.0, min(45.0, skew_x))))) * bbox.height / 2
                shear_y = abs(math.tan(math.radians(max(-45.0, min(45.0, skew_y))))) * bbox.width / 2
                expansion_x = abs(scale_x - 1) * bbox.width / 2 + abs(offset_x) + shear_x
                expansion_y = abs(scale_y - 1) * bbox.height / 2 + abs(offset_y) + shear_y
                if expansion_x > 0.5 or expansion_y > 0.5:
                    issues.append(VldtnClass(
                        severity=VldtnSeverity.WARNING,
                        code="ToFU_007",
                        message=(f"transformed text for region '{inst.id}' will be clipped at its "
                                 "bounding-box edge when it exceeds the captured cube."),
                        suggestion="reduce offset/skew/stretch, resize the region, or enable wrap.",
                        region_id=inst.id,
                    ))

        return fits


# global default instance (can be reused)
_default_tofu: Optional[ToFU] = None


def get_tofu() -> ToFU:
    """get or create a default tofu instance."""
    global _default_tofu
    if _default_tofu is None:
        _default_tofu = ToFU()
    return _default_tofu


def validate(
    asset: Any,
    targ_lang: str,
    context: Optional[Dict] = None,
    text_manifest: Optional[TextManifest] = None,
) -> VldtnReport:
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
    return get_tofu().validate(asset, targ_lang, context, text_manifest)
