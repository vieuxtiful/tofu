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
import math
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

# fit thresholds: predicted_width / bbox_width
EXPANSION_WARN = 1.05   # likely needs condensing or wrapping
EXPANSION_FAIL = 1.35   # will not fit — block before cleanse/scribe run

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
        self.font_registry: Optional[FontRegistry] = (
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

        # 3. render quality prediction (deterministic; learned model swaps in later)
        render_score = self._predict_render_quality(targ_lang, context)
        if script is not None:
            render_score = self._deterministic_render_score(
                coverage_pct, context
            )
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
        complex_scripts = {"Arab", "Hani", "Hang", "Hans", "Hant", "Jpan", "Kore", "Deva", "Thai"}
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
        ratio = (
            EXPANSION_FACTORS.get(targ_lang, 1.0)
            / EXPANSION_FACTORS.get(src_lang, 1.0)
        )
        font = (context or {}).get("font")
        fits: Dict[str, float] = {}

        for inst in text_manifest.instances:
            bbox = inst.bounding_box
            if not inst.text or bbox is None or bbox.width <= 0:
                continue

            target = inst.target_text
            fit = None
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
            if fit is None:
                if target:
                    # no metrics but actual translation: char-count proxy
                    fit = len(target) / max(1, len(inst.text))
                else:
                    # no metrics — assume source text fills its box
                    fit = ratio

            fits[inst.id] = round(fit, 3)
            if fit > EXPANSION_FAIL:
                # with an ACTUAL translation, scribe shrinks the font to
                # fit — the text cannot overflow, only render small. a
                # blocking error is reserved for the predictive
                # (pre-translation) case where planning can still react.
                issues.append(
                    VldtnClass(
                        severity=VldtnSeverity.WARNING if target else VldtnSeverity.ERROR,
                        code="ToFU_005",
                        message=(
                            f"translation for region '{inst.id}' will render well below "
                            f"source size ({fit:.0%} of available width; font shrunk to fit)."
                            if target else
                            f"translated text for region '{inst.id}' predicted to "
                            f"overflow its bounding box ({fit:.0%} of available width)."
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
