## 🍢 Verify — Layer 5
## vieuxtiful
"""
quality verification layer.

verify scores the localized asset per instance, keyed as
per_asset_instance_score[asset_id][instance_id] per the QAReport
contract. the pipeline gates success (and memory storage) on
overall_score >= PipelineCfg.qa_threshold.

metrics (all standard in the scene-text editing literature):

  1. OCR round-trip legibility — re-recognize the rendered target text
     and compare against the intended string. this is the recognition-
     accuracy protocol used to evaluate scene-text editors (SRNet, Wu et
     al. 2019; STEFANN, Roy et al. 2020): if the OCR that found the
     source text cannot read the rendered target, the render failed for
     the same reasons a human reader would struggle (font too small,
     poor contrast, clipping). reuses the cicerone EasyOCR backend, so
     no new model dependency.

  2. background reconstruction (SSIM, Wang et al. 2004) — structural
     similarity between the source and localized asset over a border
     ring just OUTSIDE each bbox. cleanse/scribe must not disturb
     pixels beyond the region; halo artifacts, inpainting spill and
     misaligned composites depress this score.

  3. ink presence (fallback legibility) — when easyocr is unavailable,
     a rendered region must still show *evidence of rendering*: pixel
     change against the source inside the bbox plus glyph edge energy
     (text is high-frequency by nature). this catches the blank-render
     and invisible-text failure classes that OCR round-trip would
     catch, without any model dependency.

  4. residual source text — re-recognize the CLEANSED (pre-scribe) crop
     and compare against the ORIGINAL source string (Phase 0's
     eval_render.py harness first; graduates here in Phase 5, same
     recognition-based erasure protocol as EnsNet/EraseNet). a high
     similarity means the source text survived the erase — a genuine
     double-exposure defect no amount of new-text legibility excuses,
     so it multiplies the instance's score down rather than averaging
     in as just another positive-evidence term.

  5. style consistency — CIE76 color ΔE between the detected source
     text color (Scene's style_profile.color) and the mean ink color
     actually sampled from the rendered crop, plus a rendered-height
     ratio against the detected source size (Phase 1 typography). both
     reuse the shared Otsu+GrabCut glyph mask (utils.imaging.text_mask)
     cleanse/typography already use, so verify agrees with them on
     which pixels are ink.

coverage accounting: untranslated non-DNT regions were real detected
text that never got addressed — they now score a real deduction
(UNTRANSLATED_SCORE), not NEUTRAL, so an incomplete localization can no
longer inflate its own gate score by simply skipping regions.
QAReport.progress carries the full regions_total/dnt/translated/
untranslated/fallback_font breakdown for the frontend to render as a
coverage summary.

degradation contract: metrics that cannot run (easyocr or numpy/PIL
missing, region outside frame) are skipped, never failed — a missing
scorer must not fail runs on placeholder data. metrics["scorer"]
records exactly which metrics contributed.
"""

import re
import unicodedata
from collections import Counter
from dataclasses import asdict
from datetime import datetime, timezone
from difflib import SequenceMatcher
from typing import Any, Dict, List, Optional, Sequence, Tuple

from tofu.core.types import (
    BBox, TextManifest, QAReport, VerificationEvidence, VerificationProject,
    VerificationRegion, VerificationReport, VerificationVisualFlag,
)
from tofu.layers import cicerone, garnish, julienne, scribe
from tofu.layers.fonts import faces_of
from tofu.layers.tofu import lang_to_script
from tofu.utils.imaging import text_mask as _text_mask

NEUTRAL_SCORE = 1.0        # DNT: correctly excluded, not a failure to judge
UNTRANSLATED_SCORE = 0.0   # real detected text never addressed -- a genuine coverage gap
OCR_WEIGHT = 0.7           # legibility dominates: unreadable text is a failed localization
SSIM_WEIGHT = 0.3
STYLE_WEIGHT = 0.2
GARNISH_WEIGHT = 0.15       # physical integration is a tiebreaker, not legibility
RING_PX = 12                    # border ring width for background-reconstruction SSIM
CROP_PAD_PX = 4                 # slack around the bbox for the OCR crop
RESIDUAL_PENALTY_THRESHOLD = 0.3  # below this, treat as OCR noise, not a real leak
CONTENT_PASS_THRESHOLD = 0.85
CONTENT_REVIEW_THRESHOLD = 0.50
OCR_CONFIDENCE_REVIEW = 0.50
COMPONENT_WEIGHTS = {
    "coverage": 0.20,
    "content_integrity": 0.20,
    "spatial_fit": 0.20,
    "typographic_intent": 0.15,
    "script_rendering_validity": 0.15,
    "contextual_fit": 0.10,
}
COLOR_DELTA_E_SCALE = 40.0      # ΔE at which the color-consistency score bottoms out at 0

# lazily-built EasyOCR readers, keyed by language tuple — reader init is
# expensive (~seconds), scoring runs per render
_readers: Dict[Tuple[str, ...], Any] = {}

_RTL_BIDI = {"R", "AL", "AN"}

# Scripts written without spaces between words.  A word error rate over
# these would measure the segmenter, not the text.
_UNSEGMENTED_SCRIPTS = {"Hani", "Hira", "Kana", "Hang", "Thai"}


def _script_and_direction(
    text: Optional[str], lang: Optional[str] = None,
) -> Tuple[Optional[str], str]:
    """Script identity and direction for one string, per UAX #24.

    Delegates to julienne, which resolves the Script property PER CHARACTER
    and reports every script present.  The single value returned here is the
    joined summary ("Hani+Kana"), not a majority winner: this used to take
    ``Counter(...).most_common(1)`` over the characters, which called
    Japanese 東京タワー "cjk" because two of five characters are ideographs
    and Korean 서울特別市 "cjk" because three of five are hanja -- in both
    cases naming the script that is not the one making the string readable.
    Callers wanting the structured breakdown should use julienne directly.
    """
    analysis = julienne.analyse(text, lang)
    return (analysis.summary if analysis.scripts else None), analysis.direction


def _asset_class(text_manifest: TextManifest, inst) -> Optional[str]:
    if inst.background_profile and inst.background_profile.semantic_label:
        return inst.background_profile.semantic_label
    b = inst.bounding_box
    cx, cy = b.x + b.width / 2, b.y + b.height / 2
    hits = [
        region for region in text_manifest.scene_regions
        if region.bbox.x <= cx <= region.bbox.x + region.bbox.width
        and region.bbox.y <= cy <= region.bbox.y + region.bbox.height
    ]
    if not hits:
        return None
    return min(hits, key=lambda region: region.bbox.width * region.bbox.height).semantic_label


def _mask_bounds(mask: Any) -> Optional[BBox]:
    try:
        bounds = mask.getbbox()
    except Exception:
        return None
    if not bounds:
        return None
    x0, y0, x1, y1 = bounds
    return BBox(x=x0, y=y0, width=x1 - x0, height=y1 - y0)


def _mask_pixel_count(mask: Any, crop: Optional[Tuple[int, int, int, int]] = None) -> int:
    try:
        image = mask.crop(crop) if crop else mask
        histogram = image.histogram()
        return sum(histogram[1:])
    except Exception:
        return 0


def _region_quad(inst, bounds: BBox) -> Optional[List[Tuple[float, float]]]:
    """The region's own footprint in image space, when it has a perspective.

    Returns None for an absent, identity or degenerate quad -- exactly the
    cases where scribe also falls back to the affine path, so the check and
    the render always agree about which shape the region occupies.
    """
    transform = getattr(inst.style_profile, "transform", None) if inst.style_profile else None
    if not transform:
        return None
    corners = scribe._parse_quad(transform.get("quad"))
    if corners is None or scribe._is_identity_quad(corners):
        return None
    placed = scribe._denormalise_quad(corners, bounds)
    return placed if scribe._quad_is_usable(placed) else None


#: Antialiasing allowance, in pixels, when testing ink against a polygon.
#:
#: The rectangular path never needed one: text fitted inside a box stays
#: inside it. A quad boundary instead cuts THROUGH antialiased glyph edges,
#: and the supersample-warp-reduce cycle legitimately spreads ink a fraction
#: of a pixel past the mathematical edge. Measured on a trapezoid region
#: that is entirely correct, a hard-edged polygon reported 0.22% of its ink
#: outside and dropped the region to review. One pixel and a half is the
#: soft edge, not overflow.
QUAD_ANTIALIAS_PAD_PX = 1.5

#: Share of ink allowed outside a QUAD before it counts as overflow.
#:
#: The rectangular path needs no tolerance and keeps none: text fitted into
#: a box lands inside it, so any ink outside is real and a strict `> 0` test
#: is right. A quad boundary is not like that. It cuts diagonally through a
#: resampled glyph raster, and every antialiased edge pixel straddling it
#: contributes a fraction. Measured on correct trapezoid renders of ordinary
#: text, that residue is 1.1-1.5% of ink and it fired spatial_overflow on
#: renders doing exactly what was asked. Two percent is the soft edge; real
#: overflow from a mis-sized string is an order of magnitude larger (the
#: same fixture measured against the wrong shape reports 18%).
QUAD_OVERFLOW_TOLERANCE = 0.02


def _inflate_quad(
    corners: List[Tuple[float, float]], pad: float,
) -> List[Tuple[float, float]]:
    """Push each corner out from the centroid by roughly ``pad`` pixels."""
    cx = sum(x for x, _ in corners) / len(corners)
    cy = sum(y for _, y in corners) / len(corners)
    out: List[Tuple[float, float]] = []
    for x, y in corners:
        dx, dy = x - cx, y - cy
        distance = (dx * dx + dy * dy) ** 0.5
        if distance < 1e-9:
            out.append((x, y))
            continue
        scale = (distance + pad) / distance
        out.append((cx + dx * scale, cy + dy * scale))
    return out


def _mask_pixel_count_in_quad(mask: Any, corners: List[Tuple[float, float]]) -> int:
    """Ink inside an arbitrary quadrilateral.

    The rectangular crop that serves every other region cannot answer this.
    Under perspective the region's footprint is a trapezoid, so a corner
    pulled outward puts correctly-seated ink outside any axis-aligned box
    drawn around the source bbox -- counted as overflow on a render doing
    exactly what was asked of it. The quad is the region's real footprint
    and is the honest shape to measure against in both directions: stricter
    where corners were pulled in, more permissive where they were pulled out.
    """
    try:
        from PIL import Image, ImageChops, ImageDraw

        shape = Image.new("L", mask.size, 0)
        ImageDraw.Draw(shape).polygon(
            [(float(x), float(y))
             for x, y in _inflate_quad(corners, QUAD_ANTIALIAS_PAD_PX)],
            fill=255,
        )
        return _mask_pixel_count(ImageChops.multiply(mask.convert("L"), shape))
    except Exception:
        return 0


def _line_count(mask: Any, bounds: Optional[BBox], direction: str) -> Optional[int]:
    """Count separated ink bands; useful deterministic evidence, not OCR."""
    if mask is None or bounds is None:
        return None
    try:
        crop = mask.crop((
            bounds.x, bounds.y, bounds.x + bounds.width, bounds.y + bounds.height
        ))
        # Horizontal text forms row bands; vertical text forms columns.
        axis = 0 if direction == "ttb" else 1
        projection = [
            bool(
                crop.crop((i, 0, i + 1, crop.height)).getbbox()
                if axis == 0 else crop.crop((0, i, crop.width, i + 1)).getbbox()
            )
            for i in range(crop.width if axis == 0 else crop.height)
        ]
        # A glyph's dot/mark may sit a pixel or two away from its body; only
        # a real inter-line gap should create another observed line.
        groups, active, empty_run = 0, False, 0
        for occupied in projection:
            if occupied:
                if not active:
                    groups += 1
                active, empty_run = True, 0
            elif active:
                empty_run += 1
                if empty_run >= 3:
                    active = False
        return groups
    except Exception:
        return None


def _resolve_font_evidence(font_registry: Any, inst, lang: str) -> Dict[str, Any]:
    # An EXPLICIT request is a font the human chose.  resolved_font_family is
    # the server's own display hint for an auto region and carries no intent,
    # so the two must not be conflated when deciding whether a substitution
    # overrode anybody.
    explicit = inst.style_profile.font_family if inst.style_profile else None
    requested = inst.resolved_font_family or explicit
    text = inst.target_text or ""
    result: Dict[str, Any] = {
        "requested_font": requested,
        "explicit_request": bool(explicit),
        "assigned_font": requested,
        "status": "unknown",
        "coverage": None,
        "missing_codepoints": [],
        "fallback_used": bool(inst.glyph_fallback),
        "overrode_explicit_choice": False,
    }
    if not text or font_registry is None or not faces_of(font_registry):
        return result

    style = inst.style_profile
    if requested:
        assigned, _ = scribe.resolve_face(
            font_registry,
            requested,
            style.font_weight if style else None,
            bool(style.italic) if style else False,
        )
    else:
        assigned = scribe.resolve_auto_font(
            font_registry,
            lang,
            text,
            weight=style.font_weight if style else None,
            italic=bool(style.italic) if style else False,
        )
    replacement, requested_covered = scribe.check_glyph_coverage(
        font_registry, assigned, scribe.press_joins(text, lang), lang
    )
    if not requested_covered and replacement:
        assigned = replacement
    result["assigned_font"] = assigned
    result["fallback_used"] = bool(inst.glyph_fallback or replacement)
    result["overrode_explicit_choice"] = bool(explicit and replacement)

    key = (
        font_registry.resolve_key(assigned)
        if assigned and hasattr(font_registry, "resolve_key")
        else assigned
    )
    face = faces_of(font_registry).get(key)
    if face is None:
        return result
    required = {
        ord(char) for char in scribe.press_joins(text, lang)
        if not char.isspace() and not unicodedata.category(char).startswith("C")
    }
    missing = sorted(required - face.codepoints)
    result["coverage"] = 1.0 if not required else (len(required) - len(missing)) / len(required)
    result["missing_codepoints"] = [f"U+{codepoint:04X}" for codepoint in missing]
    result["status"] = "pass" if not missing else "fail"
    return result


def _shaping_evidence(lang: str, script: Optional[str]) -> Dict[str, Any]:
    from tofu.layers import knead

    requires_harfbuzz = scribe.shaping_script(lang) is not None
    rtl = scribe.is_rtl_lang(lang)
    if requires_harfbuzz:
        available = knead.available()
        method = "harfbuzz" if available else "pillow_fallback"
    elif rtl:
        try:
            import bidi  # noqa: F401
            bidi_available = True
        except ImportError:
            bidi_available = False
        try:
            import arabic_reshaper  # noqa: F401
            joins_available = True
        except ImportError:
            joins_available = script != "arabic"
        available = bidi_available and joins_available
        method = "python_bidi_reshaper" if available else "pillow_fallback"
    else:
        available, method = True, "pillow"
    return {
        "required": requires_harfbuzz or rtl,
        "available": available,
        "method": method,
    }


def _target_ocr_evidence(
    clean_rendered_asset: Any,
    inst,
    bounds: Optional[BBox],
    targ_lang: str,
) -> Dict[str, Any]:
    """Re-read Scribe's clean target crop and retain all scoring evidence."""
    evidence: Dict[str, Any] = {
        "status": "unavailable",
        "expected": inst.target_text,
        "recognized": None,
        "normalized_expected": _normalize(inst.target_text or ""),
        "normalized_recognized": None,
        "similarity": None,
        "confidence": None,
        "engine": "easyocr",
    }
    if clean_rendered_asset is None or not inst.target_text or bounds is None:
        return evidence
    reader = _get_reader(targ_lang)
    if reader is None:
        return evidence
    try:
        import numpy as np
        rendered = _to_array(np, clean_rendered_asset)
        if rendered is None:
            return evidence
        height, width = rendered.shape[:2]
        x0, y0 = max(0, bounds.x - CROP_PAD_PX), max(0, bounds.y - CROP_PAD_PX)
        x1 = min(width, bounds.x + bounds.width + CROP_PAD_PX)
        y1 = min(height, bounds.y + bounds.height + CROP_PAD_PX)
        if x1 - x0 < 4 or y1 - y0 < 4:
            return evidence
        results = reader.readtext(rendered[y0:y1, x0:x1])
    except Exception:
        return evidence
    recognized = " ".join(str(result[1]) for result in results if len(result) > 1)
    confidences = [
        float(result[2]) for result in results
        if len(result) > 2 and isinstance(result[2], (int, float))
    ]
    normalized = _normalize(recognized)
    rates = error_rates(inst.target_text or "", recognized, targ_lang)
    similarity = _text_similarity(inst.target_text, recognized, targ_lang)
    confidence = sum(confidences) / len(confidences) if confidences else None
    status = (
        "pass" if similarity >= CONTENT_PASS_THRESHOLD
        else "review" if similarity >= CONTENT_REVIEW_THRESHOLD
        else "fail"
    )
    evidence.update({
        "status": status,
        "recognized": recognized,
        "normalized_recognized": normalized,
        "similarity": round(float(similarity), 4),
        # CER drives the score; WER is a reviewer's diagnostic and is None
        # for the unsegmented scripts, where it would measure a segmenter.
        "cer": rates["cer"],
        "wer": rates["wer"],
        "cer_edits": rates["cer_edits"],
        "reference_length": rates["reference_length"],
        "word_segmentable": rates["word_segmentable"],
        "confidence": round(float(confidence), 4) if confidence is not None else None,
    })
    return evidence


def _weight_value(weight: Optional[str]) -> Optional[int]:
    if not weight:
        return None
    normalized = str(weight).strip().lower().replace("-", "").replace(" ", "")
    names = {
        "thin": 100, "extralight": 200, "light": 300, "regular": 400,
        "normal": 400, "medium": 500, "semibold": 600, "bold": 700,
        "extrabold": 800, "black": 900,
    }
    if normalized in names:
        return names[normalized]
    try:
        return max(100, min(900, int(float(normalized))))
    except (TypeError, ValueError):
        return None


def _contrast_ratio(clean_rendered_asset: Any, mask: Any, bounds: BBox) -> Optional[float]:
    if clean_rendered_asset is None or mask is None:
        return None
    try:
        import numpy as np
        image = _to_array(np, clean_rendered_asset)
        coverage = np.asarray(mask) > 0
        if image is None or coverage.shape != image.shape[:2]:
            return None
        height, width = image.shape[:2]
        x0, y0 = max(0, bounds.x), max(0, bounds.y)
        x1, y1 = min(width, bounds.x + bounds.width), min(height, bounds.y + bounds.height)
        crop = image[y0:y1, x0:x1].astype("float64")
        ink = coverage[y0:y1, x0:x1]
        if not ink.any() or not (~ink).any():
            return None
        def luminance(rgb):
            channels = rgb / 255.0
            linear = np.where(
                channels <= 0.04045,
                channels / 12.92,
                ((channels + 0.055) / 1.055) ** 2.4,
            )
            return (
                0.2126 * linear[..., 0]
                + 0.7152 * linear[..., 1]
                + 0.0722 * linear[..., 2]
            )

        # A mean ink color makes an outlined glyph look falsely invisible:
        # a pale fill plus dark stroke can average to the background even
        # though the stroke supplies strong legibility. Compare robust
        # luminance bands and retain the strongest substantial text edge.
        ink_luminance = luminance(crop[ink])
        background_luminance = float(np.median(luminance(crop[~ink])))
        candidates = np.percentile(ink_luminance, (10, 50, 90))
        ratios = []
        for candidate in candidates:
            light, dark = sorted(
                (float(candidate), background_luminance), reverse=True
            )
            ratios.append((light + 0.05) / (dark + 0.05))
        return round(max(ratios), 3)
    except Exception:
        return None


def _typography_evidence(
    clean_rendered_asset: Any,
    mask: Any,
    rendered_bounds: Optional[BBox],
    approved_bounds: BBox,
    inst,
    glyph: Dict[str, Any],
    font_registry: Any,
    observed_lines: Optional[int],
) -> Tuple[Dict[str, Any], Optional[float], List[str]]:
    style = inst.style_profile
    expected_size = (
        inst.characteristics.size
        if inst.characteristics and inst.characteristics.size
        else style.font_size if style else None
    )
    expected_weight = _weight_value(style.font_weight if style else None)
    assigned_weight = None
    assigned_font = glyph.get("assigned_font")
    if assigned_font and font_registry is not None:
        key = font_registry._key(assigned_font) if hasattr(font_registry, "_key") else assigned_font
        face = faces_of(font_registry).get(key)
        assigned_weight = face.weight_class if face else None
    expected_alignment = (style.align_h if style and style.align_h else "center")
    source_lines = max(1, len((inst.text or "").splitlines()))
    wrap_enabled = bool(style and (style.transform or {}).get("wrap_text"))
    contrast = _contrast_ratio(clean_rendered_asset, mask, approved_bounds)
    checks: Dict[str, Any] = {
        "expected_weight": expected_weight,
        "assigned_weight": assigned_weight,
        "expected_size_px": expected_size,
        "rendered_height_px": rendered_bounds.height if rendered_bounds else None,
        "rendered_width_px": rendered_bounds.width if rendered_bounds else None,
        "width_fill_ratio": (
            round(rendered_bounds.width / approved_bounds.width, 4)
            if rendered_bounds and approved_bounds.width > 0 else None
        ),
        "expected_alignment": expected_alignment,
        "source_line_count": source_lines,
        "rendered_line_count": observed_lines,
        "wrap_enabled": wrap_enabled,
        "contrast_ratio": contrast,
        "hierarchy_score": None,
    }
    parts: List[float] = []
    flags: List[str] = []
    if expected_weight is not None and assigned_weight is not None:
        difference = abs(expected_weight - assigned_weight)
        weight_score = max(0.0, 100.0 - difference / 4.0)
        checks["weight_score"] = round(weight_score, 2)
        parts.append(weight_score)
        if difference >= 200:
            flags.append("typography_weight_review")
    else:
        checks["weight_score"] = None

    if expected_size and rendered_bounds:
        ratio = rendered_bounds.height / expected_size
        size_score = 100.0 * min(ratio, 1.0 / max(ratio, 1e-6))
        checks["size_ratio"] = round(ratio, 4)
        checks["size_score"] = round(size_score, 2)
        parts.append(size_score)
        if size_score < 70:
            flags.append("typography_scale_review")
    else:
        checks["size_ratio"] = checks["size_score"] = None

    if rendered_bounds and approved_bounds.width > 0:
        left_gap = rendered_bounds.x - approved_bounds.x
        right_gap = approved_bounds.x + approved_bounds.width - (
            rendered_bounds.x + rendered_bounds.width
        )
        tolerance = max(2.0, approved_bounds.width * 0.1)
        aligned = (
            left_gap <= tolerance if expected_alignment == "left"
            else right_gap <= tolerance if expected_alignment == "right"
            else abs(left_gap - right_gap) <= tolerance
        )
        alignment_score = 100.0 if aligned else 60.0
        checks["alignment_score"] = alignment_score
        parts.append(alignment_score)
        if not aligned:
            flags.append("typography_alignment_review")
    else:
        checks["alignment_score"] = None

    if observed_lines is not None and not wrap_enabled:
        line_score = 100.0 if observed_lines == source_lines else 60.0
        checks["line_count_score"] = line_score
        parts.append(line_score)
        if line_score < 100:
            flags.append("typography_line_count_review")
    else:
        checks["line_count_score"] = None

    if contrast is not None:
        contrast_score = min(100.0, 100.0 * max(0.0, contrast - 1.0) / 3.5)
        checks["contrast_score"] = round(contrast_score, 2)
        parts.append(contrast_score)
        if contrast < 1.5:
            flags.append("severe_low_contrast")
        elif contrast < 3.0:
            flags.append("typography_contrast_review")
    else:
        checks["contrast_score"] = None
    score = round(sum(parts) / len(parts), 2) if parts else None
    return checks, score, flags


def _apply_hierarchy_scores(regions: List[VerificationRegion], instances: Dict[str, Any]) -> None:
    candidates = []
    for region in regions:
        inst = instances.get(region.region_id)
        rendered = region.checks.get("geometry", {}).get("rendered_bounds")
        if not inst or not rendered or region.inventory_status != "rendered":
            continue
        source_size = (
            inst.characteristics.size
            if inst.characteristics and inst.characteristics.size
            else inst.bounding_box.height
        )
        candidates.append((region, float(source_size), float(rendered["height"])))
    for region, source_size, rendered_size in candidates:
        comparisons = []
        for other, other_source, other_rendered in candidates:
            if other is region or abs(source_size - other_source) < 1e-6:
                continue
            expected_order = source_size > other_source
            observed_order = rendered_size > other_rendered
            comparisons.append(expected_order == observed_order)
        hierarchy = (
            round(100.0 * sum(comparisons) / len(comparisons), 2)
            if comparisons else None
        )
        typography = region.checks.get("typography", {})
        typography["hierarchy_score"] = hierarchy
        if hierarchy is not None:
            current = region.scores.get("typographic_intent")
            region.scores["typographic_intent"] = (
                round((current + hierarchy) / 2.0, 2) if current is not None else hierarchy
            )
            if hierarchy < 100:
                region.flags.append("typography_hierarchy_review")
                region.recommended_action = (
                    region.recommended_action
                    or "Restore the source design's relative text hierarchy."
                )


_CONTEXT_TOKEN_PATTERN = re.compile(
    r"(?:[$€£¥]?\d+(?:[.,]\d+)?\s*(?:%|kg|g|mg|l|ml|cl|oz|lb|cm|mm|°[cf])?)",
    re.IGNORECASE,
)


def _apply_contextual_scores(
    regions: List[VerificationRegion],
    instances: Dict[str, Any],
    manifest: TextManifest,
    classification: Dict[str, Any],
) -> None:
    asset_class = classification["asset_class"]
    image_height = (manifest.img_dim or (0, 0))[1]
    source_targets: Dict[str, set[str]] = {}
    for inst in instances.values():
        if inst.text and inst.target_text:
            source_targets.setdefault(_normalize(inst.text), set()).add(
                _normalize(inst.target_text)
            )

    for region in regions:
        inst = instances.get(region.region_id)
        if not inst or region.inventory_status != "rendered":
            continue
        source, target = inst.text or "", inst.target_text or ""
        typography = region.checks.get("typography", {})
        geometry = region.checks.get("geometry", {})
        parts: List[float] = []
        checks: Dict[str, Any] = {
            "asset_class": asset_class,
            "classification_confidence": classification.get("confidence"),
            "classification_source": classification.get("source"),
            "surface_class": (
                inst.background_profile.semantic_label
                if inst.background_profile else None
            ),
        }
        failures: List[str] = []

        if asset_class == "sign":
            concise = len(target) <= 40 and len(target.split()) <= 8
            checks["concise"] = concise
            parts.append(100.0 if concise else 30.0)
            if not concise:
                failures.append("target is long for glance-readable signage")
            contrast = typography.get("contrast_ratio")
            if contrast is not None:
                legible = contrast >= 3.0
                checks["distance_contrast_ok"] = legible
                parts.append(100.0 if legible else 50.0)
                if not legible:
                    failures.append("contrast is weak for distance reading")
        elif asset_class == "poster":
            hierarchy = typography.get("hierarchy_score")
            checks["hierarchy_preserved"] = hierarchy == 100.0 if hierarchy is not None else None
            if hierarchy is not None:
                parts.append(float(hierarchy))
                if hierarchy < 100:
                    failures.append("headline hierarchy changed")
            contrast = typography.get("contrast_ratio")
            if contrast is not None:
                parts.append(min(100.0, contrast / 4.5 * 100.0))
                if contrast < 3.0:
                    failures.append("expressive text lost visual prominence")
        elif asset_class == "billboard":
            words = len(target.split())
            concise = words <= 8 and len(target) <= 60
            checks["extreme_brevity"] = concise
            parts.append(100.0 if concise else 40.0)
            if not concise:
                failures.append("copy is too long for billboard viewing")
            rendered = geometry.get("rendered_bounds")
            scale_ratio = (
                rendered["height"] / image_height
                if rendered and image_height > 0 else None
            )
            checks["rendered_height_ratio"] = round(scale_ratio, 4) if scale_ratio is not None else None
            if scale_ratio is not None:
                large = scale_ratio >= 0.08
                parts.append(100.0 if large else 45.0)
                if not large:
                    failures.append("text is too small for large-scale viewing")
        elif asset_class == "product_label":
            source_tokens = set(_CONTEXT_TOKEN_PATTERN.findall(source))
            target_tokens = set(_CONTEXT_TOKEN_PATTERN.findall(target))
            missing_tokens = sorted(source_tokens - target_tokens)
            checks["source_quantity_tokens"] = sorted(source_tokens)
            checks["target_quantity_tokens"] = sorted(target_tokens)
            checks["missing_quantity_tokens"] = missing_tokens
            if source_tokens:
                token_score = 100.0 * (len(source_tokens) - len(missing_tokens)) / len(source_tokens)
                parts.append(token_score)
                if missing_tokens:
                    failures.append("quantities or units were not preserved")
            spatial = region.scores.get("spatial_fit")
            if spatial is not None:
                parts.append(float(spatial))
                if spatial < 80:
                    failures.append("label text does not fit its small safe area")
        elif asset_class == "ui_graphic":
            expansion = len(target) / max(1, len(source))
            checks["length_expansion_ratio"] = round(expansion, 3)
            length_ok = expansion <= 1.35
            parts.append(100.0 if length_ok else max(30.0, 100.0 / expansion))
            if not length_ok:
                failures.append("control label expansion is likely too long")
            variants = source_targets.get(_normalize(source), set())
            consistent = len(variants) <= 1
            checks["terminology_consistent"] = consistent
            parts.append(100.0 if consistent else 40.0)
            if not consistent:
                failures.append("repeated UI terminology is inconsistent")

        score = round(sum(parts) / len(parts), 2) if parts else None
        checks["status"] = (
            "pass" if score is not None and score >= 70
            else "review" if score is not None else "unavailable"
        )
        checks["explanation"] = (
            f"The target fits the {asset_class.replace('_', ' ')} context."
            if not failures
            else "The target is semantically present, but " + "; ".join(failures) + "."
        )
        region.checks["contextual_fit"] = checks
        region.scores["contextual_fit"] = score
        if score is not None and score < 70:
            region.flags.append("contextual_fit_review")
            region.recommended_action = (
                region.recommended_action
                or checks["explanation"]
            )


def _apply_critical_rules(region: VerificationRegion) -> str:
    """Apply non-negotiable caps after every component has supplied evidence."""
    failures: List[str] = []
    if region.inventory_status == "missing":
        failures.append("missing_text")
    if "unsupported_glyphs" in region.flags:
        failures.append("unsupported_glyphs")
    if "unreadable_output" in region.flags:
        failures.append("unreadable_output")
    if "severe_low_contrast" in region.flags:
        failures.append("unreadable_contrast")
    if region.checks.get("geometry", {}).get("overflow_ratio", 0) >= 0.25:
        failures.append("severe_clipping")
    content = region.scores.get("content_integrity")
    if content is not None and content < CONTENT_REVIEW_THRESHOLD * 100:
        failures.append("content_integrity_failure")

    actionable_flags = [
        flag for flag in region.flags if flag != "intentional_omission"
    ]
    status = "fail" if failures else ("review" if actionable_flags else "pass")
    region.checks["critical_rules"] = {
        "status": status,
        "failures": failures,
        "score_cap": 49.0 if failures else (79.0 if actionable_flags else 100.0),
    }
    return status


def _weighted_component_score(scores: Dict[str, Optional[float]]) -> Optional[float]:
    measured = [
        (float(scores[name]), weight)
        for name, weight in COMPONENT_WEIGHTS.items()
        if scores.get(name) is not None
    ]
    if not measured:
        return None
    # Phase 5 precedes contextual QA. Unknown components are excluded and the
    # measured weights are renormalized; no neutral score is invented.
    return round(
        sum(score * weight for score, weight in measured)
        / sum(weight for _, weight in measured),
        2,
    )


def _humanize_flag(flag: str) -> str:
    labels = {
        "missing_regions": "missing text",
        "untranslated_regions": "untranslated text",
        "unsupported_glyphs": "unsupported glyphs",
        "unreadable_output": "unreadable output",
        "content_integrity_failure": "target-text mismatch",
        "spatial_overflow": "text overflow",
        "spatial_collision": "text collisions",
        "severe_low_contrast": "unreadable contrast",
        "typography_hierarchy_review": "changed text hierarchy",
    }
    return labels.get(flag, flag.replace("_", " "))


def _aggregate_presentation(
    project: VerificationProject,
    regions: List[VerificationRegion],
) -> List[VerificationVisualFlag]:
    severity_rank = {"fail": 0, "review": 1, "pass": 2}
    for region in regions:
        rules = region.checks.get("critical_rules", {})
        region.status = rules.get("status", "review")
        if region.inventory_status in {"excluded", "do_not_translate"}:
            region.overall_score = None
            region.status = "pass"
            continue
        raw = _weighted_component_score(region.scores)
        cap = float(rules.get("score_cap", 100.0))
        region.overall_score = min(raw, cap) if raw is not None else None

    actionable = [region for region in regions if region.status != "pass"]
    project.review_order = [
        region.region_id for region in sorted(
            actionable,
            key=lambda region: (
                severity_rank[region.status],
                region.overall_score if region.overall_score is not None else 101.0,
                region.region_id,
            ),
        )
    ]
    visual_flags: List[VerificationVisualFlag] = []
    for region in actionable:
        bounds = region.target_evidence.bounds or region.source_evidence.bounds
        if bounds is None:
            continue
        codes = [
            flag for flag in region.flags if flag != "intentional_omission"
        ]
        visual_flags.append(VerificationVisualFlag(
            region_id=region.region_id,
            bounds=bounds,
            severity=region.status,
            codes=codes,
            color="#dc2626" if region.status == "fail" else "#f59e0b",
            label=_humanize_flag(codes[0]) if codes else region.status,
        ))

    raw_project_score = _weighted_component_score(project.component_scores)
    project_cap = (
        49.0 if any(region.status == "fail" for region in regions)
        else 79.0 if any(region.status == "review" for region in regions)
        else 100.0
    )
    project.overall_score = (
        min(raw_project_score, project_cap)
        if raw_project_score is not None else None
    )
    failed = sum(region.status == "fail" for region in regions)
    review = sum(region.status == "review" for region in regions)
    rendered = sum(region.inventory_status == "rendered" for region in regions)
    primary = ", ".join(
        _humanize_flag(flag) for flag in project.summary_flags[:3]
    )
    if failed:
        project.summary = (
            f"Blocked: {failed} region(s) failed verification"
            + (f"; {review} additional region(s) need review" if review else "")
            + (f". Primary issues: {primary}." if primary else ".")
        )
    elif review:
        project.summary = (
            f"Needs review: {review} region(s) have verification findings"
            + (f". Primary issues: {primary}." if primary else ".")
        )
    else:
        project.summary = (
            f"Ready: {rendered} rendered region(s) passed all available "
            "deterministic checks."
        )
    return visual_flags


def build_verification_report(
    clean_rendered_asset: Any,
    text_manifest: TextManifest,
    font_registry: Any = None,
) -> VerificationReport:
    """Build Phase 1's evidence-first contract and deterministic inventory.

    This must receive Scribe's clean output, before Garnish.  When Scribe's
    per-region ``text_masks`` metadata is available it is authoritative;
    external renderers degrade to target-text plus output-presence evidence.
    """
    masks = getattr(clean_rendered_asset, "text_masks", None)
    rendered_ids = set(masks) if isinstance(masks, dict) else None
    output_present = clean_rendered_asset is not None
    manifest_ids = {inst.id for inst in text_manifest.instances}
    regions: List[VerificationRegion] = []
    context_classification = cicerone.classify_asset_context(text_manifest)
    context_asset_class = context_classification["asset_class"]
    mask_bounds = {
        region_id: _mask_bounds(mask)
        for region_id, mask in (masks.items() if isinstance(masks, dict) else [])
    }
    collisions: Dict[str, List[str]] = {}
    if isinstance(masks, dict):
        try:
            from PIL import ImageChops
            ids = sorted(masks)
            for index, left_id in enumerate(ids):
                for right_id in ids[index + 1:]:
                    if ImageChops.multiply(masks[left_id], masks[right_id]).getbbox():
                        collisions.setdefault(left_id, []).append(right_id)
                        collisions.setdefault(right_id, []).append(left_id)
        except Exception:
            collisions = {}

    for inst in text_manifest.instances:
        target_bounds = inst.adjusted_bbox or inst.bounding_box
        source_lang = inst.language or inst.detected_language or text_manifest.src_lang
        target_lang = inst.target_language or text_manifest.targ_lang or "en"
        source_analysis = julienne.analyse(inst.text, source_lang)
        target_analysis = julienne.analyse(inst.target_text, target_lang)
        source_script = source_analysis.summary if source_analysis.scripts else None
        target_script = target_analysis.summary if target_analysis.scripts else None
        source_direction, target_direction = source_analysis.direction, target_analysis.direction
        # The declared ISO 15924 code for the target language, expanded to the
        # characters it actually admits: Jpan is Han + hiragana + katakana and
        # Kore is Hangul + Han, so correctly written Japanese and Korean are
        # not "mixed-script contamination" and must not be flagged as such.
        script_check = julienne.validate(
            inst.target_text, target_lang, lang_to_script.get(target_lang),
        )
        if scribe._should_render_vertical(
            target_bounds,
            target_lang,
            inst.target_text or "",
            inst.style_profile.target_orientation if inst.style_profile else None,
        ):
            target_direction = "ttb"
        explicitly_rendered = (
            inst.id in rendered_ids if rendered_ids is not None
            else bool(output_present and inst.target_text and not inst.dnt and not inst.excluded)
        )
        if inst.dnt:
            status, flags, action = "do_not_translate", ["intentional_omission"], None
        elif inst.excluded:
            status, flags, action = "excluded", ["intentional_omission"], None
        elif not inst.target_text:
            status, flags, action = (
                "untranslated", ["coverage_untranslated"], "Provide a target translation."
            )
        elif explicitly_rendered:
            status, flags, action = "rendered", [], None
        else:
            status, flags, action = (
                "missing", ["coverage_missing"], "Render the target text for this region."
            )

        surface_class = _asset_class(text_manifest, inst)
        glyph = _resolve_font_evidence(font_registry, inst, target_lang)
        shaping = _shaping_evidence(target_lang, target_script)
        font_family = glyph["assigned_font"]
        rendered_bounds = mask_bounds.get(inst.id)
        mask = masks.get(inst.id) if isinstance(masks, dict) else None
        geometry: Dict[str, Any] = {
            "approved_bounds": asdict(target_bounds),
            "rendered_bounds": asdict(rendered_bounds) if rendered_bounds else None,
            "overflow_pixels": 0,
            "overflow_ratio": 0.0,
            "collisions": collisions.get(inst.id, []),
            "line_count": _line_count(mask, rendered_bounds, target_direction),
        }
        spatial_score: Optional[float] = None
        region_quad = _region_quad(inst, target_bounds)
        geometry["containment_shape"] = "quad" if region_quad else "bbox"
        if explicitly_rendered and mask is not None and rendered_bounds is not None:
            total_ink = _mask_pixel_count(mask)
            if region_quad:
                # Measure against the plane the region was actually placed
                # on. Against the bbox, every correctly warped region would
                # report overflow for doing what it was told.
                inside = _mask_pixel_count_in_quad(mask, region_quad)
            else:
                inside = _mask_pixel_count(mask, (
                    target_bounds.x,
                    target_bounds.y,
                    target_bounds.x + target_bounds.width,
                    target_bounds.y + target_bounds.height,
                ))
            overflow = max(0, total_ink - inside)
            ratio = overflow / total_ink if total_ink else 0.0
            tolerance = QUAD_OVERFLOW_TOLERANCE if region_quad else 0.0
            geometry["overflow_pixels"] = overflow
            geometry["measured_overflow_ratio"] = round(ratio, 4)
            geometry["overflow_tolerance"] = tolerance
            if ratio <= tolerance:
                ratio = 0.0
            geometry["overflow_ratio"] = round(ratio, 4)
            spatial_score = max(0.0, round(100.0 * (1.0 - ratio), 2))
            # `ratio`, not `overflow`: the raw pixel count is still non-zero
            # for a soft quad boundary that the tolerance has already
            # forgiven, and flagging on it would report an overflow the
            # score itself says is not there.
            if ratio:
                flags.append("spatial_overflow")
                action = action or "Adjust the text fit or enlarge the safe region."
            if collisions.get(inst.id):
                flags.append("spatial_collision")
                spatial_score = min(spatial_score, 50.0)
                action = action or "Separate this text from the overlapping region."
            try:
                width, height = mask.size
                touches_canvas = (
                    rendered_bounds.x <= 0 or rendered_bounds.y <= 0
                    or rendered_bounds.x + rendered_bounds.width >= width
                    or rendered_bounds.y + rendered_bounds.height >= height
                )
            except Exception:
                touches_canvas = False
            geometry["touches_canvas_edge"] = touches_canvas
            if touches_canvas:
                flags.append("possible_canvas_clipping")
                spatial_score = min(spatial_score, 70.0)

        direction_check = {
            "expected": target_direction,
            "alignment": inst.style_profile.align_h if inst.style_profile else None,
            "orientation": inst.style_profile.target_orientation if inst.style_profile else None,
            "status": "pass",
        }
        direction_score = 100.0
        if target_direction == "rtl" and direction_check["alignment"] == "left":
            direction_check["status"] = "review"
            direction_score = 70.0
            flags.append("rtl_alignment_review")
            action = action or "Use right or neutral alignment for this RTL region."
        if target_direction == "ttb" and direction_check["orientation"] == "horizontal":
            direction_check["status"] = "review"
            direction_score = min(direction_score, 70.0)
            flags.append("vertical_direction_review")
            action = action or "Confirm horizontal orientation for this vertical-script region."

        if glyph["status"] == "fail":
            flags.append("unsupported_glyphs")
            action = "Choose a font that covers every target codepoint."
        elif glyph["overrode_explicit_choice"]:
            # Only a substitution that overrode a HUMAN's pick is a finding.
            # An auto region has no font to betray -- "auto" means "choose one
            # that works" -- and moving off the Latin default is the correct
            # answer for every CJK target, not a defect.  Flagging it capped
            # zh/ja/ko at review on flawless renders (coverage 1.0, CER 0.0)
            # while Latin passed, so every Asian locale read amber forever.
            flags.append("glyph_fallback")
            action = action or (
                f"The chosen font could not set this text; {glyph['assigned_font']} "
                "was substituted. Review it against the intended typography."
            )
        if shaping["required"] and not shaping["available"]:
            flags.append("shaping_unavailable")
            action = action or "Enable the required shaping engine for this script."
        # Script validity is part of "script and rendering validity" by name,
        # and it is the only check that can see an orthography error: both
        # Chinese orthographies are the same Unicode script and the same
        # glyphs are present in either font, so 舊牆街 in a zh-cn region
        # passes coverage, passes OCR, and is still the wrong writing system.
        script_score = 100.0
        if script_check.get("han_variant_mismatch"):
            script_score = 40.0
            flags.append("han_variant_mismatch")
            action = action or (
                f"Target declares {script_check['declared_script']} but the text is "
                f"{script_check['han_variant']}; supply the correct orthography."
            )
        elif script_check.get("unexpected_scripts"):
            script_score = 60.0
            flags.append("unexpected_script")
            action = action or (
                "Unexpected "
                + ", ".join(script_check["unexpected_scripts"])
                + " characters for this target language; confirm the translation."
            )
        validity_parts = [direction_score, script_score]
        if glyph["status"] != "unknown":
            validity_parts.append(100.0 if glyph["status"] == "pass" else 0.0)
        if shaping["required"]:
            validity_parts.append(100.0 if shaping["available"] else 0.0)
        validity_score = round(sum(validity_parts) / len(validity_parts), 2)
        target_ocr = (
            _target_ocr_evidence(
                clean_rendered_asset, inst, rendered_bounds or target_bounds, target_lang
            )
            if status == "rendered"
            else {
                "status": "not_applicable",
                "expected": inst.target_text,
                "recognized": None,
                "normalized_expected": _normalize(inst.target_text or ""),
                "normalized_recognized": None,
                "similarity": None,
                "confidence": None,
                "engine": "easyocr",
            }
        )
        content_score = (
            round(float(target_ocr["similarity"]) * 100.0, 2)
            if target_ocr.get("similarity") is not None else None
        )
        if target_ocr["status"] == "review":
            flags.append("content_integrity_review")
            action = action or "Compare the rendered text with the intended target."
        elif target_ocr["status"] == "fail":
            if not target_ocr.get("recognized"):
                flags.append("unreadable_output")
                action = "Increase legibility and render the complete target text."
            else:
                flags.append("content_integrity_failure")
                action = "Correct the rendered target so it matches the intended text."
        if (
            target_ocr.get("confidence") is not None
            and target_ocr["confidence"] < OCR_CONFIDENCE_REVIEW
        ):
            flags.append("target_ocr_low_confidence")
            action = action or "Manually review this low-confidence OCR result."
        typography, typography_score, typography_flags = _typography_evidence(
            clean_rendered_asset,
            mask,
            rendered_bounds,
            target_bounds,
            inst,
            glyph,
            font_registry,
            geometry["line_count"],
        )
        flags.extend(typography_flags)
        if "severe_low_contrast" in typography_flags:
            action = "Increase text/background contrast until the target is readable."
        elif typography_flags:
            action = action or "Review weight, scale, alignment, and line structure."

        regions.append(VerificationRegion(
            region_id=inst.id,
            inventory_status=status,
            source_evidence=VerificationEvidence(
                text=inst.text,
                bounds=inst.bounding_box,
                language=source_lang,
                script=source_script,
                scripts=source_analysis.scripts,
                script_counts=source_analysis.counts,
                is_mixed_script=source_analysis.is_mixed,
                han_variant=source_analysis.han_variant,
                direction=source_direction,
                asset_class=context_asset_class,
            ),
            target_evidence=VerificationEvidence(
                text=inst.target_text,
                bounds=target_bounds,
                language=target_lang,
                script=target_script,
                scripts=target_analysis.scripts,
                script_counts=target_analysis.counts,
                is_mixed_script=target_analysis.is_mixed,
                han_variant=target_analysis.han_variant,
                direction=target_direction,
                font_family=font_family,
                asset_class=context_asset_class,
                clean_render_present=explicitly_rendered,
            ),
            checks={
                "glyph_coverage": glyph,
                "script_validity": script_check,
                "shaping": shaping,
                "directionality": direction_check,
                "geometry": geometry,
                "target_ocr": target_ocr,
                "typography": typography,
                "cicerone_context": {
                    "asset_class": context_asset_class,
                    "surface_class": surface_class,
                    "classification_confidence": context_classification.get("confidence"),
                },
            },
            scores={
                "coverage": 100.0 if status in {"rendered", "excluded", "do_not_translate"} else 0.0,
                "content_integrity": content_score,
                "spatial_fit": spatial_score,
                "typographic_intent": typography_score,
                "script_rendering_validity": validity_score,
                "contextual_fit": None,
            },
            flags=flags,
            confidence=inst.confidence,
            recommended_action=action,
        ))

    for region_id in sorted((rendered_ids or set()) - manifest_ids):
        regions.append(VerificationRegion(
            region_id=region_id,
            inventory_status="extra",
            source_evidence=VerificationEvidence(),
            target_evidence=VerificationEvidence(clean_render_present=True),
            scores={"coverage": 0.0},
            flags=["coverage_extra"],
            recommended_action="Remove the rendered region or add it to the manifest.",
        ))

    _apply_hierarchy_scores(
        regions, {inst.id: inst for inst in text_manifest.instances}
    )
    _apply_contextual_scores(
        regions,
        {inst.id: inst for inst in text_manifest.instances},
        text_manifest,
        context_classification,
    )
    rule_statuses = [_apply_critical_rules(region) for region in regions]
    totals = dict(Counter(region.inventory_status for region in regions))
    gaps = totals.get("missing", 0) + totals.get("untranslated", 0) + totals.get("extra", 0)
    intended = sum(
        totals.get(key, 0) for key in ("rendered", "missing", "untranslated")
    )
    coverage = 100.0 if intended == 0 else round(100.0 * totals.get("rendered", 0) / intended, 2)
    flags = []
    if totals.get("missing"):
        flags.append("missing_regions")
    if totals.get("untranslated"):
        flags.append("untranslated_regions")
    if totals.get("extra"):
        flags.append("extra_rendered_regions")
    scored_regions = [
        region for region in regions
        if region.inventory_status == "rendered"
    ]
    spatial_values = [
        region.scores["spatial_fit"] for region in scored_regions
        if region.scores.get("spatial_fit") is not None
    ]
    validity_values = [
        region.scores["script_rendering_validity"] for region in scored_regions
        if region.scores.get("script_rendering_validity") is not None
    ]
    content_values = [
        region.scores["content_integrity"] for region in scored_regions
        if region.scores.get("content_integrity") is not None
    ]
    typography_values = [
        region.scores["typographic_intent"] for region in scored_regions
        if region.scores.get("typographic_intent") is not None
    ]
    contextual_values = [
        region.scores["contextual_fit"] for region in scored_regions
        if region.scores.get("contextual_fit") is not None
    ]
    if any("unsupported_glyphs" in region.flags for region in regions):
        flags.append("unsupported_glyphs")
    if any("shaping_unavailable" in region.flags for region in regions):
        flags.append("shaping_unavailable")
    if any("spatial_overflow" in region.flags for region in regions):
        flags.append("spatial_overflow")
    if any("spatial_collision" in region.flags for region in regions):
        flags.append("spatial_collision")
    if any("unreadable_output" in region.flags for region in regions):
        flags.append("unreadable_output")
    if any("content_integrity_failure" in region.flags for region in regions):
        flags.append("content_integrity_failure")
    if any("content_integrity_review" in region.flags for region in regions):
        flags.append("content_integrity_review")
    if any("severe_low_contrast" in region.flags for region in regions):
        flags.append("severe_low_contrast")
    if any("typography_hierarchy_review" in region.flags for region in regions):
        flags.append("typography_hierarchy_review")
    if any("contextual_fit_review" in region.flags for region in regions):
        flags.append("contextual_fit_review")
    critical = "fail" in rule_statuses
    review_needed = "review" in rule_statuses
    project = VerificationProject(
        overall_status="fail" if critical else ("review" if review_needed else "pass"),
        component_scores={
            "coverage": coverage,
            "content_integrity": (
                round(sum(content_values) / len(content_values), 2)
                if content_values else None
            ),
            "spatial_fit": (
                round(sum(spatial_values) / len(spatial_values), 2)
                if spatial_values else None
            ),
            "typographic_intent": (
                round(sum(typography_values) / len(typography_values), 2)
                if typography_values else None
            ),
            "script_rendering_validity": (
                round(sum(validity_values) / len(validity_values), 2)
                if validity_values else None
            ),
            "contextual_fit": (
                round(sum(contextual_values) / len(contextual_values), 2)
                if contextual_values else None
            ),
        },
        summary_flags=flags,
        region_totals=totals,
    )
    visual_flags = _aggregate_presentation(project, regions)
    return VerificationReport(
        project=project,
        regions=regions,
        visual_flags=visual_flags,
        run_metadata={
            "contract_version": "1.0",
            "asset_id": text_manifest.asset_id,
            "stage": "post_scribe_pre_garnish",
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "clean_render_available": output_present,
            "component_weights": COMPONENT_WEIGHTS,
            "asset_classification": context_classification,
            "pending_components": [
                name for name, score in project.component_scores.items()
                if score is None
            ],
        },
    )


def _normalize(text: str) -> str:
    """Normalize compatibility forms before ignoring layout punctuation.

    NFKC folds full-width Latin/digits, Arabic presentation forms, and other
    script-specific compatibility variants into their intended characters.
    OCR round-trip then judges readable content rather than whitespace or
    punctuation placement.
    """
    compatible = unicodedata.normalize("NFKC", text or "")
    compatible = compatible.replace("\u0640", "")  # Arabic tatweel is ornamental
    return re.sub(r"[\W_]+", "", compatible, flags=re.UNICODE).casefold()


def _normalize_words(text: str) -> List[str]:
    """NFKC-fold and split on whitespace, keeping word boundaries intact.

    Distinct from ``_normalize``, which strips every separator: word error
    rate needs the separators the character rate deliberately discards.
    """
    compatible = unicodedata.normalize("NFKC", text or "")
    compatible = compatible.replace("\u0640", "")
    return [w for w in re.split(r"\s+", compatible.casefold()) if w]


def _edit_distance(reference: Sequence[Any], hypothesis: Sequence[Any]) -> int:
    """Levenshtein distance (Levenshtein 1966), two-row dynamic programming.

    Works over characters for CER and over word tokens for WER; the algorithm
    is identical and only the unit changes.
    """
    if not reference:
        return len(hypothesis)
    if not hypothesis:
        return len(reference)
    previous = list(range(len(hypothesis) + 1))
    for i, ref_item in enumerate(reference, start=1):
        current = [i]
        for j, hyp_item in enumerate(hypothesis, start=1):
            current.append(min(
                previous[j] + 1,                                  # deletion
                current[j - 1] + 1,                               # insertion
                previous[j - 1] + (ref_item != hyp_item),         # substitution
            ))
        previous = current
    return previous[-1]


def error_rates(expected: str, recognized: str, lang: Optional[str] = None) -> Dict[str, Any]:
    """Character and word error rates for one OCR round-trip.

    CER is the primary metric for EVERY language, and the reason is
    measurable: the previous score was ``SequenceMatcher.ratio()``, whose
    value depends on string length, so one identical single-character error
    cost Japanese 0.571 and German 0.952 -- an eight-fold difference in
    penalty for the same defect.  CER normalises by reference length, which
    is the standard practice in speech and OCR evaluation precisely because
    it makes short and long references comparable.

    WER is reported only for scripts that delimit words.  Chinese, Japanese
    and Korean writing does not put spaces between words, so "words" there
    would be an artifact of whatever segmenter was chosen rather than a
    property of the text.  It is evidence for a reviewer, never a score, so
    that the number the six components see stays comparable across all
    languages.
    """
    from tofu.layers import julienne

    reference = _normalize(expected)
    hypothesis = _normalize(recognized)
    result: Dict[str, Any] = {
        "cer": None, "wer": None, "cer_edits": None, "reference_length": len(reference),
        "word_segmentable": False,
    }
    if not reference:
        return result
    edits = _edit_distance(reference, hypothesis)
    result["cer_edits"] = edits
    # An unbounded CER (hypothesis far longer than reference) is still a
    # total failure; clamping keeps the derived score inside 0-100.
    result["cer"] = round(min(1.0, edits / len(reference)), 4)

    scripts = set(julienne.analyse(expected, lang).scripts)
    segmentable = bool(scripts) and not (scripts & _UNSEGMENTED_SCRIPTS)
    result["word_segmentable"] = segmentable
    if segmentable:
        ref_words, hyp_words = _normalize_words(expected), _normalize_words(recognized)
        if ref_words:
            result["wer"] = round(
                min(1.0, _edit_distance(ref_words, hyp_words) / len(ref_words)), 4
            )
    return result


def _text_similarity(expected: str, recognized: str, lang: Optional[str] = None) -> float:
    """Agreement in 0-1, as ``1 - CER``.

    Kept as a single number because every caller and the persisted evidence
    contract already speak in agreement rather than error.
    """
    a = _normalize(expected)
    if not a:
        return NEUTRAL_SCORE
    if not _normalize(recognized):
        return 0.0
    rates = error_rates(expected, recognized, lang)
    return 1.0 - (rates["cer"] if rates["cer"] is not None else 1.0)


def _get_reader(lang: str):
    """EasyOCR reader for the instance's target language (cached)."""
    try:
        import easyocr
        from tofu.layers.cicerone import _to_easyocr_lang
    except ImportError:
        return None
    code = _to_easyocr_lang(lang or "en")
    key = (code,) if code == "en" else (code, "en")
    if key not in _readers:
        try:
            _readers[key] = easyocr.Reader(list(key), gpu=False, verbose=False)
        except Exception:
            # unknown code or missing model: latin fallback still measures
            # gross failures (blank crop, clipped render)
            if ("en",) not in _readers:
                _readers[("en",)] = easyocr.Reader(["en"], gpu=False, verbose=False)
            _readers[key] = _readers[("en",)]
    return _readers[key]


def _ocr_roundtrip_score(np, localized_np, inst, targ_lang: str) -> Optional[float]:
    """similarity between the intended target text and what OCR reads
    back from the rendered crop. None if the metric cannot run."""
    b = inst.adjusted_bbox or inst.bounding_box
    h, w = localized_np.shape[:2]
    x0 = max(0, b.x - CROP_PAD_PX)
    y0 = max(0, b.y - CROP_PAD_PX)
    x1 = min(w, b.x + b.width + CROP_PAD_PX)
    y1 = min(h, b.y + b.height + CROP_PAD_PX)
    if x1 - x0 < 4 or y1 - y0 < 4:
        return None
    reader = _get_reader(inst.target_language or targ_lang)
    if reader is None:
        return None
    try:
        results = reader.readtext(localized_np[y0:y1, x0:x1])
    except Exception:
        return None
    recognized = " ".join(r[1] for r in results)
    return _text_similarity(inst.target_text or "", recognized)


def _ink_presence_score(np, source_np, localized_np, inst) -> Optional[float]:
    """fallback legibility evidence: change inside the bbox (text was
    erased + re-rendered ⇒ pixels moved) geometrically combined with
    edge energy (glyphs create gradients a flat fill cannot)."""
    if source_np is None or source_np.shape != localized_np.shape:
        return None
    b = inst.adjusted_bbox or inst.bounding_box
    h, w = localized_np.shape[:2]
    x0, y0 = max(0, b.x), max(0, b.y)
    x1, y1 = min(w, b.x + b.width), min(h, b.y + b.height)
    if x1 - x0 < 4 or y1 - y0 < 4:
        return None
    src = source_np[y0:y1, x0:x1].astype("float64")
    loc = localized_np[y0:y1, x0:x1].astype("float64")
    change = float(np.abs(loc - src).mean())          # 0 ⇒ nothing rendered
    lum = 0.299 * loc[..., 0] + 0.587 * loc[..., 1] + 0.114 * loc[..., 2]
    gy, gx = np.gradient(lum)
    edge = float(np.hypot(gx, gy).mean())             # glyph stroke energy
    # saturating ramps: full credit at modest but unambiguous evidence
    change_s = min(1.0, change / 6.0)
    edge_s = min(1.0, edge / 4.0)
    return (change_s * edge_s) ** 0.5


def _ssim(np, a, b) -> float:
    """single-window SSIM (Wang et al. 2004, eq. 13) over a pixel set.
    the windowed mean-map variant needs scipy; the global form is exact
    for our use — the ring is one homogeneous comparison unit."""
    a = a.astype("float64")
    b = b.astype("float64")
    c1, c2 = (0.01 * 255) ** 2, (0.03 * 255) ** 2
    mu_a, mu_b = a.mean(), b.mean()
    va, vb = a.var(), b.var()
    cov = ((a - mu_a) * (b - mu_b)).mean()
    return float(
        ((2 * mu_a * mu_b + c1) * (2 * cov + c2))
        / ((mu_a ** 2 + mu_b ** 2 + c1) * (va + vb + c2))
    )


def _ring_ssim_score(np, source_np, localized_np, inst) -> Optional[float]:
    """SSIM between source and localized over the border ring around the
    bbox — pixels the pipeline had no license to change."""
    if source_np.shape != localized_np.shape:
        return None
    b = inst.adjusted_bbox or inst.bounding_box
    h, w = source_np.shape[:2]
    x0, y0 = max(0, b.x - RING_PX), max(0, b.y - RING_PX)
    x1, y1 = min(w, b.x + b.width + RING_PX), min(h, b.y + b.height + RING_PX)
    if x1 - x0 < 4 or y1 - y0 < 4:
        return None
    ring = np.ones((y1 - y0, x1 - x0), dtype=bool)
    iy0, ix0 = max(0, b.y - y0), max(0, b.x - x0)
    ring[iy0:iy0 + b.height, ix0:ix0 + b.width] = False
    if not ring.any():
        return None
    # luminance (ITU-R BT.601) — SSIM is defined on intensity
    def lum(img):
        crop = img[y0:y1, x0:x1]
        return (0.299 * crop[..., 0] + 0.587 * crop[..., 1] + 0.114 * crop[..., 2])
    return _ssim(np, lum(source_np)[ring], lum(localized_np)[ring])


def _bbox_crop(np, image, inst) -> Optional[Any]:
    """Return an in-frame instance crop, or None for a degenerate bbox."""
    if image is None:
        return None
    b = inst.adjusted_bbox or inst.bounding_box
    h, w = image.shape[:2]
    x0, y0 = max(0, b.x), max(0, b.y)
    x1, y1 = min(w, b.x + b.width), min(h, b.y + b.height)
    if x1 - x0 < 3 or y1 - y0 < 3:
        return None
    return image[y0:y1, x0:x1]


def _edge_orientation_histogram(np, crop, mask) -> Optional[Tuple[Any, float]]:
    """Normalized Sobel-orientation histogram plus mean edge energy.

    Orientation captures glyph shape; the companion energy preserves the
    sharp-versus-distressed distinction that orientation alone cannot see.
    """
    if crop is None or mask is None or mask.shape != crop.shape[:2] or not mask.any():
        return None
    lum = 0.299 * crop[..., 0] + 0.587 * crop[..., 1] + 0.114 * crop[..., 2]
    gy, gx = np.gradient(lum.astype("float64"))
    magnitude = np.hypot(gx, gy)
    usable = mask & (magnitude > 1e-6)
    if not usable.any():
        return None
    orientation = (np.arctan2(gy, gx) + np.pi) % (2 * np.pi)
    hist, _ = np.histogram(
        orientation[usable], bins=16, range=(0.0, 2 * np.pi), weights=magnitude[usable],
    )
    total = float(hist.sum())
    return (hist / total, float(magnitude[usable].mean())) if total > 0 else None


def _garnish_edge_similarity_score(np, source_np, localized_np, inst) -> Optional[float]:
    """Compare source and localized glyph-edge orientation distributions.

    This deliberately measures physical edge character rather than OCR text:
    the words may differ, but a distressed, blurred, or sharp treatment should
    retain a comparable distribution of local edge directions.
    """
    if source_np is None or localized_np is None or source_np.shape != localized_np.shape:
        return None
    src_crop = _bbox_crop(np, source_np, inst)
    loc_crop = _bbox_crop(np, localized_np, inst)
    if src_crop is None or loc_crop is None:
        return None
    b = inst.adjusted_bbox or inst.bounding_box
    src_edges = _edge_orientation_histogram(np, src_crop, _text_mask(source_np, b))
    loc_edges = _edge_orientation_histogram(np, loc_crop, _text_mask(localized_np, b))
    if src_edges is None or loc_edges is None:
        return None
    src_hist, src_energy = src_edges
    loc_hist, loc_energy = loc_edges
    orientation_similarity = float(np.minimum(src_hist, loc_hist).sum())
    energy_similarity = min(src_energy, loc_energy) / max(src_energy, loc_energy, 1e-6)
    return float(orientation_similarity * energy_similarity)


def _ring_texture_statistics(np, image, inst) -> Optional[Tuple[float, float]]:
    """Edge density and chroma spread in the same non-text ring as SSIM."""
    if image is None:
        return None
    b = inst.adjusted_bbox or inst.bounding_box
    h, w = image.shape[:2]
    x0, y0 = max(0, b.x - RING_PX), max(0, b.y - RING_PX)
    x1, y1 = min(w, b.x + b.width + RING_PX), min(h, b.y + b.height + RING_PX)
    if x1 - x0 < 4 or y1 - y0 < 4:
        return None
    ring = np.ones((y1 - y0, x1 - x0), dtype=bool)
    iy0, ix0 = max(0, b.y - y0), max(0, b.x - x0)
    ring[iy0:iy0 + b.height, ix0:ix0 + b.width] = False
    if not ring.any():
        return None
    crop = image[y0:y1, x0:x1].astype("float64")
    lum = 0.299 * crop[..., 0] + 0.587 * crop[..., 1] + 0.114 * crop[..., 2]
    gy, gx = np.gradient(lum)
    # A scale-relative threshold makes this stable over both dark and bright
    # signage without pulling in another image-processing dependency.
    edge_density = float((np.hypot(gx, gy)[ring] > 12.0).mean())
    chroma = crop.max(axis=2) - crop.min(axis=2)
    chroma_std = float(chroma[ring].std())
    return edge_density, chroma_std


def _garnish_texture_match_score(np, source_np, localized_np, inst) -> Optional[float]:
    """Compare local edge density and chroma variance around the text."""
    if source_np is None or localized_np is None or source_np.shape != localized_np.shape:
        return None
    source_stats = _ring_texture_statistics(np, source_np, inst)
    localized_stats = _ring_texture_statistics(np, localized_np, inst)
    if source_stats is None or localized_stats is None:
        return None
    source_edge, source_chroma = source_stats
    localized_edge, localized_chroma = localized_stats
    # Each term is normalized independently so a colourful mural cannot hide
    # an edge-density regression (or vice versa).
    edge_distance = abs(source_edge - localized_edge) / max(source_edge, localized_edge, 0.05)
    chroma_distance = abs(source_chroma - localized_chroma) / max(source_chroma, localized_chroma, 8.0)
    return float(max(0.0, 1.0 - (edge_distance + chroma_distance) / 2.0))


def _outside_mask_preservation_score(np, cleansed_np, localized_np, inst) -> Optional[float]:
    """Check that non-glyph bbox pixels still equal the cleansed base."""
    if cleansed_np is None or localized_np is None or cleansed_np.shape != localized_np.shape:
        return None
    mask = _text_mask(localized_np, inst.adjusted_bbox or inst.bounding_box)
    clean_crop = _bbox_crop(np, cleansed_np, inst)
    localized_crop = _bbox_crop(np, localized_np, inst)
    if mask is None or clean_crop is None or localized_crop is None or mask.shape != clean_crop.shape[:2]:
        return None
    non_text = ~mask
    if not non_text.any():
        return None
    mean_abs_diff = float(np.abs(
        localized_crop.astype("float64")[non_text] - clean_crop.astype("float64")[non_text]
    ).mean())
    return float(max(0.0, 1.0 - mean_abs_diff / 255.0))


def _residual_source_text_score(np, cleansed_np, inst, targ_lang: str) -> Optional[float]:
    """re-OCR the CLEANSED (pre-scribe) crop and compare against the
    ORIGINAL source string. returns a SIMILARITY (1.0 = source text
    still fully readable = a failed erase), not a quality score — the
    caller inverts it into a penalty. None if the metric cannot run."""
    if cleansed_np is None or not inst.text:
        return None
    b = inst.bounding_box
    h, w = cleansed_np.shape[:2]
    x0 = max(0, b.x - CROP_PAD_PX)
    y0 = max(0, b.y - CROP_PAD_PX)
    x1 = min(w, b.x + b.width + CROP_PAD_PX)
    y1 = min(h, b.y + b.height + CROP_PAD_PX)
    if x1 - x0 < 4 or y1 - y0 < 4:
        return None
    try:
        from tofu.layers.ocr_verification import PaddleRegionVerifier
        result = PaddleRegionVerifier(
            languages=[inst.language or inst.detected_language or targ_lang]
        ).verify_regions(
            cleansed_np,
            [inst.bounding_box],
            [inst.text],
            inst.language or inst.detected_language or targ_lang,
        )[0]
    except Exception:
        return None
    if result.state in {"unavailable", "error"}:
        return None
    return result.similarity if result.similarity is not None else 0.0


def _rgb_to_lab(rgb) -> Tuple[float, float, float]:
    """sRGB (0-255) -> CIE Lab (D65), the standard path for a perceptual
    ΔE. no new dependency: this is the textbook piecewise formula."""
    def to_linear(c: float) -> float:
        c = c / 255.0
        return c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4
    r, g, b = (to_linear(c) for c in rgb[:3])
    x = r * 0.4124 + g * 0.3576 + b * 0.1805
    y = r * 0.2126 + g * 0.7152 + b * 0.0722
    z = r * 0.0193 + g * 0.1192 + b * 0.9505
    xn, yn, zn = 0.95047, 1.0, 1.08883

    def f(t: float) -> float:
        return t ** (1 / 3) if t > 0.008856 else (7.787 * t + 16 / 116)
    fx, fy, fz = f(x / xn), f(y / yn), f(z / zn)
    return (116 * fy - 16, 500 * (fx - fy), 200 * (fy - fz))


def _delta_e_cie76(rgb1, rgb2) -> float:
    l1, a1, b1 = _rgb_to_lab(rgb1)
    l2, a2, b2 = _rgb_to_lab(rgb2)
    return ((l1 - l2) ** 2 + (a1 - a2) ** 2 + (b1 - b2) ** 2) ** 0.5


def _parse_hex_color(color: Optional[str]) -> Optional[Tuple[int, int, int]]:
    if not color or not color.startswith("#"):
        return None
    h = color.lstrip("#")
    if len(h) == 3:
        h = "".join(c * 2 for c in h)
    if len(h) < 6:
        return None
    try:
        return (int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16))
    except ValueError:
        return None


def _style_color_score(np, localized_np, inst) -> Optional[float]:
    """CIE76 ΔE between the detected source text color
    (style_profile.color) and the mean ink color actually rendered,
    sampled via the shared glyph mask so verify agrees with cleanse/
    typography on which pixels are ink, not background."""
    sp = inst.style_profile
    target = _parse_hex_color(sp.color if sp else None)
    if target is None:
        return None
    b = inst.adjusted_bbox or inst.bounding_box
    mask = _text_mask(localized_np, b)
    if mask is None or not mask.any():
        return None
    h, w = localized_np.shape[:2]
    x0, y0 = max(0, b.x), max(0, b.y)
    x1, y1 = min(w, b.x + b.width), min(h, b.y + b.height)
    crop = localized_np[y0:y1, x0:x1]
    ink_color = tuple(float(c) for c in crop[mask].reshape(-1, 3).mean(axis=0))
    delta_e = _delta_e_cie76(ink_color, target)
    return float(max(0.0, 1.0 - delta_e / COLOR_DELTA_E_SCALE))


def _style_size_score(np, localized_np, inst) -> Optional[float]:
    """rendered ink height (from the shared glyph mask) vs. the source
    text's detected height (Phase 1 typography) — flags text rendered
    much smaller/larger than the original signage."""
    ch = inst.characteristics
    if not ch or not ch.size or ch.size <= 0:
        return None
    b = inst.adjusted_bbox or inst.bounding_box
    mask = _text_mask(localized_np, b)
    if mask is None or not mask.any():
        return None
    ys, _ = np.nonzero(mask)
    rendered_h = float(ys.max() - ys.min() + 1)
    ratio = rendered_h / ch.size
    return max(0.0, 1.0 - abs(1.0 - ratio))


def _to_array(np, asset) -> Optional[Any]:
    """PIL image | ndarray | path → RGB ndarray, else None."""
    if asset is None:
        return None
    if isinstance(asset, np.ndarray):
        return asset
    try:
        from PIL import Image
        if isinstance(asset, str):
            with Image.open(asset) as im:
                return np.asarray(im.convert("RGB"))
        if hasattr(asset, "convert"):
            return np.asarray(asset.convert("RGB"))
    except Exception:
        return None
    return None


def assess(
    localized_asset: Any,
    text_manifest: TextManifest,
    source_asset: Any = None,
    cleansed_asset: Any = None,
) -> QAReport:
    """assess localization quality of the rendered asset.

    args:
        localized_asset: output of scribe.render().
        text_manifest: the manifest used to produce it.
        source_asset: the original asset (path or image); enables the
            background-reconstruction SSIM metric when provided.
        cleansed_asset: output of cleanse.erase(), i.e. the asset BEFORE
            scribe drew the target text; enables the residual-source-
            text metric (was the erase actually complete?) when provided.

    returns:
        QAReport with overall_score and per-asset/per-instance scores.
    """
    try:
        import numpy as np
    except ImportError:
        np = None

    localized_np = _to_array(np, localized_asset) if np is not None else None
    source_np = _to_array(np, source_asset) if np is not None else None
    cleansed_np = _to_array(np, cleansed_asset) if np is not None else None

    per_instance: Dict[str, float] = {}
    ocr_scores: Dict[str, float] = {}
    ink_scores: Dict[str, float] = {}
    ssim_scores: Dict[str, float] = {}
    residual_scores: Dict[str, float] = {}
    color_scores: Dict[str, float] = {}
    size_scores: Dict[str, float] = {}
    garnish_edge_scores: Dict[str, float] = {}
    garnish_texture_scores: Dict[str, float] = {}
    outside_mask_scores: Dict[str, float] = {}
    recommendations: List[str] = []
    targ_lang = text_manifest.targ_lang or "en"

    total = len(text_manifest.instances)
    dnt_count = sum(1 for i in text_manifest.instances if i.dnt)
    excluded_count = sum(1 for i in text_manifest.instances if i.excluded)
    untranslated_count = sum(
        1 for i in text_manifest.instances
        if not i.dnt and not i.excluded and not i.target_text
    )
    fallback_count = sum(1 for i in text_manifest.instances if i.glyph_fallback)
    translated_count = total - dnt_count - excluded_count - untranslated_count

    for inst in text_manifest.instances:
        if inst.dnt or inst.excluded:
            per_instance[inst.id] = NEUTRAL_SCORE  # correctly excluded, not a failure
            continue
        if not inst.target_text:
            # real detected text that was never addressed: a genuine
            # coverage gap, not a neutral non-event — the old NEUTRAL
            # score let an incomplete localization inflate its own gate
            per_instance[inst.id] = UNTRANSLATED_SCORE
            recommendations.append(
                f"{inst.id}: no translation provided — this region was left "
                "untranslated and excluded from rendering."
            )
            continue

        parts: List[Tuple[float, float]] = []  # (score, weight)
        if localized_np is not None:
            ocr = _ocr_roundtrip_score(np, localized_np, inst, targ_lang)
            if ocr is not None:
                ocr_scores[inst.id] = round(ocr, 4)
                parts.append((ocr, OCR_WEIGHT))
                if ocr < 0.5:
                    recommendations.append(
                        f"{inst.id}: rendered text is not legible to OCR "
                        f"(expected '{inst.target_text}') — check font size, "
                        "contrast, or bbox clipping."
                    )
            else:
                ink = _ink_presence_score(np, source_np, localized_np, inst)
                if ink is not None:
                    ink_scores[inst.id] = round(ink, 4)
                    parts.append((ink, OCR_WEIGHT))
                    if ink < 0.5:
                        recommendations.append(
                            f"{inst.id}: little or no rendering evidence in "
                            "the region (blank or invisible text) — check "
                            "font resolution and color contrast."
                        )
            if source_np is not None:
                ssim = _ring_ssim_score(np, source_np, localized_np, inst)
                if ssim is not None:
                    ssim_scores[inst.id] = round(ssim, 4)
                    parts.append((max(0.0, ssim), SSIM_WEIGHT))
                    if ssim < 0.7:
                        recommendations.append(
                            f"{inst.id}: background around the region was "
                            "altered (halo/spill) — inspect cleanse output."
                        )
            style_parts = []
            color = _style_color_score(np, localized_np, inst)
            if color is not None:
                color_scores[inst.id] = round(color, 4)
                style_parts.append(color)
                if color < 0.5:
                    recommendations.append(
                        f"{inst.id}: rendered text color diverges noticeably "
                        "from the detected source color."
                    )
            size = _style_size_score(np, localized_np, inst)
            if size is not None:
                size_scores[inst.id] = round(size, 4)
                style_parts.append(size)
                if size < 0.5:
                    recommendations.append(
                        f"{inst.id}: rendered text size diverges noticeably "
                        "from the detected source size."
                    )
            if style_parts:
                parts.append((sum(style_parts) / len(style_parts), STYLE_WEIGHT))

            # Garnish is optional and its profile may be inherited from the
            # containing Scene region.  Reuse the compositor's lookup and
            # activation rules so Verify never scores an effect that Scribe/
            # Garnish could not have applied.
            profile = garnish._profile(text_manifest, inst)
            if garnish._active(profile):
                garnish_parts = []
                edge = _garnish_edge_similarity_score(np, source_np, localized_np, inst)
                if edge is not None:
                    garnish_edge_scores[inst.id] = round(float(edge), 4)
                    garnish_parts.append(edge)
                    if edge < 0.5:
                        recommendations.append(
                            f"{inst.id}: garnish edge treatment does not yet match "
                            "the source text's physical edge character."
                        )
                texture = _garnish_texture_match_score(np, source_np, localized_np, inst)
                if texture is not None:
                    garnish_texture_scores[inst.id] = round(float(texture), 4)
                    garnish_parts.append(texture)
                    if texture < 0.5:
                        recommendations.append(
                            f"{inst.id}: the local texture around the garnish differs "
                            "substantially from the source scene."
                        )
                preserved = _outside_mask_preservation_score(np, cleansed_np, localized_np, inst)
                if preserved is not None:
                    outside_mask_scores[inst.id] = round(float(preserved), 4)
                    garnish_parts.append(preserved)
                    if preserved < 0.5:
                        recommendations.append(
                            f"{inst.id}: garnish changed non-text pixels inside its "
                            "region; reduce blur, smudge, or dilation."
                        )
                if garnish_parts:
                    parts.append((sum(garnish_parts) / len(garnish_parts), GARNISH_WEIGHT))

        raw_score = (
            sum(s * w for s, w in parts) / sum(w for _, w in parts)
            if parts else NEUTRAL_SCORE
        )

        # residual source text: a genuine double-exposure defect that
        # multiplies the score down rather than averaging in as just
        # another positive-evidence term — good new-text legibility
        # doesn't excuse the original text still being visible underneath
        residual = _residual_source_text_score(np, cleansed_np, inst, targ_lang)
        if residual is not None:
            residual_scores[inst.id] = round(residual, 4)
            if residual > RESIDUAL_PENALTY_THRESHOLD:
                raw_score *= max(0.0, 1.0 - residual)
                recommendations.append(
                    f"{inst.id}: original source text may still be visible "
                    f"under the translation (residual similarity {residual:.0%}) "
                    "— check cleanse output."
                )

        per_instance[inst.id] = float(raw_score)

    # per-instance scores must be plain python floats -- a numpy scalar
    # (e.g. from a mean()/max() chain in one of the pixel-based scorers)
    # is JSON-unserializable and would 500 the whole render response;
    # this normalizes regardless of which scorer produced the value
    per_instance = {k: float(v) for k, v in per_instance.items()}
    overall = (
        sum(per_instance.values()) / len(per_instance)
        if per_instance else NEUTRAL_SCORE
    )
    overall = float(overall)

    scorer = []
    if ocr_scores:
        scorer.append("ocr-roundtrip")
    if ink_scores:
        scorer.append("ink-presence")
    if ssim_scores:
        scorer.append("ring-ssim")
    if residual_scores:
        scorer.append("residual-text")
    if color_scores or size_scores:
        scorer.append("style-consistency")
    if garnish_edge_scores or garnish_texture_scores or outside_mask_scores:
        scorer.append("garnish")
    return QAReport(
        overall_score=overall,
        per_asset_instance_score={text_manifest.asset_id: per_instance},
        progress={
            "instances_assessed": len(per_instance),
            "regions_total": total,
            "dnt": dnt_count,
            "excluded": excluded_count,
            "translated": translated_count,
            "untranslated": untranslated_count,
            "rendered": translated_count,
            "fallback_font": fallback_count,
        },
        metrics={
            "scorer": "+".join(scorer) if scorer else "neutral",
            "ocr_roundtrip": ocr_scores,
            "ink_presence": ink_scores,
            "ring_ssim": ssim_scores,
            "residual_text": residual_scores,
            "style_color": color_scores,
            "style_size": size_scores,
            "garnish_edge": garnish_edge_scores,
            "garnish_texture": garnish_texture_scores,
            "outside_mask": outside_mask_scores,
        },
        recommendations=recommendations,
    )
