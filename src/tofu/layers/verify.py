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
from difflib import SequenceMatcher
from typing import Any, Dict, List, Optional, Tuple

from tofu.core.types import TextManifest, QAReport
from tofu.utils.imaging import text_mask as _text_mask

NEUTRAL_SCORE = 1.0        # DNT: correctly excluded, not a failure to judge
UNTRANSLATED_SCORE = 0.0   # real detected text never addressed -- a genuine coverage gap
OCR_WEIGHT = 0.7           # legibility dominates: unreadable text is a failed localization
SSIM_WEIGHT = 0.3
STYLE_WEIGHT = 0.2
RING_PX = 12                    # border ring width for background-reconstruction SSIM
CROP_PAD_PX = 4                 # slack around the bbox for the OCR crop
RESIDUAL_PENALTY_THRESHOLD = 0.3  # below this, treat as OCR noise, not a real leak
COLOR_DELTA_E_SCALE = 40.0      # ΔE at which the color-consistency score bottoms out at 0

# lazily-built EasyOCR readers, keyed by language tuple — reader init is
# expensive (~seconds), scoring runs per render
_readers: Dict[Tuple[str, ...], Any] = {}


def _normalize(text: str) -> str:
    """casefold + collapse everything non-word: OCR round-trip judges
    legibility, not punctuation or spacing fidelity."""
    return re.sub(r"[\W_]+", "", text, flags=re.UNICODE).casefold()


def _text_similarity(expected: str, recognized: str) -> float:
    a, b = _normalize(expected), _normalize(recognized)
    if not a:
        return NEUTRAL_SCORE
    if not b:
        return 0.0
    return SequenceMatcher(None, a, b).ratio()


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
    b = inst.bounding_box
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
    b = inst.bounding_box
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
    b = inst.bounding_box
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
    reader = _get_reader(inst.language or inst.detected_language or targ_lang)
    if reader is None:
        return None
    try:
        results = reader.readtext(cleansed_np[y0:y1, x0:x1])
    except Exception:
        return None
    recognized = " ".join(r[1] for r in results)
    return _text_similarity(inst.text, recognized)


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
    b = inst.bounding_box
    mask = _text_mask(localized_np, b)
    if mask is None or not mask.any():
        return None
    h, w = localized_np.shape[:2]
    x0, y0 = max(0, b.x), max(0, b.y)
    x1, y1 = min(w, b.x + b.width), min(h, b.y + b.height)
    crop = localized_np[y0:y1, x0:x1]
    ink_color = crop[mask].reshape(-1, 3).mean(axis=0)
    delta_e = _delta_e_cie76(ink_color, target)
    return max(0.0, 1.0 - delta_e / COLOR_DELTA_E_SCALE)


def _style_size_score(np, localized_np, inst) -> Optional[float]:
    """rendered ink height (from the shared glyph mask) vs. the source
    text's detected height (Phase 1 typography) — flags text rendered
    much smaller/larger than the original signage."""
    ch = inst.characteristics
    if not ch or not ch.size or ch.size <= 0:
        return None
    b = inst.bounding_box
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
    recommendations: List[str] = []
    targ_lang = text_manifest.targ_lang or "en"

    total = len(text_manifest.instances)
    dnt_count = sum(1 for i in text_manifest.instances if i.dnt)
    untranslated_count = sum(
        1 for i in text_manifest.instances if not i.dnt and not i.target_text
    )
    fallback_count = sum(1 for i in text_manifest.instances if i.glyph_fallback)
    translated_count = total - dnt_count - untranslated_count

    for inst in text_manifest.instances:
        if inst.dnt:
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

        per_instance[inst.id] = raw_score

    overall = (
        sum(per_instance.values()) / len(per_instance)
        if per_instance else NEUTRAL_SCORE
    )

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
    return QAReport(
        overall_score=overall,
        per_asset_instance_score={text_manifest.asset_id: per_instance},
        progress={
            "instances_assessed": len(per_instance),
            "regions_total": total,
            "dnt": dnt_count,
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
        },
        recommendations=recommendations,
    )
