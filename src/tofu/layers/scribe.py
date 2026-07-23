## 🍢 Scribe — Layer 4
## vieuxtiful
"""
style-aware text regeneration layer.

scribe renders target-language text onto the cleansed asset, matching
the source styling captured by scene. scribe is deliberately
time-ignorant: it consumes a static RenderParams per region. the
future video compositor animates scribe's output by wrapping
RenderParams in a CameraTrack (entity anchor, trajectory, duration).

implementation: Pillow-based renderer. per region it renders
inst.target_text (untranslated regions are skipped), wrapped to fit the
bounding box (greedy word-wrap for space-delimited scripts, character-
wrap for CJK) and sized by binary search over both the wrap and the
font size together, colored/faded per StyleProfil + opacity, optionally
rotated, and alpha-composited onto the asset. if Pillow is unavailable
or the asset cannot be loaded as an image, the asset passes through
unchanged (contract preserved).

italic is rendered on a layer sized to the text's OWN extent and
sheared around its own local origin before compositing — not the whole
base-image-sized layer sheared by each pixel's absolute y-coordinate,
which was a real bug (see _render_line_layer): a shear formula applied
globally shifts text horizontally by an amount proportional to WHERE in
the image it sits, not how tall it is, which measurably bled a region's
ink far outside its detection bbox (Phase 3 measurement: region r3 on
the stylized-italic fixture, ring-SSIM 0.106 vs 0.8+ for every sibling
region, unrelated to cleanse and traced here).
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from tofu.core.types import TextManifest, RenderParams, BBox, StyleProfil

FALLBACK_FONTS = ("arial.ttf", "DejaVuSans.ttf", "segoeui.ttf")
MIN_FONT_PX = 6
ITALIC_SHEAR = 0.2  # x' = x + ITALIC_SHEAR * y, applied LOCAL to the text layer

# Font file cache: _get_font is called in a binary search loop (up to ~50
# iterations per instance) and each call re-reads the .ttf/.ttc from disk.
# The cache keys on (font_family, size) and is bounded to avoid unbounded
# growth across many font sizes.
_font_cache: Dict[Tuple[Optional[str], int], Any] = {}
_FONT_CACHE_MAX = 256

# vertical CJK column rendering: a region is treated as a vertical column
# when its bbox is narrow-and-tall AND the effective target language uses
# vertical writing conventions. the aspect threshold mirrors cicerone's
# own merge_vertical_columns() member test (COLUMN_MAX_ASPECT=1.6 for a
# CHARACTER; a merged multi-character COLUMN region is taller still), so
# a region that survived cicerone's column merge reliably crosses it.
CJK_VERTICAL_LANGS = {"ja", "zh-cn", "zh-tw", "zh-hk", "zh-mo", "zh-sg"}
VERTICAL_ASPECT_MIN = 1.3
VERTICAL_ROW_FACTOR = 1.15  # row height as a multiple of font size


def _pixel(value: float) -> int:
    """Snap a final paint coordinate to the image raster.

    Layout stays floating-point for accurate fitting and centering.  A final
    fractional origin, however, makes otherwise identical glyphs change edge
    coverage as a user nudges a capture box by one pixel.
    """
    return int(round(value))


def _pixel_bbox(bbox: BBox) -> BBox:
    """Normalize externally edited geometry before text rasterization."""
    return BBox(
        x=_pixel(bbox.x), y=_pixel(bbox.y),
        width=max(1, _pixel(bbox.width)), height=max(1, _pixel(bbox.height)),
    )


def _apply_style_transform(layer: Any, bbox: BBox, transform: Optional[Dict[str, Any]]) -> Any:
    """Apply deterministic, local text shaping before the detected rotation.

    The transform is deliberately confined to the region's text layer: a
    global affine transform would make the result depend on where the region
    lies on the canvas.  Arc is a lightweight baseline warp, not a perspective
    reconstruction, and therefore remains stable for preview and final render.
    """
    if not transform:
        return layer
    try:
        from PIL import Image
        import math
        skew_x_deg = max(-45.0, min(45.0, float(transform.get("skew_x", 0) or 0)))
        skew_y_deg = max(-45.0, min(45.0, float(transform.get("skew_y", 0) or 0)))
        sx = math.tan(math.radians(skew_x_deg))
        sy = math.tan(math.radians(skew_y_deg))
        arc = float(transform.get("arc", 0) or 0)
        preset = str(transform.get("preset", "custom") or "custom").strip().lower().replace(" ", "_")
        amount = float(transform.get("amount", arc) or 0)
        scale_x = max(0.4, min(2.5, float(transform.get("scale_x", 1) or 1)))
        scale_y = max(0.4, min(2.5, float(transform.get("scale_y", 1) or 1)))
        if preset == "none":
            preset, amount, arc = "custom", 0.0, 0.0
        offset_x = int(transform.get("offset_x", 0) or 0)
        offset_y = int(transform.get("offset_y", 0) or 0)
        if not (sx or sy or arc or amount or scale_x != 1 or scale_y != 1 or offset_x or offset_y):
            return layer

        def _quality_affine(src, coefficients):
            """Affine-transform one glyph layer with adaptive supersampling.

            Text is first rasterised by Pillow at its native target pixels.
            Repeatedly rescaling that finished raster is what produced the
            staircase edges reported by the localized canvas.  Restricting a
            supersampled pass to this instance's ink bounds keeps the detail
            in the original glyph coverage while avoiding a 16x full-image
            allocation for a large source asset.  The final LANCZOS reduction
            retains smooth contours at the actual output raster.
            """
            ink = src.getbbox()
            if not ink:
                return src
            ix0, iy0, ix1, iy1 = ink
            # Scale, shear and named warps may move ink beyond its initial
            # tight extent.  This deliberately generous local pad avoids
            # clipping an unwrapped run while remaining region-local.
            extent = max(ix1 - ix0, iy1 - iy0, bbox.width, bbox.height)
            shear_pad = abs(sx) * bbox.height + abs(sy) * bbox.width
            pad = int(max(16, extent * 1.6 + abs(offset_x) + abs(offset_y) + abs(amount) + shear_pad))
            left, top = max(0, ix0 - pad), max(0, iy0 - pad)
            right, bottom = min(src.width, ix1 + pad), min(src.height, iy1 + pad)
            if right <= left or bottom <= top:
                return src
            crop = src.crop((left, top, right, bottom))
            # Four samples per target pixel is normally enough for glyph
            # edges.  Adapt down rather than risking a pathological memory
            # spike for a very large selected text region.
            max_high_pixels = 18_000_000
            factor = min(4, max(2, int((max_high_pixels / max(1, crop.width * crop.height)) ** .5)))
            a, b, c, d, e, f = coefficients
            # Convert the global inverse map to the crop's local coordinate
            # system, then to its supersampled coordinate system.
            local = (
                a, b, (a * left + b * top + c - left) * factor,
                d, e, (d * left + e * top + f - top) * factor,
            )
            high = crop.resize((crop.width * factor, crop.height * factor), Image.Resampling.LANCZOS)
            transformed = high.transform(high.size, Image.Transform.AFFINE, local, resample=Image.Resampling.BICUBIC)
            reduced = transformed.resize(crop.size, Image.Resampling.LANCZOS)
            result = Image.new("RGBA", src.size, (0, 0, 0, 0))
            result.alpha_composite(reduced, (left, top))
            return result

        def _apply_offset(src):
            """Shift a layer by (offset_x, offset_y).

            paste WITHOUT a mask argument copies RGBA directly on a
            transparent destination (dest_alpha = src_alpha), preserving
            anti-aliased edge pixels. The previous mask=src form squared
            semi-transparent alphas (dest_alpha = src_alpha^2 / 255) and
            produced jagged, darkened edges on every positional nudge."""
            if not (offset_x or offset_y):
                return src
            shifted = Image.new("RGBA", src.size, (0, 0, 0, 0))
            shifted.paste(src, (offset_x, offset_y))
            return shifted
        anchor = str(transform.get("skew_anchor", "center") or "center").strip().lower()
        cx, cy = bbox.x + bbox.width / 2, bbox.y + bbox.height / 2
        if "left" in anchor: cx = bbox.x
        elif "right" in anchor: cx = bbox.x + bbox.width
        if "top" in anchor: cy = bbox.y
        elif "bottom" in anchor: cy = bbox.y + bbox.height
        if sx or sy or scale_x != 1 or scale_y != 1:
            # Compose inverse scale and inverse shear into one sampling pass.
            # Pillow maps destination coordinates back to source coordinates;
            # combining K * S^-1 avoids the former double resample when both
            # skew and stretch were active.  Note that y-shear depends on
            # the inverse X scale (not inverse Y scale).
            inv_x, inv_y = 1 / scale_x, 1 / scale_y
            out = _quality_affine(layer, (
                inv_x, -sx * inv_y, cx * (1 - inv_x) + sx * cy * inv_y,
                -sy * inv_x, inv_y, cy * (1 - inv_y) + sy * cx * inv_x,
            ))
        else:
            out = layer
        if not (arc or amount):
            return _apply_offset(out)
        # Shift each local column by a quadratic amount.  The work window is
        # the actual ink extent, not the captured anchor: transformed text
        # can legitimately cross an adjacent region boundary.  Include enough
        # vertical room for the largest arc displacement without allocating a
        # full-canvas remap for a small text run.
        ink = out.getbbox()
        if not ink:
            return _apply_offset(out)
        ix0, iy0, ix1, iy1 = ink
        warp_pad = int(max(2, abs(amount if amount else arc) + 2))
        x0, y0 = max(0, ix0 - 2), max(0, iy0 - warp_pad)
        x1, y1 = min(out.width, ix1 + 2), min(out.height, iy1 + warp_pad)
        warped = Image.new("RGBA", out.size, (0, 0, 0, 0))
        def offset_at(t: float) -> float:
            """Bounded Photoshop-style *deterministic* warp approximations.

            These describe text placement only; they never invent image
            pixels.  That makes the same named preset safe to use in the
            debounced canvas and final Scribe composition.  ``custom``
            retains the original quadratic Arc slider for compatibility.
            """
            magnitude = amount if amount else arc
            u = (t + 1.0) / 2.0  # 0..1 left-to-right
            if preset in {"custom", "arc", "arc_lower"}:
                return magnitude * (t * t - 1.0)
            if preset in {"arc_upper", "arch", "shell_upper", "rise"}:
                return -magnitude * (t * t - 1.0) if preset != "rise" else magnitude * (u - .5)
            if preset in {"bulge", "inflate", "fisheye"}:
                return -magnitude * (1.0 - t * t)
            if preset in {"shell_lower", "squeeze"}:
                return magnitude * (1.0 - t * t)
            if preset in {"flag", "wave", "fish"}:
                cycles = 1.0 if preset == "flag" else 2.0
                return magnitude * math.sin(cycles * math.pi * u)
            if preset == "twist":
                return magnitude * t * (1.0 - abs(t))
            return magnitude * (t * t - 1.0)

        region_w = x1 - x0
        region_h = y1 - y0
        if region_w <= 0 or region_h <= 0:
            return _apply_offset(out)
        import cv2
        import numpy as np
        region_arr = np.asarray(out.crop((x0, y0, x1, y1)))
        map_x = np.tile(np.arange(region_w, dtype=np.float32), (region_h, 1))
        map_y = np.zeros((region_h, region_w), dtype=np.float32)
        for i in range(region_w):
            t = (x0 + i - cx) / max(1.0, bbox.width / 2)
            dy = offset_at(t)
            map_y[:, i] = np.arange(region_h, dtype=np.float32) - dy
        # BORDER_TRANSPARENT leaves an uninitialised destination in OpenCV
        # when no explicit destination array is supplied, which made two
        # identical preview renders occasionally differ at the warp boundary.
        remapped = cv2.remap(region_arr, map_x, map_y, cv2.INTER_LANCZOS4,
                             borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0, 0))
        warped = Image.new("RGBA", out.size, (0, 0, 0, 0))
        warped.alpha_composite(Image.fromarray(remapped, "RGBA"), (x0, y0))
        return _apply_offset(warped)
    except Exception:
        return layer


def _load_image(asset: Any):
    """accept a PIL image, file path, or bytes; return RGBA image or None."""
    try:
        from PIL import Image
    except ImportError:
        return None
    if hasattr(asset, "convert") and hasattr(asset, "size"):  # PIL image
        return asset.convert("RGBA")
    if isinstance(asset, (str, Path)):
        try:
            return Image.open(asset).convert("RGBA")
        except Exception:
            return None
    if isinstance(asset, (bytes, bytearray)):
        import io
        try:
            return Image.open(io.BytesIO(asset)).convert("RGBA")
        except Exception:
            return None
    return None


def _get_font(font_family: Optional[str], size: int):
    """resolve a truetype font: explicit family/path first, then fallbacks.

    FontRegistry indexes collection files (.ttc/.otc) as "path#faceindex"
    (multiple font faces share one file); PIL has no such notation — it
    takes the face index as a separate `index=` argument and raises
    OSError on a literal "#N" suffix in the path string. without this
    split, any registry-resolved collection face (a very common case:
    CJK gothic/mincho families typically ship as .ttc) silently failed
    to load and fell through to the Latin-only FALLBACK_FONTS chain —
    exactly defeating both resolve_face() and check_glyph_coverage().
    """
    cache_key = (font_family, size)
    cached = _font_cache.get(cache_key)
    if cached is not None:
        return cached
    from PIL import ImageFont
    candidates = ([font_family] if font_family else []) + list(FALLBACK_FONTS)
    for cand in candidates:
        path, index = cand, 0
        if isinstance(cand, str) and "#" in cand:
            base, _, idx = cand.rpartition("#")
            if idx.isdigit():
                path, index = base, int(idx)
        try:
            font = ImageFont.truetype(path, size, index=index)
            if len(_font_cache) < _FONT_CACHE_MAX:
                _font_cache[cache_key] = font
            return font
        except Exception:
            continue
    font = ImageFont.load_default()
    if len(_font_cache) < _FONT_CACHE_MAX:
        _font_cache[cache_key] = font
    return font


def _default_fallback_path(fonts: Dict[str, Any]) -> Optional[str]:
    """first installed FALLBACK_FONTS entry, as a registry key — what
    _get_font(None, ...) resolves to structurally, before any per-text
    glyph-coverage override. shared by resolve_face/check_glyph_coverage/
    resolve_auto_font so the three "what does auto mean" call sites can't
    drift against each other."""
    for fallback_name in FALLBACK_FONTS:
        match = next(
            (p for p in fonts if Path(p.split("#")[0]).name.lower() == fallback_name.lower()),
            None,
        )
        if match:
            return match
    return None


def resolve_face(
    font_registry, font_family: Optional[str], weight: Optional[str], italic: bool,
) -> Tuple[Optional[str], bool]:
    """best real font FILE for (font_family, weight, italic), when a
    FontRegistry is available — returns (resolved_path, needs_synthetic_
    italic). a real italic FACE renders correctly at its own metrics; a
    synthetic shear (the fallback) widens the glyph beyond what the fit
    pass measured and can still leave a few px of edge softening even
    after the shear-pivot fix, so a real face is strictly better when one
    exists in the same family as font_family.

    font_family may already be an exact match (a specific weight/italic
    file the user picked, or matches what's requested) — that combination
    is returned as-is without a family search. Otherwise siblings of
    font_family's own family are searched for a subfamily name containing
    the requested weight/italic keywords.

    font_family=None ("auto") with weight or italic actually requested
    (e.g. typography detected bold/italic source text but the region was
    left on auto font selection — the common case) anchors the sibling
    search on whichever FALLBACK_FONTS entry _get_font(None, ...) would
    otherwise draw with, so a detected bold/italic lands on a real bold/
    italic FILE of that same fallback family instead of being silently
    dropped. font_family=None with NEITHER weight nor italic requested
    stays a true no-op (nothing to resolve towards). no registry, no
    match, or nothing requested: returns (font_family, italic) unchanged
    — the caller's existing behavior (literal path + synthetic shear) is
    the contract.
    """
    if not font_registry or not getattr(font_registry, "fonts", None):
        return font_family, italic
    fonts = getattr(font_registry, "_fonts", {})

    lookup_family = font_family
    if not lookup_family:
        if not (weight or italic):
            return font_family, italic
        lookup_family = _default_fallback_path(fonts)
        if not lookup_family:
            return font_family, italic

    key = font_registry._key(lookup_family) if hasattr(font_registry, "_key") else lookup_family
    current = fonts.get(key)
    if current is None:
        return font_family, italic  # not a registry-known font; nothing to resolve

    def requested_weight(value: Optional[str]) -> Optional[int]:
        """Map OS/2 classes and human face names to one CSS-like target.

        Families use different names for 900: Black, Heavy, Ultra, and Extra
        Black are all valid, distinct installed faces.  Persist the actual
        subfamily for display but resolve it by its numeric class so selecting
        Arial Heavy/Black can never be mistaken for Regular.
        """
        if value is None:
            return None
        raw = str(value).strip().lower()
        if raw.isdigit():
            return max(100, min(900, int(raw)))
        if any(token in raw for token in ("black", "heavy", "ultra", "extra bold", "extrabold")):
            return 900 if any(token in raw for token in ("black", "heavy", "ultra")) else 800
        if "bold" in raw:
            return 700
        if any(token in raw for token in ("semi", "demi")):
            return 600
        if "medium" in raw:
            return 500
        if "light" in raw or "thin" in raw:
            return 300 if "light" in raw else 100
        if any(token in raw for token in ("regular", "normal", "book", "roman")):
            return 400
        return None

    target_weight = requested_weight(weight)
    sub = (current.subfamily or "").lower()
    has_italic = "italic" in sub or "oblique" in sub
    current_weight = int(getattr(current, "weight_class", 400) or 400)
    # An explicit selected path is authoritative when it is already close to
    # the requested class.  This fixes a former bug where "Bold" was not
    # normalized, causing the resolver to replace arialbd.ttf with Regular.
    if (target_weight is None or abs(current_weight - target_weight) <= 80) and italic == has_italic:
        return font_family, False  # already the right face; no synthesis needed

    siblings = [fc for fc in fonts.values() if fc.family == current.family]
    matching_style = [
        fc for fc in siblings
        if ("italic" in (fc.subfamily or "").lower() or "oblique" in (fc.subfamily or "").lower()) == italic
    ]
    if not matching_style:
        # There is no real italic/upright sibling. Keep the selected face and
        # let the caller apply its established synthetic italic fallback.
        return font_family, italic
    if target_weight is None:
        # An italic-only request keeps the current class rather than making a
        # silent Regular substitution.
        target_weight = current_weight
    best = min(
        matching_style,
        key=lambda fc: (abs(int(getattr(fc, "weight_class", 400) or 400) - target_weight),
                        abs(int(getattr(fc, "weight_class", 400) or 400) - current_weight)),
        default=None,
    )
    if best is not None:
        return best.font_path, False
    return font_family, italic  # no matching sibling face; synthesize instead


def check_glyph_coverage(
    font_registry, font_family: Optional[str], text: str, lang: Optional[str],
) -> Tuple[Optional[str], bool]:
    """the ToFU render-time guard: does font_family actually have every
    codepoint in `text`? returns (replacement_font_path_or_None,
    all_covered). when coverage is incomplete, searches every registry
    font for the best real match against this EXACT text (not just the
    language's generic script sample — a specific string can need
    characters a script-level sample doesn't probe) and returns its path
    if it's a different, better choice than font_family; the pipeline
    logs this as a warning and the manifest instance is flagged
    glyph_fallback=True so the UI can surface it — the alternative is
    silent tofu-block glyphs reaching the user with a passing pre-flight.
    """
    if not font_registry or not text:
        return None, True
    fonts = getattr(font_registry, "_fonts", {})
    if not fonts:
        return None, True

    key = font_registry._key(font_family) if font_family and hasattr(font_registry, "_key") else None
    current = fonts.get(key) if key else None
    if current is None and font_family is None:
        # "auto" -- no explicit selection. _get_font(None, ...) resolves
        # to the first of FALLBACK_FONTS it can load; without checking
        # THAT font specifically, every auto-styled region looked like a
        # coverage failure regardless of whether the actual fallback font
        # covers the text fine (measured: flat-sign's plain English text,
        # rendered with arial.ttf via the default chain, was flagged
        # glyph_fallback on all 4 regions). resolve by filename since the
        # registry keys on full discovered paths, not the bare names in
        # FALLBACK_FONTS.
        match = _default_fallback_path(fonts)
        if match:
            key, current = match, fonts[match]

    # only a KNOWN registry font with a confirmed gap counts as evidence
    # of a real problem; an unrecognized font_family (e.g. a bare name
    # outside the registry, and no fallback match either) gives no
    # evidence either way, so it must not be reported as a coverage
    # failure without proof
    known_gap = False
    if current is not None:
        known_gap = any(ord(ch) not in current.codepoints for ch in text)
        if not known_gap:
            return None, True

    from tofu.layers.tofu import lang_to_script
    script = lang_to_script.get(lang) if lang else None
    extra_chars = set(text)
    if script:
        scored = sorted(
            ((path, font_registry.coverage(path, script, extra_chars=extra_chars))
             for path in fonts),
            key=lambda t: -t[1],
        )
    else:
        # no language hint: score purely on this text's own characters
        scored = sorted(
            (
                (path, sum(1 for ch in extra_chars if ord(ch) in fc.codepoints) / len(extra_chars))
                for path, fc in fonts.items()
            ),
            key=lambda t: -t[1],
        )
    if not scored or scored[0][1] <= 0:
        return None, not known_gap  # nothing better known anywhere
    best_path, best_cov = scored[0]
    if best_path == key or best_cov <= 0:
        return None, not known_gap  # already the best available; can't improve
    return best_path, False


def resolve_auto_font(
    font_registry, lang: Optional[str], text: str,
    weight: Optional[str] = None, italic: bool = False,
) -> Optional[str]:
    """what does "auto" (font_family=None) actually resolve to for THIS
    region's own text/language/weight/italic? render() resolves auto in
    two composed steps — resolve_face(None, weight, italic)'s sibling
    search anchored on the FALLBACK_FONTS chain (so a detected/requested
    bold or italic on an "auto" font selection lands on a real bold/
    italic FILE rather than being silently dropped), then
    check_glyph_coverage()'s override — using THAT resolved path, exactly
    as render() does once resolve_face has reassigned it — when the pick
    can't cover the text. this helper performs the exact same steps so
    its answer is guaranteed to agree with what actually gets drawn.
    exists for callers (capture-time enrichment, manifest autosave) that
    need to SHOW the user what "auto" means before a render ever happens
    — e.g. the Translate-tab preview and the Font column label — without
    duplicating font-resolution rules in a second language.

    returns a FontRegistry path (matching what familiesByLang lookups
    expect, not a bare filename), or None when there's no font_registry
    to resolve against or no FALLBACK_FONTS entry is actually installed.
    """
    if not font_registry or not text:
        return None
    fonts = getattr(font_registry, "_fonts", {})
    if not fonts:
        return None

    default_path = None
    if weight or italic:
        default_path, _ = resolve_face(font_registry, None, weight, italic)
    if default_path is None:
        default_path = _default_fallback_path(fonts)

    replacement, all_covered = check_glyph_coverage(font_registry, default_path, text, lang)
    if not all_covered and replacement:
        return replacement
    return default_path


def _line_height(font) -> float:
    try:
        ascent, descent = font.getmetrics()
        return float(ascent + descent)
    except Exception:
        return float(getattr(font, "size", 12)) * 1.2


def _wrap_lines(draw, text: str, font, max_width: float,
               tracking: float = 0, kerning: float = 0) -> List[str]:
    """greedy wrap targeting max_width: word-wrap when the text has
    space-delimited words, character-wrap otherwise (CJK and other
    scripts that don't use spaces between words).  tracking and kerning
    are included in the width estimate so wrapping matches the drawn
    result."""
    if not text:
        return [""]
    use_words = " " in text.strip()
    units = text.split(" ") if use_words else list(text)
    sep = " " if use_words else ""
    lines: List[str] = []
    cur = ""
    for unit in units:
        candidate = f"{cur}{sep}{unit}" if cur else unit
        width = draw.textlength(candidate, font=font)
        # account for inter-character spacing adjustments
        n = len(candidate)
        width += (tracking + kerning) * max(0, n - 1)
        if width <= max_width or not cur:
            cur = candidate
        else:
            lines.append(cur)
            cur = unit
    if cur or not lines:
        lines.append(cur)
    return lines


def _fit_wrapped(
    draw, text: str, bbox: BBox, font_family: Optional[str],
    explicit_size: Optional[int] = None, leading: Optional[float] = None,
    tracking: float = 0, kerning: float = 0, wrap_text: bool = False,
) -> Tuple[Any, List[str], float]:
    """Fit text with opt-in wrapping.

    When wrapping is off, a string remains one real glyph run: an explicit
    font size is never silently reflowed just because it crosses a cube edge.
    The caller can then clip it using the existing truncation controls.  When
    enabled, greedy word/character wrapping is used only once its measured
    extent crosses the cube width.
    """
    def layout(font):
        return _wrap_lines(draw, text, font, bbox.width, tracking, kerning) if wrap_text else [text or ""]

    if explicit_size and explicit_size > 0:
        font = _get_font(font_family, explicit_size)
        lines = layout(font)
        spacing = leading if leading is not None else _line_height(font) * 0.2
        return font, lines, spacing

    lo, hi = MIN_FONT_PX, max(MIN_FONT_PX + 1, bbox.height * 2)
    best_font = _get_font(font_family, MIN_FONT_PX)
    best_lines = layout(best_font)
    best_spacing = leading if leading is not None else _line_height(best_font) * 0.2
    while lo <= hi:
        mid = (lo + hi) // 2
        font = _get_font(font_family, mid)
        lines = layout(font)
        spacing = leading if leading is not None else _line_height(font) * 0.2
        # fit check uses the ACTUAL rendered ink extent (PIL's own
        # multiline layout), not font.getmetrics()'s nominal ascent+
        # descent — that nominal box accounts for glyphs the string may
        # not even contain (e.g. descenders), which over-estimates height
        # for all-caps/no-descender text and picks an unnecessarily
        # smaller font (measured: "SALE" fit at size 53 instead of the
        # correct 69, visibly shrinking a region that fit fine as-is)
        l, t, r, b = draw.multiline_textbbox((0, 0), "\n".join(lines), font=font, spacing=spacing)
        # font size is determined by raw ink extent only — tracking and
        # kerning widen the line but must not shrink the chosen size,
        # otherwise increasing spacing silently shrinks the glyphs
        if (r - l) <= bbox.width and (b - t) <= bbox.height:
            best_font, best_lines, best_spacing, lo = font, lines, spacing, mid + 1
        else:
            hi = mid - 1
    return best_font, best_lines, best_spacing


def _should_render_vertical(bbox: BBox, lang: Optional[str], text: str, orientation: Optional[str] = None) -> bool:
    """a region renders as a stacked vertical CJK column when its bbox is
    narrow-and-tall and the effective target language uses vertical
    writing — the shape cicerone's merge_vertical_columns() produces for
    genuine stacked signage. a single-character region is excluded (an
    aspect ratio near 1:1 for one glyph gives no reliable signal either
    way, and a lone character reads identically in either orientation)."""
    if orientation == "vertical":
        return True
    if orientation == "horizontal":
        return False
    if not lang or lang not in CJK_VERTICAL_LANGS:
        return False
    if bbox.width <= 0 or len(text) < 2:
        return False
    return bbox.height >= bbox.width * VERTICAL_ASPECT_MIN


def _vertical_block_height(font, n_chars: int, tsume: float) -> float:
    row_h = font.size * VERTICAL_ROW_FACTOR
    compress = (tsume or 0) * font.size * 0.1
    return row_h * n_chars - compress * max(0, n_chars - 1)


def _fit_vertical(
    draw, text: str, bbox: BBox, font_family: Optional[str],
    explicit_size: Optional[int] = None, tsume: float = 0.0,
) -> Any:
    """binary-search the largest font size whose stacked-column layout
    (row height ~1.15x font size per character, tsume-compressed) fits
    bbox — the vertical analogue of _fit_wrapped's horizontal search."""
    if explicit_size and explicit_size > 0:
        return _get_font(font_family, explicit_size)
    lo, hi = MIN_FONT_PX, max(MIN_FONT_PX + 1, bbox.width * 2)
    best = _get_font(font_family, MIN_FONT_PX)
    while lo <= hi:
        mid = (lo + hi) // 2
        font = _get_font(font_family, mid)
        max_w = max(
            (draw.textbbox((0, 0), ch, font=font)[2] - draw.textbbox((0, 0), ch, font=font)[0]
             for ch in text),
            default=0,
        )
        total_h = _vertical_block_height(font, len(text), tsume)
        if max_w <= bbox.width and total_h <= bbox.height:
            best, lo = font, mid + 1
        else:
            hi = mid - 1
    return best


def _render_vertical_layer(
    base_size, text: str, font, fill, stroke_fill, stroke_w: int,
    bbox: BBox, tsume: float,
):
    """stack glyphs top-to-bottom, centered horizontally in bbox, with
    tsume (CJK compression) reducing the gap between rows — the same
    per-character compression convention _draw_line uses for horizontal
    tracking, applied to the vertical axis instead."""
    from PIL import Image, ImageDraw

    layer = Image.new("RGBA", base_size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(layer)
    row_h = font.size * VERTICAL_ROW_FACTOR
    compress = (tsume or 0) * font.size * 0.1
    total_h = _vertical_block_height(font, len(text), tsume)
    y = bbox.y + (bbox.height - total_h) / 2
    for ch in text:
        l, t, r, b = draw.textbbox((0, 0), ch, font=font, stroke_width=stroke_w)
        w = r - l
        x = bbox.x + (bbox.width - w) / 2 - l
        draw.text((_pixel(x), _pixel(y - t)), ch, font=font, fill=fill,
                  stroke_width=stroke_w,
                  stroke_fill=stroke_fill if stroke_w > 0 else None)
        y += row_h - compress
    return layer


def _fit_vertical_words(draw, words: List[str], bbox: BBox, font_family: Optional[str], explicit_size: Optional[int] = None):
    """Fit explicit vertical word-columns, preserving each word's column."""
    if explicit_size and explicit_size > 0:
        return _get_font(font_family, explicit_size)
    lo, hi = MIN_FONT_PX, max(MIN_FONT_PX + 1, bbox.width * 2)
    best = _get_font(font_family, MIN_FONT_PX)
    while lo <= hi:
        mid = (lo + hi) // 2
        font = _get_font(font_family, mid)
        widths = [max((draw.textbbox((0, 0), ch, font=font)[2] - draw.textbbox((0, 0), ch, font=font)[0] for ch in word), default=0) for word in words]
        heights = [_vertical_block_height(font, len(word), 0.0) for word in words]
        gap = font.size * 0.3 * max(0, len(words) - 1)
        if sum(widths) + gap <= bbox.width and max(heights, default=0) <= bbox.height:
            best, lo = font, mid + 1
        else:
            hi = mid - 1
    return best


def _render_vertical_words_layer(base_size, words: List[str], font, fill, stroke_fill, stroke_w: int, bbox: BBox, word_order: Optional[str]):
    """Draw whitespace-delimited words as top-to-bottom columns."""
    from PIL import Image, ImageDraw

    ordered = list(reversed(words)) if word_order == "rtl" else words
    layer = Image.new("RGBA", base_size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(layer)
    widths = [max((draw.textbbox((0, 0), ch, font=font, stroke_width=stroke_w)[2] - draw.textbbox((0, 0), ch, font=font, stroke_width=stroke_w)[0] for ch in word), default=0) for word in ordered]
    gap = font.size * 0.3
    total_w = sum(widths) + gap * max(0, len(ordered) - 1)
    x = bbox.x + (bbox.width - total_w) / 2
    for word, column_w in zip(ordered, widths):
        total_h = _vertical_block_height(font, len(word), 0.0)
        y = bbox.y + (bbox.height - total_h) / 2
        for ch in word:
            l, t, r, _ = draw.textbbox((0, 0), ch, font=font, stroke_width=stroke_w)
            draw.text((_pixel(x + (column_w - (r - l)) / 2 - l), _pixel(y - t)), ch, font=font, fill=fill, stroke_width=stroke_w, stroke_fill=stroke_fill if stroke_w > 0 else None)
            y += font.size * VERTICAL_ROW_FACTOR
        x += column_w + gap
    return layer


def _parse_color(color: Optional[str], opacity: Optional[float]) -> Tuple[int, int, int, int]:
    """hex ('#rgb'/'#rrggbb'/'#rrggbbaa') -> RGBA; defaults to opaque black."""
    r, g, b, a = 0, 0, 0, 255
    if color and color.startswith("#"):
        h = color.lstrip("#")
        if len(h) == 3:
            h = "".join(c * 2 for c in h)
        if len(h) in (6, 8):
            try:
                r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
                if len(h) == 8:
                    a = int(h[6:8], 16)
            except ValueError:
                pass
    if opacity is not None:
        a = int(a * max(0.0, min(1.0, opacity)))
    return (r, g, b, a)


def _resolve_text_ink(
    fill_color: Optional[str],
    stroke_color: Optional[str],
    opacity: Optional[float],
) -> Tuple[Tuple[int, int, int, int], Optional[Tuple[int, int, int, int]]]:
    """Resolve fill/stroke into the exact RGBA inks Scribe composites.

    Opacity is a region-level appearance setting, so it must affect both
    components.  Applying it only to the fill leaves an opaque halo at the
    fill/stroke join even when the editor selected the very same colour for
    both.  Returning the *same tuple* for equal effective inks also makes the
    single Pillow text draw an unbroken ink shape rather than two treatments
    that merely happen to share RGB channels.
    """
    fill = _parse_color(fill_color, opacity)
    stroke = _parse_color(stroke_color, opacity) if stroke_color else None
    if stroke == fill:
        stroke = fill
    return fill, stroke


def _draw_line(draw, text: str, pos, font, fill, stroke_fill=None, stroke_w=0,
               style: Optional[StyleProfil] = None):
    """draw one line of text with optional tracking (letter spacing),
    kerning (pairwise inter-character adjustment), and tsume (CJK compression)."""
    s = style or StyleProfil()
    tracking = s.tracking or 0
    kerning = s.kerning or 0
    tsume = s.tsume or 0

    if tracking or kerning or tsume:
        x, y = pos
        n = len(text)
        for i, ch in enumerate(text):
            if tsume and i > 0:
                x -= tsume * (font.size * 0.1)
            draw.text((x, y), ch, font=font, fill=fill,
                      stroke_width=stroke_w,
                      stroke_fill=stroke_fill if stroke_w > 0 else None)
            try:
                bbox = draw.textbbox((0, 0), ch, font=font, stroke_width=stroke_w)
                x += (bbox[2] - bbox[0]) + tracking
            except Exception:
                x += font.size + tracking
            if kerning and i < n - 1:
                x += kerning
    else:
        draw.text(pos, text, font=font, fill=fill,
                  stroke_width=stroke_w,
                  stroke_fill=stroke_fill if stroke_w > 0 else None)


def _render_line_layer(
    base_size, text: str, font, fill, stroke_fill, stroke_w,
    style: Optional[StyleProfil], line_w: float, line_h: float,
    origin: Tuple[float, float],
):
    """render one line onto a layer sized to base_size, applying italic
    shear LOCAL to the line's own bounding box before compositing.

    the previous implementation applied the shear to a layer the size of
    the whole base image using each pixel's ABSOLUTE y-coordinate
    (x' = x + shear*y), which shifts text sideways by an amount
    proportional to where in the image it sits — for a region at y≈300
    that's a ~60px unwanted shift, not a slant. shearing a small layer
    sized to the line's own extent, then pasting it back at `origin`,
    bounds the shear to the line's own height regardless of its position
    in the image.
    """
    from PIL import Image, ImageDraw

    s = style or StyleProfil()
    ox, oy = origin
    if not s.italic:
        layer = Image.new("RGBA", base_size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(layer)
        _draw_line(draw, text, (ox, oy), font, fill, stroke_fill, stroke_w, s)
        return layer

    # local layer: sized to the line's own box plus shear headroom, margin
    # for stroke width, positioned so (0,0) aligns with the line's own top
    pad = stroke_w + 2
    shear_headroom = int(line_h * ITALIC_SHEAR) + pad
    local_w = max(1, int(line_w) + 2 * pad + shear_headroom)
    local_h = max(1, int(line_h) + 2 * pad)
    local = Image.new("RGBA", (local_w, local_h), (0, 0, 0, 0))
    local_draw = ImageDraw.Draw(local)
    _draw_line(local_draw, text, (pad, pad), font, fill, stroke_fill, stroke_w, s)
    # shear around this layer's own origin (y=0 at its own top) — bounded
    # to [0, shear*local_h] regardless of the line's position in the image
    m = [1, ITALIC_SHEAR, 0, 0, 1, 0]
    sheared = local.transform(
        (local_w + shear_headroom, local_h), Image.AFFINE, m, resample=Image.BICUBIC
    )
    layer = Image.new("RGBA", base_size, (0, 0, 0, 0))
    layer.paste(sheared, (int(ox - pad), int(oy - pad)), sheared)
    return layer


DEFAULT_SHADOW = {"offset_x": 2, "offset_y": 2, "blur": 2, "color": "#00000080"}


def _shadow_layer(
    base_size, text: str, font, style: Optional[StyleProfil], stroke_w: int,
    line_w: float, line_h: float, origin: Tuple[float, float],
):
    """blurred, offset, colored copy of a line — composited BENEATH the
    main text draw. StyleProfil.shadow: {offset_x, offset_y, blur, color}
    (hex RGBA), any subset; missing keys fall back to DEFAULT_SHADOW."""
    from PIL import Image, ImageDraw, ImageFilter

    s = style or StyleProfil()
    spec = {**DEFAULT_SHADOW, **(s.shadow or {})}
    dx, dy, blur = spec["offset_x"], spec["offset_y"], max(0.0, spec["blur"])
    color = _parse_color(spec["color"], None)
    ox, oy = origin

    pad = int(blur) * 3 + stroke_w + 4
    local_w = max(1, int(line_w) + 2 * pad)
    local_h = max(1, int(line_h) + 2 * pad)
    local = Image.new("RGBA", (local_w, local_h), (0, 0, 0, 0))
    local_draw = ImageDraw.Draw(local)
    _draw_line(local_draw, text, (pad, pad), font, color, None, 0, s)
    if blur > 0:
        local = local.filter(ImageFilter.GaussianBlur(radius=blur))

    layer = Image.new("RGBA", base_size, (0, 0, 0, 0))
    layer.paste(local, (int(ox - pad + dx), int(oy - pad + dy)), local)
    return layer


def _capture_text_mask(masks: Dict[str, Any], inst_id: str, layer: Any) -> None:
    """Accumulate an instance's rendered alpha coverage.

    ``layer`` is the fully transformed, image-sized render layer.  Its alpha
    channel is therefore the authoritative coverage of the glyphs, underline,
    shadow, rotation, and warp that Scribe actually produced.  Garnish uses
    this instead of trying to rediscover text by differencing the localized
    image from a possibly patched Cleanse base.
    """
    from PIL import Image
    import numpy as np

    alpha = layer.getchannel("A")
    existing = masks.get(inst_id)
    if existing is None:
        masks[inst_id] = alpha.copy()
        return
    merged = np.maximum(np.asarray(existing, dtype=np.uint8), np.asarray(alpha, dtype=np.uint8))
    masks[inst_id] = Image.fromarray(merged, mode="L")


def render(
    cleansed_asset: Any,
    text_manifest: TextManifest,
    targ_lang: str,
    render_params: Optional[Dict[str, RenderParams]] = None,
    font_registry: Optional[Any] = None,
) -> Any:
    """render localized text onto the cleansed asset.

    args:
        cleansed_asset: output of cleanse.erase().
        text_manifest: manifest with style/background profiles; each
            instance's target_text is what gets drawn (regions without a
            translation are skipped).
        targ_lang: target language code.
        render_params: optional per-region overrides
            (region_id -> RenderParams); defaults are derived from each
            instance's style_profile and bounding_box.
        font_registry: optional tofu.layers.fonts.FontRegistry. when
            given, a requested weight/italic is resolved to a REAL
            sibling face in the same family before falling back to
            synthetic bold/italic — a real face renders at its own
            metrics instead of a shear that (even after the shear-pivot
            fix) widens the glyph beyond the fitted box by a few px.

    returns:
        localized PIL image (RGB), or the input unchanged if it cannot
        be rendered (non-image asset or Pillow missing).
    """
    base = _load_image(cleansed_asset)
    if base is None:
        return cleansed_asset  # contract preserved for non-image assets

    from PIL import Image, ImageDraw

    # preserve original DPI and EXIF for production-ready output
    _orig_dpi = None
    _orig_exif = None
    _orig_icc = None
    try:
        if isinstance(cleansed_asset, (str, Path)):
            with Image.open(cleansed_asset) as src:
                _orig_dpi = src.info.get("dpi")
                _orig_exif = src.info.get("exif")
                _orig_icc = src.info.get("icc_profile")
    except Exception:
        pass

    render_params = render_params or {}
    # Downstream post-processing must remain coupled to the structured text
    # layer, not infer glyph coverage from output pixels.  These masks are
    # deliberately attached only to this transient render result: they are
    # preview/final composition data, never manifest state.
    text_masks: Dict[str, Any] = {}
    # Basil is the authoritative semantic-block → spatial-cube relation.
    # target_text is retained for editable manifest compatibility, but a
    # verified plating plan must win in final rendering just as it does in
    # the Translate preview.
    try:
        from tofu.layers.basil import plated_texts
        basil_overlay = plated_texts(text_manifest)
    except Exception:
        basil_overlay = {}
    for inst in text_manifest.instances:
        if getattr(inst, "dnt", False) or getattr(inst, "excluded", False):
            continue
        # untranslated regions are skipped, not re-rendered with source text:
        # a cleansed-but-empty region is honest; source text re-drawn in the
        # default style silently masquerades as output.
        text = basil_overlay.get(inst.id, inst.target_text)
        if not text:
            continue
        params = render_params.get(inst.id) or RenderParams(
            position=inst.bounding_box,
            style=inst.style_profile,
        )
        bbox = _pixel_bbox(params.position or inst.bounding_box)
        if bbox is None or bbox.width <= 0 or bbox.height <= 0:
            continue
        style = params.style

        measure_draw = ImageDraw.Draw(Image.new("RGBA", (1, 1)))
        s = style or StyleProfil()
        # Capture geometry is immutable semantic identity, not a visual crop
        # boundary.  Transformed ink may extend into adjacent canvas space;
        # downstream treatment and verification use the emitted alpha mask,
        # while an obsolete adjusted_bbox is cleared on every render.
        inst.adjusted_bbox = None
        effective_lang = inst.target_language or targ_lang
        if font_registry is not None and (s.font_family or s.font_weight or s.italic):
            resolved_path, synthetic_italic = resolve_face(
                font_registry, s.font_family, s.font_weight, bool(s.italic)
            )
            if resolved_path != s.font_family or synthetic_italic != bool(s.italic):
                import dataclasses
                s = dataclasses.replace(s, font_family=resolved_path, italic=synthetic_italic)

        # ToFU render-time guard: the resolved face must actually contain
        # every codepoint in this text, or the "tofu" (missing-glyph)
        # boxes the whole layer is named for reach the output despite a
        # passing pre-flight (pre-flight only ever validated the DETECTED
        # source text's script, never the SPECIFIC target string scribe
        # is about to draw). swap to the best real coverage match when
        # one improves on the current face, and flag the region either way
        # so the pipeline can log it and the UI can warn the user.
        # check_glyph_coverage degrades gracefully with no registry (True,
        # "assume covered") -- called unconditionally so glyph_fallback is
        # always explicitly set, never left stale. a manifest persists
        # across renders, and a previously-flagged region whose gap is
        # now resolved (font changed, registry attached/detached) must
        # not keep reporting glyph_fallback=True forever just because
        # this code path only ever set it, never cleared it.
        covering_path, all_covered = check_glyph_coverage(
            font_registry, s.font_family, text, effective_lang
        )
        inst.glyph_fallback = not all_covered
        if not all_covered and covering_path:
            import dataclasses
            s = dataclasses.replace(s, font_family=covering_path)

        stroke_w = int(s.stroke_width) if s.stroke_width else 0
        # Resolve both text components through the same opacity rule.  This
        # is especially important for equal fill/stroke colours: a 50%-alpha
        # fill surrounded by a 100%-alpha stroke produces a visible internal
        # boundary at close zoom instead of one continuous ink shape.
        fill, stroke_fill = _resolve_text_ink(s.color, s.stroke_color, params.opacity)

        # detected baseline rotation (Phase 1 typography) applies when no
        # explicit override was given — RenderParams.rotation always wins,
        # including an explicit 0.0 (a falsy check here would wrongly let
        # a detected rotation override an explicit "no rotation" request)
        rotation = params.rotation
        if rotation is None:
            editor_rotation = (s.transform or {}).get("rotation")
            if editor_rotation is not None:
                try:
                    rotation = float(editor_rotation)
                except (TypeError, ValueError):
                    rotation = None
        if rotation is None and inst.characteristics and inst.characteristics.positioning:
            detected_rot = inst.characteristics.positioning.get("rotation_deg")
            if detected_rot:
                rotation = detected_rot

        if _should_render_vertical(bbox, effective_lang, text, s.target_orientation):
            words = text.split()
            explicit_word_columns = s.target_orientation == "vertical" and len(words) > 1
            if explicit_word_columns:
                font = _fit_vertical_words(measure_draw, words, bbox, s.font_family, s.font_size)
                layer = _render_vertical_words_layer(
                    base.size, words, font, fill, stroke_fill, stroke_w, bbox, s.word_order
                )
            else:
                # Keep the legacy single-column path byte-for-byte for
                # orientation-unset manifests and single-token overrides.
                font = _fit_vertical(measure_draw, text, bbox, s.font_family, s.font_size, s.tsume or 0)
                layer = _render_vertical_layer(
                    base.size, text, font, fill, stroke_fill, stroke_w, bbox, s.tsume or 0
                )
            if rotation:
                center = (bbox.x + bbox.width / 2, bbox.y + bbox.height / 2)
                layer = _apply_style_transform(layer, bbox, s.transform)
                layer = layer.rotate(rotation, center=center, resample=Image.BICUBIC)
            else:
                layer = _apply_style_transform(layer, bbox, s.transform)
            _capture_text_mask(text_masks, inst.id, layer)
            base = Image.alpha_composite(base, layer)
            continue

        wrap_text = bool((s.transform or {}).get("wrap_text", False))
        font, lines, spacing = _fit_wrapped(
            measure_draw, text, bbox, s.font_family, s.font_size, s.leading,
            s.tracking or 0, s.kerning or 0, wrap_text,
        )
        # super/subscript: re-fit at a reduced size (also re-wraps, since a
        # smaller font can fit differently) rather than the fitted size —
        # the previous implementation computed this scale but never used it
        if (s.superscript or s.subscript) and not (s.font_size and s.font_size > 0):
            small_size = max(MIN_FONT_PX, round(font.size * 0.65))
            font, lines, spacing = _fit_wrapped(
                measure_draw, text, bbox, s.font_family, small_size, s.leading,
                s.tracking or 0, s.kerning or 0, wrap_text,
            )

        line_h = _line_height(font)
        _tracking = s.tracking or 0
        _kerning = s.kerning or 0
        line_metrics = []  # (line, width, left_bearing, top_bearing, bottom_bearing)
        for ln in lines:
            l, t, r, b = measure_draw.textbbox((0, 0), ln, font=font, stroke_width=stroke_w)
            # tracking and kerning widen the line beyond the raw ink bbox;
            # account for them so centering/alignment matches the drawn result
            extra = (_tracking + _kerning) * max(0, len(ln) - 1)
            line_metrics.append((ln, (r - l) + extra, l, t, b))
        # true ink extent for CENTERING (not the nominal ascent+descent
        # sum, which overestimates for text without descenders and would
        # visibly shift a single short line off-center); per-line spacing
        # for the actual row-to-row step below still uses line_h — rows
        # should be evenly spaced by the font's natural metric regardless
        # of which specific glyphs a given row happens to contain.
        _, block_t, _, block_b = measure_draw.multiline_textbbox(
            (0, 0), "\n".join(lines), font=font, spacing=spacing, stroke_width=stroke_w
        )
        block_h = block_b - block_t

        # vertical alignment of the whole block within bbox
        av = s.align_v or "middle"
        if av == "top":
            by = bbox.y
        elif av == "bottom":
            by = bbox.y + bbox.height - block_h
        else:
            by = bbox.y + (bbox.height - block_h) / 2
        if s.baseline_shift:
            by += s.baseline_shift

        if s.superscript:
            by -= block_h * 0.3
        elif s.subscript:
            by += block_h * 0.2

        layer = Image.new("RGBA", base.size, (0, 0, 0, 0))
        ah = s.align_h or "center"
        y_cursor = by
        for ln, w, l_off, t_off, b_off in line_metrics:
            if ah == "left":
                lx = bbox.x + (s.indent or 0) - l_off
            elif ah == "right":
                lx = bbox.x + bbox.width - w - l_off
            else:
                lx = bbox.x + (bbox.width - w) / 2 - l_off
            ly = y_cursor - t_off
            if s.shadow:
                shadow_layer = _shadow_layer(
                    base.size, ln, font, s, stroke_w, w, line_h, (_pixel(lx), _pixel(ly))
                )
                layer = Image.alpha_composite(layer, shadow_layer)
            line_layer = _render_line_layer(
                base.size, ln, font, fill, stroke_fill, stroke_w, s,
                w, line_h, (_pixel(lx), _pixel(ly)),
            )
            layer = Image.alpha_composite(layer, line_layer)

            if s.underline:
                # glyph bottom edge (ly + b_off is where the drawn text's
                # own bbox bottom lands), plus a detected/default drop.  The
                # offset and thickness are independent from an outline: a
                # user can tune an underline without unexpectedly changing
                # the glyph stroke itself.
                uy = _pixel(ly + b_off + (s.underline_offset if s.underline_offset is not None else 1))
                draw = ImageDraw.Draw(layer)
                draw.line([(_pixel(lx), uy), (_pixel(lx + w), uy)], fill=fill,
                          width=max(1, int(round(s.underline_width if s.underline_width is not None else (stroke_w / 2) or 1))))
            y_cursor += line_h + spacing

        layer = _apply_style_transform(layer, bbox, s.transform)
        if rotation:
            center = (bbox.x + bbox.width / 2, bbox.y + bbox.height / 2)
            layer = layer.rotate(
                rotation, center=center, resample=Image.BICUBIC
            )

        _capture_text_mask(text_masks, inst.id, layer)
        base = Image.alpha_composite(base, layer)

    result = base.convert("RGB")
    # PIL images permit transient attributes.  Keeping the coverage alongside
    # the image avoids widening Scribe's public return type and preserves the
    # existing render callers.  Garnish retains a conservative diff fallback
    # for non-Scribe image inputs and any image operation that discards it.
    result.text_masks = text_masks  # type: ignore[attr-defined]
    # attach preserved metadata so .save() callers can pass it through
    if _orig_dpi:
        result.info["dpi"] = _orig_dpi
    if _orig_exif:
        result.info["exif"] = _orig_exif
    if _orig_icc:
        result.info["icc_profile"] = _orig_icc
    return result
