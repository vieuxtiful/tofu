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

# vertical CJK column rendering: a region is treated as a vertical column
# when its bbox is narrow-and-tall AND the effective target language uses
# vertical writing conventions. the aspect threshold mirrors cicerone's
# own merge_vertical_columns() member test (COLUMN_MAX_ASPECT=1.6 for a
# CHARACTER; a merged multi-character COLUMN region is taller still), so
# a region that survived cicerone's column merge reliably crosses it.
CJK_VERTICAL_LANGS = {"ja", "zh-cn", "zh-tw", "zh-hk", "zh-mo", "zh-sg"}
VERTICAL_ASPECT_MIN = 1.3
VERTICAL_ROW_FACTOR = 1.15  # row height as a multiple of font size


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
    from PIL import ImageFont
    candidates = ([font_family] if font_family else []) + list(FALLBACK_FONTS)
    for cand in candidates:
        path, index = cand, 0
        if isinstance(cand, str) and "#" in cand:
            base, _, idx = cand.rpartition("#")
            if idx.isdigit():
                path, index = base, int(idx)
        try:
            return ImageFont.truetype(path, size, index=index)
        except Exception:
            continue
    return ImageFont.load_default()


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
    the requested weight/italic keywords. no registry, no font_family, or
    no match: returns (font_family, italic) unchanged — the caller's
    existing behavior (literal path + synthetic shear) is the contract.
    """
    if not font_registry or not getattr(font_registry, "fonts", None):
        return font_family, italic
    fonts = getattr(font_registry, "_fonts", {})
    if not font_family:
        return font_family, italic

    key = font_registry._key(font_family) if hasattr(font_registry, "_key") else font_family
    current = fonts.get(key)
    if current is None:
        return font_family, italic  # not a registry-known font; nothing to resolve

    sub = (current.subfamily or "").lower()
    wants_bold = weight == "bold"
    wants_light = weight == "light"
    has_bold = "bold" in sub
    has_italic = "italic" in sub or "oblique" in sub
    weight_ok = (wants_bold == has_bold) if (wants_bold or has_bold) else True
    if wants_light and current.weight_class > 350:
        weight_ok = False
    if weight_ok and italic == has_italic:
        return font_family, False  # already the right face; no synthesis needed

    siblings = [fc for fc in fonts.values() if fc.family == current.family]
    best = None
    for fc in siblings:
        s = (fc.subfamily or "").lower()
        s_italic = "italic" in s or "oblique" in s
        if italic != s_italic:
            continue
        s_bold = "bold" in s
        if wants_bold and not s_bold:
            continue
        if not wants_bold and not wants_light and s_bold:
            continue
        if wants_light and fc.weight_class > 350:
            continue
        best = fc
        break
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
    # only a KNOWN registry font with a confirmed gap counts as evidence
    # of a real problem; an unrecognized font_family (e.g. a bare
    # fallback name outside the registry) gives no evidence either way,
    # so it must not be reported as a coverage failure without proof
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


def _line_height(font) -> float:
    try:
        ascent, descent = font.getmetrics()
        return float(ascent + descent)
    except Exception:
        return float(getattr(font, "size", 12)) * 1.2


def _wrap_lines(draw, text: str, font, max_width: float) -> List[str]:
    """greedy wrap targeting max_width: word-wrap when the text has
    space-delimited words, character-wrap otherwise (CJK and other
    scripts that don't use spaces between words)."""
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
) -> Tuple[Any, List[str], float]:
    """binary-search the largest font size whose greedy-wrapped text fits
    bbox (both max line width and total block height), or use an
    explicit size override. returns (font, lines, line_spacing_px)."""
    if explicit_size and explicit_size > 0:
        font = _get_font(font_family, explicit_size)
        lines = _wrap_lines(draw, text, font, bbox.width)
        spacing = leading if leading is not None else _line_height(font) * 0.2
        return font, lines, spacing

    lo, hi = MIN_FONT_PX, max(MIN_FONT_PX + 1, bbox.height * 2)
    best_font = _get_font(font_family, MIN_FONT_PX)
    best_lines = _wrap_lines(draw, text, best_font, bbox.width)
    best_spacing = leading if leading is not None else _line_height(best_font) * 0.2
    while lo <= hi:
        mid = (lo + hi) // 2
        font = _get_font(font_family, mid)
        lines = _wrap_lines(draw, text, font, bbox.width)
        spacing = leading if leading is not None else _line_height(font) * 0.2
        # fit check uses the ACTUAL rendered ink extent (PIL's own
        # multiline layout), not font.getmetrics()'s nominal ascent+
        # descent — that nominal box accounts for glyphs the string may
        # not even contain (e.g. descenders), which over-estimates height
        # for all-caps/no-descender text and picks an unnecessarily
        # smaller font (measured: "SALE" fit at size 53 instead of the
        # correct 69, visibly shrinking a region that fit fine as-is)
        l, t, r, b = draw.multiline_textbbox((0, 0), "\n".join(lines), font=font, spacing=spacing)
        if (r - l) <= bbox.width and (b - t) <= bbox.height:
            best_font, best_lines, best_spacing, lo = font, lines, spacing, mid + 1
        else:
            hi = mid - 1
    return best_font, best_lines, best_spacing


def _should_render_vertical(bbox: BBox, lang: Optional[str], text: str) -> bool:
    """a region renders as a stacked vertical CJK column when its bbox is
    narrow-and-tall and the effective target language uses vertical
    writing — the shape cicerone's merge_vertical_columns() produces for
    genuine stacked signage. a single-character region is excluded (an
    aspect ratio near 1:1 for one glyph gives no reliable signal either
    way, and a lone character reads identically in either orientation)."""
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
        draw.text((x, y - t), ch, font=font, fill=fill,
                  stroke_width=stroke_w,
                  stroke_fill=stroke_fill if stroke_w > 0 else None)
        y += row_h - compress
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


def _draw_line(draw, text: str, pos, font, fill, stroke_fill=None, stroke_w=0,
               style: Optional[StyleProfil] = None):
    """draw one line of text with optional tracking (letter spacing) and
    tsume (CJK compression)."""
    s = style or StyleProfil()
    tracking = s.tracking or 0
    tsume = s.tsume or 0

    if tracking or tsume:
        x, y = pos
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
    for inst in text_manifest.instances:
        if getattr(inst, "dnt", False):
            continue
        # untranslated regions are skipped, not re-rendered with source text:
        # a cleansed-but-empty region is honest; source text re-drawn in the
        # default style silently masquerades as output.
        text = inst.target_text
        if not text:
            continue
        params = render_params.get(inst.id) or RenderParams(
            position=inst.bounding_box,
            style=inst.style_profile,
        )
        bbox = params.position or inst.bounding_box
        if bbox is None or bbox.width <= 0 or bbox.height <= 0:
            continue
        style = params.style

        measure_draw = ImageDraw.Draw(Image.new("RGBA", (1, 1)))
        s = style or StyleProfil()
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
        if font_registry is not None:
            covering_path, all_covered = check_glyph_coverage(
                font_registry, s.font_family, text, effective_lang
            )
            if not all_covered:
                inst.glyph_fallback = True
                if covering_path:
                    import dataclasses
                    s = dataclasses.replace(s, font_family=covering_path)

        stroke_w = int(s.stroke_width) if s.stroke_width else 0
        fill = _parse_color(s.color, params.opacity)
        stroke_fill = _parse_color(s.stroke_color, None) if s.stroke_color else None

        # detected baseline rotation (Phase 1 typography) applies when no
        # explicit override was given — RenderParams.rotation always wins,
        # including an explicit 0.0 (a falsy check here would wrongly let
        # a detected rotation override an explicit "no rotation" request)
        rotation = params.rotation
        if params.rotation is None and inst.characteristics and inst.characteristics.positioning:
            detected_rot = inst.characteristics.positioning.get("rotation_deg")
            if detected_rot:
                rotation = detected_rot

        if _should_render_vertical(bbox, effective_lang, text):
            font = _fit_vertical(measure_draw, text, bbox, s.font_family, s.font_size, s.tsume or 0)
            layer = _render_vertical_layer(
                base.size, text, font, fill, stroke_fill, stroke_w, bbox, s.tsume or 0
            )
            if rotation:
                center = (bbox.x + bbox.width / 2, bbox.y + bbox.height / 2)
                layer = layer.rotate(rotation, center=center, resample=Image.BICUBIC)
            base = Image.alpha_composite(base, layer)
            continue

        font, lines, spacing = _fit_wrapped(
            measure_draw, text, bbox, s.font_family, s.font_size, s.leading
        )
        # super/subscript: re-fit at a reduced size (also re-wraps, since a
        # smaller font can fit differently) rather than the fitted size —
        # the previous implementation computed this scale but never used it
        if (s.superscript or s.subscript) and not (s.font_size and s.font_size > 0):
            small_size = max(MIN_FONT_PX, round(font.size * 0.65))
            font, lines, spacing = _fit_wrapped(
                measure_draw, text, bbox, s.font_family, small_size, s.leading
            )

        line_h = _line_height(font)
        line_metrics = []  # (line, width, left_bearing, top_bearing, bottom_bearing)
        for ln in lines:
            l, t, r, b = measure_draw.textbbox((0, 0), ln, font=font, stroke_width=stroke_w)
            line_metrics.append((ln, r - l, l, t, b))
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
                    base.size, ln, font, s, stroke_w, w, line_h, (lx, ly)
                )
                layer = Image.alpha_composite(layer, shadow_layer)
            line_layer = _render_line_layer(
                base.size, ln, font, fill, stroke_fill, stroke_w, s,
                w, line_h, (lx, ly),
            )
            layer = Image.alpha_composite(layer, line_layer)

            if s.underline:
                # glyph bottom edge (ly + b_off is where the drawn text's
                # own bbox bottom lands), plus a small drop
                uy = ly + b_off + 1
                draw = ImageDraw.Draw(layer)
                draw.line([(lx, uy), (lx + w, uy)], fill=fill,
                          width=max(1, int(stroke_w / 2) or 1))
            y_cursor += line_h + spacing

        if rotation:
            center = (bbox.x + bbox.width / 2, bbox.y + bbox.height / 2)
            layer = layer.rotate(
                rotation, center=center, resample=Image.BICUBIC
            )

        base = Image.alpha_composite(base, layer)

    result = base.convert("RGB")
    # attach preserved metadata so .save() callers can pass it through
    if _orig_dpi:
        result.info["dpi"] = _orig_dpi
    if _orig_exif:
        result.info["exif"] = _orig_exif
    if _orig_icc:
        result.info["icc_profile"] = _orig_icc
    return result
