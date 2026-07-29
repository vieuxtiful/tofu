## 🍢 Knead — HarfBuzz shaping + FreeType rasterization
## vieuxtiful
"""
Text shaping seam: turn a string into positioned glyphs, then bake those
glyphs into pixels.

Pillow's own text API uses FreeType's BASIC layout engine, which maps each
codepoint to one glyph, advances by that glyph's nominal width, and stops.
It applies no OpenType GPOS (so no kerning and no mark attachment), no GSUB
(so no ligatures and no Indic conjuncts), and no reordering. The complete
fix inside Pillow is Raqm, but Pillow's Windows wheel is compiled without it
and that is a compile-time fact, not a configuration one — see the RTL note
in scribe.py. So ToFU does the shaping itself.

Two libraries, both pip-installable with Windows wheels, neither needing a C
toolchain on the target machine:
  uharfbuzz   HarfBuzz — the same shaper Chrome, Firefox and Android use
  freetype-py FreeType — rasterizes a glyph ID to a coverage bitmap

Measured against Pillow's BASIC engine on the same faces (Arial, Nirmala UI):
  "AVATAR Wave To"  396.6px -> 375.3px advance, identical ink — that 21px
                    is GPOS kerning Pillow drops on the floor entirely.
  "हिन्दी"           6 codepoints -> 4 glyphs, and the i-matra moves to the
                    left of its consonant where it belongs. Pillow draws it
                    in logical order, visibly wrong.
  "क्ष"              3 codepoints -> 1 conjunct glyph.

Everything here is optional and degrades: available() is false when either
library is missing, and every entry point returns None rather than raising,
so callers keep their Pillow path as a fallback.

DIRECTION AND BIDI. HarfBuzz shapes one direction-uniform run and emits its
glyphs in VISUAL order — for an RTL run it has already done the reversing,
and GSUB has already produced the cursive joining forms. A caller that has
its own reshaper/bidi pass must therefore hand this module LOGICAL text and
skip that pass, or the reordering happens twice and cancels out. What
HarfBuzz does NOT do is bidi itemization: splitting a mixed-direction
paragraph into runs is the caller's job (UAX #9), and passing mixed text
here shapes it all in one direction.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# script codes that shape right-to-left, as ISO 15924 (what scribe stores)
_RTL_SCRIPTS = {"Arab", "Hebr", "Syrc", "Thaa", "Nkoo", "Adlm", "Mand", "Samr"}

# a face is opened per (path, index) for HarfBuzz and per (path, index, size)
# for FreeType. reopening either per glyph is the obvious performance trap:
# scribe calls into the fitter in a binary search loop, so this is hot.
_hb_cache: Dict[Tuple[str, int], Any] = {}
_ft_cache: Dict[Tuple[str, int, int], Any] = {}
_CACHE_MAX = 64


def available() -> bool:
    """Whether both shaping libraries import. Cheap enough to call per region."""
    try:
        import uharfbuzz  # noqa: F401
        import freetype  # noqa: F401
    except ImportError:
        return False
    return True


@dataclass(frozen=True)
class ShapedGlyph:
    """One positioned glyph. Offsets and advances are pixels, not 26.6."""
    gid: int
    cluster: int
    x_advance: float
    y_advance: float
    x_offset: float
    y_offset: float


@dataclass(frozen=True)
class ShapedRun:
    glyphs: Tuple[ShapedGlyph, ...]
    advance: float      # total pen travel in pixels
    direction: str      # "ltr" | "rtl"
    script: str

    def __len__(self) -> int:
        return len(self.glyphs)


def face_identity(pil_font: Any) -> Optional[Tuple[str, int, int]]:
    """(path, face_index, pixel_size) for a Pillow FreeTypeFont, or None.

    Reuses whatever Pillow already resolved rather than re-implementing font
    lookup: scribe's fallback chain is bare filenames like "arial.ttf" that
    Pillow finds in the system font directory and FreeType would not.
    Pillow's default bitmap font carries a BytesIO in .path and has no real
    file behind it, so it is rejected here and the caller stays on Pillow.
    """
    path = getattr(pil_font, "path", None)
    if not isinstance(path, str) or not path:
        return None
    size = getattr(pil_font, "size", None)
    if not size:
        return None
    try:
        if not Path(path).is_file():
            return None
    except OSError:
        return None
    return path, int(getattr(pil_font, "index", 0) or 0), int(size)


def _hb_font(path: str, index: int, size: int):
    key = (path, index)
    face = _hb_cache.get(key)
    if face is None:
        import uharfbuzz as hb
        face = hb.Face(hb.Blob.from_file_path(path), index)
        if len(_hb_cache) < _CACHE_MAX:
            _hb_cache[key] = face
    import uharfbuzz as hb
    font = hb.Font(face)
    # HarfBuzz reports positions in 26.6 fixed point once scaled this way,
    # so every value out of the buffer is divided by 64 to reach pixels.
    font.scale = (size * 64, size * 64)
    hb.ot_font_set_funcs(font)
    return font


def _ft_face(path: str, index: int, size: int):
    key = (path, index, size)
    face = _ft_cache.get(key)
    if face is None:
        import freetype
        face = freetype.Face(path, index=index)
        face.set_pixel_sizes(0, size)
        if len(_ft_cache) < _CACHE_MAX:
            _ft_cache[key] = face
    return face


def _direction_for(script: Optional[str], override: Optional[str]) -> Optional[str]:
    if override:
        return override
    if script in _RTL_SCRIPTS:
        return "rtl"
    return None  # let HarfBuzz guess from the text's own strong characters


def knead_run(
    text: str,
    pil_font: Any,
    *,
    script: Optional[str] = None,
    direction: Optional[str] = None,
    features: Optional[Dict[str, bool]] = None,
) -> Optional[ShapedRun]:
    """Shape one direction-uniform line into positioned glyphs.

    Kneading is where the dough stops being a pile of ingredients and starts
    holding a shape: codepoints go in, an ordered run of positioned glyph IDs
    comes out, with ligatures formed, conjuncts assembled, marks attached and
    pairs kerned.

    `script` is an ISO 15924 code used only to pick a base direction; the
    actual script for shaping is detected from the text. Returns None when
    shaping is unavailable or the face cannot be opened, which is the
    caller's signal to fall back to Pillow.
    """
    if not text:
        return None
    identity = face_identity(pil_font)
    if identity is None or not available():
        return None
    path, index, size = identity
    try:
        import uharfbuzz as hb
        font = _hb_font(path, index, size)
        buf = hb.Buffer()
        buf.add_str(text)
        buf.guess_segment_properties()
        forced = _direction_for(script, direction)
        if forced:
            buf.direction = forced
        hb.shape(font, buf, features)
        glyphs = tuple(
            ShapedGlyph(
                gid=info.codepoint, cluster=info.cluster,
                x_advance=pos.x_advance / 64.0, y_advance=pos.y_advance / 64.0,
                x_offset=pos.x_offset / 64.0, y_offset=pos.y_offset / 64.0,
            )
            for info, pos in zip(buf.glyph_infos, buf.glyph_positions)
        )
    except Exception:
        return None
    if not glyphs:
        return None
    return ShapedRun(
        glyphs=glyphs,
        advance=sum(g.x_advance for g in glyphs),
        direction=str(buf.direction),
        script=str(buf.script),
    )


def proof_run(run: Optional[ShapedRun], extra_px: float) -> Optional[ShapedRun]:
    """Add letter spacing to a shaped run, at CLUSTER boundaries only.

    Proofing lets dough expand without tearing what has already been built.
    The same restraint applies here: spacing is inserted between clusters,
    never inside one.

    A cluster is the unit that survived shaping — a Devanagari conjunct, a
    base plus its attached marks, a ligature — and it can be several glyphs
    that belong to one another. Adding tracking between every GLYPH would
    prise those apart and undo the exact shaping this module exists to
    produce, so the offset lands only where the next glyph opens a new
    cluster.

    Returns a new run rather than taking a `tracking=` argument on the
    measure and draw entry points: a caller that shapes once and passes the
    same object to both cannot drift between what it measured and what it
    drew.
    """
    if run is None or not run.glyphs or not extra_px:
        return run
    glyphs = run.glyphs
    spaced = []
    for i, glyph in enumerate(glyphs):
        following = glyphs[i + 1] if i + 1 < len(glyphs) else None
        opens_new_cluster = following is not None and following.cluster != glyph.cluster
        if opens_new_cluster:
            glyph = ShapedGlyph(
                gid=glyph.gid, cluster=glyph.cluster,
                x_advance=glyph.x_advance + extra_px, y_advance=glyph.y_advance,
                x_offset=glyph.x_offset, y_offset=glyph.y_offset,
            )
        spaced.append(glyph)
    return ShapedRun(
        glyphs=tuple(spaced),
        advance=sum(g.x_advance for g in spaced),
        direction=run.direction,
        script=run.script,
    )


def run_width(text: str, pil_font: Any, **kwargs) -> Optional[float]:
    """Shaped advance width in pixels, or None to fall back to Pillow.

    This is the measurement half of the contract and it MUST agree with what
    bake_run() later draws — a fitter that measures with kerning applied and
    then draws without it will overflow its box.
    """
    run = knead_run(text, pil_font, **kwargs)
    return None if run is None else run.advance


def _glyph_bitmap(face, gid: int, stroke_width: int):
    """(coverage_array, left, top) for one glyph, stroked when asked."""
    import freetype
    import numpy as np

    if stroke_width > 0:
        face.load_glyph(gid, freetype.FT_LOAD_NO_BITMAP)
        glyph = face.glyph.get_glyph()
        stroker = freetype.Stroker()
        stroker.set(
            int(stroke_width * 64),
            freetype.FT_STROKER_LINECAP_ROUND,
            freetype.FT_STROKER_LINEJOIN_ROUND,
            0,
        )
        glyph.stroke(stroker, True)
        baked = glyph.to_bitmap(freetype.FT_RENDER_MODE_NORMAL, freetype.Vector(0, 0), True)
        bitmap, left, top = baked.bitmap, baked.left, baked.top
    else:
        face.load_glyph(gid, freetype.FT_LOAD_RENDER)
        slot = face.glyph
        bitmap, left, top = slot.bitmap, slot.bitmap_left, slot.bitmap_top

    if not bitmap.width or not bitmap.rows:
        return None, 0, 0
    # rows are padded to `pitch`; trim to the real width before use
    arr = np.array(bitmap.buffer, dtype=np.uint8).reshape(bitmap.rows, bitmap.pitch)
    return arr[:, : bitmap.width], left, top


def _stamp(coverage, arr, x: int, y: int) -> None:
    """Composite one glyph's coverage, clipped to the canvas.

    Glyphs are combined with a maximum rather than a sum: adjacent or
    overlapping glyphs (accents over letters, a stroke under its own fill,
    tightly kerned pairs) would otherwise accumulate past full coverage and
    leave a visible seam where they touch.
    """
    import numpy as np

    h, w = coverage.shape
    gh, gw = arr.shape
    x0, y0 = max(0, x), max(0, y)
    x1, y1 = min(w, x + gw), min(h, y + gh)
    if x1 <= x0 or y1 <= y0:
        return
    sub = arr[y0 - y : y1 - y, x0 - x : x1 - x]
    region = coverage[y0:y1, x0:x1]
    np.maximum(region, sub, out=region)


def bake_run(
    canvas_size: Tuple[int, int],
    run: ShapedRun,
    origin: Tuple[float, float],
    pil_font: Any,
    fill: Tuple[int, int, int, int],
    *,
    stroke_width: int = 0,
    stroke_fill: Optional[Tuple[int, int, int, int]] = None,
) -> Optional[Any]:
    """Rasterize a shaped run onto a new RGBA layer.

    `origin` is the pen start on the BASELINE (x, y), matching FreeType's
    own convention rather than Pillow's top-left default — callers that
    think in top-left boxes convert once, up front.

    The stroke is baked first and the fill composited over it, so a glyph's
    outline sits behind its own body exactly as Pillow's stroke_width does.
    """
    identity = face_identity(pil_font)
    if identity is None or run is None:
        return None
    path, index, size = identity
    try:
        import numpy as np
        from PIL import Image

        face = _ft_face(path, index, size)
        width, height = canvas_size
        layers: List[Tuple[Any, Tuple[int, int, int, int]]] = []

        for width_px, colour in ((stroke_width, stroke_fill), (0, fill)):
            if colour is None:
                continue
            if width_px == 0 and colour is stroke_fill:
                continue
            coverage = np.zeros((height, width), dtype=np.uint8)
            pen_x, pen_y = float(origin[0]), float(origin[1])
            for glyph in run.glyphs:
                arr, left, top = _glyph_bitmap(face, glyph.gid, width_px)
                if arr is not None:
                    _stamp(
                        coverage, arr,
                        int(round(pen_x + glyph.x_offset)) + left,
                        int(round(pen_y - glyph.y_offset)) - top,
                    )
                pen_x += glyph.x_advance
                pen_y -= glyph.y_advance
            layers.append((coverage, colour))

        if not layers:
            return None
        out = Image.new("RGBA", (width, height), (0, 0, 0, 0))
        for coverage, colour in layers:
            tinted = Image.new("RGBA", (width, height), tuple(colour[:3]) + (0,))
            alpha = (coverage.astype(np.float32) * (colour[3] / 255.0)).astype(np.uint8)
            tinted.putalpha(Image.fromarray(alpha, mode="L"))
            out = Image.alpha_composite(out, tinted)
        return out
    except Exception:
        return None


def run_ink_box(
    run: ShapedRun, pil_font: Any, stroke_width: int = 0
) -> Optional[Tuple[float, float, float, float]]:
    """(left, top, right, bottom) of the run's ink relative to the pen origin.

    Needed for centring: the advance width overstates the visible extent for
    text with side bearings, and the nominal ascent/descent overstates the
    height for a line with no descenders.
    """
    identity = face_identity(pil_font)
    if identity is None or run is None or not run.glyphs:
        return None
    try:
        path, index, size = identity
        face = _ft_face(path, index, size)
        pen_x = pen_y = 0.0
        left = top = float("inf")
        right = bottom = float("-inf")
        seen = False
        for glyph in run.glyphs:
            arr, bl, bt = _glyph_bitmap(face, glyph.gid, stroke_width)
            if arr is not None:
                gx = pen_x + glyph.x_offset + bl
                gy = pen_y - glyph.y_offset - bt
                left, top = min(left, gx), min(top, gy)
                right = max(right, gx + arr.shape[1])
                bottom = max(bottom, gy + arr.shape[0])
                seen = True
            pen_x += glyph.x_advance
            pen_y -= glyph.y_advance
        if not seen:
            return None
        return left, top, right, bottom
    except Exception:
        return None
