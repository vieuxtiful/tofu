## 🍢 Cleanse — Layer 3
## vieuxtiful
"""
text erasure & inpainting layer.

cleanse removes source text from the asset using stroke-level glyph
masks (shared with scene/typography via utils.imaging.text_mask) and
reconstructs the background using a strategy selected from Scene's
BgProfil.texture classification, so cleanse and scene agree on what
kind of surface they're looking at:

  - "flat"           -> border-ring median color fill (fast, exact —
                         a solid panel/sign has no texture to lose)
  - "smooth_gradient" -> per-channel linear plane reconstruction fit
                         over the border ring (the same shading model
                         Scene fits for classification, re-fit here
                         over the actual erasure footprint)
  - "textured" / unclassified -> OpenCV content-aware inpainting
                         (Telea 2004), batched into one pass, radius
                         scaled to the detected stroke width when
                         typography estimated one

every fill is alpha-feathered at the mask boundary (distance-transform
falloff) so no strategy leaves a hard seam.

the mask itself is the glyph stroke mask (Otsu + GrabCut, dilated a few
px for anti-aliased edges) rather than the whole bounding box: the
previous implementation ALSO filled the full padded bbox unconditionally,
which is exactly the "leftover artifact" failure mode described in the
project workplan (measured: stylized-italic ring-SSIM 0.655 on a clean
synthetic). the bbox rectangle is now a fallback used only when
segmentation genuinely fails, so erasure is never silently skipped.

for video (future), inpainting must be temporally consistent — flicker
is the killer failure mode.
"""

from pathlib import Path
from typing import Any, Optional, Tuple

from tofu.core.types import TextManifest
from tofu.utils.imaging import text_mask as _text_mask

DILATE_ITER = 3        # glyph-mask growth to catch anti-aliased edges (~3px)
FEATHER_PX = 3.0       # soft-edge blend width at every fill boundary
# feathering weakens toward the mask's OWN boundary (full-strength fill
# only deep inside it) — the dilation above must be generous enough that
# the true glyph ink (plus its ~1px anti-aliased fringe) sits INSIDE the
# full-strength zone, not under the taper. dilate 1px (old default) put
# the taper directly on top of the glyph's anti-aliased edge, leaving a
# thin readable outline ghost (measured: OCR still read "MAIN STREET" at
# full confidence through the outline on the flat-sign fixture).
RING_PX = 14           # border-ring width sampled for flat/gradient fills
FALLBACK_PAD = 2       # bbox-rectangle fallback padding (segmentation failed)
MIN_RING_PIXELS = 20   # below this, gradient fit degrades to flat fill
TELEA_RADIUS_DEFAULT = 3
TELEA_RADIUS_MIN, TELEA_RADIUS_MAX = 2, 8
EDGE_BORDER_PX = 2     # crop-edge sample width for the mask-polarity gate


def _load_image(asset: Any):
    try:
        from PIL import Image
    except ImportError:
        return None
    if hasattr(asset, "convert") and hasattr(asset, "size"):
        return asset.convert("RGBA")
    if isinstance(asset, (str, Path)):
        try:
            return Image.open(asset).convert("RGBA")
        except Exception:
            return None
    return None


def _plausible_polarity(np, crop, mask) -> bool:
    """sanity check that `mask` selected the INK, not the background.

    text_mask's Otsu+GrabCut segmentation assumes text is well-separated
    from a single background color; multi-color ink (e.g. a stroked/
    outlined glyph — dark fill + a very different stroke color) can have
    higher internal chromatic variance than the background, which can
    make GrabCut's GMM converge on the wrong partition (measured: the
    stylized-italic "Stroked Display" fixture's stroke+fill text got
    masked as background, leaving the actual ink completely unerased —
    residual OCR similarity 1.0). the crop's own edge pixels are almost
    certainly background (OCR boxes are tight around ink, not flush with
    it), so the masked group should sit FARTHER from the edge color than
    the unmasked group; if it's the other way around, the mask is
    inverted and the caller should fall back to the bbox rectangle.
    """
    ch, cw = crop.shape[:2]
    b = min(EDGE_BORDER_PX, ch // 2, cw // 2)
    if b < 1 or not mask.any() or mask.all():
        return True  # too small or degenerate to judge; trust it
    edge = np.concatenate([
        crop[:b].reshape(-1, 3), crop[-b:].reshape(-1, 3),
        crop[:, :b].reshape(-1, 3), crop[:, -b:].reshape(-1, 3),
    ]).mean(axis=0)
    mask_color = crop[mask].reshape(-1, 3).mean(axis=0)
    bg_color = crop[~mask].reshape(-1, 3).mean(axis=0)
    d_mask = np.linalg.norm(mask_color - edge)
    d_bg = np.linalg.norm(bg_color - edge)
    return d_mask >= d_bg


def _region_mask(np, cv2, img_array, bbox, h: int, w: int):
    """glyph-stroke mask for one instance, in full-image coordinates
    (bool array, shape (h, w)) — or None if segmentation failed (or was
    implausible) and the caller should use the bbox-rectangle fallback
    instead."""
    x0, y0 = max(0, bbox.x), max(0, bbox.y)
    x1, y1 = min(w, bbox.x + bbox.width), min(h, bbox.y + bbox.height)
    if x1 <= x0 or y1 <= y0:
        return None
    crop_mask = _text_mask(img_array, bbox)
    if crop_mask is None:
        return None
    if not _plausible_polarity(np, img_array[y0:y1, x0:x1], crop_mask):
        return None
    m8 = crop_mask.astype(np.uint8) * 255
    kernel = np.ones((3, 3), np.uint8)
    m8 = cv2.dilate(m8, kernel, iterations=DILATE_ITER)
    full = np.zeros((h, w), dtype=bool)
    full[y0:y1, x0:x1] = m8 > 0
    return full


def _bbox_fallback_mask(np, bbox, h: int, w: int):
    """padded bbox-rectangle mask — the pre-Phase-3 behavior, used only
    when glyph segmentation could not separate text from background."""
    x0 = max(0, bbox.x - FALLBACK_PAD)
    y0 = max(0, bbox.y - FALLBACK_PAD)
    x1 = min(w, bbox.x + bbox.width + FALLBACK_PAD)
    y1 = min(h, bbox.y + bbox.height + FALLBACK_PAD)
    full = np.zeros((h, w), dtype=bool)
    if x1 > x0 and y1 > y0:
        full[y0:y1, x0:x1] = True
    return full


def _local_window(np, mask, pad: int, h: int, w: int) -> Optional[Tuple[int, int, int, int]]:
    """bounding box of mask's True pixels, expanded by pad and clipped
    to image bounds — bounds fill/feather compute cost to near the
    region size instead of the whole frame."""
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        return None
    x0 = max(0, int(xs.min()) - pad)
    y0 = max(0, int(ys.min()) - pad)
    x1 = min(w, int(xs.max()) + 1 + pad)
    y1 = min(h, int(ys.max()) + 1 + pad)
    return x0, y0, x1, y1


def _border_ring(cv2, np, mask_win, ring_px: int):
    """ring_px-wide dilation shell just outside mask_win — the sampling
    context for flat/gradient reconstruction."""
    m8 = mask_win.astype(np.uint8)
    kernel = np.ones((3, 3), np.uint8)
    dilated = cv2.dilate(m8, kernel, iterations=max(1, ring_px // 2))
    return (dilated > 0) & ~mask_win


def _feathered_blend(cv2, np, working_win, mask_win, fill_win):
    """alpha-blend fill_win into working_win, softened at the mask
    boundary via a distance-transform falloff (full replace deep inside
    the mask, fading to the original pixels at the edge)."""
    m8 = mask_win.astype(np.uint8)
    dist = cv2.distanceTransform(m8, cv2.DIST_L2, 3)
    alpha = np.clip(dist / FEATHER_PX, 0.0, 1.0)[..., None]
    blended = fill_win * alpha + working_win.astype(np.float64) * (1 - alpha)
    working_win[:] = np.clip(blended, 0, 255).astype(np.uint8)


def _fill_flat(cv2, np, working, full_mask, h: int, w: int) -> None:
    win = _local_window(np, full_mask, RING_PX + int(FEATHER_PX) + 2, h, w)
    if win is None:
        return
    x0, y0, x1, y1 = win
    mask_win = full_mask[y0:y1, x0:x1]
    work_win = working[y0:y1, x0:x1]
    ring = _border_ring(cv2, np, mask_win, RING_PX)
    if ring.any():
        color = np.median(work_win[ring].reshape(-1, 3), axis=0)
    elif mask_win.any():
        color = work_win[mask_win].reshape(-1, 3).mean(axis=0)
    else:
        return
    fill = np.broadcast_to(color, work_win.shape).astype(np.float64)
    _feathered_blend(cv2, np, work_win, mask_win, fill)


def _fill_gradient(cv2, np, working, full_mask, h: int, w: int) -> None:
    win = _local_window(np, full_mask, RING_PX + int(FEATHER_PX) + 2, h, w)
    if win is None:
        return
    x0, y0, x1, y1 = win
    mask_win = full_mask[y0:y1, x0:x1]
    work_win = working[y0:y1, x0:x1]
    ring = _border_ring(cv2, np, mask_win, RING_PX)
    ys, xs = np.nonzero(ring)
    if len(xs) < MIN_RING_PIXELS:
        _fill_flat(cv2, np, working, full_mask, h, w)  # not enough context
        return
    wh, ww = mask_win.shape
    a_mat = np.column_stack([xs, ys, np.ones(len(xs))])
    yy, xx = np.mgrid[0:wh, 0:ww]
    fill = np.empty((wh, ww, 3), dtype=np.float64)
    for c in range(3):
        vals = work_win[ys, xs, c].astype(np.float64)
        try:
            coef, *_ = np.linalg.lstsq(a_mat, vals, rcond=None)
        except Exception:
            _fill_flat(cv2, np, working, full_mask, h, w)
            return
        fill[..., c] = coef[0] * xx + coef[1] * yy + coef[2]
    fill = np.clip(fill, 0, 255)
    _feathered_blend(cv2, np, work_win, mask_win, fill)


def _stroke_px(inst) -> Optional[float]:
    """detected stroke width in pixels, from typography's stroke_ratio x
    font size (Phase 1 enrichment), when available."""
    ch = getattr(inst, "characteristics", None)
    if not ch or not ch.positioning or not ch.size:
        return None
    ratio = ch.positioning.get("stroke_ratio")
    if not ratio:
        return None
    return ratio * ch.size


def erase(asset: Any, text_manifest: TextManifest) -> Any:
    """erase detected text regions and reconstruct the background.

    per-region strategy is selected from inst.background_profile.texture
    (Scene's classification): flat -> median fill, smooth_gradient ->
    local plane reconstruction, textured/unclassified -> batched
    content-aware inpainting (cv2.INPAINT_TELEA). every strategy blends
    at the mask boundary with a soft feather.

    args:
        asset: the source asset.
        text_manifest: manifest with segmentation masks and bg profiles.

    returns:
        the cleansed asset (same modality as the input).
    """
    base = _load_image(asset)
    if base is None:
        return asset

    try:
        import cv2
        import numpy as np
        from PIL import Image
    except ImportError:
        return _erase_pil_fallback(base, text_manifest)

    img_array = np.array(base.convert("RGB"))
    h, w = img_array.shape[:2]
    working = img_array.copy()

    telea_mask = np.zeros((h, w), dtype=bool)
    stroke_widths = []
    touched = False

    for inst in text_manifest.instances:
        if getattr(inst, "dnt", False):
            continue
        bbox = inst.bounding_box
        if bbox is None or bbox.width <= 0 or bbox.height <= 0:
            continue

        full_mask = _region_mask(np, cv2, img_array, bbox, h, w)
        if full_mask is None or not full_mask.any():
            full_mask = _bbox_fallback_mask(np, bbox, h, w)
        if not full_mask.any():
            continue
        touched = True

        texture = (
            inst.background_profile.texture
            if inst.background_profile else None
        )
        if texture == "flat":
            _fill_flat(cv2, np, working, full_mask, h, w)
        elif texture == "smooth_gradient":
            _fill_gradient(cv2, np, working, full_mask, h, w)
        else:
            # textured / patterned / unclassified: one batched inpaint
            # pass at the end, rather than per-region calls that could
            # produce mismatched seams between adjacent regions
            telea_mask |= full_mask
            sw = _stroke_px(inst)
            if sw:
                stroke_widths.append(sw)

    if telea_mask.any():
        radius = TELEA_RADIUS_DEFAULT
        if stroke_widths:
            avg_stroke = sum(stroke_widths) / len(stroke_widths)
            radius = int(max(TELEA_RADIUS_MIN, min(TELEA_RADIUS_MAX, round(avg_stroke / 2))))
        working = cv2.inpaint(working, telea_mask.astype(np.uint8) * 255, radius, cv2.INPAINT_TELEA)

    if touched:
        base = base.convert("RGBA")
        base.paste(Image.fromarray(working), (0, 0))

    return base


def _erase_pil_fallback(base, text_manifest: TextManifest) -> Any:
    """PIL-only fallback when OpenCV is unavailable: median fill + blur."""
    from PIL import Image, ImageFilter, ImageDraw
    import statistics

    for inst in text_manifest.instances:
        if getattr(inst, "dnt", False):
            continue
        bbox = inst.bounding_box
        if bbox is None or bbox.width <= 0 or bbox.height <= 0:
            continue

        x0 = max(0, bbox.x - 2)
        y0 = max(0, bbox.y - 2)
        x1 = min(base.width, bbox.x + bbox.width + 2)
        y1 = min(base.height, bbox.y + bbox.height + 2)

        border_pixels = []
        for px in range(x0, x1):
            for py in (y0, y1 - 1):
                if 0 <= px < base.width and 0 <= py < base.height:
                    border_pixels.append(base.getpixel((px, py))[:3])
        for py in range(y0, y1):
            for px in (x0, x1 - 1):
                if 0 <= px < base.width and 0 <= py < base.height:
                    border_pixels.append(base.getpixel((px, py))[:3])

        if not border_pixels:
            continue

        r = statistics.median(p[0] for p in border_pixels)
        g = statistics.median(p[1] for p in border_pixels)
        b = statistics.median(p[2] for p in border_pixels)
        fill_color = (int(r), int(g), int(b))

        overlay = Image.new("RGBA", base.size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay)
        draw.rectangle(
            [bbox.x, bbox.y, bbox.x + bbox.width, bbox.y + bbox.height],
            fill=fill_color + (255,),
        )
        overlay = overlay.filter(ImageFilter.GaussianBlur(radius=1.5))
        base = Image.alpha_composite(base, overlay)

    return base
