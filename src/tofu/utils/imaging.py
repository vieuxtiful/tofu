## 🍢 imaging — shared asset-loading helpers
## vieuxtiful
"""
image loading shared by layers (scene, cicerone). kept dependency-soft:
returns None when Pillow/numpy are unavailable or the asset is not an
image, so callers can preserve their pass-through contracts.

text_mask() is the widest-fanout function in the package — scene (color
estimation), typography (weight/slant), cleanse (inpaint masks), savor,
and font_matching all bottom out here — so its binarization quality
propagates to four layers at once. see its docstring for the method.
"""

from pathlib import Path
from typing import Any, Optional

# ── binarization tuning ────────────────────────────────────────────────
# a soak is "uneven" once the crop's low-frequency illumination spans more
# than this many grey levels (of 255). below it, one global cut separates
# ink from ground and Otsu is both cheaper and cleaner (no window
# artifacts); above it, no single threshold exists and Otsu necessarily
# swallows the dark end of the gradient as ink.  40/255 ≈ 16% — comfortably
# past sensor noise and JPEG blocking, well under a real lighting falloff.
UNEVEN_SOAK_FLOOR = 40.0
SAUVOLA_K = 0.2      # skimage's default; the paper's 0.5 over-erodes thin ink
SAUVOLA_R = 128.0    # dynamic range of the local standard deviation
SAUVOLA_WINDOW_MAX = 51


def load_rgb(asset: Any):
    """load an asset as an RGB numpy array, or None if not an image.

    accepts a PIL image, file path, or numpy array (assumed RGB).
    """
    try:
        import numpy as np
        from PIL import Image
    except ImportError:
        return None
    if hasattr(asset, "convert") and hasattr(asset, "size"):  # PIL image
        return np.asarray(asset.convert("RGB"))
    if isinstance(asset, (str, Path)):
        try:
            return np.asarray(Image.open(asset).convert("RGB"))
        except Exception:
            return None
    if isinstance(asset, np.ndarray):
        return asset
    return None


def crop_region(asset: Any, bbox, pad: int = 0) -> Optional[Any]:
    """crop a bbox out of an asset as a PIL RGB image, or None when the
    asset can't be loaded or the crop is degenerate. shared by memory
    (thumbnail + phash source) and anything else that needs a plain
    region crop without the glyph-segmentation logic of text_mask()."""
    try:
        from PIL import Image
    except ImportError:
        return None
    img = load_rgb(asset)
    if img is None:
        return None
    h, w = img.shape[:2]
    x0 = max(0, bbox.x - pad)
    y0 = max(0, bbox.y - pad)
    x1 = min(w, bbox.x + bbox.width + pad)
    y1 = min(h, bbox.y + bbox.height + pad)
    if x1 - x0 < 2 or y1 - y0 < 2:
        return None
    return Image.fromarray(img[y0:y1, x0:x1])


def _uneven_soak(gray, cv2) -> float:
    """How unevenly the light fell on this crop, in grey levels.

    Beans that soak unevenly never press into an even block, and a crop lit
    unevenly never splits at one global threshold.  Estimate the background
    by low-pass filtering away the ink (sigma is a quarter of the crop's
    long side, so glyph strokes vanish and the lighting trend survives) and
    report that trend's dynamic range.

    A flat, evenly lit crop scores near 0 no matter how strong its ink
    contrast is, because the blur removes the ink and leaves a constant.
    """
    h, w = gray.shape[:2]
    sigma = max(h, w) / 4.0
    background = cv2.GaussianBlur(gray, (0, 0), sigmaX=sigma, sigmaY=sigma)
    return float(background.max()) - float(background.min())


def _nigari_mask(gray, np) -> Optional[Any]:
    """Sauvola local thresholding — the coagulant that sets curd pocket by
    pocket instead of holding the whole vat at one temperature.

    Sauvola & Pietikäinen (2000), "Adaptive document image binarization",
    Pattern Recognition 33(2):225-236 — itself an adaptation of Niblack
    (1986) that adds the local-standard-deviation term which stops Niblack
    from hallucinating ink in blank background patches.  Each pixel gets
    its own threshold from the mean and standard deviation of a window
    around it, so a gradient that defeats any single global cut is handled
    locally.

    Returns dark-ink foreground, or None when scikit-image is unavailable
    (dependency-soft, per this module's contract) or the window degenerates.
    """
    try:
        from skimage.filters import threshold_sauvola  # deferred: pulls scipy
    except ImportError:
        return None
    h, w = gray.shape[:2]
    span = min(h, w)
    # window must be odd, and small enough to sit inside the crop; half the
    # short side keeps both ink and ground in view at typical text scales
    window = max(3, min(SAUVOLA_WINDOW_MAX, span // 2))
    if window % 2 == 0:
        window -= 1
    window = max(3, window)
    try:
        threshold = threshold_sauvola(gray, window_size=window, k=SAUVOLA_K, r=SAUVOLA_R)
    except Exception:
        return None
    return np.asarray(gray < threshold)


def text_mask(img, bbox, refine: bool = True) -> Optional[Any]:
    """binary glyph mask for the text inside a bbox of a full RGB image.

    returns a bool ndarray of the crop's shape (True = text stroke), or
    None when the crop is degenerate or segmentation finds no separation.

    method: the text/background split is thresholded either globally by
    Otsu (Otsu 1979) or locally by Sauvola (Sauvola & Pietikäinen 2000),
    chosen per crop by _uneven_soak(). Otsu assumes a bimodal histogram and
    one threshold for the whole crop, which is exactly right on flat
    signage and exactly wrong under a lighting gradient — the failure mode
    that dominates street photography, where it swallows the shaded end of
    the background as ink. Sauvola pays a little window noise on flat crops
    for immunity to that gradient, so each crop gets the estimator its
    illumination actually warrants rather than a blanket choice.

    whichever wins, text is assumed the MINORITY class, then refined with
    GrabCut (Rother et al. 2004) when the crop is large enough for its
    border ring, which is far more robust on textured backgrounds.

    Sauvola degrading (scikit-image absent, or a mask that comes back all
    ink or all ground) falls through to Otsu rather than failing, so this
    can only ever match or beat the previous Otsu-only behaviour.

    shared by scene (color estimation), typography (weight/slant), cleanse
    (stroke-level inpaint masks), savor, and font_matching.
    """
    try:
        import cv2
        import numpy as np
    except ImportError:
        return None
    h, w = img.shape[:2]
    x0, y0 = max(0, bbox.x), max(0, bbox.y)
    x1 = min(w, bbox.x + bbox.width)
    y1 = min(h, bbox.y + bbox.height)
    if x1 - x0 < 3 or y1 - y0 < 3:
        return None
    crop = img[y0:y1, x0:x1]
    gray = cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY)

    fg = None
    if _uneven_soak(gray, cv2) > UNEVEN_SOAK_FLOOR:
        candidate = _nigari_mask(gray, np)
        # a threshold that finds ink everywhere or nowhere has found nothing;
        # let Otsu answer instead of returning None as an all-or-nothing mask
        if candidate is not None and 0 < int(candidate.sum()) < candidate.size:
            fg = candidate
    if fg is None:
        _, otsu_mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        fg = otsu_mask > 0
    n_fg = int(fg.sum())
    if n_fg == 0 or n_fg == fg.size:
        return None

    mask = fg if n_fg <= (fg.size - n_fg) else ~fg

    # GrabCut refinement: needs a border margin; skip on tiny crops
    ch, cw = crop.shape[:2]
    if refine and ch >= 12 and cw >= 12:
        try:
            gc_mask = np.zeros((ch, cw), np.uint8)
            gc_mask[mask] = cv2.GC_FGD
            gc_mask[~mask] = cv2.GC_BGD
            border = 3
            gc_mask[:border, :] = cv2.GC_PR_BGD
            gc_mask[-border:, :] = cv2.GC_PR_BGD
            gc_mask[:, :border] = cv2.GC_PR_BGD
            gc_mask[:, -border:] = cv2.GC_PR_BGD
            bgd_model = np.zeros((1, 65), np.float64)
            fgd_model = np.zeros((1, 65), np.float64)
            cv2.grabCut(
                crop, gc_mask, None, bgd_model, fgd_model, 3,
                cv2.GC_INIT_WITH_MASK,
            )
            refined = (gc_mask == cv2.GC_FGD) | (gc_mask == cv2.GC_PR_FGD)
            if refined.sum() > 0 and refined.sum() < refined.size:
                mask = refined
        except Exception:
            pass
    return mask
