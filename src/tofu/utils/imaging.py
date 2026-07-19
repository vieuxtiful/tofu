## 🍢 imaging — shared asset-loading helpers
## vieuxtiful
"""
image loading shared by layers (scene, cicerone). kept dependency-soft:
returns None when Pillow/numpy are unavailable or the asset is not an
image, so callers can preserve their pass-through contracts.
"""

from pathlib import Path
from typing import Any, Optional


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


def text_mask(img, bbox, refine: bool = True) -> Optional[Any]:
    """binary glyph mask for the text inside a bbox of a full RGB image.

    returns a bool ndarray of the crop's shape (True = text stroke), or
    None when the crop is degenerate or segmentation finds no separation.

    method: Otsu thresholding (Otsu 1979) for the initial text/background
    split — text is assumed the MINORITY class — refined with GrabCut
    (Rother et al. 2004) when the crop is large enough for its border
    ring, which is far more robust on textured or gradient backgrounds.
    shared by scene (color estimation), typography (weight/slant), and —
    from Phase 3 — cleanse (stroke-level inpaint masks).
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
