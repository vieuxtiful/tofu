## 🍢 Inpaint -- Non-destructive treatment patches for the localized Render canvas.
## vieuxtiful

from typing import Any, Dict, List, Tuple


def make_patch(
    base_img: Any,
    polygon: List[Tuple[int, int]] | None = None,
    mode: str = "auto",
    points: List[Tuple[int, int]] | None = None,
    radius: int = 18,
    hardness: float = 0.85,
    blur_strength: float = 0.5,
    opacity: float = 1.0,
    clone_source: Tuple[int, int] | None = None,
):
    """Return an RGBA treatment crop, bbox, and applied strategy.

    ``base_img`` is the CURRENT localized treatment base (cleanse plus prior
    patches), never the source asset.  That is what makes a brush action
    context-aware and ordered.  The persisted crop remains self-contained,
    so undo/redo does not re-run a non-deterministic image operation.
    """
    from PIL import Image
    import cv2
    import numpy as np

    image = base_img.convert("RGB") if hasattr(base_img, "convert") else Image.open(base_img).convert("RGB")
    polygon = polygon or []
    points = points or []
    if len(polygon) < 3 and not points:
        raise ValueError("a lasso polygon or brush stroke is required")
    arr = np.asarray(image).copy()
    mask = np.zeros(arr.shape[:2], dtype=np.uint8)
    if len(polygon) >= 3:
        pts = np.asarray(polygon, dtype=np.int32)
        cv2.fillPoly(mask, [pts.reshape((-1, 1, 2))], 255)
    else:
        pts = np.asarray(points, dtype=np.int32)
        brush_radius = max(1, int(radius))
        if len(pts) == 1:
            cv2.circle(mask, tuple(pts[0]), brush_radius, 255, -1, lineType=cv2.LINE_AA)
        else:
            cv2.polylines(mask, [pts.reshape((-1, 1, 2))], False, 255,
                          thickness=brush_radius * 2, lineType=cv2.LINE_AA)
            for point in pts:
                cv2.circle(mask, tuple(point), brush_radius, 255, -1, lineType=cv2.LINE_AA)
    ys, xs = np.where(mask > 0)
    if not len(xs):
        raise ValueError("treatment mask is outside the image")
    x0, y0, x1, y1 = int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1
    x0, y0 = max(0, x0), max(0, y0)
    x1, y1 = min(image.width, x1), min(image.height, y1)
    if x1 <= x0 or y1 <= y0:
        raise ValueError("polygon is outside the image")
    # The canvas never asks an editor to decide whether an area is "smooth"
    # or "textured".  Auto mode estimates the immediately surrounding known
    # pixels: connected edge structure favours Navier--Stokes propagation;
    # otherwise Telea is the more conservative local heal.  The explicit
    # modes remain API-compatible for old persisted patches and tests.
    if mode == "clone":
        if clone_source is None or not points:
            raise ValueError("clone requires a source anchor and destination stroke")
        dx, dy = int(clone_source[0] - points[0][0]), int(clone_source[1] - points[0][1])
        yy, xx = np.indices(mask.shape)
        sx, sy = xx + dx, yy + dy
        valid = (mask > 0) & (sx >= 0) & (sx < image.width) & (sy >= 0) & (sy < image.height)
        cloned = arr.copy()
        cloned[valid] = arr[sy[valid], sx[valid]]
        crop = cloned[y0:y1, x0:x1]
        # Do not composite pixels whose aligned sample falls outside the asset.
        mask[~valid] = 0
        strategy = "aligned_clone"
    elif mode == "blur":
        intensity = max(0.0, min(1.0, float(blur_strength)))
        sigma = max(1.0, float(radius) * max(0.1, intensity))
        blurred = cv2.GaussianBlur(arr, (0, 0), sigma)
        # Text-aware masking: detect high-frequency text edges via Laplacian
        # and blend only those pixels, leaving smooth background untouched.
        gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
        laplacian = np.abs(cv2.Laplacian(gray, cv2.CV_64F))
        # Smooth the contrast map so the blur feather follows stroke widths
        contrast = cv2.GaussianBlur(laplacian.astype(np.float32), (0, 0), sigma * 0.5)
        cmax = float(contrast.max())
        if cmax > 0:
            contrast = contrast / cmax
        # Intensity scales how aggressively text pixels adopt the blurred version
        text_weight = np.clip(contrast * intensity * 2.0, 0, 1)[..., np.newaxis]
        result = (arr.astype(np.float32) * (1.0 - text_weight) +
                  blurred.astype(np.float32) * text_weight)
        crop = np.clip(result, 0, 255).astype(np.uint8)[y0:y1, x0:x1]
        strategy = "text_blur"
    elif mode in {"auto", "heal", "context_fill"}:
        expanded = cv2.dilate(mask, np.ones((5, 5), np.uint8), iterations=2)
        ring = (expanded > 0) & ~(mask > 0)
        gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
        edges = cv2.Canny(gray, 70, 150)
        edge_density = float((edges[ring] > 0).mean()) if ring.any() else 0.0
        strategy = "navier_stokes" if edge_density >= 0.10 else "telea"
        flag = cv2.INPAINT_NS if strategy == "navier_stokes" else cv2.INPAINT_TELEA
        out = cv2.inpaint(arr, mask, max(2, min(9, int(radius / 4) or 3)), flag)
        crop = out[y0:y1, x0:x1]
    else:
        strategy = "navier_stokes" if mode in {"texture", "navier_stokes"} else "telea"
        flag = cv2.INPAINT_NS if strategy == "navier_stokes" else cv2.INPAINT_TELEA
        out = cv2.inpaint(arr, mask, max(2, min(9, int(radius / 4) or 3)), flag)
        crop = out[y0:y1, x0:x1]
    # A slightly feathered alpha gives a healing-brush seam instead of a hard
    # stamped edge.  It never changes the actual treatment pixels.
    blur = max(0, int((1.0 - max(0.0, min(1.0, hardness))) * radius))
    alpha = cv2.GaussianBlur(mask, (0, 0), blur) if blur else mask
    alpha = np.clip(alpha.astype(np.float32) * max(0.0, min(1.0, float(opacity))), 0, 255).astype(np.uint8)
    alpha = alpha[y0:y1, x0:x1]
    rgba = np.dstack([crop, alpha])
    return (Image.fromarray(rgba, "RGBA"),
            {"x": x0, "y": y0, "width": x1 - x0, "height": y1 - y0}, strategy)
