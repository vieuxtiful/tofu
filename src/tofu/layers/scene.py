## 🍢 Scene — Layer 2
## vieuxtiful
"""
semantic context & style analysis layer.

scene has two jobs:

1. pre-pass — analyze_regions(): detect candidate text-bearing surfaces
   (signs, panels, bordered regions) BEFORE text detection runs. cicerone
   uses these to constrain detection and suppress false positives.
   backends are swappable adapters behind the SceneBackend interface:
     - ClassicalCVBackend (default): Canny edges + contour analysis +
       color statistics via OpenCV. fast, no model download, honest
       heuristic labels ("panel" / "bordered_region" / "surface").
     - SAMBackend (optional): Segment Anything automatic mask generation.
       class-agnostic masks labeled "surface"; requires segment-anything
       + a checkpoint, and is slow on CPU-only hosts. opt-in.
     - NullSceneBackend: empty results when OpenCV is unavailable.

2. enrichment — analyze(): fill per-instance StyleProfil (estimated text
   color) and BgProfil (background color, texture hint, containing scene
   region's label) used by cleanse for faithful inpainting and by scribe
   for style-matched re-rendering. existing user-set values are never
   overwritten.
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, List, Optional, Tuple

from tofu.core.types import (
    TextManifest, StyleProfil, BgProfil, SceneRegion, BBox, CharactText,
)
from tofu.utils.imaging import load_rgb as _load_rgb, text_mask as _text_mask


def _hex(rgb) -> str:
    r, g, b = (int(max(0, min(255, round(float(v))))) for v in rgb[:3])
    return f"#{r:02x}{g:02x}{b:02x}"


def _containment(inner: BBox, outer: BBox) -> float:
    """fraction of inner's area that lies inside outer."""
    ix = max(inner.x, outer.x)
    iy = max(inner.y, outer.y)
    ax = min(inner.x + inner.width, outer.x + outer.width)
    ay = min(inner.y + inner.height, outer.y + outer.height)
    if ax <= ix or ay <= iy:
        return 0.0
    inter = (ax - ix) * (ay - iy)
    area = inner.width * inner.height
    return inter / area if area > 0 else 0.0


# -- backends -----------------------------------------------------------------

class SceneBackend(ABC):
    """swappable region-detection engine adapter."""

    name: str = "base"

    @abstractmethod
    def analyze(self, asset: Any) -> List[SceneRegion]:
        """detect candidate text-bearing surfaces in the asset."""


class NullSceneBackend(SceneBackend):
    """no-engine fallback: valid empty results."""

    name = "null"

    def analyze(self, asset: Any) -> List[SceneRegion]:
        return []


class ClassicalCVBackend(SceneBackend):
    """Canny + contours + MSER + color statistics via OpenCV.

    Two complementary detectors run in parallel:
      1. Contour analysis (Canny + approxPolyDP) finds structural surfaces:
         panels, bordered regions, signs — large quadrilateral areas that
         likely contain text.
      2. MSER (Maximally Stable Extremal Regions) finds text-like blobs:
         stable connected components against varying thresholds. MSER is
         the classical CV text detector and excels at finding individual
         text characters and short strings that contour analysis misses.

    Labels are honest surface heuristics, not semantic classes:
      - "panel": near-quadrilateral contour with uniform interior color
      - "bordered_region": near-quadrilateral, non-uniform interior
      - "text_cluster": MSER blob cluster (aggregated into bounding boxes)
      - "surface": any other salient contour
    """

    name = "classical"

    def __init__(
        self,
        max_regions: int = 16,
        min_area_frac: float = 0.003,
        work_dim: int = 1600,
        uniform_std: float = 32.0,
        mser_delta: int = 5,
        mser_min_area: int = 60,
        mser_max_area: int = 14400,
        mser_merge_dist: int = 12,
    ):
        self.max_regions = max_regions
        self.min_area_frac = min_area_frac
        self.work_dim = work_dim
        self.uniform_std = uniform_std
        self.mser_delta = mser_delta
        self.mser_min_area = mser_min_area
        self.mser_max_area = mser_max_area
        self.mser_merge_dist = mser_merge_dist

    def _contour_regions(self, work, scale: float) -> List[SceneRegion]:
        import cv2
        import numpy as np
        gray = cv2.cvtColor(work, cv2.COLOR_RGB2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        edges = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=1)
        contours, _ = cv2.findContours(
            edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        min_area = self.min_area_frac * work.shape[0] * work.shape[1]
        inv = 1.0 / scale
        regions: List[SceneRegion] = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < min_area:
                continue
            x, y, bw, bh = cv2.boundingRect(cnt)
            if bw < 8 or bh < 8:
                continue
            rect_fill = area / float(bw * bh)
            approx = cv2.approxPolyDP(cnt, 0.02 * cv2.arcLength(cnt, True), True)
            is_quad = len(approx) == 4 and rect_fill > 0.6
            roi = work[y:y + bh, x:x + bw].reshape(-1, 3)
            mean = roi.mean(axis=0)
            std = float(roi.std())
            uniform = std < self.uniform_std
            if is_quad and uniform:
                label = "panel"
            elif is_quad:
                label = "bordered_region"
            else:
                label = "surface"
            confidence = round(min(1.0, 0.3 + 0.4 * rect_fill + (0.3 if uniform else 0.0)), 3)
            regions.append(SceneRegion(
                bbox=BBox(
                    x=int(x * inv), y=int(y * inv),
                    width=int(bw * inv), height=int(bh * inv),
                ),
                semantic_label=label,
                confidence=confidence,
                background_color=_hex(mean),
                border_detected=is_quad,
                polygon=[(int(p[0][0] * inv), int(p[0][1] * inv)) for p in approx],
            ))
        return regions

    def _mser_regions(self, work, scale: float) -> List[SceneRegion]:
        import cv2
        import numpy as np
        gray = cv2.cvtColor(work, cv2.COLOR_RGB2GRAY)
        mser = cv2.MSER_create(
            delta=self.mser_delta,
            min_area=self.mser_min_area,
            max_area=self.mser_max_area,
        )
        try:
            msers, bboxes = mser.detectRegions(gray)
        except Exception:
            return []
        if bboxes is None or len(bboxes) == 0:
            return []
        inv = 1.0 / scale
        raw = [
            (int(bx * inv), int(by * inv), int(bw * inv), int(bh * inv))
            for bx, by, bw, bh in bboxes
            if bw >= 4 and bh >= 4
        ]
        if not raw:
            return []
        raw.sort(key=lambda b: (b[1], b[0]))
        merged: List[Tuple[int, int, int, int]] = []
        for x, y, w, h in raw:
            cx, cy = x + w // 2, y + h // 2
            placed = False
            for i, (mx, my, mw, mh) in enumerate(merged):
                mcx, mcy = mx + mw // 2, my + mh // 2
                if (abs(cx - mcx) <= (w + mw) // 2 + self.mser_merge_dist and
                    abs(cy - mcy) <= (h + mh) // 2 + self.mser_merge_dist):
                    nx = min(x, mx)
                    ny = min(y, my)
                    nx2 = max(x + w, mx + mw)
                    ny2 = max(y + h, my + mh)
                    merged[i] = (nx, ny, nx2 - nx, ny2 - ny)
                    placed = True
                    break
            if not placed:
                merged.append((x, y, w, h))
        min_area = self.min_area_frac * work.shape[0] * work.shape[1]
        regions: List[SceneRegion] = []
        for x, y, w, h in merged:
            if w * h < min_area:
                continue
            if w < 8 or h < 8:
                continue
            aspect = w / max(h, 1)
            if aspect > 15 or aspect < 0.05:
                continue
            sx0 = max(0, int(x / inv))
            sy0 = max(0, int(y / inv))
            sx1 = min(work.shape[1], int((x + w) / inv))
            sy1 = min(work.shape[0], int((y + h) / inv))
            roi = work[sy0:sy1, sx0:sx1]
            mean = roi.reshape(-1, 3).mean(axis=0) if roi.size > 0 else np.array([128, 128, 128])
            regions.append(SceneRegion(
                bbox=BBox(x=x, y=y, width=w, height=h),
                semantic_label="text_cluster",
                confidence=0.55,
                background_color=_hex(mean),
                border_detected=False,
            ))
        return regions

    def analyze(self, asset: Any) -> List[SceneRegion]:
        img = _load_rgb(asset)
        if img is None:
            return []
        try:
            import cv2  # noqa: F401
            import numpy as np  # noqa: F401
        except ImportError:
            return []
        h, w = img.shape[:2]
        scale = 1.0
        work = img
        if max(h, w) > self.work_dim:
            scale = self.work_dim / max(h, w)
            import cv2
            work = cv2.resize(img, (max(1, int(w * scale)), max(1, int(h * scale))))

        contour_regions = self._contour_regions(work, scale)
        mser_regions = self._mser_regions(work, scale)
        all_regions = contour_regions + mser_regions

        all_regions.sort(key=lambda r: r.bbox.width * r.bbox.height, reverse=True)
        kept: List[SceneRegion] = []
        for region in all_regions:
            if any(_containment(region.bbox, k.bbox) > 0.85 for k in kept):
                continue
            kept.append(region)
            if len(kept) >= self.max_regions:
                break
        return kept


class SAMBackend(SceneBackend):
    """Segment Anything automatic mask generation (optional, opt-in).

    SAM is class-agnostic — every region is labeled "surface". requires
    the `segment-anything` package and a downloaded checkpoint (set via
    PipelineCfg.scene_model_path); automatic mask generation takes on the
    order of minutes per image on CPU-only hosts, so this is never the
    default there.
    """

    name = "sam"

    def __init__(
        self,
        checkpoint_path: Optional[str] = None,
        model_type: str = "vit_b",
        max_regions: int = 12,
    ):
        self.checkpoint_path = checkpoint_path
        self.model_type = model_type
        self.max_regions = max_regions
        self._generator = None

    def _ensure(self):
        if self._generator is not None:
            return
        from segment_anything import (  # deferred: optional heavy import
            sam_model_registry, SamAutomaticMaskGenerator,
        )
        if not self.checkpoint_path or not Path(self.checkpoint_path).exists():
            raise FileNotFoundError(
                "SAM checkpoint not found; set PipelineCfg.scene_model_path "
                "to a downloaded checkpoint (e.g. sam_vit_b_01ec64.pth)"
            )
        sam = sam_model_registry[self.model_type](checkpoint=str(self.checkpoint_path))
        self._generator = SamAutomaticMaskGenerator(sam)

    def analyze(self, asset: Any) -> List[SceneRegion]:
        img = _load_rgb(asset)
        if img is None:
            return []
        self._ensure()
        masks = self._generator.generate(img)
        masks.sort(key=lambda m: m.get("area", 0), reverse=True)
        return [
            SceneRegion(
                bbox=BBox(
                    x=int(m["bbox"][0]), y=int(m["bbox"][1]),
                    width=int(m["bbox"][2]), height=int(m["bbox"][3]),
                ),
                semantic_label="surface",
                confidence=float(m.get("stability_score", 0.5)),
            )
            for m in masks[:self.max_regions]
        ]


# -- module-level engine management ------------------------------------------

_default_backend: Optional[SceneBackend] = None


def get_backend() -> SceneBackend:
    """default backend: classical CV when OpenCV is installed, Null otherwise."""
    global _default_backend
    if _default_backend is None:
        try:
            import cv2  # noqa: F401
            _default_backend = ClassicalCVBackend()
        except ImportError:
            _default_backend = NullSceneBackend()
    return _default_backend


def set_backend(backend: SceneBackend) -> None:
    """swap the engine (e.g. SAMBackend) without touching the pipeline."""
    global _default_backend
    _default_backend = backend


# -- layer entry points -------------------------------------------------------

def analyze_regions(
    asset: Any,
    backend: Optional[SceneBackend] = None,
) -> List[SceneRegion]:
    """pre-pass: detect candidate text-bearing surfaces in the asset.

    runs BEFORE cicerone; the returned regions constrain text detection
    and seed per-instance background labels during enrichment.
    """
    engine = backend or get_backend()
    return engine.analyze(asset)


def _estimate_colors(img, bbox: BBox) -> Tuple[Optional[str], Optional[str], Optional[float]]:
    """estimate (text_color_hex, bg_color_hex, bg_std) inside a bbox.

    segmentation is the shared Otsu+GrabCut glyph mask (utils.imaging.
    text_mask) — the same mask typography and cleanse consume, so color,
    weight, and erasure all agree on which pixels are strokes.
    """
    h, w = img.shape[:2]
    x0, y0 = max(0, bbox.x), max(0, bbox.y)
    x1 = min(w, bbox.x + bbox.width)
    y1 = min(h, bbox.y + bbox.height)
    if x1 - x0 < 3 or y1 - y0 < 3:
        return None, None, None
    crop = img[y0:y1, x0:x1]
    mask = _text_mask(img, bbox)
    if mask is None:
        flat = crop.reshape(-1, 3)
        return None, _hex(flat.mean(axis=0)), float(flat.std())
    bg_pixels = crop[~mask]
    return (
        _hex(crop[mask].mean(axis=0)),
        _hex(bg_pixels.mean(axis=0)),
        float(bg_pixels.std()),
    )


def _containing_region(
    regions: List[SceneRegion], bbox: BBox
) -> Optional[SceneRegion]:
    """smallest scene region containing the bbox center, if any."""
    cx = bbox.x + bbox.width / 2
    cy = bbox.y + bbox.height / 2
    hits = [
        r for r in regions
        if r.bbox.x <= cx <= r.bbox.x + r.bbox.width
        and r.bbox.y <= cy <= r.bbox.y + r.bbox.height
    ]
    if not hits:
        return None
    return min(hits, key=lambda r: r.bbox.width * r.bbox.height)


def analyze(asset: Any, text_manifest: TextManifest) -> TextManifest:
    """analyze scene semantics and text styling for each instance.

    args:
        asset: the source asset the manifest was detected from.
        text_manifest: cicerone's output.

    returns:
        the same manifest with scene_regions populated (if not already)
        and style_profile / background_profile enriched on every
        instance. user-set profile values are never overwritten.
    """
    img = _load_rgb(asset)

    if not text_manifest.scene_regions:
        try:
            text_manifest.scene_regions = analyze_regions(asset)
        except Exception:
            text_manifest.scene_regions = []

    for inst in text_manifest.instances:
        if inst.style_profile is None:
            inst.style_profile = StyleProfil()
        if inst.background_profile is None:
            inst.background_profile = BgProfil()
        if img is None or inst.bounding_box is None:
            continue
        try:
            text_hex, bg_hex, bg_std = _estimate_colors(img, inst.bounding_box)
        except Exception:
            continue
        sp, bp = inst.style_profile, inst.background_profile
        if sp.color is None and text_hex:
            sp.color = text_hex
        if bp.dominant_color is None and bg_hex:
            bp.dominant_color = bg_hex
        if bp.texture is None and bg_std is not None:
            bp.texture = "flat" if bg_std < 24 else "textured"
        if bp.semantic_label is None:
            region = _containing_region(text_manifest.scene_regions, inst.bounding_box)
            if region is not None:
                bp.semantic_label = region.semantic_label

        # typography: weight/slant/size/rotation from the same glyph mask
        # the colors came from. user-set values are never overwritten.
        try:
            from tofu.layers.typography import analyze_region as _typo
            poly = (
                inst.segmentation_mask.polygon
                if inst.segmentation_mask else None
            )
            typo = _typo(img, inst.bounding_box, poly)
        except Exception:
            typo = None
        if typo is not None:
            if sp.font_weight is None and typo.weight in ("bold", "light"):
                sp.font_weight = typo.weight
            if sp.italic is None and typo.italic:
                sp.italic = True
            if inst.characteristics is None:
                inst.characteristics = CharactText()
            ch = inst.characteristics
            if ch.font_style is None:
                ch.font_style = typo.label()
            if ch.size is None:
                ch.size = typo.font_px
            pos = dict(ch.positioning or {})
            pos.setdefault("rotation_deg", typo.rotation_deg)
            pos.setdefault("slant_deg", typo.slant_deg)
            pos.setdefault("stroke_ratio", typo.stroke_ratio)
            ch.positioning = pos
    return text_manifest
