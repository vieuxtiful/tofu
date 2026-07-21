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
from typing import Any, Dict, List, Optional, Tuple

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


def _orientation_bucket(r: BBox) -> str:
    """coarse tall/wide/square classification, used only to decide
    whether two highly-overlapping candidate surfaces are plausibly the
    SAME content (safe to dedupe) or two DIFFERENT pieces of content
    that happen to share pixels (e.g. small horizontal English text
    written across a large vertical CJK sign) -- see _is_duplicate_surface."""
    if r.height <= 0 or r.width <= 0:
        return "square"
    ratio = r.width / r.height
    if ratio > 1.3:
        return "wide"
    if ratio < 0.77:
        return "tall"
    return "square"


def _is_duplicate_surface(candidate: BBox, kept: List[BBox]) -> bool:
    """True when `candidate` should be discarded as a near-duplicate of
    an already-kept surface.

    containment alone isn't a safe dedup signal: a small candidate can
    be >85% inside a large one while representing genuinely DIFFERENT
    content -- e.g. a small horizontal English label sitting on top of
    a large vertical CJK sign (measured live: china-street's "MING" on
    "上海明牌" was silently discarded here before cicerone ever saw it).
    requiring matching orientation keeps the dedup's original purpose
    (collapsing near-duplicate MSER/contour blobs of the SAME surface)
    while no longer discarding orthogonal overlapping content outright.
    """
    return any(
        _containment(candidate, k) > 0.85
        and _orientation_bucket(candidate) == _orientation_bucket(k)
        for k in kept
    )


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
        max_region_frac: float = 0.85,
        swt_cv_max: float = 0.65,
        mser_cluster_max_frac: float = 0.12,
    ):
        self.max_regions = max_regions
        self.min_area_frac = min_area_frac
        self.work_dim = work_dim
        self.uniform_std = uniform_std
        self.mser_delta = mser_delta
        self.mser_min_area = mser_min_area
        self.mser_max_area = mser_max_area
        self.mser_merge_dist = mser_merge_dist
        # a region covering ~the whole frame IS the frame, not a surface —
        # worse, the largest-first containment dedup would swallow every
        # real surface inside it (measured: both street photos returned
        # exactly one frame-sized region and nothing else)
        self.max_region_frac = max_region_frac
        # stroke-width coefficient-of-variation gate for MSER components
        # (SWT cascade — Epshtein 2010 / Neumann & Matas 2012): text
        # strokes have near-uniform width; blobs do not
        self.swt_cv_max = swt_cv_max
        # hard cap on a merged MSER cluster's envelope AREA, as a fraction
        # of the frame — single-linkage clustering over hundreds of
        # stroke-like components in a dense signage scene chains
        # transitively regardless of how the pairwise distance is defined;
        # only refusing merges that would exceed a real-sign-sized envelope
        # stops the snowball (measured: gemini-street collapsed 703
        # surviving components into one 1406x766 blob without this cap)
        self.mser_cluster_max_frac = mser_cluster_max_frac

    def _contour_regions(self, work, scale: float) -> List[SceneRegion]:
        """quad-shaped surfaces (panels/bordered signs), with a rescue
        pass for visually dense scenes.

        the default Canny+dilate pass can saturate: a busy street scene
        (countless small signs, neon, texture) produces an edge map so
        dense that findContours returns one blob approximating the whole
        frame, and even a large, unmistakable rectangular sign never
        emerges as its own contour — measured on japan-street.jpeg, the
        dilated edge map was 61% dense ACROSS THE ENTIRE FRAME (not just
        busy sub-areas), hiding a giant high-contrast banner in the
        noise entirely. only escalate to a second, more conservative
        pass (Gaussian blur suppresses fine background texture while
        preserving strong large-scale contrast edges; higher thresholds
        + no dilation avoid re-bridging noise back together) when the
        cheap default demonstrably found nothing real — mirrors this
        project's own established pattern of trying cheap first and
        escalating only on failure (cicerone's adaptive stage-2, surface
        probe, auto-probe language).
        """
        regions, found_real_quad = self._contour_pass(
            work, scale, blur=False, canny=(50, 150), dilate_iters=1,
        )
        if not found_real_quad:
            rescue, _ = self._contour_pass(
                work, scale, blur=True, canny=(100, 200), dilate_iters=0,
            )
            regions = regions + rescue
        return regions

    def _contour_pass(
        self, work, scale: float, blur: bool,
        canny: Tuple[int, int], dilate_iters: int,
    ) -> Tuple[List[SceneRegion], bool]:
        import cv2
        import numpy as np
        gray = cv2.cvtColor(work, cv2.COLOR_RGB2GRAY)
        if blur:
            gray = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(gray, canny[0], canny[1])
        if dilate_iters > 0:
            edges = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=dilate_iters)
        contours, _ = cv2.findContours(
            edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        min_area = self.min_area_frac * work.shape[0] * work.shape[1]
        frame_area = work.shape[0] * work.shape[1]
        inv = 1.0 / scale
        regions: List[SceneRegion] = []
        found_real_quad = False
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
            # a quad approximating the whole frame is the frame, not a
            # real sign (analyze()'s own max_region_frac filter would
            # discard it downstream anyway) — it must not count as
            # evidence that this pass "found something," or the rescue
            # pass below would never fire for exactly the dense scenes
            # it exists to rescue
            if is_quad and (bw * bh) / frame_area <= self.max_region_frac:
                found_real_quad = True
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
        return regions, found_real_quad

    def _stroke_like(self, np, cv2, pts, bx, by, bw, bh) -> bool:
        """SWT-style component gate: keep only components whose stroke
        width is near-uniform (coefficient of variation of the distance-
        transform core below swt_cv_max) — the classical text/non-text
        discriminator over MSER components."""
        comp = np.zeros((bh, bw), np.uint8)
        comp[pts[:, 1] - by, pts[:, 0] - bx] = 255
        dt = cv2.distanceTransform(comp, cv2.DIST_L2, 3)
        vals = dt[comp > 0]
        if vals.size == 0:
            return False
        peak = float(vals.max())
        if peak <= 0:
            return False
        core = vals[vals >= 0.5 * peak]
        if core.size < 4:
            return False
        mean = float(core.mean())
        if mean <= 0:
            return False
        # stroke must also be thin relative to the component — a solid
        # blob has stroke width ~ its own smaller dimension
        if 2.0 * mean > 0.7 * min(bw, bh) and min(bw, bh) > 8:
            return False
        return float(core.std()) / mean <= self.swt_cv_max

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
        raw = []
        for pts, (bx, by, bw, bh) in zip(msers, bboxes):
            if bw < 4 or bh < 4:
                continue
            try:
                if not self._stroke_like(np, cv2, pts, bx, by, bw, bh):
                    continue
            except Exception:
                pass  # gate is best-effort; never drop on internal error
            raw.append(
                (int(bx * inv), int(by * inv), int(bw * inv), int(bh * inv))
            )
        if not raw:
            return []

        # union-find over the RAW components with an envelope-area growth
        # cap. single-linkage clustering (any two "close" components merge,
        # transitively) is what naturally chains through hundreds of small
        # stroke-like components in a dense signage scene — no pairwise
        # distance definition avoids that on its own. refusing any merge
        # whose resulting envelope would exceed a real-sign-sized area
        # bounds cluster growth directly, regardless of how many nearby
        # components exist or in what order they're visited.
        n = len(raw)
        parent = list(range(n))
        envelope = [
            (raw[i][0], raw[i][1], raw[i][0] + raw[i][2], raw[i][1] + raw[i][3])
            for i in range(n)
        ]
        frame_area = (work.shape[0] * work.shape[1]) / max(scale * scale, 1e-9)
        cluster_cap = self.mser_cluster_max_frac * frame_area

        def find(i: int) -> int:
            while parent[i] != i:
                parent[i] = parent[parent[i]]
                i = parent[i]
            return i

        def close(a, b) -> bool:
            ax, ay, aw, ah = a
            bx, by, bw, bh = b
            dx = max(0, max(ax, bx) - min(ax + aw, bx + bw))
            dy = max(0, max(ay, by) - min(ay + ah, by + bh))
            return dx <= self.mser_merge_dist and dy <= self.mser_merge_dist

        for i in range(n):
            for j in range(i + 1, n):
                if not close(raw[i], raw[j]):
                    continue
                ri, rj = find(i), find(j)
                if ri == rj:
                    continue
                ex0 = min(envelope[ri][0], envelope[rj][0])
                ey0 = min(envelope[ri][1], envelope[rj][1])
                ex1 = max(envelope[ri][2], envelope[rj][2])
                ey1 = max(envelope[ri][3], envelope[rj][3])
                if (ex1 - ex0) * (ey1 - ey0) > cluster_cap:
                    continue  # would form an oversized blob — refuse
                parent[rj] = ri
                envelope[ri] = (ex0, ey0, ex1, ey1)

        groups: Dict[int, List[int]] = {}
        for i in range(n):
            groups.setdefault(find(i), []).append(i)

        merged: List[Tuple[int, int, int, int]] = []
        for members in groups.values():
            xs0 = min(raw[i][0] for i in members)
            ys0 = min(raw[i][1] for i in members)
            xs1 = max(raw[i][0] + raw[i][2] for i in members)
            ys1 = max(raw[i][1] + raw[i][3] for i in members)
            merged.append((xs0, ys0, xs1 - xs0, ys1 - ys0))

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

        # frame-sized regions are the frame, not surfaces — and since the
        # dedup below keeps largest-first, one of them would swallow every
        # real surface it contains
        frame_area = float(h * w)
        all_regions = [
            r for r in all_regions
            if (r.bbox.width * r.bbox.height) / frame_area <= self.max_region_frac
        ]

        all_regions.sort(key=lambda r: r.bbox.width * r.bbox.height, reverse=True)
        kept: List[SceneRegion] = []
        for region in all_regions:
            if _is_duplicate_surface(region.bbox, [k.bbox for k in kept]):
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


def _estimate_colors(img, bbox: BBox):
    """estimate (text_color_hex, bg_color_hex, bg_std, glyph_mask, crop)
    inside a bbox.

    segmentation is the shared Otsu+GrabCut glyph mask (utils.imaging.
    text_mask) — the same mask typography and cleanse consume, so color,
    weight, and erasure all agree on which pixels are strokes. the mask
    and crop are returned so background classification reuses them.
    """
    h, w = img.shape[:2]
    x0, y0 = max(0, bbox.x), max(0, bbox.y)
    x1 = min(w, bbox.x + bbox.width)
    y1 = min(h, bbox.y + bbox.height)
    if x1 - x0 < 3 or y1 - y0 < 3:
        return None, None, None, None, None
    crop = img[y0:y1, x0:x1]
    mask = _text_mask(img, bbox)
    if mask is None:
        flat = crop.reshape(-1, 3)
        return None, _hex(flat.mean(axis=0)), float(flat.std()), None, crop
    bg_pixels = crop[~mask]
    return (
        _hex(crop[mask].mean(axis=0)),
        _hex(bg_pixels.mean(axis=0)),
        float(bg_pixels.std()),
        mask,
        crop,
    )


# background classification thresholds (calibrated on the fixture set:
# flat-sign → flat, gradient-banner → smooth_gradient, textured-wall →
# textured)
BG_RESID_STD = 12.0   # max plane-fit residual std for flat/gradient
BG_GRAD_SPAN = 12.0   # min luminance change across the crop to call gradient


def _classify_background(crop, mask) -> Tuple[Optional[str], Optional[List[str]]]:
    """classify a region's background from its non-glyph pixels.

    method: least-squares planar shading fit over background luminance
    (the linear shading model). a low-residual fit is either "flat"
    (negligible luminance span) or "smooth_gradient" (the fitted plane's
    direction is reported as `linear <angle>°`); high residual means the
    background carries real structure → "textured".

    returns (texture_label, gradients) — gradients only for
    smooth_gradient, formatted for BgProfil.gradients.
    """
    try:
        import numpy as np
    except ImportError:
        return None, None
    if crop is None:
        return None, None
    bg = ~mask if mask is not None else np.ones(crop.shape[:2], dtype=bool)
    if bg.sum() < 40:
        return None, None
    lum = (
        0.299 * crop[..., 0].astype(np.float64)
        + 0.587 * crop[..., 1].astype(np.float64)
        + 0.114 * crop[..., 2].astype(np.float64)
    )
    h, w = lum.shape
    ys, xs = np.nonzero(bg)
    a_mat = np.column_stack([xs, ys, np.ones(len(xs))])
    try:
        coef, *_ = np.linalg.lstsq(a_mat, lum[ys, xs], rcond=None)
    except Exception:
        return None, None
    resid_std = float((lum[ys, xs] - a_mat @ coef).std())
    span = float(np.hypot(coef[0] * w, coef[1] * h))
    if resid_std < BG_RESID_STD:
        if span < BG_GRAD_SPAN:
            return "flat", None
        import math
        angle = math.degrees(math.atan2(float(coef[1]), float(coef[0])))
        return "smooth_gradient", [f"linear {angle:.0f}°"]
    return "textured", None


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
            text_hex, bg_hex, bg_std, glyph_mask, crop = _estimate_colors(
                img, inst.bounding_box
            )
        except Exception:
            continue
        sp, bp = inst.style_profile, inst.background_profile
        if sp.color is None and text_hex:
            sp.color = text_hex
        if bp.dominant_color is None and bg_hex:
            bp.dominant_color = bg_hex
        if bp.texture is None or bp.gradients is None:
            try:
                texture, gradients = _classify_background(crop, glyph_mask)
            except Exception:
                texture, gradients = None, None
            if bp.texture is None:
                if texture is not None:
                    bp.texture = texture
                elif bg_std is not None:  # classification unavailable
                    bp.texture = "flat" if bg_std < 24 else "textured"
            if bp.gradients is None and gradients:
                bp.gradients = gradients
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
