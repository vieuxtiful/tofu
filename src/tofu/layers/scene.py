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

import statistics
from abc import ABC, abstractmethod
from pathlib import Path
from threading import Lock
from typing import Any, Dict, List, Optional, Tuple

from tofu.core.types import (
    ImageLike, TextManifest, StyleProfil, BgProfil, SceneRegion, BBox, CharactText, GarnishProfile,
    InstText,
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
    def analyze(self, asset: ImageLike) -> List[SceneRegion]:
        """detect candidate text-bearing surfaces in the asset."""


class NullSceneBackend(SceneBackend):
    """no-engine fallback: valid empty results."""

    name = "null"

    def analyze(self, asset: ImageLike) -> List[SceneRegion]:
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
            # BT.601 luminance std, matching _classify_background's
            # convention -- the previous whole-array 3-channel scalar std
            # conflated CHROMA spread with brightness variation, so a
            # saturated but perfectly flat colored panel (high
            # inter-channel spread, zero spatial variation) could exceed
            # the threshold and be mislabeled bordered_region, changing
            # its downstream confidence floor.
            luma = roi @ np.array([0.299, 0.587, 0.114])
            std = float(luma.std())
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

        merged: List[Tuple[int, int, int, int, List[int]]] = []
        for members in groups.values():
            xs0 = min(raw[i][0] for i in members)
            ys0 = min(raw[i][1] for i in members)
            xs1 = max(raw[i][0] + raw[i][2] for i in members)
            ys1 = max(raw[i][1] + raw[i][3] for i in members)
            merged.append((xs0, ys0, xs1 - xs0, ys1 - ys0, members))

        min_area = self.min_area_frac * work.shape[0] * work.shape[1]
        regions: List[SceneRegion] = []
        for x, y, w, h, members in merged:
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
            # convex hull over the member components' corners: an angled
            # or stepped text run gets a polygon that hugs the actual
            # glyph footprint far tighter than its axis-aligned bbox --
            # this is what downstream polygon-aware containment (cicerone)
            # and polygon-clipped erasure (cleanse) consume. raw member
            # coords are already full-image space (see raw.append above).
            corner_pts = np.array([
                (cx, cy)
                for i in members
                for (cx, cy) in (
                    (raw[i][0], raw[i][1]),
                    (raw[i][0] + raw[i][2], raw[i][1]),
                    (raw[i][0] + raw[i][2], raw[i][1] + raw[i][3]),
                    (raw[i][0], raw[i][1] + raw[i][3]),
                )
            ], dtype=np.int32)
            try:
                hull = cv2.convexHull(corner_pts)
                polygon = [(int(p[0][0]), int(p[0][1])) for p in hull]
            except Exception:
                polygon = None
            regions.append(SceneRegion(
                bbox=BBox(x=x, y=y, width=w, height=h),
                semantic_label="text_cluster",
                confidence=0.55,
                background_color=_hex(mean),
                border_detected=False,
                polygon=polygon,
            ))
        return regions

    def analyze(self, asset: ImageLike) -> List[SceneRegion]:
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

        # region-interior texture: the same planar-shading classification
        # the per-instance enrichment applies (_classify_background), run
        # over each kept surface's own pixels. this is the SURFACE half
        # of the scene/cleanse agreement gate -- cleanse can then demand
        # that the instance-level verdict and the region-level verdict
        # agree before trusting a cheap analytic fill. best-effort: a
        # crop too small/degenerate simply leaves texture None.
        for region in kept:
            b = region.bbox
            x0, y0 = max(0, b.x), max(0, b.y)
            x1, y1 = min(w, b.x + b.width), min(h, b.y + b.height)
            if x1 - x0 < 8 or y1 - y0 < 8:
                continue
            texture, _grads = _classify_background(img[y0:y1, x0:x1], None)
            region.texture = texture
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
        self._load_lock = Lock()
        self.lifecycle_state = "missing" if not checkpoint_path else "loading"
        self.failure_reason: Optional[str] = None

    def status(self) -> Dict[str, Any]:
        """Cheap, serializable lifecycle state for diagnostics/UI."""
        return {
            "id": "sam", "state": self.lifecycle_state,
            "ready": self.lifecycle_state == "ready",
            "checkpoint_configured": bool(self.checkpoint_path and Path(self.checkpoint_path).is_file()),
            "reason": self.failure_reason,
        }

    def _ensure(self):
        if self._generator is not None:
            return
        with self._load_lock:
            if self._generator is not None:
                return
            self.lifecycle_state = "loading"
            try:
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
                self.lifecycle_state, self.failure_reason = "ready", None
            except Exception as exc:
                self._generator = None
                self.lifecycle_state = "failed"
                self.failure_reason = f"{type(exc).__name__}: {exc}"
                raise

    def analyze(self, asset: ImageLike) -> List[SceneRegion]:
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
    asset: ImageLike,
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

# Fraction of the crop width the MEDIAN horizontal line must cross before a
# run of lines counts as mortar courses.  Measured: brick fixture 0.99;
# QR code 0.31, circled pictogram 0.34, bordered text block 0.13.
MIN_COURSE_SPAN_RATIO = 0.6


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


def _describe_surface_material(crop, texture: Optional[str], semantic_label: Optional[str]) -> Optional[str]:
    """Return a conservative, user-facing material name.

    ``texture`` remains the compact routing signal consumed by Cleanse.  This
    helper deliberately does *not* turn every high-frequency crop into
    "brick": text glyphs, foliage and patterned posters all have edges too.
    Masonry needs both a repeated horizontal course and shorter vertical
    joints.  Ambiguous textured regions stay honestly labelled as a textured
    surface instead of exposing the detector's ``text_cluster`` implementation
    term in the editor.
    """
    if semantic_label in {"panel", "bordered_region"}:
        return "painted sign / panel"
    if texture == "flat":
        return "flat painted surface"
    if texture == "smooth_gradient":
        return "smooth shaded surface"
    if texture != "textured" or crop is None:
        return None

    # A bare ``text_cluster`` is the DETECTOR's edge-density heuristic, not a
    # finding about the world: it fires on a coat, foliage or pavement as
    # readily as on a sign. Measured on rue-des-martyrs, 14 of 16 proposals are
    # text_clusters blanketing the blurred street and the foreground
    # pedestrian, and every one of them was being handed the name "textured
    # surface" in the editor -- which is how a person came to be described as a
    # texture. Positive evidence still names a material (a panel border above,
    # a planar fit above, mortar courses below); absent any, the honest answer
    # for a text_cluster is no answer at all.
    #
    # This is the same standard the masonry test already holds itself to: a
    # generic label is preferable to a false one, and no label is preferable to
    # a generic one asserted about something that was never shown to be a
    # surface.
    unnamed = None if semantic_label == "text_cluster" else "textured surface"
    try:
        import cv2
        import numpy as np
        h, w = crop.shape[:2]
        if min(h, w) < 28:
            return unnamed
        gray = cv2.cvtColor(np.asarray(crop, dtype=np.uint8), cv2.COLOR_RGB2GRAY)
        edges = cv2.Canny(gray, 55, 140)
        lines = cv2.HoughLinesP(
            edges, 1, np.pi / 180, threshold=max(16, min(w, h) // 5),
            # Mortar joints are shorter than the horizontal courses, so the
            # line floor must admit both without treating one-pixel texture
            # noise as a material cue.
            minLineLength=max(10, min(w, h) // 7), maxLineGap=max(3, min(w, h) // 12),
        )
        if lines is None:
            return unnamed
        horizontal = vertical = 0
        horizontal_y: List[float] = []
        horizontal_span: List[int] = []
        for x0, y0, x1, y1 in np.asarray(lines).reshape(-1, 4):
            dx, dy = abs(int(x1) - int(x0)), abs(int(y1) - int(y0))
            if dx >= max(12, dy * 2):
                horizontal += 1
                horizontal_y.append((y0 + y1) / 2)
                horizontal_span.append(dx)
            elif dy >= max(8, dx * 1.4):
                vertical += 1
        # Brick courses repeat at several distinct y positions and include
        # perpendicular joints.  This is intentionally high precision: a
        # generic "textured surface" is preferable to a false brick label.
        distinct_courses = len({round(y / max(4, h * .08)) for y in horizontal_y})
        # ...and a course is CONTINUOUS across the surface -- that is what makes
        # it a course rather than a fragment.  Counting lines alone called every
        # high-contrast graphic masonry: measured on a medical-device label,
        # a QR code scored 29 horizontal lines over 13 courses and a circled
        # pictogram 3 over 3, both comfortably clearing the counts above.  Their
        # median course spans only 0.31 and 0.34 of the crop width, against 0.99
        # for the brick fixture's mortar lines, so continuity separates a wall
        # from a barcode where sheer line count cannot.
        spans_the_surface = (
            bool(horizontal_span)
            and statistics.median(horizontal_span) >= MIN_COURSE_SPAN_RATIO * w
        )
        if horizontal >= 3 and vertical >= 2 and distinct_courses >= 2 and spans_the_surface:
            return "brick / masonry"
    except Exception:
        pass
    return unnamed


def _material_evidence(
    crop,
    texture: Optional[str],
    semantic_label: Optional[str],
    material: Optional[str],
) -> Dict[str, Any]:
    taxonomy = {
        "painted sign / panel": "painted_panel",
        "flat painted surface": "painted_surface",
        "smooth shaded surface": "smooth_surface",
        "brick / masonry": "masonry",
        "textured surface": "textured_unknown",
    }
    material_class = taxonomy.get(material, "unknown")
    priors = {
        "painted_panel": ["fade", "abrasion", "bleed"],
        "painted_surface": ["fade", "abrasion"],
        "smooth_surface": ["fade", "blur"],
        "masonry": ["abrasion", "speckle", "occlusion"],
        "textured_unknown": ["abrasion", "speckle"],
        "unknown": [],
    }
    descriptors: Dict[str, Any] = {
        "texture": texture,
        "semantic_label": semantic_label,
    }
    try:
        import cv2
        import numpy as np

        array = np.asarray(crop, dtype=np.uint8)
        gray = cv2.cvtColor(array, cv2.COLOR_RGB2GRAY)
        descriptors.update({
            "edge_density": round(float((cv2.Canny(gray, 55, 140) > 0).mean()), 4),
            "luminance_std": round(float(gray.std()), 3),
            "color_dispersion": round(float(array.reshape(-1, 3).std(axis=0).mean()), 3),
            "sample_pixels": int(gray.size),
        })
    except Exception:
        descriptors["measurement_error"] = "descriptor_unavailable"
    return {
        "schema": 1,
        "revision": "scene-material-observation-v1",
        "material_class": material_class,
        "display_name": material,
        "descriptors": descriptors,
        "degradation_priors": [
            {"process": process, "status": "plausible_not_measured"}
            for process in priors[material_class]
        ],
        "provenance": {
            "method": "classical_surface_descriptors",
            "source": "scene_region_crop",
            "glyph_exclusion": False,
        },
        "decision_eligible": False,
        "limitations": [
            "taxonomy is heuristic and uncalibrated",
            "surface crop may contain glyph pixels",
            "degradation priors are hypotheses, not posterior probabilities",
        ],
    }


def _analyze_garnish_profile(crop, glyph_mask=None) -> GarnishProfile:
    """Estimate conservative surface-compatible text wear from local pixels.

    This is evidence for a tunable post-Scribe effect, not a generative style
    transfer.  Low-confidence/flat surfaces therefore yield near-identity
    profiles, which keeps Garnish harmless until Scene observes texture.
    """
    try:
        import cv2
        import numpy as np
        if crop is None or crop.size == 0:
            return GarnishProfile()
        gray = cv2.cvtColor(np.asarray(crop, dtype=np.uint8), cv2.COLOR_RGB2GRAY).astype(np.float32)
        low = cv2.GaussianBlur(gray, (0, 0), 1.2)
        grain = float(np.std(gray - low) / 48.0)
        lap = float(np.var(cv2.Laplacian(gray, cv2.CV_32F)))
        # A truly flat surface has no evidence for weathering.  Gate all
        # automatic treatment behind residual texture before allowing the
        # stronger (but still capped) visible recommendations below.
        texture_signal = max(0.0, min(1.0, (grain - 0.04) / 0.18))
        if texture_signal <= 0.0:
            return GarnishProfile()
        edge_blur = texture_signal * max(0.0, min(3.0, (110.0 - min(110.0, lap)) / 60.0))
        if glyph_mask is not None:
            edge_blur *= 0.75
        moments = cv2.moments(cv2.Canny(gray.astype(np.uint8), 45, 140))
        angle = 0.0
        if abs(moments.get("mu20", 0.0) - moments.get("mu02", 0.0)) > 1e-6:
            import math
            angle = math.degrees(0.5 * math.atan2(2 * moments.get("mu11", 0.0), moments.get("mu20", 0.0) - moments.get("mu02", 0.0)))
        grain = max(0.0, min(0.5, grain))
        confidence = max(0.0, min(0.8, 0.18 + grain * 1.3 + edge_blur * 0.22))
        smoothing_strength = max(0.2, min(1.0, 0.2 + texture_signal * 0.8))
        return GarnishProfile(edge_blur_px=edge_blur, edge_smoothing=True,
                              edge_smoothing_strength=smoothing_strength, grain_strength=grain,
                              smudge_strength=min(0.35, edge_blur * 0.15),
                              smudge_angle_deg=angle, source_confidence=confidence)
    except Exception:
        return GarnishProfile()


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


def analyze(asset: ImageLike, text_manifest: TextManifest) -> TextManifest:
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

    # Surface categories drive routing, while material names drive the UI.
    # Populate the latter once per scene region, not once per overlapping text
    # bbox, so a brick wall is presented as masonry even when the detector's
    # internal grouping happens to call that area ``text_cluster``.
    if img is not None:
        h, w = img.shape[:2]
        for region in text_manifest.scene_regions:
            if (
                region.material is not None
                and region.material_evidence is not None
                and region.garnish_profile is not None
            ):
                continue
            b = region.bbox
            x0, y0 = max(0, b.x), max(0, b.y)
            x1, y1 = min(w, b.x + b.width), min(h, b.y + b.height)
            crop = img[y0:y1, x0:x1]
            texture = region.texture
            if texture is None and crop.size:
                texture, _ = _classify_background(crop, None)
                region.texture = texture
            if region.material is None:
                region.material = _describe_surface_material(crop, texture, region.semantic_label)
            if region.material_evidence is None:
                region.material_evidence = _material_evidence(
                    crop, texture, region.semantic_label, region.material
                )
            if region.garnish_profile is None:
                region.garnish_profile = _analyze_garnish_profile(crop)

    for inst in text_manifest.instances:
        if inst.style_profile is None:
            inst.style_profile = StyleProfil()
        if inst.background_profile is None:
            inst.background_profile = BgProfil()
        if img is None or inst.bounding_box is None:
            continue
        sp, bp = inst.style_profile, inst.background_profile
        ch = inst.characteristics
        if (sp.color is not None and bp.dominant_color is not None
                and bp.texture is not None and bp.gradients is not None
                and bp.material is not None and ch is not None
                and ch.font_style is not None and ch.size is not None
                and ch.positioning is not None
                and inst.material_evidence is not None):
            continue
        try:
            text_hex, bg_hex, bg_std, glyph_mask, crop = _estimate_colors(
                img, inst.bounding_box
            )
        except Exception:
            continue
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
        region = _containing_region(text_manifest.scene_regions, inst.bounding_box)
        if bp.semantic_label is None:
            if region is not None:
                bp.semantic_label = region.semantic_label
                if bp.material is None:
                    bp.material = region.material
        if region is not None and inst.material_evidence is None:
            inst.material_evidence = region.material_evidence

        if bp.material is None:
            bp.material = _describe_surface_material(crop, bp.texture, bp.semantic_label)

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
            if sp.font_weight is None and typo.weight in ("heavy", "bold", "light"):
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
    # Basil runs after the source OCR has been corrected and after scene has
    # established the physical panel/sign context.  It only registers
    # semantic reading units; it never changes an rN, box, or translation.
    try:
        from tofu.layers.basil import unify_manifest
        unify_manifest(text_manifest)
    except Exception:
        # Semantic substitution is an editor enhancement.  A missing optional
        # language model must never degrade Scene/Cleanse/Scribe execution.
        pass
    return text_manifest


## --- substrate: the surface a text region sits ON, with the text removed ---
##
## Restoration is only as good as its sample of the material underneath, and
## the sample is where restoration quietly goes wrong. `cleanse` currently
## estimates from a fixed 14px ring around the glyph mask, and its own source
## records the failure mode: "a ring touching an adjacent sign can be a much
## worse estimate", mitigated by anchoring to Scene's dominant colour. A ring
## has a second contamination it does not mitigate -- on dense signage it
## samples NEIGHBOURING GLYPHS, so the "background" estimate is partly other
## people's text.
##
## Scene already holds both halves of the fix. `SceneRegion.polygon` is the
## surface outline, and every `InstText.segmentation_mask` is a glyph outline.
## Surface minus ALL glyphs is the substrate, and it is both larger and
## cleaner than any ring.
##
## This is also the input the material/degradation work needs: inferring how
## a surface has weathered requires looking at the surface, not at the letters
## on top of it. Kept OBSERVATIONAL here -- it measures and reports, and
## changes no repair. Wiring it into `cleanse` is a separate, measured step,
## because swapping the estimator under a corpus that was tuned around the
## ring is exactly the kind of change that has to be scored before it ships.

## How far each glyph mask is grown before exclusion. Anti-aliased edges
## carry glyph colour for a couple of pixels beyond the mask, and sampling
## those as substrate would drag the estimate toward the ink.
## `reasoned` -- it matches cleanse.DILATE_ITER's ~3px growth, chosen there
## for the same anti-aliasing reason. Not swept.
SUBSTRATE_EXCLUDE_PX = 3

## Below this many surviving pixels the sample is not worth reporting: a
## handful of pixels gives a median with no stability. `reasoned` against
## cleanse.MIN_RING_PIXELS = 20, which is the same judgement for the ring.
SUBSTRATE_MIN_PIXELS = 20


def substrate(
    asset: ImageLike,
    region: SceneRegion,
    instances: Optional[List[InstText]] = None,
) -> Optional[Dict[str, Any]]:
    """Sample the surface inside `region`, excluding every glyph on it.

    Returns None when the sample cannot be taken at all -- a missing image,
    no surviving pixels -- rather than returning a fabricated estimate, and
    reports `trustworthy: False` when it is too small to rely on. A
    restoration that cannot tell "no sample" from "a bad sample" is how
    hallucinated material gets treated as observed.
    """
    try:
        import cv2
        import numpy as np
    except ImportError:
        return None

    img = _load_rgb(asset)
    if img is None:
        return None
    height, width = img.shape[:2]

    ## The surface itself: its polygon when scene traced one, else its box.
    surface = np.zeros((height, width), dtype=np.uint8)
    if region.polygon:
        points = np.array([[int(x), int(y)] for x, y in region.polygon], dtype=np.int32)
        cv2.fillPoly(surface, [points], 1)
    else:
        x0 = max(0, int(region.bbox.x)); y0 = max(0, int(region.bbox.y))
        x1 = min(width, x0 + int(region.bbox.width))
        y1 = min(height, y0 + int(region.bbox.height))
        surface[y0:y1, x0:x1] = 1
    if not surface.any():
        return None
    surface_pixels = int(surface.sum())

    ## Every glyph in the IMAGE, not only those attributed to this surface.
    ## A neighbouring sign's text overlapping this polygon is contamination
    ## whoever it belongs to.
    glyphs = np.zeros((height, width), dtype=np.uint8)
    excluded = 0
    for inst in instances or []:
        mask = getattr(inst, "segmentation_mask", None)
        if mask is not None and mask.polygon:
            points = np.array([[int(x), int(y)] for x, y in mask.polygon], dtype=np.int32)
            cv2.fillPoly(glyphs, [points], 1)
            for hole in (mask.holes or []):
                hole_pts = np.array([[int(x), int(y)] for x, y in hole], dtype=np.int32)
                cv2.fillPoly(glyphs, [hole_pts], 0)
        else:
            ## No mask: fall back to the bounding box, which over-excludes.
            ## Over-excluding shrinks the sample; under-excluding poisons it.
            box = getattr(inst, "bounding_box", None)
            if box is None:
                continue
            x0 = max(0, int(box.x)); y0 = max(0, int(box.y))
            glyphs[y0:min(height, y0 + int(box.height)),
                   x0:min(width, x0 + int(box.width))] = 1
        excluded += 1

    if glyphs.any() and SUBSTRATE_EXCLUDE_PX > 0:
        kernel = np.ones((SUBSTRATE_EXCLUDE_PX * 2 + 1,) * 2, np.uint8)
        glyphs = cv2.dilate(glyphs, kernel, iterations=1)

    sample = (surface > 0) & (glyphs == 0)
    pixels = int(sample.sum())
    if pixels == 0:
        return None

    values = img[sample].reshape(-1, 3).astype(np.float64)
    median = np.median(values, axis=0)
    return {
        "schema": 1,
        "pixels": pixels,
        "surface_pixels": surface_pixels,
        ## What fraction of the surface survived exclusion. A low figure on a
        ## dense sign says the substrate is barely visible, which is a fact
        ## about the asset the caller should be able to see.
        "coverage": round(pixels / surface_pixels, 3) if surface_pixels else 0.0,
        "glyph_regions_excluded": excluded,
        "median_color": "#%02x%02x%02x" % tuple(int(round(c)) for c in median),
        ## Per-channel spread, the cheapest available roughness proxy: a flat
        ## painted panel and weathered masonry differ here before any
        ## material model is involved.
        "std": [round(float(v), 2) for v in values.std(axis=0)],
        "trustworthy": pixels >= SUBSTRATE_MIN_PIXELS,
    }


def substrate_material_evidence(
    asset: ImageLike,
    region: SceneRegion,
    instances: Optional[List[InstText]] = None,
    *,
    include_preview: bool = False,
) -> Optional[Dict[str, Any]]:
    try:
        import cv2
        import numpy as np
    except ImportError:
        return None
    summary = substrate(asset, region, instances)
    img = _load_rgb(asset)
    if summary is None or img is None:
        return None
    height, width = img.shape[:2]
    b = region.bbox
    x0, y0 = max(0, int(b.x)), max(0, int(b.y))
    x1 = min(width, x0 + int(b.width))
    y1 = min(height, y0 + int(b.height))
    if x1 <= x0 or y1 <= y0:
        return None
    crop = img[y0:y1, x0:x1].copy()
    keep = np.ones(crop.shape[:2], dtype=np.uint8)
    if region.polygon:
        keep.fill(0)
        points = np.array(
            [[int(x) - x0, int(y) - y0] for x, y in region.polygon], dtype=np.int32
        )
        cv2.fillPoly(keep, [points], 1)
    excluded = np.zeros(crop.shape[:2], dtype=np.uint8)
    fallback_boxes = 0
    for inst in instances or []:
        mask = getattr(inst, "segmentation_mask", None)
        if mask is not None and mask.polygon:
            points = np.array(
                [[int(x) - x0, int(y) - y0] for x, y in mask.polygon], dtype=np.int32
            )
            cv2.fillPoly(excluded, [points], 1)
            for hole in mask.holes or []:
                hole_points = np.array(
                    [[int(x) - x0, int(y) - y0] for x, y in hole], dtype=np.int32
                )
                cv2.fillPoly(excluded, [hole_points], 0)
        else:
            box = getattr(inst, "bounding_box", None)
            if box is None:
                continue
            bx0, by0 = int(box.x) - x0, int(box.y) - y0
            bx1, by1 = bx0 + int(box.width), by0 + int(box.height)
            cv2.rectangle(excluded, (bx0, by0), (bx1, by1), 1, thickness=-1)
            fallback_boxes += 1
    if excluded.any() and SUBSTRATE_EXCLUDE_PX > 0:
        kernel = np.ones((SUBSTRATE_EXCLUDE_PX * 2 + 1,) * 2, np.uint8)
        excluded = cv2.dilate(excluded, kernel, iterations=1)
    sample = (keep > 0) & (excluded == 0)
    if not sample.any():
        return None
    median = np.median(crop[sample].reshape(-1, 3), axis=0).astype(np.uint8)
    cleaned = crop.copy()
    cleaned[~sample] = median
    texture, _ = _classify_background(cleaned, None)
    material = _describe_surface_material(cleaned, texture, region.semantic_label)
    evidence = _material_evidence(cleaned, texture, region.semantic_label, material)
    evidence["provenance"].update({
        "source": "scene_region_substrate",
        "glyph_exclusion": True,
        "fallback_boxes": fallback_boxes,
    })
    evidence["substrate"] = summary
    evidence["limitations"] = [
        "taxonomy is heuristic and uncalibrated",
        "excluded pixels are median-filled before spatial descriptor analysis",
        "degradation priors are hypotheses, not posterior probabilities",
    ]
    if include_preview:
        evidence["analysis_preview"] = cleaned
    return evidence
