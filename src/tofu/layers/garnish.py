"""Garnish — post-Scribe natural scene integration.

Only the newly rendered text-difference layer is treated.  The cleansed base
is never modified, so neural/manual repair pixels remain reversible and QA can
assert outside-mask preservation.
"""
from __future__ import annotations

import hashlib
import logging
import math
from typing import Any, Optional

from tofu.core.types import GarnishProfile, TextManifest

logger = logging.getLogger(__name__)


def _profile(manifest: TextManifest, inst) -> Optional[GarnishProfile]:
    if inst.garnish_enabled is False:
        return None
    if inst.garnish_override is not None:
        return inst.garnish_override
    cx = inst.bounding_box.x + inst.bounding_box.width / 2
    cy = inst.bounding_box.y + inst.bounding_box.height / 2
    hits = [region for region in manifest.scene_regions or []
            if region.garnish_profile is not None and region.bbox.x <= cx <= region.bbox.x + region.bbox.width
            and region.bbox.y <= cy <= region.bbox.y + region.bbox.height]
    return min(hits, key=lambda region: region.bbox.width * region.bbox.height).garnish_profile if hits else None


def _active(profile: Optional[GarnishProfile]) -> bool:
    return bool(profile and (profile.edge_blur_px or profile.edge_smoothing or profile.erosion_px or profile.dilation_px
                             or profile.grain_strength or profile.smudge_strength
                             or abs(profile.gamma_shift - 1.0) > 1e-3))


def _motion_blur_alpha(cv2, np, alpha, strength: float, angle_deg: float):
    """Blur an alpha layer along a real rotated motion path.

    An anisotropic Gaussian can only blur horizontally/vertically; it cannot
    express the diagonal direction returned by Scene.  A normalized line
    kernel keeps this deterministic while respecting the editor's angle.
    """
    length = max(3, int(round(strength * 14)))
    if length <= 3:
        return alpha
    size = length * 2 + 1
    kernel = np.zeros((size, size), dtype=np.float32)
    center = size // 2
    radius = max(1, length // 2)
    radians = math.radians(angle_deg)
    dx, dy = int(round(math.cos(radians) * radius)), int(round(math.sin(radians) * radius))
    cv2.line(kernel, (center - dx, center - dy), (center + dx, center + dy), 1.0, 1)
    total = float(kernel.sum())
    if total <= 0:
        return alpha
    return cv2.filter2D(alpha, -1, kernel / total, borderType=cv2.BORDER_REPLICATE)


def _region_mask(cv2, np, polygon, x0: int, y0: int, width: int, height: int):
    """Rasterize one image-space editor polygon into an instance crop."""
    if not polygon or len(polygon) < 3:
        return None
    points = np.asarray([(int(x - x0), int(y - y0)) for x, y in polygon], dtype=np.int32)
    allowed = np.zeros((height, width), dtype=np.uint8)
    cv2.fillPoly(allowed, [points], 255)
    return allowed


def _smooth_coverage(cv2, np, alpha, strength: float = 0.5):
    """Round raster stair-steps with controlled edge feathering.

    A pure median only rounds alpha values WITHIN existing non-zero
    pixels — it cannot repair the perceptual jaggedness AT the coverage
    boundary, because that boundary itself stays hard.  A Gaussian
    feathers beyond the original coverage edge, which is what an editor
    asking for "softer edges" actually wants; applied alone, though, it
    reads as out-of-focus text.

    The blend between the Gaussian-feathered alpha and the original
    alpha controls how much feather escapes: ``strength * 0.4`` is the
    Gaussian weight, the rest is original.  At strength=0.5, 20% of
    the Gaussian feather leaks past the edge (conservative but visibly
    smoother); at strength=1.0, 40% leaks (noticeable softening).  No
    hard support clamp is applied — that would eliminate the only
    useful effect of the Gaussian component.
    """
    if alpha.size == 0:
        return alpha
    # Preserve Scribe's existing interior alpha exactly.  A distance ramp is
    # generated only outside the glyph, so feathering cannot soften stems.
    width_out = max(1.0, max(0.0, min(1.0, float(strength))) * 5.0)
    binary = (alpha > 0).astype(np.uint8)
    exterior_distance = cv2.distanceTransform(1 - binary, cv2.DIST_L2, 3)
    ramp = np.clip(255.0 * (1.0 - exterior_distance / width_out), 0, 255).astype(np.uint8)
    return np.where(binary > 0, alpha, ramp).astype(np.uint8)


def apply(scribed_asset: Any, text_manifest: TextManifest, base_asset: Any = None, font_registry: Optional[Any] = None) -> Any:
    """Apply deterministic text-edge treatment; identity without a base/profile."""
    if base_asset is None:
        return scribed_asset
    try:
        import numpy as np
        from PIL import Image, ImageFilter
        import cv2
        scribed = scribed_asset.convert("RGBA") if isinstance(scribed_asset, Image.Image) else Image.open(scribed_asset).convert("RGBA")
        base = base_asset.convert("RGBA") if isinstance(base_asset, Image.Image) else Image.open(base_asset).convert("RGBA")
        if scribed.size != base.size:
            return scribed_asset
        result = scribed.copy()
        source = np.asarray(scribed.convert("RGB"), dtype=np.int16)
        # Scribe's exact alpha coverage is the preferred mask.  ``clean`` is
        # retained exclusively for legacy image inputs that do not carry that
        # transient sidecar.
        clean = np.asarray(base.convert("RGB"), dtype=np.int16)
        text_masks = getattr(scribed_asset, "text_masks", None)
        for inst in text_manifest.instances:
            if not inst.target_text or inst.dnt or inst.excluded:
                continue
            # Treatment follows the rendered text when an editor applies an
            # offset/warp; source-surface matching intentionally remains
            # captured.  Prefer Scribe's real alpha bounds over the capture
            # box so an adjacent region can never crop this treatment window.
            b = inst.adjusted_bbox or inst.bounding_box
            coverage_full = text_masks.get(inst.id) if isinstance(text_masks, dict) else None
            if coverage_full is not None:
                coverage_arr = np.asarray(coverage_full, dtype=np.uint8)
                if coverage_arr.shape[:2] == (scribed.height, scribed.width):
                    ys, xs = np.nonzero(coverage_arr)
                    if xs.size:
                        # Room for blur/smudge/SDF feathering; the final alpha
                        # mask still limits every changed pixel precisely.
                        pad = 16
                        x0, y0 = max(0, int(xs.min()) - pad), max(0, int(ys.min()) - pad)
                        x1, y1 = min(scribed.width, int(xs.max()) + 1 + pad), min(scribed.height, int(ys.max()) + 1 + pad)
                    else:
                        x0 = y0 = x1 = y1 = 0
                else:
                    coverage_full = None
            if coverage_full is None:
                x0, y0 = max(0, b.x), max(0, b.y)
                x1, y1 = min(scribed.width, b.x + b.width), min(scribed.height, b.y + b.height)
            if x1 <= x0 or y1 <= y0: continue
            inherited = _profile(text_manifest, inst)
            # Whole-selection treatment is the base for every region.  In
            # per_region mode the inherited profile applies to the entire
            # instance first; sub-regions with their own profiles composite
            # on top within their polygon areas so sibling glyphs outside
            # remain untouched.
            if inst.garnish_scope == "per_region":
                targets: list = [None] + [r for r in inst.garnish_regions if r.enabled is not False and r.profile is not None]
            else:
                targets = [None]
            for region in targets:
                if region is not None and region.enabled is False:
                    continue
                profile = (region.profile if region and region.profile else inherited)
                if not _active(profile):
                    continue
                coverage = text_masks.get(inst.id) if isinstance(text_masks, dict) else None
                if coverage is not None:
                    mask_full = np.asarray(coverage, dtype=np.uint8)
                    if mask_full.shape[:2] != (scribed.height, scribed.width):
                        # A malformed sidecar must not silently offset a
                        # treatment.  The established diff path is safer.
                        coverage = None
                    else:
                        # Keep anti-aliased coverage rather than collapsing it
                        # to a binary threshold.  This is what lets edge blur
                        # and grain feather naturally at actual glyph edges.
                        mask = mask_full[y0:y1, x0:x1].copy()
                if coverage is None:
                    # Compatibility fallback for external/legacy Scribe
                    # callers.  Smart Fill patches are already in ``base``.
                    diff = np.max(np.abs(source[y0:y1, x0:x1] - clean[y0:y1, x0:x1]), axis=2).astype(np.uint8)
                    _, mask = cv2.threshold(diff, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                    if not mask.any() and int(diff.max()) > 3:
                        mask = np.where(diff > 5, 255, 0).astype(np.uint8)
                allowed = _region_mask(cv2, np, region.polygon, x0, y0, x1 - x0, y1 - y0) if region else None
                if allowed is not None:
                    mask = cv2.bitwise_and(mask, allowed)
                if not mask.any():
                    continue
                confidence = max(0.3, min(1.0, profile.source_confidence))
                edge_blur, grain_strength = profile.edge_blur_px * confidence, profile.grain_strength * confidence
                smudge_strength = profile.smudge_strength * confidence
                if profile.erosion_px > 0:
                    k = max(1, int(round(profile.erosion_px)) * 2 + 1); mask = cv2.erode(mask, np.ones((k, k), np.uint8))
                if profile.dilation_px > 0:
                    k = max(1, int(round(profile.dilation_px)) * 2 + 1); mask = cv2.dilate(mask, np.ones((k, k), np.uint8))
                if profile.edge_smoothing:
                    strength = getattr(profile, "edge_smoothing_strength", 0.5)
                    mask = _smooth_coverage(cv2, np, mask, strength)
                alpha = Image.fromarray(mask, "L")
                if edge_blur > 0: alpha = alpha.filter(ImageFilter.GaussianBlur(radius=min(10, edge_blur)))
                alpha_np = np.asarray(alpha, dtype=np.uint8)
                if smudge_strength > 0:
                    alpha_np = _motion_blur_alpha(cv2, np, alpha_np, smudge_strength, profile.smudge_angle_deg)
                if allowed is not None:
                    # Permit a small natural feather but never let a selected
                    # word's treatment leak across the rest of the instance.
                    feather_px = (max(0.0, min(1.0, float(getattr(profile, "edge_smoothing_strength", 0.5))) * 5.0)
                                  if profile.edge_smoothing else 0.0)
                    margin = max(1, int(math.ceil(edge_blur + feather_px + smudge_strength * 4)))
                    alpha_np = cv2.bitwise_and(alpha_np, cv2.dilate(allowed, np.ones((margin * 2 + 1, margin * 2 + 1), np.uint8)))
                rgb = source[y0:y1, x0:x1].astype(np.float32)
                if abs(profile.gamma_shift - 1.0) > 1e-3:
                    rgb = 255.0 * np.power(np.clip(rgb / 255.0, 0, 1), 1.0 / max(.5, min(2.0, profile.gamma_shift)))
                if grain_strength > 0:
                    seed_key = f"{inst.id}:{region.id if region else 'whole'}"
                    seed = int(hashlib.sha256(seed_key.encode()).hexdigest()[:8], 16)
                    rng = np.random.default_rng(seed)
                    noise = rng.normal(0, grain_strength * 22, rgb.shape)
                    outer = cv2.dilate(mask, np.ones((5, 5), np.uint8)); inner = cv2.erode(mask, np.ones((3, 3), np.uint8))
                    edge_weight = np.clip(cv2.subtract(outer, inner).astype(np.float32) / 255.0, 0, 1) * 0.7 + 0.3
                    rgb += noise * edge_weight[:, :, np.newaxis]
                feather_px = (max(0.0, min(1.0, float(getattr(profile, "edge_smoothing_strength", 0.5))) * 5.0)
                              if profile.edge_smoothing else 0.0)
                if edge_blur > 0 or feather_px > 0:
                    # Engrain: in the soft edge band (partial alpha after blur),
                    # blend text RGB toward the underlying surface so edges look
                    # embedded rather than casting a text-colored halo.  The
                    # parabola 4*a*(1-a) peaks at 50% alpha and vanishes at
                    # full opacity / full transparency, targeting only the
                    # feathered boundary where jagged pixels need dulling.
                    surface_rgb = np.asarray(
                        (result if region else base).crop((x0, y0, x1, y1)).convert("RGB"),
                        dtype=np.float32,
                    )
                    a = alpha_np.astype(np.float32) / 255.0
                    engrain_scale = max(0.2, min(1.0, 1.0 / max(1.0, feather_px))) if feather_px > 0 else 1.0
                    engrain = 4.0 * a * (1.0 - a) * engrain_scale
                    rgb = rgb * (1 - engrain[:, :, np.newaxis]) + surface_rgb * engrain[:, :, np.newaxis]
                layer = Image.fromarray(np.dstack([np.clip(rgb, 0, 255).astype(np.uint8), alpha_np]), "RGBA")
                # Every treatment is an alpha-masked edit over the accumulated
                # localized result.  Reconstructing a whole-instance crop from
                # ``base`` erases an earlier region wherever two capture boxes
                # overlap (for example, the V in VECCHI under MURI's box), even
                # when their glyph masks never overlap.  Capture boxes are
                # processing anchors, not replacement rectangles.
                target_crop = result.crop((x0, y0, x1, y1))
                target_crop.alpha_composite(layer)
                result.alpha_composite(target_crop, (x0, y0))
        return result.convert("RGB")
    except Exception:
        logger.exception("garnish.apply failed")
        return scribed_asset
