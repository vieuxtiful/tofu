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

import hashlib
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from tofu.core.types import InpaintAssessmentPolicy, TextManifest
from tofu.utils.imaging import text_mask as _text_mask
from tofu.layers import inpaint_providers

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
GROUP_GAP_PX = 12       # nearby glyph groups share one source-grounded repair
MIN_MASK_PIXELS = 6     # a tiny Otsu component is not safe evidence by itself


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


def _clip_to_polygon(np, cv2, mask, polygon, h: int, w: int):
    """Restrict an erasure mask to a user/detector supplied polygon."""
    if not polygon or len(polygon) < 3:
        return mask
    try:
        points = np.asarray(polygon, dtype=np.int32).reshape((-1, 1, 2))
        clip = np.zeros((h, w), dtype=np.uint8)
        cv2.fillPoly(clip, [points], 255)
        return mask & (clip > 0)
    except Exception:
        return mask  # malformed user geometry must fail open


def _select_region_mask(np, cv2, img_array, bbox, h: int, w: int,
                        instance_polygon=None, surface_polygon=None):
    """Choose a conservative text-erasure mask and preserve its evidence.

    The Scene polygon is deliberately *not* a hard erasure clip.  It is an
    excellent source of context for a repair model, but a tight or imperfect
    surface contour must not trim antialiasing, an outline, a shadow, or a
    Japanese diacritic that happens to cross the inferred surface edge.  A
    detector/user polygon can constrain the mask only when doing so keeps the
    overwhelming majority of the observed glyph evidence.
    """
    segmented = _region_mask(np, cv2, img_array, bbox, h, w)
    source = "stroke_mask"
    fallback = False
    if segmented is None or int(segmented.sum()) < MIN_MASK_PIXELS:
        selected = _bbox_fallback_mask(np, bbox, h, w)
        source = "bbox_fallback"
        fallback = True
    else:
        selected = segmented

    original_pixels = int(selected.sum())
    polygon_applied = False
    polygon_preserved = 1.0
    if instance_polygon and len(instance_polygon) >= 3 and original_pixels:
        clipped = _clip_to_polygon(np, cv2, selected, instance_polygon, h, w)
        retained = int(clipped.sum()) / max(1, original_pixels)
        # A segmentation contour is valuable, but never accept one which
        # removes a material part of the independently observed ink mask.
        if retained >= 0.78 and int(clipped.sum()) >= MIN_MASK_PIXELS:
            selected = clipped
            polygon_applied = True
            polygon_preserved = round(retained, 4)

    # Surface geometry is evidence about the repair context, not permission
    # to erase less text.  Record whether it contains the selected mask so a
    # low-confidence context route can fail safely into review.
    surface_contains = None
    if surface_polygon and len(surface_polygon) >= 3 and selected.any():
        contained = _clip_to_polygon(np, cv2, selected, surface_polygon, h, w)
        surface_contains = round(int(contained.sum()) / max(1, int(selected.sum())), 4)

    density = int(selected.sum()) / max(1, bbox.width * bbox.height)
    confidence = .96
    if fallback:
        confidence = .48
    elif density < .015 or density > .90:
        confidence = .68
    if surface_contains is not None and surface_contains < .82:
        confidence = min(confidence, .72)
    evidence = {
        "source": source,
        "pixels": int(selected.sum()),
        "bbox_density": round(density, 5),
        "confidence": round(confidence, 3),
        "fallback": fallback,
        "instance_polygon_applied": polygon_applied,
        "instance_polygon_retained": polygon_preserved,
        "surface_mask_containment": surface_contains,
        "surface_polygon_used_for_context": bool(surface_polygon and len(surface_polygon) >= 3),
    }
    return selected, evidence


def _containing_surface(text_manifest, bbox):
    """Return the scene surface that most covers an instance bbox."""
    best, best_score = None, 0.0
    for region in getattr(text_manifest, "scene_regions", []) or []:
        rb = region.bbox
        ix, iy = max(bbox.x, rb.x), max(bbox.y, rb.y)
        ax, ay = min(bbox.x + bbox.width, rb.x + rb.width), min(bbox.y + bbox.height, rb.y + rb.height)
        score = max(0, ax - ix) * max(0, ay - iy) / max(1, bbox.width * bbox.height)
        if score > best_score:
            best, best_score = region, score
    return best if best_score >= 0.5 else None


def _choose_strategy(inst, surface, image=None, mask=None, mask_evidence=None) -> str:
    """Apply provider routing and preserve its evidence on the region."""
    profile = getattr(inst, "background_profile", None)
    texture = profile.texture if profile else None
    surface_texture = getattr(surface, "texture", None) if surface else None
    surface_label = getattr(surface, "semantic_label", None) if surface else None
    if profile:
        profile.surface_texture = surface_texture

    route = inpaint_providers.route(inst, surface, image, mask)
    strategy = route.strategy
    if profile:
        profile.cleanse_strategy = strategy
    mask_evidence = mask_evidence or {}
    mask_confidence = float(mask_evidence.get("confidence", 0.0))
    # Strong Scene agreement cannot compensate for an uncertain text mask:
    # deterministic reconstruction may run, but its result is review-bound.
    auto_accept = bool(route.auto_accept and mask_confidence >= .85)
    review_required = bool(route.review_required or not auto_accept)
    reason = route.reason
    if route.auto_accept and not auto_accept:
        reason += "; mask evidence is below the auto-accept threshold"
    inst.repair_provenance = {
        "requested_provider": route.provider,
        "strategy": route.strategy,
        "confidence": route.confidence,
        "auto_accepted": auto_accept,
        "review_required": review_required,
        "reason": reason,
        "mask": mask_evidence,
        "candidates": [],
    }
    return strategy


def _masks_are_near(np, cv2, first, second) -> bool:
    """Whether two masks should be repaired in a single local context."""
    if not first.any() or not second.any():
        return False
    kernel = np.ones((3, 3), np.uint8)
    expanded = cv2.dilate(first.astype(np.uint8), kernel,
                           iterations=max(1, GROUP_GAP_PX)).astype(bool)
    return bool((expanded & second).any())


def _neural_groups(np, cv2, plans: List[Dict[str, Any]]) -> List[List[Dict[str, Any]]]:
    """Group only compatible neural repairs on the same Scene surface.

    Every group later receives the original image, never a previous model
    result.  This removes order-dependence for adjacent subtitles/sign text
    without merging unrelated surfaces merely because their boxes are close.
    """
    groups: List[List[Dict[str, Any]]] = []
    for plan in plans:
        provider = plan["provider"]
        surface = plan["surface"]
        for group in groups:
            anchor = group[0]
            if anchor["provider"] != provider or anchor["surface"] is not surface:
                continue
            if any(_masks_are_near(np, cv2, plan["mask"], member["mask"]) for member in group):
                group.append(plan)
                break
        else:
            groups.append([plan])
    return groups


def _record_candidate(inst, provider: str, accepted: bool, evidence: Dict[str, Any],
                      decision: str, group_ids: List[str]) -> None:
    if not inst.repair_provenance:
        return
    candidates = inst.repair_provenance.setdefault("candidates", [])
    candidates.append({
        "provider": provider,
        "accepted": bool(accepted),
        "decision": decision,
        "group_ids": group_ids,
        "evidence": evidence,
    })


def _repair_group_key(np, group: List[Dict[str, Any]], group_mask) -> str:
    """Stable identity for one source/mask/model repair opportunity.

    This intentionally excludes model output and acceptance state.  A local
    candidate can therefore be reviewed and applied as a patch while the same
    Cleanse plan remains reproducible on future preview/final compositions.
    """
    members = ",".join(sorted(plan["inst"].id for plan in group))
    packed_mask = np.packbits(group_mask.astype(np.uint8)).tobytes()
    digest = hashlib.sha256(members.encode("utf-8") + packed_mask).hexdigest()[:20]
    return f"repair-{digest}"


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


def _fill_flat(cv2, np, working, full_mask, h: int, w: int, dominant_color: Optional[str] = None) -> None:
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
    # A ring touching an adjacent sign can be a much worse estimate than
    # Scene's robust, instance-local dominant color.  Treat a large
    # disagreement as contamination and anchor the flat reconstruction.
    if dominant_color and isinstance(dominant_color, str) and len(dominant_color) == 7 and dominant_color.startswith("#"):
        try:
            anchor = np.array([int(dominant_color[i:i + 2], 16) for i in (1, 3, 5)], dtype=np.float64)
            if np.linalg.norm(color - anchor) > 90:
                color = anchor
        except ValueError:
            pass
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


def _perspective_repair_context(cv2, np, image, mask, inst):
    """Return rectified provider inputs plus an original-space restorer.

    The existing normalized StyleProfil quad is authoritative. Degenerate,
    identity, or implausibly large transforms fail open to the normal path.
    """
    style = getattr(inst, "style_profile", None)
    transform = getattr(style, "transform", None) if style else None
    quad = transform.get("quad") if isinstance(transform, dict) else None
    source_kind = "style_transform"
    profile = getattr(inst, "reconstruction_profile", None)
    if (
        not quad and profile is not None
        and getattr(profile, "perspective_confidence", 0.0) >= .70
    ):
        quad = getattr(profile, "perspective_quad", None)
        source_kind = "reconstruction_profile"
    if not isinstance(quad, (list, tuple)) or len(quad) != 4:
        return image, mask, None, {"applied": False, "reason": "no_valid_quad"}
    try:
        points = np.asarray(quad, dtype=np.float32).reshape(4, 2)
        if not np.isfinite(points).all():
            raise ValueError("non-finite quad")
        bbox = inst.bounding_box
        # Style quads are normalized to the immutable source bbox. A
        # reconstruction profile may carry image-space points.
        if source_kind == "style_transform":
            unit = np.asarray(((0, 0), (1, 0), (1, 1), (0, 1)), dtype=np.float32)
            if float(np.max(np.abs(points - unit))) < .01:
                return image, mask, None, {"applied": False, "reason": "identity_quad"}
            points[:, 0] = bbox.x + points[:, 0] * bbox.width
            points[:, 1] = bbox.y + points[:, 1] * bbox.height
        area = abs(float(cv2.contourArea(points)))
        top = float(np.linalg.norm(points[1] - points[0]))
        bottom = float(np.linalg.norm(points[2] - points[3]))
        left = float(np.linalg.norm(points[3] - points[0]))
        right = float(np.linalg.norm(points[2] - points[1]))
        width = int(round(max(top, bottom)))
        height = int(round(max(left, right)))
        if area < 25 or min(width, height) < 6 or max(width, height) > 4096:
            raise ValueError("degenerate quad")
        target = np.asarray(
            ((0, 0), (width - 1, 0), (width - 1, height - 1), (0, height - 1)),
            dtype=np.float32,
        )
        matrix = cv2.getPerspectiveTransform(points, target)
        inverse = cv2.getPerspectiveTransform(target, points)
        rectified_image = cv2.warpPerspective(
            image, matrix, (width, height), flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REFLECT_101,
        )
        rectified_mask = cv2.warpPerspective(
            mask.astype(np.uint8), matrix, (width, height),
            flags=cv2.INTER_NEAREST,
        ).astype(bool)
        if not rectified_mask.any():
            raise ValueError("empty rectified mask")

        def restore(candidate):
            restored = cv2.warpPerspective(
                np.asarray(candidate, dtype=np.uint8), inverse,
                (image.shape[1], image.shape[0]), flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_REFLECT_101,
            )
            composed = np.asarray(image, dtype=np.uint8).copy()
            composed[mask] = restored[mask]
            return composed

        return rectified_image, rectified_mask, restore, {
            "applied": True,
            "source": source_kind,
            "quad": points.tolist(),
            "rectified_size": [width, height],
        }
    except Exception as exc:
        return image, mask, None, {
            "applied": False,
            "reason": f"invalid_quad:{type(exc).__name__}",
        }


def _verify_final_cleanse(
    np,
    cv2,
    source,
    working,
    plans: List[Dict[str, Any]],
    policy: InpaintAssessmentPolicy,
) -> None:
    """Fresh Paddle check of final pre-Scribe pixels with bounded source retries."""
    if not plans or not policy.require_residual_verification:
        return
    from tofu.layers.ocr_verification import PaddleRegionVerifier
    verifier = PaddleRegionVerifier()
    active = [plan for plan in plans if (plan["inst"].text or "").strip()]
    if not active:
        return
    for attempt in range(max(0, policy.retry_budget) + 1):
        results = verifier.verify_regions(
            working,
            [plan["inst"].bounding_box for plan in active],
            [plan["inst"].text for plan in active],
        )
        retry: List[Dict[str, Any]] = []
        for plan, result in zip(active, results):
            inst = plan["inst"]
            evidence = result.evidence()
            evidence["attempt"] = attempt
            provenance = inst.repair_provenance or {}
            checks = provenance.setdefault("residual_checks", [])
            checks.append(evidence)
            source_like = (
                result.state == "agree"
                and result.confidence >= .50
                and (result.similarity or 0.0) >= .30
            )
            hallucinated = result.state == "disagree" and result.confidence >= .70
            unavailable = result.state in {"unavailable", "error"}
            if source_like or hallucinated or unavailable:
                provenance["auto_accepted"] = False
                provenance["review_required"] = True
                provenance["residual_verification"] = (
                    "source_text" if source_like else
                    "hallucinated_text" if hallucinated else result.state
                )
            else:
                provenance["residual_verification"] = "passed"
            inst.repair_provenance = provenance
            if (source_like or hallucinated) and attempt < policy.retry_budget:
                retry.append(plan)
        if not retry:
            break
        # Every retry is reconstructed from the untouched source, never from a
        # prior invented fill. Expand only the implicated region's mask.
        for plan in retry:
            expanded = cv2.dilate(
                plan["mask"].astype(np.uint8), np.ones((3, 3), np.uint8),
                iterations=attempt + 1,
            ).astype(bool)
            telea = cv2.inpaint(
                source, expanded.astype(np.uint8) * 255,
                TELEA_RADIUS_DEFAULT, cv2.INPAINT_TELEA,
            )
            working[expanded] = telea[expanded]
            plan["mask"] = expanded
        active = retry


def erase(
    asset: Any,
    text_manifest: TextManifest,
    candidate_observer=None,
    assessment_policy: Optional[InpaintAssessmentPolicy] = None,
) -> Any:
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

        ``candidate_observer`` is an optional server-owned callback receiving
        an unaccepted neural candidate and its exact mask.  Cleanse never
        writes candidate files itself; this keeps its image-layer contract
        pure while allowing the Render layer to expose an explicit review
        patch rather than silently accepting model output.
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
    assessment_policy = assessment_policy or InpaintAssessmentPolicy()

    # Plan every mask against the untouched source first.  Selection and
    # model-routing must never depend on pixels invented by an earlier repair.
    plans: List[Dict[str, Any]] = []
    for inst in text_manifest.instances:
        if getattr(inst, "dnt", False):
            continue
        bbox = inst.bounding_box
        if bbox is None or bbox.width <= 0 or bbox.height <= 0:
            continue
        surface = _containing_surface(text_manifest, bbox)
        instance_polygon = getattr(getattr(inst, "segmentation_mask", None), "polygon", None)
        surface_polygon = getattr(surface, "polygon", None)
        full_mask, mask_evidence = _select_region_mask(
            np, cv2, img_array, bbox, h, w, instance_polygon, surface_polygon,
        )
        if not full_mask.any():
            continue
        strategy = _choose_strategy(inst, surface, img_array, full_mask, mask_evidence)
        plans.append({
            "inst": inst,
            "surface": surface,
            "mask": full_mask,
            "strategy": strategy,
            "provider": (inst.repair_provenance or {}).get("requested_provider", "telea_fallback"),
        })

    touched = bool(plans)
    telea_mask = np.zeros((h, w), dtype=bool)
    stroke_widths: List[float] = []
    neural_plans: List[Dict[str, Any]] = []

    # Deterministic planar fills are intentionally executed separately.  They
    # are source-context fits, not generated content, and remain the safest
    # fast path where Scene and the instance agree.
    for plan in plans:
        inst, full_mask, strategy = plan["inst"], plan["mask"], plan["strategy"]
        if strategy == "flat":
            dominant_color = inst.background_profile.dominant_color if inst.background_profile else None
            _fill_flat(cv2, np, working, full_mask, h, w, dominant_color)
            if inst.repair_provenance:
                inst.repair_provenance["executed_provider"] = "analytic"
                _record_candidate(inst, "analytic", bool(inst.repair_provenance["auto_accepted"]),
                                  {"kind": "flat_reconstruction"}, "deterministic", [inst.id])
        elif strategy == "smooth_gradient":
            _fill_gradient(cv2, np, working, full_mask, h, w)
            if inst.repair_provenance:
                inst.repair_provenance["executed_provider"] = "analytic"
                _record_candidate(inst, "analytic", bool(inst.repair_provenance["auto_accepted"]),
                                  {"kind": "plane_reconstruction"}, "deterministic", [inst.id])
        elif strategy == "neural":
            neural_plans.append(plan)
        else:
            telea_mask |= full_mask
            if inst.repair_provenance:
                inst.repair_provenance["executed_provider"] = "telea_fallback"
                _record_candidate(inst, "telea_fallback", False,
                                  {"reason": "no promoted neural candidate"}, "review_required", [inst.id])
            stroke = _stroke_px(inst)
            if stroke:
                stroke_widths.append(stroke)

    # Compatible neural regions are sent together, *from img_array*.  In
    # particular, an adjacent region never sees another model's output as its
    # own context.  An unpromoted backend may be exercised for evidence but is
    # never allowed to silently become the Cleansed base.
    for group in _neural_groups(np, cv2, neural_plans):
        group_mask = np.zeros((h, w), dtype=bool)
        for plan in group:
            group_mask |= plan["mask"]
        group_ids = [plan["inst"].id for plan in group]
        group_key = _repair_group_key(np, group, group_mask)
        repair_image, repair_mask, restore_candidate, perspective = (
            _perspective_repair_context(
                cv2, np, img_array, group_mask, group[0]["inst"]
            )
            if len(group) == 1 else
            (img_array, group_mask, None, {
                "applied": False, "reason": "multi_region_group",
            })
        )
        artifacts: Dict[str, Any] = {}

        def observe_candidate(info, candidate):
            if candidate_observer is None:
                return
            try:
                artifacts[info["candidate_id"]] = candidate_observer({
                    **info, "group_key": group_key, "group_ids": group_ids,
                    "decision": "candidate",
                }, candidate, group_mask)
            except Exception:
                artifacts[info["candidate_id"]] = None

        if assessment_policy.mode == "multi":
            multi = inpaint_providers.repair_multi_candidate(
                repair_image,
                repair_mask,
                max_candidates=assessment_policy.max_neural_candidates,
                on_candidate=observe_candidate,
                candidate_transform=restore_candidate,
                quality_image=img_array if restore_candidate is not None else None,
                quality_mask=group_mask if restore_candidate is not None else None,
            )
            outcome = multi.selected
            candidate_evidence = multi.candidates
            repaired = outcome.image if outcome is not None else None
            accepted = outcome is not None
            provider_id = outcome.provider if outcome is not None else None
            decision = "accepted" if accepted else "all_candidates_rejected"
        else:
            provider_id = group[0]["provider"]
            outcome = inpaint_providers.repair(provider_id, repair_image, repair_mask)
            repaired = (
                restore_candidate(outcome.image)
                if restore_candidate is not None and outcome.image is not None
                else outcome.image
            )
            passed, quality = (
                inpaint_providers.quality_gate(img_array, repaired, group_mask)
                if repaired is not None else
                (False, {"passed": False, "score": 0.0, "reason": "provider returned no candidate"})
            )
            accepted = bool(passed and inpaint_providers.provider_spec(provider_id).promoted)
            candidate_evidence = [{
                "candidate_id": f"{provider_id}:0", "provider": provider_id,
                "execution": outcome.evidence(), "quality_gate": quality,
                "eligible": accepted, "selected": accepted,
            }]
            if repaired is not None:
                observe_candidate(candidate_evidence[0], repaired)
            decision = "accepted" if accepted else "quality_gate_rejected"
        for plan in group:
            inst = plan["inst"]
            if inst.repair_provenance:
                inst.repair_provenance["execution"] = (
                    outcome.evidence() if outcome is not None else {"ok": False}
                )
                inst.repair_provenance["repair_group"] = group_ids
                inst.repair_provenance["repair_group_key"] = group_key
                inst.repair_provenance["selection_reason"] = decision
                inst.repair_provenance["perspective"] = perspective
            for item in candidate_evidence:
                artifact = artifacts.get(item["candidate_id"])
                _record_candidate(
                    inst,
                    item["provider"],
                    bool(item.get("selected")),
                    {
                        "execution": item.get("execution"),
                        "quality_gate": item.get("quality_gate"),
                        "artifact": artifact,
                        "candidate_id": item["candidate_id"],
                    },
                    "accepted" if item.get("selected") else (
                        "unpromoted_provider" if not item.get("promoted", True)
                        else "quality_gate_rejected"
                    ),
                    group_ids,
                )
        if accepted and repaired is not None and repaired.shape == img_array.shape:
            # The provider had to preserve the full known context to pass the
            # gate.  We still blend only inside the selected erase footprint.
            for plan in group:
                full_mask = plan["mask"]
                win = _local_window(np, full_mask, RING_PX + int(FEATHER_PX) + 2, h, w)
                if win is not None:
                    x0, y0, x1, y1 = win
                    _feathered_blend(cv2, np, working[y0:y1, x0:x1],
                                      full_mask[y0:y1, x0:x1], repaired[y0:y1, x0:x1])
                if plan["inst"].repair_provenance:
                    plan["inst"].repair_provenance["executed_provider"] = provider_id
        else:
            telea_mask |= group_mask
            for plan in group:
                inst = plan["inst"]
                if inst.repair_provenance:
                    inst.repair_provenance["rejected_candidate"] = provider_id or "all"
                    inst.repair_provenance["executed_provider"] = "telea_fallback"
                    inst.repair_provenance["auto_accepted"] = False
                    inst.repair_provenance["review_required"] = True
                stroke = _stroke_px(inst)
                if stroke:
                    stroke_widths.append(stroke)

    if telea_mask.any():
        radius = TELEA_RADIUS_DEFAULT
        if stroke_widths:
            avg_stroke = sum(stroke_widths) / len(stroke_widths)
            radius = int(max(TELEA_RADIUS_MIN, min(TELEA_RADIUS_MAX, round(avg_stroke / 2))))
        # Like neural groups, deterministic fallback samples the original
        # image once, then is composited into the planned masks.  It cannot
        # accumulate order-dependent repairs across regions.
        telea = cv2.inpaint(img_array, telea_mask.astype(np.uint8) * 255, radius, cv2.INPAINT_TELEA)
        working[telea_mask] = telea[telea_mask]

    _verify_final_cleanse(
        np, cv2, img_array, working, plans, assessment_policy
    )

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
