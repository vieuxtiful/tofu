"""Deterministic frame compositor for reviewed video tracks.

Consumes `ResolvedFramePlan` and nothing else. Every temporal decision is made
before a pixel moves, which is what lets a preview be a genuine window onto the
export rather than a second opinion about it.

The range a render was asked for must not change the frames it produces. Two
pieces of state used to break that -- the temporal background evidence and the
previous frame the motion mask differences against -- because both started
empty at `start_frame`. Both are now filled by a WARM-UP pass that decodes the
frames before the range without emitting them, the same device the analyzer's
resume overlap uses.
"""
from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional
from tofu.core.types import ImageLike

from .larder import LARDER_DEPTH, BackgroundLarder
from .plan import Downgrade, PlanInputs, ResolvedFramePlan, ResolvedRegionPlan, plate
from .types import RenderKeyframe

## Enough to fill the larder and seat the previous frame. One more than the
## larder depth because the motion mask needs a predecessor of its own.
WARMUP_FRAMES = LARDER_DEPTH + 1

## Frame-to-frame difference, per channel, above which a pixel is moving.
MOTION_THRESHOLD = 32
MIN_FOREGROUND_AREA = 12


def _keyframe(data: Dict[str, Any]) -> RenderKeyframe:
    allowed = RenderKeyframe.__dataclass_fields__
    return RenderKeyframe(**{key: value for key, value in data.items() if key in allowed})


def _erase(frame: ImageLike, region: ResolvedRegionPlan, mask: Any, larder: BackgroundLarder,
           frame_index: int) -> Dict[str, Any]:
    """Remove the source text, preferring corroborated temporal evidence.

    The raw crop is banked BEFORE erasure. Banking the erased crop -- and then
    comparing the next raw crop against it -- is what made the temporal path
    unreachable: the measured difference was dominated by the text the whole
    exercise exists to remove.
    """
    import cv2
    box = region.bbox
    x, y = int(box.x), int(box.y)
    x2, y2 = x + int(box.width), y + int(box.height)
    raw = frame[y:y2, x:x2].copy()
    local_mask = mask[y:y2, x:x2]

    evidence = larder.evidence(region.track_id, raw, local_mask)
    larder.remember(region.track_id, frame_index, raw)
    if evidence is not None:
        frame[y:y2, x:x2] = evidence["patch"]
        return {"strategy": "temporal_median", "frames": evidence["frames"],
                "confidence": evidence["confidence"], "alignment": evidence["alignment"]}
    ## Spatial fallback. Bounded to the padded crop rather than run on the whole
    ## frame: the old code inpainted every pixel of every frame once per region.
    pad = region.erase_pad
    height, width = frame.shape[:2]
    px, py = max(0, x - pad), max(0, y - pad)
    px2, py2 = min(width, x2 + pad), min(height, y2 + pad)
    window = frame[py:py2, px:px2]
    repaired = cv2.inpaint(window, mask[py:py2, px:px2], 3, cv2.INPAINT_TELEA)
    frame[py:py2, px:px2] = repaired
    return {"strategy": "inpaint_telea", "frames": [], "confidence": 0.0, "alignment": []}


def _region_masks(plan: ResolvedFramePlan, region: ResolvedRegionPlan) -> Any:
    """Glyph-level mask where the analyzer gave us a polygon, bbox otherwise."""
    import cv2
    import numpy as np
    mask = np.zeros((plan.height, plan.width), np.uint8)
    polygon = region.glyph_polygon
    if polygon and len(polygon) >= 3:
        cv2.fillPoly(mask, [np.asarray(polygon, dtype=np.int32).reshape((-1, 1, 2))], 255)
        return mask
    box = region.bbox
    x, y = int(box.x), int(box.y)
    mask[y:y + int(box.height), x:x + int(box.width)] = 255
    return mask


def _foreground(plan: ResolvedFramePlan, region: ResolvedRegionPlan, mask: Any,
                motion: Optional[Any]) -> Optional[Any]:
    """Pixels that belong in front of the text and must be put back afterwards."""
    import cv2
    import numpy as np
    if region.occlusion_mask_path:
        path = Path(region.occlusion_mask_path)
        if path.is_file():
            supplied = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
            if supplied is not None and supplied.shape == mask.shape:
                return supplied > 127
        region.downgrades.append(Downgrade(
            "occlusion_mask_missing", "compose", requested=region.occlusion_mask_path,
            applied="motion", track_id=region.track_id, severity="review",
            detail="declared occlusion mask is not readable; fell back to motion"))
    if motion is None:
        return None
    box = region.bbox
    x, y = int(box.x), int(box.y)
    x2, y2 = x + int(box.width), y + int(box.height)
    region_mask = np.zeros_like(motion)
    region_mask[y:y2, x:x2] = 255
    candidate = cv2.bitwise_and(motion, region_mask)
    count, labels, stats, _ = cv2.connectedComponentsWithStats((candidate > 0).astype(np.uint8))
    accepted = np.zeros_like(candidate)
    for component in range(1, count):
        cx, cy, cw, ch, area = stats[component]
        touches_edge = cx <= x + 2 or cy <= y + 2 or cx + cw >= x2 - 2 or cy + ch >= y2 - 2
        if touches_edge and area >= MIN_FOREGROUND_AREA:
            accepted[labels == component] = 255
    accepted[mask > 0] = 0
    return accepted > 0 if accepted.any() else None


def compose(source: Path, destination: Path, manifest: Dict[str, Any], tracks: List[Dict[str, Any]],
            observations: List[Dict[str, Any]], keyframes: List[Dict[str, Any]], *,
            start_frame: int = 0, end_frame: Optional[int] = None,
            frame_dir: Optional[Path] = None,
            inputs: Optional[PlanInputs] = None,
            warmup_frames: int = WARMUP_FRAMES,
            progress: Callable[[float], None] = lambda value: None,
            cancelled: Callable[[], bool] = lambda: False,
            on_plan: Optional[Callable[[ResolvedFramePlan], None]] = None) -> None:
    import cv2
    import numpy as np
    from PIL import Image
    from tofu.core.types import InstText, TextManifest
    from tofu.layers import scribe

    end_frame = int(manifest["frame_count"] - 1 if end_frame is None else end_frame)
    fps = float(manifest.get("fps") or 30)
    width, height = int(manifest["width"]), int(manifest["height"])
    pts = manifest.get("frame_pts") or []
    inputs = inputs or PlanInputs()
    destination.parent.mkdir(parents=True, exist_ok=True)
    if frame_dir is not None: frame_dir.mkdir(parents=True, exist_ok=True)

    ## Decode from before the range so the temporal state is identical to what a
    ## full-clip render would have at this frame. Without it the first frames of
    ## a preview are erased from a cold larder and restored from no motion mask,
    ## and the reviewer approves pixels the export will never produce.
    warm_from = max(0, start_frame - max(0, int(warmup_frames)))
    cap = cv2.VideoCapture(str(source))
    cap.set(cv2.CAP_PROP_POS_FRAMES, warm_from)
    if hasattr(cv2, "CAP_PROP_ORIENTATION_AUTO"): cap.set(cv2.CAP_PROP_ORIENTATION_AUTO, 0)
    writer = None if frame_dir is not None else cv2.VideoWriter(
        str(destination), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    if not cap.isOpened() or (writer is not None and not writer.isOpened()):
        cap.release()
        if writer is not None: writer.release()
        raise RuntimeError("video compositor could not initialize codec")

    track_map = {track["id"]: track for track in tracks}
    by_frame: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
    for observation in observations:
        by_frame[int(observation["frame_index"])].append(observation)
    by_track: Dict[str, List[RenderKeyframe]] = defaultdict(list)
    for key in keyframes:
        by_track[key["track_id"]].append(_keyframe(key))

    total = max(1, end_frame - start_frame + 1)
    larder = BackgroundLarder()
    previous_original = None
    try:
        for index in range(warm_from, end_frame + 1):
            if cancelled(): raise InterruptedError("video composition cancelled")
            ok, frame = cap.read()
            if not ok: raise RuntimeError(f"decoder stopped at frame {index}")
            source_frame = frame.copy()

            plan = plate(index, pts_seconds=(pts[index] if index < len(pts) else index / fps),
                         width=width, height=height, tracks=track_map,
                         observations=by_frame.get(index, []), keyframes=by_track, inputs=inputs)
            larder.retain(region.track_id for region in plan.regions)

            motion = None
            if previous_original is not None:
                difference = cv2.cvtColor(cv2.absdiff(source_frame, previous_original),
                                          cv2.COLOR_BGR2GRAY)
                motion = ((difference > MOTION_THRESHOLD).astype(np.uint8) * 255)
                kernel = np.ones((3, 3), np.uint8)
                motion = cv2.dilate(cv2.morphologyEx(motion, cv2.MORPH_OPEN, kernel), kernel, iterations=1)

            instances: List[InstText] = []
            render_params: Dict[str, Any] = {}
            foreground_layers: List[Any] = []
            for region in plan.regions:
                mask = _region_masks(plan, region)
                region.provenance["repair"] = _erase(frame, region, mask, larder, index)
                instance = InstText(id=region.track_id, bounding_box=region.bbox,
                                    target_text=region.target_text, text=region.source_text,
                                    target_language=region.target_language,
                                    style_profile=region.style,
                                    frame_index=index, track_id=region.track_id)
                instances.append(instance)
                render_params[instance.id] = region.render_params
                if region.restore_foreground:
                    layer = _foreground(plan, region, mask, motion)
                    if layer is not None:
                        foreground_layers.append(layer)

            if instances:
                frame_manifest = TextManifest(
                    asset_id=f"video-{index}", total_regions=len(instances), instances=instances,
                    img_dim=(width, height),
                    targ_lang=next((i.target_language for i in instances if i.target_language), None))
                rendered = scribe.render(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)),
                                         frame_manifest, frame_manifest.targ_lang or "en", render_params)
                frame = cv2.cvtColor(np.asarray(rendered.convert("RGB")), cv2.COLOR_RGB2BGR)
                for layer in foreground_layers:
                    frame[layer] = source_frame[layer]

            previous_original = source_frame
            if index < start_frame:
                continue                      # warm-up: state only, no output
            if on_plan:
                on_plan(plan)
            if frame_dir is not None:
                if not cv2.imwrite(str(frame_dir / f"{index - start_frame:08d}.png"), frame):
                    raise RuntimeError("could not write rendered frame")
            else:
                writer.write(frame)
            progress((index - start_frame + 1) / total)
    finally:
        cap.release()
        if writer is not None: writer.release()
