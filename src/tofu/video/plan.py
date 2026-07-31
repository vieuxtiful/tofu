"""The single per-frame contract every video stage consumes.

`ResolvedFramePlan` is the mise en place of a frame: geometry, erasure,
typography, treatment and restoration all resolved once, before any pixel
moves. Preview and export cannot drift apart because downstream there is
nothing left to decide.

This deliberately does not live in `types.py`. That module is the versioned
wire format that lands in `data_json`; a plan is DERIVED -- recomputed from
those contracts plus keyframes plus the revision of the code that reads them.
A persisted plan is a stale plan. Only its `revision`, and the downgrades it
produced, are durable.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from tofu.core.types import BBox, GarnishProfile, RenderParams, StyleProfil
from .temporal import resolve_keyframes
from .types import RenderKeyframe

PLAN_SCHEMA_VERSION = 1
RENDERER_REVISION = "scribe-v1"

## Padding around a region when erasure runs on a crop. Cleanse needs a ring of
## untouched background outside the mask to fill from; this is comfortably
## wider than its own ring + feather.
DEFAULT_ERASE_PAD = 24


@dataclass(frozen=True)
class PlanInputs:
    """Everything a plan is a pure function of. Hashed into `revision`.

    The fields ARE the invalidation dependencies, written down. Anything absent
    here is something that can change without invalidating a rendered artifact
    -- which is how stale output gets published.
    """
    job_id: str = ""
    dependency_revision: int = 1
    track_revisions: Tuple[Tuple[str, int], ...] = ()
    keyframe_digest: str = ""
    analyzer_revision: str = ""
    renderer_revision: str = RENDERER_REVISION
    ## nothing used to invalidate when a font was installed or removed, so an
    ## export could keep a filename that promised a face it no longer used
    font_registry_digest: str = ""
    policy_digest: str = ""
    plan_schema_version: int = PLAN_SCHEMA_VERSION

    @property
    def revision(self) -> str:
        material = json.dumps(asdict(self), sort_keys=True, separators=(",", ":"), default=str)
        return hashlib.sha256(material.encode()).hexdigest()[:16]


@dataclass
class Downgrade:
    """One requested capability the pipeline could not honour verbatim.

    Recorded rather than silently applied: a stroke quietly clamped or an
    effect quietly dropped looks like a rendering bug to the reviewer, who has
    no way to tell it apart from one.
    """
    code: str
    stage: str                      # cleanse | scribe | garnish | compose | export
    requested: Any = None
    applied: Any = None
    detail: str = ""
    severity: str = "review"        # info | review | warning
    track_id: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ResolvedRegionPlan:
    """One track's fully resolved instruction for one frame."""
    track_id: str
    shot_id: str = ""
    observation_id: Optional[str] = None
    order: int = 0                                   # painter's order, ascending
    bbox: Optional[BBox] = None                      # integer, clipped to the frame
    quad: Optional[List[List[float]]] = None
    opacity: float = 1.0
    style: Optional[StyleProfil] = None
    render_params: Optional[RenderParams] = None
    source_text: Optional[str] = None
    target_text: Optional[str] = None
    target_language: Optional[str] = None

    # erasure
    erase: bool = True
    glyph_polygon: Optional[List[List[float]]] = None
    erase_pad: int = DEFAULT_ERASE_PAD

    # treatment / restoration
    garnish: Optional[GarnishProfile] = None
    occlusion_mask_path: Optional[str] = None
    restore_foreground: bool = True

    verify_level: str = "cheap"                      # none | cheap | full
    downgrades: List[Downgrade] = field(default_factory=list)
    provenance: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ResolvedFramePlan:
    frame_index: int
    pts_seconds: float = 0.0
    width: int = 0
    height: int = 0
    revision: str = ""
    regions: List[ResolvedRegionPlan] = field(default_factory=list)
    motion_mask_source: str = "auto"                 # auto | mask_path | none
    downgrades: List[Downgrade] = field(default_factory=list)

    def all_downgrades(self) -> List[Downgrade]:
        return [*self.downgrades, *(d for region in self.regions for d in region.downgrades)]


def keyframe_digest(keyframes: Sequence[Dict[str, Any]]) -> str:
    material = json.dumps(sorted((json.dumps(k, sort_keys=True, default=str) for k in keyframes)),
                          separators=(",", ":"))
    return hashlib.sha256(material.encode()).hexdigest()[:16]


def font_registry_digest(font_registry: Any) -> str:
    """Fingerprint the faces actually available, not the object identity."""
    if font_registry is None:
        return ""
    try:
        faces = sorted(getattr(font_registry, "_fonts", {}) or {})
    except Exception:
        return ""
    return hashlib.sha256("|".join(faces).encode()).hexdigest()[:16]


def plan_inputs(job: Mapping[str, Any], tracks: Sequence[Mapping[str, Any]],
                keyframes: Sequence[Dict[str, Any]], *, analyzer_revision: str = "",
                font_registry: Any = None, policy: Any = None) -> PlanInputs:
    return PlanInputs(
        job_id=str(job.get("id", "")),
        dependency_revision=int(job.get("dependency_revision", 1)),
        track_revisions=tuple(sorted((str(t.get("id", "")), int(t.get("revision", 1)))
                                     for t in tracks)),
        keyframe_digest=keyframe_digest(keyframes),
        analyzer_revision=analyzer_revision,
        font_registry_digest=font_registry_digest(font_registry),
        policy_digest=("" if policy is None else
                       hashlib.sha256(json.dumps(policy, sort_keys=True, default=str).encode()).hexdigest()[:16]),
    )


def _style(data: Optional[Dict[str, Any]], quad: Optional[List[List[float]]],
           downgrades: List[Downgrade], track_id: str) -> StyleProfil:
    fields = StyleProfil.__dataclass_fields__
    payload = dict(data or {})
    unsupported = sorted(set(payload) - set(fields))
    if unsupported:
        downgrades.append(Downgrade("style_key_unsupported", "scribe", requested=unsupported,
                                    applied=None, track_id=track_id, severity="info",
                                    detail=f"ignored style keys: {', '.join(unsupported)}"))
    style = StyleProfil(**{name: value for name, value in payload.items() if name in fields})
    if quad is not None:
        style.transform = {**(style.transform or {}), "quad": quad}
    return style


def plate(frame_index: int, *, pts_seconds: float, width: int, height: int,
          tracks: Mapping[str, Mapping[str, Any]],
          observations: Sequence[Mapping[str, Any]],
          keyframes: Mapping[str, Sequence[RenderKeyframe]],
          inputs: PlanInputs) -> ResolvedFramePlan:
    """Plate one frame: resolve every decision, record every downgrade.

    Plating is the last step before service -- everything arranged, nothing left
    to decide. Pure in (frame_index, inputs): the same frame resolves to the
    same plan regardless of which render range asked for it, which is half of
    what makes a preview a window onto the export rather than a second opinion.
    """
    plan = ResolvedFramePlan(frame_index=frame_index, pts_seconds=pts_seconds,
                             width=width, height=height, revision=inputs.revision)
    ordered = sorted(observations,
                     key=lambda o: (int((tracks.get(o["track_id"]) or {}).get("compositing_order", 0)),
                                    str(o["track_id"])))
    for observation in ordered:
        track = tracks.get(observation["track_id"])
        if not track:
            continue
        target = track.get("target_text")
        if track.get("status") == "excluded" or not target:
            continue

        resolved = resolve_keyframes(
            frame_index,
            {"bbox": observation["bbox"], "opacity": 1.0,
             "quad": observation.get("quad"), "style": track.get("style") or {}},
            keyframes.get(track["id"], []))

        box = resolved["bbox"]
        x = max(0, int(box["x"])); y = max(0, int(box["y"]))
        x2 = min(width, x + max(1, int(box["width"])))
        y2 = min(height, y + max(1, int(box["height"])))
        bbox = BBox(x=x, y=y, width=max(1, x2 - x), height=max(1, y2 - y))

        downgrades: List[Downgrade] = []
        effects = resolved.get("effects")
        if effects:
            ## the keyframe API accepts, validates, stores and resolves `effects`,
            ## and scribe implements none of them. silently dropping them made a
            ## reviewer's deliberate choice indistinguishable from a no-op.
            downgrades.append(Downgrade("effect_unsupported", "scribe", requested=effects,
                                        applied=None, track_id=track["id"], severity="review",
                                        detail="renderer implements no keyframe effects"))
        style = _style(resolved.get("style"), resolved.get("quad"), downgrades, track["id"])

        plan.regions.append(ResolvedRegionPlan(
            track_id=str(track["id"]),
            shot_id=str(observation.get("shot_id") or track.get("shot_id") or ""),
            observation_id=observation.get("id"),
            order=int(track.get("compositing_order", 0)),
            bbox=bbox, quad=resolved.get("quad"),
            opacity=float(resolved.get("opacity", 1) or 1),
            style=style,
            render_params=RenderParams(position=bbox, style=style,
                                       opacity=float(resolved.get("opacity", 1) or 1),
                                       rotation=(style.transform or {}).get("rotation")),
            source_text=track.get("source_text"),
            target_text=str(target),
            target_language=track.get("target_language"),
            glyph_polygon=observation.get("glyph_mask_polygon"),
            occlusion_mask_path=resolved.get("mask_path") or observation.get("occlusion_mask_path"),
            downgrades=downgrades,
        ))
    plan.motion_mask_source = ("mask_path" if any(r.occlusion_mask_path for r in plan.regions)
                              else "auto")
    return plan
