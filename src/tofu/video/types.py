"""Versioned contracts for video localization.

Large binary evidence (frames, crops and masks) is represented by artifact
paths.  The manifest itself remains cheap enough to page and snapshot.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class OCRCandidate:
    engine: str
    text: str
    confidence: float
    pass_name: str = "primary"
    accepted: bool = False
    evidence: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TrackObservation:
    id: str
    track_id: str
    shot_id: str
    frame_index: int
    pts_seconds: float
    bbox: Dict[str, float]
    quad: Optional[List[List[float]]] = None
    glyph_mask_path: Optional[str] = None
    glyph_mask_polygon: Optional[List[List[float]]] = None
    occlusion_mask_path: Optional[str] = None
    crop_path: Optional[str] = None
    visible: bool = True
    tracking_confidence: float = 1.0
    sharpness: Optional[float] = None
    visual_change: Optional[float] = None
    ocr_trigger: Optional[str] = None
    ocr_candidates: List[OCRCandidate] = field(default_factory=list)


@dataclass
class RenderKeyframe:
    id: str
    track_id: str
    frame_index: int
    scope: str = "frame"  # frame | range | track
    end_frame: Optional[int] = None
    bbox: Optional[Dict[str, float]] = None
    quad: Optional[List[List[float]]] = None
    opacity: Optional[float] = None
    style: Optional[Dict[str, Any]] = None
    effects: Optional[Dict[str, Any]] = None
    mask_path: Optional[str] = None
    created_at: float = 0.0


@dataclass
class TextTrack:
    id: str
    shot_id: str
    start_frame: int
    end_frame: int
    source_text: Optional[str] = None
    target_text: Optional[str] = None
    source_language: Optional[str] = None
    target_language: Optional[str] = None
    style: Dict[str, Any] = field(default_factory=dict)
    status: str = "review_required"
    consensus_confidence: float = 0.0
    ## Folded OCR evidence (see temporal.ConsensusAccumulator). Lives on the
    ## track rather than in the analyzer, so a resumed job can finish the vote
    ## for tracks that were never in this process.
    consensus_state: Dict[str, Any] = field(default_factory=dict)
    observation_count: int = 0
    ## painter's order within a frame, ascending. overlapping signage has to
    ## resolve to one deterministic stacking, and the reviewer owns it.
    compositing_order: int = 0
    revision: int = 1


@dataclass
class VideoManifest:
    asset_id: str
    width: int
    height: int
    duration: float
    frame_count: int
    fps: Optional[float]
    time_base: Optional[str]
    codec: Optional[str]
    pixel_format: Optional[str] = None
    color: Dict[str, Any] = field(default_factory=dict)
    rotation: int = 0
    audio_streams: List[Dict[str, Any]] = field(default_factory=list)
    frame_pts: List[float] = field(default_factory=list)
    tracks: List[TextTrack] = field(default_factory=list)
    version: int = 1

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
