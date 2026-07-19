## types
## vieuxtiful
"""
This module defines the core data structures 
and enums used throughout ToFu. It provides 
the foundational types for assets, 
processing states, and validation results.
"""

from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any, Literal
from enum import Enum
from collections import defaultdict
from pathlib import Path

# re:Enums

class ScrptSpprt(str, Enum): 
    FULL = "full"
    PARTIAL = "partial"
    UNSUPPORTED = "unsupported"

class VldtnSeverity(str, Enum): ## UI loc error category (leveled)
    ERROR = "error"
    WARNING = "warning"
    INFO = "info"

class LayerMode(str, Enum): ## user workflow type selection
    AUTO = "auto"
    MANUAL = "manual"
    HYBRID = "hybrid"

class AssetType(str, Enum): ## assets for processing 
    IMAGE = "image"
    VIDEO = "video"

class PrcStatus(str, Enum): ## processing status
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    MANUAL_REVIEW = "manual_review"
    MANUALLY_APPROVED = "manually_approved"
    MANUALLY_REJECTED = "manually_rejected"
    AUTO_APPROVED = "auto_approved"
    AUTO_REJECTED = "auto_rejected"
    AWAITING_REFINEMENT = "awaiting_refinement"  ## HYBRID pause checkpoint

# re:Asset Ingestion

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"}
VIDEO_EXTS = {".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v"}

@dataclass
class AssetInfo: ## ingestion descriptor resolved at load time (frontend upload or CLI)
    asset_type: AssetType = AssetType.IMAGE
    frame_count: int = 1                  ## static image defaults to 1
    fps: Optional[float] = None           ## video only
    duration: Optional[float] = None      ## video only, seconds
    source: Optional[str] = None          ## file path / URL, if known

def infer_asset_info(asset: Any) -> AssetInfo:
    """Resolve an AssetInfo from a loaded asset. Path-like inputs are
    classified by extension; anything unrecognized defaults to a static
    image (frames=1), which routes through the static pipeline."""
    if isinstance(asset, (str, Path)):
        ext = Path(str(asset).split("?")[0]).suffix.lower()
        if ext in VIDEO_EXTS:
            return AssetInfo(asset_type=AssetType.VIDEO, frame_count=0, source=str(asset))
        return AssetInfo(asset_type=AssetType.IMAGE, frame_count=1, source=str(asset))
    return AssetInfo()  ## in-memory image default: static, frames=1

# re:Core Data Structures

@dataclass
class BBox: ## B(x) = {b^t, b^c} (txt, char); bounding box
    x: int
    y: int
    width: int
    height: int

Point = tuple[int, int]  ## (x, y) pixel coordinate
Polygon = List[Point]    ## closed contour, >= 3 points

@dataclass
class Mask:
    polygon: Polygon
    confidence: float
    holes: Optional[List[Polygon]] = None  ## interior cutouts, if any

@dataclass
class SceneRegion: ## candidate text-bearing surface from scene's pre-pass
    bbox: BBox
    semantic_label: str                    ## "panel" | "bordered_region" | "surface" | backend-specific
    confidence: float
    background_color: Optional[str] = None ## dominant color, hex
    border_detected: bool = False
    polygon: Optional[Polygon] = None

@dataclass
class InstText:
    id: str
    bounding_box: BBox
    segmentation_mask: Optional[Mask] = None
    text: Optional[str] = None
    target_text: Optional[str] = None    ## translated string scribe renders (from TM or translator)
    language: Optional[str] = None
    characteristics: Optional['CharactText'] = None
    style_profile: Optional['StyleProfil'] = None
    background_profile: Optional['BgProfil'] = None
    confidence: Optional[float] = None
    detected_language: Optional[str] = None
    reading_order: Optional[int] = None
    source_asset_id: Optional[str] = None
    target_asset_id: Optional[str] = None
    frame_index: Optional[int] = None            ## video: frame this observation belongs to; None for images
    temporal_span: Optional[tuple[int, int]] = None  ## video: (first_frame, last_frame) the instance persists
    track_id: Optional[str] = None               ## video: links per-frame instances into one tracked text entity
    dnt: bool = False                            ## do-not-translate flag: excluded from export and scribe
    target_language: Optional[str] = None        ## per-region override of target language (None = use manifest default)
    glyph_fallback: Optional[bool] = None         ## True: scribe swapped the requested font for a codepoint-covering one

@dataclass
class CharactText:
    font_style: Optional[str] = None
    color: Optional[str] = None          # RGBA, CMYK, or hex
    size: Optional[int] = None
    positioning: Optional[Dict[str, Any]] = None  # warp, skew, etc.
    effects: Optional[List[str]] = None   # shadow, glow, etc.

@dataclass
class StyleProfil:
    font_family: Optional[str] = None
    font_weight: Optional[str] = None
    color: Optional[str] = None
    shadow: Optional[Dict[str, Any]] = None
    effects: Optional[List[str]] = None
    # -- text layout --
    font_size: Optional[int] = None           # explicit px override; None = auto-fit
    italic: Optional[bool] = None
    underline: Optional[bool] = None
    subscript: Optional[bool] = None
    superscript: Optional[bool] = None
    align_h: Optional[str] = None             # "left" | "right" | "center"
    align_v: Optional[str] = None             # "top" | "middle" | "bottom"
    justification: Optional[str] = None       # "last_left" | "last_right" | "justify" | "justify_center"
    indent: Optional[float] = None            # px indent for first line
    tracking: Optional[float] = None          # letter spacing in px
    kerning: Optional[float] = None           # pairwise kerning adjustment in px
    leading: Optional[float] = None           # line height in px
    baseline_shift: Optional[float] = None    # vertical baseline offset in px
    tab_width: Optional[float] = None         # tab stop width in px
    tsume: Optional[float] = None             # CJK character compression (0.0–1.0, Japanese only)
    # -- appearance --
    stroke_color: Optional[str] = None        # hex
    stroke_width: Optional[float] = None      # px

@dataclass
class BgProfil:
    semantic_label: Optional[str] = None  # "brick wall", "wood sign", etc.
    texture: Optional[str] = None
    gradients: Optional[List[str]] = None
    patterns: Optional[List[str]] = None
    dominant_color: Optional[str] = None  # hex; cleanse fill / scribe contrast hint

@dataclass
class TextManifest: ## loc task manifest via cicerone
    asset_id: str
    total_regions: int
    instances: List[InstText]
    src_lang: Optional[str] = None
    targ_lang: Optional[str] = None
    img_dim: Optional[tuple[int, int]] = None
    scene_regions: List[SceneRegion] = field(default_factory=list)
    prcssng_time: Optional[float] = None
    asset_type: AssetType = AssetType.IMAGE
    frame_count: int = 1                  ## static image == 1; video == n frames
    fps: Optional[float] = None           ## video only
    duration: Optional[float] = None      ## video only, seconds

@dataclass
class RenderParams: ## static render spec consumed by scribe (time-agnostic)
    position: Optional[BBox] = None
    style: Optional[StyleProfil] = None
    opacity: Optional[float] = None
    resolution: Optional[tuple[int, int]] = None
    rotation: Optional[float] = None      ## degrees

@dataclass
class CameraTrack: ## temporal wrapper animating a RenderParams per scene (video compositor)
    render_params: RenderParams
    anchor_track_id: Optional[str] = None     ## entity the text is anchored to (parallax control)
    trajectory: Optional[List[tuple[int, BBox]]] = None  ## (frame_index, position) keyframes
    start_frame: int = 0
    end_frame: int = 0                        ## == start_frame for static (frames=1)
    fps: Optional[float] = None
    duration: Optional[float] = None          ## seconds; None for static

@dataclass
class VldtnClass:
    severity: VldtnSeverity
    code: str
    message: str
    suggestion: Optional[str] = None
    region_id: Optional[str] = None

@dataclass
class VldtnReport:
    passed: bool
    issues: List[VldtnClass] = field(default_factory=list)
    scrpt_spprt: Dict[str, ScrptSpprt] = field(default_factory=dict)
    glyph_segmentation_score: Optional[float] = None
    render_quality_score: Optional[float] = None
    expansion_fit: Dict[str, float] = field(default_factory=dict)  ## region_id -> predicted_width / bbox_width
    suggested_actions: List[str] = field(default_factory=list)

@dataclass
class PipelineCfg:
    tofu_mode: LayerMode = LayerMode.AUTO
    cicerone_mode: LayerMode = LayerMode.AUTO
    scene_mode: LayerMode = LayerMode.AUTO
    cleanse_mode: LayerMode = LayerMode.AUTO
    scribe_mode: LayerMode = LayerMode.AUTO
    verify_mode: LayerMode = LayerMode.AUTO
    memory_mode: LayerMode = LayerMode.AUTO
    qa_threshold: float = 0.8  ## verify gate: below this, run fails and memory is skipped
    scene_backend: str = "classical"       ## "classical" | "sam" | "null"
    scene_model_path: Optional[str] = None ## checkpoint path for model backends (e.g. SAM)

@dataclass
class PipelineResult:
    success: bool
    status: PrcStatus = PrcStatus.COMPLETED
    output_asset: Optional[Any] = None  # Image or Video
    text_manifest: Optional[TextManifest] = None
    validation_report: Optional[VldtnReport] = None
    qa_report: Optional['QAReport'] = None
    """
    🍢 qa_report: pending definition 7.16.26: must be structured
    to include progress tracking and metrics for user reference 
    throughout the pipeline
    """
    memory_updates: Optional[List[Dict[str, Any]]] = None
    logs: List[Dict[str, Any]] = field(default_factory=list)  ## {ts, stage, level, message, duration_ms?}
    errors: List[str] = field(default_factory=list)

@dataclass
class QAReport:
    overall_score: Optional[float] = None
    per_asset_instance_score: Dict[str, Dict[str, float]] = field(default_factory=dict)
    progress: Dict[str, Any] = field(default_factory=dict)
    metrics: Dict[str, Any] = field(default_factory=dict)
    recommendations: List[str] = field(default_factory=list)

def map_scores_to_asset_ids(
    text_manifest: TextManifest,
    qa_report: QAReport,
) -> dict[str, dict[str, float]]:
    scores = qa_report.per_asset_instance_score.get(text_manifest.asset_id, {})

    result = defaultdict(dict)
    for inst in text_manifest.instances:
        if inst.id in scores:
            result[text_manifest.asset_id][inst.id] = scores[inst.id]

    return dict(result)