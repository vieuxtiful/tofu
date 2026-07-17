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

# re:Core Data Structures

@dataclass
class BBox: ## B(x) = {b^t, b^c} (txt, char); bounding box
    x: int
    y: int
    width: int
    height: int

@dataclass
class Mask:
    polygon: 'Polygon | Mask'
    confidence: float

@dataclass
class InstText:
    id: str
    bounding_box: BBox
    segmentation_mask: Optional[Mask] = None
    text: Optional[str] = None
    language: Optional[str] = None
    characteristics: Optional['CharactText'] = None
    style_profile: Optional['StyleProfil'] = None
    background_profile: Optional['BgProfil'] = None
    confidence: Optional[float] = None
    detected_language: Optional[str] = None
    reading_order: Optional[int] = None
    source_asset_id: Optional[str] = None
    target_asset_id: Optional[str] = None

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

@dataclass
class BgProfil:
    semantic_label: Optional[str] = None  # "brick wall", "wood sign", etc.
    texture: Optional[str] = None
    gradients: Optional[List[str]] = None
    patterns: Optional[List[str]] = None

@dataclass
class TextManifest: ## loc task manifest via cicerone
    asset_id: str
    total_regions: int
    instances: List[InstText]
    src_lang: Optional[str] = None
    targ_lang: Optional[str] = None
    img_dim: Optional[tuple[int, int]] = None
    prcssng_time: Optional[float] = None

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

@dataclass
class PipelineResult:
    success: bool
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
    logs: List[str] = field(default_factory=list)

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
    scores = qa_report.per_instance_score or {}

    result = defaultdict(dict)
    for inst in text_manifest.instances:
        if inst.id in scores:
            result[text_manifest.asset_id][inst.id] = scores[inst.id]

    return dict(result)