## types
## vieuxtiful
"""
This module defines the core data structures 
and enums used throughout ToFu. It provides 
the foundational types for assets, 
processing states, and validation results.
"""

from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any, Literal, Union, TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np
    from PIL import Image as PILImage
from enum import Enum
from collections import defaultdict
from pathlib import Path

if TYPE_CHECKING:
    import numpy as np
    from PIL import Image as PILImage

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

class CaptureMode(str, Enum):
    """How an asset is captured -- deliberately NOT LayerMode.

    `LayerMode.HYBRID` is the pipeline's pause/refinement protocol: a layer
    stops and waits for a human between stages.  This is a different axis
    entirely -- what the user asked detection to DO -- and no capture-mode
    code may branch on LayerMode.  Overloading the one enum would tie a
    detection choice to a pause checkpoint that has nothing to do with it.
    """
    AUTO = "auto"        ## detect everything; entered terms are extra evidence
    GUIDED = "guided"    ## locate the text the user supplied
    MANUAL = "manual"    ## no detection at all; the user draws

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

# re:Image Asset Type

## Images cross layer boundaries in several concrete forms: file paths
## (str / Path) from the server, PIL Images after loading, numpy arrays
## for CV operations, and occasionally raw bytes.  This Union replaces
## the ~91 `Any` annotations that preceded it so mypy can catch image
## type errors at layer boundaries.  Internal helpers that dispatch on
## the concrete type (e.g. utils.imaging.load_rgb) still accept Any at
## their entry point and narrow from there.
ImageLike = Union[str, Path, "PILImage.Image", "np.ndarray", bytes, bytearray]

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
class GarnishProfile:
    """Source-derived treatment applied only to newly rendered text pixels."""
    edge_blur_px: float = 0.0
    # Retained for backwards-compatible manifests. Feather strength is now
    # the direct control; a value of zero disables feathering.
    edge_smoothing: bool = False
    edge_smoothing_strength: float = 0.0
    erosion_px: float = 0.0
    dilation_px: float = 0.0
    grain_strength: float = 0.0
    gamma_shift: float = 1.0
    smudge_strength: float = 0.0
    smudge_angle_deg: float = 0.0
    source_confidence: float = 0.0

@dataclass
class GarnishRegion:
    """Editor-selected image-space subset of one rendered text instance."""
    id: str
    polygon: Polygon
    enabled: Optional[bool] = None
    profile: Optional[GarnishProfile] = None
    source: str = "manual"

@dataclass
class SceneRegion: ## candidate text-bearing surface from scene's pre-pass
    bbox: BBox
    semantic_label: str                    ## "panel" | "bordered_region" | "surface" | backend-specific
    confidence: float
    background_color: Optional[str] = None ## dominant color, hex
    border_detected: bool = False
    polygon: Optional[Polygon] = None
    texture: Optional[str] = None          ## region-interior classification: "flat" | "smooth_gradient" | "textured" -- the surface half of the scene/cleanse agreement gate
    material: Optional[str] = None         ## user-facing descriptor: "brick / masonry" | "painted sign" | "textured surface"
    garnish_profile: Optional[GarnishProfile] = None

@dataclass(frozen=True)
class OCRAssessmentPolicy:
    mode: Literal["off", "risk_based", "exhaustive"] = "risk_based"
    verifier: Literal["paddleocr"] = "paddleocr"
    require_verifier_for_auto_accept: bool = True
    max_region_proposals: int = 8
    calibration_revision: str = "ocr-v1"

@dataclass(frozen=True)
class InpaintAssessmentPolicy:
    mode: Literal["single", "multi"] = "multi"
    max_neural_candidates: int = 3
    retry_budget: int = 2
    require_residual_verification: bool = True
    calibration_revision: str = "inpaint-v1"

@dataclass
class ReconstructionProfile:
    material_class: str = "unknown"
    material_confidence: float = 0.0
    planar_confidence: float = 0.0
    periodic_texture_confidence: float = 0.0
    perspective_quad: Optional[Polygon] = None
    perspective_confidence: float = 0.0
    evidence: Dict[str, Any] = field(default_factory=dict)

@dataclass
class OCRObservation:
    """One raw OCR read from a single backend pass, preserved before union."""
    observation_id: str
    backend: str
    backend_revision: str
    pass_tag: str
    text: str
    raw_confidence: float
    calibrated_confidence: Optional[float] = None
    bbox: BBox = field(default_factory=lambda: BBox(0, 0, 0, 0))
    polygon: Optional[Polygon] = None
    language_hint: Optional[str] = None
    detected_script: Optional[str] = None
    runtime_ms: Optional[int] = None
    error: Optional[str] = None


@dataclass
class OCRHypothesisDecision:
    """Arbitration result for one region hypothesis formed from observations."""
    region_id: str
    member_observation_ids: List[str]
    selected_observation_id: Optional[str]
    selected_text: Optional[str]
    geometry_score: float
    transcription_score: float
    verification_state: str  # agree | disagree | no_text | unavailable | error
    verification_observation_id: Optional[str]
    auto_accepted: bool
    review_required: bool
    reason_codes: List[str]
    score_breakdown: Dict[str, Optional[float]]
    policy_revision: str


@dataclass
class InstText:
    id: str
    bounding_box: BBox
    # Render-only spatial position derived from style_profile.transform's
    # offset.  The captured bounding_box is immutable source geometry used by
    # Cicerone, Cleanse, XLIFF identity, and reading order.
    adjusted_bbox: Optional[BBox] = None
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
    excluded: bool = False                       ## user-removed from the workspace: source text left untouched (cleanse skips it, like dnt), never rendered/exported
    target_language: Optional[str] = None        ## per-region override of target language (None = use manifest default)
    glyph_fallback: Optional[bool] = None         ## True: scribe swapped the requested font for a codepoint-covering one
    tm_suggestion: Optional[Dict[str, Any]] = None  ## Memory lookup match: {target_text, score, method, source_asset_id, record_id}
    translation_attempts: List[Dict[str, Any]] = field(default_factory=list)
    translation_decision: Optional[Dict[str, Any]] = None
    translation_history: List[Dict[str, Any]] = field(default_factory=list)
    ocr_correction: Optional[Dict[str, Any]] = None  ## recognition_correct: {applied, original_text/candidate_text, corrected_text?, reason}
    ## Scene-surface attribution, recorded whether or not it was acted on:
    ## {surface_overlap, surface_label, confidence, outside_surfaces,
    ##  would_veto, vetoed}. `would_veto` is the eligibility gate's verdict,
    ## `vetoed` whether the deployment let it discard. Kept because scene
    ## non-membership is a useful NEGATIVE signal for review ordering even
    ## where it is a bad reason to discard -- see cicerone.SCENE_FILTER_VETOES.
    scene_eligibility: Optional[Dict[str, Any]] = None
    ## Review-priority feature vector, written by layers/ticket.py. Logged
    ## before any ranker exists so that reviewer outcomes, when the frontend
    ## starts reporting them, join against features already recorded rather
    ## than starting a collection period from zero. Purely descriptive: no
    ## stage may order, hide or discard on it.
    review_features: Optional[Dict[str, Any]] = None
    source_override: Optional[Dict[str, Any]] = None  ## durable applied-source attribution: {kind, text, icon, color, resource}; independent of the correction scratch slot
    recognition_history: Optional[List[Dict[str, Any]]] = None  ## immutable audit trail of engine candidates and accepted/rejected corrections
    ## The okara candidate this region was built from. In-memory only (not
    ## serialized): it exists so the lineage graph's final node can link to
    ## the chain BY IDENTITY instead of by matching coordinates, which the
    ## transforms reshape. Without it the graph documented the middle of the
    ## pipeline and went silent at the regions that actually ship.
    lineage_candidate_id: Optional[str] = None
    ocr_provenance: Optional[Dict[str, Any]] = None  ## multi-provider observations, arbitration and independent verification
    ocr_quality: Optional[Dict[str, Any]] = None  ## deterministic observability assessment; informs review-only OCR/Savor gating
    repair_provenance: Optional[Dict[str, Any]] = None  ## cleanse provider, confidence gate, fallback and review evidence
    reconstruction_profile: Optional[ReconstructionProfile] = None
    font_match: Optional[Dict[str, Any]] = None  ## evidence-gated visual font identification + installed/commercial alternatives; never silently overrides a user font choice
    semantic_assignment: Optional[Dict[str, Any]] = None  ## Basil's explicit target-span-to-immutable-region assignment provenance
    garnish_override: Optional[GarnishProfile] = None
    garnish_enabled: Optional[bool] = None  ## None = inherit enabled; False bypasses treatment without discarding the override
    garnish_scope: str = "whole_selection"  ## whole_selection | per_region; explicit so masks never change whole-selection semantics
    garnish_regions: List[GarnishRegion] = field(default_factory=list)
    resolved_font_family: Optional[str] = None    ## what "auto" (style_profile.font_family=None) currently resolves to — scribe.resolve_auto_font()'s answer, for preview/display only; never itself passed as a render override
    resolved_synthetic_italic: bool = False       ## True when render() will SHEAR an upright face (no real italic sibling exists) rather than load one — scribe.resolve_face()'s second return value, so a preview can reproduce the same choice instead of guessing from a subfamily string

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
    underline_offset: Optional[float] = None  # px below the detected glyph bottom; None = detected default
    underline_width: Optional[float] = None   # px stroke; None = detected default
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
    stroke_position: Optional[str] = None     # "outer" | "center" | "inner"
    target_orientation: Optional[str] = None  # "horizontal" | "vertical"
    word_order: Optional[str] = None          # "ltr" | "rtl" for vertical word columns
    # {skew_x, skew_y, skew_anchor, arc, preset, amount, scale_x, scale_y,
    #  offset_x, offset_y, rotation, wrap_text, locked_fields, quad}
    #
    # `quad` is the projective corner set: four [x, y] pairs ordered
    # top-left, top-right, bottom-right, bottom-left, NORMALISED to the
    # region's bounding box -- so [[0,0],[1,0],[1,1],[0,1]] is the identity
    # and a region that moves or resizes carries its perspective with it.
    # Values outside 0-1 are legal; that is what lets a corner be pulled
    # beyond the box.
    #
    # It exists because skew_x/skew_y cannot express perspective. Those are
    # a shear, and shear is affine: it keeps opposite edges parallel and
    # equal. A sign photographed at an angle has CONVERGING edges, which no
    # combination of shear, scale and rotation reproduces -- six degrees of
    # freedom against the eight a homography needs. When a non-identity quad
    # is present it replaces the affine stage; skew_x/skew_y keep working
    # unchanged for every manifest that has no quad.
    transform: Optional[Dict[str, Any]] = None

@dataclass
class BgProfil:
    semantic_label: Optional[str] = None  # "brick wall", "wood sign", etc.
    texture: Optional[str] = None
    material: Optional[str] = None          # presentational material descriptor; never drives Cleanse routing
    gradients: Optional[List[str]] = None
    patterns: Optional[List[str]] = None
    dominant_color: Optional[str] = None  # hex; cleanse fill / scribe contrast hint
    surface_texture: Optional[str] = None  # containing SceneRegion.texture
    cleanse_strategy: Optional[str] = None # "flat" | "smooth_gradient" | "telea"

@dataclass
class SemanticTextUnit:
    """A source reading unit registered against immutable Cicerone regions.

    ``region_ids`` preserves source visual cube order. The individual IDs and
    boxes never change; Basil can route a target semantic block into a
    different existing cube before Scribe renders it.
    """
    id: str
    region_ids: List[str]
    source_text: str
    bbox: BBox
    ## Durable identity, minted ONCE and never renumbered.  `id` stays "uN"
    ## for display and legacy compatibility, but it is positional -- Basil
    ## reassigns it on every re-registration -- so it can never be an
    ## interchange key.  A translator's returned file is matched on
    ## plate_uid; matching on "u2" would silently retarget a different plate
    ## after any region insert or reorder.  Deliberately random rather than
    ## content-derived: editing a plate's source or membership is a REVISION
    ## of that plate, not a new one.
    plate_uid: str = ""
    ## Reading-order position shown as "Plate 2".  Recomputed freely; never
    ## persisted as a reference and never used to resolve an import.
    display_number: Optional[int] = None
    ## derived -- Basil may regenerate this plate's membership.
    ## guided/user -- membership is PINNED: an accepted proposal or an
    ## explicit user edit, which re-registration must never silently undo.
    origin: str = "derived"
    ## Revision fingerprint over ordered region_ids + normalized source +
    ## schema. Import compares it against the exported file's copy: same UID
    ## with a different hash means the plate moved on since it was sent out,
    ## which must block rather than overwrite.
    membership_hash: str = ""
    entity_type: str = "unknown"
    confidence: float = 0.0
    analysis_provider: str = "deterministic_layout"
    semantic_roles: Dict[str, str] = field(default_factory=dict)
    review_required: bool = True
    substitution: Optional[Dict[str, Any]] = None
    pairing: Optional[Dict[str, Any]] = None  ## Basil's src/targ typology verdict: {verdict: unnecessary|possible|unknown, reasons, features} -- whether cross-region rearrangement is linguistically possible at all
    suggestion: Optional[Dict[str, Any]] = None  ## target phrase ordered by the target's own syntax from the members' existing target_text: {target_text, region_order, basis, coverage}
    ocr_repair: Optional[Dict[str, Any]] = None  ## proposed (never auto-applied) source correction when a gazetteer entity spans fragmented regions: {read, proposed, spans, similarity, evidence, accepted}

@dataclass
class GuidedAtom:
    """One indivisible piece of a Block, in the order it was written.

    Position is explicit rather than implied by list index so that duplicates
    stay distinguishable: "PARIS PARIS" is two atoms that must remain two
    separate things to locate, not one term seen twice.
    """
    id: str
    text: str
    position: int
    kind: str                       ## word | phrase | numeric | punctuation
    script: Optional[str] = None    ## ISO 15924, via layers/palate.py
    direction: Optional[str] = None ## ltr | rtl

@dataclass
class GuidedBlock:
    """Source text the user asked ToFU to find, kept exactly as typed.

    `raw_text` is never normalised away: whitespace and case are evidence
    about how the text appears in the asset, and the moment they are
    discarded a Block can no longer be matched against what was actually
    printed.  `atoms` is the derived view.
    """
    id: str
    raw_text: str
    normalized_text: str
    position: int
    source_language: Optional[str] = None
    scope: str = "asset"            ## asset | project
    find_all: bool = False
    atoms: List['GuidedAtom'] = field(default_factory=list)
    language_assessment: Optional[Dict[str, Any]] = None
    detection_assessment: Optional[Dict[str, Any]] = None

@dataclass
class ExportReadiness:
    """What the expeditor saw at the pass before a translation file left.

    ``state`` is the single verdict a caller acts on:

      ready    -- nothing to report; the file can be built
      review   -- advisory findings only; export is still permitted
      stale    -- a plate no longer describes the regions it names
      blocked  -- a structural contradiction; no file may be built

    ``diagnostics`` carries one record per finding, each
    ``{code, severity, message, plate_ids, region_ids}``. Severity is
    ``blocking`` or ``advisory`` -- ``state`` is derived from them, never
    set independently, so a caller cannot see ``ready`` alongside a
    blocking record.
    """
    state: str
    diagnostics: List[Dict[str, Any]] = field(default_factory=list)
    counts: Dict[str, int] = field(default_factory=dict)

@dataclass
class TextManifest: ## loc task manifest via cicerone
    asset_id: str
    total_regions: int
    instances: List[InstText]
    src_lang: Optional[str] = None
    targ_lang: Optional[str] = None
    img_dim: Optional[tuple[int, int]] = None
    scene_regions: List[SceneRegion] = field(default_factory=list)
    semantic_units: List[SemanticTextUnit] = field(default_factory=list)
    ## The Blocks the user asked ToFU to locate, in the order they were
    ## entered.  Persisted with the manifest rather than recomputed from the
    ## Ground Truth field: a Block carries its own language assessment and
    ## detection outcome, and re-deriving them on load would silently discard
    ## a warning the user has already seen and acknowledged.  Empty for Auto
    ## and Manual captures, so an existing manifest is unaffected.
    guided_blocks: List['GuidedBlock'] = field(default_factory=list)
    asset_class: Optional[str] = None
    asset_classification: Optional[Dict[str, Any]] = None
    ## Append-only DAG of every proposal and every merge/prune between the
    ## detector's raw output and this manifest (layers/okara.py). Present so
    ## a region absorbed by a merge stays retrievable with its ORIGINAL
    ## geometry: measured on la-bastille, CRAFT's `RUE` arrives at IoU 0.735
    ## and ships at 0.215 inside a box 4.65x too large, and without lineage
    ## the only record of the good box is gone.
    candidate_lineage: Optional[Dict[str, Any]] = None
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

## CameraTrack lived here: a RenderParams animated along a trajectory. it was
## never used. the video compositor needs the opposite direction — "given this
## frame, what do I draw" — which is tofu.video.plan.ResolvedFramePlan.

@dataclass
class VldtnClass:
    severity: VldtnSeverity
    code: str
    message: str
    suggestion: Optional[str] = None
    region_id: Optional[str] = None
    ## every region this one issue applies to. a language-level finding
    ## (script support, render quality) is true of the whole target
    ## language, not of one box — emitting it once per instance turned a
    ## single fact into N identical warnings in the preflight panel.
    ## region_id stays as the primary/first anchor for existing consumers.
    region_ids: List[str] = field(default_factory=list)

@dataclass
class VldtnInsight:
    """Non-blocking, evidence-backed advice surfaced by ToFU preflight.

    Insights deliberately sit apart from validation issues: a visual font
    retrieval result or a licensed style reference should guide an editor,
    never fail a render or overwrite their chosen typeface.
    """
    key: str
    kind: str                           # "font_substitute" | "style_reference" | provider-specific extension
    title: str
    detail: str
    severity: str = "info"              # "info" | "review" | "warning"
    region_id: Optional[str] = None
    region_ids: List[str] = field(default_factory=list)
    confidence: Optional[float] = None
    visual_score: Optional[float] = None
    family: Optional[str] = None
    subfamily: Optional[str] = None
    font_path: Optional[str] = None
    license: Optional[str] = None
    foundry: Optional[str] = None
    url: Optional[str] = None
    source: Optional[str] = None

@dataclass
class VldtnReport:
    passed: bool
    issues: List[VldtnClass] = field(default_factory=list)
    scrpt_spprt: Dict[str, ScrptSpprt] = field(default_factory=dict)
    glyph_segmentation_score: Optional[float] = None
    render_quality_score: Optional[float] = None
    render_quality_evidence: Dict[str, float] = field(default_factory=dict)
    expansion_fit: Dict[str, float] = field(default_factory=dict)  ## region_id -> predicted_width / bbox_width
    suggested_actions: List[str] = field(default_factory=list)
    insights: List[VldtnInsight] = field(default_factory=list)

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
    ocr_assessment: OCRAssessmentPolicy = field(default_factory=OCRAssessmentPolicy)
    inpaint_assessment: InpaintAssessmentPolicy = field(default_factory=InpaintAssessmentPolicy)

@dataclass
class PipelineResult:
    success: bool
    status: PrcStatus = PrcStatus.COMPLETED
    output_asset: Optional[Any] = None  # Image or Video
    text_manifest: Optional[TextManifest] = None
    validation_report: Optional[VldtnReport] = None
    verification_report: Optional['VerificationReport'] = None
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


# VerificationReport is the canonical, evidence-first Verify contract.  The
# older QAReport remains above as a compatibility view while its scorers are
# migrated into the six component scores in later phases.
VerificationStatus = Literal["pass", "review", "fail"]
InventoryStatus = Literal[
    "rendered", "missing", "untranslated", "excluded", "do_not_translate", "extra"
]


@dataclass
class VerificationEvidence:
    text: Optional[str] = None
    bounds: Optional[BBox] = None
    language: Optional[str] = None
    #: joined ISO 15924 summary, e.g. "Latn" or "Hani+Kana".  A string is not
    #: "in a script" -- its characters are (UAX #24) -- so this is a summary
    #: of `scripts`, never a single majority winner.
    script: Optional[str] = None
    #: every ISO 15924 script actually present, Common excluded
    scripts: List[str] = field(default_factory=list)
    #: character counts per script, the evidence behind `scripts`
    script_counts: Dict[str, int] = field(default_factory=dict)
    #: more than one real script present (correct for Japanese and Korean)
    is_mixed_script: bool = False
    #: "Hans" | "Hant" | "undetermined" for Chinese Han text; None otherwise.
    #: Unicode unifies both orthographies as Hani, so this comes from the
    #: national charsets rather than from any Unicode property.
    han_variant: Optional[str] = None
    direction: Optional[Literal["ltr", "rtl", "ttb"]] = None
    font_family: Optional[str] = None
    asset_class: Optional[str] = None
    clean_render_present: bool = False


@dataclass
class VerificationRegion:
    region_id: str
    inventory_status: InventoryStatus
    source_evidence: VerificationEvidence
    target_evidence: VerificationEvidence
    checks: Dict[str, Any] = field(default_factory=dict)
    scores: Dict[str, Optional[float]] = field(default_factory=dict)
    flags: List[str] = field(default_factory=list)
    confidence: Optional[float] = None
    recommended_action: Optional[str] = None
    overall_score: Optional[float] = None
    status: VerificationStatus = "review"


@dataclass
class VerificationProject:
    overall_status: VerificationStatus = "review"
    overall_score: Optional[float] = None
    component_scores: Dict[str, Optional[float]] = field(default_factory=lambda: {
        "coverage": None,
        "content_integrity": None,
        "spatial_fit": None,
        "typographic_intent": None,
        "script_rendering_validity": None,
        "contextual_fit": None,
    })
    summary_flags: List[str] = field(default_factory=list)
    region_totals: Dict[str, int] = field(default_factory=dict)
    summary: Optional[str] = None
    review_order: List[str] = field(default_factory=list)
    ## non-negotiable preconditions that were not met (roadmap P1.29). Each is
    ## {code, severity, detail}. Distinct from summary_flags: a flag says what
    ## was observed, a gate says why the verdict could not be green. Empty on a
    ## clean run. Defaults to empty so manifests written before gates existed
    ## still load -- an old report legitimately has no gate evidence, which is
    ## not the same as having passed them.
    coverage_gates: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class VerificationVisualFlag:
    region_id: str
    bounds: BBox
    severity: VerificationStatus
    codes: List[str] = field(default_factory=list)
    color: str = "#f59e0b"
    label: Optional[str] = None


@dataclass
class VerificationReport:
    project: VerificationProject
    regions: List[VerificationRegion] = field(default_factory=list)
    visual_flags: List[VerificationVisualFlag] = field(default_factory=list)
    run_metadata: Dict[str, Any] = field(default_factory=dict)

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
