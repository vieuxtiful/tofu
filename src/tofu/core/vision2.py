"""Typed contracts and inert runtime configuration for ToFU Vision 2."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Literal, TypedDict

SurvivalState = Literal["absent", "weak", "partial", "present", "unknown"]


class EvidenceSurvival(TypedDict):
    state: SurvivalState
    reasons: list[str]
    measured: dict[str, Any]
    schema: int
    policy_revision: str
    decision_eligible: bool
    ## Survival is a property of the image-channel pair, not of the scene:
    ## evidence that does not survive a crop may survive the full pipeline.
    ## Recorded so a survival verdict can never be read as channel-free.
    channel_id: str | None


class SurfaceObservationRef(TypedDict):
    surface_id: str
    observation_revision: str
    material_class: str
    calibration_status: Literal["unfitted", "calibrated"]
    substrate_trust: Literal["unmeasured", "insufficient", "measured"]
    missing_features: list[str]
    decision_weight: float


class Vision2State(str, Enum):
    OFF = "off"
    SHADOW = "shadow"
    GUIDED_REVIEW = "guided_review"
    AUTO_GT_RECOMMEND = "auto_gt_recommend"


class Vision2Decision(str, Enum):
    NOT_EVALUATED = "not_evaluated"
    SHADOW = "shadow"
    REVIEW_REQUIRED = "review_required"
    UNRESOLVABLE = "unresolvable"
    RECOMMENDED = "recommended"
    REJECTED = "rejected"


@dataclass(frozen=True)
class Vision2Config:
    state: Vision2State = Vision2State.OFF
    checkpoint_path: str | None = None
    fusion_calibration_path: str | None = None
    max_retrieval_candidates: int = 64
    max_retrieval_fonts: int = 5
    max_retrieval_regions: int = 24
    multi_view: bool = True
    component_alignment: bool = True
    alignment_top_k: int = 3
    scene_counterfactual: bool = False
    counterfactual_top_k: int = 3
    counterfactual_beam: int = 4

    @property
    def enabled(self) -> bool:
        return self.state is not Vision2State.OFF

    @property
    def may_surface_recommendations(self) -> bool:
        return self.state in {
            Vision2State.GUIDED_REVIEW,
            Vision2State.AUTO_GT_RECOMMEND,
        }


@dataclass
class ArtifactRevision:
    feature_schema: str
    model: str | None = None
    calibration: str | None = None
    corpus: str | None = None
    candidate_pool: str | None = None

    def mismatches(self, expected: ArtifactRevision) -> list[str]:
        fields = ("feature_schema", "model", "calibration", "corpus", "candidate_pool")
        return [
            name for name in fields
            if getattr(expected, name) is not None
            and getattr(self, name) != getattr(expected, name)
        ]


@dataclass
class GlyphCandidateEvidence:
    text: str
    rank: int
    support: float
    source: str
    margin: float | None = None
    contradiction: float | None = None
    ## This candidate's share of the pool's belief (layers/proof_distribution).
    ## Recorded alongside `support` rather than replacing it: `support` is the
    ## raw cosine the incumbent ranking and every fitted calibration use, and
    ## the two may not change places before a paired gate. A probability is
    ## also not a confidence -- the mass sums to one over a pool that earlier
    ## stages assembled, and says nothing if the true reading is not in it.
    probability: float | None = None
    lineage_candidate_id: str | None = None
    features: dict[str, Any] = field(default_factory=dict)


@dataclass
class GlyphMatchEvidence:
    schema: str
    decision: Vision2Decision = Vision2Decision.NOT_EVALUATED
    candidates: list[GlyphCandidateEvidence] = field(default_factory=list)
    evidence_survival_state: str | None = None
    calibrated_probability: float | None = None
    supported_domain: bool = False
    reason_codes: list[str] = field(default_factory=list)
    revisions: ArtifactRevision | None = None
    lineage_candidate_id: str | None = None
    diagnostics: dict[str, Any] = field(default_factory=dict)


@dataclass
class FusionEvidence:
    schema: str
    decision: Vision2Decision = Vision2Decision.NOT_EVALUATED
    calibrated_probability: float | None = None
    reconsideration_count: int = 0
    reason_codes: list[str] = field(default_factory=list)
    revisions: ArtifactRevision | None = None
    feature_values: dict[str, float | None] = field(default_factory=dict)
    missing_features: list[str] = field(default_factory=list)
    lineage_candidate_id: str | None = None
    reconsideration_candidate_id: str | None = None
    recommendation_revision: str | None = None

    def revision_mismatches(self, expected: ArtifactRevision) -> list[str]:
        if self.revisions is None:
            return ["revisions"]
        return self.revisions.mismatches(expected)


@dataclass
class CandidateProposal:
    candidate_id: str
    source_stage: str
    geometry: tuple[int, int, int, int]
    text: str | None = None
    confidence: float | None = None
    suppression_state: str = "active"
    rejection_reason: str | None = None
    parent_candidate_ids: list[str] = field(default_factory=list)
    scene_eligibility: dict[str, Any] | None = None
    final_eligible: bool = False
    glyph_match_evidence: GlyphMatchEvidence | None = None


@dataclass
class CandidateLedger:
    schema: str
    state: Vision2State
    proposals: list[CandidateProposal] = field(default_factory=list)
    total_candidates: int = 0
    retained_candidates: int = 0
    omitted_candidates: int = 0
    source_counts: dict[str, int] = field(default_factory=dict)
    suppression_counts: dict[str, int] = field(default_factory=dict)
    max_candidates: int = 0
