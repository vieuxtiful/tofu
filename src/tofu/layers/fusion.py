"""Fail-closed, calibrated late fusion for ToFU Vision 2."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from tofu.core.types import InstText, TextManifest
from tofu.core.vision2 import (
    ArtifactRevision,
    FusionEvidence,
    GlyphMatchEvidence,
    Vision2Config,
    Vision2Decision,
    Vision2State,
)

## Bumped from v1 on 2026-08-17. The feature vector changed twice and the
## schema is what stops a calibration fitted on the old one from being
## applied to the new: distribution features were added, and the alignment
## feature was REPAIRED -- `component_alignment_support` named a key
## `proof_alignment.align()` never produced, so it was permanently missing
## and the transport plan reached no decision in the entire life of v1. An
## artifact declaring v1 now fails to load, which is the fail-closed
## outcome rather than a silent reinterpretation of its coefficients.
FEATURE_SCHEMA = "vision2-fusion-features-v2"
CALIBRATION_SCHEMA = "vision2-fusion-calibration-v1"


@dataclass(frozen=True)
class FusionCalibration:
    revision: ArtifactRevision
    retrieval_feature_schema: str
    coefficients: dict[str, float]
    intercept: float
    recommendation_threshold: float
    rejection_threshold: float
    minimum_margin: float
    maximum_contradiction: float


def load_calibration(path_value: str | None) -> tuple[FusionCalibration | None, str | None]:
    if not path_value:
        return None, "fusion_calibration_not_configured"
    try:
        data = json.loads(Path(path_value).read_text(encoding="utf-8"))
        if data.get("schema") != CALIBRATION_SCHEMA:
            return None, "fusion_calibration_schema_mismatch"
        revision = ArtifactRevision(**data["revision"])
        calibration = FusionCalibration(
            revision=revision,
            retrieval_feature_schema=str(data["retrieval_feature_schema"]),
            coefficients={str(k): float(v) for k, v in data["coefficients"].items()},
            intercept=float(data["intercept"]),
            recommendation_threshold=float(data["recommendation_threshold"]),
            rejection_threshold=float(data.get("rejection_threshold", 0.0)),
            minimum_margin=float(data["minimum_margin"]),
            maximum_contradiction=float(data["maximum_contradiction"]),
        )
        if revision.feature_schema != FEATURE_SCHEMA:
            return None, "fusion_feature_schema_mismatch"
        if not 0.0 <= calibration.rejection_threshold < calibration.recommendation_threshold <= 1.0:
            return None, "fusion_calibration_threshold_invalid"
        return calibration, None
    except Exception:
        return None, "fusion_calibration_unavailable"


def _features(
    glyph: GlyphMatchEvidence,
    surface: dict[str, Any] | None,
) -> tuple[dict[str, float | None], list[str]]:
    top = glyph.candidates[0] if glyph.candidates else None
    values: dict[str, float | None] = {
        "retrieval_support": top.support if top else None,
        "retrieval_margin": top.margin if top else None,
        "retrieval_contradiction": top.contradiction if top else None,
        "transformation_variance": (
            top.features.get("transformation_variance") if top else None
        ),
        # Three measured transport quantities rather than one invented
        # composite. `alignment_cost` is lower-is-better and a fitted
        # coefficient can be negative; folding it into a "support" would
        # have meant choosing a combining rule before anything had measured
        # which of the three carries the signal.
        "component_alignment_cost": None,
        "component_alignment_matched_mass": None,
        "component_alignment_unmatched": None,
        # Observational Scene remains present for ablation and explicitly
        # absent from v1 fitted decisions until a later artifact names it.
        "scene_decision_weight": float((surface or {}).get("decision_weight", 0.0)),
    }
    if top:
        alignment = top.features.get("component_alignment") or {}
        values["component_alignment_cost"] = alignment.get("alignment_cost")
        values["component_alignment_matched_mass"] = alignment.get("matched_mass")
        values["component_alignment_unmatched"] = alignment.get(
            "observed_unmatched_mass"
        )
    ## Distribution shape, declared and unweighted. The top-2 margin cannot
    ## distinguish a narrow lead over a pool that is otherwise flat from a
    ## narrow lead inside a tight top-three, and these say which it is.
    ## Present in the schema so a later calibration can be fitted against
    ## them; they decide nothing until one names them in its coefficients,
    ## exactly as observational Scene entered.
    distribution = (glyph.diagnostics or {}).get("distribution") or {}
    values["retrieval_top1_probability"] = distribution.get("top1_probability")
    values["retrieval_entropy"] = distribution.get("entropy")
    values["retrieval_attempt_divergence"] = distribution.get("max_attempt_divergence")
    missing = [name for name, value in values.items() if value is None]
    return values, missing


def _compatible(
    glyph: GlyphMatchEvidence,
    calibration: FusionCalibration,
) -> list[str]:
    if glyph.revisions is None:
        return ["retrieval_revisions_missing"]
    mismatches = []
    if glyph.revisions.feature_schema != calibration.retrieval_feature_schema:
        mismatches.append("retrieval_feature_schema")
    for name in ("model", "candidate_pool"):
        expected = getattr(calibration.revision, name)
        if expected is not None and getattr(glyph.revisions, name) != expected:
            mismatches.append(name)
    return mismatches


def evaluate(
    inst: InstText,
    state: Vision2State,
    calibration: FusionCalibration | None,
    calibration_error: str | None = None,
) -> FusionEvidence:
    glyph = inst.glyph_match_evidence
    survival = inst.evidence_survival or {}
    result = FusionEvidence(
        schema=FEATURE_SCHEMA,
        decision=Vision2Decision.SHADOW if state is Vision2State.SHADOW else Vision2Decision.REVIEW_REQUIRED,
        lineage_candidate_id=inst.lineage_candidate_id,
    )
    if glyph is None:
        result.reason_codes.append("glyph_evidence_missing")
        return result
    result.feature_values, result.missing_features = _features(glyph, inst.surface_observation)
    survival_state = survival.get("state", "unknown")
    if survival_state in {"absent", "weak"}:
        result.decision = Vision2Decision.UNRESOLVABLE
        result.reason_codes.append(f"evidence_survival_{survival_state}")
        return result
    if state is Vision2State.SHADOW:
        result.reason_codes.append("shadow_only")
        return result
    if survival_state in {"partial", "unknown"}:
        result.reason_codes.append(f"evidence_survival_{survival_state}")
        return result
    if not glyph.supported_domain or len(glyph.candidates) < 2:
        result.reason_codes.append("retrieval_domain_unsupported")
        return result
    if calibration is None:
        result.reason_codes.append(calibration_error or "fusion_calibration_missing")
        return result
    mismatches = _compatible(glyph, calibration)
    if mismatches:
        result.reason_codes.extend(f"artifact_mismatch:{name}" for name in mismatches)
        return result
    required_missing = [name for name in calibration.coefficients if result.feature_values.get(name) is None]
    if required_missing:
        result.reason_codes.extend(f"feature_missing:{name}" for name in required_missing)
        return result
    logit = calibration.intercept + sum(
        weight * float(result.feature_values[name])
        for name, weight in calibration.coefficients.items()
    )
    probability = 1.0 / (1.0 + math.exp(-max(-40.0, min(40.0, logit))))
    result.calibrated_probability = round(probability, 8)
    result.revisions = calibration.revision
    margin = glyph.candidates[0].margin
    contradiction = glyph.candidates[0].contradiction
    if margin is None or margin < calibration.minimum_margin:
        result.reason_codes.append("margin_below_calibrated_minimum")
    elif contradiction is not None and contradiction > calibration.maximum_contradiction:
        result.reason_codes.append("contradiction_above_calibrated_maximum")
    elif probability >= calibration.recommendation_threshold:
        result.decision = Vision2Decision.RECOMMENDED
        result.reason_codes.append("calibrated_recommendation")
    elif probability <= calibration.rejection_threshold:
        result.decision = Vision2Decision.REJECTED
        result.reason_codes.append("calibrated_rejection")
    else:
        result.reason_codes.append("calibrated_review_band")
    return result


def assess_manifest(manifest: TextManifest, config: Vision2Config) -> int:
    if not config.enabled:
        return 0
    calibration, error = load_calibration(config.fusion_calibration_path)
    for inst in manifest.instances:
        inst.fusion_evidence = evaluate(inst, config.state, calibration, error)
        glyph = inst.glyph_match_evidence
        candidate = glyph.candidates[0].text if glyph and glyph.candidates else None
        payload = json.dumps({
            "fusion": {
                "schema": inst.fusion_evidence.schema,
                "decision": inst.fusion_evidence.decision.value,
                "probability": inst.fusion_evidence.calibrated_probability,
                "revisions": (
                    inst.fusion_evidence.revisions.__dict__
                    if inst.fusion_evidence.revisions else None
                ),
                "lineage": inst.fusion_evidence.lineage_candidate_id,
            },
            "candidate": candidate,
        }, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        inst.fusion_evidence.recommendation_revision = hashlib.sha256(
            payload.encode("utf-8")
        ).hexdigest()
    # Reconsideration is provenance-only here: it identifies at most one
    # retained proposal for review and never promotes geometry or rewrites OCR.
    if config.state is not Vision2State.SHADOW and calibration is not None:
        ledger = manifest.vision2_candidate_ledger
        eligible = [
            proposal for proposal in (ledger.proposals if ledger else [])
            if not proposal.final_eligible
            and proposal.glyph_match_evidence is not None
            and proposal.glyph_match_evidence.supported_domain
        ]
        eligible.sort(key=lambda proposal: (
            -proposal.glyph_match_evidence.candidates[0].support
            if proposal.glyph_match_evidence.candidates else 0.0,
            proposal.candidate_id,
        ))
        if eligible and manifest.instances:
            incumbent = manifest.instances[0]
            target = incumbent.fusion_evidence
            proposal = eligible[0]
            proposal_inst = InstText(
                id=f"reconsideration:{proposal.candidate_id}",
                bounding_box=incumbent.bounding_box,
                lineage_candidate_id=proposal.candidate_id,
                evidence_survival=incumbent.evidence_survival,
                glyph_match_evidence=proposal.glyph_match_evidence,
                surface_observation=incumbent.surface_observation,
            )
            proposal_result = evaluate(
                proposal_inst, config.state, calibration, error,
            )
            if (
                target is not None
                and target.reconsideration_count == 0
                and proposal_result.decision is Vision2Decision.RECOMMENDED
            ):
                target.reconsideration_count = 1
                target.reconsideration_candidate_id = proposal.candidate_id
                target.reason_codes.append("reconsideration_proposed_for_review")
    return len(manifest.instances)


def manifest_metrics(manifest: TextManifest) -> dict[str, Any]:
    decisions: dict[str, int] = {}
    outcomes: dict[str, int] = {}
    fallbacks: dict[str, int] = {}
    revisions: dict[str, int] = {}
    evaluated = 0
    for inst in manifest.instances:
        evidence = inst.fusion_evidence
        if evidence is not None:
            evaluated += 1
            decision = evidence.decision.value
            decisions[decision] = decisions.get(decision, 0) + 1
            revision = evidence.revisions.calibration if evidence.revisions else "unfitted"
            revisions[str(revision)] = revisions.get(str(revision), 0) + 1
            for reason in evidence.reason_codes:
                if "unavailable" in reason or "missing" in reason or "mismatch" in reason:
                    fallbacks[reason] = fallbacks.get(reason, 0) + 1
        for record in inst.vision2_decision_history:
            action = str(record.get("action", "unknown"))
            outcomes[action] = outcomes.get(action, 0) + 1
    total = len(manifest.instances)
    ## Channel coverage sits beside decision coverage on purpose: a decision
    ## rate that looks stable while its channel mix moves is not the same
    ## measurement two weeks running.
    from tofu.layers.flight import manifest_channels
    return {
        "regions": total,
        "evaluated": evaluated,
        "coverage": evaluated / total if total else 0.0,
        "channels": manifest_channels(manifest.instances),
        "decisions": decisions,
        "reviewer_outcomes": outcomes,
        "fallback_reasons": fallbacks,
        "calibration_revisions": revisions,
        "revision_skew": len(revisions) > 1,
        "precision": None,
        "precision_sample_size": 0,
    }
