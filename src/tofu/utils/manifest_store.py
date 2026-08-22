## 🍢 manifest_store — persistence for TextManifest
## vieuxtiful
"""
JSON-based persistence for TextManifest objects so the frontend can
auto-save and restore the interactive preview state across sessions.

Each manifest is stored as uploads/{asset_id}.manifest.json.

FIELD OWNERSHIP, and why it is written down rather than assumed.

`PUT /api/manifest/{id}` replaces the whole document, and the frontend
rebuilds that document from a TypeScript interface that knows only about the
fields the UI edits.  So any field this serializer carries but the frontend
does not is deleted by the next autosave -- silently, 1.5 seconds after the
user drags a box.  That is not hypothetical: `guided_blocks` round-tripped
here perfectly and was erased on every save anyway, and `candidate_lineage`
had no key at all, so the okara graph never once reached disk.

Hence SERVER_OWNED_FIELDS.  A field listed there is preserved by the server
when an incoming payload does not MENTION it (see `merge_server_owned` and
`server.main.put_manifest`).  Absence and emptiness are different: an
explicit `[]` or `null` still clears, because a caller that names a field is
asserting something about it.

The list is the contract.  A new server-computed field has to be added to it
or it inherits the same defect, and the round-trip test parametrises over it
so the omission fails a test rather than losing a user's work.
"""

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, Optional

from tofu.core.types import (
    TextManifest, InstText, BBox, Mask, AssetType,
    StyleProfil, BgProfil, CharactText, SceneRegion, SemanticTextUnit, GarnishProfile, GarnishRegion,
    ReconstructionProfile,
)
from tofu.core.vision2 import (
    ArtifactRevision, CandidateLedger, CandidateProposal, FusionEvidence,
    GlyphCandidateEvidence, GlyphMatchEvidence, Vision2Decision, Vision2State,
)

## Fields this serializer persists that NO frontend payload is expected to
## carry.  Everything here is computed or curated server-side.
SERVER_OWNED_FIELDS = (
    "guided_blocks",      ## Guided Blocks + their detection_assessment
    "candidate_lineage",  ## okara's append-only proposal DAG
    "vision2_candidate_ledger",
)

## Per-INSTANCE fields with the same problem, listed for the same reason.
## These live inside `instances`, so the manifest-level merge above cannot
## protect them -- a client that round-trips a region without them drops
## them. Recorded here so the next provenance field has somewhere obvious to
## be declared, and so the serializer audit has something to check against.
SERVER_OWNED_INSTANCE_FIELDS = (
    "scene_eligibility",     ## the scene filter's verdict, veto on or off
    "review_features",       ## layers/ticket.py's per-candidate ticket
    "lineage_candidate_id",  ## the join from a shipped region to its okara node
    "evidence_survival",     ## layers/decant.py's survival verdict
    "channel",               ## layers/flight.py's measurement-channel record
    "channel_id",            ## its identifier, denormalized for joins
    "glyph_match_evidence",
    "fusion_evidence",
    "surface_observation",
    "vision2_decision_history",
)


def _artifact_revision_from_dict(data: Any) -> Optional[ArtifactRevision]:
    if not isinstance(data, dict) or not data.get("feature_schema"):
        return None
    return ArtifactRevision(
        feature_schema=str(data["feature_schema"]), model=data.get("model"),
        calibration=data.get("calibration"), corpus=data.get("corpus"),
        candidate_pool=data.get("candidate_pool"),
    )


def _vision2_decision(value: Any) -> Vision2Decision:
    if isinstance(value, Vision2Decision):
        return value
    try:
        return Vision2Decision(str(value))
    except ValueError:
        return Vision2Decision.NOT_EVALUATED


def _glyph_match_from_dict(data: Any) -> Optional[GlyphMatchEvidence]:
    if not isinstance(data, dict) or not data.get("schema"):
        return None
    candidates = []
    for item in data.get("candidates") or []:
        if not isinstance(item, dict) or "text" not in item:
            continue
        candidates.append(GlyphCandidateEvidence(
            text=str(item["text"]), rank=int(item.get("rank", 0)),
            support=float(item.get("support", 0.0)), source=str(item.get("source", "unknown")),
            margin=item.get("margin"), contradiction=item.get("contradiction"),
            probability=item.get("probability"),
            lineage_candidate_id=item.get("lineage_candidate_id"),
            features=dict(item.get("features") or {}),
        ))
    return GlyphMatchEvidence(
        schema=str(data["schema"]), decision=_vision2_decision(data.get("decision")),
        candidates=candidates, evidence_survival_state=data.get("evidence_survival_state"),
        calibrated_probability=data.get("calibrated_probability"),
        supported_domain=bool(data.get("supported_domain", False)),
        reason_codes=[str(code) for code in data.get("reason_codes") or []],
        revisions=_artifact_revision_from_dict(data.get("revisions")),
        lineage_candidate_id=data.get("lineage_candidate_id"),
        diagnostics=dict(data.get("diagnostics") or {}),
    )


def _fusion_from_dict(data: Any) -> Optional[FusionEvidence]:
    if not isinstance(data, dict) or not data.get("schema"):
        return None
    return FusionEvidence(
        schema=str(data["schema"]), decision=_vision2_decision(data.get("decision")),
        calibrated_probability=data.get("calibrated_probability"),
        reconsideration_count=int(data.get("reconsideration_count", 0)),
        reason_codes=[str(code) for code in data.get("reason_codes") or []],
        revisions=_artifact_revision_from_dict(data.get("revisions")),
        feature_values=dict(data.get("feature_values") or {}),
        missing_features=[str(name) for name in data.get("missing_features") or []],
        lineage_candidate_id=data.get("lineage_candidate_id"),
        reconsideration_candidate_id=data.get("reconsideration_candidate_id"),
        recommendation_revision=data.get("recommendation_revision"),
    )


def _typed_evidence_to_dict(value: Any) -> Optional[dict]:
    if value is None:
        return None
    if isinstance(value, dict):
        return dict(value)
    return asdict(value)


def _candidate_ledger_from_dict(data: Any) -> Optional[CandidateLedger]:
    if not isinstance(data, dict) or not data.get("schema"):
        return None
    raw_state = data.get("state", "off")
    if isinstance(raw_state, Vision2State):
        state = raw_state
    else:
        try:
            state = Vision2State(str(raw_state))
        except ValueError:
            state = Vision2State.OFF
    proposals = []
    for item in data.get("proposals") or []:
        if not isinstance(item, dict) or not item.get("candidate_id"):
            continue
        geometry = item.get("geometry") or (0, 0, 0, 0)
        if len(geometry) != 4:
            continue
        proposals.append(CandidateProposal(
            candidate_id=str(item["candidate_id"]),
            source_stage=str(item.get("source_stage", "unknown")),
            geometry=tuple(int(value) for value in geometry),
            text=item.get("text"), confidence=item.get("confidence"),
            suppression_state=str(item.get("suppression_state", "active")),
            rejection_reason=item.get("rejection_reason"),
            parent_candidate_ids=[str(value) for value in item.get("parent_candidate_ids") or []],
            scene_eligibility=(
                dict(item["scene_eligibility"])
                if isinstance(item.get("scene_eligibility"), dict) else None
            ),
            final_eligible=bool(item.get("final_eligible", False)),
            glyph_match_evidence=_glyph_match_from_dict(item.get("glyph_match_evidence")),
        ))
    return CandidateLedger(
        schema=str(data["schema"]), state=state, proposals=proposals,
        total_candidates=int(data.get("total_candidates", len(proposals))),
        retained_candidates=int(data.get("retained_candidates", len(proposals))),
        omitted_candidates=int(data.get("omitted_candidates", 0)),
        source_counts={str(key): int(value) for key, value in (data.get("source_counts") or {}).items()},
        suppression_counts={str(key): int(value) for key, value in (data.get("suppression_counts") or {}).items()},
        max_candidates=int(data.get("max_candidates", 0)),
    )


def _garnish_to_dict(profile: Optional[GarnishProfile]) -> Optional[dict]:
    return None if profile is None else {
        "edge_blur_px": profile.edge_blur_px, "edge_smoothing": profile.edge_smoothing,
        "edge_smoothing_strength": profile.edge_smoothing_strength,
        "erosion_px": profile.erosion_px,
        "dilation_px": profile.dilation_px, "grain_strength": profile.grain_strength,
        "gamma_shift": profile.gamma_shift, "smudge_strength": profile.smudge_strength,
        "smudge_angle_deg": profile.smudge_angle_deg, "source_confidence": profile.source_confidence,
    }


def _garnish_from_dict(data: Optional[dict]) -> Optional[GarnishProfile]:
    if not isinstance(data, dict): return None
    legacy_smoothing = bool(data.get("edge_smoothing", False))
    values = {key: float(data.get(key, default)) for key, default in {
        "edge_blur_px": 0, "edge_smoothing_strength": (.5 if legacy_smoothing else 0), "erosion_px": 0, "dilation_px": 0, "grain_strength": 0,
        "gamma_shift": 1, "smudge_strength": 0, "smudge_angle_deg": 0, "source_confidence": 0,
    }.items()}
    return GarnishProfile(edge_smoothing=legacy_smoothing, **values)


def _garnish_region_to_dict(region: GarnishRegion) -> dict:
    return {"id": region.id, "polygon": region.polygon, "enabled": region.enabled,
            "profile": _garnish_to_dict(region.profile), "source": region.source}


def _garnish_region_from_dict(data: dict) -> GarnishRegion:
    return GarnishRegion(id=str(data.get("id", "garnish-region")),
                         polygon=[tuple(point) for point in data.get("polygon", [])],
                         enabled=data.get("enabled"), profile=_garnish_from_dict(data.get("profile")),
                         source=str(data.get("source", "manual")))


def _manifest_path(store_dir: Path, asset_id: str) -> Path:
    return store_dir / f"{asset_id}.manifest.json"


def merge_server_owned(
    incoming: Dict[str, Any], stored: Optional[TextManifest],
) -> Dict[str, Any]:
    """Carry server-owned fields forward when the payload does not name them.

    PRESENCE, not truthiness, is the test.  The endpoint receives a bare
    dict, so `{"guided_blocks": []}` and a payload with no such key are the
    same object once either is read with `.get()` -- and they mean opposite
    things.  Omission is "I have nothing to say about this"; an explicit
    empty value is "clear it".  Deciding between them on emptiness would
    make a deliberate reset unexpressible and an accidental omission
    destructive, which is exactly the pair of bugs this function exists to
    separate.

    Returns a NEW dict; the caller's payload is not mutated, so a request
    body stays what the client actually sent for logging and diffing.
    """
    if stored is None:
        return dict(incoming)
    merged = dict(incoming)
    stored_data = _manifest_to_dict(stored)
    for field_name in SERVER_OWNED_FIELDS:
        if field_name not in merged:
            merged[field_name] = stored_data.get(field_name)
    stored_instances = {
        item.get("id"): item for item in stored_data.get("instances", [])
        if isinstance(item, dict) and item.get("id") is not None
    }
    incoming_instances = merged.get("instances")
    if isinstance(incoming_instances, list):
        protected_instances = []
        for item in incoming_instances:
            if not isinstance(item, dict):
                protected_instances.append(item)
                continue
            protected = dict(item)
            previous = stored_instances.get(item.get("id"), {})
            for field_name in SERVER_OWNED_INSTANCE_FIELDS:
                if field_name not in protected and field_name in previous:
                    protected[field_name] = previous[field_name]
            protected_instances.append(protected)
        merged["instances"] = protected_instances
    return merged


## An optional check run immediately before every write.
##
## Registered rather than imported so the dependency points the right way:
## this module must not know what "valid" means -- that lives in
## `layers.couvert` and in the test-suite oracle, both of which sit ABOVE
## persistence. A callback lets the strictest available definition be
## installed without this file growing an opinion.
##
## The point is that it cannot be bypassed. Route-level settling covers the
## routes someone remembered; anything that reaches disk goes through here,
## including code paths written later by someone who has never heard of the
## invariant. Off by default -- production must not fail a save on an
## assertion -- and switched on for the whole test suite in conftest.
_SAVE_GUARD: Optional[Any] = None


def set_save_guard(guard: Optional[Any]) -> Optional[Any]:
    """Install a `guard(manifest, asset_id)` run before each save.

    Returns the previous guard, so a caller can restore it -- a test that
    deliberately persists a damaged manifest has to be able to turn this off
    for exactly that write and put it back afterwards.
    """
    global _SAVE_GUARD
    previous = _SAVE_GUARD
    _SAVE_GUARD = guard
    return previous


def save_manifest(store_dir: Path, asset_id: str, manifest: TextManifest) -> None:
    """serialize a TextManifest to JSON on disk."""
    if _SAVE_GUARD is not None:
        _SAVE_GUARD(manifest, asset_id)
    store_dir.mkdir(parents=True, exist_ok=True)
    data = _manifest_to_dict(manifest)
    _manifest_path(store_dir, asset_id).write_text(
        json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def load_manifest(store_dir: Path, asset_id: str) -> Optional[TextManifest]:
    """deserialize a TextManifest from disk, or None if not found."""
    p = _manifest_path(store_dir, asset_id)
    if not p.exists():
        return None
    data = json.loads(p.read_text(encoding="utf-8-sig"))
    return _dict_to_manifest(data)


def _manifest_to_dict(m: TextManifest) -> dict:
    return {
        "asset_id": m.asset_id,
        "total_regions": m.total_regions,
        "src_lang": m.src_lang,
        "targ_lang": m.targ_lang,
        "img_dim": list(m.img_dim) if m.img_dim else None,
        "scene_regions": [_region_to_dict(r) for r in (m.scene_regions or [])],
        "semantic_units": [_semantic_unit_to_dict(u) for u in (m.semantic_units or [])],
        "guided_blocks": [_guided_block_to_dict(b) for b in (getattr(m, "guided_blocks", None) or [])],
        ## Written verbatim: okara's graph is already a plain dict of plain
        ## values (`CandidateGraph.to_dict`), and re-shaping it here would
        ## give the append-only record a second, divergent schema.
        "candidate_lineage": getattr(m, "candidate_lineage", None),
        "vision2_candidate_ledger": _typed_evidence_to_dict(
            getattr(m, "vision2_candidate_ledger", None)
        ),
        "asset_class": m.asset_class,
        "asset_classification": m.asset_classification,
        "asset_type": m.asset_type.value if hasattr(m.asset_type, "value") else str(m.asset_type),
        "frame_count": m.frame_count,
        "fps": m.fps,
        "duration": m.duration,
        "prcssng_time": m.prcssng_time,
        "instances": [_inst_to_dict(i) for i in m.instances],
}


def _semantic_unit_to_dict(unit: SemanticTextUnit) -> dict:
    return {
        "id": unit.id,
        "plate_uid": unit.plate_uid,
        "display_number": unit.display_number,
        "origin": unit.origin,
        "membership_hash": unit.membership_hash,
        "region_ids": list(unit.region_ids),
        "source_text": unit.source_text,
        "bbox": {
            "x": unit.bbox.x, "y": unit.bbox.y,
            "width": unit.bbox.width, "height": unit.bbox.height,
        },
        "entity_type": unit.entity_type,
        "confidence": unit.confidence,
        "analysis_provider": unit.analysis_provider,
        "semantic_roles": unit.semantic_roles,
        "review_required": unit.review_required,
        "substitution": unit.substitution,
        "pairing": unit.pairing,
        "suggestion": unit.suggestion,
        "ocr_repair": unit.ocr_repair,
    }


def _dict_to_semantic_unit(data: dict) -> SemanticTextUnit:
    bbox = data.get("bbox") or {}
    return SemanticTextUnit(
        id=str(data.get("id", "u-unknown")),
        ## Defaults, never minted here: a legacy manifest gets its durable
        ## identity from basil.migrate_plate_identity() in ONE persisted
        ## pass.  Minting on read would hand the same project a different
        ## UID on every machine that opened it.
        plate_uid=str(data.get("plate_uid") or ""),
        display_number=data.get("display_number"),
        origin=str(data.get("origin") or "derived"),
        membership_hash=str(data.get("membership_hash") or ""),
        region_ids=[str(region_id) for region_id in data.get("region_ids", [])],
        source_text=str(data.get("source_text", "")),
        bbox=BBox(
            x=int(bbox.get("x", 0)), y=int(bbox.get("y", 0)),
            width=int(bbox.get("width", 0)), height=int(bbox.get("height", 0)),
        ),
        entity_type=str(data.get("entity_type", "unknown")),
        confidence=float(data.get("confidence", 0.0)),
        analysis_provider=str(data.get("analysis_provider", "deterministic_layout")),
        semantic_roles=dict(data.get("semantic_roles") or {}),
        review_required=bool(data.get("review_required", True)),
        substitution=data.get("substitution"),
        pairing=data.get("pairing"),
        suggestion=data.get("suggestion"),
        ocr_repair=data.get("ocr_repair"),
    )


def _guided_atom_to_dict(atom) -> dict:
    return {
        "id": atom.id, "text": atom.text, "position": atom.position,
        "kind": atom.kind, "script": atom.script, "direction": atom.direction,
    }


def _dict_to_guided_atom(data: dict):
    from tofu.core.types import GuidedAtom
    return GuidedAtom(
        id=str(data.get("id", "")), text=str(data.get("text", "")),
        position=int(data.get("position", 0)), kind=str(data.get("kind", "word")),
        script=data.get("script"), direction=data.get("direction"),
    )


def _guided_block_to_dict(block) -> dict:
    """A Block round-trips whole, including what was judged about it.

    `raw_text` is stored exactly as typed -- whitespace and case are evidence
    about how the text appears in the asset, and normalising on the way to
    disk would lose the thing a matcher needs.
    """
    return {
        "id": block.id,
        "raw_text": block.raw_text,
        "normalized_text": block.normalized_text,
        "position": block.position,
        "source_language": block.source_language,
        "scope": block.scope,
        "find_all": block.find_all,
        "atoms": [_guided_atom_to_dict(a) for a in (block.atoms or [])],
        "language_assessment": block.language_assessment,
        "detection_assessment": block.detection_assessment,
    }


def _dict_to_guided_block(data: dict):
    from tofu.core.types import GuidedBlock
    return GuidedBlock(
        id=str(data.get("id", "")),
        raw_text=str(data.get("raw_text", "")),
        normalized_text=str(data.get("normalized_text", "")),
        position=int(data.get("position", 0)),
        source_language=data.get("source_language"),
        scope=str(data.get("scope") or "asset"),
        find_all=bool(data.get("find_all", False)),
        atoms=[_dict_to_guided_atom(a) for a in data.get("atoms", [])],
        language_assessment=data.get("language_assessment"),
        detection_assessment=data.get("detection_assessment"),
    )


def _region_to_dict(r: SceneRegion) -> dict:
    return {
        "bbox": {"x": r.bbox.x, "y": r.bbox.y,
                 "width": r.bbox.width, "height": r.bbox.height},
        "semantic_label": r.semantic_label,
        "confidence": r.confidence,
        "background_color": r.background_color,
        "border_detected": r.border_detected,
        "polygon": r.polygon,
        "texture": r.texture,
        "material": r.material,
        "material_evidence": r.material_evidence,
        "garnish_profile": _garnish_to_dict(r.garnish_profile),
        "surface_observation": r.surface_observation,
    }


def _dict_to_region(rdict: dict) -> SceneRegion:
    b = rdict["bbox"]
    return SceneRegion(
        bbox=BBox(x=b["x"], y=b["y"], width=b["width"], height=b["height"]),
        semantic_label=rdict.get("semantic_label", "surface"),
        confidence=rdict.get("confidence", 0.0),
        background_color=rdict.get("background_color"),
        border_detected=rdict.get("border_detected", False),
        polygon=(
            [tuple(p) for p in rdict["polygon"]]
            if rdict.get("polygon") else None
        ),
        texture=rdict.get("texture"),
        material=rdict.get("material"),
        material_evidence=rdict.get("material_evidence"),
        garnish_profile=_garnish_from_dict(rdict.get("garnish_profile")),
        surface_observation=rdict.get("surface_observation"),
    )


def _inst_to_dict(inst: InstText) -> dict:
    d = {
        "id": inst.id,
        "bounding_box": {
            "x": inst.bounding_box.x,
            "y": inst.bounding_box.y,
            "width": inst.bounding_box.width,
            "height": inst.bounding_box.height,
        },
        "adjusted_bbox": ({
            "x": inst.adjusted_bbox.x,
            "y": inst.adjusted_bbox.y,
            "width": inst.adjusted_bbox.width,
            "height": inst.adjusted_bbox.height,
        } if inst.adjusted_bbox else None),
        "text": inst.text,
        "target_text": inst.target_text,
        "confidence": inst.confidence,
        "detected_language": inst.detected_language,
        "reading_order": inst.reading_order,
        "dnt": inst.dnt,
        "excluded": inst.excluded,
        "target_language": inst.target_language,
        "glyph_fallback": inst.glyph_fallback,
        "tm_suggestion": inst.tm_suggestion,
        "translation_attempts": inst.translation_attempts,
        "translation_decision": inst.translation_decision,
        "translation_history": inst.translation_history,
        "ocr_correction": inst.ocr_correction,
        "source_override": inst.source_override,
        "recognition_history": inst.recognition_history,
        "ocr_provenance": inst.ocr_provenance,
        "ocr_quality": inst.ocr_quality,
        ## Provenance that was being written in memory and dropped on save,
        ## the same defect class as candidate_lineage. `lineage_candidate_id`
        ## is the one that matters most: it is the stamp joining a shipped
        ## region to its okara node, so without it the persisted graph cannot
        ## be attributed back to the regions it produced. `scene_eligibility`
        ## records the scene filter's verdict whether or not the veto is
        ## enabled -- which is only true if it survives the write.
        "scene_eligibility": inst.scene_eligibility,
        "review_features": inst.review_features,
        "lineage_candidate_id": inst.lineage_candidate_id,
        "evidence_survival": getattr(inst, "evidence_survival", None),
        ## Which measurement channel produced this reading. Persisted because
        ## every NED, survival state and candidate distribution derived from
        ## this region is a property of the image-channel pair; a manifest
        ## that drops it puts the region back into the single-channel fiction.
        "channel": getattr(inst, "channel", None),
        "channel_id": getattr(inst, "channel_id", None),
        "glyph_match_evidence": (
            _typed_evidence_to_dict(inst.glyph_match_evidence)
        ),
        "fusion_evidence": (
            _typed_evidence_to_dict(inst.fusion_evidence)
        ),
        "vision2_decision_history": inst.vision2_decision_history,
        ## Video identity. Dropping these silently un-tracked every instance
        ## on reload.
        "source_asset_id": inst.source_asset_id,
        "target_asset_id": inst.target_asset_id,
        "frame_index": inst.frame_index,
        "temporal_span": list(inst.temporal_span) if inst.temporal_span else None,
        "track_id": inst.track_id,
        "repair_provenance": inst.repair_provenance,
        "reconstruction_profile": (
            asdict(inst.reconstruction_profile)
            if inst.reconstruction_profile is not None else None
        ),
        "material_evidence": inst.material_evidence,
        "surface_observation": inst.surface_observation,
        "font_match": inst.font_match,
        "resolved_font_family": inst.resolved_font_family,
        "resolved_synthetic_italic": inst.resolved_synthetic_italic,
        "semantic_assignment": inst.semantic_assignment,
        "garnish_override": _garnish_to_dict(inst.garnish_override),
        "garnish_enabled": inst.garnish_enabled,
        "garnish_scope": inst.garnish_scope,
        "garnish_regions": [_garnish_region_to_dict(region) for region in (inst.garnish_regions or [])],
    }
    if inst.language is not None:
        d["language"] = inst.language
    if inst.segmentation_mask:
        d["segmentation_mask"] = {
            "polygon": inst.segmentation_mask.polygon,
            "confidence": inst.segmentation_mask.confidence,
        }
        if inst.segmentation_mask.holes:
            d["segmentation_mask"]["holes"] = inst.segmentation_mask.holes
    if inst.style_profile:
        s = inst.style_profile
        d["style_profile"] = {
            "font_family": s.font_family,
            "font_weight": s.font_weight,
            "color": s.color,
            "shadow": s.shadow,
            "effects": s.effects,
            "font_size": s.font_size,
            "italic": s.italic,
            "underline": s.underline,
            "underline_offset": s.underline_offset,
            "underline_width": s.underline_width,
            "subscript": s.subscript,
            "superscript": s.superscript,
            "align_h": s.align_h,
            "align_v": s.align_v,
            "justification": s.justification,
            "indent": s.indent,
            "tracking": s.tracking,
            "kerning": s.kerning,
            "leading": s.leading,
            "baseline_shift": s.baseline_shift,
            "tab_width": s.tab_width,
            "tsume": s.tsume,
            "stroke_color": s.stroke_color,
            "stroke_width": s.stroke_width,
            "stroke_position": s.stroke_position,
            "target_orientation": s.target_orientation,
            "word_order": s.word_order,
            "transform": s.transform,
        }
    if inst.background_profile:
        d["background_profile"] = {
            "semantic_label": inst.background_profile.semantic_label,
            "texture": inst.background_profile.texture,
            "material": inst.background_profile.material,
            "gradients": inst.background_profile.gradients,
            "patterns": inst.background_profile.patterns,
            "dominant_color": inst.background_profile.dominant_color,
            "surface_texture": inst.background_profile.surface_texture,
            "cleanse_strategy": inst.background_profile.cleanse_strategy,
        }
    if inst.characteristics:
        d["characteristics"] = {
            "font_style": inst.characteristics.font_style,
            "color": inst.characteristics.color,
            "size": inst.characteristics.size,
            "positioning": inst.characteristics.positioning,
            "effects": inst.characteristics.effects,
        }
    return d


def _dict_to_manifest(data: dict) -> TextManifest:
    instances = []
    for idict in data.get("instances", []):
        bbox = BBox(
            x=idict["bounding_box"]["x"],
            y=idict["bounding_box"]["y"],
            width=idict["bounding_box"]["width"],
            height=idict["bounding_box"]["height"],
        )
        adjusted_bbox = None
        if isinstance(idict.get("adjusted_bbox"), dict):
            abox = idict["adjusted_bbox"]
            adjusted_bbox = BBox(
                x=abox["x"], y=abox["y"], width=abox["width"], height=abox["height"],
            )
        mask = None
        if idict.get("segmentation_mask"):
            mdict = idict["segmentation_mask"]
            mask = Mask(
                polygon=[tuple(p) for p in mdict["polygon"]],
                confidence=mdict["confidence"],
                holes=(
                    [[tuple(p) for p in hole] for hole in mdict["holes"]]
                    if mdict.get("holes") else None
                ),
            )
        style = None
        if idict.get("style_profile"):
            sdict = idict["style_profile"]
            style = StyleProfil(
                font_family=sdict.get("font_family"),
                font_weight=sdict.get("font_weight"),
                color=sdict.get("color"),
                shadow=sdict.get("shadow"),
                effects=sdict.get("effects"),
                font_size=sdict.get("font_size"),
                italic=sdict.get("italic"),
                underline=sdict.get("underline"),
                underline_offset=sdict.get("underline_offset"),
                underline_width=sdict.get("underline_width"),
                subscript=sdict.get("subscript"),
                superscript=sdict.get("superscript"),
                align_h=sdict.get("align_h"),
                align_v=sdict.get("align_v"),
                justification=sdict.get("justification"),
                indent=sdict.get("indent"),
                tracking=sdict.get("tracking"),
                kerning=sdict.get("kerning"),
                leading=sdict.get("leading"),
                baseline_shift=sdict.get("baseline_shift"),
                tab_width=sdict.get("tab_width"),
                tsume=sdict.get("tsume"),
                stroke_color=sdict.get("stroke_color"),
                stroke_width=sdict.get("stroke_width"),
                stroke_position=sdict.get("stroke_position"),
                target_orientation=sdict.get("target_orientation"),
                word_order=sdict.get("word_order"),
                transform=sdict.get("transform"),
            )
        bg = None
        if idict.get("background_profile"):
            gdict = idict["background_profile"]
            bg = BgProfil(
                semantic_label=gdict.get("semantic_label"),
                texture=gdict.get("texture"),
                material=gdict.get("material"),
                gradients=gdict.get("gradients"),
                patterns=gdict.get("patterns"),
                dominant_color=gdict.get("dominant_color"),
                surface_texture=gdict.get("surface_texture"),
                cleanse_strategy=gdict.get("cleanse_strategy"),
            )
        chars = None
        if idict.get("characteristics"):
            cdict = idict["characteristics"]
            chars = CharactText(
                font_style=cdict.get("font_style"),
                color=cdict.get("color"),
                size=cdict.get("size"),
                positioning=cdict.get("positioning"),
                effects=cdict.get("effects"),
            )
        instances.append(InstText(
            id=idict["id"],
            bounding_box=bbox,
            adjusted_bbox=adjusted_bbox,
            segmentation_mask=mask,
            text=idict.get("text"),
            target_text=idict.get("target_text"),
            language=idict.get("language"),
            confidence=idict.get("confidence"),
            detected_language=idict.get("detected_language"),
            reading_order=idict.get("reading_order"),
            dnt=idict.get("dnt", False),
            excluded=idict.get("excluded", False),
            target_language=idict.get("target_language"),
            glyph_fallback=idict.get("glyph_fallback"),
            tm_suggestion=idict.get("tm_suggestion"),
            translation_attempts=list(idict.get("translation_attempts") or []),
            translation_decision=idict.get("translation_decision"),
            translation_history=list(idict.get("translation_history") or []),
            ocr_correction=idict.get("ocr_correction"),
            source_override=idict.get("source_override"),
            recognition_history=idict.get("recognition_history"),
            ocr_provenance=idict.get("ocr_provenance"),
            ocr_quality=idict.get("ocr_quality"),
            scene_eligibility=idict.get("scene_eligibility"),
            review_features=idict.get("review_features"),
            lineage_candidate_id=idict.get("lineage_candidate_id"),
            evidence_survival=idict.get("evidence_survival"),
            channel=idict.get("channel"),
            channel_id=idict.get("channel_id"),
            glyph_match_evidence=_glyph_match_from_dict(idict.get("glyph_match_evidence")),
            fusion_evidence=_fusion_from_dict(idict.get("fusion_evidence")),
            vision2_decision_history=list(idict.get("vision2_decision_history") or []),
            source_asset_id=idict.get("source_asset_id"),
            target_asset_id=idict.get("target_asset_id"),
            frame_index=idict.get("frame_index"),
            temporal_span=(
                tuple(idict["temporal_span"]) if idict.get("temporal_span") else None
            ),
            track_id=idict.get("track_id"),
            repair_provenance=idict.get("repair_provenance"),
            reconstruction_profile=(
                ReconstructionProfile(**idict["reconstruction_profile"])
                if isinstance(idict.get("reconstruction_profile"), dict) else None
            ),
            material_evidence=idict.get("material_evidence"),
            surface_observation=idict.get("surface_observation"),
            garnish_override=_garnish_from_dict(idict.get("garnish_override")),
            garnish_enabled=idict.get("garnish_enabled"),
            garnish_scope=idict.get("garnish_scope", "whole_selection"),
            garnish_regions=[_garnish_region_from_dict(region) for region in idict.get("garnish_regions", []) if isinstance(region, dict)],
            font_match=idict.get("font_match"),
            resolved_font_family=idict.get("resolved_font_family"),
            resolved_synthetic_italic=bool(idict.get("resolved_synthetic_italic", False)),
            semantic_assignment=idict.get("semantic_assignment"),
            style_profile=style,
            background_profile=bg,
            characteristics=chars,
        ))
    atype = AssetType(data.get("asset_type", "image"))
    ## The PERSISTED count, not `len(instances)`. Soft-deleted regions stay in
    ## the array, so recomputing on load counted them: a manifest saved with
    ## three live regions and two deleted ones came back claiming five, on
    ## every load, no matter what the writer stored. Legacy files that predate
    ## the field fall back to the LIVE count rather than the array length.
    stored_total = data.get("total_regions")
    live_total = sum(1 for i in instances if not getattr(i, "excluded", False))
    return TextManifest(
        asset_id=data["asset_id"],
        total_regions=stored_total if isinstance(stored_total, int) else live_total,
        instances=instances,
        src_lang=data.get("src_lang"),
        targ_lang=data.get("targ_lang"),
        img_dim=tuple(data["img_dim"]) if data.get("img_dim") else None,
        scene_regions=[_dict_to_region(r) for r in data.get("scene_regions", [])],
        semantic_units=[_dict_to_semantic_unit(u) for u in data.get("semantic_units", [])],
        guided_blocks=[_dict_to_guided_block(b) for b in data.get("guided_blocks", [])],
        candidate_lineage=data.get("candidate_lineage"),
        vision2_candidate_ledger=_candidate_ledger_from_dict(
            data.get("vision2_candidate_ledger")
        ),
        asset_class=data.get("asset_class"),
        asset_classification=data.get("asset_classification"),
        asset_type=atype,
        frame_count=data.get("frame_count", 1),
        fps=data.get("fps"),
        duration=data.get("duration"),
        prcssng_time=data.get("prcssng_time"),
    )
