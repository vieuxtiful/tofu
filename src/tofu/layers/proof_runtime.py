"""Failure-isolated shadow runtime for typeface-invariant glyph retrieval."""

from __future__ import annotations

import hashlib
from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from tofu.core.types import BBox, InstText, TextManifest
from tofu.core.vision2 import (
    ArtifactRevision,
    GlyphCandidateEvidence,
    GlyphMatchEvidence,
    Vision2Config,
    Vision2Decision,
)

FEATURE_SCHEMA = "proof-runtime-v1"


@dataclass(frozen=True)
class CandidateText:
    text: str
    source: str


_MODEL_CACHE: dict[tuple[str, int], tuple[Any, str]] = {}


def candidate_pool(
    inst: InstText | None,
    ground_truth_pool: Iterable[tuple] | None,
    limit: int,
) -> list[CandidateText]:
    ordered: list[CandidateText] = []
    seen: set[str] = set()

    def add(value: Any, source: str) -> None:
        text = str(value or "").strip()
        if text and text not in seen and len(ordered) < limit:
            seen.add(text)
            ordered.append(CandidateText(text, source))

    for entry in ground_truth_pool or ():
        if isinstance(entry, list | tuple) and entry:
            add(entry[0], f"ground_truth:{entry[2] if len(entry) > 2 else 'unknown'}")
    if inst is not None:
        correction = inst.ocr_correction or {}
        for key in ("corrected_text", "candidate_text", "original_text"):
            add(correction.get(key), "ocr_correction")
        for record in inst.recognition_history or ():
            if not isinstance(record, dict):
                continue
            add(record.get("candidate_text"), f"recognition:{record.get('stage', 'unknown')}")
            add(record.get("primary_text"), f"recognition:{record.get('stage', 'unknown')}")
        add(inst.text, "incumbent_ocr")
        if inst.text:
            from tofu.layers.proof import hard_negatives
            for negative in hard_negatives(inst.text, limit=max(0, limit - len(ordered))):
                add(negative["text"], negative["source"])
    return ordered


def _pool_revision(pool: list[CandidateText]) -> str:
    material = "\n".join(f"{item.source}\0{item.text}" for item in pool)
    return hashlib.sha256(material.encode("utf-8")).hexdigest()[:16]


def _load_model(path_value: str | None) -> tuple[Any | None, str | None, str | None]:
    if not path_value:
        return None, None, "checkpoint_not_configured"
    path = Path(path_value)
    if not path.is_file():
        return None, None, "checkpoint_missing"
    key = (str(path.resolve()), path.stat().st_mtime_ns)
    cached = _MODEL_CACHE.get(key)
    if cached is not None:
        return cached[0], cached[1], None
    try:
        import torch

        from tofu.layers import proof_encoder
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
        model = proof_encoder.build_checkpoint_model(checkpoint)
        model.load_state_dict(checkpoint["model"])
        model.eval()
        revision = hashlib.sha256(path.read_bytes()).hexdigest()[:16]
        _MODEL_CACHE.clear()
        _MODEL_CACHE[key] = (model, revision)
        return model, revision, None
    except Exception:
        return None, None, "checkpoint_unavailable"


def _font_paths(
    registry: Any, pool: list[CandidateText], limit: int,
) -> list[str]:
    faces = getattr(registry, "faces", None) or {}
    ranked = []
    for path, face in faces.items():
        codepoints = getattr(face, "codepoints", set())
        coverage = sum(
            all(ord(char) in codepoints for char in item.text)
            for item in pool
        )
        if coverage:
            ranked.append((-coverage, str(path)))
    return [path for _coverage, path in sorted(ranked)[:limit]]


def _observed_mask(asset: Any, bbox: BBox, vertical: bool):
    from tofu.layers.proof_encoder import canonical_reading_direction
    from tofu.utils.imaging import load_rgb, text_mask
    image = load_rgb(asset)
    if image is None:
        return None
    mask = text_mask(image, bbox, refine=False)
    if mask is None:
        return None
    return canonical_reading_direction(mask.astype("float32"), vertical=vertical)


def _views(observed: Any, enabled: bool) -> list[Any]:
    if not enabled:
        return [observed]
    import cv2
    import numpy as np
    image = np.asarray(observed, dtype=np.float32)
    kernel = np.ones((2, 2), np.uint8)
    return [
        image,
        cv2.GaussianBlur(image, (0, 0), .45),
        cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel),
    ]


def _score(
    model: Any,
    observed: Any,
    pool: list[CandidateText],
    font_paths: list[str],
    multi_view: bool,
    component_alignment: bool,
    alignment_top_k: int,
    scene_counterfactual: bool,
    counterfactual_top_k: int,
    counterfactual_beam: int,
) -> tuple[list[GlyphCandidateEvidence], dict[str, Any]]:
    import numpy as np
    import torch

    from tofu.layers import proof, proof_distribution, proof_encoder

    rendered: list[tuple[CandidateText, Any, Any]] = []
    render_counts: dict[str, int] = defaultdict(int)
    ## Which faces actually rendered which candidate. The pool is not
    ## font-complete -- a face is chosen for how much of the pool it covers,
    ## not for covering all of it -- and knowing where the holes are is what
    ## decides whether typeface can serve as an attempt axis.
    faces_by_text: dict[str, set[str]] = defaultdict(set)
    for item in pool:
        for font_path in font_paths:
            image = proof.render(item.text, font_path)
            if image is not None:
                rendered.append((item, proof_encoder.prepare(image), image))
                render_counts[item.text] += 1
                faces_by_text[item.text].add(font_path)
    if not rendered:
        return [], {"rendered_candidates": 0, "views": 0}
    candidate_tensor = torch.from_numpy(np.stack([image for _, image, _raw in rendered])).unsqueeze(1)
    observations = torch.from_numpy(np.stack([
        proof_encoder.prepare(view) for view in _views(observed, multi_view)
    ])).unsqueeze(1)
    model.eval()
    with torch.no_grad():
        candidate_vectors = model(candidate_tensor)
        observed_vectors = model(observations)
        matrix = observed_vectors @ candidate_vectors.T
    per_text: dict[str, list[float]] = defaultdict(list)
    for index, (item, _image, _raw) in enumerate(rendered):
        per_text[item.text].extend(float(value) for value in matrix[:, index].tolist())
    source_by_text = {item.text: item.source for item in pool}
    ranked = sorted(
        ((text, float(np.mean(values)), float(np.var(values))) for text, values in per_text.items()),
        key=lambda row: (-row[1], row[0]),
    )
    ## One distribution per ATTEMPT, over the ranked pool. The attempt axis
    ## is the photometric view -- typeface is folded into each candidate's
    ## score rather than forming a second axis, because the pool is not
    ## font-complete: `_font_paths` ranks faces by how much of the pool they
    ## cover, so a candidate may render in three faces and its neighbour in
    ## one, and a softmax needs every candidate scored in every attempt.
    ## `font_complete_faces` below records whether a typeface axis is
    ## available at all, which is what a later attempt-level objective needs
    ## to know before it can use one.
    #
    # Nothing here touches `support`, `margin`, the rank order, or any
    # decision: the distribution is added alongside the incumbent ranking,
    # never in place of it, until a paired gate says otherwise.
    order = [text for text, _support, _variance in ranked]
    indices_by_text: dict[str, list[int]] = defaultdict(list)
    for index, (item, _image, _raw) in enumerate(rendered):
        indices_by_text[item.text].append(index)
    attempts = [
        proof_distribution.softmax([
            float(np.mean([matrix[view, index] for index in indices_by_text[text]]))
            for text in order
        ])
        for view in range(matrix.shape[0])
    ]
    distribution = proof_distribution.summarize(attempts)
    probability_by_text = dict(
        zip(order, distribution["consensus"], strict=True)
    ) if distribution else {}
    evidence = []
    for index, (text, support, variance) in enumerate(ranked):
        next_support = ranked[index + 1][1] if index + 1 < len(ranked) else None
        features = {
            "transformation_variance": round(variance, 8),
            "rendered_faces": render_counts[text],
        }
        if component_alignment and index < alignment_top_k:
            try:
                from tofu.layers.proof_alignment import align
                alignments = [
                    result for item, _prepared, raw in rendered
                    if item.text == text and (result := align(observed, raw)) is not None
                ]
                if alignments:
                    best = min(alignments, key=lambda item: item["alignment_cost"])
                    features["component_alignment"] = best
            except Exception:
                features["component_alignment_unavailable"] = True
        if scene_counterfactual and index < counterfactual_top_k:
            try:
                from tofu.layers.scene_counterfactual import score as scene_score
                hypotheses = [
                    result for item, _prepared, raw in rendered
                    if item.text == text
                    and (result := scene_score(
                        observed, raw, beam_limit=counterfactual_beam,
                    )) is not None
                ]
                if hypotheses:
                    features["scene_counterfactual"] = min(
                        hypotheses,
                        key=lambda result: result["best_hypothesis"]["substrate_residual"],
                    )
            except Exception:
                features["scene_counterfactual_unavailable"] = True
        evidence.append(GlyphCandidateEvidence(
            text=text, rank=index + 1, support=round(support, 6),
            source=source_by_text[text],
            margin=round(support - next_support, 6) if next_support is not None else None,
            probability=probability_by_text.get(text),
            features=features,
        ))
    diagnostics: dict[str, Any] = {
        "rendered_candidates": len(per_text),
        "rendered_candidate_faces": len(rendered),
        "views": len(_views(observed, multi_view)),
        ## How many faces rendered the WHOLE pool. A later objective that
        ## wants typeface as its own attempt axis needs this to be > 1;
        ## reporting it costs nothing and stops that question being answered
        ## by assumption.
        "font_complete_faces": sum(
            1 for path in font_paths
            if all(path in faces_by_text[text] for text in per_text)
        ),
    }
    if distribution is not None:
        diagnostics["distribution"] = distribution
    return evidence, diagnostics


def _vertical(inst: InstText | None) -> bool:
    if inst is None:
        return False
    profile = inst.style_profile
    return bool(profile and profile.target_orientation == "vertical")


def evaluate_region(
    asset: Any,
    bbox: BBox,
    inst: InstText | None,
    ground_truth_pool: Iterable[tuple] | None,
    font_registry: Any,
    config: Vision2Config,
    lineage_candidate_id: str | None = None,
) -> GlyphMatchEvidence:
    pool = candidate_pool(inst, ground_truth_pool, config.max_retrieval_candidates)
    base = GlyphMatchEvidence(
        schema=FEATURE_SCHEMA, decision=Vision2Decision.SHADOW,
        supported_domain=False, lineage_candidate_id=lineage_candidate_id,
        diagnostics={
            "candidate_pool_size": len(pool), "vertical": _vertical(inst),
            ## Which channel's observation this evidence was retrieved
            ## against. A candidate distribution is a property of the
            ## image-channel pair like every other measurement here;
            ## ledger proposals have no settled read and so carry None.
            "channel_id": getattr(inst, "channel_id", None),
        },
    )
    model, model_revision, error = _load_model(config.checkpoint_path)
    if error:
        base.reason_codes.append(error)
        return base
    if len(pool) < 2:
        base.reason_codes.append("candidate_pool_insufficient")
        return base
    fonts = _font_paths(font_registry, pool, config.max_retrieval_fonts)
    if not fonts:
        base.reason_codes.append("fonts_unavailable")
        return base
    observed = _observed_mask(asset, bbox, _vertical(inst))
    if observed is None:
        base.reason_codes.append("observed_mask_unavailable")
        return base
    try:
        candidates, diagnostics = _score(
            model, observed, pool, fonts, config.multi_view,
            config.component_alignment, config.alignment_top_k,
            config.scene_counterfactual, config.counterfactual_top_k,
            config.counterfactual_beam,
        )
    except Exception:
        base.reason_codes.append("retrieval_unavailable")
        return base
    base.candidates = candidates
    base.diagnostics.update(diagnostics)
    base.revisions = ArtifactRevision(
        feature_schema=FEATURE_SCHEMA, model=model_revision,
        candidate_pool=_pool_revision(pool),
    )
    base.supported_domain = len(candidates) >= 2
    if not base.supported_domain:
        base.reason_codes.append("renderable_candidate_pool_insufficient")
    return base


def assess_manifest(
    asset: Any,
    manifest: TextManifest,
    ground_truth_pool: Iterable[tuple] | None,
    font_registry: Any,
    config: Vision2Config,
) -> int:
    if not config.enabled:
        return 0
    written = 0
    evaluated = 0
    by_lineage = {inst.lineage_candidate_id: inst for inst in manifest.instances}
    for inst in manifest.instances:
        if evaluated < config.max_retrieval_regions:
            inst.glyph_match_evidence = evaluate_region(
                asset, inst.bounding_box, inst, ground_truth_pool, font_registry, config,
                lineage_candidate_id=inst.lineage_candidate_id,
            )
            evaluated += 1
        else:
            inst.glyph_match_evidence = GlyphMatchEvidence(
                schema=FEATURE_SCHEMA, decision=Vision2Decision.SHADOW,
                supported_domain=False, reason_codes=["retrieval_budget_exhausted"],
                lineage_candidate_id=inst.lineage_candidate_id,
            )
        written += 1
    ledger = manifest.vision2_candidate_ledger
    if ledger is not None:
        for proposal in ledger.proposals:
            if proposal.final_eligible or proposal.candidate_id in by_lineage:
                continue
            if evaluated < config.max_retrieval_regions:
                proposal.glyph_match_evidence = evaluate_region(
                    asset, BBox(*proposal.geometry), None, ground_truth_pool,
                    font_registry, config, lineage_candidate_id=proposal.candidate_id,
                )
                evaluated += 1
            else:
                proposal.glyph_match_evidence = GlyphMatchEvidence(
                    schema=FEATURE_SCHEMA, decision=Vision2Decision.SHADOW,
                    supported_domain=False, reason_codes=["retrieval_budget_exhausted"],
                    lineage_candidate_id=proposal.candidate_id,
                )
            written += 1
    return written
