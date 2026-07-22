"""Basil -- semantic registration and target-span substitution.

OCR regions are geometric facts: their IDs and boxes are durable anchors for
Cleanse and Scribe.  Translation, on the other hand, is a sentence or named
entity problem.  Basil bridges those layers without ever renumbering or
moving Cicerone regions:

* register nearby source regions as a reading unit after OCR correction;
* retain both visual/source order and a later target-language semantic order;
* align a user-supplied target phrase back to the original region anchors;
* fail open when evidence is insufficient rather than assigning words by
  incidental box order.

The optional Stanza and multilingual-MT/alignment seams are intentionally
offline-only.  Setting a model directory is an explicit deployment action;
this module never downloads a model while someone is scanning an asset.
"""

from __future__ import annotations

import os
import re
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from tofu.core.types import BBox, InstText, SemanticTextUnit, TextManifest


_TOKEN_RE = re.compile(r"[^\W_]+(?:[’'-][^\W_]+)?", re.UNICODE)

# This compact glossary is deliberately evidence, not a general MT system.
# A deployment can register a benchmarked NLLB/translation-memory adapter
# later; an unknown pair must stay review-required instead of being guessed.
_GLOSSARY: Dict[Tuple[str, str], Dict[str, str]] = {
    ("fr", "it"): {
        "rue": "via", "des": "dei", "de": "di", "du": "del",
        "la": "la", "le": "il", "les": "i", "vieux": "vecchi",
        "vieil": "vecchio", "vieille": "vecchia", "murs": "muri",
        "mur": "muro", "place": "piazza", "avenue": "viale",
    },
}
_ACTIVE_GLOSSARY: Optional[Dict[str, Any]] = None

_STREET_DESIGNATORS = {
    "rue", "avenue", "boulevard", "chemin", "place", "quai", "route",
    "via", "viale", "piazza", "corso", "strada", "street", "road",
    "lane", "drive", "highway", "calle", "avenida", "plaza", "carrer",
    "straße", "strasse", "weg", "platz",
}
_MODIFIERS = {
    "vieux", "vieil", "vieille", "vieux", "vecchio", "vecchi", "vecchia",
    "old", "new", "nouveau", "nouvelle", "grand", "grande", "petit",
    "petite", "alto", "alta", "alto", "basso", "bassa",
}
_HEAD_WORDS = {
    "mur", "murs", "muro", "muri", "porte", "pont", "bridge", "wall",
    "walls", "ville", "city", "saint", "sainte", "mont", "mount",
}


def _lang(code: Optional[str]) -> str:
    return (code or "").lower().replace("_", "-").split("-", 1)[0]


def _locale(code: Optional[str]) -> str:
    value = (code or "").strip().replace("_", "-")
    return value.lower()


def _tokens(text: Optional[str]) -> List[str]:
    return _TOKEN_RE.findall(text or "")


def _normal(token: str) -> str:
    return token.casefold().replace("’", "'")


def _visual_order(instances: Iterable[InstText]) -> List[InstText]:
    """Visual reading order independent of stable ``rN`` identifiers.

    This repeats Cicerone's useful baseline tolerance at the instance level:
    two words that start a handful of pixels apart remain on one line, then
    sort left-to-right.  It makes the rue-vieux source unit ``r1, r3, r2``
    even though ``r2`` is an immutable ID and may have been allocated first.
    """
    records = [inst for inst in instances if not inst.excluded and (inst.text or "").strip()]
    if len(records) < 2:
        return records
    heights = sorted(max(1, inst.bounding_box.height) for inst in records)
    tolerance = max(4.0, heights[len(heights) // 2] * 0.42)
    lines: List[Dict[str, Any]] = []
    for inst in sorted(records, key=lambda item: (item.bounding_box.y + item.bounding_box.height / 2, item.bounding_box.x)):
        box = inst.bounding_box
        center = box.y + box.height / 2
        line = next((candidate for candidate in lines
                     if abs(center - candidate["center"]) <= max(tolerance, candidate["height"] * 0.42)), None)
        if line is None:
            lines.append({"center": center, "height": box.height, "items": [inst]})
            continue
        line["items"].append(inst)
        count = len(line["items"])
        line["center"] += (center - line["center"]) / count
        line["height"] += (box.height - line["height"]) / count
    ordered: List[InstText] = []
    for line in sorted(lines, key=lambda candidate: candidate["center"]):
        ordered.extend(sorted(line["items"], key=lambda item: item.bounding_box.x))
    return ordered


def _contains(region_box: BBox, box: BBox) -> bool:
    cx = box.x + box.width / 2
    cy = box.y + box.height / 2
    return region_box.x <= cx <= region_box.x + region_box.width and region_box.y <= cy <= region_box.y + region_box.height


def _union_box(instances: Sequence[InstText]) -> BBox:
    x0 = min(inst.bounding_box.x for inst in instances)
    y0 = min(inst.bounding_box.y for inst in instances)
    x1 = max(inst.bounding_box.x + inst.bounding_box.width for inst in instances)
    y1 = max(inst.bounding_box.y + inst.bounding_box.height for inst in instances)
    return BBox(x=x0, y=y0, width=x1 - x0, height=y1 - y0)


def _nearby(left: InstText, right: InstText) -> bool:
    """A conservative edge for text on a common sign or label."""
    a, b = left.bounding_box, right.bounding_box
    ah, bh = max(1, a.height), max(1, b.height)
    aw, bw = max(1, a.width), max(1, b.width)
    acy, bcy = a.y + a.height / 2, b.y + b.height / 2
    same_line = abs(acy - bcy) <= max(ah, bh) * 0.46
    x_gap = max(0, max(a.x, b.x) - min(a.x + a.width, b.x + b.width))
    if same_line and x_gap <= max(ah, bh) * 2.5:
        return True
    y_gap = max(0, max(a.y, b.y) - min(a.y + a.height, b.y + b.height))
    x_overlap = max(0, min(a.x + a.width, b.x + b.width) - max(a.x, b.x))
    return y_gap <= max(ah, bh) * 1.6 and x_overlap / min(aw, bw) >= 0.18


def _components(instances: Sequence[InstText]) -> List[List[InstText]]:
    parents = list(range(len(instances)))

    def find(index: int) -> int:
        while parents[index] != index:
            parents[index] = parents[parents[index]]
            index = parents[index]
        return index

    def join(left: int, right: int) -> None:
        a, b = find(left), find(right)
        if a != b:
            parents[b] = a

    for left in range(len(instances)):
        for right in range(left + 1, len(instances)):
            if _nearby(instances[left], instances[right]):
                join(left, right)
    groups: Dict[int, List[InstText]] = {}
    for index, inst in enumerate(instances):
        groups.setdefault(find(index), []).append(inst)
    return list(groups.values())


def _role_for(inst: InstText, unit_is_street: bool, first: bool) -> str:
    words = [_normal(word) for word in _tokens(inst.text)]
    if unit_is_street and first and words and words[0] in _STREET_DESIGNATORS:
        return "street_designator"
    if any(word in _MODIFIERS for word in words):
        return "modifier"
    if any(word in _HEAD_WORDS for word in words):
        return "head"
    return "content"


def _local_semantics(instances: Sequence[InstText]) -> Tuple[str, float, Dict[str, str]]:
    phrase_words = [_normal(word) for inst in instances for word in _tokens(inst.text)]
    is_street = bool(phrase_words and phrase_words[0] in _STREET_DESIGNATORS)
    roles = {
        inst.id: _role_for(inst, is_street, index == 0)
        for index, inst in enumerate(instances)
    }
    if is_street:
        return "street_name", 0.88, roles
    # Short display labels are usually noun phrases; do not claim a named
    # entity without model evidence.
    return ("label" if len(phrase_words) <= 8 else "sentence"), 0.52, roles


def _stanza_observation(source_text: str, src_lang: Optional[str]) -> Optional[Dict[str, Any]]:
    """Read locally provisioned Stanza models, never triggering a download."""
    model_dir = os.environ.get("TOFU_BASIL_STANZA_DIR")
    if not model_dir or not Path(model_dir).is_dir():
        return None
    language = _lang(src_lang)
    if not language:
        return None
    try:
        import stanza  # type: ignore
        pipeline = stanza.Pipeline(
            lang=language, processors="tokenize,pos,lemma,depparse,ner",
            model_dir=model_dir, download_method=None, verbose=False,
        )
        document = pipeline(source_text)
    except Exception:
        return None
    entities = [
        {"text": entity.text, "type": entity.type}
        for entity in getattr(document, "entities", [])
    ]
    dependencies: List[Dict[str, str]] = []
    for sentence in getattr(document, "sentences", []):
        for word in getattr(sentence, "words", []):
            dependencies.append({"text": word.text, "upos": word.upos, "deprel": word.deprel})
    return {"entities": entities, "dependencies": dependencies}


def _unit_key(instances: Sequence[InstText]) -> Tuple[str, ...]:
    return tuple(inst.id for inst in instances)


def unify_manifest(manifest: TextManifest) -> List[SemanticTextUnit]:
    """Register source reading units without touching user translations.

    Existing applied substitution provenance is retained only when both the
    source text and immutable member IDs still match.  If OCR/source editing
    changes either, the plan becomes stale and is deliberately discarded.
    """
    visual = _visual_order(manifest.instances)
    previous = {
        (tuple(unit.region_ids), unit.source_text): unit.substitution
        for unit in (manifest.semantic_units or [])
    }
    grouped: List[List[InstText]] = []
    claimed: set[str] = set()
    # A confident panel/bordered surface is the strongest architectural
    # evidence that separated lines belong to one physical sign.
    for region in manifest.scene_regions or []:
        if region.semantic_label not in {"panel", "bordered_region"} or region.confidence < 0.35:
            continue
        members = [inst for inst in visual if inst.id not in claimed and _contains(region.bbox, inst.bounding_box)]
        if len(members) >= 2:
            grouped.append(members)
            claimed.update(inst.id for inst in members)
    remaining = [inst for inst in visual if inst.id not in claimed]
    for component in _components(remaining):
        ordered = _visual_order(component)
        grouped.append(ordered)

    # Stable unit numbering follows the actual source reading order, never
    # raw rN allocation order.
    grouped.sort(key=lambda members: min(visual.index(member) for member in members))
    units: List[SemanticTextUnit] = []
    for number, members in enumerate(grouped, 1):
        member_ids = [inst.id for inst in members]
        source = " ".join((inst.text or "").strip() for inst in members).strip()
        if not source:
            continue
        entity_type, confidence, roles = _local_semantics(members)
        observation = _stanza_observation(source, manifest.src_lang)
        provider = "deterministic_layout"
        if observation is not None:
            provider = "stanza+deterministic_layout"
            # NER evidence is intentionally advisory: store it with the unit
            # but do not turn arbitrary names into a translation decision.
            roles = {**roles, "_stanza": "available"}
        units.append(SemanticTextUnit(
            id=f"u{number}",
            region_ids=member_ids,
            source_text=source,
            bbox=_union_box(members),
            entity_type=entity_type,
            confidence=confidence,
            analysis_provider=provider,
            semantic_roles=roles,
            review_required=entity_type != "street_name",
            substitution=previous.get((tuple(member_ids), source)),
        ))
    manifest.semantic_units = units
    return units


def provider_statuses() -> List[Dict[str, Any]]:
    """Deployment-visible model seams; this never probes remote services."""
    stanza_dir = os.environ.get("TOFU_BASIL_STANZA_DIR")
    nllb_dir = os.environ.get("TOFU_BASIL_NLLB_MODEL")
    align_dir = os.environ.get("TOFU_BASIL_ALIGN_MODEL")
    providers = [
        {
            "id": "deterministic_layout", "active": True,
            "purpose": "visual reading units and verified local glossary alignment",
        },
        {
            "id": "stanza", "active": bool(stanza_dir and Path(stanza_dir).is_dir()),
            "purpose": "local token/POS/dependency/NER evidence", "model_dir": stanza_dir,
        },
        {
            "id": "nllb", "active": bool(nllb_dir and Path(nllb_dir).exists()),
            "purpose": "candidate sentence translation; benchmark-gated and never auto-applied", "model_dir": nllb_dir,
        },
        {
            "id": "awesome_align", "active": bool(align_dir and Path(align_dir).exists()),
            "purpose": "candidate cross-language token/span alignment; benchmark-gated", "model_dir": align_dir,
        },
    ]
    providers.append({
        "id": "uploaded_glossary",
        "active": bool(_ACTIVE_GLOSSARY),
        "purpose": "user-supplied termbase (global/project cascade)",
        "mode": _ACTIVE_GLOSSARY.get("mode") if _ACTIVE_GLOSSARY else None,
        "entry_count": _ACTIVE_GLOSSARY.get("entry_count", 0) if _ACTIVE_GLOSSARY else 0,
    })
    return providers


def set_active_glossary(info: Optional[Dict[str, Any]]) -> None:
    """Expose current glossary metadata to deployment/provider diagnostics."""
    global _ACTIVE_GLOSSARY
    _ACTIVE_GLOSSARY = info


def plan_substitution(
    manifest: TextManifest,
    unit_id: str,
    target_text: str,
    targ_lang: Optional[str],
    external_lexicon: Optional[Dict[Tuple[str, str], Dict[str, str]]] = None,
) -> Dict[str, Any]:
    """Plan, but never mutate, target assignments for a semantic unit."""
    unify_manifest(manifest)
    unit = next((item for item in manifest.semantic_units if item.id == unit_id), None)
    if unit is None:
        raise KeyError(f"semantic unit '{unit_id}' not found")
    phrase = (target_text or "").strip()
    base: Dict[str, Any] = {
        "schema": 1,
        "unit_id": unit.id,
        "source_text": unit.source_text,
        "target_text": phrase,
        "source_region_order": list(unit.region_ids),
        # Cubes are immutable visual boxes.  A semantic block can be plated
        # into a different cube when target syntax reorders source meaning.
        "spatial_anchor_order": list(unit.region_ids),
        "target_region_order": [],
        "assignments": [],
        "method": "manual_required",
        "confidence": 0.0,
        "review_required": True,
        "warnings": [],
    }
    if not phrase:
        base["warnings"].append("Enter the complete target phrase before planning placement.")
        return base
    effective = external_lexicon if external_lexicon is not None else _GLOSSARY
    source_locale, target_locale = _locale(manifest.src_lang), _locale(targ_lang)
    lexicon = effective.get((source_locale, target_locale))
    if lexicon is None:
        lexicon = next(
            (terms for (src_key, targ_key), terms in effective.items()
             if str(src_key).lower() == source_locale and str(targ_key).lower() == target_locale),
            None,
        )
    if lexicon is None:
        lexicon = effective.get((_lang(manifest.src_lang), _lang(targ_lang)))
    if lexicon is None and external_lexicon is not None:
        lexicon = effective.get((source_locale.split("-", 1)[0], target_locale.split("-", 1)[0]))
    if not lexicon:
        base["warnings"].append(
            "No locally verified alignment model or glossary is available for this language pair; keep per-region edits or provision a benchmarked provider."
        )
        return base

    target_words = _tokens(phrase)
    target_normal = [_normal(word) for word in target_words]
    region_word_indices: Dict[str, List[int]] = {}
    cursor = 0
    for region_id in unit.region_ids:
        inst = next((item for item in manifest.instances if item.id == region_id), None)
        count = len(_tokens(inst.text if inst else ""))
        region_word_indices[region_id] = list(range(cursor, cursor + count))
        cursor += count
    source_words = _tokens(unit.source_text)
    if len(source_words) != cursor:
        base["warnings"].append("Source unit token registration is inconsistent; review the captured text before assignment.")
        return base

    used: set[int] = set()
    mapped_positions: Dict[int, int] = {}
    for source_index, source_word in enumerate(source_words):
        expected = lexicon.get(_normal(source_word))
        if not expected:
            base["warnings"].append(f"No verified local equivalent for source token '{source_word}'.")
            return base
        found = next((index for index, target_word in enumerate(target_normal)
                      if index not in used and target_word == expected), None)
        if found is None:
            base["warnings"].append(
                f"The target phrase does not contain the verified equivalent '{expected}' for '{source_word}'."
            )
            return base
        used.add(found)
        mapped_positions[source_index] = found
    if len(used) != len(target_words):
        base["warnings"].append("Target phrase contains unaligned words; a model/provider or manual per-region placement is required.")
        return base

    semantic_assignments: List[Dict[str, Any]] = []
    for region_id in unit.region_ids:
        target_positions = sorted(mapped_positions[index] for index in region_word_indices[region_id])
        assigned = " ".join(target_words[index] for index in target_positions)
        semantic_assignments.append({
            "region_id": region_id,
            "text": assigned,
            "target_positions": target_positions,
            "method": "uploaded_glossary_span_alignment" if external_lexicon is not None else "verified_glossary_span_alignment",
            "confidence": 0.98,
        })
    # Target syntax determines the semantic block sequence; visual source
    # order determines the immutable cube sequence.  Zip the two rather than
    # assigning text back to its source rN.  For Rue des / VIEUX / MURS this
    # explicitly plates r2:Muri into r3's cube and r3:Vecchi into r2's cube.
    target_semantic_order = sorted(semantic_assignments, key=lambda item: min(item["target_positions"]))
    assignments: List[Dict[str, Any]] = []
    for anchor_id, semantic in zip(unit.region_ids, target_semantic_order):
        assignments.append({**semantic, "anchor_id": anchor_id})
    base.update({
        "assignments": assignments,
        "target_region_order": [assignment["region_id"] for assignment in target_semantic_order],
        "method": "uploaded_glossary_span_alignment" if external_lexicon is not None else "verified_glossary_span_alignment",
        "confidence": 0.98,
        "review_required": False,
    })
    return base


def apply_substitution(manifest: TextManifest, plan: Dict[str, Any], targ_lang: Optional[str]) -> SemanticTextUnit:
    """Apply an explicitly requested plan while preserving all geometry."""
    if not plan.get("assignments") or plan.get("review_required"):
        raise ValueError("Basil will not apply an unverified or incomplete substitution plan")
    unit = next((item for item in manifest.semantic_units if item.id == plan.get("unit_id")), None)
    if unit is None:
        raise KeyError(f"semantic unit '{plan.get('unit_id')}' not found")
    by_id = {inst.id: inst for inst in manifest.instances}
    anchor_ids = [assignment.get("anchor_id", assignment["region_id"]) for assignment in plan["assignments"]]
    if len(set(anchor_ids)) != len(anchor_ids):
        raise ValueError("Basil plan assigns more than one semantic block to the same spatial cube")
    original_boxes = {
        anchor_id: asdict(by_id[anchor_id].bounding_box)
        for anchor_id in anchor_ids if anchor_id in by_id
    }
    for assignment in plan["assignments"]:
        anchor_id = assignment.get("anchor_id", assignment["region_id"])
        inst = by_id.get(anchor_id)
        if inst is None or inst.excluded or inst.dnt:
            raise ValueError(f"spatial cube '{anchor_id}' cannot receive a semantic substitution")
        inst.target_text = assignment["text"]
        inst.target_language = targ_lang or inst.target_language
        inst.semantic_assignment = {
            "schema": 1,
            "unit_id": unit.id,
            "source_text": unit.source_text,
            "target_text": plan["target_text"],
            "text": assignment["text"],
            "semantic_region_id": assignment["region_id"],
            "anchor_id": anchor_id,
            "target_positions": assignment["target_positions"],
            "method": plan["method"],
            "confidence": plan["confidence"],
            "geometry_unchanged": True,
        }
    unit.substitution = {
        "schema": 1,
        "applied": True,
        "source_text": unit.source_text,
        "target_text": plan["target_text"],
        "source_region_order": plan["source_region_order"],
        "target_region_order": plan["target_region_order"],
        "spatial_anchor_order": plan.get("spatial_anchor_order", plan["source_region_order"]),
        "assignments": plan["assignments"],
        "method": plan["method"],
        "confidence": plan["confidence"],
        "geometry": original_boxes,
    }
    return unit


def plated_texts(manifest: TextManifest) -> Dict[str, str]:
    """Return the authoritative Basil block→cube text overlay.

    ``InstText.target_text`` remains a convenient persisted projection for
    editors, but Scribe and previews must use this relation as the source of
    truth.  That prevents an old autosave or a legacy manifest projection
    from undoing a verified semantic block→spatial cube placement.
    """
    migrate_legacy_plating(manifest)
    output: Dict[str, str] = {}
    for unit in manifest.semantic_units or []:
        substitution = unit.substitution if isinstance(unit.substitution, dict) else None
        if not substitution or not substitution.get("applied"):
            continue
        for assignment in substitution.get("assignments") or []:
            if not isinstance(assignment, dict):
                continue
            anchor_id = assignment.get("anchor_id", assignment.get("region_id"))
            text = assignment.get("text")
            if isinstance(anchor_id, str) and isinstance(text, str) and text.strip():
                output[anchor_id] = text
    return output


def migrate_legacy_plating(manifest: TextManifest) -> bool:
    """Upgrade pre-cube-map approved plans without changing any bbox.

    Earlier Basil plans persisted target semantic IDs but no ``anchor_id``.
    Their intended target order and source cube order are enough to recover
    the missing relation deterministically.  The migration also repairs the
    editable target projection so legacy Translate tables/tooltips agree with
    Scribe and the canvas.
    """
    changed = False
    by_id = {inst.id: inst for inst in manifest.instances}
    for unit in manifest.semantic_units or []:
        substitution = unit.substitution if isinstance(unit.substitution, dict) else None
        if not substitution or not substitution.get("applied"):
            continue
        assignments = [item for item in substitution.get("assignments") or [] if isinstance(item, dict)]
        if not assignments or all(item.get("anchor_id") for item in assignments):
            continue
        cubes = substitution.get("spatial_anchor_order") or substitution.get("source_region_order") or unit.region_ids
        semantic = sorted(assignments, key=lambda item: min(item.get("target_positions") or [10**9]))
        for cube_id, assignment in zip(cubes, semantic):
            assignment["anchor_id"] = cube_id
            inst = by_id.get(cube_id)
            if inst is not None and isinstance(assignment.get("text"), str):
                inst.target_text = assignment["text"]
                inst.semantic_assignment = {
                    **(inst.semantic_assignment or {}),
                    "schema": 1,
                    "unit_id": unit.id,
                    "source_text": unit.source_text,
                    "target_text": substitution.get("target_text", ""),
                    "text": assignment["text"],
                    "semantic_region_id": assignment.get("region_id"),
                    "anchor_id": cube_id,
                    "target_positions": assignment.get("target_positions", []),
                    "method": substitution.get("method", "verified_glossary_span_alignment"),
                    "confidence": substitution.get("confidence", 0.0),
                    "geometry_unchanged": True,
                }
            changed = True
        substitution["spatial_anchor_order"] = list(cubes)
    return changed
