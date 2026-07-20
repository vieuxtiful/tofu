## 🍢 Memory — Layer 6
## vieuxtiful
"""
visual translation memory layer.

memory stores accepted (source region, target render, style, QA score)
tuples for reuse: identical or near-identical source text regions in
future assets can be localized from memory instead of re-running the
pipeline.

QA gating is enforced HERE as well as in the pipeline: a visual TM
poisoned with bad localizations is worse than no TM, so update()
refuses to store results below the threshold regardless of caller.

storage-agnostic by design (matches the rest of src/tofu/layers/, which
never imports server/db.py): update() returns DRAFT records (plus a
ready-to-save thumbnail crop) and lookup() matches against a caller-
supplied candidate pool. server/main.py owns the actual SQLite writes
and the thumbnail-file save — that keeps this module pure and testable
with plain python objects, no live database required.
"""

from typing import Any, Dict, List, Optional

from tofu.core.types import InstText, TextManifest, QAReport
from tofu.utils import phash as phash_mod
from tofu.utils import textmatch
from tofu.utils.imaging import crop_region

DEFAULT_QA_THRESHOLD = 0.8
VISUAL_MATCH_THRESHOLD = 0.88  # stricter than fuzzy text: no text corroboration


def _style_fingerprint(inst: InstText) -> str:
    """coarse weight|italic|color bucket — a tiebreaker between candidates
    that score equally on text/visual similarity, not a primary match
    signal (fonts/colors are themselves detected, not ground truth)."""
    sp = inst.style_profile
    ch = inst.characteristics
    weight = (sp.font_weight if sp else None) or (ch.font_style if ch else None) or "regular"
    is_italic = bool(sp and sp.italic) or bool(ch and ch.font_style and "italic" in ch.font_style)
    color = (sp.color if sp else None) or (ch.color if ch else None) or "auto"
    return f"{weight}|{'italic' if is_italic else 'upright'}|{color}"


def update(
    text_manifest: TextManifest,
    targ_lang: str,
    localized_asset: Any,
    source_asset: Any = None,
    qa_report: Optional[QAReport] = None,
    qa_threshold: float = DEFAULT_QA_THRESHOLD,
) -> List[Dict[str, Any]]:
    """build draft translation-memory records for approved localizations.

    args:
        text_manifest: manifest of the completed run.
        targ_lang: target language code.
        localized_asset: the final rendered asset (unused directly here;
            kept for signature symmetry with verify.assess and because a
            future revision may thumbnail the LOCALIZED crop too).
        source_asset: the original pre-localization asset — cropped for
            the perceptual hash, since future lookups match against
            NOT-YET-TRANSLATED source images.
        qa_report: verify's report; storage is refused when the overall
            score is missing or below qa_threshold.
        qa_threshold: minimum acceptable overall score.

    returns:
        list of draft records actually eligible for storage (empty if
        gated). each dict has: asset_id, region_id, source_text,
        normalized_text, source_lang, target_lang, target_text,
        style_fingerprint, phash, qa_score, thumb_crop (PIL Image or
        None). the caller persists these (db write + thumb save).
    """
    if (
        qa_report is None
        or qa_report.overall_score is None
        or qa_report.overall_score < qa_threshold
    ):
        return []  # gate: never poison the TM with unverified results

    scores = qa_report.per_asset_instance_score.get(text_manifest.asset_id, {})
    updates: List[Dict[str, Any]] = []
    for inst in text_manifest.instances:
        if inst.dnt or not inst.text or not inst.target_text:
            continue  # nothing worth remembering without a real translation
        inst_score = scores.get(inst.id)
        if inst_score is None or inst_score < qa_threshold:
            continue  # per-instance gate

        crop = crop_region(source_asset, inst.bounding_box) if source_asset is not None else None
        region_hash = phash_mod.phash(crop) if crop is not None else None

        updates.append({
            "asset_id": text_manifest.asset_id,
            "region_id": inst.id,
            "source_text": inst.text,
            "normalized_text": textmatch.normalize_text(inst.text),
            "source_lang": text_manifest.src_lang,
            "target_lang": inst.target_language or targ_lang,
            "target_text": inst.target_text,
            "style_fingerprint": _style_fingerprint(inst),
            "phash": region_hash,
            "qa_score": inst_score,
            "thumb_crop": crop,
        })
    return updates


def _pick_best(inst: InstText, candidates: List[Dict[str, Any]], score: float, method: str) -> Dict[str, Any]:
    """among tied-score candidates, prefer a matching style fingerprint;
    candidates otherwise arrive newest-first (caller's query order)."""
    my_fp = _style_fingerprint(inst)
    ranked = sorted(candidates, key=lambda c: c.get("style_fingerprint") != my_fp)
    chosen = ranked[0]
    return {
        "target_text": chosen["target_text"],
        "score": round(score, 4),
        "method": method,
        "source_asset_id": chosen["asset_id"],
        "record_id": chosen.get("id"),
    }


def lookup(
    text_manifest: TextManifest,
    source_asset: Any,
    targ_lang: str,
    candidates: List[Dict[str, Any]],
) -> Dict[str, Dict[str, Any]]:
    """match each non-DNT detected region against a caller-supplied TM
    candidate pool (e.g. from db.find_tm_candidates — this module never
    touches storage), tiered per the plan:

      1. EXACT — normalized source text matches a stored record exactly.
      2. FUZZY — edit-distance ratio >= textmatch.FUZZY_MATCH_THRESHOLD
         against the best-scoring candidate (garbled/partial OCR text
         that's still recognizably the same string).
      3. VISUAL — perceptual-hash similarity >= VISUAL_MATCH_THRESHOLD,
         tried only when text matching found nothing and a source_asset
         is available to crop and hash (the same physical sign, but OCR
         read it differently or not at all this time).

    returns {region_id: {target_text, score, method, source_asset_id,
    record_id}} — regions with no confident match are simply absent, not
    an empty/zero entry, so callers can test `region_id in matches`.
    """
    results: Dict[str, Dict[str, Any]] = {}
    if not candidates:
        return results

    for inst in text_manifest.instances:
        if inst.dnt:
            continue

        normalized = textmatch.normalize_text(inst.text) if inst.text else ""
        best: Optional[Dict[str, Any]] = None

        if normalized:
            exact = [c for c in candidates if c.get("normalized_text") == normalized]
            if exact:
                best = _pick_best(inst, exact, 1.0, "exact")

        if best is None and inst.text:
            scored = [
                (textmatch.fuzzy_similarity(inst.text, c.get("source_text", "")), c)
                for c in candidates
            ]
            scored = [(s, c) for s, c in scored if s >= textmatch.FUZZY_MATCH_THRESHOLD]
            if scored:
                top = max(s for s, _ in scored)
                tied = [c for s, c in scored if s == top]
                best = _pick_best(inst, tied, top, "fuzzy")

        if best is None and source_asset is not None:
            crop = crop_region(source_asset, inst.bounding_box)
            inst_hash = phash_mod.phash(crop) if crop is not None else None
            if inst_hash:
                scored = [
                    (phash_mod.visual_similarity(inst_hash, c.get("phash")), c)
                    for c in candidates
                ]
                scored = [
                    (s, c) for s, c in scored
                    if s is not None and s >= VISUAL_MATCH_THRESHOLD
                ]
                if scored:
                    top = max(s for s, _ in scored)
                    tied = [c for s, c in scored if s == top]
                    best = _pick_best(inst, tied, top, "visual")

        if best is not None:
            results[inst.id] = best

    return results
