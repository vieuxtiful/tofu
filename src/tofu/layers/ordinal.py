## 🍢 Ordinal — administrative-abbreviation review course
## vieuxtiful
"""Recognize ordinal/administrative abbreviations that OCR shattered.

Paris enamel plates carry the arrondissement as an ordinal plus an abbreviated
stem with a RAISED terminal letter: ``1er Arrᵗ``. That typography defeats a
line recognizer three ways at once, all visible in one measured region on
quai-des-orfevres (75x47px, -16.6deg, light italic, textured masonry, 0.535
confidence):

  * the italic serifed ``1`` is read as ``f``;
  * the word boundary between ordinal and stem is lost;
  * the small raised ``t`` is read as ``!``.

Result: ``ferArr !``. No single-glyph correction reaches it, because nothing is
wrong with any ONE glyph in isolation -- the STRUCTURE is what identifies it.

This is deliberately not Savor's clump course. That course repairs "one emitted
character stands for multiple glyphs" and correctly refuses this region: the
component surplus here (8 observed base components against 7 recognized glyphs)
is a raised terminal and a lost space, not a clump. Two different failures need
two different courses, each able to be wrong in its own way.

Nothing here is ever auto-applied. The course emits a REVIEW PROPOSAL carrying
its structural evidence and the identity of the versioned resource that named
the pattern; a human decides. A French abbreviation dictionary applied
automatically to low-confidence plate text is exactly the broad, unsafe
substitution this design exists to avoid.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional

from tofu.core.types import ImageLike, InstText
from tofu.utils.correction_resources import (
    CorrectionResource,
    CorrectionResourceError,
    load_correction_resource,
)

FRENCH_ORDINALS = "ordinal/french_ordinals-1.0.0.json"

# A leading glyph this wide is not an ordinal digit. '1', 'f', 'I', 'J' and 'l'
# are all NARROW; the gate's job is to reject a wide letter that happens to sit
# where a digit would ('B', 'D'), not to tell '1' from 'f' -- which is exactly
# the judgement being handed to the reviewer.
ORDINAL_LEADING_MAX_ASPECT = 0.75

# No confidence gate, deliberately.
#
# An earlier version skipped reads above 0.75 on the theory that proposing a
# rewrite of a confident read is second-guessing evidence. Measured on
# quai-des-orfevres, that reasoning is wrong: once the clipped-glyph rescue
# recovers the leading digit, the region reads '1erArr' at 0.944 -- confident,
# and still missing both the word boundary and the raised terminal. Confidence
# describes how sure the recognizer is about the glyphs it emitted; it says
# nothing about the glyphs it never emitted at all.
#
# What keeps this quiet is STRUCTURE, not confidence: the text must match an
# ordinal-plus-abbreviated-stem pattern, and must differ from the canonical
# form. A plate already reading '1er Arrᵗ' proposes nothing.


def _resource() -> Optional[CorrectionResource]:
    try:
        return load_correction_resource(FRENCH_ORDINALS)
    except CorrectionResourceError:
        return None  # a malformed resource must never fail detection


def _pattern_for(entry: Dict[str, Any]) -> re.Pattern:
    """Structural matcher for one ordinal-abbreviation entry.

    Matches ``<leading><ordinal-suffix><stem><terminal>`` with the separators
    optional, because the lost word boundary is one of the symptoms.
    """
    ordinal = str(entry["ordinal"])
    digit, suffix = ordinal[0], ordinal[1:]
    leads = [digit] + [str(c) for c in entry.get("leading_confusions", [])]
    terminals = [str(entry.get("terminal", ""))] + [
        str(c) for c in entry.get("terminal_confusions", [])
    ]
    lead_alt = "|".join(re.escape(c) for c in leads if c)
    term_alt = "|".join(re.escape(c) for c in terminals if c)
    stem = re.escape(str(entry["stem"]))
    return re.compile(
        rf"^\s*(?P<lead>{lead_alt})\s*(?P<suffix>{re.escape(suffix)})\s*"
        rf"(?P<stem>{stem})\s*(?P<term>{term_alt})?\s*$",
        re.IGNORECASE,
    )


def _leading_geometry(asset: ImageLike, inst: InstText) -> Dict[str, Any]:
    """Aspect of the region's first glyph cluster, when it can be measured.

    Fails OPEN: this course only ever proposes, so missing geometry costs a
    reviewer one extra item, while refusing to propose without it would drop a
    correct reading on every asset whose plate cannot be clustered.
    """
    try:
        from tofu.layers.savor import _plate_clusters

        plate = _plate_clusters(asset, inst)
    except Exception:
        return {"state": "unavailable", "reason": "plate clustering failed"}
    if not plate:
        return {"state": "unavailable", "reason": "plate could not be clustered"}
    # _plate_clusters returns (mask, raw_components, clusters); the clusters are
    # reading-order (x, y, w, h) boxes, one per claimed glyph.
    _mask, _components, clusters = plate
    if not clusters:
        return {"state": "unavailable", "reason": "no glyph clusters"}
    first = tuple(clusters[0])
    if len(first) < 4:
        return {"state": "unavailable", "reason": "cluster carries no geometry"}
    width, height = float(first[2]), float(first[3])
    if width <= 0 or height <= 0:
        return {"state": "unavailable", "reason": "degenerate cluster"}
    aspect = width / height
    return {
        "state": "measured",
        "leading_aspect": round(aspect, 3),
        "ordinal_like": aspect <= ORDINAL_LEADING_MAX_ASPECT,
    }


def propose(
    asset: ImageLike,
    instances: List[InstText],
    language: Optional[str] = None,
) -> int:
    """Offer ordinal/abbreviation readings, applying only where the region
    is measurably reliable. Returns the count of proposals made.

    A structural match plus non-contradicting geometry is good evidence but
    it is not, on its own, licence to rewrite a region: the same pattern
    fires on a genuinely unreadable smear as on a crisp plate. So the
    canonical form is APPLIED only when Savor's own quality assessment
    already called the region reliable -- the same ``can_auto_apply`` gate
    every one of Savor's courses answers to -- and otherwise recorded for
    review exactly as before. Regions Savor never assessed (it can be
    switched off, and it runs first) have no verdict to lean on and are
    proposal-only, which is the conservative direction to fail in.

    Each decision is recorded through the same audit ledger Savor writes, so
    one region carrying several courses' opinions reads as one ordered
    history rather than competing top-level fields.
    """
    if language and not str(language).lower().startswith("fr"):
        return 0
    resource = _resource()
    if resource is None or not instances:
        return 0

    from tofu.layers.savor import OCR_QUALITY_RELIABLE, _record_correction

    patterns = [(entry, _pattern_for(entry)) for entry in resource.entries]
    proposed = 0
    for inst in instances:
        text = (inst.text or "").strip()
        if not text:
            continue
        for entry, pattern in patterns:
            match = pattern.match(text)
            if not match:
                continue
            geometry = _leading_geometry(asset, inst)
            # A MEASURED wide leading glyph is positive evidence against the
            # reading; unavailable geometry is not.
            if geometry.get("state") == "measured" and not geometry["ordinal_like"]:
                break
            canonical = str(entry["canonical"])
            if canonical == text:
                break
            quality = inst.ocr_quality or {}
            apply = quality.get("state") == OCR_QUALITY_RELIABLE
            explanation = (
                f"structural match for a French administrative abbreviation "
                f"({entry['expansion']}): ordinal + '{entry['stem']}' + raised "
                f"terminal. The leading '{match.group('lead')}' reads as an "
                f"ordinal digit, the word boundary is missing, and the raised "
                f"'{entry['terminal']}' was recognized as "
                f"'{match.group('term') or 'nothing'}'."
            )
            _record_correction(
                inst,
                applied=apply,
                original_text=text,
                corrected_text=canonical if apply else None,
                candidate_text=None if apply else canonical,
                reason=(
                    f"{explanation} Applied: the region reads reliably."
                    if apply else
                    f"{explanation} Proposal only -- the region is not "
                    f"assessed reliable; confirm against the plate."
                ),
                course="ordinal_abbreviation",
                ordinal_evidence={
                    "matched_ordinal": entry["ordinal"],
                    "stem": entry["stem"],
                    "leading_glyph": match.group("lead"),
                    "terminal_glyph": match.group("term"),
                    "geometry": geometry,
                    "ocr_quality_state": quality.get("state"),
                    "resource": resource.audit_identity(),
                },
            )
            if apply:
                inst.text = canonical
            proposed += 1
            break
    return proposed
