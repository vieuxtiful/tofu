## 🍢 decant — the careful pour that leaves the sediment behind
## vieuxtiful
"""How much identity-bearing evidence survives in this observation?

Decanting separates what can be served from what has to be left in the
bottle. This asks the same question of a captured region, and it asks it
SEPARATELY from "which candidate scores highest":

    1. Which reading wins?              <- cicerone, aboyeur, forage
    2. Is there enough here to justify
       choosing any reading at all?     <- this module

Those are different questions, and conflating them is how a system ends up
confidently transcribing a sign whose ink is gone. A candidate can rank
first because every alternative was eliminated; that is not the same as
positive evidence surviving, and a ranking alone cannot tell the two apart.

WHAT THIS DELIBERATELY DOES NOT DO.

It does not produce a score. `layers/ticket.py` records a per-candidate
feature vector and refuses to weight it, for a reason worth repeating here:
a ranker cannot be built until its effect on review effort is measurable,
and that measurement needs reviewer outcomes the application does not yet
record. Building the weighting first "would repeat the veto's original error
one layer up -- a policy nobody can price."

So this returns a STATE and the measurements that produced it. Every input
is a quantity the pipeline already computed, every threshold is named, and a
reader can see which measurement drove the verdict. When reviewer outcomes
exist, a calibrated scalar can replace the state and this becomes its
training signal rather than its competitor.

WHY THE STATES ARE THESE STATES. They mirror the taxonomy in
`scripts/eval_detector_evidence.py`, which was built to separate remedies
that cost very different amounts:

    absent    no identity-bearing evidence -> a different detector, tiles,
              or fine-tuning. Expensive, and no threshold will help.
    weak      evidence present but under the floor -> calibration or a
              targeted high-resolution retry. Cheap.
    partial   evidence present and incomplete -> the read may be truncated;
              extent, not recognition, is the suspect.
    present   enough survives to justify a decision.
    unknown   not measurable here. NOT a synonym for `absent` -- a consumer
              must never be able to confuse "not measured" with "measured
              low", which is the same rule scene_eligibility follows.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

ABSENT, WEAK, PARTIAL, PRESENT, UNKNOWN = (
    "absent", "weak", "partial", "present", "unknown",
)

## Glyph height below which a correct read cannot be relied upon.
##
## `reasoned`, not measured: EasyOCRBackend upscales anything under
## MIN_CROP_HEIGHT = 40px because below that its own reads degrade, and
## la-bastille's annotation note records +/-3px box error at 13-14px line
## height -- error large enough, relative to the box, to make IoU-0.5
## scoring noise. 12px is the point at which those two observations agree
## that the evidence is thin. It has NOT been swept.
## Registered in docs/threshold-register.md.
LEGIBLE_GLYPH_PX = 12.0

## Recognizer confidence below which the read is not, by itself, evidence
## that anything was recognized.
##
## `invented`. Confidence is known to be weak on its own here -- cicerone
## records a correctly-scripted CJK read at 0.015 against a wrong-charset
## garbage read at 0.087 -- which is exactly why it may only ever contribute
## a reason, never decide a state alone. See `_confidence_reason`.
LOW_CONFIDENCE = 0.3

## CRAFT character-region response below which the detector saw nothing at
## this location. `measured` in the sense that it is PROPOSAL_FLOOR from
## eval_detector_evidence, the loosest threshold the pipeline ever runs
## (PASS_THRESHOLDS[-1]); a peak under it was never going to form a proposal
## however the rest is tuned.
NO_ACTIVATION = 0.10
PROPOSAL_FLOOR = 0.20


def _glyph_height(inst: Any) -> Optional[float]:
    """Best available estimate of glyph height, in source pixels.

    Prefers the recognizer's own estimate over the box height: a line box
    around a short word is taller than its glyphs, and using the box would
    call thin evidence thick.
    """
    quality = getattr(inst, "ocr_quality", None) or {}
    estimated = quality.get("estimated_glyph_height")
    if isinstance(estimated, (int, float)) and estimated > 0:
        return float(estimated)
    box = getattr(inst, "bounding_box", None)
    return float(box.height) if box is not None and box.height else None


def _craft_peak(inst: Any) -> Optional[float]:
    """The detector's own response at this region, if a ticket recorded it.

    Opt-in (`TOFU_LOG_CRAFT_SCORES`), so it is usually absent -- and absent
    means UNKNOWN, never zero.
    """
    ticket = getattr(inst, "review_features", None) or {}
    peak = ticket.get("craft_region")
    return float(peak) if isinstance(peak, (int, float)) else None


def _confidence_reason(inst: Any) -> Optional[str]:
    """Low confidence as a REASON, never as a verdict.

    On CJK the recognizer's confidence sits near zero even on correct reads,
    so a state decided on confidence alone would abstain on exactly the
    scripts that need the most help.
    """
    confidence = getattr(inst, "confidence", None)
    if confidence is None:
        return None
    return "low_recognition_confidence" if confidence < LOW_CONFIDENCE else None


def survival(inst: Any) -> Dict[str, Any]:
    """What survives here, and which measurement says so.

    Order of decision is from the least recoverable failure to the most, so
    the state names the cheapest remedy that could still apply.
    """
    reasons: List[str] = []
    measured: Dict[str, Any] = {}

    peak = _craft_peak(inst)
    height = _glyph_height(inst)
    quality = getattr(inst, "ocr_quality", None) or {}
    text = (getattr(inst, "text", None) or "").strip()

    measured["craft_region"] = peak
    measured["glyph_height_px"] = round(height, 1) if height is not None else None
    measured["ocr_quality_state"] = quality.get("state")
    measured["has_text"] = bool(text)

    ## Detector evidence first, when it was recorded. It is the only signal
    ## that separates "blind to this" from "saw it and lost it downstream",
    ## and that distinction decides whether any threshold work can help.
    if peak is not None:
        if peak < NO_ACTIVATION:
            reasons.append("no_detector_activation")
            return _verdict(ABSENT, reasons, measured)
        if peak < PROPOSAL_FLOOR:
            reasons.append("detector_activation_below_proposal_floor")
            return _verdict(WEAK, reasons, measured)

    ## A region that carries no text at all has nothing positive to offer,
    ## whatever else is true of it.
    if not text:
        reasons.append("no_text_recovered")
        return _verdict(ABSENT if peak is None else WEAK, reasons, measured)

    if height is not None and height < LEGIBLE_GLYPH_PX:
        reasons.append("glyph_height_below_legible")
        return _verdict(WEAK, reasons, measured)

    ## The recognizer's own doubt about this read. `unresolvable` is its
    ## strongest statement and is taken at face value.
    state = quality.get("state")
    if state == "unresolvable":
        reasons.append("ocr_unresolvable")
        return _verdict(WEAK, reasons, measured)
    if state == "review_required":
        reasons.extend(quality.get("reasons") or ["ocr_review_required"])
        return _verdict(PARTIAL, reasons, measured)

    confidence_reason = _confidence_reason(inst)
    if confidence_reason:
        ## Contributes a reason and downgrades to `partial`, never to
        ## `weak`: see `_confidence_reason` for why confidence alone is not
        ## trusted to condemn a read.
        reasons.append(confidence_reason)
        return _verdict(PARTIAL, reasons, measured)

    ## Nothing measurable was recorded and nothing contradicts the read --
    ## which is not the same as having checked. Say so.
    if peak is None and height is None:
        return _verdict(UNKNOWN, ["nothing_measured"], measured)

    return _verdict(PRESENT, reasons, measured)


def _verdict(state: str, reasons: List[str], measured: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "state": state,
        "reasons": reasons,
        "measured": measured,
        ## No scalar. See the module docstring: weighting these into a score
        ## is the ranker's job, and the ranker cannot be priced yet.
        "schema": 1,
    }


def assess_manifest(instances: Any) -> int:
    """Attach a survival verdict to every region. Changes nothing else.

    Returns the number written. Like `ticket.write_tickets`, this never
    reorders, filters or edits -- if it starts having an opinion about which
    regions matter, it has become the ranker it exists to defer.
    """
    written = 0
    for inst in instances or []:
        inst.evidence_survival = survival(inst)
        written += 1
    return written
