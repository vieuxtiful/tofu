## 🍢 skim — lifting the scum before the curd sets
## vieuxtiful
"""
When soy milk comes to temperature a skin of scum rises to the surface, and
it is lifted off *before* the nigari goes in — once the curd sets, whatever
floated is set into the block.  Detection has the same shape: a region that
is not text has to come off before Cleanse erases its pixels and Scribe
draws over them, because by then the manifest is the block.

What this layer decides is narrow on purpose: whether a SHORT, WEAK read is
made of strokes thick enough to be lettering at all.  It does not judge
long reads, confident reads, or anything it cannot measure — those keep the
treatment they already had.

Method.  Text has a stroke width that is a real fraction of its own height;
that ratio is what makes a glyph legible at a distance, and it is bounded
below by the physics of printing and sign-painting, not by any threshold
chosen here.  Typography already measures exactly that quantity for every
region (``characteristics.positioning.stroke_ratio``, consumed today by
Cleanse for inpaint radii and by font_matching for weight targets) and
already owns a calibrated floor for it: LIGHT_RATIO, the point below which
it stops calling something light lettering.  This layer adds no new
estimator and no new constant — it observes that a read *below* the
lightest weight typography recognises is not light lettering, it is not
lettering.

Measured on the avenue-de-la-république plaque, whose masonry produces the
phantom this layer exists for:

    AVENUE              0.1549   real
    de la RÉPUBLIQUE    0.1061   real
    la rue / SANS-NOM   0.1179 / 0.1108   real
    japan-street 3F/2F  0.2714 / 0.2092   real, and only 2 characters long
    japan neon banner   0.1247   real, and recognised at 0.015 confidence
    ------------------------------ LIGHT_RATIO = 0.065
    phantom "7"         0.0552   a shadowed crack between two stones
    bare stone samples  0.045 - 0.055

The length and confidence gates are what keep the floor away from the one
measured real region that falls below it: "est. 1962" on the textured-wall
fixture reads 0.0637 — genuinely thin lettering — and is nine characters
long, so this layer never looks at it.  That is the whole reason the rule
is a conjunction rather than a threshold on the ratio alone.

CALLER CONTRACT — offer only reads that carry NO script evidence.
LIGHT_RATIO is calibrated on Latin typography (typography.py: "Arial 44px
regular ≈ 0.11") and does not transfer to CJK, which builds a glyph from
many thin strokes inside a dense square rather than a few thick ones.
Measured on gemini-street, the real Hangul regions 윗, 줄, 꿀식 and 주 all
measure below the floor, and putting them in front of this layer cost
recall 0.333 -> 0.278.  A read that decoded to no script at all is the
only case where a thin-stroke measurement stands uncontradicted — which
is also exactly the case cicerone's has_digit whitelist cannot judge.
"""

from typing import Any, List, Optional, Tuple

from tofu.core.types import BBox, InstText

## A read this short carries no internal evidence that it is language at
## all -- there is no word shape, no script run, nothing for the later
## layers to corroborate it against.  Longer reads are left alone however
## thin they measure, because thin-but-real lettering does exist and is
## exactly what the length gate protects (see "est. 1962" above).
MAX_SCUM_CHARS = 2

## Scoped to the same confidence band as cicerone's ink-support gate, and
## for the same reason: a confident read is not this layer's business even
## when it is short.  The phantom this was built for scores 0.447.
MAX_SCUM_CONFIDENCE = 0.65


def stroke_ratio(asset: Any, bbox: Optional[BBox]) -> Optional[float]:
    """Stroke width as a fraction of text height, or None when unmeasurable.

    Delegates to typography rather than re-deriving the measurement, so this
    layer and the weight classifier can never drift apart about what a
    stroke is.  None on any failure -- an unmeasurable crop must leave the
    region exactly as it was found.
    """
    if bbox is None:
        return None
    try:
        from tofu.layers import typography
        from tofu.utils.imaging import load_rgb
    except ImportError:
        return None
    img = load_rgb(asset)
    if img is None:
        return None
    try:
        profile = typography.analyze_region(img, bbox)
    except Exception:
        return None
    # analyze_region returns None outright on a crop it cannot segment --
    # a flat, inkless field being the ordinary case, and the exact case
    # this layer must not act on
    return profile.stroke_ratio if profile is not None else None


def stroke_floor() -> float:
    """The floor, borrowed from typography so there is one number, not two."""
    from tofu.layers import typography
    return typography.LIGHT_RATIO


def is_scum(inst: InstText, asset: Any) -> Tuple[bool, Optional[str]]:
    """Whether a region is a non-text detection that should be lifted off.

    Returns (verdict, reason).  Fails OPEN in every uncertain case: an
    absent measurement, a missing dependency, an unreadable crop and a
    merely-short read all return False.  A layer that removes regions has
    to be wrong in the direction of keeping them.
    """
    text = (inst.text or "").strip()
    if not text or len(text) > MAX_SCUM_CHARS:
        return False, None
    if (inst.confidence or 0) >= MAX_SCUM_CONFIDENCE:
        return False, None
    ratio = stroke_ratio(asset, inst.bounding_box)
    if ratio is None:
        return False, None
    floor = stroke_floor()
    if ratio >= floor:
        return False, None
    return True, (
        f"stroke_ratio {ratio:.4f} below typography's LIGHT_RATIO {floor:.3f} "
        f"on a {len(text)}-character read at confidence {inst.confidence or 0:.3f}"
    )


## Pixels of box width per character.  Below this a read claims more
## characters than the box has room to draw legibly: measured, the real
## short reads on japan-street sit at 9.5-12.5 px/char and the plaques at
## 17-52, while the suspect digit strings this targets sit at 2.75 ("2932"
## in an 11x21 box), 5.0 ("9113" in 20x8) and 6.2 ("88688" in 31x22).
## Deliberately set BELOW the smallest measured real value rather than
## between the populations -- this signal only nominates a region for a
## second opinion, so its job is to never nominate a real one.
MIN_PX_PER_CHAR = 8.0


def char_density(inst: InstText) -> Optional[float]:
    """Box width available per recognised character, or None if unknowable.

    Uses the long axis: a vertical CJK column packs its characters down the
    height, and dividing its narrow width by the character count would
    condemn every one of them.
    """
    text = (inst.text or "").strip()
    b = inst.bounding_box
    if not text or b is None or b.width <= 0 or b.height <= 0:
        return None
    return max(b.width, b.height) / len(text)


def needs_arbitration(inst: InstText) -> bool:
    """Whether a read is worth spending a cross-engine second opinion on.

    Deliberately NOT a verdict.  These are reads this layer declined to
    judge on stroke evidence -- too long for the length gate, or too
    confident for the confidence gate -- that nonetheless claim more
    characters than their box can hold.  The caller must still require an
    independent engine to agree before removing anything, because a thin
    box is suggestive and nothing more.
    """
    density = char_density(inst)
    return density is not None and density < MIN_PX_PER_CHAR


def skim(instances: List[InstText], asset: Any) -> List[Tuple[InstText, str]]:
    """The regions to lift, paired with why.

    Returns rather than mutates: the caller owns the manifest and the audit
    trail, and a suppression pass that quietly edited its input would be
    impossible to review after the fact.
    """
    lifted: List[Tuple[InstText, str]] = []
    for inst in instances:
        verdict, reason = is_scum(inst, asset)
        if verdict and reason:
            lifted.append((inst, reason))
    return lifted
