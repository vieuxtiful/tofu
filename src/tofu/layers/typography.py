## 🍢 Typography — per-region font-attribute estimation
## vieuxtiful
"""
estimates visual font attributes of a detected text region from pixels,
populating the manifest fields the UI and scribe consume (weight, slant/
italic, size, baseline rotation). classical, citable methods only:

  - weight: dominant stroke width from the glyph mask via the
    area/perimeter ribbon estimate cross-checked with the distance
    transform (stroke-width analysis — Epshtein, Ofek & Wexler,
    CVPR 2010), normalized by text height. thresholds calibrated on
    tests/fixtures/stylized-italic.png (Arial regular vs bold).
  - slant: shear-search maximizing vertical projection-profile energy
    (the standard de-slanting estimator — Vinciarelli & Luettin,
    Pattern Recognition Letters 2001). |slant| >= ITALIC_DEG ⇒ italic.
  - size: text row extent (ascender-to-descender) of the mask.
  - rotation: cv2.minAreaRect over the detection polygon (or mask
    points), normalized to [-45°, 45°].

every estimator degrades to None on weak evidence — a wrong "regular"
is worse than an honest unknown, because scene enrichment never
overwrites user-set values but DOES persist its own.
"""

import math
from dataclasses import dataclass
from typing import Any, Optional

from tofu.core.types import BBox, Polygon
from tofu.utils.imaging import text_mask

# weight thresholds on stroke_width / text_height (calibrated on the
# stylized-italic fixture: Arial 44px regular ≈ 0.11, bold ≈ 0.15)
# Stroke width as a share of CAP HEIGHT (see _cap_height -- it used to be a
# share of the full ink extent, which made the answer depend on whether the
# string happened to contain a descender).
#
# Re-calibrated against every region in this repo carrying weight ground
# truth (serif-vs-sans and stylized-italic, ten regions, measured through
# the same padded boxes the tests use): regular runs 0.0887 to 0.1162, bold
# 0.1323 to 0.1702. 0.125 sits between them with about 7% clearance either
# side -- narrow, and narrow for a real reason. Stem weight relative to cap
# height is genuinely typeface-dependent: Times Regular measures 0.089 and
# Arial Regular 0.116, so a light serif and a heavy sans are further apart
# than a sans is from its own bold. A universal threshold cannot do better
# than this without knowing the family, which is what font_matching is
# trying to work out from the same evidence.
BOLD_RATIO = 0.125
HEAVY_RATIO = 0.190
LIGHT_RATIO = 0.065

ITALIC_DEG = 7.0        # |slant| at or above this reads as italic
SLANT_SEARCH_DEG = 24   # shear-search half-range
SLANT_STEP_DEG = 2      # coarse step; refined ±step around the peak
MIN_MASK_PIXELS = 40    # below this the estimators are noise
MIN_ROTATION_DEG = 2.0  # snap near-axis-aligned to 0


@dataclass
class TypographyProfile:
    stroke_ratio: Optional[float] = None
    weight: Optional[str] = None          # "light" | "regular" | "bold" | "heavy"
    slant_deg: Optional[float] = None     # signed; positive = rightward lean
    italic: Optional[bool] = None
    font_px: Optional[int] = None         # ascender-to-descender extent
    rotation_deg: Optional[float] = None  # baseline angle, [-45, 45]

    def label(self) -> Optional[str]:
        """compact descriptor for CharactText.font_style / UI chips."""
        parts = []
        if self.weight and self.weight != "regular":
            parts.append(self.weight)
        if self.italic:
            parts.append("italic")
        if not parts and self.weight == "regular":
            parts.append("regular")
        return " ".join(parts) if parts else None


def _text_rows(np, mask) -> Optional[tuple]:
    """(first_row, last_row) of rows carrying real stroke mass."""
    rowsum = mask.sum(axis=1)
    if rowsum.max() == 0:
        return None
    rows = np.where(rowsum >= max(1, 0.05 * rowsum.max()))[0]
    if len(rows) == 0:
        return None
    return int(rows[0]), int(rows[-1])


def _stroke_width(np, cv2, mask) -> Optional[float]:
    """dominant stroke width of the glyph mask.

    primary: ribbon estimate 2*area/perimeter — exact for long uniform
    strokes, robust to joints. cross-check: 2x the 75th-percentile
    distance-transform value (half-width at stroke cores); the smaller
    of the two is kept so blob-like masks (failed segmentation) don't
    masquerade as ultra-bold.
    """
    m8 = mask.astype(np.uint8)
    contours, _ = cv2.findContours(m8, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return None
    area = float(mask.sum())
    perimeter = sum(cv2.arcLength(c, True) for c in contours)
    if perimeter <= 0:
        return None
    ribbon = 2.0 * area / perimeter
    dt = cv2.distanceTransform(m8, cv2.DIST_L2, 3)
    vals = dt[mask]
    if len(vals) == 0:
        return None
    core = 2.0 * float(np.percentile(vals, 75))
    return min(ribbon, core)


def _cap_height(np, mask) -> Optional[int]:
    """Ink top to BASELINE, which is the reference a stroke width means
    something against.

    Weight was measured against the full ink extent, and that extent
    depends on which letters happen to be in the string rather than on the
    lettering itself: a word with a descender is a third taller than an
    all-capital one set in the same face at the same size. So the same
    stroke divided by it reported a thinner face. Measured on the
    serif-vs-sans fixture, that inverted the answer outright -- 'Handgloves'
    in Times BOLD scored 0.1001 while 'SANS-NOM' in Arial REGULAR scored
    0.1139, and every bold region on the fixture was classified regular.

    The baseline is the last row still carrying the bulk of the ink:
    descenders are a few letters, so their rows fall far below the rows
    where every letter contributes. Above it, ink top is the cap or
    ascender line -- within a few percent of each other in text faces, and
    identical for the all-capital case.
    """
    rowsum = mask.sum(axis=1).astype(float)
    if rowsum.max() <= 0:
        return None
    inked = np.where(rowsum >= max(1.0, 0.05 * rowsum.max()))[0]
    core = np.where(rowsum >= 0.5 * rowsum.max())[0]
    if len(inked) == 0 or len(core) == 0:
        return None
    height = int(core[-1]) - int(inked[0]) + 1
    return height if height > 0 else None


def _slant(np, cv2, mask) -> Optional[float]:
    """signed slant angle via shear-search: the angle whose corrective
    shear maximizes vertical projection energy (stems become columns)."""
    h, w = mask.shape
    if h < 8 or w < 8:
        return None
    m8 = (mask * 255).astype(np.uint8)
    pad = int(h * math.tan(math.radians(SLANT_SEARCH_DEG))) + 2

    def energy(deg: float) -> float:
        t = math.tan(math.radians(deg))
        mat = np.array([[1.0, t, pad if t < 0 else 0.0], [0.0, 1.0, 0.0]])
        sheared = cv2.warpAffine(
            m8, mat, (w + pad, h), flags=cv2.INTER_NEAREST, borderValue=0
        )
        col = sheared.sum(axis=0).astype(np.float64)
        return float((col * col).sum())

    angles = list(range(-SLANT_SEARCH_DEG, SLANT_SEARCH_DEG + 1, SLANT_STEP_DEG))
    scores = {a: energy(a) for a in angles}
    best = max(scores, key=lambda a: scores[a])
    # refine at 1° around the coarse peak
    for a in (best - 1, best + 1):
        if abs(a) <= SLANT_SEARCH_DEG and a not in scores:
            scores[a] = energy(a)
    best = max(scores, key=lambda a: scores[a])
    # the corrective shear angle equals the glyph's lean; positive shear
    # straightens a rightward (italic) lean under image coordinates
    return float(best)


def _rotation(np, cv2, polygon: Optional[Polygon], mask) -> Optional[float]:
    """baseline rotation from the detection polygon (preferred — CRAFT
    quads carry real orientation) or the mask points, via minAreaRect."""
    if polygon and len(polygon) >= 3:
        pts = np.array(polygon, dtype=np.float32)
    else:
        ys, xs = np.nonzero(mask)
        if len(xs) < MIN_MASK_PIXELS:
            return None
        pts = np.column_stack([xs, ys]).astype(np.float32)
    try:
        (_, _), (rw, rh), angle = cv2.minAreaRect(pts)
    except Exception:
        return None
    if rw < rh:  # minAreaRect angle is relative to the shorter edge
        angle += 90.0
    while angle > 45.0:
        angle -= 90.0
    while angle < -45.0:
        angle += 90.0
    if abs(angle) < MIN_ROTATION_DEG:
        return 0.0
    return round(float(angle), 1)


def analyze_region(
    img: Any,
    bbox: BBox,
    polygon: Optional[Polygon] = None,
) -> Optional[TypographyProfile]:
    """estimate typography attributes for one text region of a full RGB
    image (ndarray). returns None when no usable glyph mask exists."""
    try:
        import cv2
        import numpy as np
    except ImportError:
        return None
    mask = text_mask(img, bbox)
    if mask is None or mask.sum() < MIN_MASK_PIXELS:
        return None

    profile = TypographyProfile()

    rows = _text_rows(np, mask)
    if rows is not None:
        profile.font_px = rows[1] - rows[0] + 1

    stroke = _stroke_width(np, cv2, mask)
    # font_px stays the full ink extent -- scene/tofu size rendered text from
    # it and that is what they should measure. Weight is a different question
    # and needs a reference that does not move with the string's descenders.
    cap_px = _cap_height(np, mask) or profile.font_px
    if stroke is not None and cap_px and cap_px >= 8:
        ratio = stroke / cap_px
        profile.stroke_ratio = round(ratio, 4)
        if ratio >= HEAVY_RATIO:
            profile.weight = "heavy"
        elif ratio >= BOLD_RATIO:
            profile.weight = "bold"
        elif ratio <= LIGHT_RATIO:
            profile.weight = "light"
        else:
            profile.weight = "regular"

    # polygon is in full-image coordinates; shift crop-local for the mask
    rel_poly = None
    if polygon:
        rel_poly = [(px - bbox.x, py - bbox.y) for px, py in polygon]
    profile.rotation_deg = _rotation(np, cv2, rel_poly, mask)

    slant = _slant(np, cv2, mask)
    if slant is not None:
        profile.slant_deg = slant
        # a rotated baseline leans every stem by the rotation angle —
        # italic is slant IN EXCESS of the baseline, or rotated text
        # would always read as italic
        baseline = profile.rotation_deg or 0.0
        profile.italic = abs(slant - baseline) >= ITALIC_DEG

    return profile
