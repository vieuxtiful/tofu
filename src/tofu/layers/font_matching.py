"""Evidence-gated visual font identification and licensing-aware advice.

Typography's weight/slant estimator answers *what kind* of face is present.
This layer answers the different question, *which available face is closest to
the actual glyph outlines?*  It is deliberately conservative: a local visual
match becomes a recommendation, never an implicit replacement for a user's
font selection.  Optional commercial-catalog lookup is separately gated so a
source crop is never sent to a third party without an explicit editor action.

The local course is a lightweight, deterministic analogue of a visual-font
retrieval index: render the recognised source string in known installed faces,
normalise at measured cap height, then combine silhouette and *typographic*
evidence.  It is robust to colour/background because Scene's text mask supplies
the observed glyph silhouette.  A future DeepFont-style embedding can be added
as another ranked provider without changing the persisted evidence contract.

classical, citable methods only:

  silhouette   Sørensen-Dice overlap (Dice 1945) plus symmetric Chamfer
               distance (Barrow et al. 1977; Borgefors, CVGIP 1988), which
               tolerates the one- or two-pixel boundary movement a
               photographed sign always has while still punishing a
               differently shaped bowl, leg, or terminal.

  typographic  stroke contrast and terminal-serif structure, in the tradition
               of Zramdini & Ingold, "Optical Font Recognition Using
               Typographical Features", IEEE TPAMI 20(8), 1998 — identify a
               face by the global typographical properties a designer varies,
               not by classifying glyphs.  Both are read off the medial axis
               (Blum 1967; computed by the Lee-Kashyap-Chu 1994 algorithm in
               scikit-image), which gives stroke direction and half-width in
               one pass.

Why the typographic terms had to exist: silhouette overlap is dominated by
stroke mass and proportion, so it cannot see a serif.  Measured on the
la-rue-sans-nom plaque, whose lettering is plainly a serif, the silhouette-only
kernel scored a condensed sans at 0.8130 and Centaur at 0.4634 on "SANS-NOM" —
while terminal-serif structure put the source at 2.08 against 1.45-2.76 for
serif faces and 0.79-1.23 for sans.

Deferred, and noted here so the next increment starts in the right place:
terminal *shape* (teardrop vs ball vs flat-cut) and aperture/counter structure
(single- vs double-storey g) are the remaining discriminators.  Both need
per-glyph identity, which needs a glyph segmenter that survives touching
letters — savor._plate_up() undersegments them today.
"""

from __future__ import annotations

import base64
import io
import json
import math
import os
import urllib.parse
import urllib.request
from typing import Any, Dict, Iterable, List, NamedTuple, Optional, Sequence, Tuple

from tofu.core.types import InstText, TextManifest
from tofu.layers.fonts import faces_of
from tofu.utils.imaging import text_mask


# How many families reach the silhouette stage.  Raised 72 -> 100 after the
# serif-vs-sans harness started reporting real numbers: Arial sat at position
# 81 of the weight/aspect-ordered pool for "SANS-NOM" and was cut off two
# places short of being scored at all, while ranking 1st the moment it was
# admitted.  It is free -- the pool is ordered by advance metrics the registry
# already read, and the silhouette stage is not what this layer spends its
# time on (measured over the six-region fixture: 5.3s at 72, 5.0s at 100,
# 5.4s at 128).  128 admits nothing further on this corpus.
MAX_LOCAL_FACES = 100
# How many of the silhouette-ranked pool get the expensive typographic
# reading.  Generous on purpose: on all-capital lettering the silhouette
# stage ranks the correct serif far down (measured on "SANS-NOM": Centaur
# 192nd of 206 covering families), so a tight shortlist would discard the
# answer before the terms that can recognise it ever run.
# Raising this was measured and rejected: 32 -> 64 -> 100 changed no rank on
# serif-vs-sans and cost 5.7s -> 8.7s -> 12.3s over six regions. The all-caps
# regions this was meant to help are not losing because the shortlist is too
# short; see the bold/all-caps notes on MAX_LOCAL_FACES.
PROFILE_CANDIDATES = 32
TOP_CANDIDATES = 5
MIN_GLYPH_PIXELS = 45
EXACT_CONFIDENCE = 0.82
EXACT_MARGIN = 0.075

# Scoring weights.  Silhouette still carries the majority -- it answers "is
# this the same shape at all" -- but it is no longer the only voice, because
# it cannot see a serif.
#
# Swept with scripts/eval_font_match.py (--silhouette-only for the before
# half), scripts\eval_out\font-match-baseline.json against
# font-match-serif-features.json:
#   - la-rue-sans-nom, a serif plaque: the cohort's one face went from
#     Franklin Gothic Medium (a grotesk) to Baskerville Old Face.
#   - serif-vs-sans fixture: "Handgloves" set in Times went from Berlin Sans
#     FB to Modern No. 20, and "La rue" set in Times now returns Times New
#     Roman itself at rank 1.  No sans region regressed to a serif.
#   - cost 0.8s -> 1.55s per region on the plaque, 0.32s -> 0.7s on
#     gemini-street's 27 regions.
DICE_WEIGHT = 0.36
CHAMFER_WEIGHT = 0.20
PROJECTION_WEIGHT = 0.07
ASPECT_WEIGHT = 0.05
SERIF_WEIGHT = 0.12
CONTRAST_WEIGHT = 0.20
# Stroke contrast carries the categorical gate rather than terminal serifs,
# and the measurement decided that.  On the la-rue-sans-nom plaque, contrast
# agreement separates serif from sans on BOTH lines -- serifs 0.80-0.93
# against sans 0.57-0.66 -- while terminal-serif agreement separates only
# the all-capital line (serifs 0.72-0.98 vs sans 0.46-0.51) and collapses on
# the lowercase one (Centaur 0.519 against Franklin Gothic 0.491).  "La rue"
# is why: only the L carries a full-height stem, so the terminal bands see
# x-height curves passing through rather than stems ending.  Serif structure
# stays a weighted term -- it is real evidence where it applies -- but the
# feature that fires the penalty has to be the one that is right every time.
TYPOGRAPHIC_CLASS_GAP = 0.32
CLASS_MISMATCH_PENALTY = 0.45
# Evidence floors.  Measured on the same plaque: contrast sufficiency 0.46
# for mixed-case "La rue" and 0.33 for all-capital "SANS-NOM".
MIN_CONTRAST_SUFFICIENCY = 0.15
MIN_SERIF_SUFFICIENCY = 0.20


def _split_face(path: str) -> Tuple[str, int]:
    base, marker, index = path.rpartition("#")
    if marker and index.isdigit():
        return base, int(index)
    return path, 0


def _load_font(path: str, size: int):
    from PIL import ImageFont

    base, index = _split_face(path)
    return ImageFont.truetype(base, size, index=index)


def _tight(mask):
    import numpy as np

    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        return None
    return mask[ys.min():ys.max() + 1, xs.min():xs.max() + 1]


def _render_mask(path: str, text: str, height: int):
    """Render text at a stable high resolution and return its tight mask."""
    from PIL import Image, ImageDraw
    import numpy as np

    # Rendering large then reducing to the measured source height gives all
    # candidate faces the same antialiasing budget and avoids DPI-dependent
    # font rasterisation artifacts dominating the comparison.
    font = _load_font(path, max(48, int(height * 5)))
    probe = Image.new("L", (8, 8), 0)
    box = ImageDraw.Draw(probe).textbbox((0, 0), text, font=font, stroke_width=0)
    w = max(1, box[2] - box[0])
    h = max(1, box[3] - box[1])
    canvas = Image.new("L", (w + 16, h + 16), 0)
    ImageDraw.Draw(canvas).text((8 - box[0], 8 - box[1]), text, font=font, fill=255)
    mask = _tight(np.asarray(canvas) > 96)
    if mask is None:
        return None
    import cv2
    scale = height / max(1, mask.shape[0])
    width = max(1, int(round(mask.shape[1] * scale)))
    return cv2.resize(mask.astype("uint8"), (width, height), interpolation=cv2.INTER_AREA) > 0


def _aligned_masks(source, candidate):
    """Center source/candidate in one envelope without distorting width."""
    import numpy as np

    height = max(source.shape[0], candidate.shape[0])
    width = max(source.shape[1], candidate.shape[1]) + 8
    a = np.zeros((height + 8, width), dtype=bool)
    b = np.zeros_like(a)
    ay = (a.shape[0] - source.shape[0]) // 2
    ax = (a.shape[1] - source.shape[1]) // 2
    by = (b.shape[0] - candidate.shape[0]) // 2
    bx = (b.shape[1] - candidate.shape[1]) // 2
    a[ay:ay + source.shape[0], ax:ax + source.shape[1]] = source
    b[by:by + candidate.shape[0], bx:bx + candidate.shape[1]] = candidate
    return a, b


class GlyphProfile(NamedTuple):
    """Typographic properties read off one mask's medial axis.

    Each value carries its own *sufficiency* — how much evidence the mask
    actually supplied — because both properties are measurable on some
    strings and not others, and a confident-looking number computed from
    six pixels is worse than no number at all.
    """
    contrast: Optional[float]        # vertical stroke half-width / horizontal
    contrast_sufficiency: float      # share of the skeleton running horizontally
    serif: Optional[float]           # terminal branching per stem at the extremities
    serif_sufficiency: float         # skeleton pixels available in those bands


_EMPTY_PROFILE = GlyphProfile(None, 0.0, None, 0.0)

MIN_SKELETON_PIXELS = 24
MIN_DIRECTIONAL_SAMPLES = 6
MIN_TERMINAL_STEMS = 8
# Bands at the top and bottom of the glyph box, where stems terminate and
# where a serif face therefore puts ink that a sans does not.
TERMINAL_BAND_FRACTION = 0.18
# Support window for deciding whether a skeleton pixel runs vertically or
# horizontally.  Measured on the plaque: 5px cleanly separates stems from
# crossbars at 85px ink height without merging the two at a junction.
DIRECTION_WINDOW = 5


def _glyph_profile(mask) -> GlyphProfile:
    """Stroke contrast and terminal-serif structure for one glyph mask.

    Both come off a single medial-axis pass (Blum 1967), because both want
    the same two things: where the stroke centre-lines run, and how thick
    the stroke is at each point.

    *contrast* is the ratio of median vertical to median horizontal stroke
    half-width — Zramdini & Ingold's weight-variation feature.  A serif
    face modulates its strokes (thick stems, thin hairlines); a grotesk
    holds one weight throughout.  Direction is decided on the SKELETON
    rather than the filled mask: a directional erosion of the filled glyph
    keeps a stem's whole width and contaminates the median, which measured
    on the plaque put Georgia Bold (0.837) *below* Franklin Gothic (1.000)
    — the exact inversion this feature exists to prevent.

    *serif* counts skeleton branch points inside the terminal bands,
    normalised by the stems present there.  A serif slab meets its stem at
    a T-junction; a flat-cut sans stem simply ends.  This also picks up the
    ascender/descender serif flags on b, d, h, k, p, q.

    Neither is measurable on every string, and the two fail on opposite
    ones, which is why both exist and why each reports its own sufficiency.
    Contrast wants horizontal strokes: it separates a serif from a sans on
    lowercase at every size tested (agreement 0.62-0.79 for Times against
    Arial on "La rue" from 48px to 160px ink height) and at none on
    "SANS-NOM" (0.86-1.00), because S, A, N, O and M have no hairlines to
    compare a stem against.  Serif structure wants stems that terminate
    inside the bands, so it reads capitals well and lowercase poorly --
    "La rue" puts only the L there, the rest being x-height curves passing
    through.

    Returns _EMPTY_PROFILE rather than raising when scikit-image is absent
    or the mask is too small to read — the caller redistributes the weight.
    """
    try:
        from skimage.morphology import medial_axis  # deferred: pulls scipy
    except ImportError:
        return _EMPTY_PROFILE
    import cv2
    import numpy as np

    if mask is None or mask.ndim != 2 or mask.shape[0] < 8:
        return _EMPTY_PROFILE
    try:
        # medial_axis uses random tie-breaking for equidistant foreground
        # pixels. A fixed generator keeps font evidence and tests identical
        # across process order while preserving the algorithm itself.
        try:
            skeleton, distance = medial_axis(mask, return_distance=True, rng=0)
        except TypeError:
            # scikit-image before the rng parameter: retain compatibility.
            skeleton, distance = medial_axis(mask, return_distance=True)
    except Exception:
        return _EMPTY_PROFILE
    spine = skeleton.astype(np.uint8)
    if int(spine.sum()) < MIN_SKELETON_PIXELS:
        return _EMPTY_PROFILE

    # direction: which way does this skeleton pixel's own run extend?
    window = np.ones((DIRECTION_WINDOW, 1), np.uint8)
    vertical_support = cv2.filter2D(spine, -1, window)
    horizontal_support = cv2.filter2D(spine, -1, window.T)
    vertical = skeleton & (vertical_support > horizontal_support)
    horizontal = skeleton & (horizontal_support > vertical_support)
    n_vertical, n_horizontal = int(vertical.sum()), int(horizontal.sum())

    contrast = None
    contrast_sufficiency = 0.0
    if n_vertical >= MIN_DIRECTIONAL_SAMPLES and n_horizontal >= MIN_DIRECTIONAL_SAMPLES:
        # Deliberately NOT gated on a minimum stroke width.  That was tried:
        # below roughly 90px of ink a serif's hairline and its stem quantise
        # to the same distance value and the ratio reads a flat 1.0, so a
        # floor looks like the fix.  It is not -- a thin horizontal IS the
        # feature.  Measured on the plaque, the source clears a 2.5px floor
        # (3.00) while Centaur and Baskerville do not (2.00 each) and Arial
        # sails through at 4.47, so the floor silenced exactly the candidates
        # the term exists to find, and the cohort's answer fell back from
        # Baskerville Old Face to Britannic.  Small text simply yields a
        # weak signal, which the sufficiency weighting already handles.
        contrast = float(np.median(distance[vertical])) / max(
            1e-6, float(np.median(distance[horizontal]))
        )
        # All-capital Latin is nearly all stems and diagonals, so the
        # horizontal sample is thin and the ratio is noisy; lowercase
        # supplies bowls and crossbars.  Measured on the plaque: 0.45 for
        # "La rue" against 0.32 for "SANS-NOM".
        contrast_sufficiency = min(n_vertical, n_horizontal) / max(1, n_vertical + n_horizontal)

    # branch points: a skeleton pixel with three or more skeleton neighbours
    neighbours = cv2.filter2D(spine, -1, np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]], np.uint8))
    junctions = (neighbours * spine) >= 3
    band_px = max(2, int(round(mask.shape[0] * TERMINAL_BAND_FRACTION)))
    band = np.zeros_like(skeleton)
    band[:band_px] = True
    band[-band_px:] = True
    stems = int(spine[:band_px].sum() + spine[-band_px:].sum())

    serif = None
    serif_sufficiency = 0.0
    if stems >= MIN_TERMINAL_STEMS:
        # sqrt keeps the measure comparable between a short word and a long
        # one: junction count grows with the number of stems, its own
        # normaliser, so a plain ratio would flatten every difference.
        serif = float(int((junctions & band).sum())) / (stems ** 0.5)
        serif_sufficiency = min(1.0, stems / 64.0)

    return GlyphProfile(contrast, contrast_sufficiency, serif, serif_sufficiency)


def _relative_agreement(left: Optional[float], right: Optional[float]) -> Optional[float]:
    """[0, 1] agreement between two same-unit typographic measurements."""
    if left is None or right is None:
        return None
    scale = max(abs(left), abs(right))
    if scale <= 1e-6:
        return None
    return max(0.0, 1.0 - abs(left - right) / scale)


def _visual_score(
    source, candidate, *,
    source_profile: Optional[GlyphProfile] = None,
    typographic: bool = True,
) -> Tuple[float, Dict[str, float]]:
    """Return an interpretable [0, 1] glyph-shape score.

    `source_profile` lets a caller supply a typographic reading measured
    somewhere better than this one region -- see agree_on_face(), where a
    sign's lowercase line speaks for its all-capitals line.  Omitted, the
    source is profiled on its own.

    `typographic=False` runs the silhouette terms only, for callers doing a
    cheap first pass over a whole font library.
    """
    import cv2
    import numpy as np

    a, b = _aligned_masks(source, candidate)
    intersection = float(np.logical_and(a, b).sum())
    dice = (2 * intersection) / max(1.0, float(a.sum() + b.sum()))

    # Chamfer is forgiving of the one- or two-pixel boundary movement caused
    # by a photographed sign, while strongly penalising a different R bowl,
    # leg, aperture, or terminal shape.
    da = cv2.distanceTransform((~a).astype("uint8"), cv2.DIST_L2, 3)
    db = cv2.distanceTransform((~b).astype("uint8"), cv2.DIST_L2, 3)
    chamfer = (float(da[b].mean()) if b.any() else 99.0) + (float(db[a].mean()) if a.any() else 99.0)
    chamfer /= max(1.0, math.hypot(*a.shape))
    chamfer_score = max(0.0, 1.0 - chamfer * 2.6)

    def projection_similarity(x, y) -> float:
        px, py = x.sum(axis=0).astype(float), y.sum(axis=0).astype(float)
        if px.std() < 1e-6 or py.std() < 1e-6:
            return 0.0
        return max(0.0, min(1.0, (float(np.corrcoef(px, py)[0, 1]) + 1.0) / 2.0))

    projection = (projection_similarity(a, b) + projection_similarity(a.T, b.T)) / 2.0
    aspect = 1.0 - min(1.0, abs(source.shape[1] / max(1, source.shape[0]) - candidate.shape[1] / max(1, candidate.shape[0])) / 1.25)

    if not typographic:
        # Recall stage: silhouette only, at a fifteenth of the cost.  Never
        # the final word on a candidate -- see local_match for why the
        # shortlist it produces has to be generous.
        score = (
            DICE_WEIGHT * dice + CHAMFER_WEIGHT * chamfer_score
            + PROJECTION_WEIGHT * projection + ASPECT_WEIGHT * aspect
        ) / (DICE_WEIGHT + CHAMFER_WEIGHT + PROJECTION_WEIGHT + ASPECT_WEIGHT)
        return round(float(score), 4), {
            "dice": round(float(dice), 4),
            "chamfer": round(float(chamfer_score), 4),
            "projection": round(float(projection), 4),
            "aspect": round(float(aspect), 4),
        }

    src_profile = source_profile if source_profile is not None else _glyph_profile(source)
    cand_profile = _glyph_profile(candidate)
    serif_agreement = _relative_agreement(src_profile.serif, cand_profile.serif)
    contrast_agreement = _relative_agreement(src_profile.contrast, cand_profile.contrast)
    # A typographic term only votes when BOTH masks read it confidently.
    if min(src_profile.serif_sufficiency, cand_profile.serif_sufficiency) < MIN_SERIF_SUFFICIENCY:
        serif_agreement = None
    if min(src_profile.contrast_sufficiency, cand_profile.contrast_sufficiency) < MIN_CONTRAST_SUFFICIENCY:
        contrast_agreement = None

    terms = {
        "dice": (dice, DICE_WEIGHT),
        "chamfer": (chamfer_score, CHAMFER_WEIGHT),
        "projection": (projection, PROJECTION_WEIGHT),
        "aspect": (aspect, ASPECT_WEIGHT),
    }
    if serif_agreement is not None:
        terms["serif"] = (serif_agreement, SERIF_WEIGHT)
    if contrast_agreement is not None:
        terms["contrast"] = (contrast_agreement, CONTRAST_WEIGHT)
    # Absent evidence fails open: an unmeasurable term surrenders its weight
    # to the terms that DID measure something, rather than scoring zero and
    # punishing every candidate equally for the source's own illegibility.
    total_weight = sum(weight for _, weight in terms.values())
    score = sum(value * weight for value, weight in terms.values()) / max(1e-6, total_weight)

    # Serif vs sans is categorical.  Being 5% off on Dice is a different kind
    # of error from setting a serif sign in a grotesk, and a linear term
    # cannot express that: measured on the plaque, the silhouette gap between
    # Franklin Gothic (0.8130) and Centaur (0.4634) on "SANS-NOM" is larger
    # than any defensible weight for a single feature could overcome.
    mismatch = 0.0
    if contrast_agreement is not None and contrast_agreement < 1.0 - TYPOGRAPHIC_CLASS_GAP:
        mismatch = CLASS_MISMATCH_PENALTY
        score *= 1.0 - CLASS_MISMATCH_PENALTY

    evidence = {
        "dice": round(float(dice), 4),
        "chamfer": round(float(chamfer_score), 4),
        "projection": round(float(projection), 4),
        "aspect": round(float(aspect), 4),
    }
    if serif_agreement is not None:
        evidence["serif"] = round(float(serif_agreement), 4)
    if contrast_agreement is not None:
        evidence["contrast"] = round(float(contrast_agreement), 4)
    if mismatch:
        evidence["class_mismatch_penalty"] = round(float(mismatch), 4)
    return round(float(score), 4), evidence


def _weight_target(inst: InstText) -> Optional[int]:
    """OS/2 weight class this region's lettering is asking for.

    Prefers the MEASUREMENT over the label.  Typography stores
    ``style_profile.font_weight`` only when it reads heavy, bold or light --
    a "regular" reading is written as None (scene.py) -- so the label is
    absent for the commonest case, and an absent target made every face in
    the library equidistant.  ``characteristics.positioning.stroke_ratio``
    is the number that label was derived from and is present either way
    (measured 0.1179 and 0.1108 on the la-rue-sans-nom plaque), so the
    ratio is consulted first, through typography's own thresholds rather
    than a second copy of them.
    """
    value = str((inst.style_profile.font_weight if inst.style_profile else None) or "").lower()
    if any(x in value for x in ("heavy", "black", "ultra")):
        return 900
    if "bold" in value:
        return 700
    if "semi" in value or "demi" in value:
        return 600
    if "medium" in value:
        return 500
    if "light" in value:
        return 300
    if value in {"regular", "normal", "book"}:
        return 400

    positioning = (inst.characteristics.positioning if inst.characteristics else None) or {}
    ratio = positioning.get("stroke_ratio")
    if ratio is None:
        return None
    try:
        from tofu.layers.typography import BOLD_RATIO, HEAVY_RATIO, LIGHT_RATIO
    except ImportError:
        return None
    ratio = float(ratio)
    if ratio >= HEAVY_RATIO:
        return 900
    if ratio >= BOLD_RATIO:
        return 700
    if ratio <= LIGHT_RATIO:
        return 300
    return 400


# Cap height as a share of the em.  Only used to put a face's advance
# metrics on the same footing as an ink-height measurement; the ratio is
# stable enough across text faces that the comparison stays meaningful.
CAP_HEIGHT_EM = 0.7


def _source_aspect(inst: InstText) -> Optional[float]:
    """Width-to-ink-height ratio of the lettering as it was captured.

    The same quantity _visual_score's `aspect` term compares, so the pool
    is narrowed on the axis it will later be judged on -- just without
    having to rasterise anything first.

    This is the fallback.  A caller holding the actual glyph mask should
    pass its measured aspect instead: a capture box is padded past the ink,
    and the difference is not small.  Measured on the plaque's "La rue" the
    box gives 3.88 where the ink gives 3.45, and ranking against the padded
    figure put Arial nearer the source than Centaur -- exactly backwards,
    since Centaur's rendered aspect matches the ink to within 1%.
    """
    size = (inst.characteristics.size if inst.characteristics else None) or 0
    if size <= 0 or inst.bounding_box is None or inst.bounding_box.width <= 0:
        return None
    return inst.bounding_box.width / float(size)


def _face_aspect(registry, font_path: str, text: str) -> Optional[float]:
    """What that ratio would be if this face set the same string.

    len(text) x mean advance gives the line's width in ems; dividing by the
    cap-height share converts the em to the ink height the source was
    measured in, so the em cancels and the two ratios are comparable.
    """
    if not text:
        return None
    advance = registry.avg_advance_em(font_path, text)
    if advance <= 0:
        return None
    return len(text) * advance / CAP_HEIGHT_EM


def _eligible_faces(
    registry, text: str, inst: InstText, source_aspect: Optional[float] = None,
) -> Iterable[Any]:
    """Narrow an installed library without treating coverage as visual rank.

    Ordering matters as much as the cap does.  This used to fall back to
    sorting by family NAME, which is not a typographic property at all, and
    with no weight target (the common case -- see _weight_target) every face
    scored an identical distance, so the sort collapsed to pure alphabet.
    Measured on the la-rue-sans-nom plaque: the eligible list ran "Agency
    FB" through "Gill Sans" and 134 of 219 installed families were never
    scored, Times New Roman, Palatino Linotype, Perpetua, Rockwell and
    Sylfaen among them.  A nearest-neighbour search that cannot reach two
    thirds of the library is not searching.

    So the tie-break is now the face's own natural width for this exact
    string, from the advance metrics the registry already read out of each
    hmtx table -- no rasterisation needed to rank the whole library.  A
    condensed face and an extended face are genuinely different candidates
    for lettering of a known width, and unlike a name that difference is
    measurable.
    """
    target = _weight_target(inst)
    if source_aspect is None:
        source_aspect = _source_aspect(inst)
    expected_italic = bool(inst.style_profile.italic) if inst.style_profile else False
    faces = []
    for face in faces_of(registry).values():
        if not all(ord(ch) in face.codepoints for ch in text if not ch.isspace()):
            continue
        weight_distance = abs(int(getattr(face, "weight_class", 400) or 400) - target) if target else 0
        italic = "italic" in (face.subfamily or "").lower() or "oblique" in (face.subfamily or "").lower()
        style_penalty = 120 if italic != expected_italic else 0
        aspect_distance = 0.0
        if source_aspect is not None:
            face_aspect = _face_aspect(registry, face.font_path, text)
            if face_aspect is not None:
                aspect_distance = abs(face_aspect - source_aspect) / max(1e-6, source_aspect)
        faces.append((weight_distance + style_penalty, aspect_distance, face.family.lower(), face))
    faces.sort(key=lambda item: (item[0], item[1], item[2], item[3].font_path.lower()))
    # Do not compare every face of a 300-font family collection.  The best
    # weight/style face for a family is the meaningful first-pass candidate.
    chosen, seen = [], set()
    for _, _, family, face in faces:
        if family in seen:
            continue
        chosen.append(face)
        seen.add(family)
        if len(chosen) >= MAX_LOCAL_FACES:
            break
    return chosen


def local_match(img: Any, inst: InstText, registry) -> Optional[Dict[str, Any]]:
    """Rank installed faces against a source instance's glyph silhouette."""
    if not registry or not inst.text or len(inst.text.strip()) < 2:
        return None
    try:
        import cv2
        import numpy as np
    except ImportError:
        return None
    try:
        source = text_mask(img, inst.bounding_box)
        if source is None:
            return None
        source = _tight(np.asarray(source, dtype=bool))
    except Exception:
        return None
    if source is None or int(source.sum()) < MIN_GLYPH_PIXELS or source.shape[0] < 8:
        return None

    # Rank against the INK's aspect, not the capture box's -- the box is
    # padded, and _source_aspect() records what that padding costs.
    source_aspect = source.shape[1] / max(1, source.shape[0])
    source_profile = _glyph_profile(source)

    # Two stages, because the typographic terms cost roughly fifteen times a
    # silhouette comparison (measured: 21ms against 1.4ms per face) and a
    # capture-time scan runs this for every region on the asset.  Stage one
    # renders and silhouette-scores the whole eligible pool; stage two pays
    # for the medial axis only on the shortlist.
    rendered: List[Tuple[float, Any, Any]] = []
    for face in _eligible_faces(registry, inst.text, inst, source_aspect=source_aspect):
        try:
            candidate = _render_mask(face.font_path, inst.text, source.shape[0])
            if candidate is None:
                continue
            recall, _ = _visual_score(source, candidate, typographic=False)
        except Exception:
            continue
        rendered.append((recall, face, candidate))
    rendered.sort(key=lambda item: (-item[0], item[1].family.lower()))

    ranked: List[Dict[str, Any]] = []
    for _, face, candidate in rendered[:PROFILE_CANDIDATES]:
        try:
            score, evidence = _visual_score(source, candidate, source_profile=source_profile)
        except Exception:
            continue
        ranked.append({
            "family": face.family,
            "subfamily": face.subfamily,
            "font_path": face.font_path,
            "license": "installed",
            "available": True,
            "score": score,
            "evidence": evidence,
        })
    ranked.sort(key=lambda candidate: (-candidate["score"], candidate["family"].lower()))
    ranked = ranked[:TOP_CANDIDATES]
    if not ranked:
        return None
    top = ranked[0]
    margin = top["score"] - (ranked[1]["score"] if len(ranked) > 1 else 0.0)
    confidence = max(0.0, min(1.0, 0.62 * top["score"] + 0.38 * min(1.0, margin / .16)))
    return {
        "schema": 1,
        "status": "matched" if confidence >= EXACT_CONFIDENCE and margin >= EXACT_MARGIN else "review",
        "provider": "local_glyph_retrieval",
        "confidence": round(confidence, 4),
        "margin": round(margin, 4),
        "source_text": inst.text,
        "candidates": ranked,
        "recommended_substitute": {
            "font_path": top["font_path"], "family": top["family"],
            "subfamily": top["subfamily"], "score": top["score"],
        },
    }


MAX_COHORT_CANDIDATES = 12


def _source_mask(img: Any, inst: InstText):
    """The observed glyph silhouette for one instance, or None."""
    import numpy as np

    try:
        source = text_mask(img, inst.bounding_box)
        if source is None:
            return None
        source = _tight(np.asarray(source, dtype=bool))
    except Exception:
        return None
    if source is None or int(source.sum()) < MIN_GLYPH_PIXELS or source.shape[0] < 8:
        return None
    return source


def agree_on_face(
    img: Any, manifest: TextManifest, cohorts: Sequence[Dict[str, Any]], registry,
) -> int:
    """Make every region Basil tied into one bouquet agree on one face.

    ``local_match`` ranks faces against a SINGLE region's silhouette, and a
    single region is a small sample of a typeface.  On the la-rue-sans-nom
    plaque -- one sign, one face -- "La rue" ranked Centaur top and
    "SANS-NOM" ranked Franklin Gothic Demi Cond top, because eight wide
    capitals and six mixed-case letters project different statistics.  Both
    answers cannot describe the same plaque, and a localiser who accepts
    them region by region gets a sign set in two typefaces.

    So: pool the candidates the members already surfaced, re-score every
    one of them against EVERY member's own pixels, and pick the face that
    best explains the bouquet as a whole.

    Two details make the aggregate meaningful:

      * Scores are normalised per region before being combined.  The
        achievable score depends on the string -- a long word can never
        reach a short word's Dice -- so raw scores from different regions
        are not on one scale.  Each face is judged on how close it comes
        to THAT region's own best, which is comparable.
      * The faces are ranked by their WORST member (maximin), with the
        mean only breaking ties.  The failure this pass exists to prevent
        is precisely "excellent on one region, wrong on another": on the
        plaque, Centaur is the top face for "La rue" at 1.00 normalised
        and the very worst for "SANS-NOM" at 0.57.  A mean lets a strong
        member carry a face that badly misdescribes its neighbour;
        requiring no member to be poorly served does not.

    Area weighting was tried and rejected.  It hands the vote to the
    largest box, but box size is not evidence quality: "SANS-NOM" is 66%
    of this sign's text area and is also all-capitals, and capitals are a
    far weaker typeface signature than the lowercase bowls and terminals
    in the smaller "La rue".  Weighting by area would rank the poorer
    witness higher for no defensible reason, so every member counts once.

    The per-region winner is preserved under ``region_substitute`` so the
    reconciliation stays auditable, and nothing is written to
    ``style_profile`` -- this remains evidence a human accepts, exactly as
    ``local_match`` is.

    Returns the number of instances whose evidence was updated.
    """
    if not registry or not cohorts:
        return 0
    try:
        import numpy as np  # noqa: F401
    except ImportError:
        return 0

    by_id = {inst.id: inst for inst in manifest.instances}
    updated = 0

    for cohort in cohorts:
        members = [
            by_id[rid] for rid in cohort.get("region_ids", [])
            if rid in by_id and by_id[rid].font_match
        ]
        if len(members) < 2:
            continue

        # Candidate pool: what the members themselves nominated.  A face
        # nobody ranked is not worth re-rendering for every member.
        pool: Dict[str, Dict[str, Any]] = {}
        tally: Dict[str, float] = {}
        for inst in members:
            for candidate in (inst.font_match or {}).get("candidates") or []:
                path = candidate.get("font_path")
                if not path:
                    continue  # contextual reference, not an installed face
                pool.setdefault(path, candidate)
                tally[path] = tally.get(path, 0.0) + float(candidate.get("score") or 0.0)
        if not pool:
            continue

        # A face must be able to set every member's text, or it cannot be
        # the sign's one face.
        needed = {ch for inst in members for ch in (inst.text or "") if not ch.isspace()}
        fonts = faces_of(registry)
        viable = []
        for path in pool:
            face = fonts.get(path)
            if face is not None and not all(ord(ch) in face.codepoints for ch in needed):
                continue
            viable.append(path)
        if not viable:
            continue
        viable.sort(key=lambda p: (-tally.get(p, 0.0), p.lower()))
        viable = viable[:MAX_COHORT_CANDIDATES]

        masks = {inst.id: _source_mask(img, inst) for inst in members}
        scored = [inst for inst in members if masks.get(inst.id) is not None]
        if len(scored) < 2:
            continue

        # One hand, one typographic reading.  Each member measures the sign's
        # contrast and serif structure off its own lettering, but they do not
        # measure it equally well: a line of lowercase carries bowls,
        # terminals and crossbars, and a line of capitals is nearly all stems
        # and diagonals.  Measured on the plaque, "La rue" reads contrast 2.33
        # with 46% of its skeleton available to the measurement while
        # "SANS-NOM" reads 1.49 on 33% -- and silhouette alone ranks the
        # correct serif 2nd of 206 on the first and 192nd on the second.  So
        # the bouquet pools its members' readings, each weighted by its own
        # evidence, and every member is then judged against THAT.  The line
        # that can see the typeface speaks for the line that cannot.
        profiles = {inst.id: _glyph_profile(masks[inst.id]) for inst in scored}

        def _pooled(value_of, weight_of) -> Optional[float]:
            pairs = [
                (value_of(p), weight_of(p)) for p in profiles.values()
                if value_of(p) is not None and weight_of(p) > 0
            ]
            if not pairs:
                return None
            total = sum(weight for _, weight in pairs)
            return sum(value * weight for value, weight in pairs) / max(1e-6, total)

        shared_profile = GlyphProfile(
            contrast=_pooled(lambda p: p.contrast, lambda p: p.contrast_sufficiency),
            contrast_sufficiency=max(p.contrast_sufficiency for p in profiles.values()),
            serif=_pooled(lambda p: p.serif, lambda p: p.serif_sufficiency),
            serif_sufficiency=max(p.serif_sufficiency for p in profiles.values()),
        )

        per_face: Dict[str, Dict[str, float]] = {}
        for path in viable:
            for inst in scored:
                source = masks[inst.id]
                try:
                    candidate = _render_mask(path, inst.text or "", source.shape[0])
                    if candidate is None:
                        continue
                    score, _ = _visual_score(source, candidate, source_profile=shared_profile)
                except Exception:
                    continue
                per_face.setdefault(path, {})[inst.id] = score
        # Only faces we could actually score on every member can be
        # compared; a partial column would flatter the face that failed.
        complete = {
            path: scores for path, scores in per_face.items()
            if len(scores) == len(scored)
        }
        if not complete:
            continue

        best_per_region = {
            inst.id: max(scores.get(inst.id, 0.0) for scores in complete.values()) or 1.0
            for inst in scored
        }

        def normalised(scores: Dict[str, float]) -> List[float]:
            return [score / best_per_region[rid] for rid, score in scores.items()]

        def rank(path: str) -> Tuple[float, float, int]:
            values = normalised(complete[path])
            # worst member first, mean as the tiebreak, then the members'
            # own nomination order so the choice is deterministic
            return (min(values), sum(values) / len(values), -viable.index(path))

        winner_path = max(complete, key=rank)
        winner_scores = complete[winner_path]
        winner_meta = pool[winner_path]
        worst, mean, _ = rank(winner_path)

        # Where the bouquet's answer overrules a region's own favourite,
        # say so and by how much: that is the reconciliation a reviewer
        # most needs to be able to check.
        dissent = []
        for inst in scored:
            own = (inst.font_match or {}).get("recommended_substitute") or {}
            if own.get("font_path") and own["font_path"] != winner_path:
                dissent.append({
                    "region_id": inst.id,
                    "preferred": own.get("family"),
                    "preferred_score": own.get("score"),
                    "cohort_score": round(winner_scores.get(inst.id, 0.0), 4),
                })

        for inst in members:
            evidence = inst.font_match or {}
            previous = evidence.get("recommended_substitute")
            if previous and "region_substitute" not in evidence:
                evidence["region_substitute"] = previous
            evidence["cohort"] = {
                "id": cohort.get("id"),
                "region_ids": [i.id for i in members],
                "method": "cohort_visual_consensus",
                "agreement": round(worst, 4),      # the worst-served member
                "mean_agreement": round(mean, 4),
                "per_region": {rid: round(s, 4) for rid, s in winner_scores.items()},
                "dissent": dissent,
                # what the bouquet decided the sign's typography IS, and
                # which member's evidence carried that decision
                "typography": {
                    "contrast": round(shared_profile.contrast, 4) if shared_profile.contrast is not None else None,
                    "serif": round(shared_profile.serif, 4) if shared_profile.serif is not None else None,
                    "sufficiency": {
                        rid: {
                            "contrast": round(p.contrast_sufficiency, 3),
                            "serif": round(p.serif_sufficiency, 3),
                        }
                        for rid, p in profiles.items()
                    },
                },
            }
            evidence["recommended_substitute"] = {
                "font_path": winner_path,
                "family": winner_meta.get("family"),
                "subfamily": winner_meta.get("subfamily"),
                "score": round(winner_scores.get(inst.id, 0.0), 4),
            }
            inst.font_match = evidence
            updated += 1

    return updated


def _looks_like_french_enamel_sign(img, manifest: TextManifest) -> bool:
    """Return a *style context*, never a geographic/font identification.

    Blue enamel plaques occur across many French municipalities.  They are
    not evidence that the sign is Parisian or that it used any particular
    foundry face, so callers may present only a clearly-labeled research
    reference and must not promote it as a detected font.
    """
    words = " ".join((inst.text or "").lower() for inst in manifest.instances)
    if not any(token in words.split() for token in ("rue", "avenue", "boulevard", "place", "quai")):
        return False
    try:
        import numpy as np
        arr = np.asarray(img)
        if arr.ndim != 3 or arr.shape[2] < 3:
            return False
        r, g, b = arr[..., 0], arr[..., 1], arr[..., 2]
        blue = ((b > r * 1.18) & (b > g * 1.08) & (b > 55)).mean()
        red = ((r > g * 1.3) & (r > b * .9) & (r > 85)).mean()
        return bool(blue > .08 and red > .002)
    except Exception:
        return False


def _french_enamel_reference() -> Dict[str, Any]:
    return {
        "family": "Plaak",
        "subfamily": None,
        "font_path": None,
        "license": "commercial",
        "available": False,
        "score": None,
        "source": "contextual_style_reference",
        "url": "https://www.205.tf/Plaak",
        "foundry": "205TF",
        "reason": "This is a licensed French street-sign lettering reference, not a detected match. Use catalog retrieval or human review before licensing.",
    }


def identify_manifest_fonts(asset: Any, manifest: TextManifest, registry) -> int:
    """Attach local glyph-retrieval evidence to every eligible instance."""
    try:
        from PIL import Image
        import numpy as np
        img = np.asarray(Image.open(asset).convert("RGB")) if not hasattr(asset, "shape") else asset
    except Exception:
        return 0
    enamel_context = _looks_like_french_enamel_sign(img, manifest)
    updated = 0
    for inst in manifest.instances:
        result = local_match(img, inst, registry)
        if result is None:
            continue
        if enamel_context:
            result["candidates"] = [_french_enamel_reference(), *result["candidates"]]
            result["contextual_candidates"] = ["Plaak"]
        inst.font_match = result
        updated += 1

    # A per-region winner is a small sample's opinion.  Reconcile regions
    # that Basil says were set by one hand before any of this reaches a
    # human, so the recommendation offered for a sign is consistent across
    # that sign.  Best-effort: a manifest with no bouquets, or a Basil that
    # cannot load, leaves the per-region evidence exactly as it was.
    try:
        from tofu.layers import basil

        agree_on_face(img, manifest, basil.bouquet(manifest), registry)
    except Exception as exc:
        # Best-effort, but never silent.  Swallowing this whole leaves a
        # sign recommending a different face per region with nothing
        # anywhere to say the reconciliation was even attempted, which is
        # indistinguishable from "these regions genuinely disagree".
        print(f"[tofu] font cohort reconciliation skipped: {type(exc).__name__}: {exc}")
    return updated


def external_catalog_match(asset: Any, manifest: TextManifest, registry) -> int:
    """Explicit WhatFontIs adapter for commercial/free catalog recognition.

    This function is intentionally opt-in.  It sends only an individual text
    crop and only after the UI receives an explicit consent action.  The API
    key remains server-side in ``TOFU_WHATFONTIS_API_KEY``; without it this
    records an actionable unavailable-provider state and performs no network
    request.
    """
    key = os.environ.get("TOFU_WHATFONTIS_API_KEY", "").strip()
    if not key:
        for inst in manifest.instances:
            if inst.font_match:
                inst.font_match["external_provider"] = {
                    "name": "WhatFontIs", "enabled": False,
                    "reason": "Set TOFU_WHATFONTIS_API_KEY to enable commercial-catalog matching.",
                }
        return 0
    try:
        from PIL import Image
        image = Image.open(asset).convert("RGB") if not hasattr(asset, "crop") else asset.convert("RGB")
    except Exception:
        return 0
    matched = 0
    for inst in manifest.instances:
        if not inst.text or not inst.font_match:
            continue
        box = inst.bounding_box
        crop = image.crop((max(0, box.x), max(0, box.y), max(0, box.x + box.width), max(0, box.y + box.height)))
        data = io.BytesIO()
        crop.save(data, format="PNG")
        payload = urllib.parse.urlencode({
            "API_KEY": key,
            "IMAGEBASE64": 1,
            "urlimagebase64": base64.b64encode(data.getvalue()).decode("ascii"),
            "NOTTEXTBOXSDETECTION": 0,
            "FREEFONTS": 0,
            "limit": TOP_CANDIDATES,
        }).encode("utf-8")
        try:
            request = urllib.request.Request("https://www.whatfontis.com/api2/", data=payload,
                                             headers={"Content-Type": "application/x-www-form-urlencoded"}, method="POST")
            with urllib.request.urlopen(request, timeout=20) as response:  # nosec B310: fixed HTTPS endpoint
                raw = json.loads(response.read().decode("utf-8"))
        except Exception as exc:
            inst.font_match["external_provider"] = {"name": "WhatFontIs", "enabled": True, "error": type(exc).__name__}
            continue
        candidates = []
        for item in raw if isinstance(raw, list) else []:
            candidates.append({
                "family": item.get("title", "Unknown"), "subfamily": None,
                "font_path": None,
                "license": "commercial" if str(item.get("type", "")).lower() == "commercial" else "free",
                "available": False,
                "score": None,
                "source": "whatfontis",
                "url": item.get("url"), "preview_url": item.get("image"),
                "foundry": item.get("site"),
            })
        if candidates:
            inst.font_match["external_candidates"] = candidates
            inst.font_match["external_provider"] = {"name": "WhatFontIs", "enabled": True}
            matched += 1
    return matched
