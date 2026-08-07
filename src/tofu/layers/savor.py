## 🍢 Savor — Cicerone's quality-control taste test
## vieuxtiful
"""
Savor is Cicerone's taste tester: before a batch of recognized text leaves
the kitchen, Savor samples every suspicious bite and only swallows a
correction when the evidence backs it up. It sits alongside Cicerone's
other post-recognition passes (second_look, prune_hallucinations, zoom,
adaptive) as one more quality-control step on the same raw output — not a
separate pipeline layer with its own MANUAL mode, since there's nothing
here a user would hand-author; it's pure QC on what the recognizer already
produced.

THE PROBLEM: CRNN+CTC recognizers (EasyOCR's default backend) decode
character-by-character with no language model and no notion of "does this
reading make sense." A glyph that's locally ambiguous (5/S, 0/O, 1/I, 8/B,
2/Z, 6/G) can be read wrong with HIGH confidence, because the model is
confident about the SHAPE it saw, not about whether the result is
semantically plausible — e.g. "Open 9am to 5pm" read as "Open 9am to
Spm" at 0.958 confidence. Cicerone's second_look() only re-reads regions
BELOW a confidence floor (0.55), so a case like this never gets a second
chance, and re-reading with the same model on the same pixels reliably
reproduces the same mistake anyway — a different SIGNAL is needed, not
another pass of the same recognizer.

THE MENU — courses deliberately kept separate so nothing gets
swallowed on a guess:

  0. AMUSE-BOUCHE (sniff_clumps/chew_clump): served before the menu
     proper, because it repairs the one thing every later course depends
     on — how many characters are actually there. A CTC decode can
     collapse two narrow neighbouring glyphs into a single label ("la"
     read as "J"), and from then on the plate carries more glyphs than
     the text claims characters, so every glyph-indexed course is
     addressing the wrong positions. The mark courses detect exactly
     this and correctly refuse to decide anything, which is safe but
     leaves the region permanently unrepairable. This course localizes
     the clump geometrically (word-gap segmentation, no lexicon), names
     the missing token from a stored phrase whose OTHER tokens match
     exactly, and only swallows when glyph shape AND an isolated re-read
     independently agree. See sniff_clumps/chew_clump for the split of
     evidence and why identity is never inferred from geometry.

  1. SNIFF (sniff_out): a quick, cheap smell test — is there a WHOLE
     token shaped like something a digit was expected in (currently: an
     hour immediately before "am"/"pm"), where swapping in a confusable
     glyph's digit counterpart would make it valid? This is context
     alone, and context alone is NOT enough to act on — a regex-only
     version of this idea was tried and rejected: it silently corrupts
     real text like "Sam" into "5am", because context can't tell "the
     recognizer was wrong" from "the recognizer was right and the
     context pattern is a coincidence." sniff_out() only proposes
     Morsels worth a closer look; it never rewrites anything.

  2. CHEW (chew_on): the actual bite. Segments the ambiguous glyph's own
     pixels out of the region (connected-component analysis of the
     shared `imaging.text_mask()`), renders BOTH candidate characters as
     reference glyphs at a matching size (`scribe._get_font()`), and
     compares shapes. Only a clear margin one way or the other settles
     it — a coin-flip-close comparison is left "still chewing"
     (inconclusive), not decided by tie-break.

  3. SWALLOW OR SPIT OUT (taste, the course orchestrator): a Morsel is
     only ever swallowed (the text gets rewritten) when chew_on()
     confirms the digit. Confirmed-letter Morsels are spat out — the
     text stands, nothing recorded. Still-chewing (inconclusive) Morsels
     are neither swallowed nor spat out — they're left on the plate:
     recorded on `InstText.ocr_correction` with applied=False, visible
     for human review instead of silently vanishing or silently guessing.

taste() is the module's single public entry point (mirrors every other
layer's one-verb API: cicerone.detect, scene.analyze, verify.assess,
memory.update/lookup) — called once by cicerone.detect(), on the FINAL
manifest, after every detection/refinement pass has already run.
"""

import re
from statistics import median
import unicodedata
from typing import Any, List, Optional

from tofu.core.types import BBox, ImageLike, InstText
from tofu.utils.imaging import load_rgb, text_mask
from tofu.utils.correction_resources import (
    diacritic_entries,
    fold_text,
    load_correction_resource,
    phrase_entries,
)

# glyph shapes a CRNN commonly confuses (letter <-> digit). bidirectional
# so an already-correct digit is never treated as "a letter that could be
# a digit" (see sniff_out: substitution only fires on LETTER positions).
CONFUSABLE_GLYPHS = {
    "5": "S", "S": "5",
    "0": "O", "O": "0",
    "1": "I", "I": "1",
    "8": "B", "B": "8",
    "2": "Z", "Z": "2",
    "6": "G", "G": "6",
}

# course 4 (sniff_dakuten/chew_dakuten): languages the dakuten course
# even applies to -- katakana/hiragana marks are a Japanese-specific
# orthographic feature.
DAKUTEN_LANGS = {"ja"}

# base kana -> (dakuten variant, handakuten variant or None). a small,
# linguistically-closed table of the consonant rows that take a mark --
# not a per-word answer key. katakana rows shown; hiragana mirrors them
# 1:1 since the mark works the same way in both scripts.
DAKUTEN_MAP = {
    "カ": ("ガ", None), "キ": ("ギ", None), "ク": ("グ", None), "ケ": ("ゲ", None), "コ": ("ゴ", None),
    "サ": ("ザ", None), "シ": ("ジ", None), "ス": ("ズ", None), "セ": ("ゼ", None), "ソ": ("ゾ", None),
    "タ": ("ダ", None), "チ": ("ヂ", None), "ツ": ("ヅ", None), "テ": ("デ", None), "ト": ("ド", None),
    "ハ": ("バ", "パ"), "ヒ": ("ビ", "ピ"), "フ": ("ブ", "プ"), "ヘ": ("ベ", "ペ"), "ホ": ("ボ", "ポ"),
    "ウ": ("ヴ", None),
    "か": ("が", None), "き": ("ぎ", None), "く": ("ぐ", None), "け": ("げ", None), "こ": ("ご", None),
    "さ": ("ざ", None), "し": ("じ", None), "す": ("ず", None), "せ": ("ぜ", None), "そ": ("ぞ", None),
    "た": ("だ", None), "ち": ("ぢ", None), "つ": ("づ", None), "て": ("で", None), "と": ("ど", None),
    "は": ("ば", "ぱ"), "ひ": ("び", "ぴ"), "ふ": ("ぶ", "ぷ"), "へ": ("べ", "ぺ"), "ほ": ("ぼ", "ぽ"),
    "う": ("ゔ", None),
}

# grammar for course 1 (sniff_out): a 1-2 char hour immediately before
# "am"/"pm". more digit-expected grammars (currency, ordinals, phone/
# address numbers) are a known coverage gap for a future course — see the
# plan doc; each one needs its own hand-authored pattern here.
_TIME_TOKEN = re.compile(r"^([A-Za-z0-9]{1,2})(am|pm)$", re.IGNORECASE)
_VALID_HOUR = re.compile(r"^(?:[1-9]|1[0-2])$")

# course 2 (chew_on) tuning: how much IoU advantage the winning glyph
# needs before it's "clearly" the answer, and how big a connected
# component has to be to count as a real glyph vs. noise speckle.
BITE_MARGIN = 0.12
MIN_MORSEL_AREA = 6         # px^2 floor for a connected component
TASTING_CANVAS = 24         # common square size both glyph masks are resized to for comparison
# chew_swaps small-glyph rescue: below this per-glyph extent (px), the
# crop is bicubically upscaled before masking -- tiny street-photo
# glyphs under-segment at native resolution (see chew_swaps).
GLYPH_UPSCALE_MIN_PX = 20
GLYPH_UPSCALE_FACTOR = 3

LATIN_DIACRITIC_RESOURCE = load_correction_resource("savor/latin_diacritics-1.0.0.json")
LATIN_DIACRITICS = diacritic_entries(LATIN_DIACRITIC_RESOURCE)

LATIN_PHRASE_RESOURCE = load_correction_resource("savor/latin_phrases-1.0.0.json")
LATIN_PHRASES = phrase_entries(LATIN_PHRASE_RESOURCE)
# The clump course reasons from "one glyph, one connected component", which
# holds for typeset Latin and NOT for CJK -- a single kanji is routinely
# several components (see _cluster_to_n_glyphs), so a component surplus there
# is normal rather than evidence of anything. Scoping the course to the
# languages the phrase resource actually covers keeps it from raising a
# permanent false alarm on every CJK region it cannot serve anyway.
PHRASE_LANGUAGES = {entry["language"] for entry in LATIN_PHRASES}

# course 0 (sniff_clumps/chew_clump) tuning. a word break is an outlier in
# the line's own gap distribution, so the ratio is what carries the rule and
# the pixel floor only guards a line whose glyphs nearly touch. the re-read
# pad is small on purpose: the point of the isolated crop is that it shows
# the clump WITHOUT its neighbours.
WORD_GAP_RATIO = 2.5
WORD_GAP_MIN_PX = 4
CLUMP_REREAD_PAD = 2

# OCR observability states are deliberately about evidence, not OCR
# confidence.  A high-confidence CTC decode can still be untrustworthy when
# its crop cannot resolve the marks or components that distinguish the read.
OCR_QUALITY_RELIABLE = "reliable"
OCR_QUALITY_REVIEW = "review_required"
OCR_QUALITY_UNRESOLVABLE = "unresolvable"
MARK_RESOLUTION_FLOOR = 3
LOW_GLYPH_HEIGHT = 8

# Case evidence is evaluated as a whole-token signature.  These are the
# lower-case letters whose expected vertical extent is genuinely diagnostic;
# dots and tall ascenders deliberately do not independently prove a case.
_X_HEIGHT = set("aceimnorsuvwxz")
_ASCENDERS = set("bdfhklt")
_DESCENDERS = set("gjpqy")
_CASE_MIN_LETTERS = 3


def _claimed_case_class(char: str) -> Optional[str]:
    if not char.isalpha():
        return None
    if char.isupper():
        return "tall"
    lower = char.lower()
    if lower in _X_HEIGHT:
        return "x"
    if lower in _ASCENDERS:
        return "tall"
    if lower in _DESCENDERS:
        return "desc"
    # i/j dots and unfamiliar scripts are intentionally non-diagnostic.
    return None


def _observed_case_signature(clusters: List[tuple]) -> List[str]:
    """Classify clusters relative to a robust line-level top/bottom band."""
    if not clusters:
        return []
    tops = [box[1] for box in clusters]
    bottoms = [box[3] for box in clusters]
    heights = [max(1, box[3] - box[1]) for box in clusters]
    # The upper band is the low quantile, not the median: in Title Case a
    # majority of x-height letters would otherwise redefine the baseline and
    # make the lowercase run look like capitals.  One cap anchor is enough;
    # automatic promotion below still requires that anchor in the OCR claim.
    top = sorted(tops)[max(0, len(tops) // 5 - 1)]
    bottom, h = median(bottoms), max(1.0, median(heights))
    signature: List[str] = []
    for _x0, y0, _x1, y1 in clusters:
        # A tall glyph touches both robust bands. X-height glyphs begin lower;
        # descenders also extend below the baseline.  Keep an "ambiguous"
        # result rather than inventing precision at low resolution.
        if y0 <= top + .14 * h and y1 >= bottom - .14 * h:
            signature.append("tall")
        elif y0 >= top + .20 * h and y1 > bottom + .12 * h:
            signature.append("desc")
        elif y0 >= top + .20 * h and y1 >= bottom - .14 * h:
            signature.append("x")
        else:
            signature.append("ambiguous")
    return signature


def case_signature_verdict(text: str, clusters: List[tuple]) -> tuple[Optional[str], dict]:
    """Return an unambiguous uppercase repair or review-only evidence.

    Evidence is combinatorial: every diagnostic letter in the token votes
    against both the claimed case pattern and the all-cap candidate.  This
    avoids treating one tall ``l`` or one short ``e`` as a case decision.
    Arbitrary re-casing is never automatic; only an all-cap physical plate can
    safely repair a mixed/lowercase OCR claim without language context.
    """
    chars = [ch for ch in text if not ch.isspace()]
    if len(chars) != len(clusters):
        return None, {"state": "unresolvable", "reason": "cluster_count_mismatch"}
    observed = _observed_case_signature(clusters)
    diagnostic = [
        (ch, want, got) for ch, want, got in zip(chars, map(_claimed_case_class, chars), observed)
        if want is not None and got != "ambiguous"
    ]
    letters = [item for item in diagnostic if item[0].isalpha()]
    if len(letters) < _CASE_MIN_LETTERS:
        return None, {"state": "unresolvable", "reason": "insufficient_diagnostic_letters"}
    claimed = sum(want == got or (want == "tall" and got == "tall") for _, want, got in letters) / len(letters)
    upper = sum(got == "tall" for _, _, got in letters) / len(letters)
    evidence = {
        "state": "review", "claimed_signature": "".join(want[0] for _, want, _ in letters),
        "observed_signature": "".join(got[0] for _, _, got in letters),
        "claimed_score": round(claimed, 3), "upper_score": round(upper, 3),
    }
    candidate = text.upper()
    has_cap_anchor = any(ch.isupper() for ch, _, _ in letters)
    if candidate != text and has_cap_anchor and upper >= .80 and upper >= claimed + .35:
        evidence["state"] = "upper_confirmed"
        return candidate, evidence
    return None, evidence


def _raw_components(np, cv2, mask, vertical: bool = False) -> List[dict]:
    """Raw connected components with area retained for detached-mark scans."""
    m8 = mask.astype(np.uint8) * 255
    n, _labels, stats, _centroids = cv2.connectedComponentsWithStats(m8, connectivity=8)
    components = []
    for index in range(1, n):
        area = int(stats[index, cv2.CC_STAT_AREA])
        if area < MIN_MORSEL_AREA:
            continue
        x, y = int(stats[index, cv2.CC_STAT_LEFT]), int(stats[index, cv2.CC_STAT_TOP])
        w, h = int(stats[index, cv2.CC_STAT_WIDTH]), int(stats[index, cv2.CC_STAT_HEIGHT])
        components.append({"box": (x, y, x + w, y + h), "area": area})
    components.sort(key=(lambda item: item["box"][1]) if vertical else (lambda item: item["box"][0]))
    return components


def _separate_marks(raw_components: List[dict]) -> tuple[List[dict], List[tuple[int, int]]]:
    """Split raw components into base glyphs and detached above-base marks.

    Deliberately takes no claimed character count: the segmentation course
    exists precisely BECAUSE that count disagrees with the plate, so it needs
    to see the same base/mark split that the mark courses use, without the
    reconciliation gate that (correctly) makes those courses give up.
    Returns ``(bases, detached)`` where ``detached`` is
    [(mark raw index, base raw index)] into ``raw_components``.
    """
    if not raw_components:
        return [], []
    areas = [item["area"] for item in raw_components]
    # A dotted i or diaeresis can make small components the majority.  Use
    # the upper half as the body scale, otherwise the marks redefine the
    # median and stop looking small.
    median_area = max(1.0, median(sorted(areas)[len(areas) // 2:]))
    detached: List[tuple[int, int]] = []  # (mark raw index, base raw index)
    for mark_index, mark in enumerate(raw_components):
        mx0, my0, mx1, my1 = mark["box"]
        if mark["area"] > .35 * median_area:
            continue
        candidates = []
        for base_index, base in enumerate(raw_components):
            if base_index == mark_index or base["area"] <= mark["area"]:
                continue
            bx0, by0, bx1, _by1 = base["box"]
            overlap = max(0, min(mx1, bx1) - max(mx0, bx0))
            if my1 <= by0 and overlap / max(1, mx1 - mx0) >= .45:
                candidates.append((overlap, -abs((mx0 + mx1) - (bx0 + bx1)), base_index))
        if candidates:
            detached.append((mark_index, max(candidates)[2]))
    mark_indices = {mark for mark, _base in detached}
    bases = [item for index, item in enumerate(raw_components) if index not in mark_indices]
    return bases, detached


def _detached_mark_scan(raw_components: List[dict], text: str) -> Optional[dict]:
    """Find excess detached above-base marks without consulting a lexicon.

    v1 is intentionally limited to detached marks above a Latin base glyph.
    Attached marks such as cedilla need a below-baseline ink-profile course.
    """
    chars = [char for char in text if not char.isspace()]
    if not chars or not raw_components:
        return None
    bases, detached = _separate_marks(raw_components)
    if len(bases) != len(chars):
        return None
    bases.sort(key=lambda item: item["box"][0])
    base_to_position = {id(item): index for index, item in enumerate(bases)}
    excess: dict[int, List[dict]] = {index: [] for index in range(len(chars))}
    for mark_index, base_index in detached:
        base = raw_components[base_index]
        position = base_to_position.get(id(base))
        if position is None:
            continue
        expected = 1 if chars[position].casefold() in {"i", "j"} else 0
        # Preserve deterministic left-to-right ordering for multiple dots.
        if len([pair for pair in detached if pair[1] == base_index]) > expected:
            excess[position].append(raw_components[mark_index])
    return {"base_components": [item["box"] for item in bases], "excess_marks": excess}


def _mark_zone_verdict(scan: Optional[dict], position: int) -> Optional[bool]:
    """Whether an excess detached mark is visibly present at one glyph.

    ``None`` means the plate cannot resolve a ~3px mark; callers must never
    reinterpret it as absence.  ``False`` requires a clear-enough base zone.
    """
    if scan is None or position >= len(scan["base_components"]):
        return None
    _x0, y0, _x1, y1 = scan["base_components"][position]
    mark_floor = 3
    marks = scan["excess_marks"].get(position, [])
    if marks:
        heights = [item["box"][3] - item["box"][1] for item in marks]
        return True if max(heights, default=0) >= mark_floor else None
    # The expected mark would be roughly 10-15% of the base height.  Do not
    # claim a clear zone when that signal is below physical resolution.
    return False if (y1 - y0) * .12 >= mark_floor else None


def _plate_clusters(asset: ImageLike, inst: InstText):
    """Return one stable mask/component/cluster view for Savor courses.

    Case and detached-mark verification must inspect exactly the same plate.
    Rebuilding components independently can merge a mark with a different
    neighbour and silently move a character position, so this function is the
    sole geometry boundary for those courses.
    """
    try:
        import cv2
        import numpy as np
    except ImportError:
        return None
    text = inst.text or ""
    if not text:
        return None
    image = load_rgb(asset)
    if image is None:
        return None
    mask = text_mask(image, inst.bounding_box, refine=True)
    if mask is None:
        return None
    vertical = _is_vertical_instance(inst.bounding_box)
    raw_components = _raw_components(np, cv2, mask, vertical=vertical)
    raw_boxes = [item["box"] for item in raw_components]
    clusters = _cluster_to_n_glyphs(raw_boxes, len(text.replace(" ", "")), vertical)
    if clusters is None:
        return None
    return mask, raw_components, clusters


def assess_ocr_quality(asset: ImageLike, inst: InstText, plate=None,
                       engine: Optional[Any] = None) -> dict:
    """Produce a deterministic, per-region OCR observability record.

    This is intentionally evaluated on the detected crop rather than on an
    asset-wide pixel threshold: a large image may contain unreadably small
    lettering, while a small but tightly cropped sign may be perfectly
    usable.  The result never discards a region.  It tells the UI whether a
    user needs to review it and tells Savor why an otherwise plausible repair
    was withheld.
    """
    evidence: dict[str, Any] = {
        "source_dimensions": None,
        "region_dimensions": {"width": inst.bounding_box.width, "height": inst.bounding_box.height},
        "estimated_glyph_height": None,
        "contrast": None,
        "sharpness": None,
        "raw_component_count": 0,
        "recognized_glyph_count": len([char for char in (inst.text or "") if not char.isspace()]),
        "component_surplus": None,
        "engine": getattr(engine, "name", "unavailable") if engine is not None else "unavailable",
        "reread_available": engine is not None,
    }
    reasons: list[str] = []
    image = load_rgb(asset)
    if image is None:
        return {"state": OCR_QUALITY_UNRESOLVABLE, "reasons": ["asset_unreadable"], **evidence}
    image_h, image_w = image.shape[:2]
    evidence["source_dimensions"] = {"width": int(image_w), "height": int(image_h)}
    if inst.bounding_box.width < 3 or inst.bounding_box.height < 3:
        return {"state": OCR_QUALITY_UNRESOLVABLE, "reasons": ["region_too_small"], **evidence}

    # These two image statistics are diagnostic only; they are intentionally
    # not standalone gates.  Geometry is the defensible decision signal.
    try:
        import cv2
        import numpy as np
        x0, y0 = max(0, inst.bounding_box.x), max(0, inst.bounding_box.y)
        x1, y1 = min(image_w, inst.bounding_box.x + inst.bounding_box.width), min(image_h, inst.bounding_box.y + inst.bounding_box.height)
        crop = image[y0:y1, x0:x1]
        if crop.size:
            gray = cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY)
            evidence["contrast"] = round(float(np.percentile(gray, 95) - np.percentile(gray, 5)), 2)
            evidence["sharpness"] = round(float(cv2.Laplacian(gray, cv2.CV_64F).var()), 2)
    except Exception:
        pass

    if plate is None:
        return {"state": OCR_QUALITY_UNRESOLVABLE, "reasons": ["geometry_unavailable"], **evidence}
    _mask, raw_components, clusters = plate
    bases, _marks = _separate_marks(raw_components)
    evidence["raw_component_count"] = len(raw_components)
    evidence["base_component_count"] = len(bases)
    claimed = evidence["recognized_glyph_count"]
    surplus = len(bases) - claimed
    evidence["component_surplus"] = surplus
    heights = [max(1, box[3] - box[1]) for box in (clusters or [])]
    if heights:
        evidence["estimated_glyph_height"] = round(float(median(heights)), 2)
        if evidence["estimated_glyph_height"] < MARK_RESOLUTION_FLOOR:
            reasons.append("below_mark_resolution")
        elif evidence["estimated_glyph_height"] < LOW_GLYPH_HEIGHT:
            reasons.append("low_effective_glyph_height")
    else:
        reasons.append("geometry_unavailable")
    if claimed and surplus != 0:
        # Component disagreement is always useful provenance, but it is NOT
        # automatically a review-worthy OCR error.  Serif terminals, a Q
        # tail, and weathered/engraved strokes routinely produce extra
        # connected components in an otherwise certain read.  Only a
        # *localized* disagreement (or a weak read) becomes a review gate.
        reasons.append("component_count_mismatch")
        clump_evidence: dict[str, Any] = {}
        if surplus > 0:
            try:
                _clumps, clump_evidence = sniff_clumps(
                    inst.text or "", plate, inst.detected_language or inst.language,
                    vertical=_is_vertical_instance(inst.bounding_box),
                )
            except Exception:
                clump_evidence = {"state": "unavailable"}
            evidence["segmentation_evidence"] = clump_evidence
            localized = clump_evidence.get("state") == "clump_unresolved"
            weak_read = (inst.confidence or 0.0) < 0.70
            if localized or weak_read:
                reasons.append("ambiguous_segmentation")
                if engine is None:
                    reasons.append("engine_unavailable")
    elif not claimed:
        reasons.append("no_recognized_glyphs")

    if any(reason in {"below_mark_resolution", "geometry_unavailable", "no_recognized_glyphs"} for reason in reasons):
        state = OCR_QUALITY_UNRESOLVABLE
    # ``component_count_mismatch`` by itself is diagnostic provenance, not a
    # gate: a high-confidence Q tail or fractured serif must not manufacture
    # a review task.  Every other remaining reason carries independent risk.
    elif any(reason != "component_count_mismatch" for reason in reasons):
        state = OCR_QUALITY_REVIEW
    else:
        state = OCR_QUALITY_RELIABLE
    # Preserve order while avoiding duplicate user-facing explanations.
    return {"state": state, "reasons": list(dict.fromkeys(reasons)), **evidence}


def chew_case(asset: ImageLike, inst: InstText, plate=None) -> tuple[Optional[str], dict]:
    """Measure the physical case signature of one Latin token/region."""
    text = inst.text or ""
    if not text or len(text.split()) != 1:
        return None, {"state": "unresolvable", "reason": "not_single_token"}
    plate = plate or _plate_clusters(asset, inst)
    if plate is None:
        return None, {"state": "unresolvable", "reason": "clusters_unresolvable"}
    _mask, raw_components, clusters = plate
    scan = _detached_mark_scan(raw_components, text)
    # Detached marks must not lift a cap/x-height band's top.  If the raw
    # geometry cannot reconcile to the claimed character count, retain the
    # legacy clusters but never manufacture a mark decision.
    case_boxes = scan["base_components"] if scan is not None else clusters
    return case_signature_verdict(text, case_boxes)


def _word_groups(bases: List[dict]) -> List[List[dict]]:
    """Group base glyphs into words by inter-glyph gap.

    External word segmentation of a text line: within a word the gaps cluster
    tightly around the face's own tracking, and a word break is the outlier
    above them (the classic gap-metric approach, Seni & Cohen 1994).  The
    threshold is scaled from the line's OWN median gap rather than fixed in
    pixels, so it holds at any point size; ``WORD_GAP_MIN_PX`` only stops a
    line whose glyphs nearly touch from declaring every gap a word break.
    """
    ordered = sorted(bases, key=lambda item: item["box"][0])
    if len(ordered) < 2:
        return [list(ordered)] if ordered else []
    gaps = [ordered[index + 1]["box"][0] - ordered[index]["box"][2]
            for index in range(len(ordered) - 1)]
    threshold = max(WORD_GAP_MIN_PX, WORD_GAP_RATIO * median(gaps))
    groups: List[List[dict]] = [[ordered[0]]]
    for gap, item in zip(gaps, ordered[1:]):
        if gap >= threshold:
            groups.append([])
        groups[-1].append(item)
    return groups


class ClumpMorsel:
    """one whitespace token the recognizer served as a single character
    where the plate plainly carries several glyphs -- a clump it never
    separated. unlike Morsel (one glyph, two candidate identities) the
    CARDINALITY is what is wrong here, so this carries the observed glyph
    boxes as well as the token the phrase resource proposes in its place."""
    __slots__ = ("token_index", "token_start", "token_text", "glyph_boxes",
                 "phrase", "recovered", "language")

    def __init__(self, token_index: int, token_start: int, token_text: str,
                 glyph_boxes: List[tuple], phrase: str, recovered: str,
                 language: Optional[str]):
        self.token_index = token_index      # index among the text's whitespace tokens
        self.token_start = token_start      # offset of the token within the full text
        self.token_text = token_text        # the single character the recognizer emitted, e.g. "J"
        self.glyph_boxes = glyph_boxes      # observed base-glyph boxes in mask-local coords
        self.phrase = phrase                # the stored phrase that supplied the anchor, e.g. "de la"
        self.recovered = recovered          # the token proposed in place of the clump, e.g. "la"
        self.language = language


def sniff_clumps(text: str, plate, language: Optional[str],
                 vertical: bool = False) -> tuple[List[ClumpMorsel], dict]:
    """course 0 (amuse-bouche): is a whitespace token a CLUMP -- one emitted
    character sitting on several separate glyphs?

    Served before the tasting menu proper because every later course is
    indexed by glyph position: while the plate carries more base glyphs than
    the recognizer claimed characters, the mark courses cannot reconcile the
    two and correctly refuse to decide anything (see ``_detached_mark_scan``).
    Repairing the cardinality first is what lets those courses run at all.

    Localization is purely geometric and needs no lexicon: split the line's
    base glyphs into words by gap (``_word_groups``), align those groups
    1:1 against the recognizer's OWN whitespace tokens, and require exactly
    one token to disagree with its group's glyph count.  A disagreement
    anywhere else means the line does not decompose cleanly and nothing is
    proposed.

    Identity is never inferred from geometry.  It comes only from a stored
    phrase whose OTHER tokens match the recognized text exactly and whose
    token in the disputed slot has exactly as many characters as the plate
    shows glyphs.  Returns ``(morsels, evidence)``; the evidence is recorded
    for review even when no correction is proposable, so a surplus never
    disappears silently.
    """
    evidence: dict[str, Any] = {"state": "reconciled"}
    if plate is None or vertical or not text.strip():
        return [], {"state": "unresolvable", "reason": "no_horizontal_plate"}
    if language not in PHRASE_LANGUAGES:
        # Out of scope entirely: the glyph-per-component premise this course
        # rests on does not hold for every script (see PHRASE_LANGUAGES).
        return [], {"state": "unresolvable", "reason": "language_out_of_course_scope"}
    _mask, raw_components, _clusters = plate
    bases, _detached = _separate_marks(raw_components)
    spans = [(match.start(), match.group()) for match in re.finditer(r"\S+", text)]
    tokens = [token for _start, token in spans]
    claimed = sum(len(token) for token in tokens)
    surplus = len(bases) - claimed
    if surplus <= 0:
        return [], evidence
    evidence = {"state": "surplus_unlocalized", "claimed_glyphs": claimed,
                "observed_glyphs": len(bases), "surplus": surplus}

    groups = _word_groups(bases)
    if len(groups) != len(tokens):
        evidence["reason"] = "word_groups_do_not_align_to_tokens"
        return [], evidence
    evidence["observed_word_glyphs"] = [len(group) for group in groups]
    evidence["claimed_word_glyphs"] = [len(token) for token in tokens]
    disputed = [index for index, (group, token) in enumerate(zip(groups, tokens))
                if len(group) != len(token)]
    if len(disputed) != 1:
        evidence["reason"] = "surplus_not_isolated_to_one_token"
        return [], evidence

    index = disputed[0]
    observed = len(groups[index])
    evidence.update({"state": "clump_unresolved", "token_index": index,
                     "token_text": tokens[index], "observed_token_glyphs": observed})
    # v1 scope: one emitted character standing in for several glyphs -- the
    # measured failure ("la" decoded as a single "J").  A clump spanning two
    # or more emitted characters has no single span to re-read and no
    # unambiguous claimed shape to argue against; that is a later course.
    if len(tokens[index]) != 1 or observed < 2:
        evidence["reason"] = "clump_is_not_a_single_emitted_character"
        return [], evidence

    recoveries: set[str] = set()
    phrases: set[str] = set()
    for entry in LATIN_PHRASES:
        if entry["language"] != language:
            continue
        folded, width = entry["folded"], len(entry["folded"])
        # Every window of the phrase that could cover the disputed token.
        for start in range(max(0, index - width + 1), min(index, len(tokens) - width) + 1):
            slot = index - start
            # The plate decides how many characters the recovered token may
            # have; the resource only decides WHICH characters those are.
            if len(entry["tokens"][slot]) != observed:
                continue
            anchors = [tokens[start + offset] for offset in range(width) if offset != slot]
            if any(fold_text(tokens[start + offset]) != folded[offset]
                   for offset in range(width) if offset != slot):
                continue
            recovered = entry["tokens"][slot]
            # Mirror the anchors' case: a stored lowercase article inserted
            # into an all-cap plate would otherwise read "DE la RUE".
            cased = [anchor for anchor in anchors if any(char.isalpha() for char in anchor)]
            if cased and all(anchor.isupper() for anchor in cased):
                recovered = recovered.upper()
            recoveries.add(recovered)
            phrases.add(entry["phrase"])
    if len(recoveries) != 1:
        # Zero matches: nothing stored supports this slot.  More than one:
        # ambiguous, and this course never breaks a tie by preference.
        evidence["reason"] = ("no_stored_phrase_supports_the_slot" if not recoveries
                              else "stored_phrases_disagree_on_the_slot")
        return [], evidence

    recovered = recoveries.pop()
    evidence.update({"candidate_token": recovered, "phrase": sorted(phrases)[0]})
    morsel = ClumpMorsel(
        token_index=index, token_start=spans[index][0], token_text=tokens[index],
        glyph_boxes=[item["box"] for item in groups[index]],
        phrase=sorted(phrases)[0], recovered=recovered, language=language,
    )
    return [morsel], evidence


def _reread_span(asset: ImageLike, engine: Any, bbox: BBox, span: tuple) -> Optional[str]:
    """Re-read ONE clump's own pixels as an isolated crop.

    The recognizer that produced the clump saw it inside a full line, where
    a CTC decode can collapse two narrow glyphs into one label.  Handing it
    the span alone, upscaled to the recognizer's operating height, is a
    genuinely different view of the same pixels -- the same coarse-to-fine
    argument ``second_look``/``zoom_detect`` already rest on.
    """
    if engine is None or bbox is None:
        return None
    try:
        from tofu.layers.cicerone import _compose_crop_text
        x0, y0, x1, y1 = span
        # The span is mask-local, and ``text_mask`` crops from the bbox
        # CLAMPED to the image.  Reproduce that same origin, or a region
        # hanging off the top/left edge would re-read the wrong pixels.
        crop_box = BBox(x=max(0, bbox.x) + x0, y=max(0, bbox.y) + y0,
                        width=x1 - x0, height=y1 - y0)
        composed = _compose_crop_text(engine.detect_in_regions(asset, [crop_box],
                                                              pad=CLUMP_REREAD_PAD)[0])
    except Exception:
        return None
    return (composed.text or "").strip() if composed else None


def chew_clump(asset: ImageLike, inst: InstText, morsel: ClumpMorsel, plate,
               engine: Optional[Any] = None,
               font_registry: Optional[Any] = None) -> tuple[Optional[bool], dict]:
    """course 0's bite: does the clump really separate into the proposed
    glyphs, or did the recognizer read one character correctly?

    Two independent signals must agree, and neither alone may swallow:

      SHAPE -- each observed glyph is compared against a reference render of
      the character the phrase proposes for it AND against the character the
      recognizer emitted, and the whole span is compared against that emitted
      character as one glyph.  Every proposed glyph must beat the emitted one
      on its own pixels, and the split reading must beat the single-glyph
      reading over the span.  This is what makes a genuine one-character read
      safe: a real "J" scores best as one glyph across the whole span.

      CARDINALITY -- an isolated re-read of the span must independently agree
      on HOW MANY characters are there.  It is deliberately not required to
      agree on WHICH: measured on the production case, the isolated crop of
      "la" re-reads as "Io", an I/l homoglyph confusion.  Demanding identity
      from the same recognizer that produced the clump would veto every true
      positive, while the count -- the thing actually in dispute -- is stable.

    WHAT SETTLES IDENTITY, AND WHAT CANNOT.  The pixels prove the cardinality
    and rule out the merged reading; the stored phrase names the token.  That
    division is not laziness, it is the same limit ``chew_dakuten`` already
    documents, re-measured here on the production plate -- three shape-based
    identity tests were built and rejected:

      * per-glyph IoU against the Latin alphabet, expecting the proposal to
        rank first: the true "a" scored 0.697 for "a" but 0.802 for "l"/"I".
      * aspect-ratio agreement, to recover what the shared 24x24 canvas
        normalizes away: the plaque's face is condensed (caps 12x35), so the
        observed proportions disagree with ANY reference font's by more than
        the wrong candidates do.
      * span-level IoU against the whole rendered candidate string, taking
        the argmax over its single-substitution neighbourhood: on the true
        plate "lJ" (0.636) outscored the true "la" (0.594).

    At this resolution a stretched-mask IoU separates "one glyph here" from
    "two glyphs here" decisively, and does not separate "la" from "lo".  So
    identity is gated instead by the anchors: every OTHER token of the stored
    phrase must match the recognized text exactly, in the right language.
    A plate genuinely reading "de lo ..." would be mis-repaired; the ledger
    records the resource identity and every score behind such a decision.
    Do not re-attempt shape-based identity here without a different metric.

    Returns ``(verdict, evidence)``.  ``None`` means unresolved and must
    never be read as confirmation.
    """
    evidence: dict[str, Any] = {"phrase": morsel.phrase, "candidate_token": morsel.recovered}
    if plate is None or len(morsel.recovered) != len(morsel.glyph_boxes):
        return None, {**evidence, "state": "unresolvable", "reason": "no_plate"}
    try:
        import cv2
        import numpy as np
    except ImportError:
        return None, {**evidence, "state": "unresolvable", "reason": "no_cv2"}
    mask, _raw_components, _clusters = plate
    claimed_char = morsel.token_text

    span = (min(box[0] for box in morsel.glyph_boxes), min(box[1] for box in morsel.glyph_boxes),
            max(box[2] for box in morsel.glyph_boxes), max(box[3] for box in morsel.glyph_boxes))
    span_bite = mask[span[1]:span[3], span[0]:span[2]]
    span_reference = _reference_bite(np, claimed_char, span[3] - span[1], font_registry, morsel.language)
    if span_bite.size == 0 or not span_bite.any() or span_reference is None:
        return None, {**evidence, "state": "unresolvable", "reason": "span_unreadable"}
    merged_score = _flavor_match(np, cv2, span_bite, span_reference)

    split_scores, rival_scores = [], []
    for (x0, y0, x1, y1), char in zip(morsel.glyph_boxes, morsel.recovered):
        bite = mask[y0:y1, x0:x1]
        proposed = _reference_bite(np, char, y1 - y0, font_registry, morsel.language)
        rival = _reference_bite(np, claimed_char, y1 - y0, font_registry, morsel.language)
        if bite.size == 0 or not bite.any() or proposed is None or rival is None:
            return None, {**evidence, "state": "unresolvable", "reason": "glyph_unreadable"}
        split_scores.append(_flavor_match(np, cv2, bite, proposed))
        rival_scores.append(_flavor_match(np, cv2, bite, rival))
    evidence.update({
        "merged_score": round(merged_score, 3),
        "split_scores": [round(score, 3) for score in split_scores],
        "rival_scores": [round(score, 3) for score in rival_scores],
    })
    shape_supports_split = (
        all(split >= rival + BITE_MARGIN for split, rival in zip(split_scores, rival_scores))
        and min(split_scores) >= merged_score + BITE_MARGIN
    )
    if not shape_supports_split:
        # The emitted character holds up on its own pixels: spit the
        # proposal out rather than leaving it as an open question.
        if merged_score >= min(split_scores) + BITE_MARGIN:
            return False, {**evidence, "state": "merged_reading_confirmed"}
        return None, {**evidence, "state": "shape_evidence_inconclusive"}

    reread = _reread_span(asset, engine, inst.bounding_box, span)
    if not reread:
        return None, {**evidence, "state": "reread_unavailable"}
    reread_glyphs = len([char for char in reread if not char.isspace()])
    evidence.update({"reread_text": reread, "reread_glyphs": reread_glyphs,
                     "reread_matches_identity": fold_text(reread) == fold_text(morsel.recovered)})
    if reread_glyphs != len(morsel.glyph_boxes):
        return None, {**evidence, "state": "reread_disagrees_on_glyph_count"}
    return True, {**evidence, "state": "split_confirmed"}


def _latin_diacritic_proposals(text: str, language: Optional[str]) -> List[dict]:
    """Exact stored-form lookup; never runtime-fold arbitrary OCR text."""
    if not text or not language:
        return []
    proposals = []
    offset = 0
    for token in text.split(" "):
        if not token:
            offset += 1
            continue
        matches = [entry for entry in LATIN_DIACRITICS
                   if entry["language"] == language and entry["folded"] == token.casefold()]
        # Resource loading rejects duplicate keys.  Keep this fail-closed path
        # for aggregated/future resources where ambiguity can still arise.
        if len(matches) == 1:
            canonical = matches[0]["text"]
            if len(token) == len(canonical) and token != canonical:
                positions = []
                valid = True
                for index, (seen, proposed) in enumerate(zip(token, canonical)):
                    seen_nfd, proposed_nfd = unicodedata.normalize("NFD", seen), unicodedata.normalize("NFD", proposed)
                    if seen_nfd[0].casefold() != proposed_nfd[0].casefold():
                        valid = False; break
                    marks = "".join(char for char in proposed_nfd[1:] if unicodedata.combining(char))
                    if seen != proposed:
                        if not marks and seen.casefold() != proposed.casefold():
                            valid = False; break
                        if marks:
                            positions.append((offset + index, marks))
                if valid and positions:
                    proposals.append({"token": token, "canonical": canonical, "positions": positions,
                                      "resource": matches[0]})
        offset += len(token) + 1
    return proposals


def _glyph_index(text: str, position: int) -> int:
    """Map a position in ``text`` to its index among non-space characters.

    Proposals address ``text`` (spaces included) because that is what the
    composed correction is written back into; every geometry structure here
    is indexed by GLYPH, which spaces do not occupy.  Conflating the two
    silently checks the wrong glyph's zone the moment a region holds more
    than one word.
    """
    return sum(1 for char in text[:position] if not char.isspace())


def chew_accents(inst: InstText, plate, positions: List[tuple[int, str]]) -> dict[int, Optional[bool]]:
    """Verify detached above-base marks only; attached marks are v2 scope."""
    if plate is None:
        return {position: None for position, _marks in positions}
    _mask, raw_components, _clusters = plate
    text = inst.text or ""
    scan = _detached_mark_scan(raw_components, text)
    return {
        position: _mark_zone_verdict(scan, _glyph_index(text, position))
        for position, _marks in positions
    }


def _compose_marks(text: str, positions: List[tuple[int, str]]) -> str:
    chars = list(text)
    for position, marks in positions:
        chars[position] = unicodedata.normalize("NFC", unicodedata.normalize("NFD", chars[position])[0] + marks)
    return "".join(chars)


def _record_correction(inst: InstText, *, applied: bool, original_text: str,
                       corrected_text: Optional[str] = None,
                       candidate_text: Optional[str] = None,
                       reason: str, course: str, **evidence: Any) -> None:
    """Append a correction step while preserving the OCR's earliest text.

    ``ocr_correction`` remains a dict for manifest compatibility.  The latest
    correction stays at the top level for existing UI consumers; ``steps`` is
    an append-only audit ledger for courses that co-fire on one instance.
    """
    prior = inst.ocr_correction or {}
    steps = list(prior.get("steps") or [])
    if prior and not steps:
        steps.append({key: value for key, value in prior.items() if key != "steps"})
    step: dict[str, Any] = {
        "course": course, "applied": applied, "original_text": original_text,
        "reason": reason, **evidence,
    }
    if corrected_text is not None:
        step["corrected_text"] = corrected_text
    if candidate_text is not None:
        step["candidate_text"] = candidate_text
    steps.append(step)
    record: dict[str, Any] = {
        "applied": applied,
        "original_text": prior.get("original_text", original_text),
        "reason": reason,
        "steps": steps,
        "course": course,
        **evidence,
    }
    if corrected_text is not None:
        record["corrected_text"] = corrected_text
    if candidate_text is not None:
        record["candidate_text"] = candidate_text
    inst.ocr_correction = record


class Morsel:
    """one proposed correction, waiting to be chewed on. holds enough
    context to locate the ambiguous glyph's own pixels later and to
    reconstruct the corrected string if course 2 confirms it."""
    __slots__ = ("token_start", "token_text", "char_pos_in_token", "orig_char", "digit_char", "corrected_token")

    def __init__(self, token_start: int, token_text: str, char_pos_in_token: int,
                 orig_char: str, digit_char: str, corrected_token: str):
        self.token_start = token_start              # offset of the token within the instance's full text
        self.token_text = token_text                # the token as originally recognized, e.g. "Spm"
        self.char_pos_in_token = char_pos_in_token   # index of the ambiguous char WITHIN the token
        self.orig_char = orig_char                   # what the recognizer said, e.g. "S"
        self.digit_char = digit_char                 # what it might actually be, e.g. "5"
        self.corrected_token = corrected_token        # token with the swap applied, e.g. "5pm"


def sniff_out(text: str) -> List[Morsel]:
    """course 1: whitespace-tokenize `text` and propose a Morsel for
    every token that SMELLS like a digit was expected — currently, an
    hour prefix immediately followed by "am"/"pm" where exactly one
    prefix character is a confusable letter and substituting its digit
    counterpart produces a VALID hour (1-12). tokens needing more than
    one substitution to become valid are left alone — chew_on() can only
    verify one glyph at a time, so a multi-glyph guess is never proposed
    in the first place.

    this is a SMELL TEST, not a verdict: it never touches `text`. see the
    module docstring for exactly why context alone must never be enough
    to rewrite anything (the "Sam" -> "5am" failure mode).
    """
    morsels: List[Morsel] = []
    pos = 0
    for token in text.split(" "):
        if not token:
            pos += 1
            continue
        start = text.index(token, pos)
        pos = start + len(token)

        m = _TIME_TOKEN.match(token)
        if not m:
            continue
        prefix = m.group(1)

        new_prefix_chars = []
        changed_positions = []
        for i, ch in enumerate(prefix):
            if ch.isdigit():
                new_prefix_chars.append(ch)
                continue
            mapped = CONFUSABLE_GLYPHS.get(ch)
            if mapped and mapped.isdigit():
                new_prefix_chars.append(mapped)
                changed_positions.append(i)
            else:
                new_prefix_chars.append(ch)

        if len(changed_positions) != 1:
            continue  # no ambiguity, or more glyphs than one bite can verify

        new_prefix = "".join(new_prefix_chars)
        if not _VALID_HOUR.match(new_prefix):
            continue  # e.g. "S"->"5" is plausible, "B"->"8" giving "18" is not a real hour

        char_pos = changed_positions[0]
        morsels.append(Morsel(
            token_start=start,
            token_text=token,
            char_pos_in_token=char_pos,
            orig_char=prefix[char_pos],
            digit_char=new_prefix[char_pos],
            corrected_token=new_prefix + token[len(prefix):],
        ))
    return morsels


class DakutenMorsel:
    """one proposed dakuten/handakuten correction, waiting to be chewed
    on. unlike Morsel (always exactly one ambiguous glyph), a single
    dakuten misread can span more than one character position in the
    same word (measured live: "ハンタイ" vs the real "バンダイ" differs
    at TWO positions, ハ->バ and タ->ダ) -- so this carries every
    differing position the gazetteer match implies, and chew_dakuten()
    verifies each one independently."""
    __slots__ = ("text", "lang", "candidate_word", "positions", "similarity")

    def __init__(self, text: str, lang: Optional[str], candidate_word: str,
                 positions: List[tuple], similarity: float):
        self.text = text                      # the recognized text, e.g. "ハンタイ"
        self.lang = lang
        self.candidate_word = candidate_word  # the known gazetteer name it's close to, e.g. "バンダイ"
        self.positions = positions            # List[(index, orig_char, marked_char)]
        self.similarity = similarity          # fuzzy_similarity(text, candidate_word)


def _best_gazetteer_match(text: str, lang: Optional[str]):
    """best-matching known name for `text`, with NO confidence gate --
    unlike menu.consult_menu() (which only ever checks a read the
    recognizer was already unsure about), this course exists precisely
    to catch a read the recognizer was WRONGLY confident about, so it
    must be able to look regardless of confidence. returns
    (candidate_text, similarity) or None. reuses menu.py's own gazetteer
    rather than forking a second list of known names -- menu.py has no
    reverse import of savor.py, so this is safe."""
    from tofu.layers.menu import KNOWN_PLACES
    from tofu.utils.textmatch import fuzzy_similarity
    best = None
    for candidate, cand_lang in KNOWN_PLACES:
        if lang and cand_lang != lang:
            continue
        score = fuzzy_similarity(text, candidate)
        if best is None or score > best[1]:
            best = (candidate, score)
    return best


def sniff_dakuten(text: str, lang: Optional[str]) -> List[DakutenMorsel]:
    """course 4: does `text` become a KNOWN real name (menu.py's
    gazetteer) if some of its kana are toggled to their dakuten/
    handakuten form? proposes at most one DakutenMorsel per instance,
    covering every differing position at once.

    deliberately conservative, same spirit as sniff_out: only Japanese
    text is considered; the candidate must be the same length (no
    inserted/deleted characters, just marks); and EVERY differing
    character must be a known DAKUTEN_MAP relation -- a single
    non-dakuten diff aborts the whole proposal rather than guessing
    which differences to trust. there's no narrow context grammar for
    "a dakuten might be missing" the way sniff_out has "hour before am/
    pm", so the gazetteer match itself IS the trigger.
    """
    if not text or lang not in DAKUTEN_LANGS or " " in text:
        return []
    match = _best_gazetteer_match(text, lang)
    if match is None:
        return []
    candidate, score = match
    from tofu.layers.menu import SIMILARITY_FLOOR
    if score < SIMILARITY_FLOOR or len(candidate) != len(text) or candidate == text:
        return []
    positions = []
    for i, (a, b) in enumerate(zip(text, candidate)):
        if a == b:
            continue
        variants = DAKUTEN_MAP.get(a)
        if not variants or b not in variants:
            return []  # a non-dakuten diff exists -- out of scope, don't guess
        positions.append((i, a, b))
    if not positions:
        return []
    return [DakutenMorsel(text, lang, candidate, positions, score)]


def _plate_up(np, cv2, mask, vertical: bool = False):
    """isolate individual glyph blobs on the plate for inspection:
    connected-component analysis of a text_mask, returned in reading
    order as (x0, y0, x1, y1) in mask-local (crop) coordinates. classical
    CV (no new model), the same family of technique this codebase
    already uses on glyph masks elsewhere (typography, cleanse).

    reading order is left-to-right by default (the only case the digit/
    letter course ever needs -- Latin time-strings are never vertical).
    vertical=True sorts top-to-bottom instead, for stacked CJK columns
    (see _is_vertical_instance)."""
    m8 = mask.astype(np.uint8) * 255
    n, _labels, stats, _centroids = cv2.connectedComponentsWithStats(m8, connectivity=8)
    morsels_on_plate = []
    for i in range(1, n):  # label 0 is background, not a glyph
        area = stats[i, cv2.CC_STAT_AREA]
        if area < MIN_MORSEL_AREA:
            continue
        x = stats[i, cv2.CC_STAT_LEFT]
        y = stats[i, cv2.CC_STAT_TOP]
        w = stats[i, cv2.CC_STAT_WIDTH]
        h = stats[i, cv2.CC_STAT_HEIGHT]
        morsels_on_plate.append((x, y, x + w, y + h))
    morsels_on_plate.sort(key=(lambda b: b[1]) if vertical else (lambda b: b[0]))
    return morsels_on_plate


def _reference_bite(np, char: str, size_px: int,
                     font_registry: Optional[Any] = None, lang: Optional[str] = None):
    """rasterize a single reference character at ~size_px and return its
    tight-cropped binary mask — "what a genuine glyph of this character
    is supposed to taste like." returns None if nothing rendered (the
    fallback font chain has no glyph for it).

    the default (no font_registry/lang) resolves through _get_font(None,
    ...), i.e. the Latin-only FALLBACK_FONTS chain the digit/letter
    course has always used and still relies on unchanged. a katakana/
    hiragana glyph rendered through that chain silently comes back as a
    missing-glyph box (measured live: comparing ハ/バ/パ this way scored
    IoU 1.0 between all three, because it's the SAME placeholder box
    every time) -- when a font_registry and lang ARE given, this instead
    resolves through scribe.resolve_auto_font(), the same font-coverage
    logic render() itself uses, so callers with real script coverage
    needs (e.g. the dakuten course) get an actual glyph.
    """
    try:
        from PIL import Image, ImageDraw
    except ImportError:
        return None
    from tofu.layers.scribe import _get_font  # reuse: resolves TTC notation, fallback chain

    font_path = None
    if font_registry is not None and lang:
        from tofu.layers.scribe import resolve_auto_font
        font_path = resolve_auto_font(font_registry, lang, char)
    font = _get_font(font_path, max(8, int(size_px)))
    canvas = max(16, int(size_px * 1.8))
    img = Image.new("L", (canvas, canvas), 0)
    draw = ImageDraw.Draw(img)
    draw.text((canvas * 0.1, canvas * 0.1), char, font=font, fill=255)
    arr = np.array(img)
    ys, xs = np.nonzero(arr > 40)
    if len(xs) == 0:
        return None
    crop = arr[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
    return crop > 40


def _is_vertical_instance(bbox) -> bool:
    """is this instance's own bounding box tall enough to be stacked
    vertical text, not a horizontal line? reuses the same aspect-ratio
    convention scribe.py already applies for this exact question
    (VERTICAL_ASPECT_MIN), so "vertical" means the same thing everywhere
    in the codebase."""
    from tofu.layers.scribe import VERTICAL_ASPECT_MIN
    if bbox is None or bbox.width <= 0:
        return False
    return bbox.height >= bbox.width * VERTICAL_ASPECT_MIN


def _flavor_match(np, cv2, a, b) -> float:
    """how closely two glyph masks' SHAPES match, as IoU on a common
    canvas -- the actual "taste comparison" between a real bite and a
    reference bite."""
    def plate(mask):
        m = mask.astype(np.uint8) * 255
        return cv2.resize(m, (TASTING_CANVAS, TASTING_CANVAS), interpolation=cv2.INTER_AREA) > 127
    pa, pb = plate(a), plate(b)
    inter = int(np.logical_and(pa, pb).sum())
    union = int(np.logical_or(pa, pb).sum())
    return inter / union if union > 0 else 0.0


def chew_on(asset: ImageLike, inst: InstText, morsel: Morsel) -> Optional[bool]:
    """course 2: take a real bite. isolates the ambiguous glyph's actual
    pixels and compares their SHAPE against reference renders of both
    candidate characters.

    returns True ("swallow" — the digit is confirmed, apply the
    correction), False ("spit out" — the pixels support the original
    letter, leave the text alone), or None ("still chewing" — couldn't
    reach a confident verdict, e.g. the glyph count didn't line up 1:1
    with the recognized text — a connected/cursive script, most likely —
    or the two candidates' shape scores were too close to call). None
    must NEVER be treated as "swallow anyway."
    """
    try:
        import cv2
        import numpy as np
    except ImportError:
        return None

    img = load_rgb(asset)
    if img is None:
        return None
    mask = text_mask(img, inst.bounding_box, refine=True)
    if mask is None:
        return None

    plated = _plate_up(np, cv2, mask, vertical=_is_vertical_instance(inst.bounding_box))
    full_text = inst.text or ""
    no_space = full_text.replace(" ", "")
    if len(plated) != len(no_space):
        return None  # glyph count doesn't match the recognized text 1:1 -- don't guess

    prefix_no_space = full_text[:morsel.token_start + morsel.char_pos_in_token].replace(" ", "")
    abs_index = len(prefix_no_space)
    if abs_index >= len(plated):
        return None

    x0, y0, x1, y1 = plated[abs_index]
    bite = mask[y0:y1, x0:x1]
    if bite.size == 0 or not bite.any():
        return None

    size_px = y1 - y0
    letter_reference = _reference_bite(np, morsel.orig_char, size_px)
    digit_reference = _reference_bite(np, morsel.digit_char, size_px)
    if letter_reference is None or digit_reference is None:
        return None

    letter_score = _flavor_match(np, cv2, bite, letter_reference)
    digit_score = _flavor_match(np, cv2, bite, digit_reference)
    if digit_score >= letter_score + BITE_MARGIN:
        return True   # swallow: the digit reading tastes right
    if letter_score >= digit_score + BITE_MARGIN:
        return False  # spit out: the letter reading tastes right
    return None       # still chewing: too close to call


def _cluster_to_n_glyphs(plated, n: int, vertical: bool):
    """merge `_plate_up`'s components down to exactly n reading-order
    clusters by repeatedly fusing the closest adjacent pair, or None if
    there are already FEWER components than n (a real under-
    segmentation -- two characters fused into one blob -- that can't be
    safely split back apart, so it's left as a hard "don't guess").

    scoped to chew_dakuten only, not touching chew_on/_plate_up
    themselves: a dakuten/handakuten mark routinely renders as its own
    connected component a couple pixels from its base glyph's body
    (measured live, both in a clean synthetic render AND in the real
    japan-street crop: 4 real katakana characters -- one of which is ン,
    which is canonically two disconnected strokes even in print --
    segmented into 6 components), which breaks chew_on's original
    strict 1-component-per-character assumption far more often for CJK
    than it ever does for the digit course's Latin text. merging the
    tightest gaps first is exactly the right greedy rule here: pieces of
    the SAME character (a stroke, a mark) sit closer to each other than
    to a neighboring DIFFERENT character does.
    """
    if len(plated) < n:
        return None
    if len(plated) == n:
        return list(plated)
    axis = 1 if vertical else 0  # y0 for a vertical column, x0 for a horizontal line
    clusters = [list(b) for b in sorted(plated, key=lambda b: b[axis])]
    while len(clusters) > n:
        best_gap, best_i = None, None
        for i in range(len(clusters) - 1):
            a, b = clusters[i], clusters[i + 1]
            gap = b[axis] - a[axis + 2]
            if best_gap is None or gap < best_gap:
                best_gap, best_i = gap, i
        a, b = clusters[best_i], clusters.pop(best_i + 1)
        clusters[best_i] = [min(a[0], b[0]), min(a[1], b[1]), max(a[2], b[2]), max(a[3], b[3])]
    return [tuple(c) for c in clusters]


def chew_swaps(asset: ImageLike, inst: InstText, positions: List[tuple],
                lang: Optional[str],
                font_registry: Optional[Any] = None) -> "dict[int, Optional[bool]]":
    """generic per-position glyph-swap verification: does the pixel
    evidence at each position support the PROPOSED character over the
    RECOGNIZED one? same isolate-render-compare method as chew_on(),
    generalized to arbitrary (position, recognized_char, proposed_char)
    swaps so any narrowly-triggered course (dakuten, menu's substring
    gazetteer) can pixel-verify its proposal without duplicating the
    machinery.

    `positions` is [(index, recognized_char, proposed_char)] where index
    counts characters of inst.text with spaces removed (matching the
    glyph clusters _cluster_to_n_glyphs produces). returns a verdict per
    position: True (pixels favor the proposal), False (pixels favor the
    recognized char), None (inconclusive or unverifiable). None must
    NEVER be treated as True.
    """
    verdicts: "dict[int, Optional[bool]]" = {}
    try:
        import cv2
        import numpy as np
    except ImportError:
        return verdicts

    img = load_rgb(asset)
    if img is None:
        return verdicts

    bbox = inst.bounding_box
    vertical = _is_vertical_instance(bbox)
    no_space = (inst.text or "").replace(" ", "")

    # small-glyph rescue: at street-photo scale a stacked sign's glyphs
    # can be ~13px each, where Otsu can't separate adjacent characters
    # into distinct components (measured live: japan-subs' 24x106px
    # 8-glyph post plated as only 6 components, so clustering refused
    # and every verdict came back unverifiable). upscaling the crop
    # BEFORE masking recovers the separation -- same coarse-to-fine
    # philosophy as second_look/zoom_detect. all downstream coordinates
    # (plate clusters, bites, reference sizes) live consistently in the
    # upscaled space, so nothing needs scaling back.
    per_glyph = (bbox.height if vertical else bbox.width) / max(1, len(no_space))
    if per_glyph < GLYPH_UPSCALE_MIN_PX:
        x0c, y0c = max(0, bbox.x), max(0, bbox.y)
        x1c = min(img.shape[1], bbox.x + bbox.width)
        y1c = min(img.shape[0], bbox.y + bbox.height)
        if x1c - x0c < 3 or y1c - y0c < 3:
            return verdicts
        crop = img[y0c:y1c, x0c:x1c]
        crop = cv2.resize(
            crop,
            (crop.shape[1] * GLYPH_UPSCALE_FACTOR, crop.shape[0] * GLYPH_UPSCALE_FACTOR),
            interpolation=cv2.INTER_CUBIC,
        )
        mask = text_mask(crop, BBox(x=0, y=0, width=crop.shape[1], height=crop.shape[0]), refine=True)
    else:
        mask = text_mask(img, bbox, refine=True)
    if mask is None:
        return verdicts

    plated = _plate_up(np, cv2, mask, vertical=vertical)
    plated = _cluster_to_n_glyphs(plated, len(no_space), vertical)
    if plated is None:
        return verdicts  # fewer components than characters -- can't safely split, don't guess

    for pos, orig_char, proposed_char in positions:
        if pos >= len(plated):
            continue
        x0, y0, x1, y1 = plated[pos]
        bite = mask[y0:y1, x0:x1]
        if bite.size == 0 or not bite.any():
            verdicts[pos] = None
            continue

        size_px = y1 - y0
        orig_reference = _reference_bite(np, orig_char, size_px, font_registry, lang)
        proposed_reference = _reference_bite(np, proposed_char, size_px, font_registry, lang)
        if orig_reference is None or proposed_reference is None:
            verdicts[pos] = None
            continue

        orig_score = _flavor_match(np, cv2, bite, orig_reference)
        proposed_score = _flavor_match(np, cv2, bite, proposed_reference)
        if proposed_score >= orig_score + BITE_MARGIN:
            verdicts[pos] = True   # swallow: the proposed reading tastes right
        elif orig_score >= proposed_score + BITE_MARGIN:
            verdicts[pos] = False  # spit out: the recognized reading tastes right
        else:
            verdicts[pos] = None   # still chewing: too close to call
    return verdicts


def chew_dakuten(asset: ImageLike, inst: InstText, morsel: DakutenMorsel,
                  font_registry: Optional[Any] = None) -> "dict[int, Optional[bool]]":
    """course 4's bite: verifies EACH position in `morsel.positions`
    independently against its own pixels -- a thin wrapper over
    chew_swaps() (a real case can need more than one glyph confirmed,
    see DakutenMorsel).

    only answers "does this glyph's pixels support having a mark, vs.
    the plain form" -- NOT "which mark (dakuten vs handakuten)
    specifically." whole-glyph IoU can't reliably tell dakuten and
    handakuten apart (measured live: バ vs パ reference renders score
    ~0.8 IoU against each other, far above BITE_MARGIN's separation
    requirement), so which variant is correct is left entirely to the
    gazetteer match that proposed this morsel; pixel evidence here only
    confirms whether to trust that suggestion at all.
    """
    return chew_swaps(asset, inst, morsel.positions, morsel.lang, font_registry)


def taste(asset: ImageLike, instances: List[InstText], font_registry: Optional[Any] = None,
          engine: Optional[Any] = None) -> int:
    """the full tasting menu, all courses, run across every instance.

    this is Savor's one public entry point (called once by
    cicerone.detect(), on the FINAL manifest — see module docstring).

    font_registry is optional and only used by course 4 (dakuten) --
    every existing caller that omits it keeps behaving exactly as
    before (course 1-3, the digit/letter course, never needed it).

    engine is optional and only used by course 0 (clumps), which needs a
    genuinely independent look at a disputed span before it will change the
    character COUNT of a read. without one, course 0 can still detect and
    report a clump but can never apply a correction — the same fail-closed
    contract every other course follows.

    returns the number of instances actually corrected (mirrors
    second_look()'s "count of regions improved" return contract, for
    logging/measurement). a Morsel that's still chewing (course 2
    returned None) is never applied — it's recorded on
    `inst.ocr_correction` with applied=False so it's visible for review
    instead of silently vanishing.
    """
    swallowed = 0
    for inst in instances:
        text = inst.text or ""
        if not text:
            inst.ocr_quality = {
                "state": OCR_QUALITY_UNRESOLVABLE,
                "reasons": ["no_recognized_glyphs"],
                "recognized_glyph_count": 0,
                "engine": getattr(engine, "name", "unavailable") if engine is not None else "unavailable",
                "reread_available": engine is not None,
            }
            continue
        plate = _plate_clusters(asset, inst)
        lang = inst.detected_language or inst.language
        # Course 0 (amuse-bouche): segmentation before anything glyph-indexed.
        # While the plate carries more glyphs than the recognizer claimed
        # characters, every later course is reading the wrong positions --
        # and the mark courses know it, which is why they return None here.
        vertical = _is_vertical_instance(inst.bounding_box)
        clumps, clump_evidence = sniff_clumps(inst.text or "", plate, lang, vertical=vertical)
        clump_resolved = not clump_evidence.get("surplus")
        for morsel in clumps:
            verdict, clump_evidence = chew_clump(
                asset, inst, morsel, plate, engine=engine, font_registry=font_registry
            )
            # A False verdict resolves the surplus too: the emitted character
            # held up on its own pixels, so this token is not the clump.
            clump_resolved = verdict is not None
            if verdict is True:
                original = inst.text or ""
                corrected = (original[:morsel.token_start] + morsel.recovered
                             + original[morsel.token_start + len(morsel.token_text):])
                inst.text = corrected
                _record_correction(
                    inst, applied=True, original_text=original, corrected_text=corrected,
                    reason="one emitted character stood on several glyphs; the stored phrase, "
                           "glyph shapes and an isolated re-read all support the split",
                    course="phrase_segmentation",
                    correction_resource=LATIN_PHRASE_RESOURCE.audit_identity(),
                    clump_evidence=clump_evidence,
                )
                swallowed += 1
                text = corrected
                # The claimed glyph count changed, so the plate's clusters
                # no longer describe this text.  Re-plate before the courses
                # that index by glyph position run on it.
                plate = _plate_clusters(asset, inst)
        # The cardinality course is itself a verified recovery.  Only after it
        # has either repaired or failed to repair the plate can the ordinary
        # glyph-indexed courses decide whether their automatic result is safe.
        quality_before_courses = assess_ocr_quality(asset, inst, plate, engine)
        can_auto_apply = quality_before_courses["state"] == OCR_QUALITY_RELIABLE
        if not clump_resolved:
            # A component-count surplus is never allowed to vanish.  Whether
            # it went unlocalized, unnamed by any stored phrase, or simply
            # unverified, the region is surfaced for a structured re-read
            # rather than left looking like a clean read.
            _record_correction(
                inst, applied=False, original_text=inst.text or "",
                reason="the plate carries more glyphs than the recognized text claims; "
                       "region needs a structured re-read",
                course="phrase_segmentation", clump_evidence=clump_evidence,
            )
        # Course 5a: physical case evidence.  Run before lexical/diacritic
        # work so a confirmed all-cap plate supplies the correct base letters
        # (``Republique`` -> ``REPUBLIQUE``) for any later mark repair.
        # An already all-cap read has no case repair to offer.  Fragmented
        # serif/engraved components can make its height signature look odd,
        # but a review record with an identical candidate is pure noise.
        if (inst.text or "").upper() == (inst.text or ""):
            case_candidate, case_evidence = None, {"state": "not_applicable", "reason": "already_all_caps"}
        else:
            case_candidate, case_evidence = chew_case(asset, inst, plate=plate)
        if case_candidate and can_auto_apply:
            original = inst.text or ""
            inst.text = case_candidate
            _record_correction(
                inst, applied=True, original_text=original,
                corrected_text=case_candidate,
                reason="case signature across the token confirms all-cap lettering",
                course="case_signature", case_evidence=case_evidence,
            )
            swallowed += 1
            text = case_candidate
        elif case_candidate:
            _record_correction(
                inst, applied=False, original_text=inst.text or "",
                candidate_text=case_candidate,
                reason="case evidence is present but the OCR quality gate requires review",
                course="case_signature", case_evidence=case_evidence,
                ocr_quality=quality_before_courses,
            )
        elif case_evidence.get("state") == "review":
            # Preserve evidence for mixed/lowercase contradictions without
            # guessing a title-case or arbitrary per-character correction.
            _record_correction(
                inst, applied=False, original_text=inst.text or "",
                candidate_text=(inst.text or "").upper(),
                reason="observed case signature conflicts with OCR casing; manual review required",
                course="case_signature", case_evidence=case_evidence,
            )
        # Course 5b: a stored language-scoped canonical form proposes the
        # mark, while detached-mark geometry independently decides whether it
        # exists.  No generic accent stripping or whole-glyph IoU participates.
        for proposal in _latin_diacritic_proposals(inst.text or "", lang):
            verdicts = chew_accents(inst, plate, proposal["positions"])
            if all(verdicts.get(position) is True for position, _marks in proposal["positions"]) and can_auto_apply:
                original = inst.text or ""
                corrected = _compose_marks(original, proposal["positions"])
                inst.text = corrected
                _record_correction(
                    inst, applied=True, original_text=original, corrected_text=corrected,
                    reason="stored Latin diacritic form confirmed by detached-mark geometry",
                    course="latin_diacritic",
                    correction_resource=LATIN_DIACRITIC_RESOURCE.audit_identity(),
                    mark_verdicts={str(position): verdicts[position] for position, _marks in proposal["positions"]},
                )
                swallowed += 1
                text = inst.text or text
            elif all(verdicts.get(position) is True for position, _marks in proposal["positions"]):
                _record_correction(
                    inst, applied=False, original_text=inst.text or "",
                    candidate_text=proposal["canonical"],
                    reason="diacritic evidence is present but the OCR quality gate requires review",
                    course="latin_diacritic",
                    correction_resource=LATIN_DIACRITIC_RESOURCE.audit_identity(),
                    mark_verdicts={str(position): verdicts[position] for position, _marks in proposal["positions"]},
                    ocr_quality=quality_before_courses,
                )
            elif any(verdicts.get(position) is None for position, _marks in proposal["positions"]):
                _record_correction(
                    inst, applied=False, original_text=inst.text or "",
                    candidate_text=proposal["canonical"],
                    reason="stored Latin diacritic candidate requires review; mark geometry is unresolved",
                    course="latin_diacritic",
                    correction_resource=LATIN_DIACRITIC_RESOURCE.audit_identity(),
                    mark_verdicts={str(position): verdicts[position] for position, _marks in proposal["positions"]},
                )
        for morsel in sniff_out(text):
            verdict = chew_on(asset, inst, morsel)
            if verdict is True and can_auto_apply:
                original = inst.text or text
                corrected_text = (
                    text[:morsel.token_start] + morsel.corrected_token
                    + text[morsel.token_start + len(morsel.token_text):]
                )
                inst.text = corrected_text
                _record_correction(
                    inst, applied=True, original_text=original, corrected_text=corrected_text,
                    reason=f"'{morsel.orig_char}'->'{morsel.digit_char}' in a digit-expected "
                           f"context, confirmed by glyph shape",
                    course="swap",
                )
                swallowed += 1
            elif verdict is True:
                _record_correction(
                    inst, applied=False, original_text=inst.text or text,
                    candidate_text=(text[:morsel.token_start] + morsel.corrected_token
                                    + text[morsel.token_start + len(morsel.token_text):]),
                    reason="glyph-shape evidence is present but the OCR quality gate requires review",
                    course="swap", ocr_quality=quality_before_courses,
                )
            elif verdict is None:
                _record_correction(
                    inst, applied=False, original_text=inst.text or text,
                    candidate_text=(text[:morsel.token_start] + morsel.corrected_token
                                    + text[morsel.token_start + len(morsel.token_text):]),
                    reason=f"possible '{morsel.orig_char}'->'{morsel.digit_char}' misread in a "
                           f"digit-expected context, but the pixel evidence was inconclusive",
                    course="swap",
                )
            # verdict is False: spat out, original text stands, nothing recorded

        for dmorsel in sniff_dakuten(inst.text or "", lang):
            verdicts = chew_dakuten(asset, inst, dmorsel, font_registry)
            confirmed = {p: b for p, a, b in dmorsel.positions if verdicts.get(p) is True}
            if not confirmed:
                if any(verdicts.get(p) is None for p, a, b in dmorsel.positions):
                    _record_correction(
                        inst, applied=False, original_text=inst.text or "",
                        candidate_text=dmorsel.candidate_word,
                        reason=f"possible missing dakuten/handakuten vs known name "
                               f"'{dmorsel.candidate_word}' ({dmorsel.similarity:.2f} similarity), "
                               f"pixel evidence inconclusive",
                        course="dakuten",
                    )
                continue  # verdict False on every position: spat out, nothing recorded
            if not can_auto_apply:
                _record_correction(
                    inst, applied=False, original_text=inst.text or "",
                    candidate_text=dmorsel.candidate_word,
                    reason="dakuten evidence is present but the OCR quality gate requires review",
                    course="dakuten", ocr_quality=quality_before_courses,
                )
                continue
            original = inst.text or ""
            chars = list(original)
            for p, marked_char in confirmed.items():
                chars[p] = marked_char
            corrected_text = "".join(chars)
            inst.text = corrected_text
            unresolved = [p for p, a, b in dmorsel.positions if p not in confirmed]
            reason = (
                f"missing dakuten/handakuten at position(s) {sorted(confirmed)}, confirmed by "
                f"glyph shape against known name '{dmorsel.candidate_word}'"
            )
            if unresolved:
                reason += f"; position(s) {unresolved} left unchanged, pixel evidence didn't support them"
            _record_correction(
                inst, applied=True, original_text=original, corrected_text=corrected_text,
                reason=reason, course="dakuten",
            )
            swallowed += 1
        # The final record reflects the text and plate the user will actually
        # see.  Keep the pre-course result as provenance when a verified
        # segmentation repair made an initially risky region observable.
        final_plate = _plate_clusters(asset, inst)
        final_quality = assess_ocr_quality(asset, inst, final_plate, engine)
        if final_quality != quality_before_courses:
            final_quality["pre_course"] = quality_before_courses
        inst.ocr_quality = final_quality
    return swallowed
