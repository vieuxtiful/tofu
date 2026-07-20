## 🍢 recognition_correct — Cicerone post-recognition digit/letter disambiguation
## vieuxtiful
"""
CRNN+CTC recognizers (EasyOCR's default backend) decode character-by-
character with no language model and no notion of "does this reading make
sense" — a glyph that's locally ambiguous (5/S, 0/O, 1/I, 8/B, 2/Z, 6/G)
can be read wrong with HIGH confidence, because the model is confident
about the SHAPE it saw, not about whether the result is semantically
plausible. `second_look()`'s re-read only triggers below a confidence
floor (0.55) that a case like this never crosses, and re-reading with the
same model on the same pixels reliably reproduces the same mistake — a
different SIGNAL is needed, not another pass of the same recognizer.

three stages, deliberately separated so nothing gets rewritten on a guess:

  Stage A (find_candidates): grammar-anchored — is there a WHOLE token
    matching a digit-expected shape (currently: an am/pm time suffix)
    where swapping a confusable letter for its digit counterpart yields
    a VALID result? this only proposes a candidate; it decides nothing
    about the actual pixels. (a plain regex/dict substitution without
    this next stage was tried and rejected — see the plan doc: it
    silently corrupts real text like "Sam" into "5am", because context
    alone can't distinguish "OCR was wrong" from "OCR was right and the
    context pattern is a coincidence.")

  Stage B (verify_candidate): pixel evidence — segments the ambiguous
    glyph out of the region via connected-component analysis of the
    shared `imaging.text_mask()`, renders BOTH candidate characters as
    reference glyphs at a matching size (`scribe._get_font`), and
    compares shapes via IoU. only a clear margin one way or the other
    resolves the candidate; ties return None (inconclusive).

  Stage C (orchestration in correct_instances): a candidate is only ever
    APPLIED when Stage B resolves True. an inconclusive candidate is
    never guessed into the text — it's recorded on
    `InstText.ocr_correction` (applied=False) so it surfaces as a
    reviewable signal instead of silently doing nothing OR silently
    rewriting something wrong.
"""

import re
from typing import Any, List, Optional

from tofu.core.types import InstText
from tofu.utils.imaging import load_rgb, text_mask

# letter <-> digit confusion pairs a CRNN commonly mixes up on similar
# glyph shapes. bidirectional so an already-correct digit is never
# treated as a "letter that could be a digit" (see find_candidates).
CONFUSION_PAIRS = {
    "5": "S", "S": "5",
    "0": "O", "O": "0",
    "1": "I", "I": "1",
    "8": "B", "B": "8",
    "2": "Z", "Z": "2",
    "6": "G", "G": "6",
}

_TIME_TOKEN = re.compile(r"^([A-Za-z0-9]{1,2})(am|pm)$", re.IGNORECASE)
_VALID_HOUR = re.compile(r"^(?:[1-9]|1[0-2])$")

GLYPH_MATCH_MARGIN = 0.12   # min IoU advantage the winning candidate needs
MIN_COMPONENT_AREA = 6      # px^2 floor for a connected component to count as a glyph
GLYPH_CANVAS = 24           # common canvas size glyph masks are resized to for IoU


class Candidate:
    """a proposed digit/letter correction within one InstText's text."""
    __slots__ = ("token_start", "token_text", "char_pos_in_token", "orig_char", "digit_char", "corrected_token")

    def __init__(self, token_start: int, token_text: str, char_pos_in_token: int,
                 orig_char: str, digit_char: str, corrected_token: str):
        self.token_start = token_start
        self.token_text = token_text
        self.char_pos_in_token = char_pos_in_token
        self.orig_char = orig_char
        self.digit_char = digit_char
        self.corrected_token = corrected_token


def find_candidates(text: str) -> List[Candidate]:
    """Stage A: whitespace-tokenize `text` and propose corrections for
    tokens matching a digit-expected grammar. currently: a 1-2 character
    hour prefix immediately followed by "am"/"pm", where exactly one
    prefix character is a confusable letter and substituting its digit
    counterpart produces a valid hour (1-12). multi-character ambiguity
    (more than one letter needing substitution) is skipped, not guessed —
    Stage B can only verify one glyph position at a time.

    proposals only — nothing here touches `inst.text`. see module
    docstring for why context alone must never be sufficient to rewrite.
    """
    candidates: List[Candidate] = []
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
            mapped = CONFUSION_PAIRS.get(ch)
            if mapped and mapped.isdigit():
                new_prefix_chars.append(mapped)
                changed_positions.append(i)
            else:
                new_prefix_chars.append(ch)

        if len(changed_positions) != 1:
            continue  # no ambiguity, or more than Stage B can verify at once

        new_prefix = "".join(new_prefix_chars)
        if not _VALID_HOUR.match(new_prefix):
            continue  # e.g. "S" -> "5" is fine, "B" -> "8" giving "18" is not a valid hour

        char_pos = changed_positions[0]
        candidates.append(Candidate(
            token_start=start,
            token_text=token,
            char_pos_in_token=char_pos,
            orig_char=prefix[char_pos],
            digit_char=new_prefix[char_pos],
            corrected_token=new_prefix + token[len(prefix):],
        ))
    return candidates


def _segment_glyphs(np, cv2, mask):
    """connected-component glyph blobs in a text_mask, left-to-right, as
    (x0, y0, x1, y1) in mask-local (crop) coordinates. classical
    approach (no new model) — the same family of technique as this
    codebase's other glyph-mask consumers (typography, cleanse)."""
    m8 = mask.astype(np.uint8) * 255
    n, _labels, stats, _centroids = cv2.connectedComponentsWithStats(m8, connectivity=8)
    boxes = []
    for i in range(1, n):  # label 0 is background
        area = stats[i, cv2.CC_STAT_AREA]
        if area < MIN_COMPONENT_AREA:
            continue
        x = stats[i, cv2.CC_STAT_LEFT]
        y = stats[i, cv2.CC_STAT_TOP]
        w = stats[i, cv2.CC_STAT_WIDTH]
        h = stats[i, cv2.CC_STAT_HEIGHT]
        boxes.append((x, y, x + w, y + h))
    boxes.sort(key=lambda b: b[0])
    return boxes


def _render_glyph_mask(np, char: str, size_px: int):
    """rasterize a single reference character at ~size_px and return its
    tight-cropped binary mask, or None if nothing rendered (missing
    glyph in the fallback font chain)."""
    try:
        from PIL import Image, ImageDraw
    except ImportError:
        return None
    from tofu.layers.scribe import _get_font

    font = _get_font(None, max(8, int(size_px)))
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


def _glyph_iou(np, cv2, a, b) -> float:
    def norm(mask):
        m = mask.astype(np.uint8) * 255
        return cv2.resize(m, (GLYPH_CANVAS, GLYPH_CANVAS), interpolation=cv2.INTER_AREA) > 127
    na, nb = norm(a), norm(b)
    inter = int(np.logical_and(na, nb).sum())
    union = int(np.logical_or(na, nb).sum())
    return inter / union if union > 0 else 0.0


def verify_candidate(asset: Any, inst: InstText, candidate: Candidate) -> Optional[bool]:
    """Stage B: isolate the ambiguous glyph's pixels and compare against
    rendered references of both candidate characters.

    returns True (apply the digit correction), False (evidence favors
    the original letter), or None (couldn't verify — glyph segmentation
    didn't line up 1:1 with the recognized text, e.g. a connected
    script, or the IoU scores were too close to call). None must never
    be treated as "apply anyway."
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

    glyphs = _segment_glyphs(np, cv2, mask)
    full_text = inst.text or ""
    no_space = full_text.replace(" ", "")
    if len(glyphs) != len(no_space):
        return None  # component count doesn't match 1:1 -- don't guess

    prefix_no_space = full_text[:candidate.token_start + candidate.char_pos_in_token].replace(" ", "")
    abs_index = len(prefix_no_space)
    if abs_index >= len(glyphs):
        return None

    x0, y0, x1, y1 = glyphs[abs_index]
    crop_mask = mask[y0:y1, x0:x1]
    if crop_mask.size == 0 or not crop_mask.any():
        return None

    size_px = y1 - y0
    orig_glyph = _render_glyph_mask(np, candidate.orig_char, size_px)
    digit_glyph = _render_glyph_mask(np, candidate.digit_char, size_px)
    if orig_glyph is None or digit_glyph is None:
        return None

    orig_score = _glyph_iou(np, cv2, crop_mask, orig_glyph)
    digit_score = _glyph_iou(np, cv2, crop_mask, digit_glyph)
    if digit_score >= orig_score + GLYPH_MATCH_MARGIN:
        return True
    if orig_score >= digit_score + GLYPH_MATCH_MARGIN:
        return False
    return None  # too close to call


def correct_instances(asset: Any, instances: List[InstText]) -> int:
    """run Stage A -> Stage B -> Stage C across all instances in place.

    returns the number of instances whose text was actually corrected
    (for logging/measurement — mirrors second_look()'s return contract).
    unresolved candidates are recorded on `inst.ocr_correction` with
    applied=False rather than silently dropped, so they're visible for
    review even though nothing was rewritten.
    """
    corrected = 0
    for inst in instances:
        text = inst.text or ""
        if not text:
            continue
        candidates = find_candidates(text)
        for cand in candidates:
            verdict = verify_candidate(asset, inst, cand)
            if verdict is True:
                original = inst.text or text
                corrected_text = (
                    text[:cand.token_start] + cand.corrected_token
                    + text[cand.token_start + len(cand.token_text):]
                )
                inst.text = corrected_text
                inst.ocr_correction = {
                    "applied": True,
                    "original_text": original,
                    "corrected_text": corrected_text,
                    "reason": f"'{cand.orig_char}'->'{cand.digit_char}' in digit-expected "
                              f"context, confirmed by glyph shape",
                }
                corrected += 1
            elif verdict is None:
                inst.ocr_correction = {
                    "applied": False,
                    "candidate_text": (
                        text[:cand.token_start] + cand.corrected_token
                        + text[cand.token_start + len(cand.token_text):]
                    ),
                    "reason": f"possible '{cand.orig_char}'->'{cand.digit_char}' misread in "
                              f"digit-expected context, but pixel evidence was inconclusive",
                }
            # verdict is False: original text stands, nothing recorded
    return corrected
