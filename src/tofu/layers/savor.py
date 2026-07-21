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

THE MENU — three courses, deliberately kept separate so nothing gets
swallowed on a guess:

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
from typing import Any, List, Optional

from tofu.core.types import InstText
from tofu.utils.imaging import load_rgb, text_mask

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


def chew_on(asset: Any, inst: InstText, morsel: Morsel) -> Optional[bool]:
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


def chew_dakuten(asset: Any, inst: InstText, morsel: DakutenMorsel,
                  font_registry: Optional[Any] = None) -> "dict[int, Optional[bool]]":
    """course 4's bite: verifies EACH position in `morsel.positions`
    independently against its own pixels, same isolate-render-compare
    method as chew_on(), but returns a verdict per position instead of
    a single verdict -- a real case can need more than one glyph
    confirmed (see DakutenMorsel).

    only answers "does this glyph's pixels support having a mark, vs.
    the plain form" -- NOT "which mark (dakuten vs handakuten)
    specifically." whole-glyph IoU can't reliably tell dakuten and
    handakuten apart (measured live: バ vs パ reference renders score
    ~0.8 IoU against each other, far above BITE_MARGIN's separation
    requirement), so which variant is correct is left entirely to the
    gazetteer match that proposed this morsel; pixel evidence here only
    confirms whether to trust that suggestion at all.
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
    mask = text_mask(img, inst.bounding_box, refine=True)
    if mask is None:
        return verdicts

    vertical = _is_vertical_instance(inst.bounding_box)
    plated = _plate_up(np, cv2, mask, vertical=vertical)
    no_space = (inst.text or "").replace(" ", "")
    plated = _cluster_to_n_glyphs(plated, len(no_space), vertical)
    if plated is None:
        return verdicts  # fewer components than characters -- can't safely split, don't guess

    for pos, orig_char, marked_char in morsel.positions:
        if pos >= len(plated):
            continue
        x0, y0, x1, y1 = plated[pos]
        bite = mask[y0:y1, x0:x1]
        if bite.size == 0 or not bite.any():
            verdicts[pos] = None
            continue

        size_px = y1 - y0
        plain_reference = _reference_bite(np, orig_char, size_px, font_registry, morsel.lang)
        marked_reference = _reference_bite(np, marked_char, size_px, font_registry, morsel.lang)
        if plain_reference is None or marked_reference is None:
            verdicts[pos] = None
            continue

        plain_score = _flavor_match(np, cv2, bite, plain_reference)
        marked_score = _flavor_match(np, cv2, bite, marked_reference)
        if marked_score >= plain_score + BITE_MARGIN:
            verdicts[pos] = True   # swallow: the marked reading tastes right
        elif plain_score >= marked_score + BITE_MARGIN:
            verdicts[pos] = False  # spit out: the plain reading tastes right
        else:
            verdicts[pos] = None   # still chewing: too close to call
    return verdicts


def taste(asset: Any, instances: List[InstText], font_registry: Optional[Any] = None) -> int:
    """the full tasting menu, all courses, run across every instance.

    this is Savor's one public entry point (called once by
    cicerone.detect(), on the FINAL manifest — see module docstring).

    font_registry is optional and only used by course 4 (dakuten) --
    every existing caller that omits it keeps behaving exactly as
    before (course 1-3, the digit/letter course, never needed it).

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
            continue
        for morsel in sniff_out(text):
            verdict = chew_on(asset, inst, morsel)
            if verdict is True:
                original = inst.text or text
                corrected_text = (
                    text[:morsel.token_start] + morsel.corrected_token
                    + text[morsel.token_start + len(morsel.token_text):]
                )
                inst.text = corrected_text
                inst.ocr_correction = {
                    "applied": True,
                    "original_text": original,
                    "corrected_text": corrected_text,
                    "reason": f"'{morsel.orig_char}'->'{morsel.digit_char}' in a digit-expected "
                              f"context, confirmed by glyph shape",
                }
                swallowed += 1
            elif verdict is None:
                inst.ocr_correction = {
                    "applied": False,
                    "candidate_text": (
                        text[:morsel.token_start] + morsel.corrected_token
                        + text[morsel.token_start + len(morsel.token_text):]
                    ),
                    "reason": f"possible '{morsel.orig_char}'->'{morsel.digit_char}' misread in a "
                              f"digit-expected context, but the pixel evidence was inconclusive",
                }
            # verdict is False: spat out, original text stands, nothing recorded

        lang = inst.detected_language or inst.language
        for dmorsel in sniff_dakuten(inst.text or "", lang):
            verdicts = chew_dakuten(asset, inst, dmorsel, font_registry)
            confirmed = {p: b for p, a, b in dmorsel.positions if verdicts.get(p) is True}
            if not confirmed:
                if any(verdicts.get(p) is None for p, a, b in dmorsel.positions):
                    inst.ocr_correction = {
                        "applied": False,
                        "candidate_text": dmorsel.candidate_word,
                        "reason": f"possible missing dakuten/handakuten vs known name "
                                  f"'{dmorsel.candidate_word}' ({dmorsel.similarity:.2f} similarity), "
                                  f"pixel evidence inconclusive",
                    }
                continue  # verdict False on every position: spat out, nothing recorded
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
            inst.ocr_correction = {
                "applied": True,
                "original_text": original,
                "corrected_text": corrected_text,
                "reason": reason,
            }
            swallowed += 1
    return swallowed
