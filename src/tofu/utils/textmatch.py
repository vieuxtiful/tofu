## 🍢 textmatch — normalization + fuzzy similarity for TM text matching
## vieuxtiful
"""
shared with verify.py's OCR round-trip scorer in spirit (casefold +
collapse non-word chars, difflib ratio) but kept as its own copy here:
translation-memory matching is a distinct concern (exact/fuzzy KEY
lookup across a whole store) from verify's single-string legibility
check, and the two are free to diverge without cross-coupling a QA
scorer to the memory layer's matching thresholds.
"""

import re
from difflib import SequenceMatcher
from typing import Optional

FUZZY_MATCH_THRESHOLD = 0.85  # plan-specified minimum edit-distance ratio


def normalize_text(text: Optional[str]) -> str:
    """casefold + collapse everything non-word, so exact-match lookups
    aren't defeated by whitespace/punctuation OCR noise."""
    if not text:
        return ""
    return re.sub(r"[\W_]+", "", text, flags=re.UNICODE).casefold()


def fuzzy_similarity(a: Optional[str], b: Optional[str]) -> float:
    """SequenceMatcher ratio over normalized text; 0.0 if either is empty."""
    na, nb = normalize_text(a), normalize_text(b)
    if not na or not nb:
        return 0.0
    return SequenceMatcher(None, na, nb).ratio()


# Cyrillic letters an OCR recognizer running a joint (ru, en) charset can
# emit as their Latin lookalikes, and vice versa.  This is the confusable
# set of Unicode TR39 restricted to the Cyrillic/Latin pair -- deliberately
# NOT the full confusables table: only letters whose glyphs are identical in
# ordinary typefaces are listed, so folding is lossless as a COMPARISON
# device even though it is lossy as text.
_CYRILLIC_TO_LATIN_SKELETON = {
    "А": "A", "В": "B", "Е": "E", "К": "K", "М": "M", "Н": "H", "О": "O",
    "Р": "P", "С": "C", "Т": "T", "У": "Y", "Х": "X",
    "Ѕ": "S", "І": "I", "Ј": "J", "Ү": "Y",
    "а": "a", "е": "e", "о": "o", "р": "p", "с": "c", "у": "y", "х": "x",
    "ѕ": "s", "і": "i", "ј": "j", "ԛ": "q", "ԝ": "w",
}


def pare(text: Optional[str]) -> str:
    """Pare a string down to its script-neutral skeleton.

    Cyrillic and Latin share a dozen letterforms outright (В/B, С/C, Т/T,
    а/a ...), and a recognizer given a joint ``(ru, en)`` charset picks
    between them per character with nothing to constrain it -- ``ВМЕСТЕ``
    comes back as ``BMЕСTЕ``, three characters of which are the wrong
    script and none of which look wrong.  Comparing two such strings
    literally reports them as barely related.

    Paring maps every confusable Cyrillic character onto its Latin twin so
    the comparison sees the letterforms rather than the codepoints.  Case
    is preserved: the map is case-sensitive (Cyrillic ``В`` is Latin ``B``,
    but Cyrillic ``в`` is not Latin ``b``), so casefolding first would
    destroy exactly the information the fold depends on.  Callers that want
    case-insensitivity should let :func:`normalize_text` run afterwards.
    """
    if not text:
        return ""
    return "".join(_CYRILLIC_TO_LATIN_SKELETON.get(ch, ch) for ch in text)


def pared_similarity(a: Optional[str], b: Optional[str]) -> float:
    """:func:`fuzzy_similarity` over the pared skeletons of both sides."""
    return fuzzy_similarity(pare(a), pare(b))


def best_span_similarity(needle: Optional[str], haystack: Optional[str]) -> float:
    """How well ``needle`` matches the best same-length window of ``haystack``.

    Answers "is this short read a PIECE of that longer one?", which plain
    :func:`fuzzy_similarity` cannot: a four-character fragment of a
    thirty-character line scores near zero against the whole line no matter
    how exactly it matches the part it came from.

    Both sides are pared and normalized first, so a Cyrillic fragment and a
    half-Latin line read of the same pixels still compare as the same text.
    Returns 0.0 when either side is empty or the needle is the longer of
    the two -- a needle cannot be a span of something shorter than itself.
    """
    na, nb = normalize_text(pare(needle)), normalize_text(pare(haystack))
    if not na or not nb or len(na) > len(nb):
        return 0.0
    if len(na) == len(nb):
        return SequenceMatcher(None, na, nb).ratio()
    return max(
        SequenceMatcher(None, na, nb[start:start + len(na)]).ratio()
        for start in range(len(nb) - len(na) + 1)
    )
