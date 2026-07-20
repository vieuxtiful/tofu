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
