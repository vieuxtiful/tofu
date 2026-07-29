## 🍢 Sift - ToFU typographic categoriser
## vieuxtiful
"""
sifting a flour separates it into grades; this sifts a font library into
typographic ones — serif, sans, mono, display.

Strictly a BROWSING facet for the Font Manager panel.  It exists as its own
module, and writes nothing back onto FontCoverage, for one reason: the font
LIBRARY is allowed to know what a face declares about itself, but the font
MATCHER is not.  font_matching's whole job is to reach its verdict from the
pixels of the rendered source text (see the oracle note in
scripts/eval_font_match.py) — if a `category` field lived on the dataclass
that font_matching already holds, "just peek at the metadata" would be one
attribute access away forever.  Here it is structurally out of reach.

Resolution order is most-authoritative first: the foundry's own PANOSE
declaration, then the `post` table's fixed-pitch flag, then conservative
family-name tokens, then `unknown`.  Guessing wrongly is worse than
declining to guess: an unknown font still lists and still searches, it just
does not answer a category filter.
"""

from pathlib import Path
from typing import Dict, Optional

try:
    from fontTools.ttLib import TTFont, TTCollection
    HAVE_FONTTOOLS = True
except ImportError:  # degrades to name-based sifting, same as the registry
    HAVE_FONTTOOLS = False

SERIF = "serif"
SANS = "sans"
MONO = "mono"
DISPLAY = "display"
UNKNOWN = "unknown"

CATEGORIES = (SERIF, SANS, MONO, DISPLAY, UNKNOWN)

# PANOSE 1.0, OS/2 byte 1 "Serif Style": 11 Normal Sans, 12 Obtuse Sans and
# 13 Perpendicular Sans are the sans values; 2-10 are the serif variants
# (Cove through Triangle).  The foundry's own declaration beats a hand-kept
# family list, which leaves Didones and other oddities uncategorised purely
# because nobody typed them in.
_PANOSE_SANS = {11, 12, 13}
_PANOSE_SERIF = set(range(2, 11))

# OS/2 byte 0 "Family Type": 2 Latin Text, 3 Latin Hand Written,
# 4 Latin Decorative, 5 Latin Symbol.  3 and 4 are both "display" as far as
# a person picking a typeface is concerned.
_PANOSE_FAMILY_TEXT = (0, 2)
_PANOSE_FAMILY_DISPLAY = (3, 4)

# OS/2 byte 3 "Proportion": 9 Monospaced.
_PANOSE_PROPORTION_MONO = 9

# Deliberately narrow.  Every token here is one that appears in a family
# name to describe the typeface, never incidentally: "Sans" and "Serif" as
# whole words, and mono/code faces which are the ones people most often
# filter for.  Anything vaguer would mislabel more than it labels.
_NAME_TOKENS = (
    (MONO, ("mono", "monospace", "code", "console", "courier", "consolas", "terminal")),
    (DISPLAY, ("display", "decorative", "script", "handwriting", "brush", "stencil", "dingbat")),
    (SERIF, ("serif", "roman", "georgia", "garamond", "baskerville", "didot")),
    (SANS, ("sans", "grotesk", "grotesque", "gothic", "helvetica", "arial", "verdana")),
)

# The registry is a process singleton and the installed library does not
# change while the server runs, so one dict keyed by full face path (with
# its '#n' collection suffix) is the whole cache story.
_CACHE: Dict[str, str] = {}


def _open_face(font_path: str):
    """open one face, honouring the 'file.ttc#3' convention the registry
    uses for collection members (see FontRegistry._key)."""
    base, _, index = font_path.rpartition("#")
    if base and index.isdigit():
        return TTCollection(base, lazy=True).fonts[int(index)]
    return TTFont(font_path, lazy=True)


def _from_panose(font_path: str) -> Optional[str]:
    if not HAVE_FONTTOOLS:
        return None
    try:
        panose = _open_face(font_path)["OS/2"].panose
    except Exception:
        return None
    try:
        if panose.bFamilyType in _PANOSE_FAMILY_DISPLAY:
            return DISPLAY
        if panose.bFamilyType not in _PANOSE_FAMILY_TEXT:
            return None  # symbol/pictorial faces: no honest text category
        # Monospacing is the more useful answer when a face is both, since
        # a serif monospace is what someone filtering for "mono" wants.
        if panose.bProportion == _PANOSE_PROPORTION_MONO:
            return MONO
        if panose.bSerifStyle in _PANOSE_SANS:
            return SANS
        if panose.bSerifStyle in _PANOSE_SERIF:
            return SERIF
    except AttributeError:
        return None
    return None


def _from_post_table(font_path: str) -> Optional[str]:
    """`post.isFixedPitch` is the second opinion on monospacing, and is set
    on plenty of faces whose PANOSE bytes are all zero."""
    if not HAVE_FONTTOOLS:
        return None
    try:
        if _open_face(font_path)["post"].isFixedPitch:
            return MONO
    except Exception:
        return None
    return None


def _from_name(font_path: str, family: Optional[str] = None) -> Optional[str]:
    """last resort: read the family name (or, failing that, the filename).
    Order matters — 'Mono' and 'Display' win over 'Sans' because
    "IBM Plex Sans Mono" is a monospace face first."""
    haystack = (family or Path(font_path.partition("#")[0]).stem).lower()
    for category, tokens in _NAME_TOKENS:
        if any(token in haystack for token in tokens):
            return category
    return None


def sift(font_path: str, family: Optional[str] = None) -> str:
    """Typographic category for one face: serif | sans | mono | display |
    unknown.

    UI browsing facet only.  font_matching must NEVER call this — it must
    reach its face verdict from pixels.  Passing `family` lets the name
    fallback use the registry's normalized family rather than the filename.
    """
    cached = _CACHE.get(font_path)
    if cached is not None:
        return cached
    verdict = (
        _from_panose(font_path)
        or _from_post_table(font_path)
        or _from_name(font_path, family)
        or UNKNOWN
    )
    _CACHE[font_path] = verdict
    return verdict


def clear_cache() -> None:
    """drop memoized verdicts. for tests, and for a future font-library
    reload — nothing in the running server needs it."""
    _CACHE.clear()
