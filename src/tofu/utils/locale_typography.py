## 🍢 ToFU — locale typographic spacing
##
## The space before an exclamation mark is not decoration and it is not
## universal. French (France) sets one; French (Canada) does not, for the
## same mark, in the same language. So the rule cannot be keyed on the
## language subtag alone -- "fr" is not enough to know whether "nos rues !"
## or "nos rues!" is correct, and guessing wrong is a visible typographic
## error either way.
from __future__ import annotations

import re
from typing import Dict, FrozenSet, Optional

## A REGULAR space, not U+202F. The typographically correct character before
## a French exclamation mark is a narrow no-break space, and that belongs in
## the render, where a font either has the glyph or falls back. This module
## normalises RECOGNISED TEXT, which is compared against glossaries, sent to
## translation memory and matched against ground truth -- all of which treat
## an exotic space as a different string. Kept as a named constant so a
## rendering pass can raise it to U+202F without hunting for the literal.
SPACE = " "

## Locales that space these marks, keyed on the region subtag because that is
## where the two French conventions differ.
##
##   fr-FR (and fr-BE, fr-CH): space before ! ? ; :
##   fr-CA: space before : only -- Canadian French sets ! and ? tight, which
##          is the single most common way this rule is got wrong.
##
## Anything not listed here is left exactly as recognised. Silence is the
## right default: a locale whose convention nobody has checked should not be
## "corrected" on a guess.
_SPACED_BEFORE: Dict[str, FrozenSet[str]] = {
    "fr-FR": frozenset("!?;:»"),
    "fr-BE": frozenset("!?;:»"),
    "fr-CH": frozenset("!?;:»"),
    "fr-CA": frozenset(":»"),
}

## Bare tags resolved to a default region, following the map already in
## utils/interchange.py ("fr" -> "fr-FR", "en" -> "en-US"). A manifest that
## carries a region-qualified tag always wins over this.
_DEFAULT_REGION: Dict[str, str] = {
    "fr": "fr-FR",
}

## The opening guillemet takes its space on the FOLLOWING side.
_SPACED_AFTER: Dict[str, FrozenSet[str]] = {
    "fr-FR": frozenset("«"),
    "fr-BE": frozenset("«"),
    "fr-CH": frozenset("«"),
    "fr-CA": frozenset("«"),
}


def _split(tag: Optional[str]) -> tuple[str, Optional[str]]:
    """('fr-CA') -> ('fr', 'CA'); ('fr') -> ('fr', None); ('') -> ('', None)."""
    parts = (tag or "").strip().replace("_", "-").split("-")
    base = parts[0].lower()
    for part in parts[1:]:
        if len(part) == 2 and part.isalpha():
            return base, part.upper()
    return base, None


def resolve_locale(region_lang: Optional[str], asset_lang: Optional[str] = None) -> Optional[str]:
    """The locale whose typographic convention this region should follow.

    The REGION decides the language; the asset may supply the region subtag
    the region itself is missing. That order matters and is not the obvious
    one -- taking "the first tag that parses" would let a region carrying a
    bare "fr" (which every detection emits) silently outrank an asset the
    user explicitly set to "fr-CA", and then space the exclamation marks
    Canadian French sets tight.

    The asset only lends its region when the two agree on the language, so
    a French line on a Canadian-English poster is not read as fr-CA.
    """
    base, region = _split(region_lang)
    if not base:
        base, region = _split(asset_lang)
    if not base:
        return None
    if region:
        return f"{base}-{region}"
    asset_base, asset_region = _split(asset_lang)
    if asset_region and asset_base == base:
        return f"{base}-{asset_region}"
    return _DEFAULT_REGION.get(base)


def spacing_locales() -> FrozenSet[str]:
    """Locales this module has a rule for, for callers that want to skip work."""
    return frozenset(_SPACED_BEFORE) | frozenset(_SPACED_AFTER)


def apply_punctuation_spacing(text: str, locale: Optional[str]) -> str:
    """Insert the locale's space before/after the marks that take one.

    Idempotent, and conservative in three ways that matter on recognised
    text rather than authored text:

      * An existing space of ANY kind before the mark is left alone -- a
        narrow no-break space already there is more correct than the one
        this would insert, not less.
      * A mark at the very start of the string gets nothing. "!" alone is
        a region, not a sentence, and " !" would be a leading space.
      * Repeated marks ("!!", "?!") are spaced once, before the run, which
        is what the convention actually says.
      * The mark has to END a token -- string end, or whitespace after it.
        "Attention:ici" is left alone rather than becoming "Attention :ici",
        which is worse than what was recognised. Closing the gap on BOTH
        sides would be general punctuation normalisation, and this module
        is only responsible for the space the locale asks for.

    Returns the text unchanged when the locale has no rule.
    """
    if not text or not locale:
        return text
    before = _SPACED_BEFORE.get(locale)
    after = _SPACED_AFTER.get(locale)
    result = text
    if before:
        marks = "".join(re.escape(ch) for ch in sorted(before))
        # (not already spaced)(a run of marks)(end or whitespace).
        # \S excludes every kind of existing whitespace, so this cannot
        # stack a second space onto text that was already correct.
        result = re.sub(rf"(\S)([{marks}]+)(?=\s|$)", rf"\1{SPACE}\2", result)
    if after:
        marks = "".join(re.escape(ch) for ch in sorted(after))
        result = re.sub(rf"([{marks}])(\S)", rf"\1{SPACE}\2", result)
    return result
