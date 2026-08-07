## 🍢 Wasabi — ja-JP/zh-Cn glyph normalization
## vieuxtiful
"""
PaddleOCR's "japan" language selector doesn't get a Japanese-specific
recognition model. In ToFU's build (re: paddleocr), `lang="japan"`
resolves to the SAME shared PP-OCRv6_medium_rec model used for
`"ch"`/`"chinese_cht"`/`"en"` (verification: cached
model's README, which lists `language: [en, zh]` -- Japanese isn't
even an officially supported language of it). That model's CTC decode
vocabulary contains both a simplified-Chinese-only glyph form and the
correct Japanese shinjitai form as separate valid output tokens, with
no language-conditioning to prefer the right one -- so a `ja`-labeled
read can confidently emit a Chinese-only character (measured live:
`劇場通り` read back as `剧場通`, `焼肉` read back as `烧肉`). 

This is a real, confirmed characteristic of the model itself, not something
fixable by how ToFU calls it -- thus, this module corrects it after the
fact instead.

`season()` is the module's one public entry point (mirrors every other
layer's single-verb API: cicerone.detect, scene.analyze, savor.taste,
menu.browse) -- called once by cicerone.detect(), after the PaddleOCR
rescue pass and Savor have already had their say, so it corrects
whatever text actually made it into the final manifest regardless of
which engine or pass produced it.
"""

import re
from typing import Dict, List

from tofu.core.types import InstText
from tofu.utils.correction_resources import load_correction_resource, variant_pairs

# curated, growable pairs of (simplified-Chinese-only glyph, Japanese
# shinjitai equivalent) confirmed to be confused by PaddleOCR's shared
# ja/zh recognizer. deliberately NOT a full Unihan variant table --
# most simplified forms actually MATCH Japanese shinjitai (both
# diverged independently from the same traditional form and landed on
# the same simplification), so only genuinely observed divergent pairs
# belong here. add a pair only once it's actually been seen in
# production output, the same discipline menu.py's gazetteer follows.
VARIANT_RESOURCE = load_correction_resource(
    "wasabi/simplified_to_japanese-1.0.0.json"
)
SIMPLIFIED_TO_JAPANESE: Dict[str, str] = variant_pairs(VARIANT_RESOURCE)


def normalize_japanese_kanji(text: str) -> str:
    """swap any known simplified-Chinese-only glyph in `text` for its
    Japanese shinjitai equivalent. characters not in the table (the
    vast majority, including every shinjitai form that already matches
    its simplified counterpart) pass through unchanged."""
    if not text:
        return text
    return "".join(SIMPLIFIED_TO_JAPANESE.get(ch, ch) for ch in text)


# Latin letters whose glyph IS a Cyrillic letter's glyph in ordinary
# typefaces.  Same confusable set textmatch.pare folds for comparison,
# pointed the other way: pare exists to make two strings comparable, this
# exists to repair one of them.
#
# Only pairs that are genuinely indistinguishable are listed.  Lowercase
# b, k, m, h and t are absent on purpose -- Cyrillic в, к, м, н and т do
# not look like them, so a lowercase 'b' inside a Cyrillic word is
# evidence of something else and must not be rewritten.
LATIN_TO_CYRILLIC: Dict[str, str] = {
    "A": "А", "B": "В", "E": "Е", "K": "К", "M": "М", "H": "Н", "O": "О",
    "P": "Р", "C": "С", "T": "Т", "Y": "У", "X": "Х",
    "a": "а", "e": "е", "o": "о", "p": "р", "c": "с", "y": "у", "x": "х",
}


# Cyrillic letters with no Latin lookalike -- Б Д Ж З И Й Л П Ф Ц Ч Ш Щ
# and the rest.  Derived from the confusable map rather than listed, so
# the two can never drift apart: a letter is unambiguous precisely when it
# is not something a Latin character can be mistaken for.
_CYRILLIC_HOMOGLYPHS = set(LATIN_TO_CYRILLIC.values())


def _has_unambiguous_cyrillic(text: str) -> bool:
    """Does ``text`` contain a Cyrillic letter no Latin letter looks like?

    One such character settles the script of the string it sits in, which
    is the only evidence available for a word like ``BMЕСTЕ`` -- every one
    of whose six characters is a shared letterform, leaving the word
    itself genuinely undecidable in isolation.
    """
    from tofu.layers import julienne

    return any(
        script == "Cyrl" and char not in _CYRILLIC_HOMOGLYPHS
        for char, script in julienne.analyse(text).per_character
    )


def restore_cyrillic_homoglyphs(text: str) -> str:
    """Put Latin lookalikes back into Cyrillic inside a Cyrillic region.

    EasyOCR given a joint ``(ru, en)`` charset decodes every shared
    letterform as one script or the other with nothing to constrain the
    choice, so a Russian word comes back part Latin: measured on
    russian-billboard-2, ``ВМЕСТЕ С РОССИЕЙ!`` reads as
    ``BMЕСTЕ С РОССИЕЙ!`` -- three characters of the wrong script, none of
    which look wrong, and every downstream consumer (language voting,
    script detection, translation) then reasons about a word that is not
    the word on the billboard.

    The decision is made at REGION scale and applied at WORD scale, and
    that split is the whole safety argument.

    Region scale, because a corrupted word can be evidence-free: every
    character of ``BMЕСTЕ`` is a shared letterform, so the word alone
    cannot say which script it belongs to and a per-word majority vote
    actually calls it Latin.  Its neighbour ``РОССИЕЙ`` carries И and Й,
    which no Latin letter resembles, and that settles the region.

    Word scale, because a Cyrillic region may legitimately contain a Latin
    word.  Any word with no Cyrillic character in it at all is left alone:
    russian-billboard's ``Za`` -- the Russian war symbol, genuinely Latin,
    set against Cyrillic slogans and correct as read -- is exactly that
    case, and the ground truth for that asset says so explicitly.
    """
    if not text or not _has_unambiguous_cyrillic(text):
        return text
    from tofu.layers import julienne

    out = []
    for token in re.split(r"(\s+)", text):
        counts = julienne.analyse(token).counts
        if not counts.get("Cyrl"):
            # a wholly-Latin word inside a Cyrillic region is a Latin word
            out.append(token)
            continue
        out.append("".join(LATIN_TO_CYRILLIC.get(ch, ch) for ch in token))
    return "".join(out)


def season(instances: List[InstText]) -> int:
    """the full normalization pass, run once across every instance.

    two independent courses, each keyed on what the TEXT is rather than on
    which engine produced it: shinjitai normalization for ja-labeled
    instances, and Cyrillic homoglyph restoration for any instance whose
    words are majority-Cyrillic. the second is deliberately NOT gated on a
    language label -- the defect it repairs makes a Cyrillic region look
    Latin, so the region is routinely labeled 'en' by the time this runs,
    and gating on the label would exempt exactly the regions that need it.

    corrections are recorded on inst.ocr_correction using the same shape
    savor/menu already write, so review/audit stays in one place
    regardless of which layer made the call. returns the number of
    instances corrected."""
    from tofu.layers.savor import _record_correction

    corrected = 0
    for inst in instances:
        text = inst.text or ""
        if not text:
            continue
        changed = False
        if (inst.detected_language or inst.language) == "ja":
            fixed = normalize_japanese_kanji(text)
            if fixed != text:
                _record_correction(
                    inst, applied=True, original_text=text, corrected_text=fixed,
                    reason="simplified-Chinese glyph form normalized to Japanese shinjitai",
                    course="shinjitai_variant",
                    correction_resource=VARIANT_RESOURCE.audit_identity(),
                )
                inst.text = text = fixed
                changed = True
        restored = restore_cyrillic_homoglyphs(text)
        if restored != text:
            _record_correction(
                inst, applied=True, original_text=text, corrected_text=restored,
                reason="Latin homoglyphs inside majority-Cyrillic words restored to Cyrillic",
                course="cyrillic_homoglyph",
            )
            inst.text = restored
            changed = True
        if changed:
            corrected += 1
    return corrected
