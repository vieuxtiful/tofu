## 🍢 menu — gazetteer-assisted correction for real-world place names
"""
A low-confidence OCR read of a real, well-known place or establishment
sign (a train-station gate, a named street, a chain storefront) can
often be recovered by checking it against a small list of known names,
even when the pixels alone weren't legible enough for the recognizer
to get right. Savor (glyph-confusion correction) already does something
similar for single-character digit/letter swaps, verified pixel-by-
pixel; whole-string place names can't be pixel-verified the same way
(there's no "render the candidate and compare stroke-for-stroke" for a
7-character sign), so this uses a stricter, cruder gate instead: only
touch reads the recognizer itself was already unsure about, and only
apply a candidate that's a strong fuzzy match.

`browse()` is the module's one public entry point (mirrors every other
layer's single-verb API: cicerone.detect, scene.analyze, savor.taste)
— called once by cicerone.detect(), on the FINAL manifest, after every
detection/refinement/recognition pass (including Savor's) has already
had its say.
"""

from typing import List, NamedTuple, Optional

from tofu.core.types import InstText
from tofu.utils.textmatch import fuzzy_similarity

# a read this confident is trusted over the gazetteer -- a correction
# only ever rescues text the recognizer was already unsure about, never
# overrides a confident (even if merely gazetteer-similar) read.
# measured live: japan-street's "招你み焼本練" (nonsense -- not a real
# Japanese word or name) scored 0.511 confidence, just above the old
# 0.5 floor, silently exempting it from a gazetteer check it would
# otherwise have passed (0.5 similarity against the real "お好み焼本陣"
# sign) -- 0.5-0.6 is still a genuinely uncertain read in absolute
# terms, not a confident one, so the floor moved to actually cover it.
CONFIDENCE_FLOOR = 0.6

# minimum fuzzy similarity to accept a gazetteer candidate. deliberately
# strict since, unlike Savor's per-glyph pixel check, there is no pixel
# verification step here -- a weak match is more likely coincidence
# than a genuine misread of the same sign.
SIMILARITY_FLOOR = 0.5

# known place/establishment names likely to recur in street-signage
# photos. seeded for this project's dense-CJK-signage test scenes, but
# meant to grow with whatever real signage future assets turn up --
# not a fixed answer key for one image.
KNOWN_PLACES: List[tuple] = [
    ("歌舞伎町一番街", "ja"),
    ("劇場通り", "ja"),
    ("バンダイ", "ja"),
    ("お好み焼本陣", "ja"),
    ("東南荘", "ja"),
]


class MenuMatch(NamedTuple):
    text: str
    similarity: float


def consult_menu(text: str, lang: Optional[str], confidence: Optional[float]) -> Optional[MenuMatch]:
    """check a low-confidence read against the known-places gazetteer.

    returns the best-matching known name if one clears SIMILARITY_FLOOR,
    or None if the read is already confident, empty, or nothing in the
    gazetteer resembles it closely enough.
    """
    if not text or (confidence or 0) >= CONFIDENCE_FLOOR:
        return None
    best: Optional[MenuMatch] = None
    for candidate, cand_lang in KNOWN_PLACES:
        if lang and cand_lang != lang:
            continue
        score = fuzzy_similarity(text, candidate)
        if score >= SIMILARITY_FLOOR and (best is None or score > best.similarity):
            best = MenuMatch(candidate, score)
    return best


def browse(instances: List[InstText]) -> int:
    """the full menu pass, run once across every instance.

    corrections are recorded on inst.ocr_correction using the same
    {applied, original_text, corrected_text, reason} shape Savor
    already writes, so review/audit stays in one place regardless of
    which layer made the call. returns the number of instances
    corrected (mirrors second_look()/savor.taste()'s return contract).
    """
    corrected = 0
    for inst in instances:
        text = inst.text or ""
        if not text:
            continue
        match = consult_menu(text, inst.detected_language or inst.language, inst.confidence)
        if match is None or match.text == text:
            continue
        inst.ocr_correction = {
            "applied": True,
            "original_text": text,
            "corrected_text": match.text,
            "reason": f"gazetteer match ({match.similarity:.2f} similarity) against a known place/establishment name",
        }
        inst.text = match.text
        corrected += 1
    return corrected
