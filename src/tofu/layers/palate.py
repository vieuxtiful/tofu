## 🍢 Palate -- telling one language from another before anything is cooked.
## vieuxtiful
"""Source-language agreement for text the USER typed.

Distinct from Cicerone's language identification, which reasons about OCR
output across a whole asset.  Here the input is a short string somebody
entered by hand -- a Block, a Ground Truth term -- and the only question is
whether it plausibly belongs to the project's confirmed SOURCE language.

Two rules shape everything below.

**One script vocabulary.**  ToFU has historically carried two: Cicerone's
``ScriptDetector`` answers "han", while manifests and ``lang_to_script`` say
"Hani".  Comparing them scored every CJK string against the Latin model and
raised nothing at all -- a silent failure, because both vocabularies are
individually correct.  Everything here normalises to ISO 15924 at the door.

**Absence of evidence is its own answer.**  "A", "FRI" and "5.2" are not
English, French or anything else; they are too short to carry language.
Reporting them as a mismatch trains the user to dismiss warnings, so they
get ``insufficient_evidence`` and stay silent.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Set

POLICY_REVISION = "palate-1"

## Both vocabularies in use, plus the ISO forms themselves so that
## normalise() is idempotent and can be applied defensively anywhere.
SCRIPT_ALIASES: Dict[str, str] = {
    # Cicerone's ScriptDetector families
    "latin": "Latn", "han": "Hani", "hiragana": "Hira", "katakana": "Kana",
    "hangul": "Hang", "arabic": "Arab", "hebrew": "Hebr", "cyrillic": "Cyrl",
    "greek": "Grek", "devanagari": "Deva", "thai": "Thai",
    # the frontend's coarser answer
    "cjk": "Hani", "unknown": "Zyyy", "common": "Zyyy",
    # ISO 15924, so a normalised value survives a second pass
    "latn": "Latn", "hani": "Hani", "hira": "Hira", "kana": "Kana",
    "hang": "Hang", "kore": "Kore", "jpan": "Jpan", "arab": "Arab",
    "hebr": "Hebr", "cyrl": "Cyrl", "grek": "Grek", "deva": "Deva",
    "thai": "Thai", "zyyy": "Zyyy",
}

## What a language may legitimately contain.  Japanese prose carries kanji,
## both kana AND Latin brand names, dates and abbreviations; treating a Latin
## run inside it as a foreign-language entry is simply wrong.
LANGUAGE_SCRIPTS: Dict[str, Set[str]] = {
    "ja": {"Jpan", "Hani", "Hira", "Kana", "Latn", "Zyyy"},
    "ko": {"Kore", "Hang", "Hani", "Latn", "Zyyy"},
    "zh": {"Hani", "Latn", "Zyyy"},
    "ar": {"Arab", "Latn", "Zyyy"},
    "he": {"Hebr", "Latn", "Zyyy"},
    "ru": {"Cyrl", "Latn", "Zyyy"},
    "uk": {"Cyrl", "Latn", "Zyyy"},
    "el": {"Grek", "Latn", "Zyyy"},
    "hi": {"Deva", "Latn", "Zyyy"},
    "th": {"Thai", "Latn", "Zyyy"},
}
DEFAULT_SCRIPTS: Set[str] = {"Latn", "Zyyy"}

## Languages whose composite script means mixed runs are EXPECTED rather
## than suspicious.
COMPOSITE_LANGUAGES = {"ja", "ko"}

RTL_SCRIPTS = {"Arab", "Hebr"}

## Below this many letters a string cannot carry language evidence.
MIN_ASSESSABLE_LETTERS = 4
## A stopword this long is not an OCR fragment or a two-letter coincidence.
DECISIVE_STOPWORD_LEN = 4

_WORD_RE = re.compile(r"[^\W\d_]+", re.UNICODE)

_RANGES = (
    ("Hani", ((0x4E00, 0x9FFF), (0x3400, 0x4DBF), (0xF900, 0xFAFF))),
    ("Hira", ((0x3040, 0x309F),)),
    ("Kana", ((0x30A0, 0x30FF), (0xFF66, 0xFF9D))),
    ("Hang", ((0xAC00, 0xD7AF), (0x1100, 0x11FF), (0x3130, 0x318F))),
    ("Arab", ((0x0600, 0x06FF), (0x0750, 0x077F), (0x08A0, 0x08FF))),
    ("Hebr", ((0x0590, 0x05FF),)),
    ("Cyrl", ((0x0400, 0x04FF), (0x0500, 0x052F))),
    ("Grek", ((0x0370, 0x03FF),)),
    ("Deva", ((0x0900, 0x097F),)),
    ("Thai", ((0x0E00, 0x0E7F),)),
)


def normalize_script(name: Optional[str]) -> str:
    """Any script name ToFU uses -> ISO 15924. Unknown names become Zyyy."""
    if not name:
        return "Zyyy"
    return SCRIPT_ALIASES.get(str(name).strip().lower(), "Zyyy")


def base_language(code: Optional[str]) -> str:
    """'fr-FR' -> 'fr'."""
    if not code:
        return ""
    return str(code).strip().lower().replace("_", "-").split("-")[0]


def scripts_for_language(code: Optional[str]) -> Set[str]:
    return LANGUAGE_SCRIPTS.get(base_language(code), DEFAULT_SCRIPTS)


def scripts_in(text: str) -> List[str]:
    """Every ISO 15924 script present, most frequent first.

    Digits, punctuation and whitespace map to Zyyy (Common) and are recorded
    but never establish a mismatch -- "5.2" belongs to every language.
    """
    counts: Dict[str, int] = {}
    for char in text or "":
        if char.isspace():
            continue
        code = ord(char)
        family = "Zyyy"
        if char.isalpha():
            family = "Latn"
            for name, ranges in _RANGES:
                if any(low <= code <= high for low, high in ranges):
                    family = name
                    break
        else:
            for name, ranges in _RANGES:
                if any(low <= code <= high for low, high in ranges):
                    family = name
                    break
        counts[family] = counts.get(family, 0) + 1
    return [name for name, _ in sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))]


def direction_for(scripts: Sequence[str]) -> str:
    """Writing direction, so a warning can sit at the logical trailing edge."""
    return "rtl" if any(script in RTL_SCRIPTS for script in scripts) else "ltr"


def _strip_marks(word: str) -> str:
    """Fold accents away: 'république' -> 'republique'."""
    return "".join(
        ch for ch in unicodedata.normalize("NFKD", word)
        if not unicodedata.combining(ch)
    )


def _latin_lexicon():
    ## Reused rather than forked: Cicerone already curates these, and a second
    ## copy would drift.  Imported lazily so this module stays cheap.
    from tofu.layers.cicerone import LATIN_DIACRITICS, LATIN_STOPWORDS
    return LATIN_STOPWORDS, LATIN_DIACRITICS


def identify_latin_language(text: str) -> Optional[str]:
    """Name the Latin-script language of a typed string, or None.

    Diacritics alone are not enough.  A very large share of real signage is
    written without them -- "Rue des Martyrs", "SORTIE", "Ouvert" -- and a
    detector keyed only on accents reports None for all of it, which is
    exactly how French text in an English project went unremarked.

    So stopwords count too, weighted by how much a hit actually proves:
    a long word only one language spells is decisive on its own, while a
    two-letter function word shared across languages needs corroboration.
    That is the same exclusivity rule the diacritic scoring already uses.
    """
    stopwords, diacritics = _latin_lexicon()
    languages = sorted(set(stopwords) | set(diacritics))
    scores: Dict[str, float] = {lang: 0.0 for lang in languages}
    weight: Dict[str, float] = {lang: 0.0 for lang in languages}

    words = [w.lower() for w in _WORD_RE.findall(unicodedata.normalize("NFC", text or ""))]
    if not words:
        return None

    for word in words:
        # The lexicon is stored unaccented ("republique", "eglise", "arret"),
        # so an accented spelling has to be folded before lookup or the two
        # signals never compose: "République" would score as a bare accent
        # and miss the decisive word underneath it.
        folded = _strip_marks(word)
        owners = [lang for lang, stops in stopwords.items()
                  if word in stops or folded in stops]
        if not owners:
            continue
        # Exclusive AND long enough to not be a fragment coincidence: one
        # such word is self-corroborating evidence.
        decisive = len(owners) == 1 and len(word) >= DECISIVE_STOPWORD_LEN
        for lang in owners:
            scores[lang] += 2.0 if decisive else 1.0
            weight[lang] += 2.0 if decisive else 1.0

    for word in words:
        marks = [ch for ch in word if any(ch in table for table in diacritics.values())]
        if not marks:
            continue
        owners = [lang for lang, table in diacritics.items()
                  if any(ch in table for ch in marks)]
        exclusive = len(owners) == 1
        for lang in owners:
            scores[lang] += (2.0 if exclusive else 1.0) * len(marks)
            weight[lang] += 2.0 if exclusive else 1.0

    ranked = sorted(languages, key=lambda lang: (-scores[lang], lang))
    best = ranked[0]
    if weight[best] < 2.0:
        return None
    runner_up = scores[ranked[1]] if len(ranked) > 1 else 0.0
    if scores[best] <= runner_up:
        return None
    return best


@dataclass
class LanguageAssessment:
    """Whether a typed string plausibly belongs to the source language.

    `state` is the only thing a caller should branch on.  No numeric score is
    exposed: these are not calibrated, and showing a number invites the user
    to read precision into a judgement that does not have any.
    """
    state: str                       # agree | probable_mismatch | strong_mismatch
                                     # | insufficient_evidence | mixed_script_expected
    source_language: Optional[str] = None
    detected_language: Optional[str] = None
    scripts: List[str] = field(default_factory=list)
    direction: str = "ltr"
    reasons: List[str] = field(default_factory=list)
    policy_revision: str = POLICY_REVISION

    @property
    def warns(self) -> bool:
        return self.state in {"probable_mismatch", "strong_mismatch"}


def assess(text: str, source_language: Optional[str]) -> LanguageAssessment:
    """Compare a typed string against the project's confirmed SOURCE language.

    Never blocks.  The strongest outcome is a warning the user can keep,
    review, remove or dismiss -- entered text is a statement about the asset,
    and ToFU is not in a position to overrule it.
    """
    raw = (text or "").strip()
    scripts = scripts_in(raw)
    direction = direction_for(scripts)
    source = base_language(source_language)

    if not source:
        return LanguageAssessment("insufficient_evidence", source_language, None,
                                  scripts, direction, ["no confirmed source language"])

    letters = [ch for ch in raw if ch.isalpha()]
    if len(letters) < MIN_ASSESSABLE_LETTERS:
        # Digits, punctuation and abbreviations belong to every language.
        return LanguageAssessment("insufficient_evidence", source_language, None,
                                  scripts, direction, ["too short to carry language evidence"])

    allowed = scripts_for_language(source)
    substantive = [s for s in scripts if s != "Zyyy"]
    foreign = [s for s in substantive if s not in allowed]
    if foreign:
        return LanguageAssessment(
            "strong_mismatch", source_language, None, scripts, direction,
            [f"written in {', '.join(foreign)}, which {source} does not use"],
        )

    if source in COMPOSITE_LANGUAGES and len(substantive) > 1:
        # Japanese with Latin brand names/dates is ordinary, not suspect.
        return LanguageAssessment("mixed_script_expected", source_language, None,
                                  scripts, direction,
                                  ["mixed scripts are expected in this language"])

    if substantive == ["Latn"] and scripts_for_language(source) == DEFAULT_SCRIPTS:
        detected = identify_latin_language(raw)
        if detected and detected != source:
            return LanguageAssessment(
                "probable_mismatch", source_language, detected, scripts, direction,
                [f"reads as {detected}, not {source}"],
            )
        if detected == source:
            return LanguageAssessment("agree", source_language, detected, scripts,
                                      direction, [])
        return LanguageAssessment("insufficient_evidence", source_language, None,
                                  scripts, direction,
                                  ["no distinguishing words or spelling"])

    return LanguageAssessment("agree", source_language, None, scripts, direction, [])
