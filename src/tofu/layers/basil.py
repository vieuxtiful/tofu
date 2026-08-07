"""Basil -- semantic registration and target-span substitution.

OCR regions are geometric facts: their IDs and boxes are durable anchors for
Cleanse and Scribe.  Translation, on the other hand, is a sentence or named
entity problem.  Basil bridges those layers without ever renumbering or
moving Cicerone regions:

* decide from the language PAIR whether plating can be needed at all, so a
  pairing that cannot reorder is never asked to make an arrangement;
* register source regions as a reading unit on evidence -- a lexicon entity
  spelled across them, or agreeing language, type and surface -- rather than
  on box adjacency alone;
* propose, never apply, a source correction when a known entity is spelled
  across regions one of which the recognizer misread;
* retain both visual/source order and a later target-language semantic order;
* order the regions' existing translations by the target's own syntax;
* align a user-supplied target phrase back to the original region anchors;
* fail open when evidence is insufficient rather than assigning words by
  incidental box order.

The optional Stanza and multilingual-MT/alignment seams are intentionally
offline-only.  Setting a model directory is an explicit deployment action;
this module never downloads a model while someone is scanning an asset.
"""

from __future__ import annotations

import os
import re
import unicodedata
from dataclasses import asdict
from itertools import permutations
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from tofu.core.types import BBox, InstText, SemanticTextUnit, TextManifest


_TOKEN_RE = re.compile(r"[^\W_]+(?:[’'-][^\W_]+)?", re.UNICODE)

# Scripts written without inter-word spaces.  This layer used to join every
# unit's regions with an unconditional " ", which spaced Japanese signage
# like English ("東京 都") and corrupted both source_text and every assigned
# target string for ja/zh/ko/th/km/lo/my.
_UNSPACED_RANGES = (
    (0x2E80, 0x2FDF),   # CJK radicals / Kangxi
    (0x3040, 0x30FF),   # hiragana, katakana
    (0x3400, 0x4DBF),   # CJK unified ideographs extension A
    (0x4E00, 0x9FFF),   # CJK unified ideographs
    (0xF900, 0xFAFF),   # CJK compatibility ideographs
    (0xFF66, 0xFF9F),   # halfwidth katakana
    (0x1100, 0x11FF),   # hangul jamo
    (0xAC00, 0xD7AF),   # hangul syllables
    (0x0E00, 0x0E7F),   # thai
    (0x0E80, 0x0EFF),   # lao
    (0x1000, 0x109F),   # myanmar
    (0x1780, 0x17FF),   # khmer
)
# Ideographic/kana/hangul only: these have near-square glyph boxes, which
# the proximity thresholds below have to account for separately from the
# no-spaces question (Thai is unspaced but not square).
_SQUARE_RANGES = _UNSPACED_RANGES[:8]
_RTL_RANGES = (
    (0x0590, 0x05FF),   # hebrew
    (0x0600, 0x06FF),   # arabic
    (0x0700, 0x074F),   # syriac
    (0x0750, 0x077F),   # arabic supplement
    (0x08A0, 0x08FF),   # arabic extended-A
    (0xFB1D, 0xFDFF),   # hebrew/arabic presentation forms A
    (0xFE70, 0xFEFF),   # arabic presentation forms B
)

# This compact glossary is deliberately evidence, not a general MT system.
# A deployment can register a benchmarked NLLB/translation-memory adapter
# later; an unknown pair must stay review-required instead of being guessed.
_GLOSSARY: Dict[Tuple[str, str], Dict[str, str]] = {
    ("fr", "it"): {
        "rue": "via", "des": "dei", "de": "di", "du": "del",
        "la": "la", "le": "il", "les": "i", "vieux": "vecchi",
        "vieil": "vecchio", "vieille": "vecchia", "murs": "muri",
        "mur": "muro", "place": "piazza", "avenue": "viale",
    },
}
_ACTIVE_GLOSSARY: Optional[Dict[str, Any]] = None

# Cross-region rearrangement is a typological question about the language
# PAIR, not about any one phrase, so the verdict is read off declared
# features rather than inferred per sign.  Three features actually move
# words across region boundaries in signage:
#
#   adj        order of adjective and noun (WALS 87A)
#   gen        order of genitive/possessor and noun (WALS 86A)
#   designator whether the generic type word precedes the proper name
#              ("Via dei Muri", "Улица Ленина") or follows/suffixes it
#              ("Main Street", "Bahnhofstraße", "小坂通り", "小坂路")
#
# ``prenominal_class`` is the fourth, and it is the one this project
# actually meets: French keeps a CLOSED CLASS of adjectives before the
# noun despite a postnominal default (beau, bon, grand, jeune, joli,
# mauvais, nouveau, petit, vieux ...).  fr and it agree on all three
# feature values, so nothing but this flag explains why "VIEUX MURS"
# has to become "MURI VECCHI".
#
# English is listed with gen="post": the of-genitive, not the Saxon
# genitive, is what place-name signage uses ("Osaka in Gifu").  Recording
# it as "both" would make ja->en come out unnecessary, which is wrong.
_TYPOLOGY: Dict[str, Dict[str, Any]] = {
    # Romance: postnominal adjectives, preposed designator
    "fr": {"adj": "post", "gen": "post", "designator": "pre", "prenominal_class": True},
    "it": {"adj": "post", "gen": "post", "designator": "pre", "prenominal_class": False},
    "es": {"adj": "post", "gen": "post", "designator": "pre", "prenominal_class": False},
    "pt": {"adj": "post", "gen": "post", "designator": "pre", "prenominal_class": False},
    "ca": {"adj": "post", "gen": "post", "designator": "pre", "prenominal_class": False},
    "ro": {"adj": "post", "gen": "post", "designator": "pre", "prenominal_class": False},
    # Germanic: prenominal adjectives, designator after or bound
    "en": {"adj": "pre", "gen": "post", "designator": "post", "prenominal_class": False},
    "de": {"adj": "pre", "gen": "post", "designator": "suffix", "prenominal_class": False},
    "nl": {"adj": "pre", "gen": "post", "designator": "suffix", "prenominal_class": False},
    "sv": {"adj": "pre", "gen": "post", "designator": "suffix", "prenominal_class": False},
    "da": {"adj": "pre", "gen": "post", "designator": "suffix", "prenominal_class": False},
    "no": {"adj": "pre", "gen": "post", "designator": "suffix", "prenominal_class": False},
    # Slavic
    "ru": {"adj": "pre", "gen": "post", "designator": "pre", "prenominal_class": False},
    "pl": {"adj": "pre", "gen": "post", "designator": "pre", "prenominal_class": False},
    "cs": {"adj": "pre", "gen": "post", "designator": "pre", "prenominal_class": False},
    # East Asian: uniformly head-final, modifier-first, designator suffixed
    "ja": {"adj": "pre", "gen": "pre", "designator": "suffix", "prenominal_class": False},
    "zh": {"adj": "pre", "gen": "pre", "designator": "suffix", "prenominal_class": False},
    "ko": {"adj": "pre", "gen": "pre", "designator": "suffix", "prenominal_class": False},
    "tr": {"adj": "pre", "gen": "pre", "designator": "suffix", "prenominal_class": False},
    # Semitic and SE Asian
    "ar": {"adj": "post", "gen": "post", "designator": "pre", "prenominal_class": False},
    "he": {"adj": "post", "gen": "post", "designator": "pre", "prenominal_class": False},
    "vi": {"adj": "post", "gen": "post", "designator": "pre", "prenominal_class": False},
    "th": {"adj": "post", "gen": "post", "designator": "pre", "prenominal_class": False},
}

# Role vocabulary, partitioned by language.  The previous single flat bag
# matched a Japanese sign against a French/Italian/English word list; a
# manifest with no src_lang still gets the union, preserving the old
# permissiveness for callers that never set one.
_STREET_DESIGNATORS: Dict[str, set] = {
    "fr": {"rue", "avenue", "boulevard", "chemin", "place", "quai", "route", "impasse", "allée", "cours"},
    "it": {"via", "viale", "piazza", "corso", "strada", "vicolo", "largo", "lungomare"},
    "es": {"calle", "avenida", "plaza", "paseo", "camino", "carretera", "ronda"},
    "pt": {"rua", "avenida", "praça", "travessa", "estrada", "largo"},
    "ca": {"carrer", "avinguda", "plaça", "passeig", "camí"},
    "ro": {"strada", "bulevardul", "calea", "piața"},
    "en": {"street", "road", "lane", "drive", "avenue", "highway", "boulevard", "square", "way", "court"},
    "de": {"straße", "strasse", "weg", "platz", "gasse", "allee", "ring", "damm"},
    "nl": {"straat", "laan", "plein", "weg", "gracht"},
    "ru": {"улица", "проспект", "переулок", "площадь", "шоссе"},
    # CJK designators are SUFFIXES of a token, not tokens of their own --
    # _designator_position() below tests endswith for these languages.
    # Only thoroughfare designators belong here. Stations, parks, bridges,
    # baths and shrines are named places/facilities, not streets; treating
    # every CJK place suffix as a road made phrases such as
    # ``丸太造りの駅舎飛騨小坂駅`` semantically false before Basil aligned it.
    "ja": {"通り", "街道"},
    "zh": {"路", "街", "大道", "巷"},
    "ko": {"로", "길"},
}
_MODIFIERS: Dict[str, set] = {
    "fr": {"vieux", "vieil", "vieille", "vieilles", "nouveau", "nouvelle", "grand", "grande", "petit", "petite", "haut", "haute", "bas", "basse"},
    "it": {"vecchio", "vecchi", "vecchia", "vecchie", "nuovo", "nuova", "grande", "piccolo", "piccola", "alto", "alta", "basso", "bassa"},
    "es": {"viejo", "vieja", "viejos", "nuevo", "nueva", "grande", "pequeño", "pequeña", "alto", "alta", "bajo", "baja"},
    "pt": {"velho", "velha", "novo", "nova", "grande", "pequeno", "pequena", "alto", "alta"},
    "en": {"old", "new", "great", "little", "upper", "lower", "north", "south", "east", "west"},
    "de": {"alt", "alte", "alten", "neu", "neue", "neuen", "groß", "große", "klein", "kleine", "ober", "unter"},
}
_HEAD_WORDS: Dict[str, set] = {
    "fr": {"mur", "murs", "porte", "portes", "pont", "ville", "mont", "saint", "sainte", "église", "château"},
    "it": {"muro", "muri", "porta", "porte", "ponte", "città", "monte", "santo", "santa", "chiesa", "castello"},
    "es": {"muro", "muros", "puerta", "puente", "ciudad", "monte", "san", "santa", "iglesia", "castillo"},
    "pt": {"muro", "muros", "porta", "ponte", "cidade", "monte", "são", "santa", "igreja"},
    "en": {"wall", "walls", "gate", "bridge", "city", "mount", "saint", "church", "castle", "hill"},
    "de": {"mauer", "tor", "brücke", "stadt", "berg", "kirche", "schloss", "burg"},
}


def _lang(code: Optional[str]) -> str:
    return (code or "").lower().replace("_", "-").split("-", 1)[0]


def _locale(code: Optional[str]) -> str:
    value = (code or "").strip().replace("_", "-")
    return value.lower()


def _tokens(text: Optional[str]) -> List[str]:
    return _TOKEN_RE.findall(text or "")


def _normal(token: str) -> str:
    return token.casefold().replace("’", "'")


def _in_ranges(char: str, ranges: Sequence[Tuple[int, int]]) -> bool:
    code = ord(char)
    return any(low <= code <= high for low, high in ranges)


def _script_class(text: Optional[str]) -> str:
    """``unspaced`` / ``rtl`` / ``spaced`` / ``neutral`` by character majority.

    Punctuation, digits and separators are excluded from the vote so that
    "(岐阜県下呂市)" is classified by its kanji rather than its brackets.
    """
    chars = [
        char for char in (text or "")
        if char.strip() and not unicodedata.category(char).startswith(("P", "N", "Z", "C"))
    ]
    if not chars:
        return "neutral"
    unspaced = sum(1 for char in chars if _in_ranges(char, _UNSPACED_RANGES))
    if unspaced * 2 > len(chars):
        return "unspaced"
    rtl = sum(1 for char in chars if _in_ranges(char, _RTL_RANGES))
    if rtl * 2 > len(chars):
        return "rtl"
    return "spaced"


def _is_square_script(text: Optional[str]) -> bool:
    """CJK/kana/hangul: near-square glyph boxes, unlike Latin cap-height."""
    chars = [char for char in (text or "") if char.strip()]
    if not chars:
        return False
    return sum(1 for char in chars if _in_ranges(char, _SQUARE_RANGES)) * 2 > len(chars)


def _join(parts: Sequence[str]) -> str:
    """Concatenate region texts with a script-appropriate separator.

    A space goes between two space-delimited runs and nowhere else, so a
    Japanese unit reads 東京都 rather than "東京 都" while "Rue des" +
    "VIEUX" still reads "Rue des VIEUX".
    """
    pieces = [(part or "").strip() for part in parts]
    pieces = [piece for piece in pieces if piece]
    if not pieces:
        return ""
    joined = pieces[0]
    for piece in pieces[1:]:
        unspaced_seam = (
            _in_ranges(joined[-1], _UNSPACED_RANGES)
            and _in_ranges(piece[0], _UNSPACED_RANGES)
        )
        joined += ("" if unspaced_seam else " ") + piece
    return joined


def _designator_side(features: Dict[str, Any]) -> str:
    """``before`` or ``after`` -- a bound suffix and a following word both
    put the generic type word after the proper name, and for the purpose
    of asking whether words move, that is the same fact."""
    return "before" if features.get("designator") == "pre" else "after"


def pairing(src_lang: Optional[str], targ_lang: Optional[str]) -> Dict[str, Any]:
    """Does this language pairing call for plating at all?

    Basil's whole cross-region rearrangement only earns its keep when the
    target orders the same meaning differently from the source.  Asking
    that up front stops the layer from soliciting a plating decision for a
    pair that cannot need one: ja->zh preserves modifier-head order,
    genitive order and designator position alike, so no arrangement of a
    Japanese sign's regions is ever grammatically forced in Chinese.

    Three-valued and fails open.  ``unknown`` (a language absent from the
    declared table) is never treated as ``unnecessary`` -- ignorance must
    not suppress a unit.
    """
    src, targ = _lang(src_lang), _lang(targ_lang)
    src_features, targ_features = _TYPOLOGY.get(src), _TYPOLOGY.get(targ)
    verdict = {
        "schema": 1,
        "src": src, "targ": targ,
        "features": {"src": src_features, "targ": targ_features},
    }
    if not src or not targ:
        return {**verdict, "verdict": "unknown",
                "reasons": ["source or target language is not set yet"]}
    if src == targ:
        return {**verdict, "verdict": "unnecessary",
                "reasons": ["source and target are the same language"]}
    missing = [code for code, features in ((src, src_features), (targ, targ_features)) if features is None]
    if missing:
        return {**verdict, "verdict": "unknown",
                "reasons": [f"no declared typology for {'/'.join(missing)}; failing open to review"]}

    reasons: List[str] = []
    if src_features["adj"] != targ_features["adj"]:
        reasons.append(
            f"adjective order differs ({src}: {src_features['adj']}nominal, {targ}: {targ_features['adj']}nominal)"
        )
    if src_features["gen"] != targ_features["gen"]:
        reasons.append(
            f"genitive order differs ({src}: {src_features['gen']}nominal, {targ}: {targ_features['gen']}nominal)"
        )
    if _designator_side(src_features) != _designator_side(targ_features):
        reasons.append(
            f"designator position differs ({src}: {_designator_side(src_features)} the name, "
            f"{targ}: {_designator_side(targ_features)} the name)"
        )
    if src_features["prenominal_class"] and not targ_features["prenominal_class"]:
        reasons.append(
            f"{src} has a closed class of prenominal adjectives that {targ} lacks; "
            "a source modifier may have to move after its head"
        )
    if reasons:
        return {**verdict, "verdict": "possible", "reasons": reasons}
    return {**verdict, "verdict": "unnecessary", "reasons": [
        f"{src}->{targ} preserves modifier-head order, genitive order and designator position"
    ]}


def _visual_order(instances: Iterable[InstText]) -> List[InstText]:
    """Visual reading order independent of stable ``rN`` identifiers.

    This repeats Cicerone's useful baseline tolerance at the instance level:
    two words that start a handful of pixels apart remain on one line, then
    sort left-to-right.  It makes the rue-vieux source unit ``r1, r3, r2``
    even though ``r2`` is an immutable ID and may have been allocated first.
    """
    records = [inst for inst in instances if not inst.excluded and (inst.text or "").strip()]
    if len(records) < 2:
        return records
    heights = sorted(max(1, inst.bounding_box.height) for inst in records)
    tolerance = max(4.0, heights[len(heights) // 2] * 0.42)
    lines: List[Dict[str, Any]] = []
    for inst in sorted(records, key=lambda item: (item.bounding_box.y + item.bounding_box.height / 2, item.bounding_box.x)):
        box = inst.bounding_box
        center = box.y + box.height / 2
        line = next((candidate for candidate in lines
                     if abs(center - candidate["center"]) <= max(tolerance, candidate["height"] * 0.42)), None)
        if line is None:
            lines.append({"center": center, "height": box.height, "items": [inst]})
            continue
        line["items"].append(inst)
        count = len(line["items"])
        line["center"] += (center - line["center"]) / count
        line["height"] += (box.height - line["height"]) / count
    ordered: List[InstText] = []
    rtl = sum(1 for inst in records if _script_class(inst.text) == "rtl") * 2 > len(records)
    for line in sorted(lines, key=lambda candidate: candidate["center"]):
        # Arabic/Hebrew signage reads right-to-left within a line; sorting
        # every script by ascending x silently reversed those units, and
        # therefore reversed their region_ids and source_text with them.
        ordered.extend(sorted(line["items"], key=lambda item: item.bounding_box.x, reverse=rtl))
    return ordered


def _column_order(instances: Sequence[InstText]) -> List[InstText]:
    """Vertical (tategaki) reading order: columns right-to-left, top-down.

    Load-bearing for fragmented CJK signage.  In japan-subs the two halves
    of 歓迎 sit at x=280,y=70 and x=249,y=83 -- the first glyph is to the
    RIGHT of the second, so reading them as a row gives 迎欲 and no lexicon
    can recognise it.  Read as one column the same two boxes give 欲迎, a
    single glyph away from the real sign.
    """
    records = [inst for inst in instances if not inst.excluded and (inst.text or "").strip()]
    if len(records) < 2:
        return records
    widths = sorted(max(1, inst.bounding_box.width) for inst in records)
    tolerance = max(4.0, widths[len(widths) // 2] * 0.6)
    columns: List[Dict[str, Any]] = []
    for inst in sorted(records, key=lambda item: (-(item.bounding_box.x + item.bounding_box.width / 2), item.bounding_box.y)):
        box = inst.bounding_box
        center = box.x + box.width / 2
        column = next((candidate for candidate in columns
                       if abs(center - candidate["center"]) <= tolerance), None)
        if column is None:
            columns.append({"center": center, "items": [inst]})
            continue
        column["items"].append(inst)
        count = len(column["items"])
        column["center"] += (center - column["center"]) / count
    ordered: List[InstText] = []
    for column in sorted(columns, key=lambda candidate: -candidate["center"]):
        ordered.extend(sorted(column["items"], key=lambda item: item.bounding_box.y))
    return ordered


def _reading_hypotheses(instances: Sequence[InstText]) -> List[List[InstText]]:
    """Candidate orderings for a component, one per plausible text flow.

    Latin/Cyrillic components have exactly one: the horizontal baseline
    order.  CJK components get the vertical column reading as well, because
    the same boxes spell different strings depending on which is true and
    only a lexicon can arbitrate.
    """
    row = _visual_order(instances)
    if len(row) < 2 or not any(_is_square_script(inst.text) for inst in row):
        return [row]
    column = _column_order(instances)
    if [inst.id for inst in column] == [inst.id for inst in row]:
        return [row]
    return [row, column]


def _contains(region_box: BBox, box: BBox) -> bool:
    cx = box.x + box.width / 2
    cy = box.y + box.height / 2
    return region_box.x <= cx <= region_box.x + region_box.width and region_box.y <= cy <= region_box.y + region_box.height


def _union_box(instances: Sequence[InstText]) -> BBox:
    x0 = min(inst.bounding_box.x for inst in instances)
    y0 = min(inst.bounding_box.y for inst in instances)
    x1 = max(inst.bounding_box.x + inst.bounding_box.width for inst in instances)
    y1 = max(inst.bounding_box.y + inst.bounding_box.height for inst in instances)
    return BBox(x=x0, y=y0, width=x1 - x0, height=y1 - y0)


def _nearby(left: InstText, right: InstText) -> bool:
    """A conservative edge for text on a common sign or label.

    The gap allowances are script-relative.  A Latin word box is roughly
    twice as wide as it is tall, so ``2.5 x height`` is about one word of
    whitespace; a CJK glyph box is square, so the same multiplier is two
    and a half glyphs of empty space and merrily merged neighbouring
    columns of unrelated signage.
    """
    a, b = left.bounding_box, right.bounding_box
    ah, bh = max(1, a.height), max(1, b.height)
    aw, bw = max(1, a.width), max(1, b.width)
    square = _is_square_script(left.text) or _is_square_script(right.text)
    x_allowance = 1.2 if square else 2.5
    y_allowance = 0.9 if square else 1.6
    acy, bcy = a.y + a.height / 2, b.y + b.height / 2
    same_line = abs(acy - bcy) <= max(ah, bh) * 0.46
    x_gap = max(0, max(a.x, b.x) - min(a.x + a.width, b.x + b.width))
    if same_line and x_gap <= max(ah, bh) * x_allowance:
        return True
    y_gap = max(0, max(a.y, b.y) - min(a.y + a.height, b.y + b.height))
    x_overlap = max(0, min(a.x + a.width, b.x + b.width) - max(a.x, b.x))
    if y_gap <= max(ah, bh) * y_allowance and x_overlap / min(aw, bw) >= 0.18:
        return True
    if not square:
        return False
    # A vertical CJK column stacks square glyphs whose boxes barely overlap
    # horizontally when the stroke widths differ; allow a column seam that
    # a horizontal-first test would miss.
    y_column_gap = max(0, max(a.y, b.y) - min(a.y + a.height, b.y + b.height))
    centers_aligned = abs((a.x + aw / 2) - (b.x + bw / 2)) <= max(aw, bw) * 0.6
    return centers_aligned and y_column_gap <= max(ah, bh) * 0.8


def _components(instances: Sequence[InstText], edge=None) -> List[List[InstText]]:
    """Connected components under ``edge`` (defaults to spatial adjacency).

    The predicate is injectable so the panel path can group on language and
    typography alone -- inside a bordered sign, architectural evidence has
    already answered the proximity question.
    """
    if edge is None:
        edge = _nearby
    parents = list(range(len(instances)))

    def find(index: int) -> int:
        while parents[index] != index:
            parents[index] = parents[parents[index]]
            index = parents[index]
        return index

    def join(left: int, right: int) -> None:
        a, b = find(left), find(right)
        if a != b:
            parents[b] = a

    for left in range(len(instances)):
        for right in range(left + 1, len(instances)):
            if edge(instances[left], instances[right]):
                join(left, right)
    groups: Dict[int, List[InstText]] = {}
    for index, inst in enumerate(instances):
        groups.setdefault(find(index), []).append(inst)
    return list(groups.values())


def _vocabulary(table: Dict[str, set], lang: str) -> set:
    """The language's own word set, or the union when the language is
    unknown -- which preserves the old flat-bag behaviour for manifests
    that never recorded a source language."""
    if lang in table:
        return table[lang]
    return set().union(*table.values()) if table else set()


def _designator_position(instances: Sequence[InstText], lang: str) -> Optional[int]:
    """Index of the member carrying the generic type word, or None.

    Position matters and is language-specific: Romance signage puts the
    designator first ("Rue des ..."), English and CJK put it last ("Main
    Street", "小坂通り").  The previous rule only ever looked at the first
    token of the whole unit, so every postposed designator was invisible.
    """
    designators = _vocabulary(_STREET_DESIGNATORS, lang)
    if not designators:
        return None
    features = _TYPOLOGY.get(lang)
    suffixed = features is not None and features.get("designator") == "suffix"
    candidates = []
    for index, inst in enumerate(instances):
        text = (inst.text or "").strip()
        if suffixed:
            if any(text.endswith(word) for word in designators):
                candidates.append(index)
            continue
        words = [_normal(word) for word in _tokens(text)]
        if words and (words[0] in designators or words[-1] in designators):
            candidates.append(index)
    if not candidates:
        return None
    if features is not None and _designator_side(features) == "after":
        return candidates[-1]
    return candidates[0]


def _role_for(inst: InstText, lang: str, is_designator: bool) -> str:
    if is_designator:
        return "street_designator"
    words = [_normal(word) for word in _tokens(inst.text)]
    if any(word in _vocabulary(_MODIFIERS, lang) for word in words):
        return "modifier"
    if any(word in _vocabulary(_HEAD_WORDS, lang) for word in words):
        return "head"
    return "content"


def _local_semantics(
    instances: Sequence[InstText],
    lang: str,
    entity: Optional[Dict[str, Any]] = None,
) -> Tuple[str, float, Dict[str, str]]:
    """Classify a registered unit from evidence, never from token count.

    The previous rule split ``label`` from ``sentence`` on ``len(words)
    <= 8``.  Because the tokenizer matches a whole CJK run as one token,
    that threshold was meaningless for exactly the scripts that most need
    the distinction, and nothing downstream ever branched on the answer.
    """
    designator_index = _designator_position(instances, lang)
    roles = {
        inst.id: _role_for(inst, lang, index == designator_index)
        for index, inst in enumerate(instances)
    }
    if entity is not None:
        # A name recovered from the gazetteer is the strongest evidence
        # available here; its confidence is the alignment's own similarity.
        return "gazetteer_entity", round(0.80 + 0.18 * float(entity.get("similarity", 0.0)), 4), roles
    if designator_index is not None:
        return "street_name", 0.88, roles
    if "modifier" in roles.values() and "head" in roles.values():
        return "noun_phrase", 0.60, roles
    return "label", 0.50, roles


_STANZA_PIPELINES: Dict[Tuple[str, str], Any] = {}


def _stanza_pipeline(language: str, model_dir: str) -> Optional[Any]:
    """One pipeline per (language, model dir), built at most once.

    This used to construct a fresh ``stanza.Pipeline`` inside the
    ``unify_manifest`` loop -- once per unit, per call, on a route that runs
    on every manifest read.  Loading a dependency-parsing model is seconds
    of work, so the seam was unusable in practice even when provisioned.
    """
    key = (language, model_dir)
    if key in _STANZA_PIPELINES:
        return _STANZA_PIPELINES[key]
    try:
        import stanza  # type: ignore
        pipeline = stanza.Pipeline(
            lang=language, processors="tokenize,pos,lemma,depparse,ner",
            model_dir=model_dir, download_method=None, verbose=False,
        )
    except Exception:
        pipeline = None
    _STANZA_PIPELINES[key] = pipeline
    return pipeline


def _stanza_observation(source_text: str, src_lang: Optional[str]) -> Optional[Dict[str, Any]]:
    """Read locally provisioned Stanza models, never triggering a download."""
    model_dir = os.environ.get("TOFU_BASIL_STANZA_DIR")
    if not model_dir or not Path(model_dir).is_dir():
        return None
    language = _lang(src_lang)
    if not language:
        return None
    pipeline = _stanza_pipeline(language, model_dir)
    if pipeline is None:
        return None
    try:
        document = pipeline(source_text)
    except Exception:
        return None
    entities = [
        {"text": entity.text, "type": entity.type}
        for entity in getattr(document, "entities", [])
    ]
    dependencies: List[Dict[str, str]] = []
    for sentence in getattr(document, "sentences", []):
        for word in getattr(sentence, "words", []):
            dependencies.append({"text": word.text, "upos": word.upos, "deprel": word.deprel})
    return {"entities": entities, "dependencies": dependencies}


def infuse(instances: Sequence[InstText], lang: Optional[str]) -> Optional[Dict[str, Any]]:
    """Spell a known entity out of fragmented regions -- internal plating.

    Cicerone routinely splits one two-glyph sign into one region per glyph,
    and may misrecognise one of them: japan-subs reads 歓迎 as r4 "欲"
    (confidence 0.63) and r6 "迎" (0.97).  Neither region alone resembles
    anything; concatenated in column order they are one glyph away from a
    real signage word.  Recovering that is what lets Basil register the two
    regions as ONE entity instead of propagating the fragmentation, and
    lets it report the misread without ever rewriting ``InstText.text``.

    Returns the winning span's evidence, or None.  Three gates stand
    between a fuzzy window and a claim:

    * the span must CROSS a region boundary -- a name inside one region is
      menu.browse()'s job and is already handled there;
    * every differing character must fall inside a region the recognizer
      itself was unsure about, so a correction can never contradict a
      confident read;
    * the alignment reuses menu.py's positional window/diff method rather
      than a second, differently-tuned copy of it.
    """
    from tofu.layers import menu  # layer-local import, mirrors scribe's use of basil

    members = [inst for inst in instances if (inst.text or "").strip()]
    if len(members) < 2:
        return None
    composite = _join([inst.text or "" for inst in members])
    spans: List[Tuple[int, int, InstText]] = []
    cursor = 0
    for inst in members:
        text = (inst.text or "").strip()
        spans.append((cursor, cursor + len(text), inst))
        cursor += len(text)
    if cursor != len(composite):
        # A separator was inserted, so character offsets no longer address
        # glyphs one-for-one.  Space-delimited scripts are served by the
        # token machinery in plan_substitution, not by this path.
        return None

    pool = list(menu.KNOWN_SIGNAGE) + list(menu.KNOWN_PLACES)
    candidates = (
        menu.exact_spans(composite, lang, pool)
        + menu.align_spans(composite, lang, pool, allow_equal_length=True)
    )
    best: Optional[Dict[str, Any]] = None
    for span in candidates:
        covered = [inst for start, end, inst in spans if start < span.end and span.start < end]
        if len(covered) < 2:
            continue
        unsure = True
        for index, _read, _proposed in span.diffs:
            owner = next((inst for start, end, inst in spans if start <= index < end), None)
            if owner is None or (owner.confidence is not None
                                 and owner.confidence >= menu.CONFIDENCE_FLOOR + 0.1):
                unsure = False
                break
        if not unsure:
            continue
        evidence = {
            "members": [inst.id for inst in covered],
            "candidate": span.candidate,
            "similarity": span.similarity,
            "diffs": [
                {"index": index, "read": read, "proposed": proposed}
                for index, read, proposed in span.diffs
            ],
            "read": composite[span.start:span.end],
        }
        if best is None or (span.similarity, span.end - span.start) > (best["similarity"], len(best["candidate"])):
            best = evidence
    return best


_COHESION_HEIGHT_RATIO = 1.6
# Ink distance in plain RGB, matching the tolerance cicerone already uses
# to decide whether stacked CJK fragments share one painted column
# (COLUMN_COLOR_MAX_DIST).  Sampled sign colours are never byte-identical
# -- rue-vieux's three regions read #daf0ed / #d4eaf7 / #d1f1ec off one
# painted plate -- so string equality would reject every real cluster.
_COHESION_COLOR_MAX_DIST = 90.0


def _rgb(value: Any) -> Optional[Tuple[int, int, int]]:
    if not isinstance(value, str):
        return None
    text = value.strip().lstrip("#")
    if len(text) == 3:
        text = "".join(char * 2 for char in text)
    if len(text) != 6:
        return None
    try:
        return int(text[0:2], 16), int(text[2:4], 16), int(text[4:6], 16)
    except ValueError:
        return None


def _cohesive(left: InstText, right: InstText) -> bool:
    """Typographic agreement: same-ish glyph size, same ink where known.

    ``style_profile`` and ``characteristics`` have been on every instance
    all along and this layer read neither, so two regions set in completely
    different type could share a unit purely by being adjacent.  Absent
    evidence fails open -- an unstyled manifest behaves exactly as before.
    """
    lh, rh = max(1, left.bounding_box.height), max(1, right.bounding_box.height)
    if max(lh, rh) / min(lh, rh) > _COHESION_HEIGHT_RATIO:
        return False
    left_rgb = _rgb(getattr(getattr(left, "style_profile", None), "color", None))
    right_rgb = _rgb(getattr(getattr(right, "style_profile", None), "color", None))
    if left_rgb is None or right_rgb is None:
        return True
    distance = sum((a - b) ** 2 for a, b in zip(left_rgb, right_rgb)) ** 0.5
    return distance <= _COHESION_COLOR_MAX_DIST


def _same_language(left: InstText, right: InstText) -> bool:
    """A romanisation line is parallel text, not part of the entity.

    japan-subs stacks ``ひだおさか`` over ``Hida-osaka``; grouping them
    produced a unit that mixed a language with its own transliteration.
    A missing detection is permissive so nothing regresses on manifests
    that never ran language identification.
    """
    left_lang, right_lang = _lang(left.detected_language), _lang(right.detected_language)
    return not left_lang or not right_lang or left_lang == right_lang


def _weight_bucket(inst: InstText) -> str:
    """Coarse weight class from the region's DETECTED typography.

    Reads ``characteristics.font_style`` -- what capture measured off the
    pixels -- in preference to ``style_profile.font_weight``, which is the
    localiser's own pick.  The order used to be the other way round, and
    that made cohort membership a function of the user's downstream
    choices: choose a bold face for two regions of a three-region sign and
    the third falls into a different bucket, ``_same_hand`` then fails on
    every pair crossing that boundary, and the sign's font evidence
    fragments into bouquets that go on to disagree with each other.  A
    bouquet is a claim about how the sign WAS PRINTED, and nothing a user
    selects afterwards can retroactively change that.

    The pick is still consulted when capture recorded no style at all --
    some evidence beats none, and an unstyled manifest keeps its old
    behaviour exactly.
    """
    detected = getattr(getattr(inst, "characteristics", None), "font_style", None)
    value = str((detected or "") or "").lower()
    if not value:
        style = getattr(inst, "style_profile", None)
        value = str((getattr(style, "font_weight", None) or "") or "").lower()
    if any(token in value for token in ("heavy", "black", "ultra")):
        return "heavy"
    if "bold" in value:
        return "bold"
    if "light" in value or "thin" in value:
        return "light"
    return "regular"


# A scene region only speaks for "these share one physical surface" when
# it is actually a bordered thing and the detector meant it.  Shared by
# bunch() and bouquet() so the two can never drift apart on what counts as
# a sign.
_PANEL_LABELS = {"panel", "bordered_region"}
_PANEL_MIN_CONFIDENCE = 0.35


def _same_hand(left: InstText, right: InstText) -> bool:
    """Could these two regions have been set in the SAME typeface?

    Deliberately narrower than semantic cohesion: two lines can belong to
    one phrase yet be set in different faces (a headline over its own
    fine print), and conversely two unrelated words on one shopfront can
    share a face.  So this asks only about the drawing of the letters.

    Reuses ``_cohesive``'s gates rather than inventing a second set --
    that function is already documented as "typographic agreement:
    same-ish glyph size, same ink where known" and its colour tolerance
    is the one cicerone tuned for sampled sign colour.  Proximity is
    required on top, because ink colour and glyph height alone will
    happily unite two different shops' signage across a street scene.
    """
    if not _nearby(left, right):
        return False
    if not _cohesive(left, right):
        return False
    if _weight_bucket(left) != _weight_bucket(right):
        return False
    left_italic = bool(getattr(getattr(left, "style_profile", None), "italic", False))
    right_italic = bool(getattr(getattr(right, "style_profile", None), "italic", False))
    return left_italic == right_italic


def bouquet(manifest: TextManifest) -> List[Dict[str, Any]]:
    """Tie regions that appear to be set in one typeface into a bundle.

    A bouquet garni is bound once and seasons the whole pot; these are the
    regions that should end up seasoned by one font decision rather than
    each arguing for its own.

    Why this has to exist: font_matching scores every region in ISOLATION,
    and a region's glyph silhouette is a small, noisy sample of a typeface.
    Measured on the la-rue-sans-nom plaque -- a single enamel sign, one
    face, two lines -- "La rue" ranked Centaur (a serif) top at 0.784
    while "SANS-NOM" ranked Franklin Gothic Demi Cond (a condensed sans)
    top at 0.813.  Both cannot be right about one sign.  Eight wide capitals
    and six mixed-case letters simply project different statistics, and
    whichever face happens to win each small sample wins outright.

    This layer contributes only the JUDGEMENT that a set of regions shares
    a hand; font_matching still owns every score, because the pixel
    evidence and the scoring kernel live there.

    Regions reach one bundle by either of two routes: a confident scene
    panel saying they share a physical surface, or pairwise ``_same_hand``
    proximity.  Both then have to clear the same typographic gates.

    Returns ``[{"id": "c1", "region_ids": [...]}, ...]`` for bundles of two
    or more.  A lone region is not a bouquet -- it already has its own
    answer and nothing to reconcile with.
    """
    eligible = [
        inst for inst in manifest.instances
        if not inst.excluded and (inst.text or "").strip()
    ]
    parent: Dict[str, str] = {inst.id: inst.id for inst in eligible}

    def find(node: str) -> str:
        while parent[node] != node:
            parent[node] = parent[parent[node]]
            node = parent[node]
        return node

    def union(a: str, b: str) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    def _panel_same_hand(left: InstText, right: InstText) -> bool:
        """``_same_hand`` with the proximity question already answered.

        Everything about the drawing of the letters still has to agree --
        notably ``_cohesive``, so a headline and its own fine print stay
        apart on the height ratio rather than being bound together merely
        for sharing a frame.

        The language gate is here and deliberately NOT in ``_same_hand``.
        This is the permissive path, so it carries the guard that stops a
        CJK headline being bound to its own romanisation across a wide
        sign.  It can only ever withhold a NEW union: two adjacent regions
        in different languages still meet through the pairwise pass below,
        exactly as they did before.
        """
        return (
            _same_language(left, right)
            and _cohesive(left, right)
            and _weight_bucket(left) == _weight_bucket(right)
            and bool(getattr(getattr(left, "style_profile", None), "italic", False))
            == bool(getattr(getattr(right, "style_profile", None), "italic", False))
        )

    # Scene pre-pass.  Inside a bordered sign the architecture has already
    # answered the proximity question -- the same reasoning bunch() applies
    # to panels, and the reason it groups on _alike rather than
    # _adjacent_and_alike there.  It matters here because ``_nearby``
    # allows a vertical gap of 1.6x glyph height, which a large plaque with
    # generously leaded lines exceeds while still being unmistakably one
    # sign set in one face.
    #
    # Strictly additive: union-find only ever merges, so this can add
    # groupings the pairwise pass would miss and can never break one it
    # would have made.  A manifest with no scene regions behaves
    # identically to before.
    for region in manifest.scene_regions or []:
        if region.semantic_label not in _PANEL_LABELS or region.confidence < _PANEL_MIN_CONFIDENCE:
            continue
        members = [inst for inst in eligible if _contains(region.bbox, inst.bounding_box)]
        for i, left in enumerate(members):
            for right in members[i + 1:]:
                if _panel_same_hand(left, right):
                    union(left.id, right.id)

    for i, left in enumerate(eligible):
        for right in eligible[i + 1:]:
            if _same_hand(left, right):
                union(left.id, right.id)

    order = [inst.id for inst in eligible]
    grouped: Dict[str, List[str]] = {}
    for region_id in order:
        grouped.setdefault(find(region_id), []).append(region_id)

    bundles: List[Dict[str, Any]] = []
    for root in order:
        members = grouped.get(root)
        if not members or len(members) < 2:
            continue
        bundles.append({"id": f"c{len(bundles) + 1}", "region_ids": members})
    return bundles


def _bind(
    eligible: Sequence[InstText],
    scene_regions: Optional[Sequence[Any]],
    pairwise_edge,
    panel_edge,
) -> List[Dict[str, Any]]:
    """Union-find over a panel pre-pass and a pairwise pass, as bouquet does.

    Written as its own function rather than by refactoring ``bouquet``,
    which keeps its own copy: that function's groupings are pinned by a
    long row of deliberately-measured tests, and the point of this file's
    second cohort pass is to change nothing about the first.
    """
    parent: Dict[str, str] = {inst.id: inst.id for inst in eligible}

    def find(node: str) -> str:
        while parent[node] != node:
            parent[node] = parent[parent[node]]
            node = parent[node]
        return node

    def union(a: str, b: str) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    for region in scene_regions or []:
        if region.semantic_label not in _PANEL_LABELS or region.confidence < _PANEL_MIN_CONFIDENCE:
            continue
        members = [inst for inst in eligible if _contains(region.bbox, inst.bounding_box)]
        for i, left in enumerate(members):
            for right in members[i + 1:]:
                if panel_edge(left, right):
                    union(left.id, right.id)

    for i, left in enumerate(eligible):
        for right in eligible[i + 1:]:
            if pairwise_edge(left, right):
                union(left.id, right.id)

    order = [inst.id for inst in eligible]
    grouped: Dict[str, List[str]] = {}
    for region_id in order:
        grouped.setdefault(find(region_id), []).append(region_id)

    bundles: List[Dict[str, Any]] = []
    for root in order:
        members = grouped.get(root)
        if not members or len(members) < 2:
            continue
        bundles.append({"id": f"c{len(bundles) + 1}", "region_ids": members})
    return bundles


def _one_hand_any_size(left: InstText, right: InstText) -> bool:
    """Could these be the same FAMILY, at whatever size and weight?

    The narrower question ``_same_hand`` asks -- same size, same ink, same
    weight -- is the right one for claiming two regions carry the identical
    face.  It is the wrong one for claiming they came from one family, and
    on real signage it is wrong often.  Measured on the la-bastille poster,
    every pair the eye reads as one hand was refused by a gate about
    something other than the shape of the letters:

      * height ratio: '80' over 'bis' is 2.33, 'en' over '1789' is 2.96.
        A sign sets its house number large and its qualifier small in one
        hand; glyph size is not typeface identity.
      * colour: 'la Bastille' samples #6c3f3c and 'Rue St. Antoine'
        #bbbda1 on the same engraving, a distance of 180 against a
        tolerance of 90.  Ageing and uneven lighting move sampled ink
        colour far more than a change of face does.
      * weight bucket: a 15px 'bis' reads bold and 'Avenue' reads light
        italic on a poster lettered by one hand.  At these sizes the
        weight detector is guessing.

    So this keeps only the two gates that survive a change of size:

      * ``_nearby``, which is what holds the title block apart from the
        address block below it -- and, on a street scene, one shopfront
        apart from the next.  Dropping it would make a photograph into a
        single cohort, which is not a claim anyone can defend.
      * ``_same_language``, carried on BOTH routes here rather than only
        the panel route.  In ``bouquet`` the typographic gates did this
        work incidentally; with them gone, a CJK headline sitting directly
        above its own romanisation would otherwise join it.

    Italic is NOT a gate here, though it reads like one it should be.  A
    true italic is a different drawing of the letters, so the first version
    of this refused to pair across it -- and on la-bastille that put
    "Avenue" and "Champs" in a cohort of their own, away from the address
    they belong to, on a detector reading of *light italic* for lettering
    the eye reads as upright.  At 15-40px the slant detector is guessing as
    freely as the weight one.

    Nothing is lost by dropping it, because slant is still decided -- one
    step later and with better evidence.  ``font_matching._weight_distance``
    ranks slant AHEAD of weight when each region picks its face within the
    cohort's family, so an italic region in a mixed cohort takes the
    family's Italic face while its neighbours take Regular or Bold.  Italic
    decides the FACE, not the membership.
    """
    return _nearby(left, right) and _same_language(left, right)


def mother_sauce(manifest: TextManifest) -> List[Dict[str, Any]]:
    """Regions that share a typeface FAMILY, whatever size or weight.

    The five mother sauces are the bases every daughter sauce derives
    from; a family is the same thing for type, with the weights as its
    daughters.  ``bouquet`` asks which regions carry the identical face
    and is right to be strict about it.  This asks the looser question a
    localiser actually faces -- which regions should be offered ONE family
    to choose from -- and leaves each region its own weight within it.

    Same shape as ``bouquet``: ``[{"id": "c1", "region_ids": [...]}, ...]``
    for bundles of two or more, and a lone region is not a bundle.

    Necessarily a superset of ``bouquet``'s bundles on the same manifest,
    since every gate here is one ``bouquet`` also applies.  That is the
    intended relationship: the strict pass settles the identical face
    where it can, and this one covers the rest of the sign.
    """
    eligible = [
        inst for inst in manifest.instances
        if not inst.excluded and (inst.text or "").strip()
    ]
    return _bind(
        eligible,
        manifest.scene_regions,
        _one_hand_any_size,
        # Inside a bordered panel the architecture has already answered
        # proximity, exactly as it does for bouquet.
        _same_language,
    )


def bunch(manifest: TextManifest, verdict: str) -> List[Dict[str, Any]]:
    """Group instances into candidate sprigs, gated on actual evidence.

    The previous behaviour was to take every geometric component and force
    a semantic unit onto it.  On japan-subs that produced a ten-region
    "sentence" spanning three physically separate signs plus their own
    romanisation.  A component now becomes a unit only when either the
    gazetteer confirms an entity spelled across its members, or the pair
    can actually reorder AND a confident scene panel says the members
    share one physical surface.  Otherwise nothing is emitted and the
    regions stay independent -- which is what this module's docstring has
    always promised and what the code never did.
    """
    src_lang = _lang(manifest.src_lang)
    visual = _visual_order(manifest.instances)
    order = {inst.id: index for index, inst in enumerate(visual)}
    claimed: set[str] = set()
    candidates: List[Tuple[List[InstText], bool]] = []  # (members, from_panel)

    for region in manifest.scene_regions or []:
        if region.semantic_label not in _PANEL_LABELS or region.confidence < _PANEL_MIN_CONFIDENCE:
            continue
        members = [inst for inst in visual
                   if inst.id not in claimed and _contains(region.bbox, inst.bounding_box)]
        # Inside a bordered sign the architecture has already answered the
        # proximity question, so only language and typography still gate.
        for cluster in _components(members, _alike):
            if len(cluster) >= 2:
                candidates.append((_visual_order(cluster), True))
                claimed.update(inst.id for inst in cluster)

    remaining = [inst for inst in visual if inst.id not in claimed]
    for component in _components(remaining, _adjacent_and_alike):
        if len(component) >= 2:
            candidates.append((_visual_order(component), False))

    sprigs: List[Dict[str, Any]] = []
    used: set[str] = set()
    for members, from_panel in candidates:
        entity = None
        ordering = members
        for hypothesis in _reading_hypotheses(members):
            found = infuse(hypothesis, manifest.src_lang or src_lang)
            if found is not None:
                entity, ordering = found, hypothesis
                break
        if entity is not None:
            covered = [inst for inst in ordering if inst.id in set(entity["members"])]
            if len(covered) >= 2 and not any(inst.id in used for inst in covered):
                sprigs.append({"members": covered, "entity": entity, "evidence": "gazetteer_entity"})
                used.update(inst.id for inst in covered)
                continue
        if verdict == "unnecessary" or not (from_panel or _single_line(members)):
            continue
        ordered = _visual_order(members)
        if any(inst.id in used for inst in ordered):
            continue
        sprigs.append({"members": ordered, "entity": None, "evidence": "panel_cohesion"})
        used.update(inst.id for inst in ordered)

    # Stable unit numbering follows the actual source reading order, never
    # raw rN allocation order.
    sprigs.sort(key=lambda sprig: min(order.get(inst.id, 10 ** 9) for inst in sprig["members"]))
    return sprigs


def _alike(left: InstText, right: InstText) -> bool:
    """Language and typographic agreement, without a proximity claim."""
    return _same_language(left, right) and _cohesive(left, right)


def _adjacent_and_alike(left: InstText, right: InstText) -> bool:
    """The full grouping edge outside a panel: near, same language, same type."""
    return _nearby(left, right) and _alike(left, right)


def _single_line(members: Sequence[InstText]) -> bool:
    """Every member shares one baseline.

    A confident panel is the strongest evidence that separated lines belong
    to one sign, but requiring it would make Basil depend on Scene having
    run at all.  Words sitting on a single baseline, already filtered to one
    language and one typeface and one tight gap, are their own evidence of a
    phrase -- which is what a sign's top line usually is.
    """
    if len(members) < 2:
        return False
    centers = [inst.bounding_box.y + inst.bounding_box.height / 2 for inst in members]
    heights = sorted(max(1, inst.bounding_box.height) for inst in members)
    tolerance = max(4.0, heights[len(heights) // 2] * 0.46)
    return max(centers) - min(centers) <= tolerance


def suggest_plating(
    manifest: TextManifest,
    unit: SemanticTextUnit,
    targ_lang: Optional[str],
) -> Dict[str, Any]:
    """Order the regions' existing target strings by the target's syntax.

    When the translation already exists -- entered in the Translate table,
    or round-tripped through a CAT tool or TMS -- Basil should hand back an
    arranged phrase instead of an empty box.  With fr->it the members carry
    "Via dei" (designator), "VECCHI" (modifier) and "MURI" (head); Italian
    puts the adjective after its noun, so the proposal is "Via dei MURI
    VECCHI" and the plating the user has to confirm is already written out.
    """
    by_id = {inst.id: inst for inst in manifest.instances}
    parts: List[Tuple[str, str, str]] = []
    covered = 0
    for region_id in unit.region_ids:
        inst = by_id.get(region_id)
        text = (inst.target_text or "").strip() if inst is not None else ""
        if text:
            covered += 1
        parts.append((unit.semantic_roles.get(region_id, "content"), region_id, text))
    coverage = round(covered / len(unit.region_ids), 4) if unit.region_ids else 0.0
    base = {"schema": 1, "target_text": "", "region_order": [], "coverage": coverage}
    if coverage < 1.0:
        return {**base, "basis": "waiting for every region to be translated"}
    features = _TYPOLOGY.get(_lang(targ_lang))
    if features is None:
        return {**base, "basis": f"no declared typology for '{_lang(targ_lang) or '?'}'"}

    if features["adj"] == "post":
        ranks = {"street_designator": 0, "head": 1, "content": 2, "modifier": 3}
        basis = f"{_lang(targ_lang)}: postnominal adjectives — the head precedes its modifier"
    else:
        ranks = {"street_designator": 0, "modifier": 1, "content": 2, "head": 3}
        basis = f"{_lang(targ_lang)}: prenominal adjectives — the modifier precedes its head"
    if _designator_side(features) == "after":
        ranks["street_designator"] = 9
        basis += "; the designator follows the name"
    ordered = sorted(parts, key=lambda part: (ranks.get(part[0], 5), unit.region_ids.index(part[1])))
    return {
        **base,
        "target_text": _join([text for _role, _region_id, text in ordered]),
        "region_order": [region_id for _role, region_id, _text in ordered],
        "basis": basis,
    }


def unify_manifest(manifest: TextManifest) -> List[SemanticTextUnit]:
    """Register source reading units without touching user translations.

    Existing applied substitution provenance is retained only when both the
    source text and immutable member IDs still match.  If OCR/source editing
    changes either, the plan becomes stale and is deliberately discarded.
    """
    previous = {
        (tuple(unit.region_ids), unit.source_text): unit.substitution
        for unit in (manifest.semantic_units or [])
    }
    accepted_repairs = {
        tuple(unit.region_ids)
        for unit in (manifest.semantic_units or [])
        if isinstance(unit.ocr_repair, dict) and unit.ocr_repair.get("accepted")
    }
    verdict = pairing(manifest.src_lang, manifest.targ_lang)
    src_lang = _lang(manifest.src_lang)

    units: List[SemanticTextUnit] = []
    for number, sprig in enumerate(bunch(manifest, verdict["verdict"]), 1):
        members = sprig["members"]
        entity = sprig["entity"]
        member_ids = [inst.id for inst in members]
        read = _join([inst.text or "" for inst in members])
        if not read:
            continue
        repair = None
        source = read
        if entity is not None and entity["diffs"]:
            # The corrected spelling is what the unit MEANS; the raw read is
            # kept beside it so the user can see exactly what is proposed.
            # InstText.text is never touched -- accepting is an explicit act.
            repair = {
                "schema": 1,
                "read": entity["read"],
                "proposed": entity["candidate"],
                "similarity": entity["similarity"],
                "diffs": entity["diffs"],
                "evidence": "cross_region_gazetteer_alignment",
                "accepted": tuple(member_ids) in accepted_repairs,
            }
            if repair["accepted"]:
                source = read.replace(entity["read"], entity["candidate"], 1)
        entity_type, confidence, roles = _local_semantics(members, src_lang, entity)
        provider = "deterministic_layout"
        if entity is not None:
            provider = "gazetteer+deterministic_layout"
        elif _stanza_observation(source, manifest.src_lang) is not None:
            # NER evidence is intentionally advisory: record that the seam
            # answered, but do not turn arbitrary names into a decision.
            provider = "stanza+deterministic_layout"
        unit = SemanticTextUnit(
            id=f"u{number}",
            region_ids=member_ids,
            source_text=source,
            bbox=_union_box(members),
            entity_type=entity_type,
            confidence=confidence,
            analysis_provider=provider,
            semantic_roles=roles,
            review_required=(repair is not None and not repair["accepted"]) or entity_type not in {"street_name", "gazetteer_entity"},
            substitution=previous.get((tuple(member_ids), source)),
            pairing=verdict,
            ocr_repair=repair,
        )
        unit.suggestion = suggest_plating(manifest, unit, manifest.targ_lang)
        units.append(unit)
    manifest.semantic_units = units
    return units


def accept_repair(manifest: TextManifest, unit_id: str, accepted: bool) -> SemanticTextUnit:
    """Record the user's verdict on a proposed cross-region source repair.

    Accepting changes only the unit's own ``source_text``; the regions'
    boxes, ids and OCR text all stay exactly as Cicerone left them, so a
    later re-detection or a rejected repair loses nothing.
    """
    unit = next((item for item in (manifest.semantic_units or []) if item.id == unit_id), None)
    if unit is None:
        raise KeyError(f"semantic unit '{unit_id}' not found")
    if not isinstance(unit.ocr_repair, dict):
        raise ValueError(f"semantic unit '{unit_id}' has no proposed repair")
    unit.ocr_repair = {**unit.ocr_repair, "accepted": bool(accepted)}
    read, proposed = unit.ocr_repair.get("read", ""), unit.ocr_repair.get("proposed", "")
    if accepted and read and proposed and read in unit.source_text:
        unit.source_text = unit.source_text.replace(read, proposed, 1)
    elif not accepted and read and proposed and proposed in unit.source_text:
        unit.source_text = unit.source_text.replace(proposed, read, 1)
    unit.review_required = not accepted
    return unit


def modify_unit_members(
    manifest: TextManifest,
    unit_id: str,
    *,
    add_region_id: Optional[str] = None,
    remove_region_id: Optional[str] = None,
) -> SemanticTextUnit:
    """Add or remove a single region from a semantic unit's membership.

    Only the unit's own ``region_ids``, ``source_text``, ``bbox`` and
    ``suggestion`` change.  Region boxes, ids and OCR text are untouched,
    so a later re-detection (``unify_manifest``) can still re-group them.

    Removing a region from one unit does NOT automatically add it to
    another; adding a region that already belongs to a different unit
    removes it from that unit first, so a region is always in at most one
    unit.
    """
    unit = next((item for item in (manifest.semantic_units or []) if item.id == unit_id), None)
    if unit is None:
        raise KeyError(f"semantic unit '{unit_id}' not found")
    by_id = {inst.id: inst for inst in manifest.instances}

    if remove_region_id is not None:
        if remove_region_id not in unit.region_ids:
            raise ValueError(f"region '{remove_region_id}' is not a member of unit '{unit_id}'")
        new_ids = [rid for rid in unit.region_ids if rid != remove_region_id]
    elif add_region_id is not None:
        if add_region_id in unit.region_ids:
            raise ValueError(f"region '{add_region_id}' is already a member of unit '{unit_id}'")
        if add_region_id not in by_id:
            raise ValueError(f"region '{add_region_id}' does not exist in the manifest")
        # Remove from any other unit that currently owns it.
        for other in (manifest.semantic_units or []):
            if other.id != unit_id and add_region_id in other.region_ids:
                other.region_ids = [rid for rid in other.region_ids if rid != add_region_id]
                other_members = [by_id[rid] for rid in other.region_ids if rid in by_id]
                other.source_text = _join([inst.text or "" for inst in other_members]) if other_members else ""
                other.bbox = _union_box(other_members) if other_members else other.bbox
                other.substitution = None
                other.suggestion = suggest_plating(manifest, other, manifest.targ_lang)
        new_ids = [*unit.region_ids, add_region_id]
    else:
        raise ValueError("either add_region_id or remove_region_id must be provided")

    unit.region_ids = new_ids
    members = [by_id[rid] for rid in new_ids if rid in by_id]
    unit.source_text = _join([inst.text or "" for inst in members]) if members else ""
    unit.bbox = _union_box(members) if members else unit.bbox
    # Clear stale substitution; the membership change invalidates any prior plan.
    unit.substitution = None
    unit.suggestion = suggest_plating(manifest, unit, manifest.targ_lang)
    return unit


def create_unit(manifest: TextManifest, region_ids: Optional[List[str]] = None) -> SemanticTextUnit:
    """Create a new, user-authored semantic unit (a 'plate').

    Starts empty or with the given region IDs.  The unit gets the next
    available ``uN`` id and a pairing verdict from the manifest's language
    pair.  Region boxes, ids and OCR text are untouched.
    """
    existing = {unit.id for unit in (manifest.semantic_units or [])}
    number = 1
    while f"u{number}" in existing:
        number += 1
    by_id = {inst.id: inst for inst in manifest.instances}
    member_ids = [rid for rid in (region_ids or []) if rid in by_id]
    members = [by_id[rid] for rid in member_ids]
    verdict = pairing(manifest.src_lang, manifest.targ_lang)
    unit = SemanticTextUnit(
        id=f"u{number}",
        region_ids=member_ids,
        source_text=_join([inst.text or "" for inst in members]) if members else "",
        bbox=_union_box(members) if members else BBox(0, 0, 0, 0),
        entity_type="unknown",
        confidence=0.0,
        analysis_provider="user_created",
        semantic_roles={},
        review_required=True,
        pairing=verdict,
    )
    unit.suggestion = suggest_plating(manifest, unit, manifest.targ_lang)
    manifest.semantic_units = [*manifest.semantic_units, unit]
    return unit


def delete_unit(manifest: TextManifest, unit_id: str) -> None:
    """Permanently remove a semantic unit from the manifest.

    Region boxes, ids and OCR text are untouched -- only the unit entry is
    dropped, so its former members can be re-grouped by a later
    ``unify_manifest`` re-detection or re-assigned by the user.
    """
    units = manifest.semantic_units or []
    if not any(unit.id == unit_id for unit in units):
        raise KeyError(f"semantic unit '{unit_id}' not found")
    manifest.semantic_units = [unit for unit in units if unit.id != unit_id]


def provider_statuses() -> List[Dict[str, Any]]:
    """Deployment-visible model seams; this never probes remote services."""
    stanza_dir = os.environ.get("TOFU_BASIL_STANZA_DIR")
    nllb_dir = os.environ.get("TOFU_BASIL_NLLB_MODEL")
    align_dir = os.environ.get("TOFU_BASIL_ALIGN_MODEL")
    providers = [
        {
            "id": "deterministic_layout", "active": True,
            "purpose": "visual reading units and verified local glossary alignment",
        },
        {
            "id": "stanza", "active": bool(stanza_dir and Path(stanza_dir).is_dir()),
            "purpose": "local token/POS/dependency/NER evidence", "model_dir": stanza_dir,
        },
        {
            "id": "nllb", "active": bool(nllb_dir and Path(nllb_dir).exists()),
            "purpose": "candidate sentence translation; benchmark-gated and never auto-applied", "model_dir": nllb_dir,
        },
        {
            "id": "awesome_align", "active": bool(align_dir and Path(align_dir).exists()),
            "purpose": "candidate cross-language token/span alignment; benchmark-gated", "model_dir": align_dir,
        },
    ]
    providers.append({
        "id": "uploaded_glossary",
        "active": bool(_ACTIVE_GLOSSARY),
        "purpose": "user-supplied termbase (global/project cascade)",
        "mode": _ACTIVE_GLOSSARY.get("mode") if _ACTIVE_GLOSSARY else None,
        "entry_count": _ACTIVE_GLOSSARY.get("entry_count", 0) if _ACTIVE_GLOSSARY else 0,
    })
    return providers


def set_active_glossary(info: Optional[Dict[str, Any]]) -> None:
    """Expose current glossary metadata to deployment/provider diagnostics.

    Diagnostics only.  ``plan_substitution`` never reads this; it takes the
    lexicon as an explicit argument, precisely so one project's uploaded
    termbase can never leak into another's alignment through module state.
    """
    global _ACTIVE_GLOSSARY
    _ACTIVE_GLOSSARY = info


def plan_substitution(
    manifest: TextManifest,
    unit_id: str,
    target_text: str,
    targ_lang: Optional[str],
    external_lexicon: Optional[Dict[Tuple[str, str], Dict[str, str]]] = None,
) -> Dict[str, Any]:
    """Plan, but never mutate, target assignments for a semantic unit."""
    unify_manifest(manifest)
    unit = next((item for item in manifest.semantic_units if item.id == unit_id), None)
    if unit is None:
        raise KeyError(f"semantic unit '{unit_id}' not found")
    phrase = (target_text or "").strip()
    base: Dict[str, Any] = {
        "schema": 1,
        "unit_id": unit.id,
        "source_text": unit.source_text,
        "target_text": phrase,
        "source_region_order": list(unit.region_ids),
        # Cubes are immutable visual boxes.  A semantic block can be plated
        # into a different cube when target syntax reorders source meaning.
        "spatial_anchor_order": list(unit.region_ids),
        "target_region_order": [],
        "assignments": [],
        "method": "manual_required",
        "confidence": 0.0,
        "review_required": True,
        "warnings": [],
        # Carried so a caller can see WHY plating was or was not worth
        # asking about, without re-deriving the typology itself.
        "pairing": unit.pairing or pairing(manifest.src_lang, targ_lang),
        "suggestion": unit.suggestion,
    }
    if not phrase:
        base["warnings"].append("Enter the complete target phrase before planning placement.")
        return base

    # Unspaced/CJK targets (ko/ja/zh/th/km/lo/my) have no word delimiter, so
    # the glossary-verified token alignment below cannot segment the target
    # phrase back to source regions.  The user's SlotEditor arrangement is
    # encoded in the concatenated target_text: match it against the
    # permutations of per-region translations, then tiebreak with the
    # typology-derived order from suggest_plating.
    if _script_class(phrase) == "unspaced":
        by_id = {inst.id: inst for inst in manifest.instances}
        region_texts = {
            region_id: (by_id[region_id].target_text or "").strip()
            for region_id in unit.region_ids if region_id in by_id
        }
        missing = [rid for rid in unit.region_ids if not region_texts.get(rid)]
        if missing:
            base["warnings"].append(
                f"Translate every region before plating; missing target text for {', '.join(missing)}."
            )
            return base
        candidates = [
            list(order) for order in permutations(unit.region_ids)
            if "".join(region_texts[rid] for rid in order) == phrase
        ]
        if not candidates:
            base["warnings"].append(
                "The target phrase does not match any arrangement of the per-region translations; "
                "reorder the slots in the Basil panel to match your typed phrase."
            )
            return base
        suggestion = unit.suggestion or {}
        suggested_order = suggestion.get("region_order") or unit.region_ids
        if tuple(suggested_order) in {tuple(c) for c in candidates}:
            chosen = list(suggested_order)
        else:
            chosen = candidates[0]
        assignments: List[Dict[str, Any]] = []
        for anchor_id, region_id in zip(unit.region_ids, chosen):
            assignments.append({
                "region_id": region_id,
                "text": region_texts[region_id],
                "target_positions": [],
                "method": "manual_slot_arrangement",
                "confidence": 0.9,
                "anchor_id": anchor_id,
            })
        base.update({
            "assignments": assignments,
            "target_region_order": chosen,
            "method": "manual_slot_arrangement",
            "confidence": 0.9,
            "review_required": False,
        })
        return base

    effective = external_lexicon if external_lexicon is not None else _GLOSSARY
    source_locale, target_locale = _locale(manifest.src_lang), _locale(targ_lang)
    lexicon = effective.get((source_locale, target_locale))
    if lexicon is None:
        lexicon = next(
            (terms for (src_key, targ_key), terms in effective.items()
             if str(src_key).lower() == source_locale and str(targ_key).lower() == target_locale),
            None,
        )
    if lexicon is None:
        lexicon = effective.get((_lang(manifest.src_lang), _lang(targ_lang)))
    if lexicon is None and external_lexicon is not None:
        lexicon = effective.get((source_locale.split("-", 1)[0], target_locale.split("-", 1)[0]))
    if not lexicon:
        base["warnings"].append(
            "No locally verified alignment model or glossary is available for this language pair; keep per-region edits or provision a benchmarked provider."
        )
        return base

    target_words = _tokens(phrase)
    target_normal = [_normal(word) for word in target_words]
    region_word_indices: Dict[str, List[int]] = {}
    cursor = 0
    for region_id in unit.region_ids:
        inst = next((item for item in manifest.instances if item.id == region_id), None)
        count = len(_tokens(inst.text if inst else ""))
        region_word_indices[region_id] = list(range(cursor, cursor + count))
        cursor += count
    source_words = _tokens(unit.source_text)
    if len(source_words) != cursor:
        base["warnings"].append("Source unit token registration is inconsistent; review the captured text before assignment.")
        return base

    used: set[int] = set()
    mapped_positions: Dict[int, int] = {}
    for source_index, source_word in enumerate(source_words):
        expected = lexicon.get(_normal(source_word))
        if not expected:
            base["warnings"].append(f"No verified local equivalent for source token '{source_word}'.")
            return base
        found = next((index for index, target_word in enumerate(target_normal)
                      if index not in used and target_word == expected), None)
        if found is None:
            base["warnings"].append(
                f"The target phrase does not contain the verified equivalent '{expected}' for '{source_word}'."
            )
            return base
        used.add(found)
        mapped_positions[source_index] = found
    if len(used) != len(target_words):
        base["warnings"].append("Target phrase contains unaligned words; a model/provider or manual per-region placement is required.")
        return base

    semantic_assignments: List[Dict[str, Any]] = []
    for region_id in unit.region_ids:
        target_positions = sorted(mapped_positions[index] for index in region_word_indices[region_id])
        assigned = _join([target_words[index] for index in target_positions])
        semantic_assignments.append({
            "region_id": region_id,
            "text": assigned,
            "target_positions": target_positions,
            "method": "uploaded_glossary_span_alignment" if external_lexicon is not None else "verified_glossary_span_alignment",
            "confidence": 0.98,
        })
    # Target syntax determines the semantic block sequence; visual source
    # order determines the immutable cube sequence.  Zip the two rather than
    # assigning text back to its source rN.  For Rue des / VIEUX / MURS this
    # explicitly plates r2:Muri into r3's cube and r3:Vecchi into r2's cube.
    target_semantic_order = sorted(semantic_assignments, key=lambda item: min(item["target_positions"]))
    assignments: List[Dict[str, Any]] = []
    for anchor_id, semantic in zip(unit.region_ids, target_semantic_order):
        assignments.append({**semantic, "anchor_id": anchor_id})
    base.update({
        "assignments": assignments,
        "target_region_order": [assignment["region_id"] for assignment in target_semantic_order],
        "method": "uploaded_glossary_span_alignment" if external_lexicon is not None else "verified_glossary_span_alignment",
        "confidence": 0.98,
        "review_required": False,
    })
    return base


def apply_substitution(manifest: TextManifest, plan: Dict[str, Any], targ_lang: Optional[str]) -> SemanticTextUnit:
    """Apply an explicitly requested plan while preserving all geometry."""
    if not plan.get("assignments") or plan.get("review_required"):
        raise ValueError("Basil will not apply an unverified or incomplete substitution plan")
    unit = next((item for item in manifest.semantic_units if item.id == plan.get("unit_id")), None)
    if unit is None:
        raise KeyError(f"semantic unit '{plan.get('unit_id')}' not found")
    by_id = {inst.id: inst for inst in manifest.instances}
    anchor_ids = [assignment.get("anchor_id", assignment["region_id"]) for assignment in plan["assignments"]]
    if len(set(anchor_ids)) != len(anchor_ids):
        raise ValueError("Basil plan assigns more than one semantic block to the same spatial cube")
    original_boxes = {
        anchor_id: asdict(by_id[anchor_id].bounding_box)
        for anchor_id in anchor_ids if anchor_id in by_id
    }
    for assignment in plan["assignments"]:
        anchor_id = assignment.get("anchor_id", assignment["region_id"])
        inst = by_id.get(anchor_id)
        if inst is None or inst.excluded or inst.dnt:
            raise ValueError(f"spatial cube '{anchor_id}' cannot receive a semantic substitution")
        inst.target_text = assignment["text"]
        inst.target_language = targ_lang or inst.target_language
        inst.semantic_assignment = {
            "schema": 1,
            "unit_id": unit.id,
            "source_text": unit.source_text,
            "target_text": plan["target_text"],
            "text": assignment["text"],
            "semantic_region_id": assignment["region_id"],
            "anchor_id": anchor_id,
            "target_positions": assignment["target_positions"],
            "method": plan["method"],
            "confidence": plan["confidence"],
            "geometry_unchanged": True,
        }
    unit.substitution = {
        "schema": 1,
        "applied": True,
        "source_text": unit.source_text,
        "target_text": plan["target_text"],
        "source_region_order": plan["source_region_order"],
        "target_region_order": plan["target_region_order"],
        "spatial_anchor_order": plan.get("spatial_anchor_order", plan["source_region_order"]),
        "assignments": plan["assignments"],
        "method": plan["method"],
        "confidence": plan["confidence"],
        "geometry": original_boxes,
    }
    return unit


def plated_texts(manifest: TextManifest) -> Dict[str, str]:
    """Return the authoritative Basil block→cube text overlay.

    ``InstText.target_text`` remains a convenient persisted projection for
    editors, but Scribe and previews must use this relation as the source of
    truth.  That prevents an old autosave or a legacy manifest projection
    from undoing a verified semantic block→spatial cube placement.
    """
    migrate_legacy_plating(manifest)
    output: Dict[str, str] = {}
    for unit in manifest.semantic_units or []:
        substitution = unit.substitution if isinstance(unit.substitution, dict) else None
        if not substitution or not substitution.get("applied"):
            continue
        for assignment in substitution.get("assignments") or []:
            if not isinstance(assignment, dict):
                continue
            anchor_id = assignment.get("anchor_id", assignment.get("region_id"))
            text = assignment.get("text")
            if isinstance(anchor_id, str) and isinstance(text, str) and text.strip():
                output[anchor_id] = text
    return output


def migrate_legacy_plating(manifest: TextManifest) -> bool:
    """Upgrade pre-cube-map approved plans without changing any bbox.

    Earlier Basil plans persisted target semantic IDs but no ``anchor_id``.
    Their intended target order and source cube order are enough to recover
    the missing relation deterministically.  The migration also repairs the
    editable target projection so legacy Translate tables/tooltips agree with
    Scribe and the canvas.
    """
    changed = False
    by_id = {inst.id: inst for inst in manifest.instances}
    for unit in manifest.semantic_units or []:
        substitution = unit.substitution if isinstance(unit.substitution, dict) else None
        if not substitution or not substitution.get("applied"):
            continue
        assignments = [item for item in substitution.get("assignments") or [] if isinstance(item, dict)]
        if not assignments or all(item.get("anchor_id") for item in assignments):
            continue
        cubes = substitution.get("spatial_anchor_order") or substitution.get("source_region_order") or unit.region_ids
        semantic = sorted(assignments, key=lambda item: min(item.get("target_positions") or [10**9]))
        for cube_id, assignment in zip(cubes, semantic):
            assignment["anchor_id"] = cube_id
            inst = by_id.get(cube_id)
            if inst is not None and isinstance(assignment.get("text"), str):
                inst.target_text = assignment["text"]
                inst.semantic_assignment = {
                    **(inst.semantic_assignment or {}),
                    "schema": 1,
                    "unit_id": unit.id,
                    "source_text": unit.source_text,
                    "target_text": substitution.get("target_text", ""),
                    "text": assignment["text"],
                    "semantic_region_id": assignment.get("region_id"),
                    "anchor_id": cube_id,
                    "target_positions": assignment.get("target_positions", []),
                    "method": substitution.get("method", "verified_glossary_span_alignment"),
                    "confidence": substitution.get("confidence", 0.0),
                    "geometry_unchanged": True,
                }
            changed = True
        substitution["spatial_anchor_order"] = list(cubes)
    return changed
