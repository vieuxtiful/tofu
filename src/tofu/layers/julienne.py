## 🍢 Julienne - ToFU character-level script analysis
## vieuxtiful
"""
a julienne cuts one ingredient into individual fine strips; this cuts one
string into its individual characters and asks what writing system each of
them actually belongs to.

Why it has to work this way: a string is not "in a script".  Its CHARACTERS
are, and a great many real strings mix several.  A majority vote over the
characters -- which is what Verify did before this module existed -- reports
Japanese ``東京タワー`` as Han because two of its five characters are
ideographs, and reports Korean ``서울特別市`` as Han because three of its five
are hanja.  Both answers erase the writing system that makes the string
readable.  Unicode is explicit that the Script property is a per-character
property (UAX #24), so the analysis is per-character and the summary is a
SET, never a single winner.

  script identity   Unicode Standard Annex #24, "Unicode Script Property".
                    Block ranges resolve to ISO 15924 four-letter codes.

  context resolution  UAX #24 again: characters whose Script is Common or
                    Inherited take the script of their surroundings rather
                    than forming a script of their own.  The prolonged sound
                    mark U+30FC is the case that matters here -- it is named
                    KATAKANA-HIRAGANA PROLONGED SOUND MARK and carries
                    Script_Extensions {Hira, Kana}, so in ``タワー`` it is
                    Katakana and in ``ラーメン`` likewise, but after hiragana
                    it is hiragana.

  han variant       GB 2312-80 (simplified, PRC/Singapore) against Big5
                    (traditional, Taiwan/HK/Macau), via _chardata.  Unicode
                    unifies the two orthographies into one script (Hani), so
                    no Unicode property distinguishes them and this is the
                    standards-derived way to tell 旧墙街 from 舊牆街.

The variant test is gated on the text being CHINESE.  ``東`` and ``別`` are
absent from GB 2312 and so look "traditional-exclusive", but in ``東京`` they
are Japanese kanji and in ``特別市`` Korean hanja, where the
simplified/traditional axis simply does not apply.  Asking it there would
report every Japanese sign as Traditional Chinese.
"""

from __future__ import annotations

import unicodedata
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Set, Tuple

# ISO 15924 codes used here.  Zyyy is Common (punctuation, digits, spaces):
# real characters that belong to no single script and must never make a
# string count as "mixed".
COMMON = "Zyyy"

# Block ranges -> ISO 15924.  Ordered longest-tail first only where blocks
# would otherwise overlap; CJK Unified is checked after the kana/hangul
# blocks so a kana character can never fall through to Hani.
_BLOCKS: Tuple[Tuple[int, int, str], ...] = (
    (0x3040, 0x309F, "Hira"),          # Hiragana
    (0x30A0, 0x30FF, "Kana"),          # Katakana
    (0x31F0, 0x31FF, "Kana"),          # Katakana phonetic extensions
    (0xFF66, 0xFF9D, "Kana"),          # halfwidth katakana
    (0x1B000, 0x1B0FF, "Kana"),        # kana supplement
    (0xAC00, 0xD7A3, "Hang"),          # Hangul syllables
    (0x1100, 0x11FF, "Hang"),          # Hangul jamo
    (0x3130, 0x318F, "Hang"),          # Hangul compatibility jamo
    (0xA960, 0xA97F, "Hang"),          # Hangul jamo extended-A
    (0xD7B0, 0xD7FF, "Hang"),          # Hangul jamo extended-B
    (0x4E00, 0x9FFF, "Hani"),          # CJK Unified Ideographs
    (0x3400, 0x4DBF, "Hani"),          # CJK Unified Ideographs Extension A
    (0xF900, 0xFAFF, "Hani"),          # CJK Compatibility Ideographs
    (0x20000, 0x2FA1F, "Hani"),        # Extensions B-F + compatibility supp.
    (0x0400, 0x04FF, "Cyrl"),          # Cyrillic
    (0x0500, 0x052F, "Cyrl"),          # Cyrillic supplement
    (0x2DE0, 0x2DFF, "Cyrl"),
    (0xA640, 0xA69F, "Cyrl"),
    (0x0370, 0x03FF, "Grek"),          # Greek and Coptic
    (0x1F00, 0x1FFF, "Grek"),
    (0x0590, 0x05FF, "Hebr"),          # Hebrew
    (0x0600, 0x06FF, "Arab"),          # Arabic
    (0x0750, 0x077F, "Arab"),
    (0x08A0, 0x08FF, "Arab"),
    (0xFB50, 0xFDFF, "Arab"),
    (0xFE70, 0xFEFF, "Arab"),
    (0x0900, 0x097F, "Deva"),          # Devanagari
    (0x0E00, 0x0E7F, "Thai"),          # Thai
    (0x0041, 0x005A, "Latn"),          # Basic Latin letters
    (0x0061, 0x007A, "Latn"),
    (0x00C0, 0x024F, "Latn"),          # Latin-1 supplement + extended A/B
    (0x1E00, 0x1EFF, "Latn"),          # Latin extended additional
    (0x2C60, 0x2C7F, "Latn"),
    (0xA720, 0xA7FF, "Latn"),
)

# Script_Extensions (UAX #24 §2): characters that legitimately belong to
# more than one script and must take their identity from context.  Only the
# entries that actually occur in localisation copy are listed; anything
# absent falls through to the block table.
_SCRIPT_EXTENSIONS: Dict[int, Set[str]] = {
    0x30FC: {"Hira", "Kana"},   # ー prolonged sound mark
    0x30FB: {"Hira", "Kana", "Hani"},  # ・ middle dot
    0x3005: {"Hani"},           # 々 iteration mark
    0x3006: {"Hani"},           # 〆
    0x303B: {"Hani"},           # 〻
    0x3099: {"Hira", "Kana"},   # combining voiced sound mark
    0x309A: {"Hira", "Kana"},
    0x309B: {"Hira", "Kana"},
    0x309C: {"Hira", "Kana"},
    0x309D: {"Hira"},           # ゝ hiragana iteration
    0x309E: {"Hira"},
    0x30FD: {"Kana"},           # ヽ katakana iteration
    0x30FE: {"Kana"},
}

# What each language's declared ISO 15924 code actually admits at the
# character level.  Jpan and Kore are COMPOSITE codes -- ISO 15924 defines
# Jpan as "Japanese (alias for Han + Hiragana + Katakana)" and Kore as
# "Korean (alias for Hangul + Han)" -- so a Japanese string containing both
# kanji and kana is correct, not mixed-script contamination.
_COMPOSITE_SCRIPTS: Dict[str, Set[str]] = {
    "Jpan": {"Hani", "Hira", "Kana"},
    "Kore": {"Hang", "Hani"},
    "Hans": {"Hani"},
    "Hant": {"Hani"},
}

#: languages whose Han orthography is on the simplified/traditional axis.
#: Japanese and Korean use Han characters too, but shinjitai and hanja are
#: not points on that axis and must never be judged against it.
_CHINESE_LANGS = {"zh-cn", "zh-sg", "zh-tw", "zh-hk", "zh-mo", "zh"}


@dataclass
class ScriptAnalysis:
    """Per-character script evidence for one string."""
    text: str
    #: (character, ISO 15924) for every character, Common included
    per_character: List[Tuple[str, str]] = field(default_factory=list)
    #: every script actually present, Common excluded, sorted
    scripts: List[str] = field(default_factory=list)
    #: script of the greatest number of characters, or None for empty/common
    dominant: Optional[str] = None
    #: counts per script, Common excluded
    counts: Dict[str, int] = field(default_factory=dict)
    #: more than one real script present
    is_mixed: bool = False
    direction: str = "ltr"
    #: "Hans" | "Hant" | "undetermined" | None (None when not Chinese Han)
    han_variant: Optional[str] = None
    han_evidence: Dict[str, List[str]] = field(default_factory=dict)

    @property
    def summary(self) -> str:
        """Human-readable script identity, e.g. 'Hani+Kana' or 'Latn'."""
        return "+".join(self.scripts) if self.scripts else "Zyyy"


_RTL_BIDI = {"R", "AL", "AN"}


def _block_script(codepoint: int) -> str:
    for start, end, script in _BLOCKS:
        if start <= codepoint <= end:
            return script
    return COMMON


def _resolve(text: str) -> List[Tuple[str, object]]:
    """First pass: block/Script_Extensions lookup, extensions left as sets."""
    resolved: List[Tuple[str, object]] = []
    for char in text:
        codepoint = ord(char)
        category = unicodedata.category(char)
        # Punctuation, separators, marks and control characters are Common or
        # Inherited: they take surrounding script and never make a string mixed.
        if category[0] in {"P", "Z", "C"} or char.isspace():
            resolved.append((char, COMMON))
            continue
        extensions = _SCRIPT_EXTENSIONS.get(codepoint)
        if extensions is not None:
            resolved.append((char, set(extensions) if len(extensions) > 1
                             else next(iter(extensions))))
            continue
        if category[0] == "N":  # digits belong to no script
            resolved.append((char, COMMON))
            continue
        resolved.append((char, _block_script(codepoint)))
    return resolved


def _apply_context(resolved: List[Tuple[str, object]]) -> List[Tuple[str, str]]:
    """UAX #24 context resolution for multi-script characters.

    A prolonged sound mark takes the script of the run it extends, so the
    nearest PRECEDING resolved character wins; only when nothing precedes
    does the following one decide.  Ties fall back to the alphabetically
    first candidate purely so the result is deterministic.
    """
    out: List[Tuple[str, str]] = []
    for index, (char, script) in enumerate(resolved):
        if not isinstance(script, set):
            out.append((char, script))
            continue
        chosen = None
        for j in range(index - 1, -1, -1):
            candidate = resolved[j][1]
            if isinstance(candidate, str) and candidate in script:
                chosen = candidate
                break
        if chosen is None:
            for j in range(index + 1, len(resolved)):
                candidate = resolved[j][1]
                if isinstance(candidate, str) and candidate in script:
                    chosen = candidate
                    break
        out.append((char, chosen or sorted(script)[0]))
    return out


def _han_variant(text: str) -> Tuple[Optional[str], Dict[str, List[str]]]:
    """Simplified/traditional evidence for CHINESE Han text.

    Unicode unifies both orthographies as Hani, so this reads the national
    charsets instead: a character in GB 2312 but not Big5 can only be a
    simplified form, and vice versa.  Characters in both carry no evidence
    either way, which is why short shared strings like 出口 honestly come
    back "undetermined" rather than guessing.
    """
    from tofu.layers._chardata import BIG5_CHARS, GB2312_CHARS

    simplified = [c for c in text if c in GB2312_CHARS and c not in BIG5_CHARS]
    traditional = [c for c in text if c in BIG5_CHARS and c not in GB2312_CHARS]
    evidence = {"simplified_only": simplified, "traditional_only": traditional}
    if simplified and traditional:
        return "mixed", evidence
    if simplified:
        return "Hans", evidence
    if traditional:
        return "Hant", evidence
    return "undetermined", evidence


def analyse(text: Optional[str], lang: Optional[str] = None) -> ScriptAnalysis:
    """Cut ``text`` into per-character script evidence.

    ``lang`` only gates the Han simplified/traditional test; the script
    identification itself is a property of the characters and never of the
    language someone declared.
    """
    value = text or ""
    analysis = ScriptAnalysis(text=value)
    if not value:
        return analysis

    per_character = _apply_context(_resolve(value))
    analysis.per_character = per_character

    counts: Dict[str, int] = {}
    rtl = 0
    for char, script in per_character:
        if script != COMMON:
            counts[script] = counts.get(script, 0) + 1
        if unicodedata.bidirectional(char) in _RTL_BIDI:
            rtl += 1
    analysis.counts = counts
    analysis.scripts = sorted(counts)
    analysis.is_mixed = len(counts) > 1
    analysis.dominant = max(counts, key=lambda s: (counts[s], s)) if counts else None
    analysis.direction = "rtl" if rtl else "ltr"

    if "Hani" in counts and _is_chinese(lang):
        analysis.han_variant, analysis.han_evidence = _han_variant(value)
    return analysis


def _is_chinese(lang: Optional[str]) -> bool:
    return (lang or "").strip().lower() in _CHINESE_LANGS


def expected_scripts(declared: Optional[str]) -> Set[str]:
    """Character-level scripts admitted by a declared ISO 15924 code.

    Expands the composite codes: Jpan legitimately contains Han, hiragana
    AND katakana, so a Japanese string is not "mixed script" in any sense
    that should be flagged -- it is correctly written Japanese.
    """
    if not declared:
        return set()
    return set(_COMPOSITE_SCRIPTS.get(declared, {declared}))


def validate(text: Optional[str], lang: Optional[str],
             declared_script: Optional[str]) -> Dict[str, object]:
    """Check a string's characters against the script its language declares.

    Returns evidence, never a verdict: the caller decides what an unexpected
    script is worth.  Foreign scripts are split into two findings because
    they mean opposite things.

    Brand names, model numbers and proper nouns legitimately stay in the
    source script -- "TOFU برو" is correctly localised Arabic, 「iPhoneを使う」
    correctly localised Japanese -- so a foreign script is not a defect by
    itself.  What separates the two is whether ANY of the expected script is
    there: a region carrying none of it was never translated, while a region
    carrying it plus a Latin run has an embedded token.

    Character SHARE was tried first and is the wrong measure: Latin spends
    far more characters per unit of meaning than Han or Arabic, so "iPhone
    を使う" is 67% Latin by character and entirely ordinary Japanese by
    content.  Presence is density-independent, which is what lets one rule
    serve every script pair.
    """
    analysis = analyse(text, lang)
    allowed = expected_scripts(declared_script)
    foreign = sorted(s for s in analysis.scripts if allowed and s not in allowed)
    scripted = sum(analysis.counts.values())
    foreign_count = sum(analysis.counts.get(s, 0) for s in foreign)
    share = (foreign_count / scripted) if scripted else 0.0
    expected_present = any(analysis.counts.get(s) for s in allowed)
    # foreign characters AND none of the language's own script
    dominant_foreign = bool(foreign) and not expected_present
    result: Dict[str, object] = {
        "scripts": analysis.scripts,
        "summary": analysis.summary,
        "dominant": analysis.dominant,
        "is_mixed": analysis.is_mixed,
        "counts": analysis.counts,
        "declared_script": declared_script,
        "expected_scripts": sorted(allowed),
        #: none of the declared script is present: the region reads as
        #: never having been translated
        "unexpected_scripts": foreign if dominant_foreign else [],
        #: foreign script alongside the declared one: a brand, a model
        #: number, a proper noun -- reported, never scored against
        "embedded_scripts": [] if dominant_foreign else foreign,
        "expected_script_present": expected_present,
        "foreign_share": round(share, 4),
        "direction": analysis.direction,
        "status": "pass" if not dominant_foreign else "review",
    }
    if analysis.han_variant is not None:
        result["han_variant"] = analysis.han_variant
        result["han_evidence"] = {
            key: "".join(chars) for key, chars in analysis.han_evidence.items()
        }
        # zh-cn declares Hans; traditional-exclusive characters in it mean the
        # wrong orthography reached the render, which no glyph-coverage check
        # can see because both orthographies are the same Unicode script.
        if declared_script in {"Hans", "Hant"} and analysis.han_variant in {"Hans", "Hant"} \
                and analysis.han_variant != declared_script:
            result["status"] = "review"
            result["han_variant_mismatch"] = True
    return result
