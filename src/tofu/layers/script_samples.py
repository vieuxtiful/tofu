## 🍢 ToFU — exhaustive per-script sample sets
## vieuxtiful
"""
exhaustive character sets per ISO 15924 script code, used to score real
font cmap coverage. CCJK codes (Hans/Hant/Jpan/Kore) are preserved as
distinct, standards-derived sets — never collapsed into Hani — so that
e.g. a simplified-only font correctly fails Hant.

unassigned codepoints are filtered so range-based sets never penalize
fonts for characters that do not exist.
"""

import unicodedata
from typing import Dict, Set

from tofu.layers._chardata import (
    GB2312_CHARS,
    BIG5_CHARS,
    JIS_X_0208_KANJI,
    KS_X_1001_HANJA,
)


def _r(*ranges: tuple) -> Set[str]:
    """build a character set from inclusive codepoint ranges,
    dropping unassigned codepoints (category 'Cn')."""
    return {
        chr(c)
        for lo, hi in ranges
        for c in range(lo, hi + 1)
        if unicodedata.category(chr(c)) != "Cn"
    }


# kana + hangul building blocks
HIRAGANA: Set[str] = _r((0x3041, 0x3096), (0x309D, 0x309F))
KATAKANA: Set[str] = _r((0x30A1, 0x30FA), (0x30FC, 0x30FF), (0xFF66, 0xFF9D))
HANGUL_SYLLABLES: Set[str] = _r((0xAC00, 0xD7A3))  # all 11,172 precomposed
HANGUL_JAMO: Set[str] = _r((0x1100, 0x1112), (0x1161, 0x1175), (0x11A8, 0x11C2))
CJK_UNIFIED: Set[str] = _r((0x4E00, 0x9FFF))  # URO — generic Hani fallback only

SCRIPT_SAMPLES: Dict[str, Set[str]] = {
    # Latn includes Latin-1, Extended-A/B (Ơ/Ư at 0x01A0-0x01B0), and Latin
    # Extended Additional (0x1E00-0x1EFF) — the block carrying the full
    # Vietnamese diacritic set (ạ ả ấ ầ ẩ ẫ ậ ... ỹ).
    "Latn": _r(
        (0x0041, 0x005A), (0x0061, 0x007A), (0x00C0, 0x00FF),
        (0x0100, 0x017F), (0x0180, 0x024F), (0x1E00, 0x1EFF),
    ),
    "Cyrl": _r((0x0400, 0x04FF), (0x0500, 0x052F)),
    "Grek": _r((0x0370, 0x03FF), (0x1F00, 0x1FFE)),
    "Arab": _r((0x0600, 0x06FF), (0x0750, 0x077F), (0xFB50, 0xFDFF), (0xFE70, 0xFEFF)),
    "Hebr": _r((0x0590, 0x05FF), (0xFB1D, 0xFB4F)),
    "Deva": _r((0x0900, 0x097F), (0xA8E0, 0xA8FF)),
    "Thai": _r((0x0E00, 0x0E7F)),
    "Beng": _r((0x0980, 0x09FF)),
    "Guru": _r((0x0A00, 0x0A7F)),
    "Gujr": _r((0x0A80, 0x0AFF)),
    "Taml": _r((0x0B80, 0x0BFF)),
    "Telu": _r((0x0C00, 0x0C7F)),
    "Knda": _r((0x0C80, 0x0CFF)),
    "Mlym": _r((0x0D00, 0x0D7F)),
    "Sinh": _r((0x0D80, 0x0DFF)),
    "Mymr": _r((0x1000, 0x109F)),
    "Khmr": _r((0x1780, 0x17FF)),
    "Laoo": _r((0x0E80, 0x0EFF)),
    "Ethi": _r((0x1200, 0x137F)),
    "Armn": _r((0x0530, 0x058F)),
    "Geor": _r((0x10A0, 0x10FF)),

    # CCJK — distinct, standards-derived (never collapsed into Hani)
    "Hans": GB2312_CHARS,                              # mainland china, singapore
    "Hant": BIG5_CHARS,                                # taiwan, hong kong, macau
    "Jpan": HIRAGANA | KATAKANA | JIS_X_0208_KANJI,    # kana + JIS X 0208 kanji
    "Kore": HANGUL_SYLLABLES | HANGUL_JAMO,            # hangul (hanja optional tier)
    "Hani": CJK_UNIFIED,                               # generic fallback only
    "Hang": HANGUL_SYLLABLES,
    "Hira": HIRAGANA,
    "Kana": KATAKANA,
}

# optional strict tiers (not required for FULL classification)
SCRIPT_OPTIONAL_TIERS: Dict[str, Set[str]] = {
    "Kore": KS_X_1001_HANJA,
}

# language-level hard requirements beyond the script base set.
# these are non-negotiable: a font missing any of them cannot be FULL
# for that language, regardless of overall script coverage.
_VI_UPPER = (
    "ÀÁẢÃẠĂẰẮẲẴẶÂẦẤẨẪẬÐĐ"
    "ÈÉẺẼẸÊỀẾỂỄỆ"
    "ÌÍỈĨỊ"
    "ÒÓỎÕỌÔỒỐỔỖỘƠỜỚỞỠỢ"
    "ÙÚỦŨỤƯỪỨỬỮỰ"
    "ỲÝỶỸỴ"
)
LANG_REQUIRED: Dict[str, Set[str]] = {
    "vi": set(_VI_UPPER) | set(_VI_UPPER.lower()),
    "tr": set("ĞğİıŞşÇçÖöÜü"),
    "pl": set("ĄąĆćĘęŁłŃńÓóŚśŹźŻż"),
    "de": set("ÄäÖöÜüßẞ"),
    "fr": set("ÀàÂâÆæÇçÉéÈèÊêËëÎîÏïÔôŒœÙùÛûÜüŸÿ"),
    "es": set("ÁáÉéÍíÑñÓóÚúÜü¿¡"),
    "pt": set("ÁáÂâÃãÀàÇçÉéÊêÍíÓóÔôÕõÚú"),
    "cs": set("ÁáČčĎďÉéĚěÍíŇňÓóŘřŠšŤťÚúŮůÝýŽž"),
    "hu": set("ÁáÉéÍíÓóÖöŐőÚúÜüŰű"),
    "ro": set("ĂăÂâÎîȘșȚț"),
    "is": set("ÁáÐðÉéÍíÓóÚúÝýÞþÆæÖö"),
}
