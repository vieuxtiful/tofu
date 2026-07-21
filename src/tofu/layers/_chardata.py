## 🍢 ToFU — CCJK charset data
## vieuxtiful
"""
standards-derived CCJK character sets, generated at import time from
python's built-in legacy codecs (no giant literals, no external data files).

- GB2312_CHARS:     simplified hanzi (mainland china, singapore) — 6,763 chars
- BIG5_CHARS:       traditional hanzi (taiwan, hong kong, macau) — ~13,000 chars
- JIS_X_0208_KANJI: japanese kanji (JIS X 0208 levels 1+2) — ~6,300 chars
- KS_X_1001_HANJA:  korean hanja (KS X 1001) — 4,888 chars
"""

import unicodedata
from typing import Iterable, Set


def _decode_two_byte(codec: str, leads: Iterable[int], trails: Iterable[int]) -> Set[str]:
    """decode every two-byte code in the given lead/trail ranges, keeping
    single letter-category characters. invalid codes are skipped."""
    chars: Set[str] = set()
    for lead in leads:
        for trail in trails:
            try:
                ch = bytes((lead, trail)).decode(codec)
            except (UnicodeDecodeError, LookupError):
                continue
            if len(ch) == 1 and unicodedata.category(ch).startswith("L"):
                chars.add(ch)
    return chars


# GB2312 hanzi: EUC-CN rows 16-87 (lead 0xB0-0xF7, trail 0xA1-0xFE)
GB2312_CHARS: Set[str] = _decode_two_byte(
    "gb2312", range(0xB0, 0xF8), range(0xA1, 0xFF)
)

# Big5 hanzi: common block 0xA4-0xC6 + less-common block 0xC9-0xF9
# (trail bytes 0x40-0x7E and 0xA1-0xFE)
BIG5_CHARS: Set[str] = _decode_two_byte(
    "big5",
    list(range(0xA4, 0xC7)) + list(range(0xC9, 0xFA)),
    list(range(0x40, 0x7F)) + list(range(0xA1, 0xFF)),
)

# JIS X 0208 kanji: EUC-JP rows 16-84 (lead 0xB0-0xF4, trail 0xA1-0xFE)
JIS_X_0208_KANJI: Set[str] = _decode_two_byte(
    "euc_jp", range(0xB0, 0xF5), range(0xA1, 0xFF)
)

# KS X 1001 hanja: EUC-KR lead 0xCA-0xFD, trail 0xA1-0xFE
KS_X_1001_HANJA: Set[str] = _decode_two_byte(
    "euc_kr", range(0xCA, 0xFE), range(0xA1, 0xFF)
)
