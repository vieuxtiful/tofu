## 🍢 make_guided_fixtures — fixtures the Guided corpus was missing
## vieuxtiful
"""Generate the strata the derived Guided corpus could not fill.

Reading the existing fixtures by CONTENT rather than by name exposed three
gaps that the file names had hidden:

  * `rtl-sign` is the Latin word "Welcome" -- a sign to be localized INTO an
    RTL target, not RTL text.  The corpus had **no RTL/bidi coverage at all**.
  * `cjk-horizontal` is annotated "City Library" / "Open today".
  * repeated text -- the duplicate-occurrence case where set-based matching
    quietly fails -- appeared in only six Blocks.

Everything here is rendered, so the bounding boxes and strings are EXACT
rather than hand-annotated.  Arabic and Hebrew are shaped and reordered for
display with the same libraries Scribe uses, while the ground truth records
the LOGICAL string -- which is precisely the round trip a Guided match has to
survive.

Deterministic: fixed seed, no timestamps, stable ordering.

usage: .venv/Scripts/python scripts/make_guided_fixtures.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "tests" / "fixtures"
CANVAS = (1000, 520)

_FONTS = {
    "regular": ("arial.ttf", "segoeui.ttf", "DejaVuSans.ttf"),
    "bold": ("arialbd.ttf", "segoeuib.ttf", "DejaVuSans-Bold.ttf"),
    "arabic": ("tahoma.ttf", "arial.ttf", "segoeui.ttf"),
    "hebrew": ("arial.ttf", "tahoma.ttf", "DejaVuSans.ttf"),
    "cjk": ("msgothic.ttc", "meiryo.ttc", "YuGothM.ttc"),
}


def _font(kind: str, size: int):
    for candidate in _FONTS[kind]:
        try:
            return ImageFont.truetype(candidate, size)
        except Exception:
            continue
    return ImageFont.load_default()


def _shape_rtl(text: str) -> str:
    """Display form for Arabic/Hebrew, via the same seam Scribe renders with.

    The ground truth keeps the logical string: a detector reads glyphs off
    the image and a matcher has to get back to what the user typed, so
    recording the visual form would quietly grade the wrong thing.
    """
    try:
        import arabic_reshaper
        from bidi.algorithm import get_display
        return get_display(arabic_reshaper.reshape(text))
    except Exception:
        try:
            from bidi.algorithm import get_display
            return get_display(text)
        except Exception:
            return text


def _draw_text(draw, position, text, font, fill, *, logical=None, rtl=False):
    """Draw, and return the ground-truth record for what was drawn."""
    display = _shape_rtl(text) if rtl else text
    draw.text(position, display, font=font, fill=fill)
    left, top, right, bottom = draw.textbbox(position, display, font=font)
    return {
        "bbox": [int(left), int(top), int(right - left), int(bottom - top)],
        "text": logical if logical is not None else text,
    }


def _save(name: str, image: Image.Image, regions: list) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    image.save(OUT / f"{name}.png")
    (OUT / f"{name}.gt.json").write_text(
        json.dumps({"regions": regions}, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"  {name}.png  ({len(regions)} region(s))")


# --- RTL / bidi -----------------------------------------------------------

def arabic_street() -> None:
    """Arabic street signage with Western digits — the bidi run case."""
    image = Image.new("RGB", CANVAS, (240, 236, 226))
    draw = ImageDraw.Draw(image)
    draw.rectangle([80, 90, 920, 300], fill=(24, 88, 60))
    regions = [
        _draw_text(draw, (620, 120), "شارع النيل", _font("arabic", 58), (255, 255, 255), rtl=True),
        _draw_text(draw, (620, 210), "القاهرة", _font("arabic", 40), (232, 232, 220), rtl=True),
        _draw_text(draw, (140, 210), "12", _font("bold", 40), (232, 232, 220)),
        _draw_text(draw, (140, 350), "مفتوح", _font("arabic", 36), (30, 30, 30), rtl=True),
        _draw_text(draw, (420, 350), "مفتوح", _font("arabic", 36), (30, 30, 30), rtl=True),
    ]
    _save("arabic-street", image, regions)


def hebrew_notice() -> None:
    """Hebrew notice board, mixed with a Latin brand token."""
    image = Image.new("RGB", CANVAS, (246, 244, 238))
    draw = ImageDraw.Draw(image)
    draw.rectangle([100, 80, 900, 260], fill=(28, 52, 110))
    regions = [
        _draw_text(draw, (600, 110), "רחוב הרצל", _font("hebrew", 54), (255, 255, 255), rtl=True),
        _draw_text(draw, (600, 190), "תל אביב", _font("hebrew", 36), (226, 226, 236), rtl=True),
        _draw_text(draw, (150, 320), "TOFU", _font("bold", 44), (28, 52, 110)),
        _draw_text(draw, (150, 400), "סגור", _font("hebrew", 34), (40, 40, 40), rtl=True),
    ]
    _save("hebrew-notice", image, regions)


def arabic_bilingual_panel() -> None:
    """Arabic over Latin — the same panel in two scripts, one above the other."""
    image = Image.new("RGB", CANVAS, (238, 238, 234))
    draw = ImageDraw.Draw(image)
    draw.rectangle([60, 60, 940, 460], fill=(250, 250, 248), outline=(120, 120, 120), width=3)
    regions = [
        _draw_text(draw, (600, 100), "مطار دولي", _font("arabic", 52), (20, 20, 20), rtl=True),
        _draw_text(draw, (120, 190), "International Airport", _font("bold", 40), (20, 20, 20)),
        _draw_text(draw, (620, 280), "بوابة", _font("arabic", 40), (60, 60, 60), rtl=True),
        _draw_text(draw, (120, 300), "Gate", _font("regular", 38), (60, 60, 60)),
        _draw_text(draw, (120, 380), "24", _font("bold", 44), (20, 20, 20)),
    ]
    _save("arabic-bilingual-panel", image, regions)


# --- Japanese horizontal --------------------------------------------------

def japanese_shopfront() -> None:
    image = Image.new("RGB", CANVAS, (244, 240, 232))
    draw = ImageDraw.Draw(image)
    draw.rectangle([70, 80, 930, 240], fill=(190, 40, 40))
    regions = [
        _draw_text(draw, (130, 110), "東京古書店", _font("cjk", 62), (255, 250, 240)),
        _draw_text(draw, (130, 300), "営業中", _font("cjk", 44), (30, 30, 30)),
        _draw_text(draw, (480, 300), "定休日", _font("cjk", 44), (30, 30, 30)),
        _draw_text(draw, (130, 400), "10:00", _font("bold", 36), (40, 40, 40)),
        _draw_text(draw, (330, 400), "19:00", _font("bold", 36), (40, 40, 40)),
    ]
    _save("japanese-shopfront", image, regions)


def japanese_poster() -> None:
    """Japanese with Latin and a decimal date — the mixed-run poster case."""
    image = Image.new("RGB", CANVAS, (250, 248, 240))
    draw = ImageDraw.Draw(image)
    regions = [
        _draw_text(draw, (100, 90), "5.2", _font("bold", 56), (200, 30, 30)),
        _draw_text(draw, (230, 100), "FRI", _font("bold", 44), (30, 30, 30)),
        _draw_text(draw, (100, 210), "世界を救え", _font("cjk", 54), (20, 20, 20)),
        _draw_text(draw, (100, 320), "サンダーボルツ", _font("cjk", 46), (20, 20, 20)),
        _draw_text(draw, (100, 420), "ACME", _font("bold", 32), (90, 90, 90)),
    ]
    _save("japanese-poster", image, regions)


# --- repeated occurrences -------------------------------------------------

def repeated_exits() -> None:
    """The same word four times. Set-based matching reports one and passes."""
    image = Image.new("RGB", CANVAS, (236, 236, 236))
    draw = ImageDraw.Draw(image)
    font = _font("bold", 40)
    regions = []
    for index, (x, y) in enumerate([(90, 90), (520, 90), (90, 330), (520, 330)]):
        draw.rectangle([x - 20, y - 15, x + 220, y + 65], fill=(20, 110, 60))
        regions.append(_draw_text(draw, (x, y), "SORTIE", font, (255, 255, 255)))
    regions.append(_draw_text(draw, (420, 220), "12", _font("bold", 36), (30, 30, 30)))
    _save("repeated-exits", image, regions)


def repeated_platform() -> None:
    image = Image.new("RGB", CANVAS, (242, 240, 236))
    draw = ImageDraw.Draw(image)
    regions = []
    for y in (90, 200, 310):
        regions.append(_draw_text(draw, (110, y), "PLATFORM", _font("bold", 38), (25, 25, 25)))
        regions.append(_draw_text(draw, (520, y), "PLATFORM", _font("bold", 38), (25, 25, 25)))
    _save("repeated-platform", image, regions)


# --- Latin volume ---------------------------------------------------------

def transit_board() -> None:
    image = Image.new("RGB", CANVAS, (18, 18, 22))
    draw = ImageDraw.Draw(image)
    rows = [
        ("PARIS", "08:15"), ("LYON", "09:40"),
        ("MARSEILLE", "11:05"), ("BORDEAUX", "13:20"),
    ]
    regions = []
    for index, (place, time_text) in enumerate(rows):
        y = 90 + index * 90
        regions.append(_draw_text(draw, (110, y), place, _font("bold", 40), (250, 200, 40)))
        regions.append(_draw_text(draw, (640, y), time_text, _font("bold", 40), (250, 250, 250)))
    _save("transit-board", image, regions)


def museum_labels() -> None:
    image = Image.new("RGB", CANVAS, (248, 246, 240))
    draw = ImageDraw.Draw(image)
    entries = [
        "Musee des Arts", "Salle 4", "Ouvert", "Entree libre", "Fermeture 18h",
    ]
    regions = []
    for index, text in enumerate(entries):
        y = 80 + index * 78
        regions.append(_draw_text(draw, (120, y), text, _font("regular", 36), (30, 30, 30)))
    _save("museum-labels", image, regions)


def arabic_market() -> None:
    """Dense Arabic stall signage with repeated words and Western prices."""
    image = Image.new("RGB", CANVAS, (243, 238, 226))
    draw = ImageDraw.Draw(image)
    font = _font("arabic", 40)
    regions = []
    for index, y in enumerate((90, 190, 290, 390)):
        regions.append(_draw_text(draw, (640, y), "سوق", font, (25, 25, 25), rtl=True))
        regions.append(_draw_text(draw, (200, y), str(10 + index * 5), _font("bold", 36), (140, 30, 30)))
    _save("arabic-market", image, regions)


def japanese_station() -> None:
    """Station board: kanji place names with repeated platform wording."""
    image = Image.new("RGB", CANVAS, (26, 28, 34))
    draw = ImageDraw.Draw(image)
    regions = []
    places = ["新宿", "渋谷", "品川"]
    for index, place in enumerate(places):
        y = 90 + index * 110
        regions.append(_draw_text(draw, (120, y), place, _font("cjk", 52), (250, 250, 240)))
        regions.append(_draw_text(draw, (420, y), "番線", _font("cjk", 40), (200, 220, 250)))
        regions.append(_draw_text(draw, (700, y), f"{index + 1}", _font("bold", 44), (250, 210, 60)))
    _save("japanese-station", image, regions)


def parking_levels() -> None:
    """Repeated level markers — many occurrences of very few strings."""
    image = Image.new("RGB", CANVAS, (235, 235, 232))
    draw = ImageDraw.Draw(image)
    regions = []
    for index, y in enumerate((80, 180, 280, 380)):
        regions.append(_draw_text(draw, (120, y), "NIVEAU", _font("bold", 38), (30, 30, 30)))
        regions.append(_draw_text(draw, (400, y), str(index + 1), _font("bold", 38), (30, 30, 30)))
        regions.append(_draw_text(draw, (600, y), "COMPLET", _font("bold", 34), (170, 30, 30)))
    _save("parking-levels", image, regions)


def cjk_vertical_menu() -> None:
    """Stacked vertical Japanese columns — one character per row, shared x.

    The stratum this fills was previously carried by two dense street photos
    that a geometry heuristic mislabelled as vertical; it needs signage that
    genuinely reads top to bottom.
    """
    image = Image.new("RGB", CANVAS, (242, 236, 224))
    draw = ImageDraw.Draw(image)
    font = _font("cjk", 54)
    regions = []
    columns = [("居酒屋", 180), ("定食屋", 420), ("喫茶店", 660)]
    for text, x in columns:
        y = 90
        for character in text:
            draw.text((x, y), character, font=font, fill=(30, 28, 26))
            y += 74
        left, top = x, 90
        right, bottom = x + 58, y - 14
        regions.append({"bbox": [int(left), int(top), int(right - left), int(bottom - top)],
                        "text": text})
    _save("cjk-vertical-menu", image, regions)


def cjk_vertical_banner() -> None:
    """Vertical banner columns beside a horizontal Latin caption."""
    image = Image.new("RGB", CANVAS, (28, 30, 36))
    draw = ImageDraw.Draw(image)
    font = _font("cjk", 48)
    regions = []
    for text, x in (("新春大売出", 200), ("本日開店", 480)):
        y = 70
        for character in text:
            draw.text((x, y), character, font=font, fill=(250, 244, 230))
            y += 66
        regions.append({"bbox": [int(x), 70, 52, int(y - 70 - 14)], "text": text})
    regions.append(_draw_text(draw, (640, 200), "SALE", _font("bold", 44), (250, 210, 60)))
    regions.append(_draw_text(draw, (640, 300), "2026", _font("bold", 36), (200, 200, 210)))
    _save("cjk-vertical-banner", image, regions)


FIXTURES = (
    arabic_street, hebrew_notice, arabic_bilingual_panel,
    japanese_shopfront, japanese_poster,
    repeated_exits, repeated_platform,
    transit_board, museum_labels,
    arabic_market, japanese_station, parking_levels,
    cjk_vertical_menu, cjk_vertical_banner,
)


def main() -> int:
    print(f"writing guided fixtures into {OUT.relative_to(ROOT)}")
    for fixture in FIXTURES:
        fixture()
    print(f"\n{len(FIXTURES)} fixture(s) written")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
