## 🍢 make_fixtures — deterministic eval fixture generator
## vieuxtiful
"""
Generates the synthetic half of the Phase-0 fixture set into
tests/fixtures/: images with EXACT ground truth (we rendered the text,
so bboxes and strings are known, not hand-annotated).

Each fixture writes {name}.png + {name}.gt.json in the format
eval_detect.py consumes: {"regions": [{"bbox": [x,y,w,h], "text": ...}]}.

Fixture matrix (one failure mode each):
  flat-sign        white text on uniform panels     → cleanse: flat fill
  gradient-banner  text over a linear gradient      → cleanse: gradient fill
  textured-wall    text over procedural noise       → cleanse: content-aware
  stylized-italic  bold / italic / stroked text     → cicerone: typography
  serif-vs-sans    same strings, serif and sans     → font_matching: serif
                                                      discrimination
  expansion-en     long single-line region          → scribe: wrap (de ~1.35x)
  cjk-vertical     stacked vertical Japanese column → cicerone: column merge,
                                                      scribe: vertical render

The two REAL fixtures (images/gemini-street.png, japan-street.jpeg) are
referenced by the baseline runs directly; only synthetic ones are
generated here. Deterministic: fixed seed, no timestamps.

usage: .venv/Scripts/python scripts/make_fixtures.py
"""

import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "tests" / "fixtures"

SEED = 20260718
CANVAS = (960, 640)
BG = (226, 228, 232)

# windows font chain; every entry is optional so the generator degrades
_FONTS = {
    "regular": ("arial.ttf", "segoeui.ttf", "DejaVuSans.ttf"),
    "bold": ("arialbd.ttf", "segoeuib.ttf", "DejaVuSans-Bold.ttf"),
    "italic": ("ariali.ttf", "segoeuii.ttf", "DejaVuSans-Oblique.ttf"),
    "cjk": ("msgothic.ttc", "meiryo.ttc", "YuGothM.ttc", "malgun.ttf"),
    # every other Latin fixture here is sans-serif, which left the font
    # matcher's serif/sans discrimination with nothing to be tested against
    "serif": ("times.ttf", "georgia.ttf", "DejaVuSerif.ttf"),
    "serif-bold": ("timesbd.ttf", "georgiab.ttf", "DejaVuSerif-Bold.ttf"),
}


def _font(kind: str, size: int):
    for cand in _FONTS[kind]:
        try:
            return ImageFont.truetype(cand, size)
        except Exception:
            continue
    return ImageFont.load_default()


def _font_file(kind: str) -> str | None:
    """which face actually got used — recorded as ground truth."""
    for cand in _FONTS[kind]:
        try:
            ImageFont.truetype(cand, 12)
            return cand
        except Exception:
            continue
    return None


def _text_bbox(draw, pos, text, font):
    l, t, r, b = draw.textbbox(pos, text, font=font)
    return [int(l), int(t), int(r - l), int(b - t)]


def _save(name: str, img: Image.Image, regions: list) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    img.save(OUT / f"{name}.png")
    (OUT / f"{name}.gt.json").write_text(
        json.dumps({"regions": regions}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"  {name}.png  ({len(regions)} region(s))")


def flat_sign() -> None:
    """road-sign style: uniform color panels, high-contrast text."""
    img = Image.new("RGB", CANVAS, BG)
    draw = ImageDraw.Draw(img)
    regions = []
    # green highway panel
    draw.rounded_rectangle([80, 80, 620, 240], radius=12, fill=(21, 101, 52))
    for text, y, size in [("MAIN STREET", 110, 52), ("EXIT 25", 180, 40)]:
        font = _font("bold", size)
        pos = (120, y)
        draw.text(pos, text, font=font, fill=(255, 255, 255))
        regions.append({"bbox": _text_bbox(draw, pos, text, font), "text": text})
    # red stop-style panel
    draw.rounded_rectangle([660, 100, 900, 200], radius=10, fill=(155, 28, 28))
    font = _font("bold", 44)
    pos = (700, 125)
    draw.text(pos, "STOP", font=font, fill=(255, 255, 255))
    regions.append({"bbox": _text_bbox(draw, pos, "STOP", font), "text": "STOP"})
    # small info plate
    draw.rectangle([80, 420, 480, 500], fill=(38, 38, 42))
    font = _font("regular", 30)
    pos = (100, 442)
    draw.text(pos, "Open 9am to 5pm", font=font, fill=(240, 240, 240))
    regions.append({"bbox": _text_bbox(draw, pos, "Open 9am to 5pm", font),
                    "text": "Open 9am to 5pm"})
    _save("flat-sign", img, regions)


def gradient_banner() -> None:
    """text over a smooth linear gradient (cleanse must reconstruct it)."""
    w, h = CANVAS
    x = np.linspace(0, 1, w)
    grad = np.zeros((h, w, 3), dtype=np.uint8)
    grad[..., 0] = (40 + 140 * x)[None, :]
    grad[..., 1] = (70 + 90 * x)[None, :]
    grad[..., 2] = (140 - 60 * x)[None, :]
    img = Image.fromarray(grad)
    draw = ImageDraw.Draw(img)
    regions = []
    for text, pos, size, fill in [
        ("SUMMER SALE", (140, 180), 64, (255, 255, 255)),
        ("fifty percent off", (200, 300), 40, (250, 230, 140)),
    ]:
        font = _font("bold" if size > 50 else "regular", size)
        draw.text(pos, text, font=font, fill=fill)
        regions.append({"bbox": _text_bbox(draw, pos, text, font), "text": text})
    _save("gradient-banner", img, regions)


def textured_wall() -> None:
    """text over correlated noise (brick-ish): the content-aware case."""
    rng = np.random.default_rng(SEED)
    w, h = CANVAS
    noise = rng.normal(0, 1, (h // 4, w // 4, 3))
    # upsample for spatial correlation, then add horizontal brick bands
    tex = np.asarray(Image.fromarray(
        ((noise - noise.min()) / (np.ptp(noise) + 1e-9) * 70 + 110).astype(np.uint8)
    ).resize((w, h), Image.BICUBIC)).astype(np.int16)
    for row in range(0, h, 64):
        tex[row:row + 3, :] -= 45  # mortar lines
    tex[..., 0] = np.clip(tex[..., 0] + 35, 0, 255)  # brick-red bias
    tex[..., 2] = np.clip(tex[..., 2] - 25, 0, 255)
    img = Image.fromarray(np.clip(tex, 0, 255).astype(np.uint8))
    draw = ImageDraw.Draw(img)
    regions = []
    for text, pos, size in [("BAKERY", (280, 140), 72), ("est. 1962", (360, 280), 36)]:
        font = _font("bold" if size > 50 else "regular", size)
        draw.text(pos, text, font=font, fill=(245, 240, 228),
                  stroke_width=2, stroke_fill=(40, 30, 25))
        regions.append({"bbox": _text_bbox(draw, pos, text, font), "text": text,
                        "style": {"color": "#f5f0e4", "stroke_color": "#281e19",
                                  "stroke_width": 2}})
    _save("textured-wall", img, regions)


def stylized_italic() -> None:
    """typography ground truth: known weight/slant/stroke per region."""
    img = Image.new("RGB", CANVAS, (245, 244, 240))
    draw = ImageDraw.Draw(img)
    regions = []
    cases = [
        ("Regular Weight", (100, 90), 44, "regular", None),
        ("Bold Statement", (100, 190), 44, "bold", None),
        ("Italic Emphasis", (100, 290), 44, "italic", None),
        ("Stroked Display", (100, 390), 48, "bold", (30, 60, 160)),
    ]
    for text, pos, size, kind, stroke in cases:
        font = _font(kind, size)
        kw = {"stroke_width": 3, "stroke_fill": stroke} if stroke else {}
        draw.text(pos, text, font=font, fill=(30, 30, 34), **kw)
        regions.append({
            "bbox": _text_bbox(draw, pos, text, font), "text": text,
            "style": {"weight": "bold" if kind == "bold" else "regular",
                      "italic": kind == "italic", "stroked": stroke is not None,
                      "color": "#1e1e22",
                      "stroke_color": "#1e3ca0" if stroke else None,
                      "stroke_width": 3 if stroke else None},
        })
    _save("stylized-italic", img, regions)


def serif_vs_sans() -> None:
    """font-matching ground truth: the SAME strings in a serif and a sans.

    The matcher's job is to name the face a sign was set in, and until this
    fixture existed nothing in tests/fixtures could tell whether it could
    make the most basic typographic distinction there is — every other
    Latin fixture here renders through the same sans-serif chain.

    Both a mixed-case and an all-capital string, because the two carry very
    different evidence: lowercase supplies bowls, terminals and crossbars,
    while capitals are nearly all stems and diagonals.  Measured on the
    la-rue-sans-nom plaque, silhouette overlap alone ranks the correct
    serif 2nd of 206 installed families on a lowercase line and 192nd on an
    all-capital one from the very same sign.
    """
    img = Image.new("RGB", CANVAS, (247, 246, 243))
    draw = ImageDraw.Draw(img)
    regions = []
    cases = [
        ("La rue", (90, 80), 64, "serif"),
        ("SANS-NOM", (90, 190), 64, "serif"),
        ("La rue", (500, 80), 64, "regular"),
        ("SANS-NOM", (500, 190), 64, "regular"),
        ("Handgloves", (90, 320), 56, "serif-bold"),
        ("Handgloves", (500, 320), 56, "bold"),
    ]
    for text, pos, size, kind in cases:
        font = _font(kind, size)
        draw.text(pos, text, font=font, fill=(24, 26, 32))
        regions.append({
            "bbox": _text_bbox(draw, pos, text, font), "text": text,
            "style": {
                "serif": kind.startswith("serif"),
                "weight": "bold" if kind.endswith("bold") else "regular",
                "font_file": _font_file(kind),
            },
        })
    _save("serif-vs-sans", img, regions)


def expansion_en() -> None:
    """long single-line english on a tight panel: german (~1.35x) must
    wrap, not shrink — the scribe line-wrapping acceptance fixture."""
    img = Image.new("RGB", CANVAS, BG)
    draw = ImageDraw.Draw(img)
    regions = []
    draw.rectangle([60, 240, 900, 330], fill=(28, 46, 88))
    text = "Please keep this area clean"
    font = _font("regular", 42)
    pos = (100, 262)
    draw.text(pos, text, font=font, fill=(255, 255, 255))
    regions.append({"bbox": _text_bbox(draw, pos, text, font), "text": text})
    _save("expansion-en", img, regions)


def cjk_vertical() -> None:
    """stacked vertical japanese column + horizontal line: column-merge
    and vertical-render fixture."""
    img = Image.new("RGB", CANVAS, (238, 234, 228))
    draw = ImageDraw.Draw(img)
    regions = []
    # vertical column: one kanji/kana per row, shared x — like signage
    col_text = "居酒屋"
    font = _font("cjk", 64)
    x, y = 160, 100
    ys = []
    for ch in col_text:
        draw.rectangle([x - 12, y - 6, x + 76, y + 70], fill=(180, 40, 40))
        draw.text((x, y), ch, font=font, fill=(255, 250, 240))
        ys.append(y)
        y += 90
    # ground truth: ONE region for the whole column (what merge should give)
    regions.append({
        "bbox": [x - 12, ys[0] - 6, 88, (ys[-1] + 76) - (ys[0] - 6)],
        "text": col_text, "vertical": True,
    })
    # horizontal japanese line for contrast
    htext = "ようこそ"
    hfont = _font("cjk", 48)
    hpos = (420, 260)
    draw.text(hpos, htext, font=hfont, fill=(30, 30, 34))
    regions.append({"bbox": _text_bbox(draw, hpos, htext, hfont), "text": htext})
    _save("cjk-vertical", img, regions)


def dense_layout() -> None:
    """Four neighboring text blocks with deliberately narrow gutters."""
    img = Image.new("RGB", CANVAS, (236, 238, 242))
    draw = ImageDraw.Draw(img)
    regions = []
    cards = [
        ("NEWS", (70, 70, 430, 210), 50),
        ("WEATHER", (530, 70, 930, 210), 42),
        ("SPORT", (70, 260, 430, 400), 48),
        ("CULTURE", (530, 260, 930, 400), 44),
    ]
    for text, panel, size in cards:
        draw.rounded_rectangle(panel, radius=12, fill=(32, 56, 96))
        font = _font("bold", size)
        text_pos = (panel[0] + 28, panel[1] + 40)
        draw.text(text_pos, text, font=font, fill=(250, 252, 255))
        regions.append({"bbox": _text_bbox(draw, text_pos, text, font), "text": text})
    _save("dense-layout", img, regions)


def product_label() -> None:
    """Label hierarchy with a product name, quantity, and warning."""
    img = Image.new("RGB", CANVAS, (245, 238, 213))
    draw = ImageDraw.Draw(img)
    draw.rounded_rectangle([180, 55, 820, 520], radius=30, fill=(250, 247, 234),
                           outline=(90, 58, 32), width=5)
    regions = []
    rows = [
        ("ALMOND TONIC", (270, 120), 58, "bold"),
        ("500 ml", (420, 250), 40, "regular"),
        ("CONTAINS NUTS", (330, 385), 34, "bold"),
    ]
    for text, pos, size, kind in rows:
        font = _font(kind, size)
        draw.text(pos, text, font=font, fill=(72, 43, 25))
        regions.append({"bbox": _text_bbox(draw, pos, text, font), "text": text,
                        "style": {"weight": kind}})
    _save("product-label", img, regions)


def ui_controls() -> None:
    """Compact UI labels whose terminology and length must remain stable."""
    img = Image.new("RGB", CANVAS, (245, 247, 250))
    draw = ImageDraw.Draw(img)
    regions = []
    for text, pos, fill in [
        ("Save", (180, 150), (36, 110, 210)),
        ("Cancel", (430, 150), (90, 96, 108)),
        ("Delete account", (300, 320), (180, 38, 52)),
    ]:
        font = _font("bold", 36)
        bbox = _text_bbox(draw, pos, text, font)
        x, y, w, h = bbox
        draw.rounded_rectangle([x - 24, y - 14, x + w + 24, y + h + 14],
                               radius=12, fill=fill)
        draw.text(pos, text, font=font, fill=(255, 255, 255))
        regions.append({"bbox": bbox, "text": text})
    _save("ui-controls", img, regions)


def shadow_effects() -> None:
    """High-contrast display text with a visible shadow treatment."""
    img = Image.new("RGB", CANVAS, (223, 232, 245))
    draw = ImageDraw.Draw(img)
    regions = []
    for text, pos, size in [("NIGHT MARKET", (170, 150), 66), ("Every Friday", (310, 300), 42)]:
        font = _font("bold" if size > 50 else "italic", size)
        draw.text((pos[0] + 5, pos[1] + 6), text, font=font, fill=(35, 44, 62))
        draw.text(pos, text, font=font, fill=(246, 92, 74))
        regions.append({"bbox": _text_bbox(draw, pos, text, font), "text": text,
                        "style": {"weight": "bold" if size > 50 else "regular",
                                  "italic": size <= 50, "color": "#f65c4a",
                                  "shadow": {"color": "#232c3e", "offset_x": 5,
                                             "offset_y": 6, "blur": 0}}})
    _save("shadow-effects", img, regions)


def tight_space() -> None:
    """A deliberately tight banner used to prove wrapping/review behavior."""
    img = Image.new("RGB", CANVAS, (250, 248, 242))
    draw = ImageDraw.Draw(img)
    draw.rounded_rectangle([250, 210, 750, 330], radius=16, fill=(40, 92, 78))
    text = "Members only"
    font = _font("bold", 50)
    pos = (330, 240)
    draw.text(pos, text, font=font, fill=(255, 255, 255))
    _save("tight-space", img, [{"bbox": _text_bbox(draw, pos, text, font), "text": text}])


def rtl_sign() -> None:
    """English source sign localized to a right-to-left target."""
    img = Image.new("RGB", CANVAS, (238, 233, 220))
    draw = ImageDraw.Draw(img)
    draw.rectangle([180, 160, 820, 370], fill=(38, 72, 112))
    text = "Welcome"
    font = _font("bold", 72)
    pos = (340, 225)
    draw.text(pos, text, font=font, fill=(255, 255, 255))
    _save("rtl-sign", img, [{"bbox": _text_bbox(draw, pos, text, font), "text": text}])


def mixed_script() -> None:
    """Latin product token plus localized non-Latin descriptor and digits."""
    img = Image.new("RGB", CANVAS, (249, 245, 236))
    draw = ImageDraw.Draw(img)
    draw.rounded_rectangle([140, 145, 860, 390], radius=18, fill=(86, 42, 98))
    regions = []
    for text, pos, size in [("TOFU PRO", (250, 190), 64), ("Version 2", (360, 300), 38)]:
        font = _font("bold" if size > 50 else "regular", size)
        draw.text(pos, text, font=font, fill=(255, 248, 225))
        regions.append({"bbox": _text_bbox(draw, pos, text, font), "text": text})
    _save("mixed-script", img, regions)


def cjk_horizontal() -> None:
    """Wide Latin source regions localized to horizontal Japanese."""
    img = Image.new("RGB", CANVAS, (244, 241, 235))
    draw = ImageDraw.Draw(img)
    regions = []
    for text, pos, size in [("City Library", (220, 150), 62), ("Open today", (330, 285), 42)]:
        font = _font("bold" if size > 50 else "regular", size)
        draw.text(pos, text, font=font, fill=(30, 38, 48))
        regions.append({"bbox": _text_bbox(draw, pos, text, font), "text": text})
    _save("cjk-horizontal", img, regions)


def indic_shaping() -> None:
    """Indic/SEA complex shaping: conjuncts and vowel signs that exercise
    HarfBuzz + FreeType shaping in knead.py and scribe rendering."""
    img = Image.new("RGB", CANVAS, (245, 240, 230))
    draw = ImageDraw.Draw(img)
    regions = []
    cases = [
        ("नमस्ते", (200, 120), 56, "regular"),
        ("สวัสดี", (200, 260), 56, "regular"),
        ("வணக்கம்", (200, 400), 48, "regular"),
    ]
    for text, pos, size, kind in cases:
        font = _font(kind, size)
        draw.text(pos, text, font=font, fill=(30, 30, 34))
        regions.append({"bbox": _text_bbox(draw, pos, text, font), "text": text})
    _save("indic-shaping", img, regions)


def perspective_sign() -> None:
    """Text rendered on a perspective-tilted plane: tests quad extraction
    and perspective-aware reconstruction in Cleanse."""
    img = Image.new("RGB", CANVAS, (230, 232, 238))
    draw = ImageDraw.Draw(img)
    regions = []
    text = "CAFE"
    font = _font("bold", 80)
    quad = [(220, 140), (620, 100), (660, 260), (260, 320)]
    draw.polygon(quad, fill=(42, 62, 88))
    cx = sum(p[0] for p in quad) / 4
    cy = sum(p[1] for p in quad) / 4
    bbox_l = min(p[0] for p in quad)
    bbox_t = min(p[1] for p in quad)
    bbox_r = max(p[0] for p in quad)
    bbox_b = max(p[1] for p in quad)
    draw.text((cx - 80, cy - 30), text, font=font, fill=(255, 255, 255))
    regions.append({
        "bbox": [int(bbox_l), int(bbox_t), int(bbox_r - bbox_l), int(bbox_b - bbox_t)],
        "text": text,
        "quad": [[int(p[0]), int(p[1])] for p in quad],
    })
    _save("perspective-sign", img, regions)


def multicolour_text() -> None:
    """Multi-colour text on a single sign: exercises text_mask's
    multi-colour binarization weakness identified in the readiness doc."""
    img = Image.new("RGB", CANVAS, (240, 238, 242))
    draw = ImageDraw.Draw(img)
    draw.rounded_rectangle([120, 140, 840, 380], radius=16, fill=(28, 28, 32))
    regions = []
    font = _font("bold", 64)
    colours = [(255, 80, 80), (80, 200, 120), (120, 160, 255)]
    x = 180
    for ch, colour in zip("ABC", colours):
        pos = (x, 210)
        draw.text(pos, ch, font=font, fill=colour)
        x += 120
    regions.append({
        "bbox": [180, 210, 300, 64],
        "text": "ABC",
        "style": {"multicolour": True, "colours": ["#ff5050", "#50c878", "#78a0ff"]},
    })
    _save("multicolour-text", img, regions)


def mixed_orientation() -> None:
    """Horizontal and vertical text in the same frame: tests detection
    of mixed orientation without false merges."""
    img = Image.new("RGB", CANVAS, (236, 238, 230))
    draw = ImageDraw.Draw(img)
    regions = []
    h_text = "OPEN"
    h_font = _font("bold", 56)
    h_pos = (140, 140)
    draw.text(h_pos, h_text, font=h_font, fill=(30, 30, 34))
    regions.append({"bbox": _text_bbox(draw, h_pos, h_text, h_font), "text": h_text})
    v_text = "店"
    v_font = _font("cjk", 56)
    for i, ch in enumerate(v_text):
        pos = (500, 120 + i * 80)
        draw.text(pos, ch, font=v_font, fill=(180, 40, 40))
    regions.append({"bbox": [500, 120, 56, 56], "text": v_text, "vertical": True})
    _save("mixed-orientation", img, regions)


def low_confidence() -> None:
    """Low-contrast, blurred text that produces low OCR confidence:
    exercises the confidence-decay OCR trigger and review-required
    provenance path."""
    img = Image.new("RGB", CANVAS, (232, 230, 224))
    draw = ImageDraw.Draw(img)
    regions = []
    font = _font("bold", 48)
    pos = (200, 200)
    draw.text(pos, "FADED", font=font, fill=(200, 198, 192))
    regions.append({"bbox": _text_bbox(draw, pos, "FADED", font), "text": "FADED",
                    "style": {"low_contrast": True}})
    _save("low-confidence", img, regions)


def degraded_ink() -> None:
    """Text with simulated degraded ink (partial erosion, broken strokes):
    exercises text_mask's weakness with multicolour and degraded ink
    identified in the readiness doc."""
    img = Image.new("RGB", CANVAS, (234, 236, 228))
    draw = ImageDraw.Draw(img)
    regions = []
    font = _font("bold", 60)
    pos = (180, 180)
    draw.text(pos, "OLD SIGN", font=font, fill=(60, 58, 52))
    rng = np.random.default_rng(SEED + 99)
    arr = np.asarray(img).astype(np.int16)
    noise = rng.integers(0, 2, arr.shape[:2])
    for c in range(3):
        arr[..., c] = np.where(noise, arr[..., c] + rng.integers(-40, 40, arr.shape[:2]), arr[..., c])
    img = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))
    regions.append({"bbox": _text_bbox(draw, pos, "OLD SIGN", font), "text": "OLD SIGN",
                    "style": {"degraded_ink": True}})
    _save("degraded-ink", img, regions)


def main() -> None:
    print(f"writing fixtures to {OUT}")
    flat_sign()
    gradient_banner()
    textured_wall()
    stylized_italic()
    serif_vs_sans()
    expansion_en()
    cjk_vertical()
    dense_layout()
    product_label()
    ui_controls()
    shadow_effects()
    tight_space()
    rtl_sign()
    mixed_script()
    cjk_horizontal()
    indic_shaping()
    perspective_sign()
    multicolour_text()
    mixed_orientation()
    low_confidence()
    degraded_ink()
    print("done.")


if __name__ == "__main__":
    main()
