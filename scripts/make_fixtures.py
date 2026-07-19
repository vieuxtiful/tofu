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
}


def _font(kind: str, size: int):
    for cand in _FONTS[kind]:
        try:
            return ImageFont.truetype(cand, size)
        except Exception:
            continue
    return ImageFont.load_default()


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
        regions.append({"bbox": _text_bbox(draw, pos, text, font), "text": text})
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
                      "italic": kind == "italic", "stroked": stroke is not None},
        })
    _save("stylized-italic", img, regions)


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


def main() -> None:
    print(f"writing fixtures to {OUT}")
    flat_sign()
    gradient_banner()
    textured_wall()
    stylized_italic()
    expansion_en()
    cjk_vertical()
    print("done.")


if __name__ == "__main__":
    main()
