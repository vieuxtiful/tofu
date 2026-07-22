"""Visual font retrieval remains evidence only; it never mutates the style."""

from pathlib import Path

import pytest
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from tofu.core.types import BBox, InstText, TextManifest
from tofu.layers.font_matching import external_catalog_match, local_match
from tofu.layers.fonts import FontCoverage, FontRegistry


def _font_path(name: str) -> str | None:
    candidates = [
        Path("C:/Windows/Fonts") / name,
        Path("/usr/share/fonts/truetype/dejavu") / name,
        Path("/Library/Fonts") / name,
    ]
    for candidate in candidates:
        if candidate.exists():
            return str(candidate)
    return None


def _registry(exact: str, other: str) -> FontRegistry:
    glyphs = {ord(ch) for ch in "MURS "}
    registry = FontRegistry()
    registry._fonts = {
        exact: FontCoverage(exact, "Exact Sans", "Bold", 700, glyphs),
        other: FontCoverage(other, "Different Serif", "Bold", 700, glyphs),
    }
    return registry


def test_local_glyph_match_ranks_exact_face_without_changing_user_style():
    exact = _font_path("arialbd.ttf") or _font_path("DejaVuSans-Bold.ttf")
    other = _font_path("timesbd.ttf") or _font_path("DejaVuSerif-Bold.ttf")
    if not exact or not other:
        pytest.skip("two known system font faces are required for glyph retrieval test")

    font = ImageFont.truetype(exact, 54)
    image = Image.new("RGB", (300, 120), "white")
    draw = ImageDraw.Draw(image)
    box = draw.textbbox((0, 0), "MURS", font=font)
    draw.text((16 - box[0], 24 - box[1]), "MURS", font=font, fill="black")
    inst = InstText(id="r4", bounding_box=BBox(10, 18, box[2] - box[0] + 12, box[3] - box[1] + 12), text="MURS")

    result = local_match(np.asarray(image), inst, _registry(exact, other))

    assert result is not None
    assert result["candidates"][0]["family"] == "Exact Sans"
    assert result["recommended_substitute"]["font_path"] == exact
    assert inst.style_profile is None  # retrieval must never silently select a font


def test_unconfigured_catalog_never_sends_a_crop(monkeypatch):
    monkeypatch.delenv("TOFU_WHATFONTIS_API_KEY", raising=False)
    inst = InstText(id="r1", bounding_box=BBox(0, 0, 20, 20), text="Rue", font_match={"candidates": []})
    manifest = TextManifest(asset_id="a", total_regions=1, instances=[inst])

    assert external_catalog_match("not-opened.png", manifest, None) == 0
    assert inst.font_match["external_provider"]["enabled"] is False
