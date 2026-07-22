## 🍢 background classification against fixture ground truth
import json

import numpy as np
from PIL import Image

from conftest import FIXTURES
from tofu.core.types import BBox
from tofu.layers.scene import _classify_background, _describe_surface_material, _estimate_colors


def region_inputs(name: str, region_idx: int, pad: int = 6):
    img = np.asarray(Image.open(FIXTURES / f"{name}.png").convert("RGB"))
    gt = json.loads((FIXTURES / f"{name}.gt.json").read_text(encoding="utf-8"))
    x, y, w, h = gt["regions"][region_idx]["bbox"]
    bbox = BBox(x=x - pad, y=y - pad, width=w + 2 * pad, height=h + 2 * pad)
    _, _, _, mask, crop = _estimate_colors(img, bbox)
    return crop, mask


class TestClassifyBackground:
    def test_flat_sign_is_flat(self):
        crop, mask = region_inputs("flat-sign", 0)  # white on green panel
        texture, gradients = _classify_background(crop, mask)
        assert texture == "flat"
        assert gradients is None

    def test_gradient_banner_is_smooth_gradient(self):
        crop, mask = region_inputs("gradient-banner", 0)
        texture, gradients = _classify_background(crop, mask)
        assert texture == "smooth_gradient"
        assert gradients and gradients[0].startswith("linear")

    def test_textured_wall_is_textured(self):
        crop, mask = region_inputs("textured-wall", 0)
        texture, gradients = _classify_background(crop, mask)
        assert texture == "textured"

    def test_degenerate_input_returns_none(self):
        assert _classify_background(None, None) == (None, None)

    def test_tiny_background_returns_none(self):
        crop = np.full((6, 6, 3), 128, dtype=np.uint8)
        mask = np.ones((6, 6), dtype=bool)  # everything is glyph
        assert _classify_background(crop, mask) == (None, None)

    def test_material_label_keeps_routing_texture_separate(self):
        # A bounded sign is still a textured crop for Cleanse routing, but
        # users need an actionable material label rather than "text_cluster".
        crop = np.full((60, 100, 3), (30, 80, 150), dtype=np.uint8)
        assert _describe_surface_material(crop, "textured", "bordered_region") == "painted sign / panel"

    def test_brick_pattern_is_named_masonry_when_evidence_is_present(self):
        crop = np.full((90, 120, 3), 180, dtype=np.uint8)
        # three mortar courses and staggered joints: enough independent
        # structure for the intentionally conservative brick label.
        crop[20:23, :, :] = 55
        crop[48:51, :, :] = 55
        crop[76:79, :, :] = 55
        crop[3:20, 30:33, :] = 55
        crop[23:48, 72:75, :] = 55
        crop[51:76, 38:41, :] = 55
        assert _describe_surface_material(crop, "textured", "text_cluster") == "brick / masonry"
