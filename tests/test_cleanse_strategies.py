## 🍢 cleanse: stroke-level masks + background-adaptive fill strategies
import json

import numpy as np
from PIL import Image

from conftest import FIXTURES
from tofu.core.types import BBox, BgProfil, InstText, Mask, TextManifest
from tofu.layers import cleanse, scene
from tofu.utils.imaging import text_mask


def an_ink_pixel(img, bbox):
    """a real glyph-stroke pixel inside bbox — the bbox's geometric
    center is NOT reliable (it can land in a letter counter/gap now
    that masking is stroke-precise instead of blanket-rectangle)."""
    mask = text_mask(np.array(img), bbox)
    assert mask is not None, "fixture region must segment"
    ys, xs = np.nonzero(mask)
    assert len(ys) > 0
    i = len(ys) // 2
    return bbox.y + int(ys[i]), bbox.x + int(xs[i])


def load_fixture(name: str):
    img = Image.open(FIXTURES / f"{name}.png").convert("RGB")
    gt = json.loads((FIXTURES / f"{name}.gt.json").read_text(encoding="utf-8"))
    return img, gt["regions"]


def manifest_from_gt(name: str, region_idx: int = 0, texture: str | None = None):
    """build a single-instance manifest from fixture ground truth, with
    a real glyph-stroke segmentation mask (not a synthetic rectangle) —
    exercises the actual text_mask() path these tests target."""
    img, regions = load_fixture(name)
    r = regions[region_idx]
    x, y, w, h = r["bbox"]
    pad = 4
    bbox = BBox(x=x - pad, y=y - pad, width=w + 2 * pad, height=h + 2 * pad)
    inst = InstText(
        id="r1", bounding_box=bbox,
        segmentation_mask=Mask(
            polygon=[(bbox.x, bbox.y), (bbox.x + bbox.width, bbox.y),
                     (bbox.x + bbox.width, bbox.y + bbox.height), (bbox.x, bbox.y + bbox.height)],
            confidence=0.9,
        ),
        text=r["text"], target_text=r["text"],
        background_profile=BgProfil(texture=texture) if texture else None,
    )
    manifest = TextManifest(asset_id="a", total_regions=1, instances=[inst])
    return img, manifest, bbox


class TestStrokeLevelMaskReplacesRectangle:
    """the actual regression test for the bug: pixels inside the OLD
    padded-bbox-rectangle mask but OUTSIDE the dilated glyph strokes
    must now be UNCHANGED (or nearly so — feathering blends a few px
    at the true stroke edges, but the bulk of the background inside
    the bbox and away from glyphs must survive untouched)."""

    def test_background_between_glyphs_survives(self):
        img, manifest, bbox = manifest_from_gt("flat-sign", 0, texture="flat")
        src = np.array(img)
        out = cleanse.erase(img, manifest)
        out_np = np.array(out.convert("RGB"))

        # a pixel just inside the bbox, in the corner (far from any glyph
        # stroke and far from the region's own feather radius) must be
        # bit-identical -- proof the fill no longer blankets the bbox
        corner_y, corner_x = bbox.y + 2, bbox.x + 2
        assert tuple(out_np[corner_y, corner_x]) == tuple(src[corner_y, corner_x])

    def test_glyph_pixels_are_actually_erased(self):
        img, manifest, bbox = manifest_from_gt("flat-sign", 0, texture="flat")
        src = np.array(img)
        cy, cx = an_ink_pixel(img, bbox)
        out = cleanse.erase(img, manifest)
        out_np = np.array(out.convert("RGB"))
        assert not np.array_equal(out_np[cy, cx], src[cy, cx])


class TestNoBoundaryGhost:
    """regression for a real bug found during Phase 3 measurement: the
    dilated glyph mask must be generous enough that the feather taper
    (weakest fill right AT the mask boundary) lands past the ink's own
    anti-aliased edge, in real background territory — not on top of it.
    with only 1px of dilation, a thin partially-filled OUTLINE of the
    original letterforms survived at full readability (EasyOCR still
    read "MAIN STREET" off the flat-sign fixture at full confidence
    through the ghost). checked over the WHOLE dilated mask, not a
    single sampled pixel, since a spot-check can miss a thin ring."""

    def test_entire_masked_region_reaches_background_color(self):
        from tofu.layers.cleanse import _region_mask
        import cv2
        img, manifest, bbox = manifest_from_gt("flat-sign", 0, texture="flat")
        arr = np.array(img)
        h, w = arr.shape[:2]
        dilated_mask = _region_mask(np, cv2, arr, bbox, h, w)
        assert dilated_mask is not None and dilated_mask.any()

        out = np.array(cleanse.erase(img, manifest).convert("RGB"))
        bg_reference = arr[bbox.y - 8, bbox.x + bbox.width // 2].astype(int)
        masked_pixels = out[dilated_mask].astype(int)
        # every pixel the mask claims to have erased must be close to
        # the panel's background color -- a surviving ghost outline
        # would show up as a subset of these pixels still near the ink
        # color instead
        max_dev = np.abs(masked_pixels - bg_reference).max(axis=1)
        assert max_dev.max() < 60, (
            f"worst masked pixel deviates {max_dev.max()} from background "
            "-- a boundary ghost likely survived"
        )


class TestFlatFill:
    def test_flat_region_fills_near_background_color(self):
        img, manifest, bbox = manifest_from_gt("flat-sign", 0, texture="flat")
        out = np.array(cleanse.erase(img, manifest).convert("RGB"))
        # sample the panel's known background color just outside the region
        bg_sample = out[bbox.y - 6, bbox.x + bbox.width // 2]
        center = out[bbox.y + bbox.height // 2, bbox.x + bbox.width // 2]
        assert np.abs(center.astype(int) - bg_sample.astype(int)).mean() < 25


class TestGradientFill:
    def test_gradient_region_no_visible_seam(self):
        img, manifest, bbox = manifest_from_gt("gradient-banner", 0, texture="smooth_gradient")
        out = np.array(cleanse.erase(img, manifest).convert("RGB"))
        # the gradient continues across the erased region: color just
        # left of the region and just right of it should differ (real
        # gradient), but the filled interior shouldn't jump abruptly
        # from either edge — check the horizontal gradient trend holds
        left = out[bbox.y + bbox.height // 2, max(0, bbox.x - 3)].astype(int)
        mid = out[bbox.y + bbox.height // 2, bbox.x + bbox.width // 2].astype(int)
        right = out[bbox.y + bbox.height // 2, min(out.shape[1] - 1, bbox.x + bbox.width + 3)].astype(int)
        # mid should sit between left and right on the dominant (red) channel
        lo, hi = sorted([left[0], right[0]])
        assert lo - 15 <= mid[0] <= hi + 15


class TestTexturedFallsBackToInpaint:
    def test_textured_region_still_gets_filled(self):
        img, manifest, bbox = manifest_from_gt("textured-wall", 0, texture="textured")
        src = np.array(img)
        cy, cx = an_ink_pixel(img, bbox)
        out = np.array(cleanse.erase(img, manifest).convert("RGB"))
        assert not np.array_equal(out[cy, cx], src[cy, cx])

    def test_unclassified_texture_defaults_to_inpaint_not_skipped(self):
        img, manifest, bbox = manifest_from_gt("textured-wall", 0, texture=None)
        src = np.array(img)
        cy, cx = an_ink_pixel(img, bbox)
        out = np.array(cleanse.erase(img, manifest).convert("RGB"))
        assert not np.array_equal(out[cy, cx], src[cy, cx])


class TestMaskPolarityGate:
    """regression for a real bug found during Phase 3 measurement: the
    stylized-italic fixture's "Stroked Display" text (dark fill + a very
    different blue stroke color) has higher internal chromatic variance
    than its background, which made GrabCut's GMM converge on the WRONG
    partition — the mask selected the background, leaving the actual ink
    completely unerased (residual OCR similarity 1.0, i.e. the erase was
    a complete no-op). the fix rejects a mask whose selected pixels sit
    CLOSER to the crop's own edge color (near-certainly background) than
    the unselected pixels do."""

    def test_stroked_text_is_not_left_fully_readable(self):
        img, manifest, bbox = manifest_from_gt("stylized-italic", 3, texture="flat")
        src = np.array(img)
        cy, cx = an_ink_pixel(img, bbox)
        out = np.array(cleanse.erase(img, manifest).convert("RGB"))
        # this exact pixel was the reproduction case: mask inversion left
        # it bit-identical to the source (complete erase failure)
        assert not np.array_equal(out[cy, cx], src[cy, cx])

    def test_inverted_mask_is_rejected_not_used(self):
        from tofu.layers.cleanse import _region_mask
        import cv2
        img, regions = load_fixture("stylized-italic")
        x, y, w, h = regions[3]["bbox"]
        bbox = BBox(x=x, y=y, width=w, height=h)
        arr = np.array(img)
        H, W = arr.shape[:2]
        result = _region_mask(np, cv2, arr, bbox, H, W)
        # either the mask was rejected (None -> bbox fallback fires in
        # erase()) or a plausible one was found; it must never be the
        # inverted (background-selected) mask this fixture triggers
        if result is not None:
            crop = arr[y:y + h, x:x + w]
            sub = result[y:y + h, x:x + w]
            assert sub.any()
            ink_mean = crop[sub].reshape(-1, 3).mean(axis=0)
            bg_mean = crop[~sub].reshape(-1, 3).mean(axis=0)
            # the background here is near-white (~245,244,240); the
            # selected group must NOT be the light one
            assert ink_mean.mean() < bg_mean.mean()


class TestSegmentationFailureFallback:
    def test_degenerate_region_still_erases_via_bbox_fallback(self):
        # a uniform-color crop defeats Otsu (text_mask returns None) —
        # the bbox-rectangle fallback must still fire, not silently skip
        img = Image.new("RGB", (200, 120), (100, 150, 100))
        bbox = BBox(x=50, y=40, width=60, height=30)
        inst = InstText(
            id="r1", bounding_box=bbox, text="X", target_text="X",
            background_profile=BgProfil(texture="flat"),
        )
        manifest = TextManifest(asset_id="a", total_regions=1, instances=[inst])
        out = np.array(cleanse.erase(img, manifest).convert("RGB"))
        # uniform source means fill color == source color, so we can't
        # check pixel change; instead confirm erase() didn't crash and
        # returned a same-shape image (degenerate-mask path exercised
        # without exception)
        assert out.shape == (120, 200, 3)


class TestDntAndBoundsUnchanged:
    def test_dnt_region_untouched(self):
        img, manifest, bbox = manifest_from_gt("flat-sign", 0, texture="flat")
        manifest.instances[0].dnt = True
        src = np.array(img)
        out = np.array(cleanse.erase(img, manifest).convert("RGB"))
        assert np.array_equal(out, src)

    def test_far_corner_pixels_bit_identical(self):
        img, manifest, _ = manifest_from_gt("flat-sign", 0, texture="flat")
        src = np.array(img)
        out = np.array(cleanse.erase(img, manifest).convert("RGB"))
        assert np.array_equal(out[:10, :10], src[:10, :10])


class TestSceneIntegration:
    """end-to-end: scene.analyze() populates background_profile.texture,
    and cleanse consumes it directly — the actual pipeline wiring."""

    def test_scene_enriched_manifest_drives_strategy_selection(self):
        img, regions = load_fixture("gradient-banner")
        x, y, w, h = regions[0]["bbox"]
        pad = 4
        bbox = BBox(x=x - pad, y=y - pad, width=w + 2 * pad, height=h + 2 * pad)
        inst = InstText(id="r1", bounding_box=bbox, text=regions[0]["text"],
                        target_text=regions[0]["text"])
        manifest = TextManifest(asset_id="a", total_regions=1, instances=[inst])
        manifest = scene.analyze(img, manifest)
        assert manifest.instances[0].background_profile.texture == "smooth_gradient"
        # cleanse must not crash consuming a REAL scene-enriched manifest
        out = cleanse.erase(img, manifest)
        assert out is not None
