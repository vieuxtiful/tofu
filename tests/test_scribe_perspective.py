## 🍢 projective text placement
"""
skew_x/skew_y are a shear, and shear is affine: it keeps opposite edges
parallel and equal.  A sign photographed at an angle has CONVERGING edges,
which no combination of shear, scale and rotation reproduces -- six degrees
of freedom against the eight a homography needs.

So the tests that matter here are about the property, not the matrix: a quad
must be able to make one edge shorter than its opposite, and skew must not.
"""

import numpy as np
import pytest
from PIL import Image, ImageDraw

from tofu.core.types import BBox, InstText, StyleProfil, TextManifest
from tofu.layers import scribe, verify

BOX = BBox(80, 60, 240, 90)
IDENTITY = [[0, 0], [1, 0], [1, 1], [0, 1]]
#: top corners pulled inward -- the trapezoid a sign makes when its far edge
#: is further from the camera than its near edge
CONVERGING = [[0.25, 0.0], [0.75, 0.0], [1.0, 1.0], [0.0, 1.0]]
#: a parallelogram: shear expressed as corners, still affine
SHEARED = [[0.25, 0.0], [1.25, 0.0], [1.0, 1.0], [0.0, 1.0]]


def _block() -> Image.Image:
    """A filled rectangle exactly covering BOX: edge lengths are readable."""
    image = Image.new("RGBA", (460, 260), (0, 0, 0, 0))
    ImageDraw.Draw(image).rectangle(
        (BOX.x, BOX.y, BOX.x + BOX.width, BOX.y + BOX.height), fill=(0, 0, 0, 255)
    )
    return image


def _alpha(image: Image.Image) -> np.ndarray:
    return np.asarray(image.split()[-1]) > 8


def _row_width(image: Image.Image, fraction: float) -> int:
    """Width of the ink band at a given height through the ink extent."""
    mask = _alpha(image)
    ys, xs = np.nonzero(mask)
    if not len(xs):
        return 0
    y = int(ys.min() + fraction * (ys.max() - ys.min()))
    row = np.nonzero(mask[y])[0]
    return int(row.max() - row.min() + 1) if len(row) else 0


def _ink_bounds(image: Image.Image):
    mask = _alpha(image)
    ys, xs = np.nonzero(mask)
    return None if not len(xs) else (xs.min(), ys.min(), xs.max(), ys.max())


class TestQuadHelpers:
    def test_absent_and_identity_alike_mean_no_perspective(self):
        assert scribe._is_identity_quad(None)
        assert scribe._is_identity_quad(scribe._parse_quad(IDENTITY))

    def test_malformed_quads_parse_to_none_rather_than_raising(self):
        for bad in (None, [], [[0, 0]], [[0, 0], [1, 0], [1, 1]], "quad",
                    [[0, 0], [1, 0], [1, 1], ["x", 1]]):
            assert scribe._parse_quad(bad) is None

    def test_denormalise_places_the_identity_on_the_box(self):
        assert scribe._denormalise_quad(scribe._parse_quad(IDENTITY), BOX) == [
            (80.0, 60.0), (320.0, 60.0), (320.0, 150.0), (80.0, 150.0)
        ]

    @pytest.mark.parametrize("corners,usable", [
        ([(0, 0), (240, 0), (240, 90), (0, 90)], True),      # the box
        ([(60, 0), (180, 0), (240, 90), (0, 90)], True),     # trapezoid
        ([(0, 0), (120, 0), (240, 0), (0, 90)], False),      # collinear
        ([(0, 0), (240, 0), (0, 90), (240, 90)], False),     # bowtie
    ])
    def test_degenerate_quads_are_rejected_before_the_solve(self, corners, usable):
        assert scribe._quad_is_usable(corners) is usable


class TestProjectiveRender:
    def test_an_identity_quad_changes_nothing(self):
        out = scribe._apply_style_transform(_block(), BOX, {"quad": IDENTITY})
        assert _ink_bounds(out) == _ink_bounds(_block())

    def test_converging_corners_shorten_one_edge(self):
        """The property affine cannot produce, and the whole reason the quad
        exists."""
        out = scribe._apply_style_transform(_block(), BOX, {"quad": CONVERGING})
        top, bottom = _row_width(out, 0.05), _row_width(out, 0.95)
        assert top < bottom * 0.6

    def test_skew_keeps_both_edges_equal(self):
        """The control: 30 degrees of shear leans the block without
        narrowing it, because a shear maps parallel lines to parallel
        lines."""
        out = scribe._apply_style_transform(_block(), BOX, {"skew_x": 30})
        assert _row_width(out, 0.05) == pytest.approx(_row_width(out, 0.95), abs=3)

    def test_a_parallelogram_quad_is_affine_and_stays_affine(self):
        """A quad CAN express a shear, and when it does it must behave like
        one -- equal opposite edges, no convergence sneaking in."""
        out = scribe._apply_style_transform(_block(), BOX, {"quad": SHEARED})
        assert _row_width(out, 0.05) == pytest.approx(_row_width(out, 0.95), abs=3)

    def test_the_quad_replaces_the_affine_stage_rather_than_composing(self):
        """Corner positions already encode any shear; applying skew as well
        would apply it twice."""
        with_skew = scribe._apply_style_transform(
            _block(), BOX, {"quad": CONVERGING, "skew_x": 25}
        )
        without = scribe._apply_style_transform(_block(), BOX, {"quad": CONVERGING})
        assert _ink_bounds(with_skew) == _ink_bounds(without)

    @pytest.mark.parametrize("quad", [
        [[0, 0], [0.5, 0], [1, 0], [0, 1]],      # collinear
        [[0, 0], [1, 0], [0, 1], [1, 1]],        # self-intersecting
        [[0, 0], [1, 0]],                        # malformed
    ])
    def test_a_degenerate_quad_falls_back_instead_of_raising(self, quad):
        """A singular solve must never surface as an exception mid-render."""
        out = scribe._apply_style_transform(_block(), BOX, {"quad": quad, "skew_x": 20})
        assert _ink_bounds(out) is not None

    def test_a_corner_pulled_outside_the_box_is_not_clipped(self):
        """Ink lands where the corner went, so the working crop has to
        follow it -- otherwise the pad that sizes the crop clips the warp."""
        out = scribe._apply_style_transform(
            _block(), BOX, {"quad": [[-0.4, 0.0], [1.4, 0.0], [1.0, 1.0], [0.0, 1.0]]}
        )
        assert _row_width(out, 0.05) > BOX.width * 1.3


class TestVerifyContainment:
    """Verify measures ink against the region's footprint.  Under
    perspective that footprint is the quad, not the source rectangle."""

    def _region(self, transform):
        inst = InstText(
            id="r1", bounding_box=BBox(60, 50, 280, 80),
            text="City Library", target_text="City Library", target_language="en",
            style_profile=StyleProfil(color="#000000", transform=transform),
        )
        manifest = TextManifest(
            asset_id="a", total_regions=1, instances=[inst],
            src_lang="en", targ_lang="en",
        )
        rendered = scribe.render(
            Image.new("RGB", (500, 260), (255, 255, 255)), manifest, "en"
        )
        return verify.build_verification_report(rendered, manifest).regions[0], manifest

    def test_a_region_without_a_quad_is_measured_against_its_bbox(self):
        region, _ = self._region(None)
        assert region.checks["geometry"]["containment_shape"] == "bbox"

    def test_a_warped_region_is_measured_against_its_quad(self):
        region, _ = self._region({"quad": CONVERGING})
        assert region.checks["geometry"]["containment_shape"] == "quad"
        assert region.checks["geometry"]["overflow_ratio"] == 0.0
        assert region.scores["spatial_fit"] == 100.0

    def test_the_bbox_would_have_reported_false_overflow(self, monkeypatch):
        """The regression this exists to prevent: a corner pulled outside
        the box puts correctly-seated ink outside the rectangle, and the
        rectangle calls that overflow."""
        outward = {"quad": [[-0.25, 0.0], [1.25, 0.0], [1.0, 1.0], [0.0, 1.0]]}
        region, _ = self._region(outward)
        assert region.checks["geometry"]["overflow_ratio"] == 0.0

        monkeypatch.setattr(verify, "_region_quad", lambda inst, bounds: None)
        stale, _ = self._region(outward)
        assert stale.checks["geometry"]["overflow_ratio"] > 0.1
        assert "spatial_overflow" in stale.flags

    def test_a_degenerate_quad_falls_back_in_the_check_as_it_does_in_the_render(self):
        """The check and the render must agree about which shape the region
        occupies, or one of them is measuring something that was not drawn."""
        region, _ = self._region({"quad": [[0, 0], [0.5, 0], [1, 0], [0, 1]]})
        assert region.checks["geometry"]["containment_shape"] == "bbox"
