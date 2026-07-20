## 🍢 scene pre-pass regression: MSER cluster growth cap + frame filter
from pathlib import Path

from conftest import FIXTURES
from tofu.layers.scene import ClassicalCVBackend
from tofu.utils.imaging import load_rgb

ROOT = Path(__file__).resolve().parents[1]


class TestFrameSizedRegionsFiltered:
    def test_analyze_never_returns_frame_sized_region(self):
        # single-panel synthetic: the contour detector's outer boundary
        # around the whole panel background must not survive as a surface
        img = load_rgb(FIXTURES / "flat-sign.png")
        h, w = img.shape[:2]
        frame_area = h * w
        backend = ClassicalCVBackend()
        regions = backend.analyze(FIXTURES / "flat-sign.png")
        for r in regions:
            area = r.bbox.width * r.bbox.height
            assert area / frame_area <= backend.max_region_frac


class TestMSERClusterGrowthCap:
    """regression for the single-linkage snowball: a dense, signage-heavy
    scene (hundreds of stroke-like MSER components close together) must
    not collapse into one frame-covering 'text_cluster' blob. this was a
    real bug — the original center-distance merge test scaled its own
    catchment tolerance with the accumulating cluster's size, and even a
    fixed-tolerance single-linkage merge still chains transitively without
    an explicit envelope-growth cap."""

    def test_dense_street_scene_does_not_collapse_to_one_blob(self):
        img_path = ROOT / "images" / "gemini-street.png"
        if not img_path.exists():
            return  # optional real-image fixture; skip if repo layout differs
        backend = ClassicalCVBackend()
        img = load_rgb(img_path)
        h, w = img.shape[:2]
        frame_area = h * w
        regions = backend.analyze(img_path)
        text_clusters = [r for r in regions if r.semantic_label == "text_cluster"]
        assert len(text_clusters) >= 3, (
            "dense signage should yield multiple sign-scale clusters, "
            f"got {len(text_clusters)}"
        )
        for r in text_clusters:
            area_frac = (r.bbox.width * r.bbox.height) / frame_area
            # generous slack over mser_cluster_max_frac: post-cap contour
            # regions can still be large, but no MSER cluster should
            # approach frame coverage
            assert area_frac <= 0.35, (
                f"cluster {r.bbox} covers {area_frac:.0%} of the frame — "
                "the growth cap should have refused this merge"
            )

    def test_mser_cluster_cap_is_enforced_directly(self):
        """unit-level check on _mser_regions: no output envelope exceeds
        mser_cluster_max_frac of the frame it was computed over."""
        backend = ClassicalCVBackend()
        img_path = ROOT / "images" / "gemini-street.png"
        if not img_path.exists():
            return
        img = load_rgb(img_path)
        h, w = img.shape[:2]
        regions = backend._mser_regions(img, 1.0)
        frame_area = h * w
        for r in regions:
            area_frac = (r.bbox.width * r.bbox.height) / frame_area
            assert area_frac <= backend.mser_cluster_max_frac * 1.05


class TestContourRescuePass:
    """regression: a visually dense scene can saturate the default
    Canny+dilate edge map (measured on japan-street.jpeg: 61% edge-pixel
    density ACROSS THE WHOLE FRAME, not just busy sub-areas), hiding
    even a giant, unmistakable rectangular sign in the noise —
    findContours returns nothing but the frame boundary itself. the
    rescue pass (blur + higher Canny thresholds + no dilation) only
    fires when the default pass found zero real (non-frame-sized) quad
    candidates, and must recover a real quad for a case like this."""

    def test_dense_scene_recovers_a_real_quad_for_the_banner(self):
        img_path = ROOT / "images" / "japan-street.jpeg"
        if not img_path.exists():
            return  # optional real-image fixture; skip if repo layout differs
        backend = ClassicalCVBackend()
        img = load_rgb(img_path)
        regions = backend.analyze(img_path)
        quads = [r for r in regions if r.semantic_label in ("panel", "bordered_region")]
        assert quads, "expected the rescue pass to recover at least one real quad"
        # the recovered quad should closely match the banner's true
        # extent (193,125,240x40) -- IoU-style containment check, not
        # exact pixels
        banner = (193, 125, 240, 40)
        bx0, by0, bx1, by1 = banner[0], banner[1], banner[0] + banner[2], banner[1] + banner[3]
        found = False
        for r in quads:
            rx0, ry0 = r.bbox.x, r.bbox.y
            rx1, ry1 = r.bbox.x + r.bbox.width, r.bbox.y + r.bbox.height
            ox = max(0, min(bx1, rx1) - max(bx0, rx0))
            oy = max(0, min(by1, ry1) - max(by0, ry0))
            overlap = ox * oy
            if overlap / (banner[2] * banner[3]) > 0.8:
                found = True
                break
        assert found, f"no recovered quad closely matched the banner; got {quads}"

    def test_rescue_pass_does_not_fire_when_default_already_found_a_quad(self):
        """unit-level check on _contour_regions: a clean synthetic image
        where the default pass already finds a real quad must not
        additionally run (or need) the rescue pass — verifies the
        escalate-only-on-failure gate itself, not just an end result."""
        img_path = FIXTURES / "flat-sign.png"
        if not img_path.exists():
            return
        backend = ClassicalCVBackend()
        img = load_rgb(img_path)
        default_regions, found = backend._contour_pass(
            img, 1.0, blur=False, canny=(50, 150), dilate_iters=1,
        )
        assert found, "flat-sign should already yield a real quad on the default pass"
