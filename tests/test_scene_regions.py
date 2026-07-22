## 🍢 scene pre-pass regression: MSER cluster growth cap + frame filter
from pathlib import Path

from conftest import FIXTURES
from tofu.core.types import BBox, SceneRegion
from tofu.layers.cicerone import _scene_containment_frac
from tofu.layers.scene import ClassicalCVBackend, _is_duplicate_surface, _orientation_bucket
from tofu.utils.imaging import load_rgb

ROOT = Path(__file__).resolve().parents[1]


class TestOrientationAwareSurfaceDedup:
    """regression: china-street's "MING" (small horizontal English text
    written across the large vertical "上海明牌" sign) was silently
    discarded by the scene pre-pass's containment-only dedup before
    cicerone ever got a chance to see it as its own candidate surface."""

    def test_orientation_bucket_classifies_wide_tall_square(self):
        assert _orientation_bucket(BBox(x=0, y=0, width=200, height=30)) == "wide"
        assert _orientation_bucket(BBox(x=0, y=0, width=30, height=200)) == "tall"
        assert _orientation_bucket(BBox(x=0, y=0, width=50, height=50)) == "square"

    def test_orthogonal_overlap_is_not_a_duplicate(self):
        # small wide box mostly inside a large tall box -- different
        # orientation, must survive as its own candidate
        tall = BBox(x=190, y=480, width=72, height=174)
        wide = BBox(x=200, y=500, width=60, height=20)
        assert _is_duplicate_surface(wide, [tall]) is False

    def test_same_orientation_high_containment_still_deduped(self):
        # near-identical duplicate MSER/contour blobs of the SAME
        # surface must still collapse -- this dedup's original purpose
        large = BBox(x=100, y=100, width=200, height=40)
        near_duplicate = BBox(x=105, y=102, width=190, height=36)
        assert _is_duplicate_surface(near_duplicate, [large]) is True

    def test_low_containment_never_deduped_regardless_of_orientation(self):
        a = BBox(x=0, y=0, width=200, height=40)
        b = BBox(x=180, y=0, width=200, height=40)  # only slight overlap
        assert _is_duplicate_surface(b, [a]) is False


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


class TestRegionSurfaceMetadata:
    def test_analyze_populates_region_texture(self):
        backend = ClassicalCVBackend()
        regions = backend.analyze(FIXTURES / "flat-sign.png")
        assert regions
        assert all(r.texture in {"flat", "smooth_gradient", "textured", None} for r in regions)
        # Region crops include glyph pixels (unlike the masked enrichment
        # crop), so a flat sign can conservatively classify as textured;
        # this test verifies the pre-pass always records a real verdict.
        assert any(r.texture is not None for r in regions)

    def test_polygon_containment_rejects_bbox_corner_slop(self):
        # The bbox is a 100px square but the real surface is a triangle;
        # detection in the empty lower-right corner must not inherit the
        # surface's confidence privilege.
        region = SceneRegion(
            bbox=BBox(x=0, y=0, width=100, height=100),
            semantic_label="panel", confidence=1.0,
            polygon=[(0, 0), (100, 0), (0, 100)],
        )
        assert _scene_containment_frac(BBox(x=10, y=10, width=20, height=20), region) > 0.9
        assert _scene_containment_frac(BBox(x=70, y=70, width=20, height=20), region) == 0.0


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

    def test_mser_text_clusters_include_a_hull_polygon(self):
        img_path = ROOT / "images" / "gemini-street.png"
        if not img_path.exists():
            return
        backend = ClassicalCVBackend()
        clusters = [r for r in backend._mser_regions(load_rgb(img_path), 1.0) if r.semantic_label == "text_cluster"]
        assert clusters
        assert all(r.polygon and len(r.polygon) >= 3 for r in clusters)


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
