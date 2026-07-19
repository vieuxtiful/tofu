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
