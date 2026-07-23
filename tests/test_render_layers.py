## 🍢 cleanse / scribe / verify units on synthetic images
import numpy as np
from PIL import Image

from tofu.core.types import BBox, InstText, Mask, RenderParams, StyleProfil, TextManifest
from tofu.layers import cleanse, scribe, verify
from tofu.layers.scribe import _apply_style_transform, _pixel_bbox


def make_asset(w=300, h=200, color=(90, 140, 90)):
    return Image.new("RGB", (w, h), color)


def make_manifest(instances):
    return TextManifest(asset_id="a", total_regions=len(instances),
                        instances=instances, src_lang="en")


def text_inst(x=50, y=50, w=120, h=40, text="HELLO", target=None, dnt=False, excluded=False):
    return InstText(
        id="r1", bounding_box=BBox(x=x, y=y, width=w, height=h),
        segmentation_mask=Mask(
            polygon=[(x, y), (x + w, y), (x + w, y + h), (x, y + h)],
            confidence=0.9,
        ),
        text=text, target_text=target, dnt=dnt, excluded=excluded,
    )


class TestCleanse:
    def test_erase_changes_region_pixels(self):
        img = make_asset()
        d = np.asarray(img).copy()
        d[60:80, 60:160] = (255, 255, 255)  # fake text strokes
        asset = Image.fromarray(d)
        out = cleanse.erase(asset, make_manifest([text_inst()]))
        out_np = np.asarray(out.convert("RGB"))
        # white strokes must be gone (inpainted toward surrounding green)
        assert out_np[70, 100].tolist() != [255, 255, 255]

    def test_dnt_region_untouched(self):
        img = make_asset()
        d = np.asarray(img).copy()
        d[60:80, 60:160] = (255, 255, 255)
        asset = Image.fromarray(d)
        out = cleanse.erase(asset, make_manifest([text_inst(dnt=True)]))
        out_np = np.asarray(out.convert("RGB"))
        assert out_np[70, 100].tolist() == [255, 255, 255]

    def test_excluded_region_still_erased(self):
        # unlike dnt, a user-excluded region must still be cleansed --
        # only its rendering (scribe) is skipped, not its erasure
        img = make_asset()
        d = np.asarray(img).copy()
        d[60:80, 60:160] = (255, 255, 255)
        asset = Image.fromarray(d)
        out = cleanse.erase(asset, make_manifest([text_inst(excluded=True)]))
        out_np = np.asarray(out.convert("RGB"))
        assert out_np[70, 100].tolist() != [255, 255, 255]

    def test_pixels_outside_bbox_preserved(self):
        asset = make_asset()
        out = cleanse.erase(asset, make_manifest([text_inst()]))
        out_np = np.asarray(out.convert("RGB"))
        src_np = np.asarray(asset)
        # corner far from the region must be bit-identical
        assert (out_np[:20, :20] == src_np[:20, :20]).all()


class TestScribe:
    def test_editor_geometry_is_snapped_before_rasterization(self):
        # JSON/client values may be floats even though persisted BBox fields
        # are typed as ints.  Rendering must not pass half-pixel origins to
        # Pillow after a position or size edit.
        snapped = _pixel_bbox(BBox(x=20.49, y=30.51, width=99.51, height=39.49))
        assert snapped == BBox(x=20, y=31, width=100, height=39)

    def test_renders_target_text_pixels(self):
        asset = make_asset(color=(255, 255, 255))
        inst = text_inst(target="HELLO")
        out = scribe.render(asset, make_manifest([inst]), "en")
        out_np = np.asarray(out)
        b = inst.bounding_box
        region = out_np[b.y:b.y + b.height, b.x:b.x + b.width]
        assert (region < 200).any()  # dark glyphs on white

    def test_emits_exact_per_instance_text_coverage_masks(self):
        asset = make_asset(color=(255, 255, 255))
        first = text_inst(x=20, y=30, w=100, h=35, text="one", target="ONE")
        second = text_inst(x=145, y=90, w=100, h=35, text="two", target="TWO")
        second.id = "r2"
        out = scribe.render(asset, make_manifest([first, second]), "en")
        assert set(out.text_masks) == {"r1", "r2"}
        assert all(mask.mode == "L" and mask.size == out.size for mask in out.text_masks.values())
        first_mask = np.asarray(out.text_masks["r1"])
        second_mask = np.asarray(out.text_masks["r2"])
        assert first_mask[30:65, 20:120].max() > 0
        assert second_mask[30:65, 20:120].max() == 0

    def test_same_color_stroke_uses_one_effective_ink_at_region_opacity(self):
        """A same-colour stroke must not stay opaque around translucent fill."""
        asset = make_asset(color=(255, 255, 255))
        inst = text_inst(x=30, y=30, w=170, h=70, target="TOFU")
        inst.style_profile = StyleProfil(
            font_size=42,
            color="#204a87",
            stroke_color="#204a87",
            stroke_width=4,
        )
        out = scribe.render(
            asset,
            make_manifest([inst]),
            "en",
            render_params={
                "r1": RenderParams(position=inst.bounding_box, style=inst.style_profile, opacity=0.5)
            },
        )
        coverage = np.asarray(out.text_masks["r1"])
        ink = coverage[coverage > 0]
        assert ink.size > 0
        # 50%-opaque fill and stroke must share the same maximum alpha.  The
        # old split treatment produced an opaque (255) outline around a 127
        # alpha fill, which made the join visibly discernible when zoomed.
        assert 120 <= ink.max() <= 128

    def test_untranslated_region_left_empty(self):
        asset = make_asset(color=(255, 255, 255))
        out = scribe.render(asset, make_manifest([text_inst(target=None)]), "en")
        assert (np.asarray(out) == 255).all()

    def test_dnt_region_not_rendered(self):
        asset = make_asset(color=(255, 255, 255))
        out = scribe.render(
            asset, make_manifest([text_inst(target="HELLO", dnt=True)]), "en"
        )
        assert (np.asarray(out) == 255).all()

    def test_excluded_region_not_rendered(self):
        asset = make_asset(color=(255, 255, 255))
        out = scribe.render(
            asset, make_manifest([text_inst(target="HELLO", excluded=True)]), "en"
        )
        assert (np.asarray(out) == 255).all()

    def test_non_image_asset_passes_through(self):
        sentinel = object()
        assert scribe.render(sentinel, make_manifest([]), "en") is sentinel

    def test_named_warp_and_stretch_are_deterministic(self):
        from PIL import ImageDraw
        layer = Image.new("RGBA", (160, 90), (0, 0, 0, 0))
        ImageDraw.Draw(layer).rectangle((35, 30, 125, 55), fill=(0, 0, 0, 255))
        bbox = BBox(x=30, y=20, width=100, height=50)
        spec = {"preset": "wave", "amount": 12, "scale_x": 1.15, "scale_y": .9}
        a = _apply_style_transform(layer, bbox, spec)
        b = _apply_style_transform(layer, bbox, spec)
        assert np.array_equal(np.asarray(a), np.asarray(b))
        assert not np.array_equal(np.asarray(a), np.asarray(layer))

    def test_transformed_text_can_extend_beyond_its_capture_anchor(self):
        """Capture boxes are anchors, never crop masks for neighbouring text."""
        asset = make_asset(color=(255, 255, 255))
        inst = text_inst(x=70, y=60, w=90, h=36, target="LOCALIZED")
        inst.style_profile = StyleProfil(
            color="#000000",
            transform={"offset_x": 45, "offset_y": -12, "skew_x": 30, "scale_x": 1.3},
        )
        out = np.asarray(scribe.render(asset, make_manifest([inst]), "en"))
        changed = np.any(out != 255, axis=2)
        # The transformed visual ink can cross its source-region edge.  It is
        # still bounded by the output image itself, and the original box
        # remains immutable for capture, cleanup, and semantic identity.
        assert np.any(changed[:, 160:])
        assert inst.adjusted_bbox is None


class TestVerifyMetrics:
    def test_text_similarity_ignores_case_and_punct(self):
        assert verify._text_similarity("Main-Street!", "main street") == 1.0

    def test_text_similarity_empty_recognized_is_zero(self):
        assert verify._text_similarity("HELLO", "") == 0.0

    def test_ssim_identical_is_one(self):
        a = np.random.default_rng(0).uniform(0, 255, (40, 40))
        assert abs(verify._ssim(np, a, a.copy()) - 1.0) < 1e-9

    def test_untouched_background_ring_scores_high(self):
        rng = np.random.default_rng(1)
        src = rng.uniform(0, 255, (200, 300, 3))
        inst = text_inst(target="X")
        score = verify._ring_ssim_score(np, src, src.copy(), inst)
        assert score > 0.99

    def test_disturbed_ring_scores_low(self):
        rng = np.random.default_rng(2)
        src = rng.uniform(0, 255, (200, 300, 3))
        loc = src.copy()
        loc[40:100, 40:180] = 255  # spill well beyond the bbox
        inst = text_inst(target="X")
        assert verify._ring_ssim_score(np, src, loc, inst) < 0.7

    def test_assess_neutral_for_dnt_only(self):
        asset = make_asset()
        qa = verify.assess(asset, make_manifest([text_inst(dnt=True)]), asset)
        assert qa.overall_score == verify.NEUTRAL_SCORE

    def test_assess_neutral_for_excluded_only(self):
        # an excluded region was deliberately erased-not-rendered -- QA
        # must not penalize it as an incomplete/illegible translation
        asset = make_asset()
        qa = verify.assess(
            asset, make_manifest([text_inst(target="X", excluded=True)]), asset
        )
        assert qa.overall_score == verify.NEUTRAL_SCORE
        assert qa.progress["excluded"] == 1
        assert qa.progress["translated"] == 0
