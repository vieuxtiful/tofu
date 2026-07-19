## 🍢 cleanse / scribe / verify units on synthetic images
import numpy as np
from PIL import Image

from tofu.core.types import BBox, InstText, Mask, TextManifest
from tofu.layers import cleanse, scribe, verify


def make_asset(w=300, h=200, color=(90, 140, 90)):
    return Image.new("RGB", (w, h), color)


def make_manifest(instances):
    return TextManifest(asset_id="a", total_regions=len(instances),
                        instances=instances, src_lang="en")


def text_inst(x=50, y=50, w=120, h=40, text="HELLO", target=None, dnt=False):
    return InstText(
        id="r1", bounding_box=BBox(x=x, y=y, width=w, height=h),
        segmentation_mask=Mask(
            polygon=[(x, y), (x + w, y), (x + w, y + h), (x, y + h)],
            confidence=0.9,
        ),
        text=text, target_text=target, dnt=dnt,
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

    def test_pixels_outside_bbox_preserved(self):
        asset = make_asset()
        out = cleanse.erase(asset, make_manifest([text_inst()]))
        out_np = np.asarray(out.convert("RGB"))
        src_np = np.asarray(asset)
        # corner far from the region must be bit-identical
        assert (out_np[:20, :20] == src_np[:20, :20]).all()


class TestScribe:
    def test_renders_target_text_pixels(self):
        asset = make_asset(color=(255, 255, 255))
        inst = text_inst(target="HELLO")
        out = scribe.render(asset, make_manifest([inst]), "en")
        out_np = np.asarray(out)
        b = inst.bounding_box
        region = out_np[b.y:b.y + b.height, b.x:b.x + b.width]
        assert (region < 200).any()  # dark glyphs on white

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

    def test_non_image_asset_passes_through(self):
        sentinel = object()
        assert scribe.render(sentinel, make_manifest([]), "en") is sentinel


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
