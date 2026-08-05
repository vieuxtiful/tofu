from PIL import Image, ImageDraw

from tofu.layers.inpaint import make_patch


def _base():
    image = Image.new("RGB", (80, 60), "#d8c39c")
    ImageDraw.Draw(image).rectangle((30, 20, 45, 35), fill="#202020")
    return image


def test_brush_patch_is_a_bounded_rgba_crop_with_strategy():
    patch, bbox, strategy = make_patch(
        _base(), points=[(28, 28), (45, 28)], radius=6, hardness=.7,
    )
    assert patch.mode == "RGBA"
    assert strategy == "telea"
    assert bbox["width"] > 0 and bbox["height"] > 0
    assert patch.size == (bbox["width"], bbox["height"])


def test_texture_lasso_selects_texture_context_strategy():
    patch, bbox, strategy = make_patch(
        _base(), polygon=[(25, 18), (50, 18), (50, 38), (25, 38)], mode="texture",
    )
    assert strategy == "navier_stokes"
    assert patch.getbbox() is not None
    # Polygon endpoints are inclusive in the OpenCV raster mask.
    assert bbox == {"x": 25, "y": 18, "width": 26, "height": 21}


def test_clone_uses_aligned_source_pixels_and_records_strategy():
    image = Image.new("RGB", (40, 24), "#ffffff")
    draw = ImageDraw.Draw(image)
    draw.rectangle((4, 8, 10, 14), fill="#e02020")
    patch, bbox, strategy = make_patch(
        image,
        points=[(24, 11)],
        radius=3,
        hardness=1,
        clone_source=(7, 11),
        mode="clone",
    )
    assert strategy == "aligned_clone"
    assert bbox["x"] <= 24 < bbox["x"] + bbox["width"]
    assert bbox["y"] <= 11 < bbox["y"] + bbox["height"]
    # The destination centre samples the red source-anchor centre.
    assert patch.getpixel((24 - bbox["x"], 11 - bbox["y"]))[:3] == (224, 32, 32)


def test_cleanup_opacity_scales_patch_alpha():
    patch, _, _ = make_patch(
        _base(), points=[(36, 28)], radius=4, hardness=1,
        opacity=.5, mode="heal",
    )
    assert 126 <= patch.getpixel((4, 4))[3] <= 128
