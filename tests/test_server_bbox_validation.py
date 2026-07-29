"""HTTP-boundary geometry guards for manual capture and OCR crops."""

import sys

import pytest
from fastapi import HTTPException
from PIL import Image


@pytest.fixture
def server_main(tmp_path, monkeypatch):
    # The backend imports ``db`` as a top-level server-local module.
    sys.path.insert(0, "server")
    try:
        import main
    finally:
        sys.path.pop(0)
    monkeypatch.setattr(main, "UPLOAD_DIR", tmp_path)
    Image.new("RGB", (100, 60), "white").save(tmp_path / "asset.png")
    return main


def test_bbox_is_clipped_to_asset_bounds(server_main):
    bbox = server_main._normalized_asset_bbox("asset", {
        "x": -10, "y": 40, "width": 140, "height": 30,
    })

    assert (bbox.x, bbox.y, bbox.width, bbox.height) == (0, 40, 100, 20)


@pytest.mark.parametrize("raw, detail", [
    ({"x": 0, "y": 0, "width": 0, "height": 10}, "positive"),
    ({"x": 101, "y": 0, "width": 10, "height": 10}, "intersect"),
    ({"x": 0, "y": 61, "width": 10, "height": 10}, "intersect"),
    ({"x": 0, "y": 0, "width": "wide", "height": 10}, "integer"),
])
def test_invalid_or_outside_bbox_is_a_clear_422(server_main, raw, detail):
    with pytest.raises(HTTPException) as exc:
        server_main._normalized_asset_bbox("asset", raw)

    assert exc.value.status_code == 422
    assert detail in exc.value.detail
