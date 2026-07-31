import pytest
from PIL import Image

from tofu.video.media import available, binaries, encode_vfr_sequence


def test_media_capability_requires_both_binaries():
    found = binaries()
    assert set(found) == {"ffmpeg", "ffprobe"}
    assert available() is bool(found["ffmpeg"] and found["ffprobe"])


def test_vfr_sequence_encoder_accepts_irregular_pts(tmp_path):
    if not binaries()["ffmpeg"]:
        pytest.skip("ffmpeg is unavailable")
    frames = tmp_path / "frames"; frames.mkdir()
    for index, color in enumerate(("red", "green", "blue")):
        Image.new("RGB", (32, 24), color).save(frames / f"{index:08d}.png")
    output = tmp_path / "vfr.mp4"
    encode_vfr_sequence(frames, [0.0, 0.04, 0.15], output)
    assert output.is_file() and output.stat().st_size > 0
