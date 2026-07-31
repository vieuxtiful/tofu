from pathlib import Path

import cv2
import numpy as np

from tofu.video.compositor import WARMUP_FRAMES, compose


def _video(path: Path, frames: int = 3) -> None:
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 10, (96, 64))
    assert writer.isOpened()
    for _ in range(frames):
        image = np.full((64, 96, 3), 220, np.uint8)
        cv2.putText(image, "OLD", (12, 36), cv2.FONT_HERSHEY_SIMPLEX, .7, (0, 0, 0), 2)
        writer.write(image)
    writer.release()


def _moving_video(path: Path, frames: int = 40) -> None:
    """A near-static textured background with mild per-frame noise, plus text.

    Both properties are needed to reach the code paths that diverge:

    - the background must be STABLE enough that the compositor's temporal
      history passes its mean-abs-difference test and the median fill is
      actually used (a fast-moving background fails that test on every frame,
      so the history never fires and both ranges fall through to inpaint);
    - it must still VARY frame to frame, or the median of the history equals
      the current crop and using it changes no pixels.
    """
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 10, (96, 64))
    assert writer.isOpened()
    rng = np.random.default_rng(20260730)
    texture = rng.integers(90, 190, (64, 96, 3), dtype=np.uint8).astype(np.int16)
    for index in range(frames):
        noise = rng.integers(-9, 10, texture.shape)
        image = np.clip(texture + noise, 0, 255).astype(np.uint8)
        cv2.putText(image, "OLD", (14, 40), cv2.FONT_HERSHEY_SIMPLEX, .6, (0, 0, 0), 2)
        writer.write(image)
    writer.release()


def _fixture(frames: int = 40):
    manifest = {"frame_count": frames, "fps": 10, "width": 96, "height": 64}
    tracks = [{"id": "t1", "target_text": "NEW", "status": "translated",
               "style": {"color": "#ff0000"}}]
    ## a glyph polygon strictly inside the bbox. without one the compositor sets
    ## mask = region_mask and then zeroes the motion mask by that same region,
    ## so the occlusion path can never fire and half the divergence hides.
    polygon = [[14, 26], [56, 26], [56, 44], [14, 44]]
    observations = [{"id": f"o{i}", "track_id": "t1", "frame_index": i,
                     "bbox": {"x": 8, "y": 20, "width": 56, "height": 30},
                     "glyph_mask_polygon": polygon}
                    for i in range(frames)]
    return manifest, tracks, observations


def test_preview_and_export_frames_are_byte_identical(tmp_path):
    """A preview is a window onto the export, not a different render.

    Rendering [0,39] and rendering [12,27] must agree exactly on the frames they
    share. They did not: the temporal background history and the previous-frame
    motion mask were both warmed from start_frame, so a preview beginning
    mid-clip erased and restored differently than the export covering the same
    frame -- the reviewer approved pixels the export would never produce.
    """
    source = tmp_path / "source.mp4"
    _moving_video(source)
    manifest, tracks, observations = _fixture()

    export_dir, preview_dir = tmp_path / "export", tmp_path / "preview"
    compose(source, tmp_path / "e.mp4", manifest, tracks, observations, [],
            start_frame=0, end_frame=39, frame_dir=export_dir)
    compose(source, tmp_path / "p.mp4", manifest, tracks, observations, [],
            start_frame=12, end_frame=27, frame_dir=preview_dir)

    divergent = []
    for index in range(12, 28):
        exported = cv2.imread(str(export_dir / f"{index:08d}.png"))
        previewed = cv2.imread(str(preview_dir / f"{index - 12:08d}.png"))
        assert exported is not None and previewed is not None
        if not np.array_equal(exported, previewed):
            divergent.append(index)
    assert not divergent, f"preview and export disagree on frames {divergent}"


def _diverging_frames(tmp_path, warmup):
    source = tmp_path / f"source-{warmup}.mp4"
    _moving_video(source)
    manifest, tracks, observations = _fixture()
    export_dir, preview_dir = tmp_path / f"e{warmup}", tmp_path / f"p{warmup}"
    compose(source, tmp_path / f"e{warmup}.mp4", manifest, tracks, observations, [],
            start_frame=0, end_frame=39, frame_dir=export_dir, warmup_frames=warmup)
    compose(source, tmp_path / f"p{warmup}.mp4", manifest, tracks, observations, [],
            start_frame=12, end_frame=27, frame_dir=preview_dir, warmup_frames=warmup)
    return [index for index in range(12, 28)
            if not np.array_equal(cv2.imread(str(export_dir / f"{index:08d}.png")),
                                  cv2.imread(str(preview_dir / f"{index - 12:08d}.png")))]


def test_without_warmup_the_first_preview_frames_diverge(tmp_path):
    """Proves the parity test above has teeth.

    Temporal reconstruction and the motion mask both look backwards, so a range
    that starts cold reconstructs its opening frames differently. The divergence
    window is exactly the depth of that lookback -- which is why warm-up is
    sized from it rather than picked.
    """
    divergent = _diverging_frames(tmp_path, warmup=0)
    assert divergent, "expected a cold start to diverge; the parity test would be vacuous"
    assert max(divergent) - 12 < WARMUP_FRAMES, "divergence outlasted the warm-up window"
    assert not _diverging_frames(tmp_path, warmup=WARMUP_FRAMES)


def test_compositor_replaces_track_region(tmp_path):
    source, output = tmp_path / "source.mp4", tmp_path / "output.mp4"
    _video(source)
    manifest = {"frame_count": 3, "fps": 10, "width": 96, "height": 64}
    tracks = [{"id": "t1", "target_text": "NEW", "status": "translated", "style": {"color": "#ff0000"}}]
    observations = [{"id": f"o{i}", "track_id": "t1", "frame_index": i,
                     "bbox": {"x": 8, "y": 16, "width": 55, "height": 28}}
                    for i in range(3)]
    compose(source, output, manifest, tracks, observations, [])
    capture = cv2.VideoCapture(str(output)); ok, rendered = capture.read(); capture.release()
    assert ok
    assert not np.array_equal(rendered, cv2.VideoCapture(str(source)).read()[1])
    assert rendered[16:44, 8:63].std() > 0
