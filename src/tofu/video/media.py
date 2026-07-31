"""FFmpeg-backed ingest and export helpers.

No shell is used: paths are passed as argv entries and subprocess output is
captured for actionable API errors.
"""
from __future__ import annotations

import json
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional

from .types import VideoManifest


class MediaError(RuntimeError):
    pass


def binaries() -> Dict[str, Optional[str]]:
    return {name: shutil.which(name) for name in ("ffmpeg", "ffprobe")}


def available() -> bool:
    return all(binaries().values())


def _run(argv: List[str]) -> subprocess.CompletedProcess[str]:
    try:
        result = subprocess.run(argv, capture_output=True, text=True, check=False)
    except OSError as exc:
        raise MediaError(str(exc)) from exc
    if result.returncode:
        raise MediaError((result.stderr or result.stdout or "media command failed")[-2000:])
    return result


def probe(path: Path, asset_id: str) -> VideoManifest:
    exe = binaries()["ffprobe"]
    if not exe:
        raise MediaError("ffprobe is not installed or not on PATH")
    payload = json.loads(_run([exe, "-v", "error", "-show_streams", "-show_format",
                               "-print_format", "json", str(path)]).stdout)
    streams = payload.get("streams", [])
    video = next((s for s in streams if s.get("codec_type") == "video"), None)
    if not video:
        raise MediaError("input has no decodable video stream")
    pts: List[float] = []
    process = subprocess.Popen([exe, "-v", "error", "-select_streams", "v:0", "-show_entries",
                                "frame=best_effort_timestamp_time,pts_time", "-of", "csv=p=0", str(path)],
                               stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    assert process.stdout is not None
    for line in process.stdout:
        values = [value.strip() for value in line.strip().split(",") if value.strip() and value.strip() != "N/A"]
        for value in values:
            try: pts.append(float(value)); break
            except ValueError: continue
    stderr = process.stderr.read() if process.stderr else ""
    if process.wait() != 0: raise MediaError(stderr[-2000:] or "ffprobe frame scan failed")
    rate = video.get("avg_frame_rate", "0/0")
    numerator, denominator = (rate.split("/") + ["1"])[:2]
    fps = float(numerator) / float(denominator) if float(denominator) else None
    tags = video.get("tags", {})
    side_data = video.get("side_data_list", [])
    rotation = int(tags.get("rotate", 0) or next((d.get("rotation", 0) for d in side_data if "rotation" in d), 0))
    duration = float(video.get("duration") or payload.get("format", {}).get("duration") or (pts[-1] if pts else 0.0))
    audio = [{k: s.get(k) for k in ("index", "codec_name", "sample_rate", "channels", "channel_layout", "duration")}
             for s in streams if s.get("codec_type") == "audio"]
    return VideoManifest(asset_id=asset_id, width=int(video["width"]), height=int(video["height"]),
                         duration=duration, frame_count=len(pts) or int(video.get("nb_frames") or 0),
                         fps=fps, time_base=video.get("time_base"), codec=video.get("codec_name"),
                         pixel_format=video.get("pix_fmt"), rotation=rotation, audio_streams=audio,
                         frame_pts=pts, color={k: video.get(k) for k in
                         ("color_range", "color_space", "color_transfer", "color_primaries") if video.get(k)})


def make_proxy(source: Path, destination: Path) -> None:
    exe = binaries()["ffmpeg"]
    if not exe:
        raise MediaError("ffmpeg is not installed or not on PATH")
    destination.parent.mkdir(parents=True, exist_ok=True)
    _run([exe, "-y", "-v", "error", "-i", str(source), "-map", "0:v:0", "-map", "0:a?",
          "-vf", "scale='min(1280,iw)':-2", "-c:v", "libx264", "-preset", "veryfast",
          "-crf", "24", "-movflags", "+faststart", "-c:a", "aac", "-b:a", "128k",
          str(destination)])


def mux_rendered_video(rendered: Path, source: Path, destination: Path, *,
                       color: Optional[Dict[str, Any]] = None, rotation: int = 0) -> None:
    """Mux rendered video with source audio, copying audio when MP4 permits."""
    exe = binaries()["ffmpeg"]
    if not exe:
        raise MediaError("ffmpeg is not installed or not on PATH")
    destination.parent.mkdir(parents=True, exist_ok=True)
    common = [exe, "-y", "-v", "error", "-i", str(rendered), "-i", str(source),
              "-map", "0:v:0", "-map", "1:a?", "-c:v", "libx264", "-pix_fmt", "yuv420p",
              "-map_metadata", "1", "-metadata:s:v:0", f"rotate={rotation}",
              "-movflags", "+faststart", "-shortest"]
    option_map = {"color_range":"-color_range","color_space":"-colorspace","color_transfer":"-color_trc","color_primaries":"-color_primaries"}
    for key, option in option_map.items():
        if color and color.get(key): common.extend([option, str(color[key])])
    try:
        _run(common + ["-c:a", "copy", str(destination)])
    except MediaError:
        _run(common + ["-c:a", "aac", "-b:a", "192k", str(destination)])


def encode_vfr_sequence(frame_dir: Path, frame_pts: List[float], destination: Path) -> None:
    """Encode numbered PNG frames using their actual presentation deltas."""
    exe = binaries()["ffmpeg"]
    if not exe: raise MediaError("ffmpeg is not installed or not on PATH")
    frames = sorted(frame_dir.glob("*.png"))
    if not frames: raise MediaError("render produced no frames")
    destination.parent.mkdir(parents=True, exist_ok=True)
    durations = [max(.001, frame_pts[i+1]-frame_pts[i]) for i in range(min(len(frames)-1, len(frame_pts)-1))]
    fallback = durations[-1] if durations else 1/30
    lines: List[str] = []
    for index, frame in enumerate(frames):
        escaped = str(frame.resolve()).replace("'", "'\\''")
        lines.extend([f"file '{escaped}'", f"duration {durations[index] if index < len(durations) else fallback:.9f}"])
    escaped = str(frames[-1].resolve()).replace("'", "'\\''"); lines.append(f"file '{escaped}'")
    listing = frame_dir / "frames.ffconcat"
    listing.write_text("ffconcat version 1.0\n" + "\n".join(lines) + "\n", encoding="utf-8")
    _run([exe, "-y", "-v", "error", "-f", "concat", "-safe", "0", "-i", str(listing),
          "-vsync", "vfr", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(destination)])


def mux_preview_audio(rendered: Path, source: Path, destination: Path, start_seconds: float, duration: float) -> None:
    exe = binaries()["ffmpeg"]
    if not exe: raise MediaError("ffmpeg is not installed or not on PATH")
    base = [exe, "-y", "-v", "error", "-i", str(rendered), "-ss", f"{max(0,start_seconds):.9f}",
            "-t", f"{max(.001,duration):.9f}", "-i", str(source), "-map", "0:v:0", "-map", "1:a?",
            "-c:v", "copy", "-movflags", "+faststart", "-shortest"]
    try: _run(base + ["-c:a", "copy", str(destination)])
    except MediaError: _run(base + ["-c:a", "aac", "-b:a", "160k", str(destination)])


def verify_output(path: Path, *, expected_width: int, expected_height: int,
                  expected_duration: float, max_drift: float = .02) -> Dict[str, Any]:
    exe = binaries()["ffprobe"]
    if not exe: raise MediaError("ffprobe is not installed or not on PATH")
    payload = json.loads(_run([exe, "-v", "error", "-show_streams", "-show_format", "-of", "json", str(path)]).stdout)
    streams = payload.get("streams", []); video = next((item for item in streams if item.get("codec_type")=="video"), None)
    if not video: raise MediaError("encoded artifact has no video stream")
    if int(video.get("width",0)) != expected_width or int(video.get("height",0)) != expected_height:
        raise MediaError("encoded dimensions do not match the source")
    duration = float(video.get("duration") or payload.get("format",{}).get("duration") or 0)
    drift = abs(duration-expected_duration)
    if drift > max_drift: raise MediaError(f"encoded duration drift {drift:.6f}s exceeds {max_drift:.6f}s")
    audio = [item for item in streams if item.get("codec_type")=="audio"]
    return {"duration": duration, "duration_drift": drift, "video_codec": video.get("codec_name"),
            "pixel_format": video.get("pix_fmt"), "audio_streams": len(audio),
            "color": {key: video.get(key) for key in ("color_range","color_space","color_transfer","color_primaries") if video.get(key)}}
