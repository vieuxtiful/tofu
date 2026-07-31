## 🍢 video_fixtures — mise en place for the video pipeline
## vieuxtiful
"""
Mise en place: everything the video tests need, cut to size before the
cooking starts.

Generates synthetic clips into tests/fixtures/video/ with EXACT ground
truth. We author the timeline, so cut frames, burned-in strings, frame
counts and timestamps are known rather than measured -- the same bargain
make_fixtures.py strikes for images.

Each fixture writes {name}.mp4 + {name}.gt.json:
  {"name", "kind", "width", "height", "fps", "frame_count", "duration",
   "cuts": [frame, ...], "transitions": [{"kind","start","end"}, ...],
   "text": [{"frame_start","frame_end","x","y","text"}, ...],
   "audio": {"streams": n, "offset_seconds": f} | None,
   "expect": {...}}          # per-kind assertions the tests read

Fixture matrix (one failure mode each):
  cfr-30         constant 30fps baseline        → the control
  vfr-mixed      three different frame periods  → PTS-tick assertions
  rotated-90     rotate=90 side data            → export metadata survival
  hdr-tagged     bt2020 / smpte2084 tags        → HDR metadata survival
  audio-delayed  audio offset 250ms             → A/V drift verification
  hard-cuts      4 abrupt scene changes         → shot-detection precision
  fade-dissolve  fade to black + 1s dissolve    → cut vs. gradual transition
  dense-text     12 simultaneous text regions   → OCR/tracking worst case
  sparse-text    1 region, long dwell           → OCR trigger economy
  4k-30min       3840x2160, 30 minutes          → memory bound (opt-in)

Everything is deterministic: fixed lavfi sources, fixed seeds, no
timestamps in the output, `-fflags +bitexact` so the muxer writes no
encoder string. Regenerating gives byte-identical files.

The clips are gitignored; only this generator is committed. 4k-30min is
opt-in (--include-heavy) because it is ~2GB and takes minutes.

usage (from repo root):
  .venv/Scripts/python scripts/video_fixtures.py
  .venv/Scripts/python scripts/video_fixtures.py --only hard-cuts --force
  .venv/Scripts/python scripts/video_fixtures.py --include-heavy
"""

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tofu.layers.fonts import pantry  # noqa: E402

OUT = ROOT / "tests" / "fixtures" / "video"

## Small enough that a full-decode test stays quick, large enough that
## drawtext regions survive h264 at crf 18 well enough for OCR.
WIDTH, HEIGHT, FPS = 640, 360, 30

## bitexact strips the encoder/creation tags that would otherwise make two
## runs of this script differ byte-for-byte and defeat fixture caching.
BITEXACT = ["-fflags", "+bitexact", "-flags:v", "+bitexact"]


class FixtureError(RuntimeError):
    pass


def _ffmpeg() -> str:
    exe = shutil.which("ffmpeg")
    if not exe:
        raise FixtureError("ffmpeg is not installed or not on PATH")
    return exe


def _run(args: List[str]) -> None:
    """argv only, never shell=True -- fixture names reach the command line."""
    proc = subprocess.run([_ffmpeg(), "-y", "-v", "error", "-nostdin", *args],
                          capture_output=True, text=True)
    if proc.returncode != 0:
        tail = (proc.stderr or "").strip().splitlines()[-6:]
        raise FixtureError("ffmpeg failed:\n  " + "\n  ".join(tail))


def _fontfile() -> str:
    """A real font path for drawtext, from the same pantry the pipeline uses.

    drawtext needs a file, not a family. Resolving through pantry() means a
    machine that can run ToFU at all can generate fixtures -- no hardcoded
    C:\\Windows\\Fonts, no separate answer to "where do fonts live".
    """
    root = pantry()
    if not root:
        raise FixtureError("no font library found; fixtures need one for drawtext")
    for pattern in ("arial.ttf", "Arial.ttf", "DejaVuSans.ttf", "*.ttf"):
        match = next(Path(root).rglob(pattern), None)
        if match:
            ## drawtext parses ':' as an option separator and '\' as an escape.
            return str(match).replace("\\", "/").replace(":", "\\:")
    raise FixtureError(f"no .ttf under {root}")


def _draw(text: str, x: int, y: int, size: int = 28,
          start: Optional[float] = None, end: Optional[float] = None) -> str:
    clause = (f"drawtext=fontfile='{_fontfile()}':text='{text}'"
              f":x={x}:y={y}:fontsize={size}:fontcolor=white:box=1:boxcolor=black@0.75:boxborderw=6")
    if start is not None and end is not None:
        clause += f":enable='between(t,{start},{end})'"
    return clause


def _encode(dest: Path, inputs: List[str], filtergraph: Optional[str],
            extra: Optional[List[str]] = None, *, crf: int = 18) -> None:
    args = [*inputs]
    if filtergraph:
        args += ["-vf", filtergraph]
    args += ["-c:v", "libx264", "-preset", "veryfast", "-crf", str(crf),
             "-pix_fmt", "yuv420p", "-movflags", "+faststart", *BITEXACT]
    args += list(extra or [])
    args += [str(dest)]
    _run(args)


# --- individual fixtures ---------------------------------------------------


def _cfr_30(dest: Path) -> Dict[str, Any]:
    """Constant 30fps, one static caption. The control every other clip is read against."""
    seconds = 4
    _encode(dest,
            ["-f", "lavfi", "-i", f"testsrc2=size={WIDTH}x{HEIGHT}:rate={FPS}:duration={seconds}"],
            _draw("CONTROL CAPTION", 60, 240))
    return {"fps": FPS, "frame_count": FPS * seconds, "duration": float(seconds), "cuts": [],
            "text": [{"frame_start": 0, "frame_end": FPS * seconds - 1, "x": 60, "y": 240,
                      "text": "CONTROL CAPTION"}],
            "expect": {"constant_frame_rate": True, "max_pts_jitter_seconds": 1e-6}}


def _vfr_mixed(dest: Path) -> Dict[str, Any]:
    """Three real frame rates spliced end to end: 30fps, 15fps, 60fps.

    Built by concatenating separately-encoded segments rather than by
    remapping one CFR source with setpts. Measured, the setpts route does not
    survive: lavfi hands the filter a 1/rate timebase, so the fast segment's
    1/60s spacing quantizes onto a single tick and the file ships pairs of
    frames sharing one PTS -- and settb, -fps_mode and -video_track_timescale
    together did not fix it. Splicing is also the honest fixture: this is how
    real VFR arises, from sources joined at different rates.

    Everything is -c copy, so the concat cannot resample the timestamps it
    exists to preserve. Segment boundaries land on frames 60 and 90.
    """
    rates, seconds = (FPS, FPS // 2, FPS * 2), 2
    segments: List[Path] = []
    listing = dest.with_name(dest.stem + "-segments.txt")
    try:
        for rate in rates:
            segment = dest.with_name(f"{dest.stem}-{rate}fps.mp4")
            _encode(segment,
                    ["-f", "lavfi", "-i", f"testsrc2=size={WIDTH}x{HEIGHT}:rate={rate}:duration={seconds}"],
                    _draw(f"{rate} FPS", 40, 40),
                    ["-video_track_timescale", "90000"])
            segments.append(segment)
        listing.write_text("".join(f"file '{p.name}'\n" for p in segments), encoding="utf-8")
        _run(["-f", "concat", "-safe", "0", "-i", str(listing), "-c", "copy",
              "-fps_mode", "passthrough", "-video_track_timescale", "90000",
              "-movflags", "+faststart", *BITEXACT, str(dest)])
    finally:
        listing.unlink(missing_ok=True)
        for segment in segments:
            segment.unlink(missing_ok=True)
    counts = [rate * seconds for rate in rates]
    return {"fps": None, "frame_count": sum(counts),
            "duration": float(seconds * len(rates)), "cuts": [],
            "expect": {"constant_frame_rate": False,
                       "distinct_frame_periods_at_least": 3,
                       "expected_periods_seconds": [round(1 / rate, 6) for rate in rates],
                       "segment_boundary_frames": [counts[0], counts[0] + counts[1]],
                       "monotonic_pts": True,
                       "pts_tolerance_ticks": 1}}


def _rotated_90(dest: Path) -> Dict[str, Any]:
    """Portrait display metadata on a landscape stream -- rotation must survive export.

    Two passes on purpose. ffmpeg 7+ dropped the writable `rotate` stream tag
    in favour of Display Matrix side data, and `-display_rotation` is an INPUT
    option -- so the rotation can only be stamped while reading an existing
    file. Encode, then remux with `-c copy`: the pixels are untouched and only
    the side data differs.
    """
    intermediate = dest.with_name(dest.stem + "-unrotated.mp4")
    try:
        _encode(intermediate,
                ["-f", "lavfi", "-i", f"testsrc2=size={WIDTH}x{HEIGHT}:rate={FPS}:duration=3"],
                _draw("ROTATED", 40, 40))
        _run(["-display_rotation", "90", "-i", str(intermediate), "-c", "copy",
              "-movflags", "+faststart", *BITEXACT, str(dest)])
    finally:
        intermediate.unlink(missing_ok=True)
    return {"fps": FPS, "frame_count": FPS * 3, "duration": 3.0, "cuts": [],
            "expect": {"rotation": 90}}


def _hdr_tagged(dest: Path) -> Dict[str, Any]:
    """BT.2020 / PQ tags. MP4+H.264 keeps these; the export must not drop them silently.

    The colour description is forced into the H.264 VUI with h264_metadata
    rather than left to `-color_primaries`/`-color_trc`. Measured: those two
    reach the encoder but never the bitstream (primaries and transfer probe
    back as 'unknown' at 8- and 10-bit alike), while `-colorspace` does land.
    A fixture that only carries one of the three would let an export drop the
    other two and still pass.
    """
    _encode(dest,
            ["-f", "lavfi", "-i", f"gradients=size={WIDTH}x{HEIGHT}:rate={FPS}:duration=3:nb_colors=4"],
            _draw("HDR", 40, 40),
            ["-colorspace", "bt2020nc", "-color_range", "tv",
             "-bsf:v", "h264_metadata=colour_primaries=9:transfer_characteristics=16"
                       ":matrix_coefficients=9:video_full_range_flag=0"])
    return {"fps": FPS, "frame_count": FPS * 3, "duration": 3.0, "cuts": [],
            "expect": {"color_primaries": "bt2020", "color_transfer": "smpte2084",
                       "color_space": "bt2020nc", "color_range": "tv"}}


def _audio_delayed(dest: Path) -> Dict[str, Any]:
    """Audio starting 250ms late. Total duration hides this; end-of-file drift does not."""
    offset = 0.25
    _run(["-f", "lavfi", "-i", f"testsrc2=size={WIDTH}x{HEIGHT}:rate={FPS}:duration=5",
          "-itsoffset", str(offset), "-f", "lavfi", "-i", "sine=frequency=440:duration=5",
          "-vf", _draw("A/V", 40, 40),
          "-c:v", "libx264", "-preset", "veryfast", "-crf", "18", "-pix_fmt", "yuv420p",
          "-c:a", "aac", "-b:a", "128k", "-movflags", "+faststart", *BITEXACT, str(dest)])
    return {"fps": FPS, "frame_count": FPS * 5, "duration": 5.0, "cuts": [],
            "audio": {"streams": 1, "offset_seconds": offset},
            "expect": {"max_av_drift_seconds": 0.020}}


def _hard_cuts(dest: Path) -> Dict[str, Any]:
    """Five 1-second segments, abrupt boundaries at exact frame indices.

    Ground truth for shot-detection precision/recall: any boundary the
    detector reports away from these frames is a false positive, full stop.
    """
    segments = ["red", "blue", "green", "0x202020", "white"]
    seconds = 1
    inputs: List[str] = []
    for colour in segments:
        inputs += ["-f", "lavfi", "-i",
                   f"color=c={colour}:size={WIDTH}x{HEIGHT}:rate={FPS}:duration={seconds}"]
    chain = "".join(f"[{i}:v]" for i in range(len(segments)))
    graph = (f"{chain}concat=n={len(segments)}:v=1:a=0[cat];"
             f"[cat]{_draw('CUT %{eif\\:floor(t)\\:d}', 40, 40)}[out]")
    _run([*inputs, "-filter_complex", graph, "-map", "[out]",
          "-c:v", "libx264", "-preset", "veryfast", "-crf", "18", "-pix_fmt", "yuv420p",
          "-movflags", "+faststart", *BITEXACT, str(dest)])
    cuts = [FPS * i for i in range(1, len(segments))]
    return {"fps": FPS, "frame_count": FPS * seconds * len(segments),
            "duration": float(seconds * len(segments)), "cuts": cuts,
            "expect": {"cut_frames": cuts, "cut_tolerance_frames": 1,
                       "max_false_positives": 0}}


def _fade_dissolve(dest: Path) -> Dict[str, Any]:
    """A 1s dissolve between two sources. Gradual, so it must NOT read as a hard cut."""
    _run(["-f", "lavfi", "-i", f"color=c=red:size={WIDTH}x{HEIGHT}:rate={FPS}:duration=3",
          "-f", "lavfi", "-i", f"color=c=blue:size={WIDTH}x{HEIGHT}:rate={FPS}:duration=3",
          "-filter_complex",
          f"[0:v][1:v]xfade=transition=fade:duration=1:offset=2[x];[x]{_draw('DISSOLVE', 40, 40)}[out]",
          "-map", "[out]", "-c:v", "libx264", "-preset", "veryfast", "-crf", "18",
          "-pix_fmt", "yuv420p", "-movflags", "+faststart", *BITEXACT, str(dest)])
    return {"fps": FPS, "frame_count": FPS * 5, "duration": 5.0, "cuts": [],
            "transitions": [{"kind": "dissolve", "start": FPS * 2, "end": FPS * 3}],
            "expect": {"cut_frames": [], "gradual_frames": [FPS * 2, FPS * 3],
                       "max_false_positives": 0}}


def _dense_text(dest: Path) -> Dict[str, Any]:
    """Twelve simultaneous regions over moving texture: the OCR/tracking worst case."""
    regions: List[Dict[str, Any]] = []
    clauses: List[str] = []
    for row in range(4):
        for column in range(3):
            x, y = 30 + column * 200, 30 + row * 85
            text = f"REGION {row * 3 + column + 1:02d}"
            regions.append({"frame_start": 0, "frame_end": FPS * 4 - 1, "x": x, "y": y, "text": text})
            clauses.append(_draw(text, x, y, size=20))
    _encode(dest,
            ["-f", "lavfi", "-i", f"testsrc2=size={WIDTH}x{HEIGHT}:rate={FPS}:duration=4"],
            ",".join(clauses))
    return {"fps": FPS, "frame_count": FPS * 4, "duration": 4.0, "cuts": [], "text": regions,
            "expect": {"region_count": len(regions), "track_count": len(regions)}}


def _sparse_text(dest: Path) -> Dict[str, Any]:
    """One region held for 8 seconds: re-OCR should fire a handful of times, not 240."""
    seconds = 8
    _encode(dest,
            ["-f", "lavfi", "-i", f"color=c=0x303030:size={WIDTH}x{HEIGHT}:rate={FPS}:duration={seconds}"],
            _draw("SINGLE STATIC SIGN", 120, 160))
    return {"fps": FPS, "frame_count": FPS * seconds, "duration": float(seconds), "cuts": [],
            "text": [{"frame_start": 0, "frame_end": FPS * seconds - 1, "x": 120, "y": 160,
                      "text": "SINGLE STATIC SIGN"}],
            "expect": {"region_count": 1, "track_count": 1, "max_ocr_invocations": 12}}


def _4k_30min(dest: Path) -> Dict[str, Any]:
    """3840x2160 for 30 minutes. Not a correctness fixture -- the memory bound.

    Peak RSS across this must not scale with duration; that is the whole
    claim bounded chunking makes.
    """
    seconds = 30 * 60
    _encode(dest,
            ["-f", "lavfi", "-i", f"testsrc2=size=3840x2160:rate={FPS}:duration={seconds}"],
            _draw("4K STRESS", 200, 1000, size=96),
            ["-g", "120"], crf=30)
    return {"width": 3840, "height": 2160, "fps": FPS, "frame_count": FPS * seconds,
            "duration": float(seconds), "cuts": [],
            "expect": {"memory_slope_tolerance_mb_per_minute": 8.0}}


BUILDERS = {
    "cfr-30": _cfr_30,
    "vfr-mixed": _vfr_mixed,
    "rotated-90": _rotated_90,
    "hdr-tagged": _hdr_tagged,
    "audio-delayed": _audio_delayed,
    "hard-cuts": _hard_cuts,
    "fade-dissolve": _fade_dissolve,
    "dense-text": _dense_text,
    "sparse-text": _sparse_text,
}
HEAVY = {"4k-30min": _4k_30min}
ALL = {**BUILDERS, **HEAVY}


# --- public API ------------------------------------------------------------


def prep(kind: str, dest: Optional[Path] = None, *, force: bool = False) -> Path:
    """Build one fixture and its ground truth; return the clip path.

    Cached by default: tests call this freely, and only the first call in a
    working tree pays the encode.
    """
    if kind not in ALL:
        raise FixtureError(f"unknown fixture {kind!r}; known: {', '.join(sorted(ALL))}")
    destination = Path(dest) if dest else OUT / f"{kind}.mp4"
    truth = destination.with_suffix(".gt.json")
    if destination.is_file() and truth.is_file() and not force:
        return destination
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.unlink(missing_ok=True)
    ground = ALL[kind](destination)
    payload = {"name": kind, "kind": kind, "width": WIDTH, "height": HEIGHT,
               "cuts": [], "transitions": [], "text": [], "audio": None, **ground}
    truth.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return destination


def ground_truth(kind: str) -> Dict[str, Any]:
    """Ground truth for a fixture, building it first if it is missing."""
    return json.loads(prep(kind).with_suffix(".gt.json").read_text(encoding="utf-8"))


def main() -> int:
    parser = argparse.ArgumentParser(description="generate synthetic video fixtures")
    parser.add_argument("--only", action="append", choices=sorted(ALL),
                        help="build just this fixture (repeatable)")
    parser.add_argument("--include-heavy", action="store_true",
                        help="also build 4k-30min (~2GB, several minutes)")
    parser.add_argument("--force", action="store_true", help="rebuild even if cached")
    args = parser.parse_args()

    wanted = args.only or (list(BUILDERS) + (list(HEAVY) if args.include_heavy else []))
    failures = 0
    for kind in wanted:
        try:
            path = prep(kind, force=args.force)
            size = path.stat().st_size / 1024
            print(f"  ok  {kind:<14} {size:>9.1f} KiB  {path}")
        except FixtureError as exc:
            failures += 1
            print(f"  FAIL {kind:<14} {exc}", file=sys.stderr)
    print(f"\n{len(wanted) - failures}/{len(wanted)} fixtures in {OUT}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
