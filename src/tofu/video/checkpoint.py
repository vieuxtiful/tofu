"""Durable tracker state for resumable analysis.

A braise can come off the heat and go back on. Everything the analysis loop
needs to continue is either in this blob or re-derivable by re-decoding a short
overlap.

Nothing here is a full frame. The one piece of large state -- the grayscale
frame optical flow needs -- is deliberately NOT persisted. A 4K luma plane is
~8MB, and a two-hour clip at chunk_size=240 produces ~1800 checkpoints, so
storing it would cost gigabytes to protect a few hundred lines of tracker.
Re-decoding twelve frames reproduces it exactly, costs milliseconds, and unlike
a stored copy it *proves* the decoder resynchronized on the same pixels.

`previous_hist` (32 floats) IS persisted -- not because deriving it is
expensive, but because it is the cheap decode-determinism token: re-deriving it
from the replay and comparing is exactly the statistic the shot detector runs
on, so agreement proves the resumed decode would not invent a cut.
"""
from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence
from tofu.core.types import ImageLike

CHECKPOINT_VERSION = 1

## Twelve frames of overlap. The binding constraint is decoder
## resynchronization, not tracker state: seeking a long-GOP H.264/HEVC stream
## lands on a preceding keyframe and decodes forward, and with B-frame
## reordering and open GOP, frame-accurate seek is not guaranteed across
## containers. H.264 Annex A caps max_num_ref_frames at 16 and High-profile
## level-4.x encoders use <=4 in practice, so twelve covers the realistic
## reference window with margin. The tracker itself needs only one prior frame
## (pyramidal Lucas-Kanade), and twelve is well inside its own decay horizon --
## confidence falls 0.03/frame from 1.0, so ~15 frames to the re-OCR trigger,
## and a track retires after 2 missed keyframes. Cost: 0.4s of video at 30fps.
OVERLAP_FRAMES = 12

## A corrupt checkpoint is likelier than a broken decoder, and rolling back
## costs only one chunk of rework -- but not indefinitely.
RESUME_ROLLBACK_LIMIT = 2

## Perceptual, not byte-exact: the same frame decoded via a different seek path
## can differ by a least-significant bit on some FFmpeg builds, and demanding
## bit-equality would make resume fail spuriously. 5/64 is the conventional
## near-duplicate threshold for a 64-bit difference hash.
HAMMING_TOLERANCE = 5

## Bhattacharyya distance between a track's stored crop histogram and the same
## box re-sampled from the replayed frame. Loose enough to survive re-encode
## noise, tight enough to catch a genuinely different frame.
APPEARANCE_TOLERANCE = .35
APPEARANCE_QUORUM = .8


class ResumeMismatch(RuntimeError):
    """The replayed overlap does not match what the checkpoint recorded."""

    def __init__(self, reason: str, evidence: Optional[Dict[str, Any]] = None) -> None:
        super().__init__(reason)
        self.reason = reason
        self.evidence = evidence or {}


@dataclass
class TrackSnapshot:
    """One active track's carry-over state at a chunk boundary."""
    track: Dict[str, Any]                        # asdict(TextTrack), incl. consensus_state
    last_observation: Dict[str, Any]             # asdict(TrackObservation) at next_frame - 1
    missed: int = 0
    appearance: Optional[List[float]] = None     # 32-bin luma histogram of the last OCR crop
    ## unused at v1. the escape hatch for a learned re-identification embedding,
    ## which is the one descriptor that would outgrow a JSON column.
    descriptor_blob_path: Optional[str] = None


@dataclass
class TrackerCheckpoint:
    """Everything needed to resume, minus what the overlap replay rebuilds.

    Note what is absent: RETIRED tracks. Their consensus accumulators live on
    the video_tracks row, which is already written every chunk and already keyed
    per track. That one decision bounds this blob at O(tracks on screen) instead
    of O(tracks in the clip), and is what makes the incremental chunk write
    coherent -- the same change solves both.
    """
    job_id: str = ""
    chunk_index: int = -1
    next_frame: int = 0                          # first frame NOT yet analyzed
    shot_index: int = 0
    force_next: bool = False
    last_cut_frame: int = 0                      # newest boundary: the free restart point
    previous_hist: List[float] = field(default_factory=list)
    overlap_fingerprints: List[str] = field(default_factory=list)
    overlap_frames: List[int] = field(default_factory=list)
    overlap_cuts: List[int] = field(default_factory=list)
    active: Dict[str, TrackSnapshot] = field(default_factory=dict)
    params: Dict[str, Any] = field(default_factory=dict)
    source_fingerprint: str = ""
    version: int = CHECKPOINT_VERSION
    analyzer_revision: str = ""
    created_at: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Optional[Dict[str, Any]]) -> Optional["TrackerCheckpoint"]:
        if not data:
            return None
        payload = dict(data)
        payload["active"] = {
            track_id: TrackSnapshot(**snapshot) if isinstance(snapshot, dict) else snapshot
            for track_id, snapshot in (payload.get("active") or {}).items()
        }
        known = cls.__dataclass_fields__
        return cls(**{key: value for key, value in payload.items() if key in known})


def source_fingerprint(path: Any) -> str:
    """Identify the file a checkpoint belongs to.

    Path, size and mtime rather than a content hash: hashing a multi-gigabyte
    source on every resume would cost more than the analysis we are trying to
    save, and this is enough to catch the case that matters -- a checkpoint
    being replayed against a different or re-encoded file.
    """
    file = Path(path)
    try:
        stat = file.stat()
        material = f"{file.resolve()}|{stat.st_size}|{stat.st_mtime_ns}"
    except OSError:
        material = str(file)
    return hashlib.sha256(material.encode()).hexdigest()[:32]


def frame_fingerprint(gray: ImageLike) -> str:
    """64-bit difference hash (Zauner 2010) of a grayscale frame.

    Rows of adjacent-pixel comparisons: invariant to overall brightness and to
    the small absolute differences a re-decode can introduce, while still
    distinguishing one frame of a shot from its neighbour.
    """
    import cv2
    small = cv2.resize(gray, (9, 8), interpolation=cv2.INTER_AREA)
    value = 0
    for bit in (small[:, 1:] > small[:, :-1]).reshape(-1):
        value = (value << 1) | int(bit)
    return f"{value:016x}"


def fingerprint_distance(left: str, right: str) -> int:
    try:
        return bin(int(left, 16) ^ int(right, 16)).count("1")
    except (TypeError, ValueError):
        return 64


def histogram_distance(left: Sequence[float], right: Sequence[float]) -> float:
    """Bhattacharyya distance -- the same metric the shot detector compares on."""
    import cv2
    import numpy as np
    if not len(left) or not len(right) or len(left) != len(right):
        return 1.0
    a = np.asarray(left, dtype=np.float32).reshape(-1, 1)
    b = np.asarray(right, dtype=np.float32).reshape(-1, 1)
    return float(cv2.compareHist(a, b, cv2.HISTCMP_BHATTACHARYYA))


class OverlapReconciler:
    """Decides whether a resumed decode may be trusted to continue a checkpoint.

    Four checks, all of which must pass. They are deliberately different in
    kind: two about the pixels (are these the same frames), one about the shot
    state (would this decode invent a boundary), one about the tracks (is the
    state we are about to carry forward still describing what is on screen).

    Nothing here ever silently continues. A resume that quietly degrades
    produces track-identity churn, which surfaces to the viewer as duplicated
    or flickering subtitles -- far worse than paying for one chunk of rework.
    """

    def __init__(self, checkpoint: TrackerCheckpoint, warm_from: int, *,
                 cut_threshold: float = .45) -> None:
        self.checkpoint = checkpoint
        self.warm_from = warm_from
        self.cut_threshold = cut_threshold
        self.replayed: List[Dict[str, Any]] = []
        self.settled = False
        self.evidence: Dict[str, Any] = {}

    def observe(self, frame_index: int, replay: Dict[str, Any]) -> None:
        self.replayed.append({"frame_index": frame_index,
                              "fingerprint": frame_fingerprint(replay["gray"]),
                              "cut": bool(replay["cut"])})

    def settle(self, braise: Any) -> Dict[str, Any]:
        """Run every check against the just-replayed overlap. Raises ResumeMismatch."""
        evidence: Dict[str, Any] = {"replayed": len(self.replayed),
                                    "warm_from": self.warm_from,
                                    "next_frame": self.checkpoint.next_frame}

        if not self.replayed:
            ## resuming at frame 0 with nothing to replay is not a resume
            if self.checkpoint.next_frame > 0:
                raise ResumeMismatch("decoder produced no overlap frames", evidence)
            self.settled = True
            self.evidence = evidence
            return evidence

        evidence["fingerprint_distance"] = self._check_decode_identity()
        evidence["histogram_distance"] = self._check_histogram_continuity(braise)
        evidence["cuts"] = self._check_shot_consistency()
        evidence["appearance"] = self._check_track_appearance(braise)

        self.settled = True
        self.evidence = evidence
        return evidence

    def _check_decode_identity(self) -> int:
        """Are these the same frames, perceptually?"""
        stored = list(self.checkpoint.overlap_fingerprints)
        replayed = [entry["fingerprint"] for entry in self.replayed]
        if not stored:
            return 0
        ## align from the END: both sequences run up to next_frame, but a resume
        ## near the head of the clip may have had fewer frames to replay
        depth = min(len(stored), len(replayed))
        worst = max(fingerprint_distance(a, b)
                    for a, b in zip(stored[-depth:], replayed[-depth:]))
        if worst > HAMMING_TOLERANCE:
            raise ResumeMismatch(
                f"overlap frames differ from the checkpoint (hamming {worst} > {HAMMING_TOLERANCE})",
                {"hamming": worst, "compared": depth})
        return worst

    def _check_histogram_continuity(self, braise: Any) -> float:
        """Would the resumed decode read a cut where the original read none?"""
        stored = list(self.checkpoint.previous_hist or [])
        if not stored or braise.previous_hist is None:
            return 0.0
        current = [float(v) for v in braise.previous_hist.reshape(-1)]
        distance = histogram_distance(stored, current)
        limit = self.cut_threshold / 4
        if distance > limit:
            raise ResumeMismatch(
                f"luma histogram drifted across the resume ({distance:.4f} > {limit:.4f})",
                {"distance": distance, "limit": limit})
        return distance

    def _check_shot_consistency(self) -> List[int]:
        """One invented cut desynchronizes every later shot id, and with it every
        shot barrier the association rule depends on."""
        window = range(self.warm_from + 1, self.checkpoint.next_frame)
        expected = sorted(f for f in self.checkpoint.overlap_cuts if f in window)
        observed = sorted(entry["frame_index"] for entry in self.replayed
                          if entry["cut"] and entry["frame_index"] in window)
        if expected != observed:
            raise ResumeMismatch(
                f"shot boundaries differ across the resume (expected {expected}, replayed {observed})",
                {"expected": expected, "observed": observed})
        return observed

    def _check_track_appearance(self, braise: Any) -> Dict[str, Any]:
        """Is the state we are about to carry forward still describing the screen?

        This is the check that catches a decoder handing back a plausible frame
        from the wrong place: the pixels may hash close, but the specific boxes
        we are tracking will not look like themselves.
        """
        snapshots = self.checkpoint.active
        if not snapshots or braise.previous_gray is None:
            return {"checked": 0, "passed": 0}
        checked = passed = 0
        worst = 0.0
        for snapshot in snapshots.values():
            if not snapshot.appearance:
                continue
            box = (snapshot.last_observation or {}).get("bbox")
            if not box:
                continue
            current = braise._crop_appearance(braise.previous_gray, box)
            if current is None:
                continue
            checked += 1
            distance = histogram_distance(snapshot.appearance, current)
            worst = max(worst, distance)
            if distance <= APPEARANCE_TOLERANCE:
                passed += 1
        if checked and passed / checked < APPEARANCE_QUORUM:
            raise ResumeMismatch(
                f"tracked regions do not match the checkpoint ({passed}/{checked} agreed)",
                {"checked": checked, "passed": passed, "worst_distance": worst})
        return {"checked": checked, "passed": passed, "worst_distance": worst}
