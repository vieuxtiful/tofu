"""Adaptive keyframe OCR and conservative shot-bounded linking."""
from __future__ import annotations

import time
import uuid
from collections import deque
from dataclasses import asdict
from typing import Any, Callable, Deque, Dict, List, Optional, Sequence, Set, Tuple

from tofu.layers import cicerone
from tofu.core.types import AssetInfo, AssetType
from .checkpoint import (OVERLAP_FRAMES, OverlapReconciler, ResumeMismatch,
                         TrackSnapshot, TrackerCheckpoint, frame_fingerprint)
from .temporal import (ConsensusAccumulator, association_score, settle,
                       should_run_ocr)
from .types import OCRCandidate, TextTrack, TrackObservation, VideoManifest

## Bumped whenever a change here would make a stored checkpoint describe state
## this code no longer produces. Resume refuses a checkpoint from another
## revision rather than silently continuing with mismatched semantics.
ANALYZER_REVISION = "braise-v1"

## Association needs BOTH geometric plausibility and combined evidence. The
## floor stops a distant region with identical text from teleporting into a
## track; the threshold is set so that "quarter overlap, no text opinion"
## (0.65*0.25 + 0.35*0.5) still just passes, preserving the old pure-IoU
## behaviour while letting text agreement rescue a weak overlap and text
## disagreement veto a strong one.
ASSOCIATION_MIN_IOU = .10
ASSOCIATION_THRESHOLD = .34

## Frame-to-frame flow decay, and the point at which a track's geometry is too
## stale to trust without a fresh read.
CONFIDENCE_DECAY = .03
CONFIDENCE_RECOVERY = .02
CONFIDENCE_CEILING = .99
FLOW_FAILURE_PENALTY = .6
MISSED_KEYFRAMES_BEFORE_RETIREMENT = 2


class Braise:
    """Long, low-heat analysis that survives coming off the burner.

    A braise runs for hours, covered, and does not mind being taken off the
    heat and put back on. That is the property this class exists to give the
    analysis loop: the ten interdependent locals the loop used to carry are
    fields, so the whole tracker can be captured at a chunk boundary and
    reconstructed later.

    Invariant: every field mutated by `step` is either captured by `snapshot()`
    or re-derivable by replaying already-analyzed frames through `warm()`.
    That invariant is what makes a resume exact rather than approximate.

    Driving it directly -- `step(frame_array)` in a loop -- needs no decoder,
    which is what makes the resume and memory properties testable.
    """

    def __init__(self, manifest: VideoManifest, *, languages: Optional[Sequence[str]] = None,
                 key_interval: int = 30, cut_threshold: float = .45,
                 start_frame: int = 0, overlap: int = OVERLAP_FRAMES,
                 resume: Optional[TrackerCheckpoint] = None) -> None:
        self.manifest = manifest
        self.languages = languages
        self.key_interval = max(1, int(key_interval))
        self.cut_threshold = float(cut_threshold)

        self.frame = int(start_frame)
        self.shot_index = 0
        self.force_next = False
        ## newest shot boundary seen. at a cut the tracker carries nothing but
        ## this counter, which makes a cut the one place a cold restart is free.
        self.last_cut_frame = 0
        self.previous_hist: Optional[Any] = None
        self.previous_gray: Optional[Any] = None

        self.tracks: Dict[str, TextTrack] = {}
        self.active: Dict[str, TrackObservation] = {}
        self.missed: Dict[str, int] = {}
        ## 32-bin luma histogram of each active track's last OCR crop: cheap
        ## enough to checkpoint, distinctive enough to tell "the decoder gave us
        ## the frame we expected" from "it gave us a different one".
        self.appearance: Dict[str, List[float]] = {}

        ## Only tracks touched since the last drain are written. Sending every
        ## track every chunk made the write O(chunks^2) over a long clip.
        self.dirty: Set[str] = set()
        self.observations: List[TrackObservation] = []

        ## Rolling perceptual record of the frames a resume will have to replay.
        ## Bounded by the overlap, so it costs the same on a 30-second clip as
        ## on a three-hour one.
        self.recent: Deque[Dict[str, Any]] = deque(maxlen=max(1, int(overlap)))

        if resume is not None:
            self._restore(resume)

    def _restore(self, checkpoint: TrackerCheckpoint) -> None:
        """Seat the state that cannot be replayed; leave the rest to `warm()`.

        `previous_hist` and `previous_gray` stay None on purpose. They are
        rebuilt by replaying the overlap, and seeding them with the values from
        the far side of the gap would make the first replayed frame compare
        against a non-adjacent one -- which is precisely how a resume invents a
        shot boundary that was never there.
        """
        self.shot_index = int(checkpoint.shot_index)
        self.force_next = bool(checkpoint.force_next)
        self.last_cut_frame = int(checkpoint.last_cut_frame)
        for track_id, snapshot in checkpoint.active.items():
            track = TextTrack(**{k: v for k, v in snapshot.track.items()
                                 if k in TextTrack.__dataclass_fields__})
            self.tracks[track_id] = track
            observation = snapshot.last_observation
            self.active[track_id] = TrackObservation(
                **{k: v for k, v in observation.items()
                   if k in TrackObservation.__dataclass_fields__ and k != "ocr_candidates"},
                ocr_candidates=[OCRCandidate(**c) for c in (observation.get("ocr_candidates") or [])])
            if snapshot.missed:
                self.missed[track_id] = int(snapshot.missed)
            if snapshot.appearance:
                self.appearance[track_id] = list(snapshot.appearance)

    def snapshot(self, *, job_id: str = "", chunk_index: int = -1,
                 source_fingerprint: str = "", params: Optional[Dict[str, Any]] = None,
                 analyzer_revision: str = ANALYZER_REVISION) -> TrackerCheckpoint:
        """Capture everything a resume cannot re-derive from the overlap."""
        return TrackerCheckpoint(
            job_id=job_id, chunk_index=chunk_index, next_frame=self.frame,
            shot_index=self.shot_index, force_next=self.force_next,
            last_cut_frame=self.last_cut_frame,
            previous_hist=([float(v) for v in self.previous_hist.reshape(-1)]
                           if self.previous_hist is not None else []),
            overlap_fingerprints=[entry["fingerprint"] for entry in self.recent],
            overlap_frames=[entry["frame_index"] for entry in self.recent],
            overlap_cuts=[entry["frame_index"] for entry in self.recent if entry["cut"]],
            active={track_id: TrackSnapshot(
                        track=asdict(self.tracks[track_id]),
                        last_observation=asdict(observation),
                        missed=self.missed.get(track_id, 0),
                        appearance=self.appearance.get(track_id))
                    for track_id, observation in self.active.items()
                    if track_id in self.tracks},
            params=dict(params or {}),
            source_fingerprint=source_fingerprint,
            analyzer_revision=analyzer_revision,
            created_at=time.time())

    # -- per-frame state ---------------------------------------------------

    @property
    def shot_id(self) -> str:
        return f"shot-{self.shot_index:05d}"

    def _histogram(self, gray: Any) -> Any:
        import cv2
        hist = cv2.calcHist([gray], [0], None, [32], [0, 256])
        cv2.normalize(hist, hist)
        return hist

    def _visual_change(self, hist: Any) -> float:
        """Bhattacharyya distance to the previous frame: 0 identical, 1 disjoint."""
        import cv2
        if self.previous_hist is None:
            return 0.0
        return float(cv2.compareHist(self.previous_hist, hist, cv2.HISTCMP_BHATTACHARYYA))

    def _pts(self, frame: int) -> float:
        pts = self.manifest.frame_pts
        return pts[frame] if frame < len(pts) else frame / (self.manifest.fps or 30)

    def _crop_appearance(self, gray: Any, box: Dict[str, float]) -> Optional[List[float]]:
        x, y, w, h = (int(box[k]) for k in ("x", "y", "width", "height"))
        crop = gray[max(0, y):max(0, y + h), max(0, x):max(0, x + w)]
        if not crop.size:
            return None
        return [float(v) for v in self._histogram(crop).reshape(-1)]

    # -- the loop body -----------------------------------------------------

    def warm(self, image: Any) -> Dict[str, Any]:
        """Replay an already-analyzed frame: refresh derived state, emit nothing.

        This is how `previous_gray` comes back after a resume instead of being
        persisted. A 4K grayscale frame is ~8MB and would have to be stored at
        every chunk boundary; re-deriving it costs one decode we already owe the
        reconciler, and unlike a stored copy it *proves* the decoder handed us
        the same pixels as last time.

        Returns the evidence the overlap reconciler compares.
        """
        import cv2
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        hist = self._histogram(gray)
        change = self._visual_change(hist)
        cut = self.previous_hist is not None and change > self.cut_threshold
        self.previous_hist = hist
        self.previous_gray = gray
        replayed = self.frame
        self.frame += 1
        return {"frame_index": replayed, "visual_change": change, "cut": cut,
                "histogram": [float(v) for v in hist.reshape(-1)], "gray": gray}

    def _remember(self, gray: Any, cut: bool) -> None:
        self.recent.append({"frame_index": self.frame,
                            "fingerprint": frame_fingerprint(gray),
                            "cut": bool(cut)})

    def step(self, image: Any) -> None:
        """Analyze one new frame, appending to `self.observations`."""
        import cv2
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        hist = self._histogram(gray)
        change = self._visual_change(hist)
        cut = self.previous_hist is not None and change > self.cut_threshold
        if cut:
            self.shot_index += 1
            self.last_cut_frame = self.frame
            ## nothing survives a cut. this is also why a cut is the cheapest
            ## possible restart point: only shot_index crosses it.
            self.active = {}
            self.missed.clear()
            self.appearance.clear()
        self.previous_hist = hist
        self._remember(gray, cut)

        weakest = min((o.tracking_confidence for o in self.active.values()), default=1.0)
        trigger = should_run_ocr(
            is_keyframe=self.frame % self.key_interval == 0,
            track_birth=self.frame == 0,
            scene_cut=cut,
            confidence=weakest,
            visual_change=change,
            forced=self.force_next,
        )
        self.force_next = False

        if trigger is not None:
            self._read(image, gray, trigger, change)
        else:
            self._propagate(gray, change)

        self.frame += 1
        self.previous_gray = gray

    def _read(self, image: Any, gray: Any, trigger: str, change: float) -> None:
        """Run OCR on this frame and re-associate detections to active tracks."""
        import cv2
        ## an ndarray, not a PIL Image: the backend forwards any non-str asset
        ## straight to the reader, which takes arrays/paths/bytes and rejects
        ## Image objects outright. the frame is already an array -- wrapping it
        ## only bought a ValueError on every keyframe.
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        result = cicerone.detect(rgb, asset_info=AssetInfo(asset_type=AssetType.IMAGE),
                                 languages=self.languages)
        unmatched = set(self.active)
        for inst in result.instances:
            box = asdict(inst.bounding_box)
            candidates = [OCRCandidate((inst.ocr_provenance or {}).get("accepted_engine", "cicerone"),
                                       inst.text or "", float(inst.confidence or 0), accepted=True)]
            x, y, w, h = (int(box[k]) for k in ("x", "y", "width", "height"))
            crop = gray[max(0, y):max(0, y + h), max(0, x):max(0, x + w)]
            sharpness = float(cv2.Laplacian(crop, cv2.CV_64F).var()) if crop.size else 0.0
            observation = TrackObservation(uuid.uuid4().hex[:16], "", self.shot_id, self.frame,
                                           self._pts(self.frame), box,
                                           glyph_mask_polygon=(inst.segmentation_mask.polygon
                                                               if inst.segmentation_mask else None),
                                           tracking_confidence=1.0, sharpness=sharpness,
                                           visual_change=change, ocr_trigger=trigger,
                                           ocr_candidates=candidates)
            observation.track_id = self._associate(observation, unmatched)
            self._record(observation, gray)

        for track_id in list(unmatched):
            self.missed[track_id] = self.missed.get(track_id, 0) + 1
            if self.missed[track_id] >= MISSED_KEYFRAMES_BEFORE_RETIREMENT:
                self.active.pop(track_id, None)
                self.appearance.pop(track_id, None)
                self.missed.pop(track_id, None)

    def _associate(self, observation: TrackObservation, unmatched: Set[str]) -> str:
        """Pick the best surviving track for a detection, or open a new one.

        Scoring is `association_score` -- overlap AND text agreement, with an
        absolute shot barrier -- rather than overlap alone. The IoU floor keeps
        the text term from matching a region on the far side of the frame just
        because it reads the same.
        """
        from .temporal import bbox_iou
        best, best_score = None, 0.0
        for track_id in unmatched:
            previous = self.active[track_id]
            if bbox_iou(previous.bbox, observation.bbox) < ASSOCIATION_MIN_IOU:
                continue
            score = association_score(previous, observation)
            if score > best_score:
                best, best_score = track_id, score
        if best is not None and best_score >= ASSOCIATION_THRESHOLD:
            unmatched.discard(best)
            return best
        track_id = uuid.uuid4().hex[:16]
        self.tracks[track_id] = TextTrack(track_id, self.shot_id, self.frame, self.frame)
        self.dirty.add(track_id)
        return track_id

    def _record(self, observation: TrackObservation, gray: Any) -> None:
        """Attach an OCR observation to its track and fold the evidence away."""
        track_id = observation.track_id
        self.observations.append(observation)
        self.active[track_id] = observation
        self.missed[track_id] = 0
        appearance = self._crop_appearance(gray, observation.bbox)
        if appearance is not None:
            self.appearance[track_id] = appearance
        ## invariant: active ⊆ tracks. drain() only evicts tracks that are NOT
        ## active, and _associate seats a new track before _record runs, so a
        ## lookup here cannot miss.
        track = self.tracks[track_id]
        track.end_frame = observation.frame_index
        track.observation_count += 1
        ## fold immediately: holding every keyframe observation until the end of
        ## the clip was the one structure here that grew with duration.
        accumulator = ConsensusAccumulator.from_dict(track.consensus_state)
        accumulator.fold(observation)
        track.consensus_state = accumulator.to_dict()
        self.dirty.add(track_id)

    def _propagate(self, gray: Any, change: float) -> None:
        """Carry every active box forward on sparse optical flow."""
        import cv2
        import numpy as np
        for track_id, prior in list(self.active.items()):
            box = dict(prior.bbox)
            confidence = max(.0, prior.tracking_confidence - CONFIDENCE_DECAY)
            delta = (0.0, 0.0)
            if self.previous_gray is not None:
                x, y, w, h = (float(box[k]) for k in ("x", "y", "width", "height"))
                points = np.array([[[x, y]], [[x + w, y]], [[x + w, y + h]], [[x, y + h]],
                                   [[x + w / 2, y + h / 2]]], dtype=np.float32)
                moved, status, _ = cv2.calcOpticalFlowPyrLK(self.previous_gray, gray, points, None)
                valid = status.reshape(-1).astype(bool) if status is not None else np.zeros(5, dtype=bool)
                if moved is not None and valid.sum() >= 3:
                    shift = np.median((moved - points).reshape(-1, 2)[valid], axis=0)
                    delta = (float(shift[0]), float(shift[1]))
                    box["x"], box["y"] = max(0.0, x + delta[0]), max(0.0, y + delta[1])
                    confidence = min(CONFIDENCE_CEILING, confidence + CONFIDENCE_RECOVERY * int(valid.sum()))
                else:
                    confidence *= FLOW_FAILURE_PENALTY
            if confidence < .55:
                self.force_next = True
            shifted = ([[point[0] + delta[0], point[1] + delta[1]] for point in prior.glyph_mask_polygon]
                       if prior.glyph_mask_polygon else None)
            observation = TrackObservation(uuid.uuid4().hex[:16], track_id, self.shot_id, self.frame,
                                           self._pts(self.frame), box, quad=prior.quad,
                                           glyph_mask_polygon=shifted, tracking_confidence=confidence,
                                           visual_change=change)
            self.observations.append(observation)
            self.active[track_id] = observation
            track = self.tracks[track_id]          # active ⊆ tracks; see _record
            track.end_frame = observation.frame_index
            track.observation_count += 1
            self.dirty.add(track_id)

    # -- chunk boundary ----------------------------------------------------

    def drain(self) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """Hand off this chunk's writes and let go of everything already durable.

        Returns (touched tracks, buffered observations) and clears both. Tracks
        that are neither active nor newly touched are dropped from memory: they
        are rows now, and `settle()` reads them back from there. This is what
        keeps the resident set proportional to the regions on screen rather than
        to the length of the clip.
        """
        tracks = [asdict(self.tracks[track_id]) for track_id in sorted(self.dirty)
                  if track_id in self.tracks]
        observations = [asdict(observation) for observation in self.observations]
        self.dirty.clear()
        self.observations.clear()
        for track_id in [t for t in self.tracks if t not in self.active]:
            del self.tracks[track_id]
        return tracks, observations


def _seek(cap: Any, target: int, *, mode: str = "fast") -> None:
    """Position the decoder at `target`.

    'fast' asks the container to seek, which on a long-GOP stream lands on a
    preceding keyframe and decodes forward -- usually right, occasionally off by
    a frame. 'exact' rewinds and grabs forward without ever converting to BGR,
    which is slower but cannot be wrong; it is what the reconciler escalates to
    when the fast path fails its checks.
    """
    import cv2
    if target <= 0:
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        return
    if mode == "exact":
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        for _ in range(target):
            if not cap.grab():
                break
        return
    cap.set(cv2.CAP_PROP_POS_FRAMES, target)


def analyze(source: str, manifest: VideoManifest, *, languages: Optional[Sequence[str]] = None,
            cancelled: Callable[[], bool] = lambda: False,
            progress: Callable[[float], None] = lambda value: None,
            checkpoint: Optional[Callable[..., None]] = None,
            chunk_size: int = 240, key_interval: int = 30, cut_threshold: float = .45,
            resume: Optional[TrackerCheckpoint] = None, overlap: int = OVERLAP_FRAMES,
            seek_mode: str = "fast", job_id: str = "", fingerprint: str = "",
            on_resume: Optional[Callable[[Dict[str, Any]], None]] = None,
            ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    import cv2
    cap = cv2.VideoCapture(source)
    if hasattr(cv2, "CAP_PROP_ORIENTATION_AUTO"): cap.set(cv2.CAP_PROP_ORIENTATION_AUTO, 0)
    if not cap.isOpened():
        raise RuntimeError("video decoder could not open input")
    chunk_size = max(1, int(chunk_size))
    start = int(resume.next_frame) if resume else 0
    warm_from = max(0, start - int(overlap)) if resume else 0
    _seek(cap, warm_from, mode=seek_mode)
    braise = Braise(manifest, languages=languages, key_interval=key_interval,
                    cut_threshold=cut_threshold, start_frame=warm_from,
                    overlap=overlap, resume=resume)
    reconciler = (OverlapReconciler(resume, warm_from, cut_threshold=cut_threshold)
                  if resume else None)
    params = {"chunk_size": chunk_size, "key_interval": key_interval,
              "cut_threshold": cut_threshold, "overlap": overlap, "seek_mode": seek_mode,
              "languages": list(languages) if languages else None}

    ## Braise lets go of a track once it is durable, but this wrapper still owes
    ## its caller every track. Keyed by id, last write wins -- the same merge
    ## INSERT OR REPLACE performs. A server that settles over the rows instead
    ## can ignore the first element of the return and never hold this at all.
    seen: Dict[str, Dict[str, Any]] = {}

    def collect(tracks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        for track in tracks:
            seen[track["id"]] = track
        return tracks

    def commit(index: int, end_frame: int) -> None:
        tracks, observations = braise.drain()
        ## the snapshot is handed over in the same call as the rows it describes,
        ## so a caller can write both in one transaction. a checkpoint that
        ## landed without its observations would resume past work never done.
        checkpoint(index, end_frame, collect(tracks), observations,
                   braise.snapshot(job_id=job_id, chunk_index=index,
                                   source_fingerprint=fingerprint, params=params))

    try:
        while True:
            if cancelled():
                raise InterruptedError("video analysis cancelled")
            ok, image = cap.read()
            if not ok:
                break
            if reconciler is not None and braise.frame < start:
                ## still inside the overlap: replay without emitting anything
                reconciler.observe(braise.frame, braise.warm(image))
                continue
            if reconciler is not None and not reconciler.settled:
                evidence = reconciler.settle(braise)
                if on_resume:
                    on_resume(evidence)
            analyzed = braise.frame
            braise.step(image)
            if analyzed % 10 == 0:
                progress(min(1.0, analyzed / max(1, manifest.frame_count)))
            ## chunk c covers [c*chunk_size, (c+1)*chunk_size-1], matching how
            ## create_video_job cut the rows. one expression, used identically
            ## here and for the trailing flush, so the two cannot disagree.
            if checkpoint and (analyzed + 1) % chunk_size == 0:
                commit(analyzed // chunk_size, analyzed)
    finally:
        cap.release()
    if reconciler is not None and not reconciler.settled:
        ## the decoder ran dry inside the overlap -- the resume never began
        raise ResumeMismatch("decoder ended before the resume point",
                             {"next_frame": start, "reached": braise.frame})
    ## Flush the trailing partial chunk here rather than leaving it to the
    ## caller. The chunk index is the same expression used in the loop, so the
    ## two cannot drift, and the final chunk gets a snapshot like every other.
    if checkpoint and (braise.observations or braise.dirty):
        commit(max(0, (braise.frame - 1)) // chunk_size, braise.frame - 1)
    tracks, observations = braise.drain()
    collect(tracks)
    settled, issues = settle(seen.values())
    return settled, observations, issues
