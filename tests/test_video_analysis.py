"""Analyzer internals that must hold without a decoder.

Everything here is pure: no cv2, no ffmpeg, no database. That is deliberate --
these are the properties (bounded memory, lossless resume, idempotent
finalization) that are impossible to test convincingly through a video file.
"""
import json
import random

from tofu.video.temporal import (CONSENSUS_TOP_K, ConsensusAccumulator, settle,
                                 temporal_consensus)
from tofu.video.types import OCRCandidate, TrackObservation


def obs(text=None, confidence=.9, sharpness=100, frame=0):
    candidates = [] if text is None else [OCRCandidate("test", text, confidence, accepted=True)]
    return TrackObservation(f"o{frame}", "t1", "s1", frame, frame / 30,
                            {"x": 0, "y": 0, "width": 10, "height": 10},
                            sharpness=sharpness, ocr_candidates=candidates)


def test_streaming_fold_matches_the_batch_vote():
    """The wrapper and the accumulator must not drift into two different votes."""
    rng = random.Random(20260730)
    vocabulary = ["TOFU", "T0FU", "TDFU", "tofu"]
    for _ in range(50):
        observations = [obs(rng.choice(vocabulary), rng.uniform(.2, 1.0),
                            rng.uniform(10, 400), index)
                        for index in range(rng.randint(1, 30))]
        accumulator = ConsensusAccumulator()
        for observation in observations:
            accumulator.fold(observation)
        assert accumulator.verdict() == temporal_consensus(observations)


def test_accumulator_is_bounded_and_never_inflates_confidence():
    """10k distinct readings must not become 10k dict entries, and evicting the
    losers must not make the winner look more certain than the evidence is."""
    dominant = [obs("REAL SIGN", .99, 400, index) for index in range(40)]
    noise = [obs(f"junk-{index}", .05, 10, 1000 + index) for index in range(10_000)]

    accumulator = ConsensusAccumulator()
    for observation in dominant + noise:
        accumulator.fold(observation)

    assert len(accumulator.scores) <= CONSENSUS_TOP_K
    assert accumulator.evicted_variants > 0
    verdict = accumulator.verdict()
    assert verdict["text"] == "REAL SIGN"
    assert verdict["disagreement"] is True
    # the evicted weight stays in the denominator, so the bounded answer is
    # never more confident than the exhaustive one
    assert verdict["confidence"] <= temporal_consensus(dominant + noise)["confidence"] + 1e-9


def test_accumulator_survives_a_json_roundtrip():
    """It rides in video_tracks.data_json, so a lossy roundtrip would silently
    reset the vote at every chunk boundary."""
    accumulator = ConsensusAccumulator()
    for index, text in enumerate(["TOFU", "T0FU", "TOFU"]):
        accumulator.fold(obs(text, .8, 150, index))
    restored = ConsensusAccumulator.from_dict(accumulator.to_dict())
    assert restored.verdict() == accumulator.verdict()
    # and folding continues correctly on the restored one
    restored.fold(obs("TOFU", .9, 200, 9))
    accumulator.fold(obs("TOFU", .9, 200, 9))
    assert restored.verdict() == accumulator.verdict()


def test_fold_ignores_blank_readings_without_counting_the_observation():
    accumulator = ConsensusAccumulator()
    accumulator.fold(obs(None))
    accumulator.fold(obs("   "))
    assert accumulator.observed == 0
    assert accumulator.verdict() == {"text": None, "confidence": 0.0,
                                     "evidence_count": 0, "disagreement": False}


def _track(track_id, state, status="review_required", start=0):
    return {"id": track_id, "shot_id": "s1", "start_frame": start, "end_frame": start + 10,
            "status": status, "consensus_state": state}


def _state(*readings):
    accumulator = ConsensusAccumulator()
    for index, (text, confidence, sharpness) in enumerate(readings):
        accumulator.fold(obs(text, confidence, sharpness, index))
    return accumulator.to_dict()


def test_settle_reads_the_verdict_off_the_row():
    agreed = _track("clean", _state(("STOP", .95, 300), ("STOP", .95, 300)))
    split = _track("split", _state(("STOP", .9, 300), ("5TOP", .9, 300)))
    settled, issues = settle([agreed, split])

    by_id = {track["id"]: track for track in settled}
    assert by_id["clean"]["source_text"] == "STOP"
    assert by_id["clean"]["status"] == "recognized"
    assert by_id["split"]["status"] == "review_required"
    assert [issue["track_id"] for issue in issues] == ["split"]
    assert issues[0]["code"] == "ocr_disagreement"


def test_settle_is_idempotent():
    """It re-runs after a partial resume, so running it twice must not change
    the answer or duplicate issues."""
    rows = [_track("a", _state(("GO", .95, 300))), _track("b", _state(("GO", .5, 20), ("G0", .5, 20)))]
    once, issues_once = settle(rows)
    twice, issues_twice = settle(once)
    assert once == twice
    assert issues_once == issues_twice


def test_settle_never_revives_an_excluded_track():
    """'excluded' is a reviewer decision the compositor honours by skipping the
    region; recomputing over it would put burned-in source text back on screen."""
    excluded = _track("dropped", _state(("SIGN", .99, 400)), status="excluded")
    settled, issues = settle([excluded])
    assert settled[0]["status"] == "excluded"
    assert settled[0]["source_text"] == "SIGN"   # evidence still resolved
    assert issues == []


def test_settle_handles_a_track_that_never_saw_ocr():
    settled, issues = settle([_track("silent", {})])
    assert settled[0]["source_text"] is None
    assert settled[0]["status"] == "review_required"
    assert len(issues) == 1


# --- Braise: driven with synthetic frames, no decoder ----------------------

import numpy as np
import pytest

from tofu.core.types import BBox, InstText, TextManifest
from tofu.video import analysis as analysis_module
from tofu.video.analysis import Braise
from tofu.video.types import VideoManifest


def manifest(frames=120, width=160, height=120):
    return VideoManifest("asset", width, height, frames / 30, frames, 30.0, "1/30", "h264",
                         frame_pts=[index / 30 for index in range(frames)])


def frame(fill=40, box=None, width=160, height=120, tag=None):
    """A flat frame with an optional bright rectangle standing in for a sign.

    `tag` writes the frame index into the corner pixel. The fake reader decodes
    it, so what it "sees" is a function of the IMAGE rather than of how many
    times it has been called -- without that, a resumed run gets a fresh counter
    and is effectively shown a different clip, which would make the resume
    comparison meaningless.
    """
    image = np.full((height, width, 3), fill, dtype=np.uint8)
    if box:
        x, y, w, h = box
        image[y:y + h, x:x + w] = 235
    if tag is not None:
        image[0, 0] = (tag % 256,) * 3
    return image


class FakeCicerone:
    """Stands in for the OCR layer so tracker behaviour is deterministic.

    Records every call, so tests can assert on how often OCR was asked to run
    -- which is the whole point of the adaptive trigger.
    """

    def __init__(self, plan, frame_indexed=False):
        self.plan = plan          # callable(key) -> [(text, bbox), ...]
        self.frame_indexed = frame_indexed
        self.calls = 0

    def detect(self, asset, **_):
        key = int(asset[0, 0, 0]) if self.frame_indexed else self.calls
        self.calls += 1
        instances = [
            InstText(id=f"i{index}", bounding_box=BBox(*box), text=text, confidence=.9)
            for index, (text, box) in enumerate(self.plan(key))
        ]
        return TextManifest(asset_id="fake", total_regions=len(instances), instances=instances)


@pytest.fixture
def fake_ocr(monkeypatch):
    def install(plan, frame_indexed=False):
        stub = FakeCicerone(plan, frame_indexed)
        monkeypatch.setattr(analysis_module, "cicerone", stub)
        return stub
    return install


def test_a_track_never_survives_a_shot_cut(fake_ocr):
    """The barrier is absolute: a sign in the same place either side of a cut is
    two different signs, because the shot changed underneath it."""
    box = (20, 20, 60, 25)
    fake_ocr(lambda _: [("SIGN", box)])
    braise = Braise(manifest(), key_interval=1, cut_threshold=.45)

    for _ in range(3):
        braise.step(frame(fill=40, box=box))
    before = set(braise.tracks)
    braise.step(frame(fill=225, box=box))     # violent luma change -> cut
    assert braise.shot_index == 1
    assert braise.last_cut_frame == 3
    after = set(braise.active)
    assert after.isdisjoint(before)
    assert len(braise.tracks) == len(before) + 1


def test_after_a_cut_no_tracker_state_references_the_previous_shot(fake_ocr):
    """The property the shot-aligned cold restart depends on: a cut leaves
    nothing behind but the shot counter, so restarting there is free.

    The cut itself clears `active`, then the scene_cut trigger re-reads in the
    same step -- so the assertion is not 'empty' but 'nothing from before'.
    """
    box = (20, 20, 60, 25)
    fake_ocr(lambda _: [("SIGN", box)])
    braise = Braise(manifest(), key_interval=1, cut_threshold=.45)
    for _ in range(3):
        braise.step(frame(fill=40, box=box))
    old_shot, old_tracks = braise.shot_id, set(braise.active)

    braise.step(frame(fill=225, box=box))     # violent luma change -> cut

    assert braise.shot_id != old_shot
    assert braise.last_cut_frame == 3
    assert set(braise.active).isdisjoint(old_tracks)
    assert {o.shot_id for o in braise.active.values()} == {braise.shot_id}
    ## the bookkeeping dicts must be pruned with it, or they leak for the rest
    ## of the clip and bloat every checkpoint after this one
    assert set(braise.missed) <= set(braise.active)
    assert set(braise.appearance) <= set(braise.active)


def test_ocr_runs_on_the_trigger_ladder_not_on_every_frame(fake_ocr):
    """A static sign held for 90 frames must not cost 90 OCR invocations."""
    box = (20, 20, 60, 25)
    stub = fake_ocr(lambda _: [("STATIC", box)])
    braise = Braise(manifest(), key_interval=30)
    for _ in range(90):
        braise.step(frame(fill=40, box=box))

    assert stub.calls <= 6, f"adaptive trigger ran OCR {stub.calls} times over 90 frames"
    assert len(braise.tracks) == 1
    triggers = {o.ocr_trigger for o in braise.observations if o.ocr_trigger}
    assert triggers <= {"track_birth", "keyframe", "scene_cut", "confidence_decay",
                        "visual_change", "forced", "occlusion_recovery"}
    ## the old inline ladder labelled every ordinary interval keyframe
    ## 'confidence_decay' via `frame % key_interval` -- exactly backwards
    assert any(o.ocr_trigger == "keyframe" for o in braise.observations)
    assert braise.observations[0].ocr_trigger == "track_birth"


def test_drain_writes_only_touched_tracks_and_releases_the_rest(fake_ocr):
    """The O(chunks^2) fix: a track that stopped changing must stop being written."""
    early, late = (10, 10, 40, 20), (100, 80, 40, 20)

    def plan(call):
        return [("EARLY", early)] if call < 2 else [("LATE", late)]

    fake_ocr(plan)
    braise = Braise(manifest(), key_interval=1)

    for _ in range(2):
        braise.step(frame(box=early))
    first, _ = braise.drain()
    assert {t["source_text"] is None for t in first} == {True}   # settle() fills text later
    assert len(first) == 1

    for _ in range(4):
        braise.step(frame(box=late))
    second, _ = braise.drain()
    written = {t["id"] for t in second}
    assert first[0]["id"] not in written, "an untouched track was rewritten"
    ## and it is gone from memory, because it is a row now
    assert first[0]["id"] not in braise.tracks
    assert len(braise.tracks) == len(braise.active)


def test_resident_track_memory_does_not_grow_with_clip_length(fake_ocr):
    """One region on screen at a time, 40 chunks: the resident set must stay flat."""
    def plan(call):
        return [(f"SIGN {call}", (10 + (call % 3) * 40, 20, 35, 20))]

    fake_ocr(plan)
    braise = Braise(manifest(frames=1200), key_interval=1)
    resident = []
    for chunk in range(40):
        for _ in range(5):
            braise.step(frame(box=(10 + (chunk % 3) * 40, 20, 35, 20)))
        braise.drain()
        resident.append(len(braise.tracks))

    assert max(resident) <= 4, f"resident tracks grew: {resident}"
    assert resident[-1] <= resident[4] + 1


def test_association_needs_geometry_not_just_matching_text(fake_ocr):
    """Two signs reading the same words in different places stay two tracks."""
    left, right = (5, 20, 40, 20), (110, 80, 40, 20)

    def plan(call):
        return [("EXIT", left)] if call == 0 else [("EXIT", right)]

    fake_ocr(plan)
    braise = Braise(manifest(), key_interval=1)
    braise.step(frame(box=left))
    braise.step(frame(box=right))
    assert len(braise.tracks) == 2, "identical text teleported across the frame"


# --- checkpoint and resume ------------------------------------------------

from tofu.video.checkpoint import (OVERLAP_FRAMES, OverlapReconciler, ResumeMismatch,
                                   TrackerCheckpoint, frame_fingerprint,
                                   fingerprint_distance)


def _script(index):
    """A moving sign, so tracking state actually has to carry across the gap."""
    return [(f"KM {index // 4}", (8 + (index % 9) * 6, 30, 48, 22))]


def _drive(braise, frames):
    for index, image in enumerate(frames):
        braise.step(image)


def _clip(count=60):
    return [frame(fill=40, box=_script(index)[0][1], tag=index) for index in range(count)]


def _comparable(observations):
    """Ignore the uuids -- they are meant to differ between runs."""
    return [(o.frame_index, o.shot_id, {k: round(float(v), 3) for k, v in o.bbox.items()},
             round(o.tracking_confidence, 6), o.ocr_trigger) for o in observations]


def test_resume_from_a_checkpoint_matches_an_uninterrupted_run(fake_ocr):
    """The headline property. If this does not hold, the durability story is
    decoration: a job that resumes to a different answer than it would have
    reached uninterrupted is not resumable, it is merely restartable."""
    frames = _clip(60)
    cut_at = 40

    fake_ocr(_script, frame_indexed=True)
    straight = Braise(manifest(frames=60), key_interval=4)
    _drive(straight, frames)
    expected = _comparable(straight.observations)

    # --- run again, stopping at cut_at, and carry the state through JSON ---
    fake_ocr(_script, frame_indexed=True)
    first = Braise(manifest(frames=60), key_interval=4)
    _drive(first, frames[:cut_at])
    stored = json.loads(json.dumps(first.snapshot(job_id="j", chunk_index=0).to_dict()))
    before = _comparable(first.observations)

    checkpoint = TrackerCheckpoint.from_dict(stored)
    assert checkpoint.next_frame == cut_at

    warm_from = cut_at - OVERLAP_FRAMES
    fake_ocr(_script, frame_indexed=True)
    second = Braise(manifest(frames=60), key_interval=4,
                    start_frame=warm_from, resume=checkpoint)
    reconciler = OverlapReconciler(checkpoint, warm_from)
    for index in range(warm_from, cut_at):
        reconciler.observe(index, second.warm(frames[index]))
    reconciler.settle(second)          # must not raise
    assert second.frame == cut_at
    _drive(second, frames[cut_at:])

    assert before + _comparable(second.observations) == expected


def test_the_replayed_overlap_emits_nothing(fake_ocr):
    """warm() must not double-count: those frames were already analyzed."""
    frames = _clip(40)
    fake_ocr(_script, frame_indexed=True)
    braise = Braise(manifest(frames=40), key_interval=4)
    _drive(braise, frames[:30])
    checkpoint = TrackerCheckpoint.from_dict(braise.snapshot().to_dict())

    fake_ocr(_script, frame_indexed=True)
    resumed = Braise(manifest(frames=40), key_interval=4,
                     start_frame=30 - OVERLAP_FRAMES, resume=checkpoint)
    for index in range(30 - OVERLAP_FRAMES, 30):
        resumed.warm(frames[index])
    assert resumed.observations == []
    assert resumed.dirty == set()


def test_checkpoint_survives_a_json_roundtrip(fake_ocr):
    fake_ocr(_script, frame_indexed=True)
    braise = Braise(manifest(frames=40), key_interval=4)
    _drive(braise, _clip(24))
    original = braise.snapshot(job_id="job", chunk_index=2, source_fingerprint="abc")
    restored = TrackerCheckpoint.from_dict(json.loads(json.dumps(original.to_dict())))
    assert restored.to_dict() == original.to_dict()
    assert set(restored.active) == set(braise.active)


def test_checkpoint_holds_only_active_tracks(fake_ocr):
    """Retired tracks live on their row. Carrying them here would make the blob
    grow with the clip and undo the incremental write in the same stroke."""
    def plan(call):
        return [(f"SIGN {call}", (5 + call * 30, 20, 25, 18))]   # marches off screen

    fake_ocr(plan)
    braise = Braise(manifest(frames=60), key_interval=1)
    for index in range(6):
        braise.step(frame(box=(5 + index * 30, 20, 25, 18)))
    checkpoint = braise.snapshot()
    assert set(checkpoint.active) == set(braise.active)
    assert len(checkpoint.active) < len(braise.tracks)


def test_a_tampered_overlap_refuses_to_resume(fake_ocr):
    """Perceptually different frames must stop the resume, not be averaged over."""
    frames = _clip(40)
    fake_ocr(_script, frame_indexed=True)
    braise = Braise(manifest(frames=40), key_interval=4)
    _drive(braise, frames[:30])
    checkpoint = TrackerCheckpoint.from_dict(braise.snapshot().to_dict())
    ## complement each hash, so the tamper is maximally distant whatever the
    ## synthetic frames happened to hash to
    checkpoint.overlap_fingerprints = [f"{int(fp, 16) ^ ((1 << 64) - 1):016x}"
                                       for fp in checkpoint.overlap_fingerprints]

    warm_from = 30 - OVERLAP_FRAMES
    fake_ocr(_script, frame_indexed=True)
    resumed = Braise(manifest(frames=40), key_interval=4, start_frame=warm_from, resume=checkpoint)
    reconciler = OverlapReconciler(checkpoint, warm_from)
    for index in range(warm_from, 30):
        reconciler.observe(index, resumed.warm(frames[index]))
    with pytest.raises(ResumeMismatch, match="overlap frames differ"):
        reconciler.settle(resumed)


def test_an_invented_shot_boundary_refuses_to_resume(fake_ocr):
    """One spurious cut desynchronizes every later shot id, and with it every
    shot barrier the association rule leans on."""
    frames = _clip(40)
    fake_ocr(_script, frame_indexed=True)
    braise = Braise(manifest(frames=40), key_interval=4)
    _drive(braise, frames[:30])
    checkpoint = TrackerCheckpoint.from_dict(braise.snapshot().to_dict())
    checkpoint.overlap_cuts = [25]          # a boundary the replay will not find

    warm_from = 30 - OVERLAP_FRAMES
    fake_ocr(_script, frame_indexed=True)
    resumed = Braise(manifest(frames=40), key_interval=4, start_frame=warm_from, resume=checkpoint)
    reconciler = OverlapReconciler(checkpoint, warm_from)
    for index in range(warm_from, 30):
        reconciler.observe(index, resumed.warm(frames[index]))
    with pytest.raises(ResumeMismatch, match="shot boundaries differ"):
        reconciler.settle(resumed)


def test_moved_regions_refuse_to_resume(fake_ocr):
    """The check that catches a decoder returning a plausible frame from the
    wrong place: the pixels may hash close, the tracked boxes will not."""
    frames = _clip(40)
    fake_ocr(_script, frame_indexed=True)
    braise = Braise(manifest(frames=40), key_interval=4)
    _drive(braise, frames[:30])
    checkpoint = TrackerCheckpoint.from_dict(braise.snapshot().to_dict())
    for snapshot in checkpoint.active.values():
        snapshot.appearance = [0.0] * 31 + [1.0]     # nothing on screen looks like this

    warm_from = 30 - OVERLAP_FRAMES
    fake_ocr(_script, frame_indexed=True)
    resumed = Braise(manifest(frames=40), key_interval=4, start_frame=warm_from, resume=checkpoint)
    reconciler = OverlapReconciler(checkpoint, warm_from)
    for index in range(warm_from, 30):
        reconciler.observe(index, resumed.warm(frames[index]))
    with pytest.raises(ResumeMismatch, match="tracked regions do not match"):
        reconciler.settle(resumed)


def test_frame_fingerprint_separates_neighbours_but_tolerates_noise():
    rng = np.random.default_rng(20260730)
    base = rng.integers(0, 255, (120, 160), dtype=np.uint8)
    other = rng.integers(0, 255, (120, 160), dtype=np.uint8)
    noisy = np.clip(base.astype(np.int16) + rng.integers(-1, 2, base.shape), 0, 255).astype(np.uint8)

    assert fingerprint_distance(frame_fingerprint(base), frame_fingerprint(noisy)) <= 5
    assert fingerprint_distance(frame_fingerprint(base), frame_fingerprint(other)) > 5


def test_association_survives_a_wobble_when_the_text_agrees(fake_ocr):
    """Partial overlap plus agreeing text is one sign, not two."""
    first, nudged = (20, 20, 60, 25), (34, 26, 60, 25)

    def plan(call):
        return [("PHARMACIE", first if call == 0 else nudged)]

    fake_ocr(plan)
    braise = Braise(manifest(), key_interval=1)
    braise.step(frame(box=first))
    braise.step(frame(box=nudged))
    assert len(braise.tracks) == 1
