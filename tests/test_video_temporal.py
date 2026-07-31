from tofu.video.temporal import association_score, resolve_keyframes, should_run_ocr, temporal_consensus
from tofu.video.types import OCRCandidate, RenderKeyframe, TrackObservation


def obs(oid, shot, frame, bbox, text=None, confidence=.9, sharpness=100):
    candidates = [] if text is None else [OCRCandidate("test", text, confidence, accepted=True)]
    return TrackObservation(oid, "t1", shot, frame, frame / 30, bbox,
                            sharpness=sharpness, ocr_candidates=candidates)


def test_association_never_crosses_shot_boundary():
    box = {"x": 1, "y": 2, "width": 30, "height": 10}
    assert association_score(obs("a", "s1", 1, box, "hello"), obs("b", "s2", 2, box, "hello")) == 0
    assert association_score(obs("a", "s1", 1, box, "hello"), obs("b", "s1", 2, box, "hello")) == 1


def test_adaptive_ocr_reports_the_highest_priority_trigger():
    assert should_run_ocr(is_keyframe=True, forced=True, confidence=.1) == "forced"
    assert should_run_ocr(is_keyframe=False, confidence=.4) == "confidence_decay"
    assert should_run_ocr(is_keyframe=False, confidence=.9, visual_change=.1) is None


def test_temporal_consensus_weights_crop_quality_and_reports_disagreement():
    box = {"x": 0, "y": 0, "width": 10, "height": 10}
    observations = [obs("a", "s", 1, box, "TOFU", .95, 200),
                    obs("b", "s", 2, box, "T0FU", .8, 25),
                    obs("c", "s", 3, box, "T0FU", .8, 25)]
    result = temporal_consensus(observations)
    assert result["text"] == "TOFU"
    assert result["disagreement"] is True


def test_keyframe_bbox_interpolation_and_track_override():
    keys = [RenderKeyframe("track", "t", 0, scope="track", opacity=.5),
            RenderKeyframe("a", "t", 0, bbox={"x": 0, "y": 0, "width": 10, "height": 10}),
            RenderKeyframe("b", "t", 10, bbox={"x": 10, "y": 20, "width": 20, "height": 30})]
    resolved = resolve_keyframes(5, {}, keys)
    assert resolved["opacity"] == .5
    assert resolved["bbox"] == {"x": 5, "y": 10, "width": 15, "height": 20}
