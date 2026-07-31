"""ResolvedFramePlan: the contract that keeps preview and export honest."""
from tofu.core.types import BBox
from tofu.video.plan import (PLAN_SCHEMA_VERSION, Downgrade, PlanInputs,
                             plan_inputs, plate)
from tofu.video.types import RenderKeyframe


def _track(track_id="t1", **overrides):
    return {"id": track_id, "shot_id": "shot-00000", "revision": 1, "status": "translated",
            "source_text": "SORTIE", "target_text": "EXIT", "target_language": "en",
            "style": {"font_size": 20}, "compositing_order": 0, **overrides}


def _observation(frame_index, track_id="t1", x=10, **overrides):
    return {"id": f"o{frame_index}", "track_id": track_id, "shot_id": "shot-00000",
            "frame_index": frame_index,
            "bbox": {"x": x, "y": 20, "width": 40, "height": 18}, **overrides}


def _plate(frame_index, observations, keyframes=None, tracks=None, inputs=None):
    return plate(frame_index, pts_seconds=frame_index / 30, width=160, height=120,
                 tracks={t["id"]: t for t in (tracks or [_track()])},
                 observations=observations, keyframes=keyframes or {},
                 inputs=inputs or PlanInputs(job_id="job"))


def test_plate_is_independent_of_the_range_that_asked_for_it():
    """A preview resolving frame 20 must reach the same plan as an export that
    happened to also load frames 0-19 on the way there."""
    everything = [_observation(index, x=10 + index) for index in range(40)]
    window = [o for o in everything if 12 <= o["frame_index"] <= 27]

    wide = _plate(20, [o for o in everything if o["frame_index"] == 20])
    narrow = _plate(20, [o for o in window if o["frame_index"] == 20])
    assert wide.regions[0].bbox == narrow.regions[0].bbox
    assert wide.revision == narrow.revision
    assert wide.regions[0].target_text == narrow.regions[0].target_text


def test_plate_is_deterministic():
    observations = [_observation(5)]
    first, second = _plate(5, observations), _plate(5, observations)
    assert first.regions[0].bbox == second.regions[0].bbox
    assert first.revision == second.revision


def test_geometry_is_clipped_to_the_frame():
    """A dragged box may leave the canvas; the renderer must not be handed one."""
    plan = _plate(0, [_observation(0, x=150)])       # 40 wide on a 160 canvas
    box = plan.regions[0].bbox
    assert box.x + box.width <= 160
    assert box.width >= 1 and box.height >= 1


def test_excluded_and_untranslated_tracks_are_not_plated():
    excluded = _track("t1", status="excluded")
    untranslated = _track("t2", target_text=None)
    plan = plate(0, pts_seconds=0, width=160, height=120,
                 tracks={"t1": excluded, "t2": untranslated},
                 observations=[_observation(0, "t1"), _observation(0, "t2")],
                 keyframes={}, inputs=PlanInputs())
    assert plan.regions == []


def test_regions_are_ordered_by_compositing_order():
    """Overlapping signage has to resolve to one deterministic stacking."""
    back = _track("back", compositing_order=2)
    front = _track("front", compositing_order=1)
    plan = plate(0, pts_seconds=0, width=160, height=120,
                 tracks={"back": back, "front": front},
                 observations=[_observation(0, "back"), _observation(0, "front")],
                 keyframes={}, inputs=PlanInputs())
    assert [region.track_id for region in plan.regions] == ["front", "back"]


def test_keyframe_geometry_reaches_the_plan():
    keys = [RenderKeyframe("k1", "t1", 0, bbox={"x": 0, "y": 0, "width": 20, "height": 10}),
            RenderKeyframe("k2", "t1", 10, bbox={"x": 20, "y": 40, "width": 20, "height": 10})]
    plan = _plate(5, [_observation(5)], keyframes={"t1": keys})
    assert plan.regions[0].bbox == BBox(x=10, y=20, width=20, height=10)


def test_a_quad_reaches_scribe_through_the_style_transform():
    quad = [[0, 0], [1, 0], [1, 1], [0, 1]]
    keys = [RenderKeyframe("k", "t1", 0, scope="track", quad=quad)]
    plan = _plate(3, [_observation(3)], keyframes={"t1": keys})
    assert plan.regions[0].style.transform["quad"] == quad
    assert plan.regions[0].quad == quad


def test_unimplemented_keyframe_effects_are_recorded_not_dropped():
    """The keyframe API accepts, validates, stores and resolves `effects`, and
    scribe implements none of them. Silently dropping them made a reviewer's
    deliberate choice indistinguishable from a no-op."""
    keys = [RenderKeyframe("k", "t1", 0, scope="track", effects={"glow": 3})]
    plan = _plate(0, [_observation(0)], keyframes={"t1": keys})
    codes = [d.code for d in plan.all_downgrades()]
    assert "effect_unsupported" in codes
    downgrade = next(d for d in plan.all_downgrades() if d.code == "effect_unsupported")
    assert downgrade.requested == {"glow": 3}
    assert downgrade.track_id == "t1"


def test_unknown_style_keys_are_recorded_not_silently_ignored():
    plan = _plate(0, [_observation(0)], tracks=[_track(style={"font_size": 12, "wobble": 4})])
    assert "style_key_unsupported" in [d.code for d in plan.all_downgrades()]
    assert plan.regions[0].style.font_size == 12


# --- invalidation ----------------------------------------------------------

def _inputs(**overrides):
    job = {"id": "job", "dependency_revision": 1}
    return plan_inputs({**job, **overrides.pop("job", {})},
                       overrides.pop("tracks", [_track()]),
                       overrides.pop("keyframes", []), **overrides)


def test_plan_revision_is_stable_for_identical_inputs():
    assert _inputs().revision == _inputs().revision


def test_editing_a_track_changes_the_plan_revision():
    assert _inputs().revision != _inputs(tracks=[_track(revision=2)]).revision


def test_moving_a_keyframe_changes_the_plan_revision():
    keys = [{"id": "k", "track_id": "t1", "frame_index": 3, "bbox": {"x": 1}}]
    assert _inputs().revision != _inputs(keyframes=keys).revision


def test_changing_the_font_library_changes_the_plan_revision():
    """Nothing used to invalidate when a font was installed or removed, so an
    export kept a filename promising a face it no longer rendered with."""
    class Registry:
        def __init__(self, faces): self._fonts = dict.fromkeys(faces)

    before = _inputs(font_registry=Registry(["Arial", "Noto Sans"]))
    after = _inputs(font_registry=Registry(["Arial"]))
    assert before.revision != after.revision


def test_schema_version_participates_in_the_revision():
    assert PlanInputs(plan_schema_version=PLAN_SCHEMA_VERSION).revision != \
           PlanInputs(plan_schema_version=PLAN_SCHEMA_VERSION + 1).revision


def test_downgrade_serializes_for_the_ledger():
    downgrade = Downgrade("font_fallback", "scribe", requested="Noto Sans JP",
                          applied="Arial", track_id="t1")
    payload = downgrade.to_dict()
    assert payload["code"] == "font_fallback" and payload["stage"] == "scribe"
    assert payload["severity"] == "review"
