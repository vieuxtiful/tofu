from PIL import Image

from tofu.core.events import PipelineEvent, PipelineEventStatus
from tofu.core.pipeline import TofuPipeline
from tofu.core.types import (
    AssetInfo,
    AssetType,
    QAReport,
    TextManifest,
    VldtnReport,
)


def test_pipeline_event_serializes_enum_for_transport():
    event = PipelineEvent(
        stage="cicerone",
        operation="detect",
        status=PipelineEventStatus.COMPLETED,
        progress=0.35,
        payload={"region_count": 2},
        duration_ms=14,
    )

    encoded = event.to_dict()

    assert encoded["status"] == "completed"
    assert encoded["payload"] == {"region_count": 2}
    assert encoded["timestamp"].endswith("+00:00")


def test_static_pipeline_emits_ordered_canonical_stage_events(monkeypatch):
    events = []
    pipeline = TofuPipeline(on_event=events.append)
    manifest = TextManifest(asset_id="fixture", total_regions=0, instances=[])
    valid = VldtnReport(passed=True)

    monkeypatch.setattr(pipeline, "_run_tofu", lambda *args, **kwargs: valid)
    monkeypatch.setattr(pipeline, "_run_cicerone", lambda *args, **kwargs: manifest)
    monkeypatch.setattr(pipeline, "_run_scene", lambda asset, value: value)
    monkeypatch.setattr(pipeline, "_run_cleanse", lambda asset, value: asset)
    monkeypatch.setattr(pipeline, "_run_scribe", lambda asset, *args, **kwargs: asset)
    monkeypatch.setattr(
        pipeline, "_run_verify",
        lambda *args, **kwargs: QAReport(overall_score=0.95),
    )
    monkeypatch.setattr(pipeline, "_run_memory", lambda *args, **kwargs: [])
    monkeypatch.setattr("tofu.core.pipeline.scene.analyze_regions", lambda asset: [])
    monkeypatch.setattr("tofu.core.pipeline.garnish.apply", lambda asset, *args: asset)
    # Inventory capture is independently non-fatal and immaterial to progress.
    monkeypatch.setattr(
        "tofu.core.pipeline.verify.build_verification_report",
        lambda *args: (_ for _ in ()).throw(RuntimeError("not part of fixture")),
    )

    result = pipeline.process(
        Image.new("RGB", (16, 16)),
        "fr",
        AssetInfo(asset_type=AssetType.IMAGE),
    )

    assert result.success
    completed = [
        (event.stage, event.operation)
        for event in events
        if event.status == PipelineEventStatus.COMPLETED
    ]
    assert completed == [
        ("tofu", "preflight"),
        ("scene", "prepass"),
        ("cicerone", "detect"),
        ("tofu", "revalidate"),
        ("scene", "enrich"),
        ("cleanse", "erase"),
        ("scribe", "render"),
        ("verify", "assess"),
        ("memory", "update"),
        ("pipeline", "process"),
    ]
    assert events[0].progress == 0.0
    assert events[-1].progress == 1.0


def test_observer_failure_does_not_fail_processing():
    def broken_observer(event):
        raise ConnectionError("client disconnected")

    pipeline = TofuPipeline(on_event=broken_observer)
    result = pipeline.process(
        object(),
        "en",
        AssetInfo(asset_type=AssetType.VIDEO),
    )

    assert not result.success
    assert result.errors == ["video pipeline not yet implemented"]
    assert pipeline._observer_errors
