import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "server"))

import main  # noqa: E402
from tofu.core.events import PipelineEvent, PipelineEventStatus  # noqa: E402


def _event(stage, operation, status, **payload):
    return PipelineEvent(
        stage=stage,
        operation=operation,
        status=status,
        payload=payload,
    )


def test_render_sse_adapter_preserves_public_stage_contract():
    core = [
        _event("pipeline", "process", PipelineEventStatus.STARTED),
        _event("tofu", "preflight", PipelineEventStatus.STARTED),
        _event(
            "tofu", "preflight", PipelineEventStatus.COMPLETED, passed=True
        ),
        _event("cicerone", "detect", PipelineEventStatus.COMPLETED),
        _event("tofu", "revalidate", PipelineEventStatus.COMPLETED),
        _event("scene", "enrich", PipelineEventStatus.STARTED),
        _event("scene", "enrich", PipelineEventStatus.COMPLETED),
        _event("cleanse", "erase", PipelineEventStatus.STARTED),
        _event("cleanse", "erase", PipelineEventStatus.COMPLETED),
        _event("scribe", "render", PipelineEventStatus.STARTED),
        _event("scribe", "render", PipelineEventStatus.COMPLETED),
        _event("verify", "assess", PipelineEventStatus.STARTED),
        _event(
            "verify", "assess", PipelineEventStatus.COMPLETED,
            overall_score=0.91,
        ),
        _event("memory", "update", PipelineEventStatus.COMPLETED),
        _event("pipeline", "process", PipelineEventStatus.COMPLETED),
    ]

    public = [
        payload
        for item in core
        if (payload := main._render_sse_event_payload(item)) is not None
    ]

    assert [(item["stage"], item["status"]) for item in public] == [
        ("tofu", "running"),
        ("tofu", "complete"),
        ("scene", "running"),
        ("scene", "complete"),
        ("cleanse", "running"),
        ("cleanse", "complete"),
        ("scribe", "running"),
        ("scribe", "complete"),
        ("verify", "running"),
        ("verify", "complete"),
    ]
    assert public[1]["passed"] is True
    assert public[-1]["score"] == 0.91


def test_render_sse_adapter_surfaces_core_failures():
    payload = main._render_sse_event_payload(
        _event("cleanse", "run", PipelineEventStatus.FAILED)
    )

    assert payload == {"stage": "cleanse", "status": "error"}
