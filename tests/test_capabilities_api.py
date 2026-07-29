"""Unified runtime-capability contract."""

import asyncio
import sys
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "server"))

import main  # noqa: E402


def _get_capabilities():
    async def request():
        transport = httpx.ASGITransport(app=main.app)
        async with httpx.AsyncClient(
            transport=transport, base_url="http://testserver",
        ) as client:
            return await client.get("/api/capabilities")

    return asyncio.run(request())


def test_capabilities_contract_is_complete_and_safe(monkeypatch):
    monkeypatch.setattr(
        main.inpaint_providers,
        "provider_statuses",
        lambda: [{"id": "analytic", "available": True, "promoted": True}],
    )
    monkeypatch.setattr(
        "tofu.layers.basil.provider_statuses",
        lambda: [{"id": "deterministic_layout", "active": True}],
    )

    response = _get_capabilities()

    assert response.status_code == 200
    body = response.json()
    assert body["schema_version"] == "1.0"
    assert body["build"]["version"] == main.app.version
    assert {
        "compute", "ocr", "scene", "inpainting", "shaping", "semantics",
        "translation", "fonts", "video",
    } <= body.keys()
    assert body["video"]["available"] is False
    assert body["video"]["project_creation_enabled"] is False
    assert {item["id"] for item in body["ocr"]["providers"]} == {
        "easyocr", "paddleocr",
    }
    assert body["inpainting"]["providers"][0]["ready"] is True


def test_every_provider_has_common_status_fields():
    body = _get_capabilities().json()
    groups = (
        body["ocr"]["providers"],
        body["scene"]["providers"],
        body["inpainting"]["providers"],
        body["shaping"]["providers"],
        body["semantics"]["providers"],
        body["translation"]["providers"],
    )
    for provider in (item for group in groups for item in group):
        assert {"id", "available", "ready", "version", "reason"} <= provider.keys()
