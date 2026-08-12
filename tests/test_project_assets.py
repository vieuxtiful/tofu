import io
import sys
from pathlib import Path

import pytest
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "server"))

import main  # noqa: E402
import db  # noqa: E402
from tofu.core.types import BBox, InstText, TextManifest  # noqa: E402


@pytest.fixture
def client(tmp_path, monkeypatch):
    uploads = tmp_path / "uploads"
    uploads.mkdir()
    monkeypatch.setattr(main, "UPLOAD_DIR", uploads)
    monkeypatch.setattr(db, "DB_PATH", tmp_path / "test.db")
    db.init_db()
    from fastapi.testclient import TestClient
    with TestClient(main.app) as test_client:
        yield test_client


def png():
    output = io.BytesIO()
    Image.new("RGB", (8, 8), "white").save(output, "PNG")
    output.seek(0)
    return output


def project(client):
    return client.post("/api/projects", json={
        "name": "assets", "target_lang": "fr", "asset_kind": "image",
    }).json()


def upload(client, project_id, name, *, activate=True):
    response = client.post(
        "/api/assets", params={"project_id": project_id, "activate": activate},
        files={"file": (name, png(), "image/png")},
    )
    assert response.status_code == 200, response.text
    return response.json()


def test_nonactivating_upload_keeps_the_current_asset_and_exposes_all_urls(client):
    pid = project(client)["id"]
    first = upload(client, pid, "first.png")
    second = upload(client, pid, "second.png", activate=False)
    fetched = client.get(f"/api/projects/{pid}").json()
    assert fetched["active_asset"]["asset_id"] == first["asset_id"]
    assert len(fetched["assets"]) == 2
    assert all(item["asset_url"] for item in fetched["assets"])
    assert second["asset_id"] != first["asset_id"]


def test_deleting_active_asset_selects_the_newest_survivor(client):
    pid = project(client)["id"]
    first = upload(client, pid, "first.png")
    second = upload(client, pid, "second.png", activate=False)
    response = client.delete(f"/api/projects/{pid}/assets/{first['asset_id']}")
    assert response.json()["active_asset_id"] == second["asset_id"]
    assert client.get(f"/api/projects/{pid}").json()["active_asset"]["asset_id"] == second["asset_id"]


def test_deleting_inactive_asset_does_not_change_the_active_asset(client):
    pid = project(client)["id"]
    first = upload(client, pid, "first.png")
    second = upload(client, pid, "second.png", activate=False)
    response = client.delete(f"/api/projects/{pid}/assets/{second['asset_id']}")
    assert response.json()["active_asset_id"] is None
    assert client.get(f"/api/projects/{pid}").json()["active_asset"]["asset_id"] == first["asset_id"]


def test_project_limit_is_enforced_inside_link_transaction(client):
    pid = project(client)["id"]
    for index in range(30):
        db.link_asset(pid, f"a{index}", f"{index}.png", make_active=index == 0)
    response = client.post(
        "/api/assets", params={"project_id": pid, "activate": False},
        files={"file": ("overflow.png", png(), "image/png")},
    )
    assert response.status_code == 409
    assert len(db.get_project(pid)["assets"]) == 30


def test_switching_away_and_back_loads_each_assets_own_manifest(client):
    pid = project(client)["id"]
    first = upload(client, pid, "first.png")
    second = upload(client, pid, "second.png", activate=False)
    for uploaded, text in ((first, "FIRST"), (second, "SECOND")):
        main.save_manifest(main.UPLOAD_DIR, uploaded["asset_id"], TextManifest(
            asset_id=uploaded["asset_id"], total_regions=1,
            instances=[InstText(
                id="r1", reading_order=0, text=text,
                bounding_box=BBox(0, 0, 4, 4),
            )],
        ))

    switched = client.post(
        f"/api/projects/{pid}/assets/{second['asset_id']}/activate",
    )
    assert switched.status_code == 200
    assert switched.json()["manifest"]["instances"][0]["text"] == "SECOND"
    restored = client.post(
        f"/api/projects/{pid}/assets/{first['asset_id']}/activate",
    )
    assert restored.status_code == 200
    assert restored.json()["manifest"]["instances"][0]["text"] == "FIRST"
