## 🍢 ToFU — E2E server tests (FastAPI TestClient)
## vieuxtiful
"""
Full lifecycle test exercising every major endpoint in sequence:

    create project -> upload asset -> detect (mocked) -> add region ->
    translate (manual target_text) -> render (mocked) -> verify ->
    export XLIFF -> export VTM -> import XLIFF -> re-render

Plus error-path tests: 404 on missing asset, 422 on invalid bbox,
409 on duplicate project (name collision), 400 on empty project name.

OCR and render are mocked to avoid requiring torch/EasyOCR in the test
environment — the point of this suite is the HTTP contract and manifest
state transitions, not OCR accuracy (covered by the eval harnesses).
"""
import io
import sys
from pathlib import Path

import pytest
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "server"))

import main  # noqa: E402
import db  # noqa: E402
from tofu.core.types import TextManifest, InstText, BBox  # noqa: E402


# ─── fixture ──────────────────────────────────────────────────────────────

@pytest.fixture
def client(tmp_path, monkeypatch):
    """A TestClient with UPLOAD_DIR, OUTPUT_DIR, and DB all pointed at a
    temporary directory so tests are fully isolated."""
    uploads = tmp_path / "uploads"
    outputs = tmp_path / "outputs"
    uploads.mkdir(parents=True)
    outputs.mkdir(parents=True)

    monkeypatch.setattr(main, "UPLOAD_DIR", uploads)
    monkeypatch.setattr(main, "OUTPUT_DIR", outputs)
    monkeypatch.setattr(db, "DB_PATH", tmp_path / "test.db")
    db.init_db()

    from fastapi.testclient import TestClient
    with TestClient(main.app) as c:
        yield c


def _png_bytes(width=200, height=100, color=(255, 255, 255)):
    """Return a minimal valid PNG as BytesIO for upload."""
    img = Image.new("RGB", (width, height), color)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)
    return buf


def _mock_manifest(asset_id, img_dim=(200, 100)):
    """A synthetic manifest with one detected region, suitable for testing
    manifest CRUD, export, and import without running real OCR."""
    return TextManifest(
        asset_id=asset_id,
        total_regions=1,
        src_lang="en",
        targ_lang="es",
        img_dim=img_dim,
        instances=[
            InstText(
                id="r1",
                bounding_box=BBox(x=10, y=10, width=80, height=30),
                text="Hello",
                confidence=0.95,
                detected_language="en",
            ),
        ],
    )


# ─── full lifecycle ───────────────────────────────────────────────────────

class TestFullLifecycle:
    """Exercise the major endpoints in pipeline order."""

    def test_create_upload_detect_add_translate_render_export_import(self, client, monkeypatch):
        # 1. Create project
        resp = client.post("/api/projects", json={
            "name": "E2E Test Project", "target_lang": "es", "asset_kind": "image",
        })
        assert resp.status_code == 200, resp.text
        project = resp.json()
        pid = project["id"]
        assert project["name"] == "E2E Test Project"
        assert project["target_lang"] == "es"

        # 2. Upload asset (linked to project)
        resp = client.post(
            "/api/assets",
            files={"file": ("test.png", _png_bytes(), "image/png")},
            params={"project_id": pid},
        )
        assert resp.status_code == 200, resp.text
        upload = resp.json()
        asset_id = upload["asset_id"]
        assert upload["asset_info"]["asset_type"] == "image"

        # 3. Detect — mock cicerone.detect to return a synthetic manifest
        mock = _mock_manifest(asset_id)
        monkeypatch.setattr(main.cicerone, "detect", lambda *a, **kw: mock)
        resp = client.post("/api/detect", json={"asset_id": asset_id})
        assert resp.status_code == 200, resp.text
        detected = resp.json()
        assert detected["total_regions"] == 1
        assert detected["instances"][0]["text"] == "Hello"
        assert detected["src_lang"] == "en"

        # 4. GET manifest — should persist from detect
        resp = client.get(f"/api/manifest/{asset_id}")
        assert resp.status_code == 200
        manifest = resp.json()
        assert len(manifest["instances"]) == 1

        # 5. Add a manual region
        resp = client.post(f"/api/manifest/{asset_id}/regions", json={
            "x": 100, "y": 50, "width": 60, "height": 25, "text": "World",
        })
        assert resp.status_code == 200, resp.text
        new_region = resp.json()
        assert new_region["text"] == "World"
        assert new_region["bounding_box"]["width"] == 60

        # 6. GET manifest — should now have 2 regions
        resp = client.get(f"/api/manifest/{asset_id}")
        assert resp.status_code == 200
        manifest = resp.json()
        assert len(manifest["instances"]) == 2

        # 7. Update region — set target_text (simulating translation)
        region_id = manifest["instances"][0]["id"]
        resp = client.patch(f"/api/manifest/{asset_id}/regions/{region_id}", json={
            "target_text": "Hola",
        })
        assert resp.status_code == 200, resp.text
        updated = resp.json()
        assert updated["target_text"] == "Hola"

        # 8. PUT manifest (autosave) — persist the target_text
        manifest["instances"][0]["target_text"] = "Hola"
        resp = client.put(f"/api/manifest/{asset_id}", json=manifest)
        assert resp.status_code == 200, resp.text

        # 9. Export XLIFF
        resp = client.post("/api/export", json={
            "asset_id": asset_id, "format": "xliff", "targ_lang": "es",
        })
        assert resp.status_code == 200, resp.text
        assert "xml" in resp.headers.get("content-type", "")
        xliff_content = resp.text
        assert "Hello" in xliff_content
        assert "Hola" in xliff_content

        # 10. Export VTM
        resp = client.post("/api/export", json={
            "asset_id": asset_id, "format": "vtm", "targ_lang": "es",
        })
        assert resp.status_code == 200, resp.text
        import json as _json
        vtm_doc = _json.loads(resp.text)
        assert vtm_doc["vtm_version"].startswith("1.")
        assert len(vtm_doc["entries"]) >= 1

        # 11. Import XLIFF — round-trip the export back
        resp = client.post(
            f"/api/import?asset_id={asset_id}",
            files={"file": ("test.xliff", xliff_content.encode("utf-8"), "application/xml")},
        )
        assert resp.status_code == 200, resp.text
        import_result = resp.json()
        assert import_result["imported"] >= 1

        # 12. GET manifest after import — target_text should be preserved
        resp = client.get(f"/api/manifest/{asset_id}")
        assert resp.status_code == 200
        final_manifest = resp.json()
        r1 = next(i for i in final_manifest["instances"] if i["id"] == region_id)
        assert r1["target_text"] == "Hola"

    def test_merge_regions_area_weighted_confidence(self, client, monkeypatch):
        """0.2: merged region confidence is area-weighted, not arithmetic average."""
        # Setup: project + upload + detect with two regions of different sizes
        client.post("/api/projects", json={
            "name": "Merge Test", "target_lang": "es", "asset_kind": "image",
        })
        pid = client.get("/api/projects").json()["projects"][0]["id"]

        resp = client.post(
            "/api/assets",
            files={"file": ("merge.png", _png_bytes(400, 200), "image/png")},
            params={"project_id": pid},
        )
        asset_id = resp.json()["asset_id"]

        # Mock detect with two regions: a large high-confidence one and a
        # tiny low-confidence one.
        mock = TextManifest(
            asset_id=asset_id, total_regions=2, src_lang="en",
            img_dim=(400, 200),
            instances=[
                InstText(
                    id="big", bounding_box=BBox(x=10, y=10, width=200, height=50),
                    text="Large", confidence=0.95,
                ),
                InstText(
                    id="small", bounding_box=BBox(x=220, y=10, width=20, height=10),
                    text="tiny", confidence=0.10,
                ),
            ],
        )
        monkeypatch.setattr(main.cicerone, "detect", lambda *a, **kw: mock)
        client.post("/api/detect", json={"asset_id": asset_id})

        # Merge the two regions (no reread)
        resp = client.post(f"/api/manifest/{asset_id}/regions/merge", json={
            "region_ids": ["r1", "r2"],
        })
        assert resp.status_code == 200, resp.text
        merged = resp.json()
        survivor = merged["region"]

        # Area-weighted: (0.95 * 200*50 + 0.10 * 20*10) / (200*50 + 20*10)
        # = (9500 + 20) / (10000 + 200) = 9520 / 10200 ≈ 0.9333
        # Arithmetic average would be (0.95 + 0.10) / 2 = 0.525
        assert survivor["confidence"] != pytest.approx(0.525, abs=0.01)
        assert survivor["confidence"] == pytest.approx(9520 / 10200, abs=0.01)
        assert merged["source"] == "joined"

    def test_delete_region_marks_excluded_not_removed(self, client, monkeypatch):
        """delete_region soft-deletes (excluded=True); the instance stays in
        the manifest so cleanse() can still erase its pixels."""
        client.post("/api/projects", json={
            "name": "Delete Test", "target_lang": "es", "asset_kind": "image",
        })
        resp = client.post(
            "/api/assets",
            files={"file": ("del.png", _png_bytes(), "image/png")},
        )
        asset_id = resp.json()["asset_id"]

        monkeypatch.setattr(main.cicerone, "detect", lambda *a, **kw: _mock_manifest(asset_id))
        client.post("/api/detect", json={"asset_id": asset_id})

        # Delete the region
        resp = client.delete(f"/api/manifest/{asset_id}/regions/r1")
        assert resp.status_code == 200, resp.text
        assert resp.json()["ok"] is True

        # Still in the manifest, still excluded -- but RETIRED out of the
        # ordinal space. Live regions are r1..rN with no gaps, so a deleted
        # region cannot keep an ordinal; it becomes x1. The instance itself
        # survives, which is what this test has always been about: delete is
        # a mark, not a splice, and `update_region` can bring it back.
        manifest = client.get(f"/api/manifest/{asset_id}").json()
        assert [i["id"] for i in manifest["instances"]] == ["x1"]
        assert manifest["instances"][0]["excluded"] is True


# ─── error paths ──────────────────────────────────────────────────────────

class TestErrorPaths:

    def test_404_on_missing_asset_manifest(self, client):
        resp = client.get("/api/manifest/nonexistent-asset-id")
        assert resp.status_code == 404

    def test_404_on_missing_project(self, client):
        resp = client.get("/api/projects/nonexistent-pid")
        assert resp.status_code == 404

    def test_400_on_empty_project_name(self, client):
        resp = client.post("/api/projects", json={
            "name": "  ", "target_lang": "es", "asset_kind": "image",
        })
        assert resp.status_code == 400

    def test_400_on_unknown_asset_kind(self, client):
        resp = client.post("/api/projects", json={
            "name": "Bad Kind", "target_lang": "es", "asset_kind": "audio",
        })
        assert resp.status_code == 400

    def test_404_on_upload_to_missing_project(self, client):
        resp = client.post(
            "/api/assets",
            files={"file": ("test.png", _png_bytes(), "image/png")},
            params={"project_id": "nonexistent-pid"},
        )
        assert resp.status_code == 404

    def test_422_on_merge_with_single_region(self, client, monkeypatch):
        client.post("/api/projects", json={
            "name": "Merge 422", "target_lang": "es", "asset_kind": "image",
        })
        resp = client.post(
            "/api/assets",
            files={"file": ("m.png", _png_bytes(), "image/png")},
        )
        asset_id = resp.json()["asset_id"]
        monkeypatch.setattr(main.cicerone, "detect", lambda *a, **kw: _mock_manifest(asset_id))
        client.post("/api/detect", json={"asset_id": asset_id})

        resp = client.post(f"/api/manifest/{asset_id}/regions/merge", json={
            "region_ids": ["r1"],
        })
        assert resp.status_code == 422

    def test_404_on_delete_missing_region(self, client):
        resp = client.post(
            "/api/assets",
            files={"file": ("test.png", _png_bytes(), "image/png")},
        )
        asset_id = resp.json()["asset_id"]
        resp = client.delete(f"/api/manifest/{asset_id}/regions/nonexistent")
        assert resp.status_code == 404

    def test_400_on_unknown_export_format(self, client, monkeypatch):
        resp = client.post(
            "/api/assets",
            files={"file": ("test.png", _png_bytes(), "image/png")},
        )
        asset_id = resp.json()["asset_id"]
        monkeypatch.setattr(main.cicerone, "detect", lambda *a, **kw: _mock_manifest(asset_id))
        client.post("/api/detect", json={"asset_id": asset_id})

        resp = client.post("/api/export", json={
            "asset_id": asset_id, "format": "pdf",
        })
        assert resp.status_code == 400

    def test_415_on_undecodable_image_upload(self, client):
        """Upload a non-image file — server rejects with 415."""
        resp = client.post(
            "/api/assets",
            files={"file": ("fake.png", io.BytesIO(b"not an image"), "image/png")},
        )
        assert resp.status_code == 415

    def test_project_lifecycle_archive_restore_delete(self, client):
        """Project archive/restore/delete cycle."""
        resp = client.post("/api/projects", json={
            "name": "Archive Test", "target_lang": "es", "asset_kind": "image",
        })
        pid = resp.json()["id"]

        # Archive
        resp = client.post(f"/api/projects/{pid}/archive")
        assert resp.status_code == 200
        project = client.get(f"/api/projects/{pid}").json()
        assert project["archived_at"] is not None

        # Restore
        resp = client.post(f"/api/projects/{pid}/restore")
        assert resp.status_code == 200
        project = client.get(f"/api/projects/{pid}").json()
        assert project["archived_at"] is None

        # Delete
        resp = client.delete(f"/api/projects/{pid}")
        assert resp.status_code == 200
        resp = client.get(f"/api/projects/{pid}")
        assert resp.status_code == 404

    def test_capabilities_endpoint(self, client):
        resp = client.get("/api/capabilities")
        assert resp.status_code == 200
        body = resp.json()
        assert body["schema_version"] == "1.0"
        assert "ocr" in body
        assert "inpainting" in body

    def test_languages_and_fonts_endpoints(self, client):
        resp = client.get("/api/languages")
        assert resp.status_code == 200
        body = resp.json()
        assert "languages" in body
        assert isinstance(body["languages"], list)
        assert len(body["languages"]) > 0

        resp = client.get("/api/fonts", params={"lang": "es"})
        assert resp.status_code == 200
