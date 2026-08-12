## 🍢 ToFU — Basil export surface (readiness + prepare/download)
## vieuxtiful
"""
Release 1 moved export ownership to Basil.  Two properties matter and are
pinned here:

  1. the prepare/download flow and the legacy /api/export route produce
     BYTE-IDENTICAL files, because they share one writer;
  2. the expeditor's verdict distinguishes advisory findings (export still
     allowed) from structural contradictions (no file is built at all).

Readiness is checked at Basil's boundary rather than inside each format
writer, so these tests drive `expedite()` directly as well as over HTTP.
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
from tofu.core.types import TextManifest, InstText, BBox, SemanticTextUnit  # noqa: E402
from tofu.layers.basil import expedite  # noqa: E402


@pytest.fixture
def client(tmp_path, monkeypatch):
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


def _region(rid, text, target="", **kw):
    return InstText(
        id=rid, bounding_box=BBox(x=10, y=10, width=80, height=30),
        text=text, target_text=target, confidence=0.9, **kw,
    )


def _manifest(asset_id="a1", units=None, **kw):
    manifest = TextManifest(
        asset_id=asset_id, total_regions=2,
        src_lang=kw.pop("src_lang", "fr"), targ_lang=kw.pop("targ_lang", "en"),
        img_dim=(200, 100),
        instances=kw.pop("instances", [_region("r1", "Rue", "Street"), _region("r2", "des", "of the")]),
    )
    manifest.semantic_units = units if units is not None else [
        SemanticTextUnit(
            id="u1", region_ids=["r1", "r2"], source_text="Rue des",
            bbox=BBox(x=10, y=10, width=80, height=30), review_required=False,
        )
    ]
    return manifest


def _upload(client, manifest):
    """Register a project + asset, then overwrite its manifest with ours."""
    project = client.post("/api/projects", json={
        "name": f"p-{manifest.asset_id}", "target_lang": "en", "asset_kind": "image",
    })
    assert project.status_code == 200, project.text
    img = Image.new("RGB", (200, 100), (255, 255, 255))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)
    up = client.post(
        "/api/assets",
        files={"file": ("a.png", buf, "image/png")},
        params={"project_id": project.json()["id"]},
    )
    assert up.status_code == 200, up.text
    asset_id = up.json()["asset_id"]
    manifest.asset_id = asset_id
    from tofu.utils.manifest_store import set_save_guard
    damaged = any(
        rid not in {item.id for item in manifest.instances}
        for unit in manifest.semantic_units for rid in unit.region_ids
    )
    if damaged:
        previous = set_save_guard(None)
        try:
            main.save_manifest(main.UPLOAD_DIR, asset_id, manifest)
        finally:
            set_save_guard(previous)
    else:
        main.save_manifest(main.UPLOAD_DIR, asset_id, manifest)
    return asset_id


# ─── expeditor verdicts ───────────────────────────────────────────────────

class TestExpedite:
    def test_clean_manifest_is_ready(self):
        readiness = expedite(_manifest())
        assert readiness.state == "ready"
        assert readiness.diagnostics == []
        assert readiness.counts["regions"] == 2
        assert readiness.counts["plates"] == 1

    def test_missing_region_blocks(self):
        manifest = _manifest()
        manifest.semantic_units[0].region_ids = ["r1", "r2", "ghost"]
        readiness = expedite(manifest)
        assert readiness.state == "blocked"
        assert [d["code"] for d in readiness.diagnostics] == ["plate_missing_region"]
        assert readiness.diagnostics[0]["region_ids"] == ["ghost"]

    def test_region_in_two_plates_blocks(self):
        manifest = _manifest(units=[
            SemanticTextUnit(id="u1", region_ids=["r1"], source_text="Rue",
                             bbox=BBox(10, 10, 80, 30), review_required=False),
            SemanticTextUnit(id="u2", region_ids=["r1"], source_text="Rue",
                             bbox=BBox(10, 10, 80, 30), review_required=False),
        ])
        readiness = expedite(manifest)
        assert readiness.state == "blocked"
        assert "region_in_multiple_plates" in [d["code"] for d in readiness.diagnostics]

    def test_duplicate_plate_id_blocks(self):
        manifest = _manifest(units=[
            SemanticTextUnit(id="u1", region_ids=["r1"], source_text="Rue",
                             bbox=BBox(10, 10, 80, 30), review_required=False),
            SemanticTextUnit(id="u1", region_ids=["r2"], source_text="des",
                             bbox=BBox(10, 10, 80, 30), review_required=False),
        ])
        assert expedite(manifest).state == "blocked"
        assert "duplicate_plate_id" in [d["code"] for d in expedite(manifest).diagnostics]

    def test_missing_language_metadata_blocks(self):
        assert expedite(_manifest(targ_lang=None)).state == "blocked"
        # an explicit target passed at prepare time satisfies it
        assert expedite(_manifest(targ_lang=None), "en-US").state == "ready"

    def test_drifted_plate_source_is_stale_not_blocked(self):
        manifest = _manifest()
        manifest.semantic_units[0].source_text = "Something else entirely"
        readiness = expedite(manifest)
        assert readiness.state == "stale"

    def test_accepted_ocr_repair_is_not_drift(self):
        """A repaired plate deliberately differs from what its members read."""
        manifest = _manifest()
        manifest.semantic_units[0].source_text = "Rue del"
        manifest.semantic_units[0].ocr_repair = {"accepted": True}
        assert expedite(manifest).state == "ready"

    def test_review_is_advisory_only(self):
        manifest = _manifest()
        manifest.semantic_units[0].review_required = True
        readiness = expedite(manifest)
        assert readiness.state == "review"
        assert all(d["severity"] == "advisory" for d in readiness.diagnostics)
        assert readiness.counts["plates_needing_review"] == 1

    def test_untranslated_regions_are_advisory(self):
        manifest = _manifest(instances=[_region("r1", "Rue"), _region("r2", "des")])
        manifest.semantic_units = [SemanticTextUnit(
            id="u1", region_ids=["r1", "r2"], source_text="Rue des",
            bbox=BBox(10, 10, 80, 30), review_required=False)]
        readiness = expedite(manifest)
        assert readiness.state == "review"
        assert "untranslated_regions" in [d["code"] for d in readiness.diagnostics]

    def test_state_never_contradicts_its_diagnostics(self):
        """`ready` must be impossible alongside a blocking record."""
        for manifest in (_manifest(), _manifest(targ_lang=None)):
            readiness = expedite(manifest)
            blocking = [d for d in readiness.diagnostics if d["severity"] == "blocking"]
            assert (readiness.state in {"blocked", "stale"}) == bool(blocking)


# ─── HTTP surface ─────────────────────────────────────────────────────────

class TestPrepareAndDownload:
    def test_prepare_then_download_matches_legacy_export_bytes(self, client):
        """The gate for Release 1: relocation must not change the file."""
        asset_id = _upload(client, _manifest())
        legacy = client.post("/api/export", json={
            "asset_id": asset_id, "format": "xliff", "variant": "sdl", "targ_lang": "en-US",
        })
        assert legacy.status_code == 200

        prepared = client.post("/api/basil/export/prepare", json={
            "asset_id": asset_id, "format": "xliff", "variant": "sdl", "target_language": "en-US",
        })
        assert prepared.status_code == 200, prepared.text
        body = prepared.json()
        assert body["export_id"]
        assert body["filename"].endswith(".xliff")

        fetched = client.get(f"/api/basil/export/{body['export_id']}/download")
        assert fetched.status_code == 200
        assert fetched.content == legacy.content

    @pytest.mark.parametrize("fmt", ["xliff", "tmx", "tsv", "csv", "txt"])
    def test_every_offered_format_round_trips_identically(self, client, fmt):
        asset_id = _upload(client, _manifest())
        legacy = client.post("/api/export", json={
            "asset_id": asset_id, "format": fmt, "variant": "standard", "targ_lang": "en-US",
        })
        prepared = client.post("/api/basil/export/prepare", json={
            "asset_id": asset_id, "format": fmt, "variant": "standard", "target_language": "en-US",
        }).json()
        fetched = client.get(f"/api/basil/export/{prepared['export_id']}/download")
        assert fetched.content == legacy.content

    def test_blocked_manifest_yields_no_download_id(self, client):
        manifest = _manifest()
        manifest.semantic_units[0].region_ids = ["r1", "r2", "ghost"]
        asset_id = _upload(client, manifest)
        body = client.post("/api/basil/export/prepare", json={
            "asset_id": asset_id, "format": "xliff", "variant": "standard", "target_language": "en-US",
        }).json()
        assert body["readiness"]["state"] == "blocked"
        assert body["export_id"] is None
        assert body["filename"] is None

    def test_review_findings_still_export(self, client):
        manifest = _manifest()
        manifest.semantic_units[0].review_required = True
        asset_id = _upload(client, manifest)
        body = client.post("/api/basil/export/prepare", json={
            "asset_id": asset_id, "format": "xliff", "variant": "standard", "target_language": "en-US",
        }).json()
        assert body["readiness"]["state"] == "review"
        assert body["export_id"]

    def test_review_units_can_be_refused(self, client):
        manifest = _manifest()
        manifest.semantic_units[0].review_required = True
        asset_id = _upload(client, manifest)
        body = client.post("/api/basil/export/prepare", json={
            "asset_id": asset_id, "format": "xliff", "variant": "standard",
            "target_language": "en-US", "include_review_units": False,
        }).json()
        assert body["export_id"] is None

    def test_unknown_export_id_is_404(self, client):
        assert client.get("/api/basil/export/deadbeef/download").status_code == 404

    def test_prepare_on_missing_asset_is_404(self, client):
        assert client.post("/api/basil/export/prepare", json={
            "asset_id": "nope", "format": "xliff", "target_language": "en",
        }).status_code == 404

    def test_prepared_exports_do_not_grow_without_bound(self, client):
        asset_id = _upload(client, _manifest())
        for _ in range(main._PREPARED_EXPORT_LIMIT + 5):
            client.post("/api/basil/export/prepare", json={
                "asset_id": asset_id, "format": "txt", "target_language": "en-US",
            })
        assert len(main._PREPARED_EXPORTS) <= main._PREPARED_EXPORT_LIMIT


# ─── semantic profile (Release 3) ─────────────────────────────────────────

class TestSemanticProfileOverHttp:
    def test_semantic_profile_exports_plates_not_regions(self, client):
        asset_id = _upload(client, _manifest())
        region_based = client.post("/api/basil/export/prepare", json={
            "asset_id": asset_id, "format": "xliff", "variant": "standard",
            "target_language": "en-US", "profile": "translation",
        }).json()
        semantic = client.post("/api/basil/export/prepare", json={
            "asset_id": asset_id, "format": "xliff", "variant": "standard",
            "target_language": "en-US", "profile": "semantic",
        }).json()
        assert region_based["export_id"] and semantic["export_id"]
        a = client.get(f"/api/basil/export/{region_based['export_id']}/download").text
        b = client.get(f"/api/basil/export/{semantic['export_id']}/download").text
        assert a != b
        assert "plate:" not in a and "plate:" in b

    def test_default_profile_is_unchanged_region_based_output(self, client):
        """Callers that do not opt in must keep getting the legacy file."""
        asset_id = _upload(client, _manifest())
        legacy = client.post("/api/export", json={
            "asset_id": asset_id, "format": "xliff", "variant": "standard", "targ_lang": "en-US",
        }).text
        default = client.post("/api/basil/export/prepare", json={
            "asset_id": asset_id, "format": "xliff", "variant": "standard",
            "target_language": "en-US",
        }).json()
        assert client.get(f"/api/basil/export/{default['export_id']}/download").text == legacy

    def test_tmx_profiles_are_selectable(self, client):
        asset_id = _upload(client, _manifest())
        counts = {}
        for profile in ("semantic", "atomic", "both"):
            prepared = client.post("/api/basil/export/prepare", json={
                "asset_id": asset_id, "format": "tmx",
                "target_language": "en-US", "profile": profile,
            }).json()
            counts[profile] = client.get(
                f"/api/basil/export/{prepared['export_id']}/download"
            ).text.count("<tu ")
        assert counts["both"] == counts["semantic"] + counts["atomic"]

    def test_import_detects_the_scheme_from_the_file(self, client):
        asset_id = _upload(client, _manifest())
        prepared = client.post("/api/basil/export/prepare", json={
            "asset_id": asset_id, "format": "xliff",
            "target_language": "en-US", "profile": "semantic",
        }).json()
        semantic_file = client.get(
            f"/api/basil/export/{prepared['export_id']}/download"
        ).text
        report = client.post(
            f"/api/import?asset_id={asset_id}",
            files={"file": ("back.xliff", semantic_file.encode("utf-8"), "application/xml")},
        ).json()
        assert report["scheme"] == "semantic"

        legacy_file = client.post("/api/export", json={
            "asset_id": asset_id, "format": "xliff", "targ_lang": "en-US",
        }).text
        legacy_report = client.post(
            f"/api/import?asset_id={asset_id}",
            files={"file": ("legacy.xliff", legacy_file.encode("utf-8"), "application/xml")},
        ).json()
        assert legacy_report["scheme"] == "legacy-region"
