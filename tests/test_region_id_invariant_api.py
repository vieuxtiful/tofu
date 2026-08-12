## 🍢 ToFU — the contiguity invariant, enforced at every write boundary
## vieuxtiful
"""Every mutating request leaves a manifest whose live regions are r1…rN.

WHY THIS IS AN AUTOUSE WRAPPER AND NOT A LIST OF CASES. The defect this
guards against is a mutation path nobody remembered to settle. A hand-written
test per endpoint covers the endpoints someone thought of -- which is exactly
the set that is already correct. Wrapping the client checks the ones nobody
thought of, including any added later.
"""
import io
import sys
from pathlib import Path

import pytest
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "server"))
sys.path.insert(0, str(ROOT))

import db  # noqa: E402
import main  # noqa: E402

from tofu.core.types import BBox, InstText, TextManifest  # noqa: E402
from tests.support.integrity import assert_manifest_integrity  # noqa: E402

MUTATING = {"POST", "PUT", "PATCH", "DELETE"}


class _Checked:
    """A TestClient that validates the persisted manifest after every write.

    The check runs on the manifest ON DISK, not on a response body: a route
    that returns a tidy payload while saving something inconsistent is
    precisely the failure being hunted.
    """

    def __init__(self, inner, uploads):
        self._inner = inner
        self._uploads = uploads

    def __getattr__(self, name):
        attr = getattr(self._inner, name)
        if name.upper() not in MUTATING:
            return attr

        def checked(url, *args, **kwargs):
            response = attr(url, *args, **kwargs)
            for asset_id in self._known_assets():
                manifest = main.load_manifest(self._uploads, asset_id)
                if manifest is not None:
                    assert_manifest_integrity(
                        manifest, context=f"after {name.upper()} {url}",
                    )
            return response

        return checked

    def _known_assets(self):
        return [path.name.split(".manifest.json")[0]
                for path in self._uploads.glob("*.manifest.json")]


@pytest.fixture
def client(tmp_path, monkeypatch):
    uploads, outputs = tmp_path / "uploads", tmp_path / "outputs"
    uploads.mkdir(parents=True)
    outputs.mkdir(parents=True)
    monkeypatch.setattr(main, "UPLOAD_DIR", uploads)
    monkeypatch.setattr(main, "OUTPUT_DIR", outputs)
    monkeypatch.setattr(db, "DB_PATH", tmp_path / "test.db")
    main._OPERATION_IDS.clear()
    db.init_db()
    from fastapi.testclient import TestClient

    with TestClient(main.app) as raw:
        yield _Checked(raw, uploads)


def _asset(client, blocks=()):
    pid = client.post("/api/projects", json={
        "name": "invariant", "target_lang": "en-US", "asset_kind": "image",
        "source_lang": "fr-FR",
    }).json()["id"]
    buf = io.BytesIO()
    Image.new("RGB", (400, 200), (255, 255, 255)).save(buf, format="PNG")
    buf.seek(0)
    asset_id = client.post(
        "/api/assets", files={"file": ("a.png", buf, "image/png")},
        params={"project_id": pid},
    ).json()["asset_id"]
    if blocks:
        client.put(f"/api/assets/{asset_id}/guided-blocks", json={"blocks": list(blocks)})
    return asset_id


def _draw(client, asset_id, *, block_id=None, x=10):
    body = {"x": x, "y": 10, "width": 60, "height": 20}
    if block_id:
        body["guided_block_id"] = block_id
    response = client.post(f"/api/manifest/{asset_id}/regions", json=body)
    assert response.status_code == 200, response.text
    return response.json()


def _ids(asset_id):
    manifest = main.load_manifest(main.UPLOAD_DIR, asset_id)
    return [i.id for i in manifest.instances if not i.excluded]


class TestContiguityAtEveryWrite:
    def test_an_empty_manifest_starts_at_r1(self, client):
        asset_id = _asset(client, ["SORTIE"])
        assert _draw(client, asset_id, block_id="g1")["id"] == "r1"

    def test_successive_draws_number_upward(self, client):
        asset_id = _asset(client, ["A", "B", "C"])
        for index, block in enumerate(["g1", "g2", "g3"], start=1):
            _draw(client, asset_id, block_id=block, x=10 + index * 70)
        assert _ids(asset_id) == ["r1", "r2", "r3"]

    def test_deleting_the_middle_region_closes_the_gap(self, client):
        asset_id = _asset(client, ["A", "B", "C"])
        for index, block in enumerate(["g1", "g2", "g3"], start=1):
            _draw(client, asset_id, block_id=block, x=10 + index * 70)
        client.delete(f"/api/manifest/{asset_id}/regions/r2")
        assert _ids(asset_id) == ["r1", "r2"]

    def test_deleting_the_first_region_shifts_every_survivor(self, client):
        asset_id = _asset(client, ["A", "B", "C"])
        for index, block in enumerate(["g1", "g2", "g3"], start=1):
            _draw(client, asset_id, block_id=block, x=10 + index * 70)
        client.delete(f"/api/manifest/{asset_id}/regions/r1")
        assert _ids(asset_id) == ["r1", "r2"]

    def test_a_draw_after_a_delete_does_not_reopen_a_gap(self, client):
        ## The old allocator's exact failure: a delete burned the number, the
        ## next draw started above it, and nothing ever brought it back down.
        asset_id = _asset(client, ["A", "B", "C"])
        _draw(client, asset_id, block_id="g1")
        _draw(client, asset_id, block_id="g2", x=90)
        client.delete(f"/api/manifest/{asset_id}/regions/r1")
        _draw(client, asset_id, block_id="g3", x=170)
        assert _ids(asset_id) == ["r1", "r2"]

    def test_guided_evidence_follows_the_renumbering(self, client):
        ## The half that matters. The ids moved, so every assessment naming
        ## them had to move too, or it now points at a different region.
        asset_id = _asset(client, ["A", "B"])
        _draw(client, asset_id, block_id="g1")
        _draw(client, asset_id, block_id="g2", x=90)
        client.delete(f"/api/manifest/{asset_id}/regions/r1")
        blocks = client.get(f"/api/assets/{asset_id}/guided-blocks").json()["blocks"]
        surviving = [b for b in blocks if b["detection_assessment"]["region_ids"]]
        assert len(surviving) == 1
        assert surviving[0]["raw_text"] == "B"
        assert surviving[0]["detection_assessment"]["region_ids"] == ["r1"]
        assert surviving[0]["detection_assessment"]["explicit_region_ids"] == ["r1"]

    def test_a_soft_deleted_region_leaves_the_ordinal_space(self, client):
        asset_id = _asset(client, ["A", "B"])
        _draw(client, asset_id, block_id="g1")
        _draw(client, asset_id, block_id="g2", x=90)
        client.delete(f"/api/manifest/{asset_id}/regions/r1")
        manifest = main.load_manifest(main.UPLOAD_DIR, asset_id)
        assert [i.id for i in manifest.instances if i.excluded] == ["x1"]

    def test_restoring_a_region_returns_it_to_the_ordinal_space(self, client):
        asset_id = _asset(client, ["A", "B"])
        _draw(client, asset_id, block_id="g1")
        _draw(client, asset_id, block_id="g2", x=90)
        client.delete(f"/api/manifest/{asset_id}/regions/r1")
        client.patch(f"/api/manifest/{asset_id}/regions/x1", json={"excluded": False})
        assert sorted(_ids(asset_id)) == ["r1", "r2"]

    def test_a_full_manifest_save_is_compacted_too(self, client):
        ## Undo, redo and snapshot restore all arrive as an ordinary full save,
        ## which is the route where a whole set of regions can appear at once.
        asset_id = _asset(client, ["A"])
        _draw(client, asset_id, block_id="g1")
        response = client.put(f"/api/manifest/{asset_id}", json={
            "asset_id": asset_id, "total_regions": 2, "src_lang": "fr-FR",
            "targ_lang": "en-US", "img_dim": [400, 200], "scene_regions": [],
            "semantic_units": [], "asset_type": "image", "frame_count": 1,
            "fps": None, "duration": None, "prcssng_time": None,
            "instances": [
                {"id": "r7", "bounding_box": {"x": 10, "y": 10, "width": 60, "height": 20},
                 "text": "A", "reading_order": 1},
                {"id": "r9", "bounding_box": {"x": 90, "y": 10, "width": 60, "height": 20},
                 "text": "B", "reading_order": 0},
            ],
        })
        assert response.status_code == 200, response.text
        ## Array order is preserved; the ORDINALS follow reading order, so the
        ## second entry (reading_order 0) becomes r1.
        assert _ids(asset_id) == ["r2", "r1"]


class TestTheBoundaryIsHard:
    """A manifest that cannot be made consistent does not reach disk."""

    def test_durable_corruption_refuses_the_write(self, client):
        ## A semantic unit naming a region that does not exist is curated
        ## state pointing at nothing -- the prem-sais-gt failure. Nothing
        ## regenerates it, so it cannot be reconciled away, and renumbering
        ## around it would make the stale reference resolve to a DIFFERENT
        ## region. The write is refused instead.
        asset_id = _asset(client, ["A"])
        _draw(client, asset_id, block_id="g1")
        before = (main.UPLOAD_DIR / f"{asset_id}.manifest.json").read_text(encoding="utf-8")

        response = client.put(f"/api/manifest/{asset_id}", json={
            "asset_id": asset_id, "total_regions": 1, "src_lang": "fr-FR",
            "targ_lang": "en-US", "img_dim": [400, 200], "scene_regions": [],
            "semantic_units": [{
                "id": "u1", "region_ids": ["r9"], "source_text": "gone",
                "bbox": {"x": 1, "y": 1, "width": 2, "height": 2},
            }],
            "asset_type": "image", "frame_count": 1,
            "fps": None, "duration": None, "prcssng_time": None,
            "instances": [{
                "id": "r1", "bounding_box": {"x": 10, "y": 10, "width": 60, "height": 20},
                "text": "A", "reading_order": 0,
            }],
        })
        assert response.status_code == 409, response.text
        body = response.json()
        assert body["asset_id"] == asset_id
        assert "r9" in body["detail"]
        ## The point of refusing: the stored manifest is untouched.
        after = (main.UPLOAD_DIR / f"{asset_id}.manifest.json").read_text(encoding="utf-8")
        assert after == before

    def test_stale_guided_evidence_is_reconciled_not_refused(self, client):
        ## The other half. `detection_assessment` is DERIVED -- reconcile
        ## rewrites it from the live regions -- so a full save that replaces
        ## the instances must succeed, with coverage recomputed rather than
        ## the write rejected.
        asset_id = _asset(client, ["A"])
        _draw(client, asset_id, block_id="g1")
        response = client.put(f"/api/manifest/{asset_id}", json={
            "asset_id": asset_id, "total_regions": 1, "src_lang": "fr-FR",
            "targ_lang": "en-US", "img_dim": [400, 200], "scene_regions": [],
            "semantic_units": [], "asset_type": "image", "frame_count": 1,
            "fps": None, "duration": None, "prcssng_time": None,
            "instances": [{
                "id": "r4", "bounding_box": {"x": 10, "y": 10, "width": 60, "height": 20},
                "text": "A", "reading_order": 0,
            }],
        })
        assert response.status_code == 200, response.text
        assert _ids(asset_id) == ["r1"]


def test_export_refuses_a_noncontiguous_manifest():
    manifest = TextManifest(
        asset_id="bad-export", total_regions=2,
        instances=[
            InstText(id="r1", bounding_box=BBox(0, 0, 10, 10), text="A"),
            InstText(id="r3", bounding_box=BBox(20, 0, 10, 10), text="B"),
        ],
    )
    with pytest.raises(main.CouvertCompactionError, match="not contiguous"):
        main._render_export(manifest, "xliff", "standard", "en")
