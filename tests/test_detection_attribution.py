## 🍢 ToFU — why this asset has the regions it has
## vieuxtiful
""""No boxes were drawn" is six different events wearing one face.

Zero regions on screen can mean the OCR engine was missing, the detector
proposed nothing, the detector proposed plenty and everything was
suppressed, or regions shipped and were all excluded afterwards. Those send
a user -- and an engineer -- to completely different remedies, and the old
toast ("no text regions detected. you can draw them manually.") said the
same thing for all of them.

This route reads the graph the shipping run recorded. It re-runs nothing:
a fresh permissive detector pass is a DIFFERENT run, and comparing one
against what shipped is exactly how an earlier attribution reported six
pipeline-destroyed regions where the production ancestry has one.
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

from tofu.core.types import BBox, InstText, TextManifest  # noqa: E402
from tofu.layers.okara import CandidateGraph  # noqa: E402


@pytest.fixture
def client(tmp_path, monkeypatch):
    uploads, outputs = tmp_path / "uploads", tmp_path / "outputs"
    uploads.mkdir(parents=True)
    outputs.mkdir(parents=True)
    monkeypatch.setattr(main, "UPLOAD_DIR", uploads)
    monkeypatch.setattr(main, "OUTPUT_DIR", outputs)
    monkeypatch.setattr(db, "DB_PATH", tmp_path / "test.db")
    db.init_db()
    from fastapi.testclient import TestClient
    with TestClient(main.app) as c:
        yield c


def _asset(client, *, instances=(), lineage=None):
    pid = client.post("/api/projects", json={
        "name": "attribution", "target_lang": "en-US", "asset_kind": "image",
    }).json()["id"]
    buf = io.BytesIO()
    Image.new("RGB", (200, 100), (255, 255, 255)).save(buf, format="PNG")
    buf.seek(0)
    asset_id = client.post(
        "/api/assets", files={"file": ("a.png", buf, "image/png")},
        params={"project_id": pid},
    ).json()["asset_id"]
    manifest = TextManifest(
        asset_id=asset_id, total_regions=sum(not item.excluded for item in instances), img_dim=(200, 100),
        instances=list(instances),
    )
    manifest.candidate_lineage = lineage
    main._settle_manifest(manifest)
    main.save_manifest(main.UPLOAD_DIR, asset_id, manifest)
    return asset_id


def _region(rid="r1", excluded=False):
    return InstText(id=rid, bounding_box=BBox(10, 10, 50, 20), text="SORTIE",
                    excluded=excluded)


def _graph(*, proposals=2, suppress=0):
    """A production graph with `proposals` raw candidates, `suppress` of them
    pruned at the merge stage."""
    graph = CandidateGraph(image_id="a", run_kind="production")
    ids = []
    for index in range(proposals):
        ids.append(graph.add(
            stage="raw_craft",
            geometry=(10 * index, 10, 50, 20),
            text="SORTIE", confidence=0.9,
        ))
    for cid in ids[:suppress]:
        graph.suppress(cid, "pruned", reason="prune_contained_fragments")
    return graph.to_dict()


class TestTheLadderIsNamed:
    def test_regions_present_is_the_ordinary_case(self, client):
        asset_id = _asset(client, instances=[_region()], lineage=_graph())
        body = client.get(f"/api/manifest/{asset_id}/detection-attribution").json()
        assert body["rung"] == "regions_present"

    def test_the_detector_proposing_nothing_is_its_own_rung(self, client):
        """Blindness at this resolution. Remedy: tiles, another detector, or
        fine-tuning -- all expensive."""
        asset_id = _asset(client, lineage=_graph(proposals=0))
        body = client.get(f"/api/manifest/{asset_id}/detection-attribution").json()
        assert body["rung"] == "no_proposals"
        assert body["lineage"]["raw_proposals"] == 0

    def test_everything_suppressed_is_a_different_rung(self, client):
        """The detector saw the text and the pipeline discarded it. Remedy:
        score calibration or a targeted retry -- much cheaper. Reporting
        this as blindness would send the work to the wrong place."""
        asset_id = _asset(client, lineage=_graph(proposals=3, suppress=3))
        body = client.get(f"/api/manifest/{asset_id}/detection-attribution").json()
        assert body["rung"] == "all_suppressed"
        assert body["lineage"]["raw_proposals"] == 3
        assert body["lineage"]["by_state"]["active"] == 0

    def test_regions_captured_then_all_excluded(self, client):
        asset_id = _asset(client, instances=[_region(excluded=True)], lineage=_graph())
        body = client.get(f"/api/manifest/{asset_id}/detection-attribution").json()
        assert body["rung"] == "all_excluded"
        assert body["regions"] == {"total": 1, "active": 0, "excluded": 1}

    def test_a_missing_graph_is_reported_as_missing_not_as_blindness(self, client):
        """A manifest captured before lineage was persisted has no evidence
        either way. Calling that "the detector proposed nothing" would be
        inventing a finding."""
        asset_id = _asset(client, lineage=None)
        body = client.get(f"/api/manifest/{asset_id}/detection-attribution").json()
        assert body["rung"] == "no_lineage"
        assert body["lineage"] is None

    def test_every_rung_carries_a_sentence(self, client):
        asset_id = _asset(client, lineage=_graph(proposals=0))
        body = client.get(f"/api/manifest/{asset_id}/detection-attribution").json()
        assert body["explanation"] == main.DETECTION_RUNGS[body["rung"]]


class TestWhatTheGraphSays:
    def test_suppression_is_attributed_to_a_stage_and_a_reason(self, client):
        asset_id = _asset(client, lineage=_graph(proposals=3, suppress=2))
        lineage = client.get(f"/api/manifest/{asset_id}/detection-attribution").json()["lineage"]
        assert lineage["suppressed_by_stage"]["raw_craft"]["pruned"] == 2
        assert lineage["suppression_reasons"]["prune_contained_fragments"] == 2

    def test_the_conservation_invariant_is_reported_not_assumed(self, client):
        asset_id = _asset(client, lineage=_graph(proposals=2))
        lineage = client.get(f"/api/manifest/{asset_id}/detection-attribution").json()["lineage"]
        assert lineage["orphans"] == 0

    def test_the_run_kind_is_carried(self, client):
        """`raw` is not a sufficient provenance label -- one image can have
        several raw proposal sets, and mixing them makes any claim about
        what the pipeline lost unfalsifiable."""
        asset_id = _asset(client, lineage=_graph())
        lineage = client.get(f"/api/manifest/{asset_id}/detection-attribution").json()["lineage"]
        assert lineage["run_kind"] == "production"

    def test_nothing_is_recomputed(self, client):
        """Byte-identical on repeat: the route reports a recorded run, and a
        re-detection would be a different one."""
        asset_id = _asset(client, lineage=_graph(proposals=3, suppress=1))
        first = client.get(f"/api/manifest/{asset_id}/detection-attribution").json()
        second = client.get(f"/api/manifest/{asset_id}/detection-attribution").json()
        assert first == second

    def test_a_missing_asset_is_404(self, client):
        assert client.get("/api/manifest/nope/detection-attribution").status_code == 404
