import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "server"))

import db  # noqa: E402
import main  # noqa: E402


def test_project_and_asset_ground_truth_round_trip(monkeypatch, tmp_path):
    monkeypatch.setattr(db, "DB_PATH", tmp_path / "ground-truth.sqlite")
    db.init_db()
    project = db.create_project("signs", "en")
    assert project["ground_truth"] == []

    project = db.update_project(project["id"], ground_truth=["湯屋", "下島"])
    assert project and project["ground_truth"] == ["湯屋", "下島"]

    db.link_asset(project["id"], "a1", "sign.png")
    asset = db.update_asset_ground_truth("a1", ["濁河温泉"])
    assert asset and asset["ground_truth"] == ["濁河温泉"]

    restored = db.get_project(project["id"])
    assert restored and restored["ground_truth"] == ["湯屋", "下島"]
    assert restored["active_asset"]["ground_truth"] == ["濁河温泉"]


def test_effective_pool_normalizes_and_asset_scope_wins(monkeypatch):
    monkeypatch.setattr(main.db, "project_for_asset", lambda _asset_id: "p1")
    monkeypatch.setattr(main.db, "get_project", lambda _pid: {
        "source_lang": "ja",
        "ground_truth": [" 湯屋 ", "下島", "湯屋"],
        "assets": [{
            "asset_id": "a1",
            "ground_truth": ["湯屋", "濁河温泉"],
        }],
    })

    assert main._ground_truth_pool("a1", None) == [
        ("湯屋", "ja", "asset"),
        ("下島", "ja", "project"),
        ("濁河温泉", "ja", "asset"),
    ]
