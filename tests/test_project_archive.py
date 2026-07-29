import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "server"))

import db  # noqa: E402


def test_archive_filters_and_restores_without_losing_project(monkeypatch, tmp_path):
    monkeypatch.setattr(db, "DB_PATH", tmp_path / "projects.sqlite")
    db.init_db()
    project = db.create_project("Rue République", "en")
    assert [item["id"] for item in db.list_projects(archived=False)] == [project["id"]]

    archived = db.set_project_archived(project["id"], True)
    assert archived and archived["archived_at"] is not None
    assert db.list_projects(archived=False) == []
    assert db.list_projects(archived=True)[0]["id"] == project["id"]

    restored = db.set_project_archived(project["id"], False)
    assert restored and restored["archived_at"] is None
    assert db.list_projects(query="répub", sort="name")[0]["id"] == project["id"]
