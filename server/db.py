## 🍢 ToFU — project database (SQLite)
## vieuxtiful
"""
Persistent project / session storage for localization management.

Schema:
  projects        one row per localization project (name, langs, timestamps)
  project_assets  assets uploaded into a project; one may be "active"
                  (the asset currently open in the editing session)
  snapshots       full manifest JSON captured over time (autosave, pre-erase
                  guards, imports) — the "data is never lost" ledger
  events          human-readable session history (uploads, detections,
                  imports, renders, restores)

All writes are idempotent and safe under FastAPI's threaded request model:
each call opens a short-lived connection (SQLite serializes writers; WAL
keeps readers unblocked).
"""

import json
import sqlite3
import time
import uuid
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional

DB_PATH = Path(__file__).resolve().parent / "tofu.db"

# snapshots kept per asset; oldest autosaves pruned first, but protected
# reasons (pre-erase, import, manual) are never auto-pruned
SNAPSHOT_CAP = 50
PROTECTED_REASONS = ("pre-erase", "import", "manual", "restore-backup")

_SCHEMA = """
CREATE TABLE IF NOT EXISTS projects (
  id          TEXT PRIMARY KEY,
  name        TEXT NOT NULL,
  target_lang TEXT NOT NULL,
  source_lang TEXT,
  asset_kind  TEXT NOT NULL DEFAULT 'image',
  created_at  REAL NOT NULL,
  updated_at  REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS project_assets (
  asset_id     TEXT PRIMARY KEY,
  project_id   TEXT NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
  filename     TEXT,
  uploaded_at  REAL NOT NULL,
  is_active    INTEGER NOT NULL DEFAULT 0,
  content_hash TEXT
);
CREATE TABLE IF NOT EXISTS snapshots (
  id            INTEGER PRIMARY KEY AUTOINCREMENT,
  project_id    TEXT NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
  asset_id      TEXT NOT NULL,
  manifest_json TEXT NOT NULL,
  reason        TEXT NOT NULL DEFAULT 'autosave',
  region_count  INTEGER NOT NULL DEFAULT 0,
  translated    INTEGER NOT NULL DEFAULT 0,
  created_at    REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_snapshots_asset ON snapshots(asset_id, created_at DESC);
CREATE TABLE IF NOT EXISTS events (
  id         INTEGER PRIMARY KEY AUTOINCREMENT,
  project_id TEXT NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
  kind       TEXT NOT NULL,
  detail     TEXT,
  created_at REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_events_project ON events(project_id, created_at DESC);
CREATE TABLE IF NOT EXISTS tm_records (
  id                INTEGER PRIMARY KEY AUTOINCREMENT,
  project_id        TEXT NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
  asset_id          TEXT NOT NULL,
  region_id         TEXT NOT NULL,
  source_text       TEXT NOT NULL,
  normalized_text   TEXT NOT NULL,
  source_lang       TEXT,
  target_lang       TEXT NOT NULL,
  target_text       TEXT NOT NULL,
  style_fingerprint TEXT,
  phash             TEXT,
  thumb_path        TEXT,
  qa_score          REAL NOT NULL,
  created_at        REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_tm_project ON tm_records(project_id, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_tm_lookup ON tm_records(project_id, source_lang, target_lang);
CREATE INDEX IF NOT EXISTS idx_tm_normalized ON tm_records(normalized_text);
"""


@contextmanager
def _conn() -> Iterator[sqlite3.Connection]:
    """open → commit/rollback → CLOSE. sqlite3's bare `with conn:` commits
    but never closes; leaked handles on Windows escalate into
    'database is locked' (HTTP 500s) under normal frontend traffic."""
    con = sqlite3.connect(DB_PATH, timeout=10)
    con.row_factory = sqlite3.Row
    con.execute("PRAGMA journal_mode=WAL")
    con.execute("PRAGMA foreign_keys=ON")
    try:
        yield con
        con.commit()
    except Exception:
        con.rollback()
        raise
    finally:
        con.close()


def init_db() -> None:
    with _conn() as con:
        con.executescript(_SCHEMA)
        # migration: projects created before the image/video paradigm
        cols = {r["name"] for r in con.execute("PRAGMA table_info(projects)")}
        if "asset_kind" not in cols:
            con.execute(
                "ALTER TABLE projects ADD COLUMN asset_kind TEXT NOT NULL DEFAULT 'image'"
            )
        # migration: assets uploaded before duplicate-image detection existed
        asset_cols = {r["name"] for r in con.execute("PRAGMA table_info(project_assets)")}
        if "content_hash" not in asset_cols:
            con.execute("ALTER TABLE project_assets ADD COLUMN content_hash TEXT")
        con.execute("CREATE INDEX IF NOT EXISTS idx_assets_hash ON project_assets(content_hash)")


# --- projects ---

def create_project(name: str, target_lang: str,
                   asset_kind: str = "image") -> Dict[str, Any]:
    pid = uuid.uuid4().hex[:12]
    now = time.time()
    with _conn() as con:
        con.execute(
            "INSERT INTO projects (id, name, target_lang, asset_kind, created_at, updated_at)"
            " VALUES (?, ?, ?, ?, ?, ?)",
            (pid, name, target_lang, asset_kind, now, now),
        )
    log_event(pid, "project-created",
              f"{asset_kind} project '{name}' created (target: {target_lang})")
    return get_project(pid)  # type: ignore[return-value]


def list_projects() -> List[Dict[str, Any]]:
    with _conn() as con:
        rows = con.execute(
            """SELECT p.*,
                      (SELECT COUNT(*) FROM project_assets a WHERE a.project_id = p.id) AS asset_count,
                      (SELECT COUNT(*) FROM snapshots s WHERE s.project_id = p.id) AS snapshot_count
               FROM projects p ORDER BY p.updated_at DESC"""
        ).fetchall()
    return [dict(r) for r in rows]


def get_project(pid: str) -> Optional[Dict[str, Any]]:
    with _conn() as con:
        row = con.execute("SELECT * FROM projects WHERE id = ?", (pid,)).fetchone()
        if row is None:
            return None
        assets = con.execute(
            "SELECT * FROM project_assets WHERE project_id = ?"
            " ORDER BY uploaded_at DESC", (pid,)
        ).fetchall()
    out = dict(row)
    out["assets"] = [dict(a) for a in assets]
    out["active_asset"] = next((dict(a) for a in assets if a["is_active"]), None)
    return out


def update_project(pid: str, *, name: Optional[str] = None,
                   target_lang: Optional[str] = None,
                   source_lang: Optional[str] = None) -> Optional[Dict[str, Any]]:
    sets, vals = [], []
    if name is not None:
        sets.append("name = ?"); vals.append(name)
    if target_lang is not None:
        sets.append("target_lang = ?"); vals.append(target_lang)
    if source_lang is not None:
        sets.append("source_lang = ?"); vals.append(source_lang)
    if not sets:
        return get_project(pid)
    sets.append("updated_at = ?"); vals.append(time.time())
    vals.append(pid)
    with _conn() as con:
        con.execute(f"UPDATE projects SET {', '.join(sets)} WHERE id = ?", vals)
    return get_project(pid)


def delete_project(pid: str) -> bool:
    with _conn() as con:
        cur = con.execute("DELETE FROM projects WHERE id = ?", (pid,))
    return cur.rowcount > 0


def touch_project(pid: str) -> None:
    with _conn() as con:
        con.execute("UPDATE projects SET updated_at = ? WHERE id = ?", (time.time(), pid))


# --- assets ---

def link_asset(pid: str, asset_id: str, filename: Optional[str],
               content_hash: Optional[str] = None) -> None:
    """attach an uploaded asset to a project and make it the active one."""
    now = time.time()
    with _conn() as con:
        con.execute(
            "UPDATE project_assets SET is_active = 0 WHERE project_id = ?", (pid,)
        )
        con.execute(
            "INSERT OR REPLACE INTO project_assets"
            " (asset_id, project_id, filename, uploaded_at, is_active, content_hash)"
            " VALUES (?, ?, ?, ?, 1, ?)",
            (asset_id, pid, filename, now, content_hash),
        )
        con.execute("UPDATE projects SET updated_at = ? WHERE id = ?", (now, pid))


def find_asset_by_hash(content_hash: str) -> Optional[Dict[str, Any]]:
    """most recent asset (across all projects) matching an exact content
    hash, joined with its project's name -- the duplicate-upload check."""
    with _conn() as con:
        row = con.execute(
            """SELECT a.asset_id, a.filename, a.project_id, p.name AS project_name
               FROM project_assets a JOIN projects p ON p.id = a.project_id
               WHERE a.content_hash = ?
               ORDER BY a.uploaded_at DESC LIMIT 1""",
            (content_hash,),
        ).fetchone()
    return dict(row) if row else None


def set_active_asset(pid: str, asset_id: str) -> None:
    with _conn() as con:
        con.execute("UPDATE project_assets SET is_active = 0 WHERE project_id = ?", (pid,))
        con.execute(
            "UPDATE project_assets SET is_active = 1"
            " WHERE project_id = ? AND asset_id = ?", (pid, asset_id)
        )
        con.execute("UPDATE projects SET updated_at = ? WHERE id = ?", (time.time(), pid))


def unlink_asset(pid: str, asset_id: str) -> bool:
    """remove an asset from a project. snapshots are the never-lose-data
    ledger and are deliberately retained; the upload file stays on disk."""
    with _conn() as con:
        cur = con.execute(
            "DELETE FROM project_assets WHERE project_id = ? AND asset_id = ?",
            (pid, asset_id),
        )
        con.execute("UPDATE projects SET updated_at = ? WHERE id = ?", (time.time(), pid))
    return cur.rowcount > 0


def delete_snapshot(sid: int) -> bool:
    with _conn() as con:
        cur = con.execute("DELETE FROM snapshots WHERE id = ?", (sid,))
    return cur.rowcount > 0


def project_for_asset(asset_id: str) -> Optional[str]:
    with _conn() as con:
        row = con.execute(
            "SELECT project_id FROM project_assets WHERE asset_id = ?", (asset_id,)
        ).fetchone()
    return row["project_id"] if row else None


# --- snapshots (the never-lose-data ledger) ---

def add_snapshot(pid: str, asset_id: str, manifest: Dict[str, Any],
                 reason: str = "autosave") -> Optional[int]:
    """store a full manifest snapshot. identical consecutive autosaves are
    deduped so the ledger records change, not chatter."""
    payload = json.dumps(manifest, ensure_ascii=False)
    instances = manifest.get("instances") or []
    region_count = len(instances)
    translated = sum(1 for i in instances if i.get("target_text"))
    now = time.time()
    with _conn() as con:
        if reason == "autosave":
            last = con.execute(
                "SELECT manifest_json FROM snapshots WHERE asset_id = ?"
                " ORDER BY created_at DESC LIMIT 1", (asset_id,)
            ).fetchone()
            if last and last["manifest_json"] == payload:
                return None
        cur = con.execute(
            "INSERT INTO snapshots (project_id, asset_id, manifest_json, reason,"
            " region_count, translated, created_at) VALUES (?, ?, ?, ?, ?, ?, ?)",
            (pid, asset_id, payload, reason, region_count, translated, now),
        )
        # prune: keep newest SNAPSHOT_CAP autosaves per asset; protected
        # reasons stay forever
        con.execute(
            """DELETE FROM snapshots WHERE asset_id = ? AND reason = 'autosave'
               AND id NOT IN (
                 SELECT id FROM snapshots WHERE asset_id = ? AND reason = 'autosave'
                 ORDER BY created_at DESC LIMIT ?
               )""",
            (asset_id, asset_id, SNAPSHOT_CAP),
        )
        con.execute("UPDATE projects SET updated_at = ? WHERE id = ?", (now, pid))
        return cur.lastrowid


def list_snapshots(pid: str, asset_id: Optional[str] = None,
                   limit: int = 100) -> List[Dict[str, Any]]:
    """snapshot metadata (no payload — fetch individually to restore)."""
    q = ("SELECT id, project_id, asset_id, reason, region_count, translated,"
         " created_at FROM snapshots WHERE project_id = ?")
    vals: List[Any] = [pid]
    if asset_id:
        q += " AND asset_id = ?"; vals.append(asset_id)
    q += " ORDER BY created_at DESC LIMIT ?"; vals.append(limit)
    with _conn() as con:
        rows = con.execute(q, vals).fetchall()
    return [dict(r) for r in rows]


def get_snapshot(sid: int) -> Optional[Dict[str, Any]]:
    with _conn() as con:
        row = con.execute("SELECT * FROM snapshots WHERE id = ?", (sid,)).fetchone()
    if row is None:
        return None
    out = dict(row)
    out["manifest"] = json.loads(out.pop("manifest_json"))
    return out


# --- events (session history) ---

def log_event(pid: str, kind: str, detail: str = "") -> None:
    with _conn() as con:
        con.execute(
            "INSERT INTO events (project_id, kind, detail, created_at) VALUES (?, ?, ?, ?)",
            (pid, kind, detail, time.time()),
        )


def list_events(pid: str, limit: int = 200) -> List[Dict[str, Any]]:
    with _conn() as con:
        rows = con.execute(
            "SELECT * FROM events WHERE project_id = ? ORDER BY created_at DESC LIMIT ?",
            (pid, limit),
        ).fetchall()
    return [dict(r) for r in rows]


# --- translation memory (visual TM: Phase 6) ---

def store_tm_record(
    pid: str, asset_id: str, region_id: str, source_text: str,
    normalized_text: str, source_lang: Optional[str], target_lang: str,
    target_text: str, style_fingerprint: Optional[str], phash: Optional[str],
    thumb_path: Optional[str], qa_score: float,
) -> int:
    with _conn() as con:
        cur = con.execute(
            "INSERT INTO tm_records (project_id, asset_id, region_id, source_text,"
            " normalized_text, source_lang, target_lang, target_text,"
            " style_fingerprint, phash, thumb_path, qa_score, created_at)"
            " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (pid, asset_id, region_id, source_text, normalized_text, source_lang,
             target_lang, target_text, style_fingerprint, phash, thumb_path,
             qa_score, time.time()),
        )
        return cur.lastrowid


def find_tm_candidates(
    pid: str, target_lang: str, source_lang: Optional[str] = None,
    limit: int = 2000,
) -> List[Dict[str, Any]]:
    """the candidate pool for a lookup: same project + target language
    (source language narrows further when known). SQLite can't rank by
    edit-distance or hamming distance, so matching itself happens in
    python (memory.lookup()) over this pool -- fine at MVP scale (a
    project's TM), not meant to scale to a global cross-project index."""
    q = "SELECT * FROM tm_records WHERE project_id = ? AND target_lang = ?"
    vals: List[Any] = [pid, target_lang]
    if source_lang:
        q += " AND (source_lang IS NULL OR source_lang = ?)"
        vals.append(source_lang)
    q += " ORDER BY created_at DESC LIMIT ?"
    vals.append(limit)
    with _conn() as con:
        rows = con.execute(q, vals).fetchall()
    return [dict(r) for r in rows]


def list_tm_records(pid: str, limit: int = 200) -> List[Dict[str, Any]]:
    with _conn() as con:
        rows = con.execute(
            "SELECT * FROM tm_records WHERE project_id = ?"
            " ORDER BY created_at DESC LIMIT ?", (pid, limit),
        ).fetchall()
    return [dict(r) for r in rows]


def delete_tm_record(rid: int) -> bool:
    with _conn() as con:
        cur = con.execute("DELETE FROM tm_records WHERE id = ?", (rid,))
    return cur.rowcount > 0


def tm_count(pid: str) -> int:
    with _conn() as con:
        row = con.execute(
            "SELECT COUNT(*) AS n FROM tm_records WHERE project_id = ?", (pid,)
        ).fetchone()
    return row["n"] if row else 0
