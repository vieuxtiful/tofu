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

## Beside the uploads it describes, wherever those live.
##
## This used to be Path(__file__).parent / "tofu.db" -- next to the source,
## not next to the data. In a container that is inside the image layer while
## uploads/, outputs/ and tm_thumbs/ follow TOFU_DATA_DIR onto the volume, so
## a rebuild silently discarded every project, snapshot and translation memory
## while leaving all the files those rows pointed at sitting on disk. The
## database and the data it indexes have to move together or neither is a
## backup of anything.
from config import settings  # noqa: E402  (settings reads the environment once)

DB_PATH = settings.data_dir / "tofu.db"
DB_PATH.parent.mkdir(parents=True, exist_ok=True)

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
  archived_at REAL,
  created_at  REAL NOT NULL,
  updated_at  REAL NOT NULL,
  ground_truth TEXT NOT NULL DEFAULT '[]'
);
CREATE TABLE IF NOT EXISTS project_assets (
  asset_id     TEXT PRIMARY KEY,
  project_id   TEXT NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
  filename     TEXT,
  uploaded_at  REAL NOT NULL,
  is_active    INTEGER NOT NULL DEFAULT 0,
  content_hash TEXT,
  ground_truth TEXT NOT NULL DEFAULT '[]'
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
-- What a reviewer actually did with each detected candidate.
--
-- The measurement gap this closes: removing the scene-membership veto
-- recovered four ground-truth regions and grew the candidate set from 99 to
-- 134 across the fixture corpus. Whether those 35 extra candidates are worth
-- the review effort is a question about REVIEWER TIME, and no metric the
-- pipeline computes -- mean IoU, garbage fraction, edit distance -- can
-- answer it. A ranker built without this table would be as unmeasurable as
-- the veto it replaces.
--
-- `features` and `eligibility` are SNAPSHOTS taken at emit time, not foreign
-- keys into a manifest. Deliberate: a manifest is edited continuously, so a
-- join resolved later would describe the candidate as it ended up rather
-- than as it was when the reviewer judged it -- which inverts cause and
-- effect for the exact question being asked.
CREATE TABLE IF NOT EXISTS review_events (
  id           TEXT PRIMARY KEY,
  project_id   TEXT REFERENCES projects(id) ON DELETE CASCADE,
  asset_id     TEXT NOT NULL,
  session_id   TEXT,
  candidate_id TEXT NOT NULL,
  action_type  TEXT NOT NULL,     -- accepted|rejected|edited_geometry|edited_text|merged
  dwell_ms     INTEGER,
  modified_bbox TEXT,             -- JSON [x,y,w,h], only when geometry changed
  features     TEXT,              -- JSON snapshot of InstText.review_features
  eligibility  TEXT,              -- JSON snapshot of InstText.scene_eligibility
  created_at   REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_review_asset ON review_events(asset_id, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_review_action ON review_events(action_type);
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
CREATE TABLE IF NOT EXISTS video_jobs (
  id TEXT PRIMARY KEY, project_id TEXT, asset_id TEXT NOT NULL,
  status TEXT NOT NULL, stage TEXT NOT NULL, progress REAL NOT NULL DEFAULT 0,
  cancel_requested INTEGER NOT NULL DEFAULT 0, error TEXT,
  manifest_json TEXT, dependency_revision INTEGER NOT NULL DEFAULT 1,
  pipeline_version TEXT NOT NULL DEFAULT 'production-v2', upgrade_required INTEGER NOT NULL DEFAULT 0,
  chunk_size INTEGER NOT NULL DEFAULT 240,
  created_at REAL NOT NULL, updated_at REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_video_jobs_asset ON video_jobs(asset_id, updated_at DESC);
CREATE TABLE IF NOT EXISTS video_chunks (
  job_id TEXT NOT NULL REFERENCES video_jobs(id) ON DELETE CASCADE,
  chunk_index INTEGER NOT NULL, start_frame INTEGER NOT NULL, end_frame INTEGER NOT NULL,
  status TEXT NOT NULL DEFAULT 'pending', stage TEXT, error TEXT, checksum TEXT,
  dependency_revision INTEGER NOT NULL DEFAULT 1, attempts INTEGER NOT NULL DEFAULT 0,
  completed_at REAL,
  checkpoint_json TEXT, checkpoint_version INTEGER NOT NULL DEFAULT 0, analyzer_revision TEXT,
  PRIMARY KEY(job_id, chunk_index)
);
CREATE INDEX IF NOT EXISTS idx_video_chunks_resume ON video_chunks(job_id, status, chunk_index DESC);
CREATE TABLE IF NOT EXISTS video_tracks (
  job_id TEXT NOT NULL REFERENCES video_jobs(id) ON DELETE CASCADE,
  track_id TEXT NOT NULL, shot_id TEXT NOT NULL, start_frame INTEGER NOT NULL,
  end_frame INTEGER NOT NULL, data_json TEXT NOT NULL, PRIMARY KEY(job_id, track_id)
);
CREATE TABLE IF NOT EXISTS video_observations (
  job_id TEXT NOT NULL REFERENCES video_jobs(id) ON DELETE CASCADE,
  observation_id TEXT NOT NULL, track_id TEXT NOT NULL, frame_index INTEGER NOT NULL,
  pts_seconds REAL NOT NULL, data_json TEXT NOT NULL, PRIMARY KEY(job_id, observation_id)
);
CREATE INDEX IF NOT EXISTS idx_video_obs_range ON video_observations(job_id, frame_index);
CREATE TABLE IF NOT EXISTS video_keyframes (
  job_id TEXT NOT NULL REFERENCES video_jobs(id) ON DELETE CASCADE,
  keyframe_id TEXT NOT NULL, track_id TEXT NOT NULL, frame_index INTEGER NOT NULL,
  data_json TEXT NOT NULL, PRIMARY KEY(job_id, keyframe_id)
);
CREATE TABLE IF NOT EXISTS video_issues (
  id INTEGER PRIMARY KEY AUTOINCREMENT, job_id TEXT NOT NULL,
  track_id TEXT, frame_index INTEGER, code TEXT NOT NULL, severity TEXT NOT NULL,
  detail TEXT, resolved INTEGER NOT NULL DEFAULT 0
);
CREATE TABLE IF NOT EXISTS video_artifacts (
  id TEXT PRIMARY KEY, job_id TEXT NOT NULL, kind TEXT NOT NULL,
  path TEXT NOT NULL, status TEXT NOT NULL DEFAULT 'ready', created_at REAL NOT NULL,
  verification_json TEXT
);
CREATE TABLE IF NOT EXISTS video_operations (
  id TEXT PRIMARY KEY, job_id TEXT NOT NULL, kind TEXT NOT NULL,
  start_frame INTEGER NOT NULL, end_frame INTEGER NOT NULL,
  status TEXT NOT NULL, error TEXT, spec_json TEXT NOT NULL DEFAULT '{}',
  lease_owner TEXT, lease_expires REAL, heartbeat_at REAL,
  attempts INTEGER NOT NULL DEFAULT 0, max_attempts INTEGER NOT NULL DEFAULT 3,
  cancellation_generation INTEGER NOT NULL DEFAULT 0,
  dependency_revision INTEGER NOT NULL DEFAULT 1, dedupe_key TEXT,
  created_at REAL NOT NULL, updated_at REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_video_operations_recovery ON video_operations(status, updated_at);
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
        if "archived_at" not in cols:
            con.execute("ALTER TABLE projects ADD COLUMN archived_at REAL")
        if "ground_truth" not in cols:
            con.execute("ALTER TABLE projects ADD COLUMN ground_truth TEXT NOT NULL DEFAULT '[]'")
        # migration: assets uploaded before duplicate-image detection existed
        asset_cols = {r["name"] for r in con.execute("PRAGMA table_info(project_assets)")}
        if "content_hash" not in asset_cols:
            con.execute("ALTER TABLE project_assets ADD COLUMN content_hash TEXT")
        if "ground_truth" not in asset_cols:
            con.execute("ALTER TABLE project_assets ADD COLUMN ground_truth TEXT NOT NULL DEFAULT '[]'")
        con.execute("CREATE INDEX IF NOT EXISTS idx_assets_hash ON project_assets(content_hash)")
        operation_cols = {r["name"] for r in con.execute("PRAGMA table_info(video_operations)")}
        operation_additions = {
            "spec_json": "TEXT NOT NULL DEFAULT '{}'", "lease_owner": "TEXT",
            "lease_expires": "REAL", "heartbeat_at": "REAL",
            "attempts": "INTEGER NOT NULL DEFAULT 0", "max_attempts": "INTEGER NOT NULL DEFAULT 3",
            "cancellation_generation": "INTEGER NOT NULL DEFAULT 0",
            "dependency_revision": "INTEGER NOT NULL DEFAULT 1", "dedupe_key": "TEXT",
        }
        for name, declaration in operation_additions.items():
            if name not in operation_cols:
                con.execute(f"ALTER TABLE video_operations ADD COLUMN {name} {declaration}")
        con.execute("CREATE INDEX IF NOT EXISTS idx_video_operations_dedupe ON video_operations(job_id,kind,dedupe_key)")
        job_cols = {r["name"] for r in con.execute("PRAGMA table_info(video_jobs)")}
        if "dependency_revision" not in job_cols:
            con.execute("ALTER TABLE video_jobs ADD COLUMN dependency_revision INTEGER NOT NULL DEFAULT 1")
        if "pipeline_version" not in job_cols:
            con.execute("ALTER TABLE video_jobs ADD COLUMN pipeline_version TEXT NOT NULL DEFAULT 'prototype-v1'")
        if "upgrade_required" not in job_cols:
            con.execute("ALTER TABLE video_jobs ADD COLUMN upgrade_required INTEGER NOT NULL DEFAULT 1")
        # migration: chunk_size used to live only in the operation spec, so /resume and
        # /upgrade guessed 240 while the chunk rows had been cut on a different stride.
        if "chunk_size" not in job_cols:
            con.execute("ALTER TABLE video_jobs ADD COLUMN chunk_size INTEGER NOT NULL DEFAULT 240")
        artifact_cols = {r["name"] for r in con.execute("PRAGMA table_info(video_artifacts)")}
        if "verification_json" not in artifact_cols:
            con.execute("ALTER TABLE video_artifacts ADD COLUMN verification_json TEXT")
        con.execute("UPDATE video_artifacts SET status='stale' WHERE job_id IN (SELECT id FROM video_jobs WHERE pipeline_version='prototype-v1') AND kind IN ('preview','export')")
        con.execute("DELETE FROM video_observations WHERE job_id IN (SELECT id FROM video_jobs WHERE pipeline_version='prototype-v1')")
        con.execute("DELETE FROM video_keyframes WHERE NOT EXISTS (SELECT 1 FROM video_tracks t WHERE t.job_id=video_keyframes.job_id AND t.track_id=video_keyframes.track_id)")
        chunk_cols = {r["name"] for r in con.execute("PRAGMA table_info(video_chunks)")}
        for name, declaration in {"checksum": "TEXT", "dependency_revision": "INTEGER NOT NULL DEFAULT 1",
                                  "attempts": "INTEGER NOT NULL DEFAULT 0", "completed_at": "REAL",
                                  "checkpoint_json": "TEXT",
                                  "checkpoint_version": "INTEGER NOT NULL DEFAULT 0",
                                  "analyzer_revision": "TEXT"}.items():
            if name not in chunk_cols: con.execute(f"ALTER TABLE video_chunks ADD COLUMN {name} {declaration}")


def create_video_job(asset_id: str, project_id: Optional[str], manifest: Dict[str, Any],
                     chunk_size: int = 240) -> Dict[str, Any]:
    job_id, now = uuid.uuid4().hex[:16], time.time()
    frame_count = int(manifest.get("frame_count", 0))
    with _conn() as con:
        # chunk_size is persisted, not re-guessed: the video_chunks rows below are cut on
        # this stride, and every later resume must address the very same rows.
        con.execute("INSERT INTO video_jobs (id,project_id,asset_id,status,stage,manifest_json,pipeline_version,upgrade_required,chunk_size,created_at,updated_at) VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                    (job_id, project_id, asset_id, "queued", "ingest", json.dumps(manifest), "production-v2", 0, chunk_size, now, now))
        for index, start in enumerate(range(0, frame_count, chunk_size)):
            con.execute("INSERT INTO video_chunks (job_id,chunk_index,start_frame,end_frame,status) VALUES (?,?,?,?,?)",
                        (job_id, index, start, min(frame_count - 1, start + chunk_size - 1), "pending"))
    return get_video_job(job_id)  # type: ignore[return-value]


def get_video_job(job_id: str) -> Optional[Dict[str, Any]]:
    with _conn() as con:
        row = con.execute("SELECT * FROM video_jobs WHERE id=?", (job_id,)).fetchone()
        if not row:
            return None
        chunks = con.execute("SELECT * FROM video_chunks WHERE job_id=? ORDER BY chunk_index", (job_id,)).fetchall()
    out = dict(row)
    out["manifest"] = json.loads(out.pop("manifest_json") or "{}")
    out["chunks"] = [dict(c) for c in chunks]
    return out


def latest_video_job(asset_id: str) -> Optional[Dict[str, Any]]:
    with _conn() as con:
        row = con.execute("SELECT id FROM video_jobs WHERE asset_id=? ORDER BY updated_at DESC LIMIT 1", (asset_id,)).fetchone()
    return get_video_job(row["id"]) if row else None


def video_artifact_url(job_id: str, kind: str) -> Optional[str]:
    with _conn() as con:
        row = con.execute("SELECT path FROM video_artifacts WHERE job_id=? AND kind=? AND status='ready' ORDER BY created_at DESC LIMIT 1",
                          (job_id, kind)).fetchone()
    if not row:
        return None
    try:
        relative = Path(row["path"]).resolve().relative_to((Path(__file__).resolve().parents[1] / "server" / "outputs").resolve())
    except ValueError:
        return None
    return "/outputs/" + relative.as_posix()


def update_video_job(job_id: str, **fields: Any) -> Optional[Dict[str, Any]]:
    allowed = {"status", "stage", "progress", "error", "cancel_requested", "manifest_json"}
    values = {k: (json.dumps(v) if k == "manifest_json" and not isinstance(v, str) else v)
              for k, v in fields.items() if k in allowed}
    if values:
        values["updated_at"] = time.time()
        with _conn() as con:
            con.execute("UPDATE video_jobs SET " + ",".join(f"{k}=?" for k in values) + " WHERE id=?",
                        (*values.values(), job_id))
    return get_video_job(job_id)


def list_video_timeline(job_id: str, start: int, end: int, limit: int = 1000, cursor: int = 0) -> Dict[str, Any]:
    with _conn() as con:
        tracks = con.execute("SELECT data_json FROM video_tracks WHERE job_id=? ORDER BY start_frame", (job_id,)).fetchall()
        obs = con.execute("SELECT data_json FROM video_observations WHERE job_id=? AND frame_index BETWEEN ? AND ? ORDER BY frame_index,observation_id LIMIT ? OFFSET ?",
                          (job_id, start, end, limit + 1, cursor)).fetchall()
        keys = con.execute("SELECT data_json FROM video_keyframes WHERE job_id=? AND frame_index BETWEEN ? AND ? ORDER BY frame_index",
                           (job_id, start, end)).fetchall()
        issues = con.execute("SELECT * FROM video_issues WHERE job_id=? AND (frame_index IS NULL OR frame_index BETWEEN ? AND ?) ORDER BY frame_index",
                             (job_id, start, end)).fetchall()
    truncated = len(obs) > limit; page = obs[:limit]
    return {"tracks": [json.loads(r[0]) for r in tracks], "observations": [json.loads(r[0]) for r in page],
            "keyframes": [json.loads(r[0]) for r in keys], "issues": [dict(r) for r in issues],
            "truncated": truncated, "next_cursor": cursor + limit if truncated else None}


def upsert_video_keyframe(job_id: str, data: Dict[str, Any]) -> None:
    with _conn() as con:
        con.execute("INSERT OR REPLACE INTO video_keyframes (job_id,keyframe_id,track_id,frame_index,data_json) VALUES (?,?,?,?,?)",
                    (job_id, data["id"], data["track_id"], data["frame_index"], json.dumps(data)))
        con.execute("UPDATE video_artifacts SET status='stale' WHERE job_id=? AND kind IN ('preview','export')", (job_id,))
        con.execute("UPDATE video_jobs SET dependency_revision=dependency_revision+1,updated_at=? WHERE id=?", (time.time(), job_id))


def reset_video_analysis(job_id: str, *, from_frame: int = 0) -> None:
    """Clear analysis output at or after `from_frame`.

    Scoped, because /resume calls this: wiping every chunk back to pending is
    exactly what made a resume start over from frame zero. Tracks that straddle
    the boundary are re-emitted by the analyzer -- they are in the checkpoint's
    active set -- so their end_frame self-heals through INSERT OR REPLACE.
    Tracks born after the boundary and absent from the checkpoint are orphans,
    and deleting them is the point.
    """
    with _conn() as con:
        con.execute("DELETE FROM video_observations WHERE job_id=? AND frame_index>=?", (job_id, from_frame))
        con.execute("DELETE FROM video_tracks WHERE job_id=? AND start_frame>=?", (job_id, from_frame))
        con.execute("DELETE FROM video_issues WHERE job_id=? AND (frame_index IS NULL OR frame_index>=?)",
                    (job_id, from_frame))
        con.execute("UPDATE video_chunks SET status='pending',stage=NULL,error=NULL,checksum=NULL,"
                    "completed_at=NULL,checkpoint_json=NULL,checkpoint_version=0,analyzer_revision=NULL "
                    "WHERE job_id=? AND start_frame>=?", (job_id, from_frame))


def commit_video_analysis_chunk(job_id: str, chunk_index: int, tracks: List[Dict[str, Any]],
                                observations: List[Dict[str, Any]], checksum: str,
                                dependency_revision: int, *,
                                checkpoint: Optional[Dict[str, Any]] = None,
                                analyzer_revision: Optional[str] = None) -> None:
    """Write a chunk's rows and the tracker state that describes them, atomically.

    One transaction is load-bearing: a checkpoint that survived without its
    observations would resume past work that was never persisted, and
    observations without a checkpoint would be re-analyzed and duplicated.
    """
    now = time.time()
    with _conn() as con:
        con.executemany("INSERT OR REPLACE INTO video_tracks (job_id,track_id,shot_id,start_frame,end_frame,data_json) VALUES (?,?,?,?,?,?)",
                        [(job_id, t["id"], t["shot_id"], t["start_frame"], t["end_frame"], json.dumps(t)) for t in tracks])
        con.executemany("INSERT OR REPLACE INTO video_observations (job_id,observation_id,track_id,frame_index,pts_seconds,data_json) VALUES (?,?,?,?,?,?)",
                        [(job_id, o["id"], o["track_id"], o["frame_index"], o["pts_seconds"], json.dumps(o)) for o in observations])
        con.execute("UPDATE video_chunks SET status='completed',stage='analysis',checksum=?,dependency_revision=?,"
                    "attempts=attempts+1,completed_at=?,checkpoint_json=?,checkpoint_version=?,analyzer_revision=? "
                    "WHERE job_id=? AND chunk_index=?",
                    (checksum, dependency_revision, now,
                     json.dumps(checkpoint) if checkpoint is not None else None,
                     int((checkpoint or {}).get("version", 0)), analyzer_revision,
                     job_id, chunk_index))


def latest_video_checkpoint(job_id: str, *, analyzer_revision: str, dependency_revision: int,
                            source_fingerprint: str) -> Optional[Dict[str, Any]]:
    """Newest usable checkpoint, or None to start over.

    Every rejection here is a case where continuing would be worse than
    redoing the work: a checkpoint written by different analyzer code, one
    belonging to a superseded revision of the job, or one describing a
    different source file.
    """
    with _conn() as con:
        rows = con.execute(
            "SELECT chunk_index,checkpoint_json,analyzer_revision,dependency_revision "
            "FROM video_chunks WHERE job_id=? AND status='completed' AND checkpoint_json IS NOT NULL "
            "ORDER BY chunk_index DESC", (job_id,)).fetchall()
    for row in rows:
        if row["analyzer_revision"] != analyzer_revision:
            continue
        if int(row["dependency_revision"]) != int(dependency_revision):
            continue
        try:
            payload = json.loads(row["checkpoint_json"])
        except (TypeError, ValueError):
            continue
        if payload.get("source_fingerprint") and payload["source_fingerprint"] != source_fingerprint:
            continue
        return payload
    return None


def record_video_issues(job_id: str, issues: List[Dict[str, Any]]) -> None:
    """Append issues without disturbing what is already there.

    finalize_video_analysis replaces the table wholesale; this does not. Resume
    provenance is written while analysis is still running, so it has to survive
    the finalization that follows it.
    """
    with _conn() as con:
        con.executemany("INSERT INTO video_issues (job_id,track_id,frame_index,code,severity,detail) VALUES (?,?,?,?,?,?)",
                        [(job_id, i.get("track_id"), i.get("frame_index"), i["code"],
                          i.get("severity", "review"), i.get("detail")) for i in issues])


def list_video_tracks(job_id: str) -> List[Dict[str, Any]]:
    with _conn() as con:
        rows = con.execute("SELECT data_json FROM video_tracks WHERE job_id=? ORDER BY start_frame",
                           (job_id,)).fetchall()
    return [json.loads(row["data_json"]) for row in rows]


def video_analysis_progress(job_id: str) -> Dict[str, Any]:
    """How much of the analysis is already durable."""
    with _conn() as con:
        row = con.execute(
            "SELECT COUNT(*) AS total, SUM(status='completed') AS done, "
            "MAX(CASE WHEN status='completed' THEN end_frame END) AS last_frame "
            "FROM video_chunks WHERE job_id=?", (job_id,)).fetchone()
    return {"chunks": int(row["total"] or 0), "completed_chunks": int(row["done"] or 0),
            "last_completed_frame": row["last_frame"]}


def finalize_video_analysis(job_id: str, tracks: List[Dict[str, Any]], issues: List[Dict[str, Any]]) -> None:
    with _conn() as con:
        con.executemany("INSERT OR REPLACE INTO video_tracks (job_id,track_id,shot_id,start_frame,end_frame,data_json) VALUES (?,?,?,?,?,?)",
                        [(job_id, t["id"], t["shot_id"], t["start_frame"], t["end_frame"], json.dumps(t)) for t in tracks])
        con.execute("DELETE FROM video_issues WHERE job_id=?", (job_id,))
        con.executemany("INSERT INTO video_issues (job_id,track_id,frame_index,code,severity,detail) VALUES (?,?,?,?,?,?)",
                        [(job_id, i.get("track_id"), i.get("frame_index"), i["code"], i.get("severity", "review"), i.get("detail")) for i in issues])


def get_video_track(job_id: str, track_id: str) -> Optional[Dict[str, Any]]:
    with _conn() as con:
        row = con.execute("SELECT data_json FROM video_tracks WHERE job_id=? AND track_id=?", (job_id, track_id)).fetchone()
    return json.loads(row["data_json"]) if row else None


def update_video_track(job_id: str, track_id: str, changes: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    track = get_video_track(job_id, track_id)
    if not track:
        return None
    track.update(changes)
    track["revision"] = int(track.get("revision", 1)) + 1
    with _conn() as con:
        con.execute("UPDATE video_tracks SET data_json=? WHERE job_id=? AND track_id=?",
                    (json.dumps(track), job_id, track_id))
        con.execute("UPDATE video_artifacts SET status='stale' WHERE job_id=? AND kind IN ('preview','export')", (job_id,))
        con.execute("UPDATE video_jobs SET dependency_revision=dependency_revision+1,updated_at=? WHERE id=?", (time.time(), job_id))
    return track


def video_frame_count(job_id: str) -> Optional[int]:
    job = get_video_job(job_id)
    return int(job["manifest"].get("frame_count", 0)) if job else None


def mark_video_upgrade(job_id: str, *, completed: bool) -> None:
    with _conn() as con:
        con.execute("UPDATE video_jobs SET pipeline_version='production-v2',upgrade_required=?,updated_at=? WHERE id=?",
                    (0 if completed else 1, time.time(), job_id))


def video_composition_data(job_id: str, start: int, end: int) -> Dict[str, Any]:
    with _conn() as con:
        tracks = con.execute("SELECT data_json FROM video_tracks WHERE job_id=?", (job_id,)).fetchall()
        observations = con.execute("SELECT data_json FROM video_observations WHERE job_id=? AND frame_index BETWEEN ? AND ? ORDER BY frame_index",
                                   (job_id, start, end)).fetchall()
        keyframes = con.execute("SELECT data_json FROM video_keyframes WHERE job_id=? ORDER BY frame_index", (job_id,)).fetchall()
    return {"tracks": [json.loads(row[0]) for row in tracks],
            "observations": [json.loads(row[0]) for row in observations],
            "keyframes": [json.loads(row[0]) for row in keyframes]}


def save_video_artifact(job_id: str, kind: str, path: str,
                        verification: Optional[Dict[str, Any]] = None) -> str:
    artifact_id = uuid.uuid4().hex[:16]
    with _conn() as con:
        con.execute("UPDATE video_artifacts SET status='stale' WHERE job_id=? AND kind=?", (job_id, kind))
        con.execute("INSERT INTO video_artifacts (id,job_id,kind,path,status,created_at,verification_json) VALUES (?,?,?,?,?,?,?)",
                    (artifact_id, job_id, kind, path, "ready", time.time(),
                     json.dumps(verification) if verification is not None else None))
    return artifact_id


def prune_video_artifact_records(job_id: str, kind: str, keep: int) -> List[str]:
    with _conn() as con:
        rows = con.execute("SELECT id,path FROM video_artifacts WHERE job_id=? AND kind=? ORDER BY created_at DESC", (job_id, kind)).fetchall()
        removed = rows[max(0, keep):]
        if removed: con.executemany("DELETE FROM video_artifacts WHERE id=?", [(row["id"],) for row in removed])
    return [str(row["path"]) for row in removed]


def create_video_operation(job_id: str, kind: str, start: int, end: int, *,
                           spec: Optional[Dict[str, Any]] = None, dependency_revision: int = 1,
                           dedupe_key: Optional[str] = None, max_attempts: int = 3) -> str:
    operation_id, now = uuid.uuid4().hex[:16], time.time()
    with _conn() as con:
        if dedupe_key:
            existing = con.execute("SELECT id FROM video_operations WHERE job_id=? AND kind=? AND dedupe_key=? AND status IN ('queued','running','completed') ORDER BY created_at DESC LIMIT 1",
                                   (job_id, kind, dedupe_key)).fetchone()
            if existing: return str(existing["id"])
        if kind == "preview":
            con.execute("UPDATE video_operations SET status='superseded',updated_at=? WHERE job_id=? AND kind='preview' AND status='queued'", (now, job_id))
        con.execute("INSERT INTO video_operations (id,job_id,kind,start_frame,end_frame,status,spec_json,max_attempts,dependency_revision,dedupe_key,created_at,updated_at) VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
                    (operation_id, job_id, kind, start, end, "queued", json.dumps(spec or {}), max_attempts,
                     dependency_revision, dedupe_key, now, now))
    return operation_id


def update_video_operation(operation_id: str, status: str, error: Optional[str] = None) -> None:
    with _conn() as con:
        con.execute("UPDATE video_operations SET status=?,error=?,updated_at=? WHERE id=?",
                    (status, error, time.time(), operation_id))


def claim_video_operation(worker_id: str, lease_seconds: int = 30) -> Optional[Dict[str, Any]]:
    now = time.time()
    with _conn() as con:
        con.execute("BEGIN IMMEDIATE")
        # The lease alone is exclusive per operation, which does not stop two workers from
        # picking up two *different* operations of the same kind for one job and racing over
        # the same rows.  The NOT EXISTS guard makes exclusion a property of the claim itself,
        # so it holds whether or not the caller remembered a dedupe key.
        row = con.execute(
            "SELECT * FROM video_operations o WHERE (o.status='queued' OR (o.status='running' AND o.lease_expires<?))"
            " AND o.attempts<o.max_attempts"
            " AND NOT EXISTS (SELECT 1 FROM video_operations b WHERE b.job_id=o.job_id AND b.kind=o.kind"
            "                 AND b.status='running' AND b.lease_expires>=? AND b.id<>o.id)"
            " ORDER BY o.created_at LIMIT 1",
            (now, now),
        ).fetchone()
        if not row: return None
        expires = now + lease_seconds
        changed = con.execute(
            "UPDATE video_operations SET status='running',lease_owner=?,lease_expires=?,heartbeat_at=?,attempts=attempts+1,updated_at=? WHERE id=? AND (status='queued' OR lease_expires<?)",
            (worker_id, expires, now, now, row["id"], now),
        ).rowcount
        if not changed: return None
        claimed = con.execute("SELECT * FROM video_operations WHERE id=?", (row["id"],)).fetchone()
    out = dict(claimed); out["spec"] = json.loads(out.pop("spec_json") or "{}"); return out


def heartbeat_video_operation(operation_id: str, worker_id: str, lease_seconds: int = 30) -> bool:
    now = time.time()
    with _conn() as con:
        changed = con.execute("UPDATE video_operations SET heartbeat_at=?,lease_expires=?,updated_at=? WHERE id=? AND status='running' AND lease_owner=?",
                              (now, now + lease_seconds, now, operation_id, worker_id)).rowcount
    return bool(changed)


def finish_video_operation(operation_id: str, worker_id: str, status: str, error: Optional[str] = None) -> bool:
    with _conn() as con:
        changed = con.execute("UPDATE video_operations SET status=?,error=?,lease_owner=NULL,lease_expires=NULL,updated_at=? WHERE id=? AND lease_owner=?",
                              (status, error, time.time(), operation_id, worker_id)).rowcount
    return bool(changed)


def cancel_video_operations(job_id: str) -> None:
    with _conn() as con:
        con.execute("UPDATE video_operations SET cancellation_generation=cancellation_generation+1,status=CASE WHEN status='queued' THEN 'cancelled' ELSE status END,updated_at=? WHERE job_id=? AND status IN ('queued','running')",
                    (time.time(), job_id))


def video_operation_cancelled(operation_id: str, generation: int) -> bool:
    with _conn() as con:
        row = con.execute("SELECT status,cancellation_generation FROM video_operations WHERE id=?", (operation_id,)).fetchone()
    return not row or row["status"] == "cancelled" or int(row["cancellation_generation"]) != generation


def requeue_abandoned_video_operations() -> int:
    """Return operations whose worker died mid-lease to the queue.

    A crashed worker leaves rows 'running' with a lease nobody will ever renew.
    claim_video_operation can already steal an expired lease, but only while
    attempts<max_attempts -- so an operation killed on its final attempt would
    otherwise sit 'running' forever.  Startup is the one moment we know no
    lease of ours is live, so expiring them here is safe and unblocks recovery.
    """
    now = time.time()
    with _conn() as con:
        changed = con.execute(
            "UPDATE video_operations SET status='queued',lease_owner=NULL,lease_expires=NULL,updated_at=?"
            " WHERE status='running' AND (lease_expires IS NULL OR lease_expires<?)",
            (now, now),
        ).rowcount
    return int(changed)


# --- projects ---

def _with_ground_truth(record: Dict[str, Any]) -> Dict[str, Any]:
    """Expose persisted JSON as the array used by the API."""
    out = dict(record)
    raw = out.get("ground_truth")
    try:
        parsed = json.loads(raw) if isinstance(raw, str) else raw
    except (TypeError, ValueError):
        parsed = []
    out["ground_truth"] = parsed if isinstance(parsed, list) else []
    return out

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


def list_projects(*, archived: Optional[bool] = None, query: Optional[str] = None,
                  sort: str = "updated") -> List[Dict[str, Any]]:
    """Return project summaries without loading manifests.

    Archive is a reversible lifecycle state, deliberately distinct from delete:
    projects and their snapshot ledger remain recoverable until explicitly
    deleted.  Existing rows migrate as active (``archived_at IS NULL``).
    """
    where: List[str] = []
    vals: List[Any] = []
    if archived is not None:
        where.append("p.archived_at IS NOT NULL" if archived else "p.archived_at IS NULL")
    if query:
        where.append("lower(p.name) LIKE ?")
        vals.append(f"%{query.casefold()}%")
    order = {
        "updated": "p.updated_at DESC",
        "created": "p.created_at DESC",
        "name": "lower(p.name) ASC",
    }.get(sort, "p.updated_at DESC")
    with _conn() as con:
        rows = con.execute(
            """SELECT p.*,
                      (SELECT COUNT(*) FROM project_assets a WHERE a.project_id = p.id) AS asset_count,
                      (SELECT COUNT(*) FROM snapshots s WHERE s.project_id = p.id) AS snapshot_count
               FROM projects p"""
            + (" WHERE " + " AND ".join(where) if where else "")
            + f" ORDER BY {order}", vals
        ).fetchall()
    return [_with_ground_truth(dict(r)) for r in rows]


def get_project(pid: str) -> Optional[Dict[str, Any]]:
    with _conn() as con:
        row = con.execute("SELECT * FROM projects WHERE id = ?", (pid,)).fetchone()
        if row is None:
            return None
        assets = con.execute(
            "SELECT * FROM project_assets WHERE project_id = ?"
            " ORDER BY uploaded_at DESC", (pid,)
        ).fetchall()
    out = _with_ground_truth(dict(row))
    out["assets"] = [_with_ground_truth(dict(a)) for a in assets]
    out["active_asset"] = next(
        (_with_ground_truth(dict(a)) for a in assets if a["is_active"]), None
    )
    return out


def update_project(pid: str, *, name: Optional[str] = None,
                   target_lang: Optional[str] = None,
                   source_lang: Optional[str] = None,
                   ground_truth: Optional[List[str]] = None,
                   archived: Optional[bool] = None) -> Optional[Dict[str, Any]]:
    sets, vals = [], []
    if name is not None:
        sets.append("name = ?"); vals.append(name)
    if target_lang is not None:
        sets.append("target_lang = ?"); vals.append(target_lang)
    if source_lang is not None:
        sets.append("source_lang = ?"); vals.append(source_lang)
    if ground_truth is not None:
        sets.append("ground_truth = ?")
        vals.append(json.dumps(ground_truth, ensure_ascii=False))
    if archived is not None:
        sets.append("archived_at = ?")
        vals.append(time.time() if archived else None)
    if not sets:
        return get_project(pid)
    sets.append("updated_at = ?"); vals.append(time.time())
    vals.append(pid)
    with _conn() as con:
        con.execute(f"UPDATE projects SET {', '.join(sets)} WHERE id = ?", vals)
    return get_project(pid)


def set_project_archived(pid: str, archived: bool) -> Optional[Dict[str, Any]]:
    project = update_project(pid, archived=archived)
    if project is not None:
        log_event(pid, "project-archived" if archived else "project-restored",
                  "project archived" if archived else "project restored")
    return project


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


def update_asset_ground_truth(asset_id: str, ground_truth: List[str]) -> Optional[Dict[str, Any]]:
    now = time.time()
    with _conn() as con:
        row = con.execute(
            "SELECT project_id FROM project_assets WHERE asset_id = ?", (asset_id,)
        ).fetchone()
        if row is None:
            return None
        con.execute(
            "UPDATE project_assets SET ground_truth = ? WHERE asset_id = ?",
            (json.dumps(ground_truth, ensure_ascii=False), asset_id),
        )
        con.execute(
            "UPDATE projects SET updated_at = ? WHERE id = ?", (now, row["project_id"])
        )
        asset = con.execute(
            "SELECT * FROM project_assets WHERE asset_id = ?", (asset_id,)
        ).fetchone()
    return _with_ground_truth(dict(asset)) if asset else None


def asset_by_id(asset_id: str) -> Optional[Dict[str, Any]]:
    with _conn() as con:
        row = con.execute(
            "SELECT * FROM project_assets WHERE asset_id = ?", (asset_id,)
        ).fetchone()
    return _with_ground_truth(dict(row)) if row else None


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


# --- review telemetry (feeds the review-priority ranker) ---

## Actions a reviewer can take on a candidate, and what each one indicts.
## Derived from ordinary editing gestures rather than asked for explicitly:
## a reviewer who has to grade every region is doing the ranker's data entry,
## and would stop.
##
##   accepted          left untouched through to save/export -- pipeline was right
##   rejected          region deleted -- a candidate that cost review time and
##                     produced nothing, which is what a ranker should bury
##   edited_geometry   handles dragged -- localization wrong, recognition may
##                     have been fine (the gemini-street near-miss class)
##   edited_text       transcript overtyped -- localization right, recognizer wrong
##   merged            two regions joined -- over-segmentation upstream
REVIEW_ACTIONS = ("accepted", "rejected", "edited_geometry", "edited_text", "merged")


def record_review_events(events: List[Dict[str, Any]]) -> int:
    """Append reviewer outcomes. Returns the count actually written.

    Unknown action types are dropped rather than stored: this table's whole
    value is that a later join can trust the label, and a typo'd action that
    silently becomes a category would corrupt exactly that.
    """
    rows = []
    now = time.time()
    for e in events:
        action = str(e.get("action_type") or "")
        if action not in REVIEW_ACTIONS or not e.get("candidate_id") or not e.get("asset_id"):
            continue
        rows.append((
            str(uuid.uuid4()), e.get("project_id"), e["asset_id"], e.get("session_id"),
            e["candidate_id"], action,
            int(e["dwell_ms"]) if e.get("dwell_ms") is not None else None,
            json.dumps(e["modified_bbox"]) if e.get("modified_bbox") else None,
            json.dumps(e.get("features")) if e.get("features") else None,
            json.dumps(e.get("eligibility")) if e.get("eligibility") else None,
            float(e.get("created_at") or now),
        ))
    if not rows:
        return 0
    with _conn() as con:
        con.executemany(
            "INSERT OR REPLACE INTO review_events (id, project_id, asset_id, session_id,"
            " candidate_id, action_type, dwell_ms, modified_bbox, features, eligibility,"
            " created_at) VALUES (?,?,?,?,?,?,?,?,?,?,?)",
            rows,
        )
    return len(rows)


def review_summary(asset_id: Optional[str] = None,
                   project_id: Optional[str] = None) -> Dict[str, Any]:
    """The review-effort numbers a ranker has to move.

    `candidates_per_accepted` is the headline: how many regions a human had
    to look at for each one that survived. Mean IoU cannot see it, and it is
    the metric that decides whether the candidates recovered by dropping the
    scene veto are practically manageable.
    """
    where, params = [], []
    if asset_id:
        where.append("asset_id = ?"); params.append(asset_id)
    if project_id:
        where.append("project_id = ?"); params.append(project_id)
    clause = (" WHERE " + " AND ".join(where)) if where else ""
    with _conn() as con:
        rows = con.execute(
            f"SELECT action_type, COUNT(*) n FROM review_events{clause} GROUP BY action_type",
            params,
        ).fetchall()
        reviewed = con.execute(
            f"SELECT COUNT(DISTINCT candidate_id) n FROM review_events{clause}", params,
        ).fetchone()["n"]
    counts = {a: 0 for a in REVIEW_ACTIONS}
    counts.update({r["action_type"]: r["n"] for r in rows})
    accepted = counts["accepted"]
    return {
        "counts": counts,
        "candidates_reviewed": reviewed,
        "accepted": accepted,
        # None rather than a divide-by-zero sentinel: no accepted regions
        # means the ratio is undefined, not infinite.
        "candidates_per_accepted": round(reviewed / accepted, 3) if accepted else None,
    }


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
