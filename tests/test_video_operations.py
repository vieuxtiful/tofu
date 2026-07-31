import json
import time

import pytest

from server import db


def _database(monkeypatch, tmp_path):
    monkeypatch.setattr(db, "DB_PATH", tmp_path / "operations.db")
    db.init_db()
    job = db.create_video_job("asset", None, {"frame_count": 10}, chunk_size=4)
    return job


def test_operation_claim_is_exclusive_and_heartbeat_extends_lease(monkeypatch, tmp_path):
    job = _database(monkeypatch, tmp_path)
    operation = db.create_video_operation(job["id"], "analysis", 0, 9, spec={"asset_id": "asset"})
    claimed = db.claim_video_operation("worker-a", lease_seconds=2)
    assert claimed["id"] == operation
    assert claimed["attempts"] == 1
    assert db.claim_video_operation("worker-b") is None
    previous = claimed["lease_expires"]
    assert db.heartbeat_video_operation(operation, "worker-a", lease_seconds=5)
    with db._conn() as con:
        current = con.execute("SELECT lease_expires FROM video_operations WHERE id=?", (operation,)).fetchone()[0]
    assert current > previous
    assert db.finish_video_operation(operation, "worker-a", "completed")


def test_export_dedupe_and_preview_supersession(monkeypatch, tmp_path):
    job = _database(monkeypatch, tmp_path)
    first = db.create_video_operation(job["id"], "export", 0, 9, dedupe_key="rev1")
    assert db.create_video_operation(job["id"], "export", 0, 9, dedupe_key="rev1") == first
    old = db.create_video_operation(job["id"], "preview", 0, 2)
    new = db.create_video_operation(job["id"], "preview", 3, 5)
    with db._conn() as con:
        assert con.execute("SELECT status FROM video_operations WHERE id=?", (old,)).fetchone()[0] == "superseded"
        assert con.execute("SELECT status FROM video_operations WHERE id=?", (new,)).fetchone()[0] == "queued"


def test_expired_lease_is_reclaimable(monkeypatch, tmp_path):
    job = _database(monkeypatch, tmp_path)
    operation = db.create_video_operation(job["id"], "analysis", 0, 9)
    db.claim_video_operation("dead-worker", lease_seconds=1)
    with db._conn() as con:
        con.execute("UPDATE video_operations SET lease_expires=? WHERE id=?", (time.time() - 1, operation))
    reclaimed = db.claim_video_operation("replacement")
    assert reclaimed["id"] == operation
    assert reclaimed["attempts"] == 2


def test_job_remembers_the_chunk_size_its_rows_were_cut_on(monkeypatch, tmp_path):
    """chunk_size used to live only in the operation spec, so /resume guessed 240
    while the chunk rows had been cut on a different stride -- a resume then
    addressed chunk indices that correspond to no row."""
    job = _database(monkeypatch, tmp_path)
    assert job["chunk_size"] == 4
    indices = [c["chunk_index"] for c in job["chunks"]]
    assert indices == [0, 1, 2]  # 10 frames on a stride of 4
    assert db.get_video_job(job["id"])["chunk_size"] == 4


def test_second_worker_cannot_claim_a_sibling_operation_for_the_same_job(monkeypatch, tmp_path):
    """The lease is exclusive per operation, which alone does not stop two
    workers taking two different analysis operations for one job and racing
    over the same rows."""
    job = _database(monkeypatch, tmp_path)
    first = db.create_video_operation(job["id"], "analysis", 0, 9)
    second = db.create_video_operation(job["id"], "analysis", 0, 9)
    assert second != first
    assert db.claim_video_operation("worker-a", lease_seconds=30)["id"] == first
    assert db.claim_video_operation("worker-b", lease_seconds=30) is None
    # a different KIND is still free to run concurrently
    db.create_video_operation(job["id"], "preview", 0, 4)
    assert db.claim_video_operation("worker-c", lease_seconds=30)["kind"] == "preview"


def test_abandoned_running_operations_are_requeued_at_startup(monkeypatch, tmp_path):
    """A worker killed on its final attempt leaves a row 'running' that
    claim_video_operation can never steal, because attempts==max_attempts."""
    job = _database(monkeypatch, tmp_path)
    operation = db.create_video_operation(job["id"], "analysis", 0, 9, max_attempts=1)
    db.claim_video_operation("doomed-worker", lease_seconds=1)
    with db._conn() as con:
        con.execute("UPDATE video_operations SET lease_expires=? WHERE id=?", (time.time() - 1, operation))
    assert db.claim_video_operation("replacement") is None
    assert db.requeue_abandoned_video_operations() == 1
    with db._conn() as con:
        row = con.execute("SELECT status,lease_owner FROM video_operations WHERE id=?", (operation,)).fetchone()
    assert row["status"] == "queued" and row["lease_owner"] is None


# --- checkpoint durability -------------------------------------------------

def _observation(frame_index, track_id="trk"):
    return {"id": f"obs{frame_index}", "track_id": track_id, "frame_index": frame_index,
            "pts_seconds": frame_index / 30, "bbox": {"x": 1, "y": 2, "width": 3, "height": 4}}


def _track_row(track_id="trk", start=0, end=3):
    return {"id": track_id, "shot_id": "shot-00000", "start_frame": start, "end_frame": end}


def _state(next_frame, chunk_index=0, fingerprint="fp"):
    return {"next_frame": next_frame, "chunk_index": chunk_index, "shot_index": 0,
            "source_fingerprint": fingerprint, "version": 1, "active": {}}


def _commit(job_id, chunk_index, frames, *, revision=1, fingerprint="fp",
            analyzer_revision="braise-v1"):
    observations = [_observation(f) for f in frames]
    db.commit_video_analysis_chunk(
        job_id, chunk_index, [_track_row(start=frames[0], end=frames[-1])], observations,
        "digest", revision, checkpoint=_state(frames[-1] + 1, chunk_index, fingerprint),
        analyzer_revision=analyzer_revision)


def test_resume_keeps_completed_chunks_and_their_observations(monkeypatch, tmp_path):
    """The bug this whole phase exists to kill: /resume used to wipe every chunk
    back to pending, so a resume was a restart from frame zero."""
    job = _database(monkeypatch, tmp_path)          # 10 frames, chunk_size 4
    _commit(job["id"], 0, [0, 1, 2, 3])
    _commit(job["id"], 1, [4, 5, 6, 7])

    db.reset_video_analysis(job["id"], from_frame=8)

    chunks = {c["chunk_index"]: c["status"] for c in db.get_video_job(job["id"])["chunks"]}
    assert chunks[0] == "completed" and chunks[1] == "completed"
    assert chunks[2] == "pending"
    with db._conn() as con:
        kept = con.execute("SELECT COUNT(*) FROM video_observations WHERE job_id=?",
                           (job["id"],)).fetchone()[0]
    assert kept == 8, "a scoped reset deleted work that was already durable"


def test_reset_from_zero_still_clears_everything(monkeypatch, tmp_path):
    job = _database(monkeypatch, tmp_path)
    _commit(job["id"], 0, [0, 1, 2, 3])
    db.reset_video_analysis(job["id"], from_frame=0)
    assert all(c["status"] == "pending" for c in db.get_video_job(job["id"])["chunks"])
    with db._conn() as con:
        assert con.execute("SELECT COUNT(*) FROM video_observations WHERE job_id=?",
                           (job["id"],)).fetchone()[0] == 0


def test_latest_checkpoint_is_the_newest_completed_chunk(monkeypatch, tmp_path):
    job = _database(monkeypatch, tmp_path)
    _commit(job["id"], 0, [0, 1, 2, 3])
    _commit(job["id"], 1, [4, 5, 6, 7])
    state = db.latest_video_checkpoint(job["id"], analyzer_revision="braise-v1",
                                       dependency_revision=1, source_fingerprint="fp")
    assert state["next_frame"] == 8 and state["chunk_index"] == 1


def test_a_checkpoint_from_other_analyzer_code_is_refused(monkeypatch, tmp_path):
    """Resuming across a behaviour change would splice two different trackers'
    output into one timeline."""
    job = _database(monkeypatch, tmp_path)
    _commit(job["id"], 0, [0, 1, 2, 3], analyzer_revision="braise-v0")
    assert db.latest_video_checkpoint(job["id"], analyzer_revision="braise-v1",
                                      dependency_revision=1, source_fingerprint="fp") is None


def test_a_checkpoint_for_a_different_source_file_is_refused(monkeypatch, tmp_path):
    job = _database(monkeypatch, tmp_path)
    _commit(job["id"], 0, [0, 1, 2, 3], fingerprint="fingerprint-of-some-other-file")
    assert db.latest_video_checkpoint(job["id"], analyzer_revision="braise-v1",
                                      dependency_revision=1, source_fingerprint="fp") is None


def test_a_checkpoint_from_a_superseded_revision_is_refused(monkeypatch, tmp_path):
    job = _database(monkeypatch, tmp_path)
    _commit(job["id"], 0, [0, 1, 2, 3], revision=1)
    assert db.latest_video_checkpoint(job["id"], analyzer_revision="braise-v1",
                                      dependency_revision=2, source_fingerprint="fp") is None


def test_checkpoint_and_observations_are_written_in_one_transaction(monkeypatch, tmp_path):
    """A checkpoint that outlived its observations would resume past work that
    was never persisted -- silently dropping a chunk from the output."""
    job = _database(monkeypatch, tmp_path)
    unserializable = {"next_frame": 4, "version": 1, "active": {1, 2, 3}}
    with pytest.raises(TypeError):
        db.commit_video_analysis_chunk(job["id"], 0, [_track_row()],
                                       [_observation(f) for f in (0, 1, 2, 3)],
                                       "digest", 1, checkpoint=unserializable,
                                       analyzer_revision="braise-v1")
    with db._conn() as con:
        assert con.execute("SELECT COUNT(*) FROM video_observations WHERE job_id=?",
                           (job["id"],)).fetchone()[0] == 0
        assert con.execute("SELECT status FROM video_chunks WHERE job_id=? AND chunk_index=0",
                           (job["id"],)).fetchone()[0] == "pending"


def test_analysis_progress_reports_what_is_durable(monkeypatch, tmp_path):
    job = _database(monkeypatch, tmp_path)
    _commit(job["id"], 0, [0, 1, 2, 3])
    progress = db.video_analysis_progress(job["id"])
    assert progress == {"chunks": 3, "completed_chunks": 1, "last_completed_frame": 3}


def test_artifact_verification_is_persisted(monkeypatch, tmp_path):
    """verify_output's report used to be computed and dropped on the floor."""
    job = _database(monkeypatch, tmp_path)
    report = {"duration": 4.0, "duration_drift": 0.001, "video_codec": "h264"}
    db.save_video_artifact(job["id"], "export", str(tmp_path / "out.mp4"), report)
    with db._conn() as con:
        stored = con.execute("SELECT verification_json FROM video_artifacts WHERE job_id=? AND status='ready'",
                             (job["id"],)).fetchone()[0]
    assert json.loads(stored) == report
