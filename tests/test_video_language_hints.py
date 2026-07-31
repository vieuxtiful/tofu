"""Video analysis must inherit the project's source language.

Every keyframe in a video job goes through the same cicerone.detect() an
image does, so the same source-language priority applies: the charset drives
what the recognizer can read. The server used to call the analyzer without
`languages`, so video ran on the default reader no matter what the project
declared, and a resume could silently stitch two different charsets into one
timeline.
"""
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "server"))

import db  # noqa: E402
import main  # noqa: E402

from tofu.video.analysis import ANALYZER_REVISION  # noqa: E402
from tofu.video.checkpoint import TrackerCheckpoint  # noqa: E402

MANIFEST = {
    "asset_id": "asset-1", "width": 320, "height": 240, "duration": 1.0,
    "frame_count": 10, "fps": 10.0, "time_base": "1/10", "codec": "h264",
}


@pytest.fixture
def video_job(monkeypatch, tmp_path):
    """A job whose asset belongs to a project with a locked source language."""
    monkeypatch.setattr(db, "DB_PATH", tmp_path / "hints.db")
    db.init_db()
    project = db.create_project("subs", "en", asset_kind="video")
    db.update_project(project["id"], source_lang="ja-JP")
    db.link_asset(project["id"], "asset-1", "clip.mp4")
    job = db.create_video_job("asset-1", project["id"], dict(MANIFEST), chunk_size=4)

    monkeypatch.setattr(main, "make_proxy", lambda source, proxy: None)
    monkeypatch.setattr(main, "video_source_fingerprint", lambda source: "fp-1")

    calls = []
    def fake_analyze(source, manifest, **kwargs):
        calls.append(kwargs)
        return [], []
    monkeypatch.setattr(main, "analyze_video", fake_analyze)
    return job, calls, tmp_path / "clip.mp4"


def _issues(job_id: str):
    return db.list_video_timeline(job_id, 0, MANIFEST["frame_count"])["issues"]


def _seed_checkpoint(job_id: str, languages, *, next_frame: int = 4) -> None:
    """Commit one completed chunk carrying a resumable tracker checkpoint."""
    checkpoint = TrackerCheckpoint(
        job_id=job_id, chunk_index=0, next_frame=next_frame,
        params={"chunk_size": 4, "seek_mode": "fast", "languages": languages},
        source_fingerprint="fp-1", analyzer_revision=ANALYZER_REVISION,
    )
    db.commit_video_analysis_chunk(
        job_id, 0, [], [], "digest-0", 1,
        checkpoint=checkpoint.to_dict(), analyzer_revision=ANALYZER_REVISION)


def test_analysis_inherits_the_project_source_language(video_job):
    job, calls, source = video_job
    main._prepare_video_job(job["id"], source)
    assert calls[0]["languages"] == ["ja-JP"]


def test_analysis_runs_language_blind_without_a_project_source(video_job):
    """auto-detect projects must keep the unconstrained reader."""
    job, calls, source = video_job
    # update_project treats None as "leave unchanged", so an auto-detect
    # project is one that never had a source language set.
    auto = db.create_project("auto", "en", asset_kind="video")
    db.link_asset(auto["id"], "asset-2", "clip2.mp4")
    auto_job = db.create_video_job("asset-2", auto["id"], {**MANIFEST, "asset_id": "asset-2"}, chunk_size=4)
    main._prepare_video_job(auto_job["id"], source)
    assert calls[0]["languages"] is None


def test_resume_keeps_a_checkpoint_analyzed_with_the_same_language(video_job):
    job, calls, source = video_job
    _seed_checkpoint(job["id"], ["ja-JP"])
    main._prepare_video_job(job["id"], source, mode="resume")
    resumed = calls[0]["resume"]
    assert resumed is not None and resumed.next_frame == 4
    codes = {issue["code"] for issue in _issues(job["id"])}
    assert "resume_language_changed" not in codes


def test_resume_across_a_language_change_restarts_cold(video_job):
    """A different charset produces different text for the same pixels, so
    continuing would merge two incompatible analyses into one timeline."""
    job, calls, source = video_job
    _seed_checkpoint(job["id"], ["en"])
    main._prepare_video_job(job["id"], source, mode="resume")
    assert calls[0]["resume"] is None
    assert calls[0]["languages"] == ["ja-JP"]
    changed = [i for i in _issues(job["id"]) if i["code"] == "resume_language_changed"]
    assert len(changed) == 1 and changed[0]["severity"] == "warning"
