import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_quarantine_ledger_matches_the_bytes_and_excludes_repair():
    ledger = json.loads((ROOT / "evidence" / "manifest-quarantine-v1.json").read_text(encoding="utf-8"))
    assert ledger["schema"] == "manifest-quarantine/v1"
    quarantined = {entry["asset_id"]: entry for entry in ledger["entries"]}
    recoverable = set(ledger["not_quarantined"]["asset_ids"])
    assert quarantined.keys().isdisjoint(recoverable)
    for entry in quarantined.values():
        raw = (ROOT / entry["path"]).read_bytes()
        assert len(raw) == entry["bytes"]
        assert hashlib.sha256(raw).hexdigest().startswith(entry["sha256_prefix"])
        assert raw and not raw.strip(b"\x00")
