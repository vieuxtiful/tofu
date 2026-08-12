"""Fit glyph-margin calibration from labelled encoder evaluation rows."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("evidence", nargs="+", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--revision", default="proof-margin-v1")
    args = parser.parse_args()

    from tofu.layers import proof_calibration

    rows = []
    for path in args.evidence:
        document = json.loads(path.read_text(encoding="utf-8"))
        rows.extend(document.get("rows") or [])
    usable = [row for row in rows if row.get("margin") is not None and "top1" in row]
    calibration = proof_calibration.fit(
        [row["margin"] for row in usable],
        [row["top1"] for row in usable],
        revision=args.revision,
    )
    proof_calibration.save(calibration, args.out)
    print(json.dumps(calibration.to_dict(), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
