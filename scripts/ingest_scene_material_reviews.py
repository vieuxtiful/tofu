"""Validate completed material reviews and emit agreement/eligibility evidence."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--queue", type=Path, required=True)
    parser.add_argument("--schema", type=Path, required=True)
    parser.add_argument("--reviews-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    from tofu.layers import material_review

    queue = json.loads(args.queue.read_text(encoding="utf-8"))
    schema = json.loads(args.schema.read_text(encoding="utf-8"))
    queued = {
        (record["asset_id"], record["surface_id"]): record
        for record in queue["records"]
    }
    completed = []
    outcomes = []
    seen = set()
    files = sorted(args.reviews_dir.glob("*.json")) if args.reviews_dir.exists() else []
    for path in files:
        record = json.loads(path.read_text(encoding="utf-8"))
        errors = material_review.validate(record, schema)
        key = (record.get("asset_id"), record.get("surface_id"))
        if key not in queued:
            errors.append("surface_id: record does not belong to the frozen review queue")
        if key in seen:
            errors.append("surface_id: duplicate completed review record")
        seen.add(key)
        result = material_review.outcome(record, errors)
        source = queued.get(key)
        result.update({
            "file": str(path),
            "substrate_available": bool(source and source.get("glyph_excluded_crop")),
            "substrate_scoring_eligible": bool(
                result["label_eligible"] and source and source.get("glyph_excluded_crop")
            ),
        })
        outcomes.append(result)
        if not errors:
            completed.append(record)

    eligible = sum(result["label_eligible"] for result in outcomes)
    report = {
        "schema": 1,
        "kind": "scene_material_review_status",
        "queue": str(args.queue),
        "review_schema": str(args.schema),
        "queued": len(queued),
        "submitted": len(files),
        "valid": sum(result["valid"] for result in outcomes),
        "invalid": sum(not result["valid"] for result in outcomes),
        "label_eligible": eligible,
        "substrate_scoring_eligible": sum(
            result["substrate_scoring_eligible"] for result in outcomes
        ),
        "pending": len(queued) - len({key for key in seen if key in queued}),
        "agreement": material_review.agreement(completed),
        "production_decision_eligible": False,
        "outcomes": outcomes,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: report[key] for key in (
        "queued", "submitted", "valid", "invalid", "label_eligible",
        "substrate_scoring_eligible", "pending", "agreement",
        "production_decision_eligible",
    )}, indent=2))
    return 1 if report["invalid"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
