"""Inventory observational Scene material evidence without changing routing."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=Path, action="append", required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    from tofu.core.types import TextManifest
    from tofu.layers import scene

    assets = []
    total = Counter()
    for path in args.image:
        manifest = TextManifest(asset_id=path.stem, total_regions=0, instances=[])
        scene.analyze(str(path), manifest)
        rows = []
        for index, region in enumerate(manifest.scene_regions):
            evidence = region.material_evidence
            material_class = evidence["material_class"] if evidence else "unmeasured"
            total[material_class] += 1
            rows.append({
                "region": index + 1,
                "bbox": {
                    "x": region.bbox.x,
                    "y": region.bbox.y,
                    "width": region.bbox.width,
                    "height": region.bbox.height,
                },
                "surface_label": region.semantic_label,
                "detector_confidence": region.confidence,
                "material_evidence": evidence,
            })
        assets.append({
            "image": str(path),
            "regions": len(rows),
            "classes": dict(sorted(Counter(
                row["material_evidence"]["material_class"]
                if row["material_evidence"] else "unmeasured"
                for row in rows
            ).items())),
            "rows": rows,
        })

    report = {
        "schema": 1,
        "kind": "scene_material_evidence_inventory",
        "observational_only": True,
        "decision_eligible": False,
        "assets": assets,
        "regions": sum(asset["regions"] for asset in assets),
        "classes": dict(sorted(total.items())),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
