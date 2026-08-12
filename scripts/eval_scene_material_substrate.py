"""Compare contaminated and glyph-excluded Scene material observations."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def _instances(path: Path):
    from tofu.core.types import BBox, InstText, Mask

    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = []
    if "regions" in payload:
        for index, region in enumerate(payload["regions"]):
            x, y, width, height = region["bbox"]
            rows.append(InstText(
                id=f"gt-{index + 1}",
                bounding_box=BBox(x, y, width, height),
                text=region.get("text"),
            ))
        return rows, "tracked_gt_boxes"
    for index, instance in enumerate(payload.get("instances", [])):
        box = instance["bounding_box"]
        mask_data = instance.get("segmentation_mask")
        mask = None
        if isinstance(mask_data, dict) and mask_data.get("polygon"):
            mask = Mask(
                polygon=[tuple(point) for point in mask_data["polygon"]],
                confidence=float(mask_data.get("confidence", 0.0)),
                holes=[
                    [tuple(point) for point in hole]
                    for hole in mask_data.get("holes", [])
                ] or None,
            )
        rows.append(InstText(
            id=str(instance.get("id") or f"manifest-{index + 1}"),
            bounding_box=BBox(**box),
            segmentation_mask=mask,
            text=instance.get("text"),
        ))
    return rows, "manifest_masks_and_boxes"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", action="append", required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    from tofu.core.types import TextManifest
    from tofu.layers import scene

    assets = []
    transitions = Counter()
    for case in args.case:
        image_text, annotation_text = case.split("=", 1)
        image, annotations = Path(image_text), Path(annotation_text)
        instances, exclusion_source = _instances(annotations)
        manifest = TextManifest(
            asset_id=image.stem,
            total_regions=len(instances),
            instances=instances,
        )
        scene.analyze(str(image), manifest)
        rows = []
        for index, region in enumerate(manifest.scene_regions):
            contaminated = region.material_evidence
            excluded = scene.substrate_material_evidence(
                str(image), region, manifest.instances
            )
            before = contaminated["material_class"] if contaminated else "unmeasured"
            after = excluded["material_class"] if excluded else "unmeasured"
            transitions[(before, after)] += 1
            rows.append({
                "surface_id": f"{image.stem}:surface-{index + 1}",
                "contaminated": contaminated,
                "glyph_excluded": excluded,
                "class_changed": before != after,
            })
        assets.append({
            "image": str(image),
            "annotations": str(annotations),
            "exclusion_source": exclusion_source,
            "glyph_regions": len(instances),
            "surfaces": len(rows),
            "rows": rows,
        })

    report = {
        "schema": 1,
        "kind": "scene_material_substrate_ablation",
        "observational_only": True,
        "decision_eligible": False,
        "review_schema": "evidence/scene-material-review-schema-v1.json",
        "surfaces": sum(asset["surfaces"] for asset in assets),
        "glyph_excluded_measurable": sum(
            row["glyph_excluded"] is not None
            for asset in assets for row in asset["rows"]
        ),
        "glyph_excluded_unmeasurable": sum(
            row["glyph_excluded"] is None
            for asset in assets for row in asset["rows"]
        ),
        "class_changes_among_measurable": sum(
            row["class_changed"] and row["glyph_excluded"] is not None
            for asset in assets for row in asset["rows"]
        ),
        "mean_substrate_coverage": (
            sum(
                row["glyph_excluded"]["substrate"]["coverage"]
                for asset in assets for row in asset["rows"]
                if row["glyph_excluded"] is not None
            )
            / sum(
                row["glyph_excluded"] is not None
                for asset in assets for row in asset["rows"]
            )
        ),
        "transitions": [
            {"contaminated": before, "glyph_excluded": after, "count": count}
            for (before, after), count in sorted(transitions.items())
        ],
        "assets": assets,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "surfaces": report["surfaces"],
        "glyph_excluded_measurable": report["glyph_excluded_measurable"],
        "glyph_excluded_unmeasurable": report["glyph_excluded_unmeasurable"],
        "class_changes_among_measurable": report["class_changes_among_measurable"],
        "mean_substrate_coverage": report["mean_substrate_coverage"],
        "transitions": report["transitions"],
    }, ensure_ascii=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
