"""How many regions could Scene actually condition a corruption on? (S6)

THE GATING QUESTION, ASKED BEFORE BUILDING ON THE ANSWER. Stage S6 wants
training-time candidate corruptions drawn from the operator family the
observed surface could have produced -- `D_{M,P}(R(g_i); z_D)` of the
technical paper's eq. (8). That requires a material class Scene has
measured, on substrate it actually sampled, from a classifier someone
calibrated. This script counts how many annotated regions clear that bar.

It is deliberately cheap: Scene's surface pass and a geometric join, no
detection, no OCR, no encoder. The number it reports is the ceiling on what
S6 can possibly affect, so it is worth having before any training arm is
built rather than after one has been trained and measured to no effect.

WHY THE ANSWER IS NOT "MOST OF THEM". Three independent gates have to pass
and each fails on the frozen corpus for its own reason: a region must land
on a surface at all, that surface must carry a material class other than
`unknown`, and the observation must be `calibrated` rather than `unfitted`.
Scene ships `calibration_status: "unfitted"` unconditionally today, so the
third gate alone closes every region -- which is correct behaviour, not a
defect. Uncalibrated Scene evidence is MISSING evidence, and the programme's
standing rule is that it may not condition anything.

Usage:
    .venv/Scripts/python scripts/eval_scene_conditioning.py
    .venv/Scripts/python scripts/eval_scene_conditioning.py --out evidence/x.json
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

REPORT_SCHEMA = "scene-conditioning-coverage-v1"
DEFAULT_OUT = ROOT / "evidence" / "scene-conditioning-coverage-v1.json"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", type=Path, default=ROOT / "evidence" / "corpus-v1.json")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    from eval_channel_envelope import _script_of

    from tofu.core.types import BBox, InstText, TextManifest
    from tofu.layers import marinade, scene

    corpus = json.loads(args.corpus.read_text(encoding="utf-8"))
    materials: Counter[str] = Counter()
    trust: Counter[str] = Counter()
    calibration: Counter[str] = Counter()
    reasons: Counter[str] = Counter()
    by_script: dict[str, dict[str, int]] = {}
    rows: list[dict[str, Any]] = []

    for stem, entry in sorted(corpus["fixtures"].items()):
        annotation = json.loads(
            (ROOT / entry["annotation"]["path"]).read_text(encoding="utf-8")
        )
        regions = [
            region for region in annotation.get("regions", [])
            if (region.get("text") or "").strip()
        ]
        if not regions:
            continue
        surfaces = scene.analyze_regions(str(ROOT / entry["image"]["path"]))
        instances = [
            InstText(id=f"r{index}", bounding_box=BBox(*region["bbox"]),
                     text=region["text"])
            for index, region in enumerate(regions)
        ]
        manifest = TextManifest(
            asset_id=stem, total_regions=len(instances),
            instances=instances, scene_regions=surfaces,
        )
        scene._attach_surface_observations(manifest)

        for instance, region in zip(instances, regions, strict=True):
            observation = instance.surface_observation
            family = marinade.family_for_observation(observation)
            record = observation or {}
            script = _script_of(region["text"])
            materials[str(record.get("material_class") or "no_surface")] += 1
            trust[str(record.get("substrate_trust") or "no_surface")] += 1
            calibration[str(record.get("calibration_status") or "no_surface")] += 1
            for reason in family["reasons"]:
                reasons[reason] += 1
            bucket = by_script.setdefault(script, {"regions": 0, "conditioned": 0})
            bucket["regions"] += 1
            bucket["conditioned"] += int(family["conditioned"])
            rows.append({
                "fixture": stem,
                "text": region["text"],
                "stratum_script": script,
                "material_class": record.get("material_class"),
                "substrate_trust": record.get("substrate_trust"),
                "calibration_status": record.get("calibration_status"),
                "conditioned": family["conditioned"],
                "processes": list(family["processes"]),
                "reasons": family["reasons"],
            })

    conditioned = sum(1 for row in rows if row["conditioned"])
    report = {
        "schema": REPORT_SCHEMA,
        "corpus": str(args.corpus),
        "corpus_id": corpus.get("corpus_id"),
        "regions": len(rows),
        "conditioned": conditioned,
        "conditioned_rate": round(conditioned / len(rows), 4) if rows else None,
        "unconditioned_family_size": len(marinade.UNCONDITIONED),
        "material_class": dict(materials.most_common()),
        "substrate_trust": dict(trust.most_common()),
        "calibration_status": dict(calibration.most_common()),
        ## Which gate closed, counted. Three independent ones have to pass,
        ## and knowing WHICH fails is what says whether the remedy is more
        ## surface detection, better material classification, or simply
        ## calibrating the classifier that already exists.
        "declined_reasons": dict(reasons.most_common()),
        "by_script": by_script,
        "note": (
            "conditioned == 0 is the expected result while Scene reports "
            "calibration_status 'unfitted'; uncalibrated evidence is missing "
            "evidence and may not condition a corruption"
        ),
        "rows": rows,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8",
    )
    print(json.dumps(
        {key: value for key, value in report.items() if key != "rows"},
        ensure_ascii=False, indent=2,
    ))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
