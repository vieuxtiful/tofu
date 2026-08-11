"""Run the frozen corpus and report stratified detection quality (P1.1).

`eval_detect.py` measures one image and answers "did this change help on
gemini-street". This answers the question the roadmap actually gates on:
what is held-out recall, per declared stratum, at two IoU thresholds, and
when a region is lost, *how* was it lost.

It refuses to run against a corpus that has drifted from
`evidence/corpus-v1.json`. A stratified report over an unfrozen corpus is
a number without a denominator.

Two properties worth stating because they are easy to lose:

  **Dev and holdout are reported separately, holdout last.** Tuning against
  a number you have just read is the failure the split exists to prevent,
  and putting both in one column invites exactly that.

  **Precision is withheld on partial annotations.** Nine of eleven fixtures
  are partial, so a correct detection of unannotated text would count as a
  false positive. Garbage fraction carries that signal instead.

Usage:
    .venv/Scripts/python scripts/eval_detect_corpus.py
    .venv/Scripts/python scripts/eval_detect_corpus.py --split dev
    .venv/Scripts/python scripts/eval_detect_corpus.py --fixture japan-street
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from detection_metrics import (  # noqa: E402
    MATCHED,
    OUTCOMES,
    Candidate,
    GroundTruthRegion,
    aggregate_strata,
    bootstrap_ci,
    evaluate_fixture,
    floor_violations,
    rect_to_polygon,
)

MANIFEST = ROOT / "evidence" / "corpus-v1.json"
OUT_DIR = ROOT / "scripts" / "eval_out"


def load_manifest() -> dict[str, Any]:
    if not MANIFEST.exists():
        raise SystemExit(
            f"no frozen corpus at {MANIFEST}.\n"
            f"run: .venv/Scripts/python scripts/freeze_corpus.py"
        )
    return json.loads(MANIFEST.read_text(encoding="utf-8"))


def assert_corpus_frozen() -> None:
    """A stratified report over a drifted corpus is meaningless."""
    from freeze_corpus import build

    manifest = load_manifest()
    current = build()
    if current["split_hash"] != manifest["split_hash"]:
        raise SystemExit(
            "corpus has drifted since the freeze -- refusing to report.\n"
            "run: .venv/Scripts/python scripts/freeze_corpus.py --verify"
        )


def load_ground_truth(stem: str, entry: dict[str, Any]) -> list[GroundTruthRegion]:
    data = json.loads((ROOT / entry["annotation"]["path"]).read_text(encoding="utf-8"))
    regions = []
    for index, region in enumerate(data.get("regions", [])):
        regions.append(
            GroundTruthRegion(
                region_id=f"{stem}#{index}",
                polygon=rect_to_polygon(region["bbox"]),
                text=region.get("text"),
                language=region.get("lang"),
                vertical=bool(region.get("vertical", False)),
            )
        )
    return regions


def _ancestor_counts(manifest_obj: Any) -> dict[tuple[float, ...], int]:
    """Distinct detector proposals under each shipped region, keyed by geometry.

    Keyed by geometry rather than id because instance ids (`r1`) and
    candidate ids (`fina-0fc9a50…`) are different namespaces -- an id-keyed
    lookup silently misses every time and reports one ancestor for
    everything, which reads as "the detector did it" for every over-merge.

    KNOWN GAP, measured 2026-08-10: this map is usually empty. `final` nodes
    do not carry the shipped geometry -- instances differ from the last
    recorded node by 1-3px (e.g. raw (239,163,75,46) ships as
    (239,162,75,47)), so some transform after the last lineage record is
    unrecorded. Rather than fuzzy-matching across that gap -- inference from
    final geometry, which has produced phantom defects here before -- the
    lookup misses and the over-merge is reported as *unattributed*. Closing
    the gap means recording that transform in cicerone; until then the
    report says it does not know, which is true.
    """
    lineage = getattr(manifest_obj, "candidate_lineage", None)
    if not lineage:
        return {}
    try:
        from tofu.layers.okara import CandidateGraph

        graph = CandidateGraph.from_dict(lineage)
    except Exception:
        return {}

    # Stages where a proposal ORIGINATES rather than being derived, owned by
    # okara so this cannot drift from what the graph actually records.
    # `graph.raw_ancestors()` counts only raw_craft, which is right for its
    # own callers but would report "no ancestry" for a region whose chain is
    # entirely zoom- or surface-probe-derived -- and those are exactly the
    # dense CJK regions where over-merges live.
    from tofu.layers.okara import ORIGIN_STAGES

    origin_stages = set(ORIGIN_STAGES)

    counts: dict[tuple[float, ...], int] = {}
    for node in graph.nodes("final"):
        try:
            origins = {
                n.candidate_id
                for n in graph.ancestors(node.candidate_id)
                if n.stage in origin_stages
            }
        except Exception:
            continue
        if origins:
            key = tuple(round(float(v), 2) for v in node.geometry)
            counts[key] = len(origins)
    return counts


def _instance_polygon(inst: Any) -> tuple[list[tuple[float, float]], bool]:
    """The instance's quad, or its box with a degradation flag.

    The quad lives on `segmentation_mask.polygon`, not on a `polygon`
    attribute -- an earlier version of this looked for the latter, found
    nothing, and reported `polygon_scored_fraction: 0.0` for the whole
    corpus. Box-scoring every fixture while claiming to be polygon-aware is
    the kind of quiet degradation this field exists to make visible, so it
    is worth naming where the geometry actually is.
    """
    mask = getattr(inst, "segmentation_mask", None)
    poly = getattr(mask, "polygon", None) if mask is not None else None
    if poly and len(poly) >= 3:
        return [(float(p[0]), float(p[1])) for p in poly], False
    box = inst.bounding_box
    return rect_to_polygon((box.x, box.y, box.width, box.height)), True


def fixture_language(gt_regions: list[GroundTruthRegion]) -> str | None:
    """Majority annotated language, as a property of the corpus.

    Taken from the annotation rather than from a previous baseline run, so
    the hint describes the fixture rather than describing what some earlier
    measurement happened to pass in.

    The tags here are BCP-47 (`zh-cn`) while the engines use their own codes
    (`ch_sim`) -- the same two-vocabulary split that once scored every CJK
    region against the Latin model. `cicerone._to_easyocr_lang` is the one
    place that translation belongs, so it is called rather than reimplemented.
    """
    from collections import Counter

    counts = Counter(r.language for r in gt_regions if r.language)
    return counts.most_common(1)[0][0] if counts else None


def run_fixture(stem: str, entry: dict[str, Any]) -> dict[str, Any]:
    from tofu.layers import cicerone, scene

    image = ROOT / entry["image"]["path"]
    gt_regions = load_ground_truth(stem, entry)
    language = fixture_language(gt_regions)

    started = time.time()
    # Mirror eval_detect.py's production path: scene pre-pass, then detect
    # with a per-fixture language hint. The recorded baseline was measured
    # under exactly these conditions ("language_hints: per-fixture").
    scene_regions = scene.analyze_regions(str(image))
    detect_kwargs: dict[str, Any] = {}
    if language:
        detect_kwargs["languages"] = [language]
    text_manifest = cicerone.detect(
        str(image), scene_regions=scene_regions, **detect_kwargs
    )
    elapsed = time.time() - started

    ancestors = _ancestor_counts(text_manifest)
    candidates = []
    resolved = 0
    for inst in text_manifest.instances:
        polygon, degraded = _instance_polygon(inst)
        box = inst.bounding_box
        key = tuple(round(float(v), 2) for v in (box.x, box.y, box.width, box.height))
        count = ancestors.get(key)
        resolved += count is not None
        candidates.append(
            Candidate(
                candidate_id=inst.id,
                polygon=polygon,
                degraded_to_box=degraded,
                raw_ancestor_count=count,
                text=inst.text,
                confidence=inst.confidence,
            )
        )

    report = evaluate_fixture(
        gt_regions, candidates, partial_gt=entry["annotation"]["partial"]
    )
    report["stratum"] = entry["stratum"]
    report["split"] = entry["split"]
    report["language_hint"] = language
    report["seconds"] = round(elapsed, 2)
    # How many SHIPPED regions the lineage could actually be resolved for --
    # not merely whether a graph exists. The graph is always present; the
    # question that matters is whether it reaches the regions being scored,
    # and conflating the two reports full provenance for a run that has none.
    report["lineage_resolved_regions"] = resolved
    report["lineage_coverage"] = (
        round(resolved / len(candidates), 3) if candidates else None
    )
    return report


def build_report(selected: dict[str, dict[str, Any]], manifest: dict[str, Any]) -> dict[str, Any]:
    per_fixture = {stem: run_fixture(stem, entry) for stem, entry in selected.items()}
    strata = {stem: entry["stratum"] for stem, entry in selected.items()}

    by_split: dict[str, Any] = {}
    for split in ("dev", "holdout"):
        members = {s: r for s, r in per_fixture.items() if r["split"] == split}
        if not members:
            continue
        hits: list[float] = []
        counts = {o: 0 for o in OUTCOMES}
        regions = matched = 0
        for report in members.values():
            block = report["by_threshold"]["iou_0.5"]
            regions += block["gt_regions"]
            matched += block["outcomes"][MATCHED]
            hits.extend(1.0 if r["outcome"] == MATCHED else 0.0 for r in block["regions"])
            for outcome, count in block["outcomes"].items():
                counts[outcome] += count
        by_split[split] = {
            "fixtures": sorted(members),
            "gt_regions": regions,
            "matched": matched,
            "recall": round(matched / regions, 4) if regions else 0.0,
            "recall_ci95": bootstrap_ci(hits),
            "outcomes": counts,
        }

    strata_05 = aggregate_strata(per_fixture, strata, "iou_0.5")
    strata_075 = aggregate_strata(per_fixture, strata, "iou_0.75")

    return {
        "corpus_id": manifest["corpus_id"],
        "split_hash": manifest["split_hash"],
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "by_split": by_split,
        "strata_iou_0.5": strata_05,
        "strata_iou_0.75": strata_075,
        "floor_violations_iou_0.5": floor_violations(strata_05),
        "per_fixture": per_fixture,
    }


def print_report(report: dict[str, Any]) -> None:
    print(f"\ncorpus {report['corpus_id']}  split {report['split_hash'][:12]}")

    for split in ("dev", "holdout"):
        block = report["by_split"].get(split)
        if not block:
            continue
        lo, hi = block["recall_ci95"]
        print(
            f"\n{split.upper():8s} recall {block['recall']:.3f} "
            f"[{lo:.3f}, {hi:.3f}]  "
            f"({block['matched']}/{block['gt_regions']} regions)"
        )
        losses = {k: v for k, v in block["outcomes"].items() if k != MATCHED and v}
        if losses:
            print("         losses: " + ", ".join(f"{k}={v}" for k, v in sorted(losses.items())))

    print(f"\n{'stratum':18s} {'n':>4s} {'R@.5':>7s} {'CI95':>16s} {'R@.75':>7s}")
    s05, s75 = report["strata_iou_0.5"], report["strata_iou_0.75"]
    for name in sorted(s05):
        a = s05[name]
        lo, hi = a["recall_ci95"]
        mark = "" if a["sufficient_sample"] else "  (insufficient sample)"
        print(
            f"{name:18s} {a['gt_regions']:4d} {a['recall']:7.3f} "
            f"  [{lo:.2f},{hi:.2f}]  {s75[name]['recall']:7.3f}{mark}"
        )

    violations = report["floor_violations_iou_0.5"]
    print(f"\nfloor violations (recall < 0.80): {len(violations)}")
    for v in violations:
        print(f"  ! {v}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--split", choices=("dev", "holdout", "all"), default="all")
    ap.add_argument("--fixture", help="run a single fixture by stem")
    ap.add_argument("--out", default=None, help="write the JSON report here")
    ap.add_argument(
        "--allow-drift",
        action="store_true",
        help="report even if the corpus no longer matches its freeze (diagnostic only)",
    )
    args = ap.parse_args()

    if not args.allow_drift:
        assert_corpus_frozen()

    manifest = load_manifest()
    selected = {
        stem: entry
        for stem, entry in manifest["fixtures"].items()
        if (args.split == "all" or entry["split"] == args.split)
        and (args.fixture is None or stem == args.fixture)
    }
    if not selected:
        raise SystemExit("no fixtures selected")

    print(f"running {len(selected)} fixture(s) -- roughly {len(selected) * 70}s")
    report = build_report(selected, manifest)
    print_report(report)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = Path(args.out) if args.out else OUT_DIR / "corpus-detection.json"
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
