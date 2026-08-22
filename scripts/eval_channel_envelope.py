"""Measure D_{e,c} per channel, and the oracle envelope D* over them (S2).

The question this answers, and it is a gating one: **does the channel family
contain a materially better measurement for any stratum?** If it does not,
the learned channel selector of stage S7 has nothing to select and should
not be built. If it does, the size and location of the gap say where.

WHY THE NUMBER EXISTS AT ALL. Every NED on record was written as though the
system observed each region through one immutable channel. It does not: the
fixed crop path, the multipass rungs, zoom, the merges and the Paddle rescue
are different decoders, and the measurement actually obtained is
D_{e,c}(X) = NED(f_{e,c}(X), t*). A crop is a deterministic transformation
of the available image and cannot ADD information about the latent reading,
but what it loses is stratum-dependent -- approximately sufficient for
Latin, a strict garbling for CJK where context and component merging carry
identity-bearing evidence. Reporting one channel's score as a scene-level
legibility fact is the category error this harness exists to stop.

TWO ARMS, AND THEY ARE NOT LIKE FOR LIKE.

  region_crop     `detect_in_regions` at the ANNOTATED box. Recognition
                  measured from geometry it was handed. This is the arm the
                  tau_unreadable convention is defined against ("from a crop
                  already known to be geometrically valid").

  full_pipeline   the shipped `detect()`, matched back to the annotation by
                  IoU. Recognition AND localization: a region the pipeline
                  never proposed scores 1.0, because from the consumer's
                  side an unfound region and an unreadable one are equally
                  absent.

The asymmetry is real and is not a defect of the harness -- it is what the
two channels are. The report therefore gives every per-channel figure twice:
over all annotated regions, and over the subset the pipeline localized, where
the comparison is recognition against recognition. The envelope is taken over
all regions, because "the best measurement this family can produce for this
region" is the training reference regardless of which half failed.

WHAT IS NOT MEASURED. Zoom and the merge routes are named channels in the
paper and are NOT probed here: they are not separately invocable per region
without reimplementing their trigger conditions, and a probe that fired on
different regions than the pipeline does would measure the harness. They
appear in the report only where `full_pipeline` happened to route through
them, which the channel stamp records. Paddle is probed only if available.

D* IS AN ORACLE. It is computed from t* and is a training and benchmarking
envelope, never an inference-time score. The guard is in
`flight.oracle_envelope`, not in this file's discipline.

Usage:
    .venv/Scripts/python scripts/eval_channel_envelope.py
    .venv/Scripts/python scripts/eval_channel_envelope.py --split holdout
    .venv/Scripts/python scripts/eval_channel_envelope.py --fixture china-street
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import unicodedata
from collections import defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from tofu.layers import flight  # noqa: E402
from tofu.utils.distance import TAU_UNREADABLE  # noqa: E402

OUT = ROOT / "evidence" / "channel-envelope-v1.json"
REPORT_SCHEMA = "channel-envelope-v1"

## IoU at which a shipped region is taken to BE the annotated one. Inherited
## from the detection harness's primary threshold so that "the pipeline found
## this region" means the same thing in both reports.
MATCH_IOU = 0.5

## The script vocabulary this report strata by. Derived from the annotated
## text with `unicodedata`, NOT from the fixture name and NOT from
## `ScriptDetector` -- the pipeline has two script-name vocabularies in it
## already ("han" vs "Hani"), and a third one that silently disagreed with
## both would put every CJK region in the wrong stratum. Named and versioned
## so a later comparison can tell which one produced a number.
SCRIPT_VOCABULARY = "envelope-script-v1"


def _script_of(text: str) -> str:
    """Which script this annotation is written in, from its codepoints."""
    kinds = set()
    for char in text or "":
        if not char.isalpha():
            continue
        name = unicodedata.name(char, "")
        if "CJK UNIFIED" in name:
            kinds.add("han")
        elif "HIRAGANA" in name or "KATAKANA" in name:
            kinds.add("kana")
        elif "HANGUL" in name:
            kinds.add("hangul")
        elif "CYRILLIC" in name:
            kinds.add("cyrillic")
        elif "ARABIC" in name:
            kinds.add("arabic")
        elif "LATIN" in name:
            kinds.add("latin")
        else:
            kinds.add("other")
    if not kinds:
        return "unscripted"
    ## Japanese mixes han and kana within one sign as a matter of course;
    ## calling that "mixed" would scatter the single most important CJK
    ## stratum across two buckets.
    if kinds <= {"han", "kana"}:
        return "japanese_or_han" if "kana" in kinds else "han"
    return next(iter(kinds)) if len(kinds) == 1 else "mixed"


def _iou(a: tuple[float, ...], b: tuple[float, ...]) -> float:
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    left, top = max(ax, bx), max(ay, by)
    right, bottom = min(ax + aw, bx + bw), min(ay + ah, by + bh)
    if right <= left or bottom <= top:
        return 0.0
    overlap = (right - left) * (bottom - top)
    return overlap / (aw * ah + bw * bh - overlap)


def _crop_reading(backend: Any, image: Path, bbox: tuple[int, int, int, int]) -> str:
    """The fixed crop path's reading of one annotated box.

    Goes through `detect_in_regions` -- the pipeline's own crop path, with
    its upscale to MIN_CROP_HEIGHT -- rather than a bare readtext on a
    padded slice. A diagnostic weaker than the thing it diagnoses reports
    the pipeline's strengths as the image's failures.
    """
    from tofu.core.types import BBox

    x, y, width, height = bbox
    found = backend.detect_in_regions(
        str(image), [BBox(x=x, y=y, width=width, height=height)],
    )
    detections = found[0] if found else []
    ## The crop path returns fragments in reading order and the pipeline
    ## joins them, so the joined text is what must be compared.
    return "".join((det.text or "") for det in detections)


def _fixture_language(regions: list[dict[str, Any]]) -> str | None:
    for region in regions:
        if region.get("lang"):
            return str(region["lang"])
    return None


def measure_fixture(stem: str, entry: dict[str, Any]) -> list[dict[str, Any]]:
    """Every annotated region of one fixture, measured through every probe."""
    from tofu.layers import cicerone, scene
    from tofu.layers.cicerone import EasyOCRBackend

    image = ROOT / entry["image"]["path"]
    annotation = json.loads(
        (ROOT / entry["annotation"]["path"]).read_text(encoding="utf-8")
    )
    gt_regions = annotation.get("regions") or []
    language = _fixture_language(gt_regions)

    ## Mirror the corpus harness's production path exactly: scene pre-pass,
    ## then detect with a per-fixture language hint. An arm called
    ## "full_pipeline" that ran a different pipeline would measure nothing.
    started = time.time()
    scene_regions = scene.analyze_regions(str(image))
    detect_kwargs: dict[str, Any] = {"languages": [language]} if language else {}
    manifest = cicerone.detect(
        str(image), scene_regions=scene_regions, **detect_kwargs
    )
    pipeline_seconds = time.time() - started

    backend = EasyOCRBackend(
        languages=tuple([language]) if language else ("en",), gpu=False,
    )

    shipped = [
        (
            (inst.bounding_box.x, inst.bounding_box.y,
             inst.bounding_box.width, inst.bounding_box.height),
            inst,
        )
        for inst in manifest.instances
        if inst.bounding_box is not None
    ]

    rows = []
    for index, region in enumerate(gt_regions):
        truth = (region.get("text") or "").strip()
        if not truth:
            continue
        bbox = tuple(int(value) for value in region["bbox"])

        best_iou, matched = 0.0, None
        for geometry, inst in shipped:
            score = _iou(bbox, geometry)
            if score > best_iou:
                best_iou, matched = score, inst
        localized = best_iou >= MATCH_IOU

        crop_channel = flight.Channel(
            engine="easyocr", route=flight.REGION_CROP, view="crop", crop=bbox,
        )
        measurements = [
            {
                "probe": "region_crop",
                **flight.measure(
                    _crop_reading(backend, image, bbox), truth,
                    engine="easyocr", channel_id=crop_channel.channel_id,
                ),
                "route": crop_channel.route,
                "localized": True,   ## handed the annotated box by construction
            },
            {
                "probe": "full_pipeline",
                **flight.measure(
                    (matched.text if localized and matched else "") or "", truth,
                    engine="easyocr",
                    ## Whatever route flight stamped on the shipped region.
                    ## An unlocalized region has no shipped read and so no
                    ## route -- said explicitly rather than defaulted.
                    channel_id=(
                        (matched.channel_id if localized and matched else None)
                        or f"{flight.UNREGISTERED}-unlocalized"
                    ),
                ),
                "route": (
                    (matched.channel or {}).get("route")
                    if localized and matched else flight.UNREGISTERED
                ),
                "localized": localized,
            },
        ]
        envelope = flight.oracle_envelope(measurements, context="benchmark")
        rows.append({
            "fixture": stem,
            "region_id": f"{stem}#{index}",
            "stratum_declared": entry["stratum"],
            "stratum_script": _script_of(truth),
            "split": entry["split"],
            "text": truth,
            "bbox": list(bbox),
            "vertical": bool(region.get("vertical", False)),
            "best_iou": round(best_iou, 3),
            "localized": localized,
            "measurements": measurements,
            "envelope": envelope,
            "pipeline_seconds": round(pipeline_seconds, 2),
        })
    return rows


def _aggregate(rows: list[dict[str, Any]], key: str) -> dict[str, Any]:
    """Per-stratum, per-probe curves, plus the envelope and the gap it opens."""
    def summarize(values: list[float]) -> dict[str, Any]:
        if not values:
            return {"n": 0, "mean": None, "median": None, "unreadable_rate": None}
        ordered = sorted(values)
        middle = len(ordered) // 2
        median = (
            ordered[middle] if len(ordered) % 2
            else (ordered[middle - 1] + ordered[middle]) / 2
        )
        return {
            "n": len(values),
            "mean": round(sum(values) / len(values), 4),
            "median": round(median, 4),
            "unreadable_rate": round(
                sum(1 for value in values if value >= TAU_UNREADABLE) / len(values), 4,
            ),
        }

    strata: dict[str, dict[str, Any]] = {}
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[row[key]].append(row)

    for stratum, members in sorted(grouped.items()):
        probes: dict[str, list[float]] = defaultdict(list)
        probes_localized: dict[str, list[float]] = defaultdict(list)
        wins: dict[str, int] = defaultdict(int)
        routes: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
        envelope_values: list[float] = []
        envelope_localized: list[float] = []
        for row in members:
            envelope_values.append(row["envelope"]["best_ned"])
            if row["localized"]:
                envelope_localized.append(row["envelope"]["best_ned"])
            best_channel = row["envelope"]["best_channel_id"]
            for measurement in row["measurements"]:
                probes[measurement["probe"]].append(measurement["ned"])
                routes[measurement["probe"]][str(measurement.get("route"))] += 1
                if row["localized"]:
                    probes_localized[measurement["probe"]].append(measurement["ned"])
                if measurement["channel_id"] == best_channel:
                    wins[measurement["probe"]] += 1
        envelope = summarize(envelope_values)
        localized_envelope = summarize(envelope_localized)
        ## The S7 statistic, stated once rather than left to be derived:
        ## how much the best FIXED channel gives up against the envelope,
        ## measured only on regions the pipeline actually localized. Over all
        ## regions this quantity is inflated by a localization failure
        ## wearing a channel-selection costume -- an unfound region scores
        ## 1.0 for the pipeline while the crop path, handed the annotated
        ## box, may score below it. No inference-time selector can recover
        ## that, because at inference there is no annotated box.
        fixed_gap_localized = None
        if localized_envelope["mean"] is not None:
            fixed_means = [
                sum(values) / len(values)
                for values in probes_localized.values() if values
            ]
            if fixed_means:
                fixed_gap_localized = round(
                    min(fixed_means) - localized_envelope["mean"], 4,
                )
                ## The envelope is a minimum over the same channels, so this
                ## cannot truly be negative; rounding just produces -0.0.
                ## Normalized rather than clamped: a genuinely negative value
                ## would be a bug and must stay visible.
                if fixed_gap_localized == 0:
                    fixed_gap_localized = 0.0
        strata[stratum] = {
            "regions": len(members),
            "localized": sum(1 for row in members if row["localized"]),
            "envelope": envelope,
            "envelope_localized": localized_envelope,
            "fixed_gap_localized": fixed_gap_localized,
            ## A probe is not a channel. `region_crop` is one route;
            ## `full_pipeline` is a MIXTURE over whichever routes the
            ## pipeline happened to take, and the equivalence verdicts below
            ## are therefore between probes, not between single channels.
            ## This block is what says how much of a mixture it is -- and if
            ## one route dominates a stratum, that route is the thing S4 may
            ## eventually tie at channel granularity.
            "route_mixture": {
                probe: dict(sorted(counts.items(), key=lambda item: (-item[1], item[0])))
                for probe, counts in sorted(routes.items())
            },
            "probes": {
                name: {
                    **summarize(values),
                    ## Recognition against recognition: the pipeline is not
                    ## charged for regions it never proposed.
                    "localized_only": summarize(probes_localized[name]),
                    "argmin_wins": wins[name],
                    ## How much this channel gives up against the best the
                    ## family could have measured. THIS is the S7 gate: a
                    ## gap of zero means there is nothing to select.
                    "envelope_gap": (
                        round(summarize(values)["mean"] - envelope["mean"], 4)
                        if envelope["mean"] is not None else None
                    ),
                }
                for name, values in sorted(probes.items())
            },
        }
    return strata


def _equivalence(rows: list[dict[str, Any]], key: str) -> dict[str, Any]:
    """Which channel pairs may be tied by the S4 consistency regularizer.

    Fitted on the DEV split only, and only on regions where both channels
    produced a measurement. A pair that is not certified equivalent here may
    not be forced to agree later: a strict garbling and the richer channel
    disagreeing is evidence about channel quality, not a violation.
    """
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if row["split"] == "dev":
            grouped[row[key]].append(row)

    classes: dict[str, Any] = {}
    for stratum, members in sorted(grouped.items()):
        names = sorted({m["probe"] for row in members for m in row["measurements"]})
        pairs = {}
        for left_index, left in enumerate(names):
            for right in names[left_index + 1:]:
                differences = []
                for row in members:
                    by_probe = {m["probe"]: m["ned"] for m in row["measurements"]}
                    if left in by_probe and right in by_probe:
                        differences.append(by_probe[left] - by_probe[right])
                pairs[f"{left}|{right}"] = flight.equivalence_verdict(differences)
        classes[stratum] = pairs
    return classes


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", choices=("dev", "holdout", "all"), default="all")
    parser.add_argument("--fixture", action="append", default=[])
    parser.add_argument("--out", type=Path, default=OUT)
    ## Re-derive strata, envelope and equivalence from an existing report's
    ## per-region rows, without re-running detection. The measurements are
    ## the expensive, non-deterministic-in-wall-clock part; the aggregation
    ## over them is neither, and a report whose derived blocks can only be
    ## regenerated by a fifteen-minute run is a report nobody re-derives.
    parser.add_argument("--reaggregate", type=Path)
    args = parser.parse_args()

    from eval_detect_corpus import assert_corpus_frozen, load_manifest

    if args.reaggregate:
        previous = json.loads(args.reaggregate.read_text(encoding="utf-8"))
        rows = previous["rows"]
        manifest = {
            "corpus_id": previous.get("corpus_id"),
            "split_hash": previous.get("split_hash"),
        }
        split = previous.get("split", args.split)
    else:
        assert_corpus_frozen()
        manifest = load_manifest()
        selected = {
            stem: entry for stem, entry in manifest["fixtures"].items()
            if (args.split == "all" or entry["split"] == args.split)
            and (not args.fixture or stem in args.fixture)
        }
        if not selected:
            raise SystemExit("no fixtures selected")
        rows = []
        for stem, entry in sorted(selected.items()):
            print(f"  {stem} ...", flush=True)
            rows.extend(measure_fixture(stem, entry))
        split = args.split

    report = {
        "schema": REPORT_SCHEMA,
        "corpus_id": manifest.get("corpus_id"),
        "split_hash": manifest.get("split_hash"),
        "split": split,
        "script_vocabulary": SCRIPT_VOCABULARY,
        "match_iou": MATCH_IOU,
        "tau_unreadable": TAU_UNREADABLE,
        "regions": len(rows),
        ## Said in the artifact, not only in the prose: a consumer reading
        ## this file must not be able to mistake the envelope for a score.
        "oracle_notice": (
            "envelope and regret are computed from ground truth; training, "
            "benchmarking and error analysis only, never inference"
        ),
        "limitations": [
            "region_crop is measured from the ANNOTATED box and full_pipeline "
            "includes localization; per-probe means over all regions are not a "
            "like-for-like recognition comparison. Use localized_only for that.",
            "zoom and the merge routes are not separately probed; they appear "
            "only where full_pipeline routed through them.",
            "nine of eleven fixtures carry partial annotations, so these are "
            "per-annotated-region measurements and imply nothing about precision.",
            f"equivalence classes are fitted on the dev split only, at "
            f"margin={flight.EQUIV_MARGIN}, alpha={flight.EQUIV_ALPHA}, "
            f"min_pairs={flight.EQUIV_MIN_PAIRS}; 'underpowered' is not 'equivalent'.",
        ],
        "by_declared_stratum": _aggregate(rows, "stratum_declared"),
        "by_script_stratum": _aggregate(rows, "stratum_script"),
        "equivalence_by_declared_stratum": _equivalence(rows, "stratum_declared"),
        "equivalence_by_script_stratum": _equivalence(rows, "stratum_script"),
        "rows": rows,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8",
    )

    ## `fixed_gap` is the S7 decision: how much the BEST single fixed channel
    ## gives up against the envelope. Zero means one channel already achieves
    ## the best the family can measure, and a selector has nothing to select.
    print(
        f"\n{'stratum':<20}{'n':>4}{'loc':>5}"
        f"{'crop':>8}{'pipe':>8}{'D*':>8}{'gap':>8}{'gap|loc':>9}"
    )
    for stratum, block in report["by_script_stratum"].items():
        probes = block["probes"]
        crop = probes.get("region_crop", {}).get("mean")
        pipe = probes.get("full_pipeline", {}).get("mean")
        best = block["envelope"]["mean"]
        gap = min(
            (p["envelope_gap"] for p in probes.values() if p["envelope_gap"] is not None),
            default=None,
        )
        localized_gap = block["fixed_gap_localized"]
        print(
            f"{stratum:<20}{block['regions']:>4}{block['localized']:>5}"
            f"{crop if crop is not None else '-':>8}"
            f"{pipe if pipe is not None else '-':>8}"
            f"{best if best is not None else '-':>8}"
            f"{gap if gap is not None else '-':>8}"
            f"{localized_gap if localized_gap is not None else '-':>9}"
        )
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
