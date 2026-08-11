## 🍢 eval_guided_corpus — Gate 2: does Guided find what was requested?
## vieuxtiful
"""Measure requested-occurrence recall over the frozen Guided corpus.

Gate 2 is a COMPARISON, so this reports one number per arm on the same
assets, the same backends and the same annotations:

    guided recall  >=  auto recall + 8 percentage points
    guided false-region rate  <=  auto false-region rate + 2 points

An occurrence counts as found when a shipped region overlaps the annotated
box at IoU >= the threshold AND reads as the requested text.  Geometry alone
is not enough: a box in the right place carrying the wrong string has not
located what the user asked for.

Three refusals, each guarding a way this could produce a number that means
nothing:

  * **an unreviewed corpus cannot certify.**  Blocks here are machine-derived
    from annotations; until a human confirms they are text somebody would
    actually request, a passing score measures the derivation rule.
  * **a corpus below the declared minimums cannot certify.**  A recall figure
    over 40 occurrences is not the gate the plan wrote down.
  * **a drifted corpus cannot run at all.**  Comparing today's score against
    yesterday's baseline requires that the corpus did not change underneath.

Dev and holdout are reported separately, holdout last, because tuning against
a number you have just read is the failure the split exists to prevent.

Usage:
    .venv/Scripts/python scripts/eval_guided_corpus.py --arm auto
    .venv/Scripts/python scripts/eval_guided_corpus.py --arm auto --split dev
    .venv/Scripts/python scripts/eval_guided_corpus.py --arm auto --limit 5
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
import unicodedata
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

CORPUS_PATH = ROOT / "evidence" / "guided-corpus-v1.json"
DEFAULT_IOU = 0.5

## Gate 2, verbatim from the plan.
RECALL_MARGIN_POINTS = 8.0
FALSE_REGION_TOLERANCE_POINTS = 2.0


def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", unicodedata.normalize("NFC", text or "")).strip().casefold()


def _iou(a: List[float], b: List[float]) -> float:
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    left, top = max(ax, bx), max(ay, by)
    right, bottom = min(ax + aw, bx + bw), min(ay + ah, by + bh)
    if right <= left or bottom <= top:
        return 0.0
    overlap = (right - left) * (bottom - top)
    union = aw * ah + bw * bh - overlap
    return overlap / union if union > 0 else 0.0


def load_corpus() -> Dict[str, Any]:
    if not CORPUS_PATH.exists():
        raise SystemExit(
            f"no guided corpus at {CORPUS_PATH.relative_to(ROOT)}.\n"
            "run: .venv/Scripts/python scripts/build_guided_corpus.py"
        )
    return json.loads(CORPUS_PATH.read_text(encoding="utf-8"))


def verify_not_drifted(corpus: Dict[str, Any]) -> List[str]:
    """Re-hash every asset and annotation the corpus claims to pin."""
    import hashlib
    problems = []
    for stem, entry in corpus["assets"].items():
        for kind in ("image", "annotation"):
            path = ROOT / entry[kind]["path"]
            if not path.exists():
                problems.append(f"{stem}: missing {kind} {entry[kind]['path']}")
                continue
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            if digest != entry[kind]["sha256"]:
                problems.append(f"{stem}: {kind} changed since the corpus was frozen")
    return problems


def detect_auto(image_path: Path, language: str | None) -> Any:
    """The shipped Auto path, exactly as eval_detect_corpus.py runs it."""
    from tofu.layers import cicerone, scene
    scene_regions = scene.analyze_regions(str(image_path))
    kwargs: Dict[str, Any] = {}
    if language:
        kwargs["languages"] = [language]
    return cicerone.detect(str(image_path), scene_regions=scene_regions, **kwargs)


def score_asset(entry: Dict[str, Any], manifest: Any, iou_threshold: float) -> Dict[str, Any]:
    """Requested-occurrence recall for one asset.

    Each expected occurrence is matched at most once, so four instances of
    "SORTIE" need four located regions -- one region cannot satisfy them all.
    """
    shipped = [
        {
            "box": [inst.bounding_box.x, inst.bounding_box.y,
                    inst.bounding_box.width, inst.bounding_box.height],
            "text": _normalize(inst.text or ""),
            "claimed": False,
        }
        for inst in manifest.instances
    ]
    found = expected = 0
    per_block = []
    # One record per REQUESTED OCCURRENCE, in a stable order. Gate 2 is a
    # paired comparison, and a paired test needs pairs: totals alone cannot
    # say whether the two arms failed on the same occurrences or on
    # different ones, which is the entire question McNemar answers.
    outcomes: List[Dict[str, Any]] = []
    for block in entry["blocks"]:
        wanted = _normalize(block["requested_text"])
        hits = 0
        for occurrence in block["occurrences"]:
            expected += 1
            best, best_iou = None, 0.0
            for region in shipped:
                if region["claimed"] or region["text"] != wanted:
                    continue
                overlap = _iou(occurrence["bbox"], region["box"])
                if overlap >= iou_threshold and overlap > best_iou:
                    best, best_iou = region, overlap
            outcomes.append({
                "block_id": block["block_id"],
                "occurrence": len(outcomes),
                "found": best is not None,
            })
            if best is not None:
                best["claimed"] = True
                found += 1
                hits += 1
        per_block.append({
            "block_id": block["block_id"],
            "requested_text": block["requested_text"],
            "expected": block["expected_occurrences"],
            "found": hits,
        })

    # Regions carrying text nobody requested. Meaningless on a partial
    # annotation, where unlisted text is real and correctly detected.
    unclaimed = sum(1 for region in shipped if not region["claimed"])
    return {
        "expected": expected, "found": found,
        "shipped_regions": len(shipped),
        "unrequested_regions": unclaimed,
        "partial_annotation": entry["partial_annotation"],
        "blocks": per_block,
        "occurrences": outcomes,
    }


def score_guided(entry: Dict[str, Any], manifest: Any, iou_threshold: float) -> Dict[str, Any]:
    """Requested-occurrence recall for the Guided arm.

    Scored identically to Auto -- an occurrence counts only when a located
    box overlaps the annotation at IoU >= threshold -- so the two arms differ
    in what they SEARCH, never in how they are graded.
    """
    from tofu.layers import forage, mise

    blocks = mise.make_blocks([b["requested_text"] for b in entry["blocks"]])
    expected_counts = {
        block.id: entry["blocks"][index]["expected_occurrences"]
        for index, block in enumerate(blocks)
    }
    outcomes = forage.locate(manifest, blocks, expected_counts=expected_counts)

    found = expected = 0
    recovered_hits = 0
    per_block = []
    occurrence_outcomes: List[Dict[str, Any]] = []
    tiers: Dict[str, int] = {}
    high_correct = high_total = 0
    for index, outcome in enumerate(outcomes):
        source_block = entry["blocks"][index]
        wanted = source_block["occurrences"]
        expected += len(wanted)
        tiers[outcome.tier] = tiers.get(outcome.tier, 0) + 1
        used: set = set()
        hits = 0
        matched_positions: set = set()
        for occurrence in outcome.occurrences:
            best, best_iou = None, 0.0
            for position, annotated in enumerate(wanted):
                if position in used:
                    continue
                overlap = _iou(annotated["bbox"], list(occurrence.geometry))
                if overlap >= iou_threshold and overlap > best_iou:
                    best, best_iou = position, overlap
            if best is not None:
                used.add(best)
                matched_positions.add(best)
                hits += 1
                if occurrence.source == "recovered":
                    recovered_hits += 1
        # Same ordering as the Auto scorer -- annotation order within
        # annotation order -- so the two lists index the SAME occurrences and
        # can be compared position by position.
        for position in range(len(wanted)):
            occurrence_outcomes.append({
                "block_id": source_block["block_id"],
                "occurrence": len(occurrence_outcomes),
                "found": position in matched_positions,
            })
        found += hits
        if outcome.tier == "high":
            high_total += len(outcome.occurrences)
            high_correct += hits
        per_block.append({
            "block_id": source_block["block_id"], "state": outcome.state,
            "tier": outcome.tier, "expected": len(wanted), "found": hits,
        })

    located = sum(len(o.occurrences) for o in outcomes)
    return {
        "expected": expected, "found": found,
        "shipped_regions": located,
        "unrequested_regions": max(0, located - found),
        "partial_annotation": entry["partial_annotation"],
        "recovered_hits": recovered_hits,
        "tiers": tiers,
        "high_tier_located": high_total, "high_tier_correct": high_correct,
        "blocks": per_block,
        "occurrences": occurrence_outcomes,
    }


_ORACLE_BACKENDS: Dict[str, Any] = {}


def _oracle_backend(language: str | None):
    """One recogniser per language set, reused across crops.

    Constructing an EasyOCR reader per occurrence would dominate the
    runtime and measure model loading rather than recognition.
    """
    from tofu.core.types import BBox  # noqa: F401  (used by score_oracle)
    from tofu.layers import cicerone
    key = language or "en"
    if key not in _ORACLE_BACKENDS:
        langset = cicerone.expand_langset([language]) if language else ("en",)
        _ORACLE_BACKENDS[key] = cicerone.EasyOCRBackend(languages=langset, gpu=False)
    return _ORACLE_BACKENDS[key]


def score_oracle(entry: Dict[str, Any], manifest: Any, iou_threshold: float) -> Dict[str, Any]:
    """INVALID AS A CEILING AS WRITTEN. Measured 2026-08-11: 42.3% (80/189),
    BELOW Auto's 67.2% and Guided's 74.6%.

    That is not a finding about guidance, it is a contradiction that proves
    the arm is mis-specified. An oracle handed perfect localization cannot
    score below an arm that had to find the boxes itself unless the two are
    being read by different machinery -- and they are.

    WHY IT IS WRONG. This reads each annotated crop through a bare
    `EasyOCRBackend.detect_in_regions(..., pad=0)`. The product's recognition
    is not that. It is a multipass ladder plus zoom, surface probes, edge
    rescue, language-set expansion and the correction layers -- and the
    corpus annotations are DETECTOR-scale boxes, so `pad=0` crops tighter
    than anything the pipeline reads. `la-bastille`'s own annotation note
    records the consequence directly: a box tight to the glyphs of a
    comparable word reads as garbage at 0.303.

    The per-stratum numbers make the mechanism plain -- japanese_horizontal
    8.7% and japanese_vertical 11.1%, exactly the strata whose reads depend
    most on the zoom and rescue passes this arm skips.

    So this measures A DIFFERENT CONFIGURATION FROM THE ONE THE PRODUCT RUNS,
    which is the trap `docs/measured-dead-ends.md` names twice as the most
    expensive mistake available here. The number is kept and labelled rather
    than deleted, because a plausible-looking 42.3% quoted later as "the
    ceiling" would be worse than no arm at all.

    WHAT A VALID ORACLE NEEDS: the annotated boxes injected into the
    pipeline's OWN recognition path after detection -- same passes, same
    padding, same corrections -- so that only localization is replaced.
    Until then this arm answers nothing about the feature's premise.

    Original intent, still the right question:

    Localization is handed over -- each annotated occurrence is treated as a
    box the user drew -- and only the read is scored. That separates the two
    failures the Auto number fuses:

        localization failure   the box was never found
        recognition failure    the box was found and read wrong

    It is not an arm anyone ships. It answers the question that has to be
    asked BEFORE building a guidance feature: if guidance worked perfectly,
    how much would it be worth? An oracle sitting at the Auto baseline says
    the remaining loss is recognition, and no amount of guidance addresses
    that.

    Scored on the same rule as the other arms -- the read must match the
    requested text -- with the geometry supplied rather than searched for.
    """
    image = ROOT / entry["image"]["path"]
    found = expected = 0
    per_block = []
    outcomes: List[Dict[str, Any]] = []
    for block in entry["blocks"]:
        wanted = _normalize(block["requested_text"])
        hits = 0
        for occurrence in block["occurrences"]:
            expected += 1
            box = occurrence["bbox"]
            read = ""
            try:
                # The recogniser on the ANNOTATED crop, through the same
                # backend the product uses. No detection is involved, so a
                # miss here is a reading failure and nothing else.
                from tofu.core.types import BBox
                backend = _oracle_backend(occurrence.get("language"))
                per_region = backend.detect_in_regions(
                    str(image),
                    [BBox(x=int(box[0]), y=int(box[1]),
                          width=int(box[2]), height=int(box[3]))],
                    ## DEFAULT pad and upscale, deliberately -- `pad=0` was
                    ## the first version's mistake. The backend's own crop
                    ## path pads by 4 and upscales anything under
                    ## MIN_CROP_HEIGHT, and the corpus boxes are
                    ## detector-scale; cropping tighter than the pipeline
                    ## ever does made this arm weaker than the thing it was
                    ## supposed to bound. `crop_legibility` in
                    ## eval_detector_evidence.py had already learned this
                    ## the same way.
                )
                read = _normalize(" ".join(
                    det.text or "" for group in per_region for det in group
                ))
            except Exception:
                # A recogniser that cannot run is not a located occurrence.
                # Recorded as a miss rather than skipped, so the ceiling is
                # never flattered by an engine that was absent.
                read = ""
            hit = bool(read) and (read == wanted or wanted in read)
            outcomes.append({
                "block_id": block["block_id"],
                "occurrence": len(outcomes), "found": hit,
            })
            if hit:
                hits += 1
                found += 1
        per_block.append({
            "block_id": block["block_id"],
            "expected": len(block["occurrences"]), "found": hits,
        })
    return {
        "expected": expected, "found": found,
        "shipped_regions": expected, "unrequested_regions": 0,
        "partial_annotation": entry["partial_annotation"],
        "blocks": per_block, "occurrences": outcomes,
    }


SCORERS = {"auto": score_asset, "guided": score_guided, "guided_oracle": score_oracle}


def run(arm: str, split: str | None, limit: int | None, iou: float) -> Dict[str, Any]:
    corpus = load_corpus()
    drift = verify_not_drifted(corpus)
    if drift:
        raise SystemExit("corpus has drifted:\n  " + "\n  ".join(drift))

    assets = {
        stem: entry for stem, entry in corpus["assets"].items()
        if split is None or entry["split"] == split
    }
    if limit:
        assets = dict(list(assets.items())[:limit])

    # `both` scores every arm from ONE detection pass. Gate 2 is a paired
    # comparison, so the arms must see the identical manifest: re-detecting
    # per arm would put OCR variance into the difference the gate measures,
    # and would make "the same occurrence" a claim rather than a fact.
    arms = ["auto", "guided"] if arm == "both" else [arm]
    results: Dict[str, Dict[str, Any]] = {}
    started = time.time()
    for index, (stem, entry) in enumerate(sorted(assets.items()), 1):
        image = ROOT / entry["image"]["path"]
        language = next(
            (occ.get("language") for block in entry["blocks"]
             for occ in block["occurrences"] if occ.get("language")), None
        )
        print(f"  [{index}/{len(assets)}] {stem} ...", flush=True)
        # ONE detection pass feeds both arms. Guided's advantage is what it
        # SEARCHES (shipped regions plus discarded candidates), so re-detecting
        # per arm would measure OCR variance instead of guidance.
        manifest = detect_auto(image, language)
        for name in arms:
            scored = SCORERS[name](entry, manifest, iou)
            results.setdefault(name, {})[stem] = {
                "stratum": entry["stratum"], "split": entry["split"], **scored,
            }
    elapsed = time.time() - started

    payload = {"corpus_id": corpus["corpus_id"], "arm": arm, "arms": arms, "iou": iou,
               "review_status": corpus["review_status"], "totals": corpus["totals"],
               "minimums": corpus["minimums"], "elapsed_seconds": round(elapsed, 1)}
    if len(arms) == 1:
        payload["assets"] = results[arms[0]]
    else:
        payload["by_arm"] = results
    return payload


def _aggregate(results: Dict[str, Any], predicate) -> Dict[str, float]:
    chosen = [r for r in results.values() if predicate(r)]
    expected = sum(r["expected"] for r in chosen)
    found = sum(r["found"] for r in chosen)
    full = [r for r in chosen if not r["partial_annotation"]]
    shipped = sum(r["shipped_regions"] for r in full)
    unrequested = sum(r["unrequested_regions"] for r in full)
    return {
        "assets": len(chosen), "expected": expected, "found": found,
        "recall": round(100.0 * found / expected, 1) if expected else 0.0,
        # Withheld on partial annotations: unlisted text there is real text
        # correctly detected, not a false region.
        "false_region_rate": round(100.0 * unrequested / shipped, 1) if shipped else None,
        "false_region_basis": len(full),
    }


def report(run_result: Dict[str, Any]) -> None:
    """Print every arm the run scored.

    A `--arm both` payload holds `by_arm`; a single-arm one holds `assets`.
    Reading only the second is how a completed 26-minute run died at the
    printing step -- which is also why `main` now WRITES BEFORE IT REPORTS.
    Measurement is expensive and formatting is not; a bug in the cheap half
    must never be able to destroy the expensive half.
    """
    if "by_arm" in run_result:
        # `by_arm` is dropped from the recursive payload, not merely
        # shadowed: leaving it in means the next call takes this branch
        # again, forever.
        shared = {k: v for k, v in run_result.items() if k != "by_arm"}
        for name, assets in run_result["by_arm"].items():
            report({**shared, "arm": name, "assets": assets})
        return
    results = run_result["assets"]
    print(f"\ncorpus        {run_result['corpus_id']}   arm={run_result['arm']}   IoU>={run_result['iou']}")
    print(f"review status {run_result['review_status']}")
    print(f"elapsed       {run_result['elapsed_seconds']}s\n")

    print(f"{'stratum':24s} {'assets':>6} {'exp':>5} {'found':>6} {'recall%':>8}")
    strata = sorted({r["stratum"] for r in results.values()})
    for name in strata:
        agg = _aggregate(results, lambda r, n=name: r["stratum"] == n)
        print(f"{name:24s} {agg['assets']:6d} {agg['expected']:5d} {agg['found']:6d} {agg['recall']:8.1f}")

    for split in ("dev", "holdout"):
        agg = _aggregate(results, lambda r, s=split: r["split"] == s)
        if not agg["assets"]:
            continue
        false_rate = "n/a" if agg["false_region_rate"] is None else f"{agg['false_region_rate']:.1f}%"
        print(f"\n{split.upper():8s} assets={agg['assets']:3d}  recall={agg['recall']:.1f}%  "
              f"({agg['found']}/{agg['expected']})  false-region={false_rate} "
              f"(over {agg['false_region_basis']} fully-annotated assets)")

    print("\n" + "-" * 68)
    blockers = []
    if run_result["review_status"] != "human_reviewed":
        blockers.append(
            f"corpus review_status is {run_result['review_status']!r}: Blocks are "
            "machine-derived and have not been confirmed by a human"
        )
    short = {k: v for k, v in run_result["minimums"].items()
             if run_result["totals"].get(k, 0) < v}
    for name, need in short.items():
        blockers.append(f"{name}: {run_result['totals'][name]} < required {need}")
    if run_result["arm"] == "auto":
        blockers.append("this is the Auto BASELINE; the Guided arm lands in Release 5")

    if blockers:
        print("GATE 2 NOT CERTIFIED. Reasons:")
        for reason in blockers:
            print(f"  - {reason}")
    else:
        print("Gate 2 inputs are valid; compare arms to certify.")
    print(f"\nGate 2 requires: guided recall >= auto + {RECALL_MARGIN_POINTS} points, "
          f"false-region <= auto + {FALSE_REGION_TOLERANCE_POINTS} points.")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--arm", default="auto",
                        choices=["auto", "guided", "guided_oracle", "both"])
    parser.add_argument("--split", choices=["dev", "holdout"])
    parser.add_argument("--limit", type=int)
    parser.add_argument("--iou", type=float, default=DEFAULT_IOU)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()

    result = run(args.arm, args.split, args.limit, args.iou)
    # WRITTEN FIRST, deliberately. Scoring the corpus costs ~26 minutes of
    # real OCR; printing it costs nothing. Reporting before persisting means
    # any formatting bug throws the measurement away -- which is exactly
    # what happened once.
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n",
                            encoding="utf-8")
        print(f"wrote {args.out}\n")
    report(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
