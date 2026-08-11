## 🍢 compare_guided_arms — Gate 2 as a PAIRED comparison
## vieuxtiful
"""Does Guided locate more of what was requested than Auto, on the same corpus?

Gate 2 asks for an 8-point recall margin. Two single-arm percentages cannot
answer that honestly at this corpus size: the holdout's n=40 gives roughly
±15.5pp on a point estimate, so an 8-point difference sits comfortably
inside the noise of either number taken alone.

But the arms are not independent samples. They score THE SAME occurrences,
so every occurrence is a matched pair, and the right question is not "are
the two rates different" but "of the occurrences the two arms disagreed
about, did Guided win more of them than chance allows". That is McNemar's
test, and on paired data it is far more powerful than the single-proportion
intervals suggest -- it throws away the (usually large) agreeing majority
and tests only the discordant pairs.

    b = found by Auto, missed by Guided
    c = found by Guided, missed by Auto

Under the null "guidance changes nothing", each discordant occurrence is a
coin flip, so the exact test is a two-sided binomial on c out of b+c at
p=0.5. Exact rather than the chi-square approximation because b+c here is in
the dozens, not the hundreds.

WHAT THIS CANNOT DO. It cannot certify Gate 2 on a corpus whose Blocks are
machine-derived: a significant margin would then be evidence about the
derivation rule, not about guidance. The refusal is inherited from the
corpus file and repeated here rather than assumed to have been checked
upstream.

Usage:
    .venv/Scripts/python scripts/compare_guided_arms.py evidence/guided-paired-v1.json
    .venv/Scripts/python scripts/compare_guided_arms.py --split holdout <file>
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

ROOT = Path(__file__).resolve().parents[1]

RECALL_MARGIN_POINTS = 8.0
FALSE_REGION_TOLERANCE_POINTS = 2.0
ALPHA = 0.05


def binomial_two_sided(successes: int, trials: int, p: float = 0.5) -> float:
    """Exact two-sided binomial p-value, by the method of small likelihoods.

    Sums every outcome at most as probable as the observed one, which is the
    definition that stays correct when the distribution is asymmetric --
    doubling the one-sided tail does not, and at p=0.5 the two agree anyway.
    Written out rather than imported so this harness has no scipy dependency
    and its arithmetic is auditable in place.
    """
    if trials == 0:
        return 1.0
    observed = math.comb(trials, successes) * (p ** successes) * ((1 - p) ** (trials - successes))
    total = 0.0
    for k in range(trials + 1):
        probability = math.comb(trials, k) * (p ** k) * ((1 - p) ** (trials - k))
        # 1e-12 slack: exact ties are common at p=0.5 and float equality
        # would drop the mirrored outcome, halving the p-value.
        if probability <= observed * (1 + 1e-12):
            total += probability
    return min(1.0, total)


def _outcomes(payload: Dict[str, Any], arm: str, split: str | None) -> Dict[Tuple[str, str, int], bool]:
    """Every requested occurrence for one arm, keyed so the arms line up.

    The key is (asset, block, occurrence index). Position within a block is
    the annotation's own order, which both scorers walk identically -- that
    is what makes these pairs rather than two lists of the same length.
    """
    assets = payload.get("by_arm", {}).get(arm) or payload.get("assets")
    if assets is None:
        raise SystemExit(f"no results for arm {arm!r} in this file")
    found: Dict[Tuple[str, str, int], bool] = {}
    for stem, result in assets.items():
        if split and result.get("split") != split:
            continue
        for record in result.get("occurrences", []):
            found[(stem, record["block_id"], record["occurrence"])] = bool(record["found"])
    return found


def _recall(outcomes: Dict[Tuple[str, str, int], bool]) -> Tuple[int, int, float]:
    total = len(outcomes)
    hits = sum(1 for v in outcomes.values() if v)
    return hits, total, (100.0 * hits / total if total else 0.0)


def compare(payload: Dict[str, Any], split: str | None) -> Dict[str, Any]:
    auto = _outcomes(payload, "auto", split)
    guided = _outcomes(payload, "guided", split)

    shared = sorted(set(auto) & set(guided))
    if not shared:
        raise SystemExit("the two arms share no occurrences; they cannot be paired")
    unpaired = len(set(auto) ^ set(guided))

    both = neither = b = c = 0
    for key in shared:
        if auto[key] and guided[key]:
            both += 1
        elif auto[key] and not guided[key]:
            b += 1
        elif guided[key] and not auto[key]:
            c += 1
        else:
            neither += 1

    discordant = b + c
    p_value = binomial_two_sided(c, discordant)
    auto_hits, auto_total, auto_recall = _recall({k: auto[k] for k in shared})
    guided_hits, guided_total, guided_recall = _recall({k: guided[k] for k in shared})

    return {
        "split": split or "full",
        "paired_occurrences": len(shared),
        "unpaired_occurrences": unpaired,
        "both_found": both, "neither_found": neither,
        "auto_only": b, "guided_only": c, "discordant": discordant,
        "auto": {"found": auto_hits, "of": auto_total, "recall": round(auto_recall, 1)},
        "guided": {"found": guided_hits, "of": guided_total, "recall": round(guided_recall, 1)},
        "margin_points": round(guided_recall - auto_recall, 1),
        "p_value": round(p_value, 5),
        "significant": p_value < ALPHA,
        "meets_margin": (guided_recall - auto_recall) >= RECALL_MARGIN_POINTS,
    }


def report(payload: Dict[str, Any], results: List[Dict[str, Any]]) -> None:
    print(f"corpus        {payload.get('corpus_id')}")
    print(f"review status {payload.get('review_status')}")
    print(f"arms          {', '.join(payload.get('arms', []))}\n")

    for result in results:
        print(f"--- {result['split'].upper()} "
              f"({result['paired_occurrences']} paired occurrences)")
        print(f"  auto    {result['auto']['found']:3d}/{result['auto']['of']:<3d}"
              f"  {result['auto']['recall']:5.1f}%")
        print(f"  guided  {result['guided']['found']:3d}/{result['guided']['of']:<3d}"
              f"  {result['guided']['recall']:5.1f}%   margin {result['margin_points']:+.1f} pts")
        print(f"  agreement: both {result['both_found']}, neither {result['neither_found']}")
        print(f"  discordant: guided-only {result['guided_only']}, "
              f"auto-only {result['auto_only']}  (n={result['discordant']})")
        print(f"  McNemar exact p = {result['p_value']}"
              f"  ({'significant' if result['significant'] else 'not significant'} at {ALPHA})")
        if result["unpaired_occurrences"]:
            print(f"  ! {result['unpaired_occurrences']} occurrence(s) present in one arm only "
                  "-- excluded from the pairing")
        print()

    print("-" * 68)
    blockers = []
    if payload.get("review_status") != "human_reviewed":
        blockers.append(
            f"corpus review_status is {payload.get('review_status')!r}: a significant "
            "margin here is evidence about the Block-derivation rule, not about guidance"
        )
    full = next((r for r in results if r["split"] == "full"), results[0])
    if not full["meets_margin"]:
        blockers.append(
            f"margin {full['margin_points']:+.1f} pts < required "
            f"+{RECALL_MARGIN_POINTS} pts on the full corpus"
        )
    if not full["significant"]:
        blockers.append(f"McNemar p = {full['p_value']} is not significant at {ALPHA}")

    if blockers:
        print("GATE 2 NOT CERTIFIED. Reasons:")
        for reason in blockers:
            print(f"  - {reason}")
    else:
        print("Gate 2 margin and significance are met on the full corpus.")
    print("\nDev and holdout are reported separately for transparency. Certification "
          "reads the FULL corpus: n=40 on the holdout gives about +/-15.5pp on a "
          "single-arm estimate, which cannot support or refute an 8-point margin.")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("results", type=Path,
                        help="a --arm both run from eval_guided_corpus.py")
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()

    payload = json.loads(args.results.read_text(encoding="utf-8"))
    if "by_arm" not in payload:
        raise SystemExit(
            "this file holds a single arm. Re-run with --arm both so the two "
            "arms score the SAME detection pass -- pairing across separate runs "
            "would put OCR variance into the difference being tested."
        )

    results = [compare(payload, None)]
    for split in ("dev", "holdout"):
        try:
            results.append(compare(payload, split))
        except SystemExit:
            continue
    report(payload, results)

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(
            {"corpus_id": payload.get("corpus_id"),
             "review_status": payload.get("review_status"),
             "comparisons": results}, indent=2) + "\n", encoding="utf-8")
        print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
