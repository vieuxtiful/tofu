"""Does component alignment separate the true reading? (stage S5)

THE MEASUREMENT THAT WAS SUPPOSED TO HAPPEN AND DID NOT. The fusion plan's
Phase 5 required an ablation of the transport plan "on exactly the same
regions and candidate pools as the global encoder". That ablation was
reported, but it could not have measured anything: `fusion._features` read
an `alignment.get("support")` key that `proof_alignment.align()` has never
produced, so the feature was permanently `None` and the plan reached no
decision. Two tests asserted its absence and passed. This script is the
first time the numbers exist.

WHAT IS ASKED. For every scored region, every candidate string is rendered
and transported onto the observed ink. Three questions, in increasing
order of what they would license:

  1. does the argmin of alignment cost pick the truth more often than
     chance?  -- is there any signal at all
  2. is the truth's cost lower than the wrong candidates' on average?
     -- separation, which is what a fitted coefficient would exploit
  3. does alignment rank the truth first where the encoder does not?
     -- incremental value, which is the only thing that justifies the
     runtime it costs

A "no" to (2) is decisive on its own: a feature that does not separate
cannot be given a useful weight by any calibration, and adding it to the
vector would buy latency and a missing-feature abstention path.

ALIGNMENT CANNOT RECOMMEND. Whatever this reports, the transport plan may
not independently promote a candidate -- solver failure, timeout or a
degenerate mask must produce missing evidence, never a negative identity
verdict. This measures whether it deserves a coefficient, not whether it
deserves a vote.

Usage:
    .venv/Scripts/python scripts/eval_alignment_ablation.py \
        --corpus evidence/proof-real-ink-corpus-v1.json \
        --checkpoint evidence/proof-encoder-zh-hant-v1.pt \
        --font C:/Windows/Fonts/msjh.ttc --font C:/Windows/Fonts/msgothic.ttc \
        --out evidence/alignment-ablation-v1.json
"""
from __future__ import annotations

import argparse
import json
import sys
import unicodedata
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from tofu.layers.flight import sign_test  # noqa: E402

REPORT_SCHEMA = "alignment-ablation-v1"


def _contains_han(text: str) -> bool:
    return any("CJK UNIFIED IDEOGRAPH" in unicodedata.name(char, "") for char in text)


def _regions(path: Path, han_only: bool = True) -> list[dict]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = []
    for index, region in enumerate(payload.get("regions", [])):
        text = (region.get("text") or "").strip()
        if not text or (han_only and not _contains_han(text)):
            continue
        x, y, width, height = region["bbox"]
        rows.append({
            "id": f"gt-{index + 1}", "text": text,
            "box": {"x": x, "y": y, "width": width, "height": height},
            "vertical": bool(region.get("vertical")),
        })
    return rows


def _assets(corpus: dict, path: Path) -> list[dict]:
    """Normalize the two corpus formats this can read.

    The real-ink corpus (`assets`, with eligibility policy) is nine CJK
    regions -- too few to say anything about separation with confidence.
    The frozen detection corpus (`fixtures`, hashed and split) is 66
    annotated regions across six scripts, and alignment needs none of the
    machinery the eligibility policy exists to gate: no encoder, no
    candidate generation, just masks, renderings and a transport plan.

    Reading both is what lets the same question be asked at 7x the sample.
    """
    if "assets" in corpus:
        return [
            {
                "id": asset["id"],
                "image": ROOT / asset["image"],
                "annotations": ROOT / asset["annotations"],
                "han_only": True,
            }
            for asset in corpus["assets"] if asset["score_eligible"]
        ]
    return [
        {
            "id": stem,
            "image": ROOT / entry["image"]["path"],
            "annotations": ROOT / entry["annotation"]["path"],
            "han_only": False,
        }
        for stem, entry in sorted(corpus.get("fixtures", {}).items())
    ]


def _mask(image, box):
    import numpy as np

    from tofu.core.types import BBox
    from tofu.utils.imaging import text_mask

    mask = text_mask(image, BBox(**box))
    if mask is None or not mask.any():
        return None
    rows = np.any(mask, axis=1)
    columns = np.any(mask, axis=0)
    y0, y1 = np.where(rows)[0][[0, -1]]
    x0, x1 = np.where(columns)[0][[0, -1]]
    return mask[y0 : y1 + 1, x0 : x1 + 1]


def _alignment_cost(observed, candidate_text, fonts, weights, revision):
    """Best transport cost for one candidate, over the faces that render it.

    Best rather than mean, matching `proof_runtime._score`: a face the
    candidate does not suit is evidence about the FACE, and letting it drag
    the candidate's cost up would measure the font cohort.
    """
    from tofu.layers import proof
    from tofu.layers.proof_alignment import align

    best = None
    for font in fonts:
        rendered = proof.render(candidate_text, font)
        if rendered is None:
            continue
        result = align(observed, rendered, weights=weights, revision=revision)
        if result is None:
            continue
        if best is None or result["alignment_cost"] < best["alignment_cost"]:
            best = result
    return best


def _by_script(scored: list[dict[str, Any]]) -> dict[str, Any]:
    """Per-script separation, top-1 and its chance rate."""
    from collections import defaultdict

    from eval_channel_envelope import _script_of

    from tofu.layers.flight import sign_test

    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in scored:
        grouped[_script_of(row["truth"])].append(row)
    report = {}
    for script, rows in sorted(grouped.items()):
        differences = [row["wrong_cost_mean"] - row["truth_cost"] for row in rows]
        report[script] = {
            "regions": len(rows),
            "alignment_top1": sum(row["alignment_top1"] for row in rows),
            "chance": round(sum(1 / row["candidates"] for row in rows) / len(rows), 4),
            "truth_cheaper_than_wrong": sum(1 for value in differences if value > 0),
            "separation_p_value": round(sign_test(differences), 6),
        }
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--font", action="append", required=True)
    parser.add_argument("--cost-weights", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    import torch

    from tofu.layers import proof_encoder
    from tofu.layers.proof_alignment import load_cost_weights
    from tofu.utils.imaging import load_rgb

    weights, revision = load_cost_weights(
        str(args.cost_weights) if args.cost_weights else None
    )
    corpus = json.loads(args.corpus.read_text(encoding="utf-8"))
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    model = proof_encoder.build_checkpoint_model(checkpoint)
    model.load_state_dict(checkpoint["model"])
    model.eval()

    from eval_proof_real_ink_corpus import _encoder_rank

    rows: list[dict[str, Any]] = []
    for asset in _assets(corpus, args.corpus):
        image = load_rgb(str(asset["image"]))
        if image is None:
            continue
        regions = _regions(asset["annotations"], han_only=asset["han_only"])
        candidates = sorted({region["text"] for region in regions})
        for region in regions:
            observed = _mask(image, region["box"])
            if observed is None:
                continue
            canonical = proof_encoder.canonical_reading_direction(
                observed, vertical=region["vertical"]
            )
            costs = {}
            for candidate in candidates:
                result = _alignment_cost(
                    canonical, candidate, args.font, weights, revision
                )
                if result is not None:
                    costs[candidate] = result
            if len(costs) < 2:
                continue
            ranked = sorted(costs, key=lambda text: costs[text]["alignment_cost"])
            encoder = _encoder_rank(model, canonical, candidates, args.font)
            truth = region["text"]
            wrong = [text for text in costs if text != truth]
            rows.append({
                "asset": asset["id"],
                "region_id": region["id"],
                "truth": truth,
                "candidates": len(costs),
                "alignment_best": ranked[0],
                "alignment_top1": ranked[0] == truth,
                "encoder_best": encoder["best"],
                "encoder_top1": encoder["best"] == truth,
                "truth_cost": (
                    costs[truth]["alignment_cost"] if truth in costs else None
                ),
                "wrong_cost_mean": (
                    sum(costs[text]["alignment_cost"] for text in wrong) / len(wrong)
                    if wrong else None
                ),
                "truth_matched_mass": (
                    costs[truth]["matched_mass"] if truth in costs else None
                ),
                "costs": {
                    text: round(costs[text]["alignment_cost"], 6) for text in ranked
                },
            })

    scored = [row for row in rows if row["truth_cost"] is not None]
    separated = [
        row for row in scored if row["truth_cost"] < row["wrong_cost_mean"]
    ]
    ## Chance for argmin over a pool: the mean of 1/K, not 1/mean(K).
    chance = (
        sum(1.0 / row["candidates"] for row in scored) / len(scored) if scored else None
    )
    report = {
        "schema": REPORT_SCHEMA,
        "corpus": str(args.corpus),
        "checkpoint": str(args.checkpoint),
        "cost_weights": list(weights),
        "cost_weights_revision": revision,
        "fonts": list(args.font),
        "scored_regions": len(scored),
        "alignment_top1": sum(row["alignment_top1"] for row in scored),
        "encoder_top1": sum(row["encoder_top1"] for row in scored),
        "alignment_top1_chance": round(chance, 4) if chance else None,
        ## Question 2, and the decisive one: on how many regions is the
        ## truth cheaper to transport than the average wrong candidate? At
        ## chance this is half of them.
        "truth_cheaper_than_wrong": len(separated),
        "mean_truth_cost": (
            round(sum(row["truth_cost"] for row in scored) / len(scored), 6)
            if scored else None
        ),
        "mean_wrong_cost": (
            round(sum(row["wrong_cost_mean"] for row in scored) / len(scored), 6)
            if scored else None
        ),
        ## Question 3: regions the encoder gets wrong and alignment gets
        ## right. This is the only bucket that justifies the runtime.
        "alignment_rescues_encoder": sum(
            1 for row in scored if row["alignment_top1"] and not row["encoder_top1"]
        ),
        "alignment_breaks_encoder": sum(
            1 for row in scored if row["encoder_top1"] and not row["alignment_top1"]
        ),
        "separation_p_value": round(
            sign_test([row["wrong_cost_mean"] - row["truth_cost"] for row in scored]), 6
        ) if scored else None,
        ## Stratified because the headline hides the mechanism. Component
        ## features are the centroid, log-area and log-aspect of connected
        ## components -- for alphabetic scripts a component is roughly a
        ## letter and their arrangement is identity-bearing, while a single
        ## Han or Hangul glyph is many components of radicals and strokes.
        ## A pooled number would report the alphabetic result as the
        ## system's.
        "by_script": _by_script(scored),
        "limitations": [
            "alignment may not independently recommend a candidate; this "
            "measures whether it deserves a fitted coefficient, not a vote",
            "candidate pools are the other annotated readings of the same "
            "asset, which is the retrieval harness's convention and is "
            "smaller and easier than a production pool",
        ],
        "rows": rows,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8",
    )
    print(json.dumps(
        {k: v for k, v in report.items() if k not in ("rows", "limitations")},
        ensure_ascii=False, indent=2,
    ))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
