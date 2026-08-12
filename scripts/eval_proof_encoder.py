"""Evaluate a proof-encoder checkpoint on unseen strings and typefaces."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--texts", type=Path, required=True)
    parser.add_argument("--observed-font", required=True)
    parser.add_argument("--candidate-font", required=True)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()

    import numpy as np
    import torch

    from tofu.layers import proof, proof_encoder

    texts = [
        line.strip() for line in args.texts.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    model = proof_encoder.build_checkpoint_model(checkpoint)
    model.load_state_dict(checkpoint["model"])
    model.eval()

    candidates = []
    for text in texts:
        image = proof.render(text, args.candidate_font)
        if image is not None:
            candidates.append((text, proof_encoder.prepare(image)))
    candidate_tensor = torch.from_numpy(np.stack([image for _, image in candidates])).unsqueeze(1)
    with torch.no_grad():
        candidate_vectors = model(candidate_tensor)

    rows = []
    for truth in texts:
        observed = proof.render(truth, args.observed_font)
        if observed is None:
            continue
        tensor = torch.from_numpy(proof_encoder.prepare(observed)).unsqueeze(0).unsqueeze(0)
        with torch.no_grad():
            if hasattr(model, "forward_with_length"):
                vector, length_logits = model.forward_with_length(tensor)
                predicted_length = int(length_logits.argmax(dim=1)[0]) + 1
            else:
                vector = model(tensor)
                predicted_length = None
            scores = candidate_vectors @ vector[0]
        order = torch.argsort(scores, descending=True)
        ranked = [candidates[int(index)][0] for index in order]
        truth_rank = ranked.index(truth) + 1 if truth in ranked else None
        rows.append({
            "truth": truth,
            "best": ranked[0],
            "top1": ranked[0] == truth,
            "rank": truth_rank,
            "support": round(float(scores[int(order[0])]), 6),
            "margin": round(float(scores[int(order[0])] - scores[int(order[1])]), 6),
            "predicted_length": predicted_length,
            "length_top1": predicted_length == len(truth) if predicted_length else None,
        })

    top1 = sum(row["top1"] for row in rows)
    report = {
        "schema": 1,
        "checkpoint": str(args.checkpoint),
        "observed_font": args.observed_font,
        "candidate_font": args.candidate_font,
        "scored": len(rows),
        "top1": top1,
        "top1_rate": top1 / len(rows) if rows else None,
        "length_top1_rate": (
            sum(row["length_top1"] for row in rows) / len(rows)
            if rows and rows[0]["length_top1"] is not None else None
        ),
        "rows": rows,
    }
    if args.out:
        args.out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
