"""Compare learned and deterministic GT retrieval on annotated real ink."""

from __future__ import annotations

import argparse
import json
import sys
import unicodedata
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def _contains_han(text: str) -> bool:
    return any("CJK UNIFIED IDEOGRAPH" in unicodedata.name(char, "") for char in text)


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


def _encoder_rank(model, observed, candidates, fonts):
    import numpy as np
    import torch

    from tofu.layers import proof, proof_encoder

    observations = torch.from_numpy(proof_encoder.prepare(observed)).unsqueeze(0).unsqueeze(0)
    rendered = []
    owners = []
    for index, text in enumerate(candidates):
        for font in fonts:
            image = proof.render(text, font)
            if image is not None:
                rendered.append(proof_encoder.prepare(image))
                owners.append(index)
    if not rendered:
        return None
    with torch.no_grad():
        observed_vector = model(observations)[0]
        vectors = model(torch.from_numpy(np.stack(rendered)).unsqueeze(1))
    prototypes = []
    for index in range(len(candidates)):
        selected = vectors[[owner == index for owner in owners]].mean(dim=0)
        prototypes.append(torch.nn.functional.normalize(selected, dim=0))
    scores = torch.stack(prototypes) @ observed_vector
    order = torch.argsort(scores, descending=True)
    best, runner = int(order[0]), int(order[1])
    return {
        "best": candidates[best],
        "support": round(float(scores[best]), 6),
        "margin": round(float(scores[best] - scores[runner]), 6),
    }


def _deterministic_rank(observed, candidates, fonts):
    from tofu.layers import proof

    best = None
    for font in fonts:
        result = proof.match(observed, candidates, font)
        if result is not None and (best is None or result["support"] > best["support"]):
            best = result
    return best


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--font", action="append", required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    import torch

    from tofu.layers import proof_encoder
    from tofu.utils.imaging import load_rgb

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    instances = [
        instance for instance in manifest.get("instances", [])
        if _contains_han((instance.get("text") or "").strip())
    ]
    candidates = sorted({instance["text"].strip() for instance in instances})
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    model = proof_encoder.build_checkpoint_model(checkpoint)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    image = load_rgb(str(args.image))

    rows = []
    for instance in instances:
        truth = instance["text"].strip()
        observed = _mask(image, instance["bounding_box"])
        if observed is None:
            continue
        encoder = _encoder_rank(model, observed, candidates, args.font)
        deterministic = _deterministic_rank(observed, candidates, args.font)
        rows.append({
            "region_id": instance.get("id"),
            "truth": truth,
            "bbox": instance["bounding_box"],
            "label_source": (
                "ground_truth_rescue"
                if (instance.get("source_override") or {}).get("kind") == "gt_rescue"
                else "manifest_read"
            ),
            "encoder": {**encoder, "top1": encoder["best"] == truth},
            "deterministic": {
                "best": deterministic["best"],
                "support": deterministic["support"],
                "margin": deterministic["margin"],
                "top1": deterministic["best"] == truth,
            },
        })

    report = {
        "schema": 1,
        "kind": "real_ink_pilot",
        "image": str(args.image),
        "manifest": str(args.manifest),
        "checkpoint": str(args.checkpoint),
        "fonts": args.font,
        "partial_annotation": True,
        "scored": len(rows),
        "candidate_count": len(candidates),
        "encoder_top1": sum(row["encoder"]["top1"] for row in rows),
        "deterministic_top1": sum(row["deterministic"]["top1"] for row in rows),
        "rows": rows,
    }
    args.out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
