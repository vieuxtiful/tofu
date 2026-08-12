"""Evaluate Proof retrieval only on corpus rows whose provenance permits scoring."""

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


def _regions(path: Path) -> list[dict]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = []
    for index, region in enumerate(payload.get("regions", [])):
        text = (region.get("text") or "").strip()
        if not text or not _contains_han(text):
            continue
        x, y, width, height = region["bbox"]
        rows.append({
            "id": f"gt-{index + 1}",
            "text": text,
            "box": {"x": x, "y": y, "width": width, "height": height},
            "vertical": bool(region.get("vertical")),
        })
    return rows


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

    rendered, owners = [], []
    for index, text in enumerate(candidates):
        for font in fonts:
            image = proof.render(text, font)
            if image is not None:
                rendered.append(proof_encoder.prepare(image))
                owners.append(index)
    if not rendered:
        return None
    with torch.no_grad():
        observed_tensor = torch.from_numpy(
            proof_encoder.prepare(observed)
        ).unsqueeze(0).unsqueeze(0)
        if hasattr(model, "forward_with_length"):
            observed_batch, length_logits = model.forward_with_length(observed_tensor)
            observed_vector = observed_batch[0]
            length_probability = torch.softmax(length_logits[0], dim=0)
            predicted_length = int(length_probability.argmax()) + 1
            length_support = float(length_probability.max())
        else:
            observed_vector = model(observed_tensor)[0]
            predicted_length = None
            length_support = None
        vectors = model(torch.from_numpy(np.stack(rendered)).unsqueeze(1))
    prototypes = []
    for index in range(len(candidates)):
        selected = vectors[[owner == index for owner in owners]].mean(dim=0)
        prototypes.append(torch.nn.functional.normalize(selected, dim=0))
    scores = torch.stack(prototypes) @ observed_vector
    order = torch.argsort(scores, descending=True)
    best = int(order[0])
    runner = int(order[1]) if len(order) > 1 else best
    return {
        "best": candidates[best],
        "support": round(float(scores[best]), 6),
        "margin": round(float(scores[best] - scores[runner]), 6),
        "predicted_length": predicted_length,
        "length_support": round(length_support, 6) if length_support is not None else None,
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
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--geometry-checkpoint", type=Path)
    parser.add_argument("--font", action="append", required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    import torch

    from tofu.layers import proof_encoder
    from tofu.utils.imaging import load_rgb

    corpus = json.loads(args.corpus.read_text(encoding="utf-8"))
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    model = proof_encoder.build_checkpoint_model(checkpoint)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    geometry_model = None
    if args.geometry_checkpoint:
        from tofu.layers import proof_geometry

        geometry_checkpoint = torch.load(
            args.geometry_checkpoint, map_location="cpu", weights_only=True
        )
        geometry_model = proof_geometry.build_model(geometry_checkpoint["max_length"])
        geometry_model.load_state_dict(geometry_checkpoint["model"])
        geometry_model.eval()

    assets = []
    all_rows = []
    for asset in corpus["assets"]:
        record = {
            "id": asset["id"],
            "stratum": asset["stratum"],
            "score_eligible": asset["score_eligible"],
            "primary_eligible": asset["primary_eligible"],
            "calibration_eligible": asset["calibration_eligible"],
            "reason": asset["reason"],
            "rows": [],
        }
        if asset["score_eligible"]:
            image = load_rgb(str(ROOT / asset["image"]))
            regions = _regions(ROOT / asset["annotations"])
            candidates = sorted({region["text"] for region in regions})
            for region in regions:
                observed = _mask(image, region["box"])
                if observed is None:
                    continue
                from tofu.layers import proof_encoder

                canonical = proof_encoder.canonical_reading_direction(
                    observed, vertical=region["vertical"]
                )
                geometry = None
                if geometry_model is not None:
                    from tofu.layers import proof_geometry

                    geometry = proof_geometry.predict(geometry_model, canonical)
                encoder_raw = _encoder_rank(model, observed, candidates, args.font)
                deterministic_raw = _deterministic_rank(observed, candidates, args.font)
                encoder = _encoder_rank(model, canonical, candidates, args.font)
                deterministic = _deterministic_rank(canonical, candidates, args.font)
                row = {
                    "region_id": region["id"],
                    "truth": region["text"],
                    "characters": len(region["text"]),
                    "vertical": region["vertical"],
                    "geometry": geometry,
                    "raw": {
                        "encoder_best": encoder_raw["best"],
                        "encoder_top1": encoder_raw["best"] == region["text"],
                        "deterministic_best": deterministic_raw["best"],
                        "deterministic_top1": deterministic_raw["best"] == region["text"],
                    },
                    "encoder": {**encoder, "top1": encoder["best"] == region["text"]},
                    "deterministic": {
                        "best": deterministic["best"],
                        "support": deterministic["support"],
                        "margin": deterministic["margin"],
                        "top1": deterministic["best"] == region["text"],
                    },
                }
                record["rows"].append(row)
                all_rows.append(row)
        record["scored"] = len(record["rows"])
        record["encoder_top1"] = sum(row["encoder"]["top1"] for row in record["rows"])
        record["deterministic_top1"] = sum(
            row["deterministic"]["top1"] for row in record["rows"]
        )
        record["raw_encoder_top1"] = sum(
            row["raw"]["encoder_top1"] for row in record["rows"]
        )
        record["raw_deterministic_top1"] = sum(
            row["raw"]["deterministic_top1"] for row in record["rows"]
        )
        record["length_top1"] = sum(
            row["geometry"]["length"] == row["characters"]
            for row in record["rows"]
            if row["geometry"] is not None
        )
        geometry_rows = [row for row in record["rows"] if row["geometry"] is not None]
        record["length_mae"] = (
            sum(abs(row["geometry"]["length"] - row["characters"]) for row in geometry_rows)
            / len(geometry_rows)
            if geometry_rows else None
        )
        assets.append(record)

    report = {
        "schema": 1,
        "kind": "proof_real_ink_corpus_evaluation",
        "corpus": str(args.corpus),
        "checkpoint": str(args.checkpoint),
        "geometry_checkpoint": str(args.geometry_checkpoint) if args.geometry_checkpoint else None,
        "normalization": {
            "vertical": "rotate counter-clockwise so top-to-bottom becomes left-to-right",
            "length": "report exact Unicode code-point length strata; do not use the unknown answer to filter candidates",
        },
        "scored": len(all_rows),
        "primary_scored": sum(
            asset["scored"] for asset in assets if asset["primary_eligible"]
        ),
        "calibration_scored": sum(
            asset["scored"] for asset in assets if asset["calibration_eligible"]
        ),
        "encoder_top1": sum(row["encoder"]["top1"] for row in all_rows),
        "deterministic_top1": sum(row["deterministic"]["top1"] for row in all_rows),
        "raw_encoder_top1": sum(row["raw"]["encoder_top1"] for row in all_rows),
        "raw_deterministic_top1": sum(
            row["raw"]["deterministic_top1"] for row in all_rows
        ),
        "length_top1": sum(
            row["geometry"]["length"] == row["characters"]
            for row in all_rows
            if row["geometry"] is not None
        ),
        "length_mae": (
            sum(
                abs(row["geometry"]["length"] - row["characters"])
                for row in all_rows if row["geometry"] is not None
            )
            / sum(row["geometry"] is not None for row in all_rows)
            if any(row["geometry"] is not None for row in all_rows) else None
        ),
        "length_strata": {
            str(length): {
                "scored": sum(row["characters"] == length for row in all_rows),
                "encoder_top1": sum(
                    row["characters"] == length and row["encoder"]["top1"]
                    for row in all_rows
                ),
                "deterministic_top1": sum(
                    row["characters"] == length and row["deterministic"]["top1"]
                    for row in all_rows
                ),
            }
            for length in sorted({row["characters"] for row in all_rows})
        },
        "assets": assets,
    }
    args.out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
