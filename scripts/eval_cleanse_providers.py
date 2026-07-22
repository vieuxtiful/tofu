"""Versioned Cleanse-provider benchmark and promotion gate.

Creates synthetic cases with a known clean background, overlays text, then
compares Cleanse's candidate result only inside the true text mask while also
requiring bit-stable pixels outside the mask.  Real assets can be added as
review-only cases; they never manufacture a ground truth score.

Run: .venv/Scripts/python scripts/eval_cleanse_providers.py [--tag baseline]
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tofu.core.types import BBox, BgProfil, InstText, Mask, SceneRegion, TextManifest
from tofu.layers import cleanse, inpaint_providers
from tofu.utils.imaging import text_mask


def _font(size):
    for name in ("arialbd.ttf", "DejaVuSans-Bold.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            pass
    return ImageFont.load_default()


def _case(name):
    w, h = 480, 260
    if name == "flat":
        clean = Image.new("RGB", (w, h), "#21633b")
        texture, label = "flat", "panel"
    elif name == "gradient":
        x = np.linspace(0, 1, w)
        arr = np.zeros((h, w, 3), dtype=np.uint8)
        arr[..., 0], arr[..., 1], arr[..., 2] = 35 + 150*x, 65 + 80*x, 145 - 55*x
        clean, texture, label = Image.fromarray(arr), "smooth_gradient", "panel"
    elif name == "repeating":
        yy, xx = np.mgrid[0:h, 0:w]
        arr = np.zeros((h, w, 3), dtype=np.uint8)
        arr[..., 0] = 60 + ((xx // 18) % 2) * 35
        arr[..., 1] = 95 + ((yy // 18) % 2) * 35
        arr[..., 2] = 125 + (((xx + yy) // 24) % 2) * 25
        clean, texture, label = Image.fromarray(arr), "textured", "surface"
    else:
        rng = np.random.default_rng(71)
        arr = rng.normal(125, 20, (h // 4, w // 4, 3)).clip(0, 255).astype("uint8")
        clean, texture, label = Image.fromarray(arr).resize((w, h), Image.Resampling.BICUBIC), "textured", "surface"
    source = clean.copy()
    draw = ImageDraw.Draw(source)
    font = _font(52)
    box = draw.textbbox((100, 95), "TOFU TEST", font=font)
    draw.text((100, 95), "TOFU TEST", font=font, fill="white", stroke_width=2 if name == "outlined" else 1, stroke_fill="black")
    bbox = BBox(box[0]-3, box[1]-3, box[2]-box[0]+6, box[3]-box[1]+6)
    inst = InstText("r1", bbox, text="TOFU TEST", target_text="TOFU TEST",
                    segmentation_mask=Mask([(bbox.x,bbox.y),(bbox.x+bbox.width,bbox.y),(bbox.x+bbox.width,bbox.y+bbox.height),(bbox.x,bbox.y+bbox.height)], .9),
                    background_profile=BgProfil(texture=texture))
    return clean, source, TextManifest(name, 1, [inst], scene_regions=[SceneRegion(bbox, label, 1., texture=texture)])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="baseline")
    ap.add_argument("--max-mask-mae", type=float, default=30.0)
    ap.add_argument("--max-outside-mae", type=float, default=.01)
    ap.add_argument("--min-mask-recall", type=float, default=.94)
    ap.add_argument("--strict", action="store_true", help="exit non-zero when a protected case fails")
    args = ap.parse_args()
    cases = []
    for name in ("flat", "gradient", "textured", "repeating", "outlined"):
        clean, source, manifest = _case(name)
        before = np.asarray(source)
        b = manifest.instances[0].bounding_box
        # The actual Cleanse mask is a diagnostic, not the ground truth.  The
        # synthetic overlay lets us measure whether it covers all changed text
        # pixels, including anti-aliasing/strokes, before judging repair
        # fidelity.  This prevents a too-small mask from looking artificially
        # good just because evaluation used the same deficient mask.
        import cv2
        full, mask_evidence = cleanse._select_region_mask(
            np, cv2, before, b, before.shape[0], before.shape[1],
            manifest.instances[0].segmentation_mask.polygon, None,
        )
        clean_arr = np.asarray(clean)
        truth_mask = np.any(before != clean_arr, axis=2)
        covered_truth = int((full & truth_mask).sum())
        mask_recall = covered_truth / max(1, int(truth_mask.sum()))
        candidate_observations = []

        def observe_candidate(info, candidate, candidate_mask):
            """Score the actual neural proposal before Cleanse falls back.

            An unpromoted provider intentionally never changes ``output``.
            Evaluation must nevertheless measure its candidate, otherwise a
            Telea fallback could be mistakenly reported as model evidence.
            """
            candidate_arr = np.asarray(candidate, dtype=np.uint8)
            repair_mask = np.asarray(candidate_mask, dtype=bool)
            candidate_observations.append({
                "provider": info["provider"],
                "group_ids": info["group_ids"],
                "decision": info["decision"],
                "mask_mae": round(float(np.abs(candidate_arr[repair_mask].astype(float) - clean_arr[repair_mask]).mean()), 4),
                "outside_mae": round(float(np.abs(candidate_arr[~repair_mask].astype(float) - before[~repair_mask]).mean()), 6),
                "mask_recall": round(int((repair_mask & truth_mask).sum()) / max(1, int(truth_mask.sum())), 5),
                "quality_gate": info["quality_gate"],
                "execution": info["execution"],
            })
            return None

        t0 = time.time()
        output = cleanse.erase(source, manifest, candidate_observer=observe_candidate).convert("RGB")
        elapsed = time.time() - t0
        after = np.asarray(output)
        mask_mae = float(np.abs(after[full].astype(float) - np.asarray(clean)[full]).mean())
        outside_mae = float(np.abs(after[~full].astype(float) - before[~full]).mean())
        repair = manifest.instances[0].repair_provenance or {}
        cases.append({
            "case": name,
            "mask_mae": round(mask_mae, 4),
            "outside_mae": round(outside_mae, 6),
            "mask_recall": round(mask_recall, 5),
            "mask_evidence": mask_evidence,
            "seconds": round(elapsed, 3),
            "repair": repair,
            "provider": repair.get("executed_provider", "none"),
            "candidate_observations": candidate_observations,
        })
    baseline_passed = all(
        c["mask_mae"] <= args.max_mask_mae
        and c["outside_mae"] <= args.max_outside_mae
        and c["mask_recall"] >= args.min_mask_recall
        for c in cases
    )
    provider_reports = {}
    for provider in {c["provider"] for c in cases}:
        provider_cases = [c for c in cases if c["provider"] == provider]
        provider_reports[provider] = {
            "cases": [c["case"] for c in provider_cases],
            "eligible": bool(provider_cases) and all(
                c["mask_mae"] <= args.max_mask_mae
                and c["outside_mae"] <= args.max_outside_mae
                and c["mask_recall"] >= args.min_mask_recall
                for c in provider_cases
            ),
        }
    configured = {p["id"]: p for p in inpaint_providers.provider_statuses()}
    candidate_reports = {}
    for provider, status in configured.items():
        if status.get("kind") not in {"self_hosted", "experimental"}:
            continue
        observations = [
            observation for case in cases for observation in case["candidate_observations"]
            if observation["provider"] == provider
        ]
        if not observations:
            continue
        candidate_reports[provider] = {
            "cases": sorted({case["case"] for case in cases if any(
                observation["provider"] == provider for observation in case["candidate_observations"]
            )}),
            "candidate_count": len(observations),
            "synthetic_eligible": all(
                observation["mask_mae"] <= args.max_mask_mae
                and observation["outside_mae"] <= args.max_outside_mae
                and observation["mask_recall"] >= args.min_mask_recall
                for observation in observations
            ),
        }
    promotion_eligible = {
        # Synthetic backgrounds prove file-contract and basic fidelity, not
        # correctness on protected real assets.  A provider stays review-only
        # until its real-fixture benchmark extends this report.
        provider: False
        for provider in candidate_reports
    }
    result = {
        "schema": 4,
        "tag": args.tag,
        "baseline_eligible": baseline_passed,
        "provider_promotion_eligible": promotion_eligible,
        "provider_reports": provider_reports,
        "candidate_reports": candidate_reports,
        "provider_status": list(configured.values()),
        "promotion_reason": "A neural provider must pass every synthetic and protected real-fixture gate; synthetic-only success records candidate fidelity but is insufficient for production promotion.",
        "thresholds": {"max_mask_mae": args.max_mask_mae, "max_outside_mae": args.max_outside_mae, "min_mask_recall": args.min_mask_recall},
        "cases": cases,
    }
    out = ROOT / "scripts" / "eval_out" / f"cleanse-providers-{args.tag}.json"; out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    print(f"report: {out}")
    if args.strict and not baseline_passed:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
