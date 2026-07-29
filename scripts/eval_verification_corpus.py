"""Run the deterministic verification corpus.

The default mode deliberately disables model-backed re-OCR. It proves the
stable Scribe -> canonical VerificationReport contract while leaving OCR
evidence unavailable rather than fabricated. Use ``--ocr real`` to diagnose
the installed OCR stack without turning dependency/model drift into a gate.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import sys
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tofu.core.types import BBox, InstText, StyleProfil, TextManifest  # noqa: E402
from tofu.layers import cleanse, scene, scribe, verify  # noqa: E402

DEFAULT_SPEC = ROOT / "scripts" / "verification_corpus.json"
DEFAULT_OUTPUT = ROOT / "scripts" / "eval_out" / "verification-corpus"


def load_spec(path: Path) -> dict:
    spec = json.loads(path.read_text(encoding="utf-8"))
    if "waves" in spec:
        spec["cases"] = [
            {**case, "wave": wave["id"]}
            for wave in spec["waves"]
            for case in wave["cases"]
        ]
    if not isinstance(spec.get("cases"), list) or not spec["cases"]:
        raise ValueError("corpus spec must contain a non-empty 'cases' list")
    return spec


def build_manifest(case: dict) -> TextManifest:
    gt_path = ROOT / case["ground_truth"]
    ground_truth = json.loads(gt_path.read_text(encoding="utf-8"))
    instances = []
    for index, region in enumerate(ground_truth["regions"], start=1):
        x, y, width, height = region["bbox"]
        style_data = region.get("style", {})
        style = StyleProfil(
            font_family=style_data.get("font_file"),
            font_weight=style_data.get("weight"),
            italic=style_data.get("italic"),
            color=style_data.get("color"),
            stroke_color=style_data.get("stroke_color"),
            stroke_width=style_data.get("stroke_width"),
            shadow=style_data.get("shadow"),
            target_orientation="vertical" if region.get("vertical") else None,
            # A case-level transform applies to every region, which is what a
            # perspective case wants: the whole sign sits on one plane.
            transform=style_data.get("transform") or case.get("transform"),
        )
        source = region["text"]
        targets = case.get("targets")
        target = targets[index - 1] if targets else (
            source if case.get("target_mode", "identity") == "identity" else None
        )
        instances.append(
            InstText(
                id=f"r{index}",
                bounding_box=BBox(x=x, y=y, width=width, height=height),
                text=source,
                target_text=target,
                language=case["source_language"],
                detected_language=case["source_language"],
                target_language=case["target_language"],
                confidence=1.0,
                reading_order=index - 1,
                style_profile=style,
            )
        )
    return TextManifest(
        asset_id=case["id"],
        total_regions=len(instances),
        instances=instances,
        src_lang=case["source_language"],
        targ_lang=case["target_language"],
        asset_class=case["asset_class"],
        asset_classification={
            "asset_class": case["asset_class"],
            "confidence": 1.0,
            "source": "verification_corpus",
        },
    )


def run_case(case: dict, output_dir: Path, *, real_ocr: bool = False) -> dict:
    image_path = ROOT / case["image"]
    manifest = build_manifest(case)
    if manifest.total_regions != case["expected_regions"]:
        raise AssertionError(
            f"{case['id']}: expected {case['expected_regions']} regions, "
            f"ground truth produced {manifest.total_regions}"
        )

    # Ground-truth files intentionally contain only immutable text geometry.
    # Run the same source-style/background enrichment used by the production
    # render path so Scribe does not fall back to black text indiscriminately.
    manifest = scene.analyze(str(image_path), manifest)
    cleansed = cleanse.erase(str(image_path), manifest)
    localized = scribe.render(cleansed, manifest, manifest.targ_lang or "en")
    if real_ocr:
        report = verify.build_verification_report(localized, manifest)
    else:
        with patch.object(verify, "_get_reader", return_value=None):
            report = verify.build_verification_report(localized, manifest)

    output_dir.mkdir(parents=True, exist_ok=True)
    localized_path = output_dir / f"{case['id']}.localized.png"
    report_path = output_dir / f"{case['id']}.verification.json"
    localized.convert("RGB").save(localized_path)
    report_path.write_text(
        json.dumps(dataclasses.asdict(report), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    actual = report.project.overall_status
    return {
        "id": case["id"],
        "wave": case.get("wave"),
        "expected_regions": case["expected_regions"],
        "actual_regions": len(report.regions),
        "target_language": case["target_language"],
        "asset_class": case["asset_class"],
        "expected_outcome": case["expected_outcome"],
        "actual_outcome": actual,
        "matched": actual == case["expected_outcome"],
        "overall_score": report.project.overall_score,
        "summary": report.project.summary,
        "report": str(report_path.relative_to(ROOT)),
        "localized": str(localized_path.relative_to(ROOT)),
    }


def run_wave(
    spec_path: Path,
    output_dir: Path,
    *,
    real_ocr: bool = False,
    case_ids: set[str] | None = None,
) -> dict:
    spec = load_spec(spec_path)
    selected = [
        case for case in spec["cases"]
        if not case_ids or case["id"] in case_ids
    ]
    missing = (case_ids or set()) - {case["id"] for case in selected}
    if missing:
        raise ValueError(f"unknown corpus case(s): {', '.join(sorted(missing))}")
    cases = [run_case(case, output_dir, real_ocr=real_ocr) for case in selected]
    result = {
        "suite": spec.get("suite", spec.get("wave")),
        "ocr_mode": "real" if real_ocr else "disabled_deterministic",
        "passed": all(case["matched"] for case in cases),
        "case_count": len(cases),
        "cases": cases,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "summary.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--spec", type=Path, default=DEFAULT_SPEC)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--ocr", choices=("off", "real"), default="off")
    parser.add_argument("--case", action="append", dest="cases",
                        help="run only this case id; repeat to select multiple")
    args = parser.parse_args()
    result = run_wave(
        args.spec.resolve(),
        args.output.resolve(),
        real_ocr=args.ocr == "real",
        case_ids=set(args.cases) if args.cases else None,
    )
    for case in result["cases"]:
        marker = "PASS" if case["matched"] else "FAIL"
        print(
            f"{marker} {case['id']}: {case['actual_outcome']} "
            f"(expected {case['expected_outcome']}, score={case['overall_score']})"
        )
    print(f"summary: {args.output.resolve() / 'summary.json'}")
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
