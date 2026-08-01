"""Generate baseline_metrics.json from the eval harnesses.

Runs each harness programmatically, collects metrics per fixture, and
writes them to tests/fixtures/regression/baseline_metrics.json using
the tolerances module.  Tolerances default to 0 (exact match); widen
them manually after reviewing the initial baseline.

usage: .venv/Scripts/python tests/regression/generate_baseline.py [--harness NAME]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from tolerances import update_baseline  # noqa: E402

FIXTURES_DIR = ROOT / "tests" / "fixtures"


def _fixture_names() -> list[str]:
    return sorted(gt.name.removesuffix(".gt.json") for gt in FIXTURES_DIR.glob("*.gt.json"))


def gen_cleanse() -> dict[str, dict[str, float]]:
    from eval_cleanse_providers import _case
    from tofu.layers import cleanse
    import numpy as np
    import cv2

    results = {}
    for name in ("flat", "gradient", "textured", "repeating", "outlined"):
        clean, source, manifest = _case(name)
        before = np.asarray(source)
        b = manifest.instances[0].bounding_box
        full, _ = cleanse._select_region_mask(
            np, cv2, before, b, before.shape[0], before.shape[1],
            manifest.instances[0].segmentation_mask.polygon, None,
        )
        output = cleanse.erase(source, manifest).convert("RGB")
        after = np.asarray(output)
        clean_arr = np.asarray(clean)
        mask_mae = float(np.abs(after[full].astype(float) - clean_arr[full]).mean())
        outside_mae = float(np.abs(after[~full].astype(float) - before[~full]).mean())
        truth_mask = np.any(before != clean_arr, axis=2)
        mask_recall = int((full & truth_mask).sum()) / max(1, int(truth_mask.sum()))
        results[name] = {
            "mask_mae": round(mask_mae, 4),
            "outside_mae": round(outside_mae, 6),
            "mask_recall": round(mask_recall, 5),
        }
    return results


def gen_detect() -> dict[str, dict[str, float]]:
    from tofu.layers import cicerone, scene
    from eval_detect import evaluate as detect_evaluate, _font_registry

    results = {}
    for fixture in _fixture_names():
        image_path = FIXTURES_DIR / f"{fixture}.png"
        gt_path = FIXTURES_DIR / f"{fixture}.gt.json"
        if not image_path.exists():
            continue
        try:
            scene_regions = scene.analyze_regions(str(image_path))
        except Exception:
            scene_regions = []
        manifest = cicerone.detect(
            str(image_path), scene_regions=scene_regions,
            font_registry=_font_registry(),
        )
        metrics = detect_evaluate(image_path, manifest, gt_path, scene_regions)
        entry: dict[str, float] = {}
        if metrics.get("precision") is not None:
            entry["precision"] = float(metrics["precision"])
        if metrics.get("recall") is not None:
            entry["recall"] = float(metrics["recall"])
        if metrics.get("f1") is not None:
            entry["f1"] = float(metrics["f1"])
        entry["mean_norm_ed"] = float(metrics["mean_norm_ed"])
        results[fixture] = entry
        print(f"  detect/{fixture}: {entry}")
    return results


def gen_projects() -> dict[str, dict[str, float]]:
    """Baseline over the REAL project corpus in images/.

    Reuses the harness itself rather than duplicating the run logic, so the
    numbers recorded here are produced by exactly the code the test asserts on.
    """
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from test_projects_regression import _fixture_names as project_fixtures, _project_metrics

    results = {}
    for fixture in project_fixtures():
        try:
            entry = _project_metrics(fixture)
        except Exception as exc:  # a missing image skips in pytest; here it is data
            print(f"  projects/{fixture}: SKIPPED ({type(exc).__name__}: {exc})")
            continue
        results[fixture] = entry
        print(f"  projects/{fixture}: {entry}")
    return results


def gen_savor() -> dict[str, dict[str, float]]:
    from tofu.core.types import AssetInfo, AssetType
    from tofu.layers import cicerone

    results = {}
    for fixture in _fixture_names():
        image_path = FIXTURES_DIR / f"{fixture}.png"
        if not image_path.exists():
            continue
        info = AssetInfo(asset_type=AssetType.IMAGE, frame_count=1, source=str(image_path))
        manifest = cicerone.detect(str(image_path), info)
        applied = sum(1 for i in manifest.instances if (i.ocr_correction or {}).get("verdict") == "applied")
        unresolved = sum(1 for i in manifest.instances if (i.ocr_correction or {}).get("verdict") == "unresolved")
        results[fixture] = {"applied": float(applied), "unresolved": float(unresolved)}
        print(f"  savor/{fixture}: applied={applied} unresolved={unresolved}")
    return results


def gen_tofu() -> dict[str, dict[str, float]]:
    import statistics
    from tofu.core.pipeline import TofuPipeline
    from tofu.layers import cicerone, scene
    from tofu.layers.fonts import FontRegistry, pantry
    from tofu.layers.tofu import ToFU, lang_to_script
    from tofu.utils.manifest_store import _dict_to_manifest

    cache = ROOT / "scripts" / "eval_out" / "tofu_manifests"
    langs = ("en", "de", "ja", "ar", "hi", "th")
    results = {}

    for fixture in _fixture_names():
        image_path = FIXTURES_DIR / f"{fixture}.png"
        if not image_path.exists():
            continue
        cached = cache / f"{fixture}.json"
        if cached.exists():
            manifest = _dict_to_manifest(json.loads(cached.read_text(encoding="utf-8")))
        else:
            try:
                surfaces = scene.analyze_regions(str(image_path))
            except Exception:
                surfaces = []
            manifest = cicerone.detect(str(image_path), scene_regions=surfaces)
            try:
                manifest = scene.analyze(str(image_path), manifest)
            except Exception:
                pass

        font_dir = pantry()
        registry = FontRegistry(font_dir) if font_dir else None
        validator = ToFU(font_library_path=font_dir)

        all_glyph, all_render = [], []
        for lang in langs:
            script = lang_to_script.get(lang)
            font = None
            if registry and script:
                ranked = registry.recommend(script, lang, limit=1)
                if ranked and ranked[0][1] > 0:
                    font = ranked[0][0]
            context = TofuPipeline._tofu_context(font, manifest)
            report = validator.validate(str(image_path), lang, context, manifest)
            if report.glyph_segmentation_score is not None:
                all_glyph.append(report.glyph_segmentation_score)
            if report.render_quality_score is not None:
                all_render.append(report.render_quality_score)

        entry: dict[str, float] = {}
        if all_glyph:
            entry["glyph_median"] = round(statistics.median(all_glyph), 4)
            entry["glyph_min"] = round(min(all_glyph), 4)
        if all_render:
            entry["render_median"] = round(statistics.median(all_render), 4)
            entry["render_min"] = round(min(all_render), 4)
        results[fixture] = entry
        print(f"  tofu/{fixture}: {entry}")
    return results


def gen_render() -> dict[str, dict[str, float]]:
    from tofu.layers import cicerone, cleanse, scribe, verify
    from tofu.layers.fonts import FontRegistry, pantry
    from tofu.utils.manifest_store import _dict_to_manifest

    cache = ROOT / "scripts" / "eval_out"
    results = {}

    for fixture in _fixture_names():
        image_path = FIXTURES_DIR / f"{fixture}.png"
        if not image_path.exists():
            continue
        cached = cache / f"{fixture}.manifest.json"
        if cached.exists():
            manifest = _dict_to_manifest(json.loads(cached.read_text(encoding="utf-8")))
        else:
            from tofu.layers import scene
            try:
                scene_regions = scene.analyze_regions(str(image_path))
            except Exception:
                scene_regions = []
            manifest = cicerone.detect(str(image_path), scene_regions=scene_regions)

        for inst in manifest.instances:
            if inst.text and not inst.target_text:
                inst.target_text = inst.text

        font_dir = pantry()
        registry = FontRegistry(font_dir) if font_dir else None
        targ_lang = manifest.src_lang or "en"

        cleansed = cleanse.erase(image_path, manifest)
        localized = scribe.render(cleansed, manifest, targ_lang, font_registry=registry)
        qa = verify.assess(localized, manifest, str(image_path), cleansed_asset=cleansed)

        entry: dict[str, float] = {}
        if qa.overall_score is not None:
            entry["qa_overall"] = round(float(qa.overall_score), 4)
        ring = qa.metrics.get("ring_ssim", {})
        if ring:
            entry["mean_ring_ssim"] = round(sum(ring.values()) / len(ring), 4)
        ocr_rt = qa.metrics.get("ocr_roundtrip", {})
        if ocr_rt:
            entry["mean_ocr_roundtrip"] = round(sum(ocr_rt.values()) / len(ocr_rt), 4)
        residual = qa.metrics.get("residual_text", {})
        if residual:
            entry["mean_residual_similarity"] = round(sum(residual.values()) / len(residual), 4)
            entry["max_residual_similarity"] = round(max(residual.values()), 4)
        results[fixture] = entry
        print(f"  render/{fixture}: {entry}")
    return results


def gen_font_match() -> dict[str, dict[str, float]]:
    from tofu.layers import font_matching
    from tofu.layers.fonts import FontRegistry, faces_of, pantry
    from tofu.utils.manifest_store import load_manifest

    fixture = "serif-vs-sans"
    image_path = FIXTURES_DIR / f"{fixture}.png"
    gt_path = FIXTURES_DIR / f"{fixture}.gt.json"
    if not image_path.exists():
        return {}

    gt = json.loads(gt_path.read_text(encoding="utf-8"))
    font_dir = pantry()
    if not font_dir:
        return {}
    registry = FontRegistry(font_dir)
    faces = faces_of(registry)
    manifest = load_manifest(image_path)
    if manifest is None:
        return {}

    results: dict[str, dict[str, float]] = {}
    metrics: dict[str, float] = {}
    for i, inst in enumerate(manifest.instances):
        style = gt["regions"][i].get("style", {}) if i < len(gt["regions"]) else {}
        gt_font_file = style.get("font_file")
        if not gt_font_file:
            continue
        try:
            ranked = font_matching.match(inst, registry, faces, manifest, str(image_path))
        except Exception:
            continue
        for idx, (face, score) in enumerate(ranked, 1):
            if face.file_path and gt_font_file in face.file_path:
                metrics[f"rank_{inst.id}"] = float(idx)
                break
    results[fixture] = metrics
    print(f"  font_match/{fixture}: {metrics}")
    return results


def gen_verification() -> dict[str, dict[str, float]]:
    from eval_verification_corpus import load_spec, run_case

    spec_path = ROOT / "scripts" / "verification_corpus.json"
    if not spec_path.exists():
        return {}
    spec = load_spec(spec_path)
    output_dir = ROOT / "scripts" / "eval_out" / "verification-baseline"
    output_dir.mkdir(parents=True, exist_ok=True)

    total = 0
    matched = 0
    per_wave: dict[str, int] = {}
    per_wave_matched: dict[str, int] = {}

    for case in spec["cases"]:
        wave = case.get("wave", "default")
        result = run_case(case, output_dir, real_ocr=False)
        total += 1
        per_wave[wave] = per_wave.get(wave, 0) + 1
        if result["matched"]:
            matched += 1
            per_wave_matched[wave] = per_wave_matched.get(wave, 0) + 1

    metrics: dict[str, float] = {"pass_rate": round(matched / total, 4) if total else 0.0}
    for wave in per_wave:
        w_rate = per_wave_matched.get(wave, 0) / per_wave[wave] if per_wave[wave] else 0.0
        metrics[f"pass_rate_{wave}"] = round(w_rate, 4)
    print(f"  verification_corpus/overall: {metrics}")
    return {"overall": metrics}


def gen_memory() -> dict[str, dict[str, float]]:
    import io
    from PIL import Image
    from tofu.core.types import AssetInfo, AssetType, BBox, QAReport, TextManifest
    from tofu.layers import cicerone, memory
    from eval_memory import iou

    SCALE = 0.85
    JPEG_Q = 80
    IOU_MATCH = 0.3
    results = {}

    for fixture in _fixture_names():
        fx = FIXTURES_DIR / f"{fixture}.png"
        if not fx.exists():
            continue
        info_a = AssetInfo(asset_type=AssetType.IMAGE, frame_count=1, source=str(fx))
        manifest_a = cicerone.detect(str(fx), info_a)
        instances_with_text = [i for i in manifest_a.instances if i.text]
        fixture_gt = []
        for inst in instances_with_text:
            inst.target_text = f"T::{fx.stem}::{inst.id}"
            fixture_gt.append((inst.bounding_box, inst.target_text))
        if instances_with_text:
            qa = QAReport(
                overall_score=1.0,
                per_asset_instance_score={manifest_a.asset_id: {i.id: 1.0 for i in instances_with_text}},
            )
            candidates = memory.update(manifest_a, "es", None, str(fx), qa, qa_threshold=0.5)
        else:
            candidates = []

        img = Image.open(fx).convert("RGB")
        w, h = img.size
        resized = img.resize((max(1, int(w * SCALE)), max(1, int(h * SCALE))))
        buf = io.BytesIO()
        resized.save(buf, format="JPEG", quality=JPEG_Q)
        buf.seek(0)
        perturbed = Image.open(buf).convert("RGB")
        tmp_path = ROOT / "scripts" / "eval_out" / f"_memeval_{fx.stem}.jpg"
        tmp_path.parent.mkdir(parents=True, exist_ok=True)
        perturbed.save(tmp_path)

        info_b = AssetInfo(asset_type=AssetType.IMAGE, frame_count=1, source=str(tmp_path))
        manifest_b = cicerone.detect(str(tmp_path), info_b)

        predicted = 0
        correct = 0
        for inst in manifest_b.instances:
            if not inst.text:
                continue
            b = inst.bounding_box
            orig_box = BBox(
                x=int(b.x / SCALE), y=int(b.y / SCALE),
                width=int(b.width / SCALE), height=int(b.height / SCALE),
            )
            best_iou, best_target = 0.0, None
            for gt_box, gt_target in fixture_gt:
                score = iou(orig_box, gt_box)
                if score > best_iou:
                    best_iou, best_target = score, gt_target
            if best_iou < IOU_MATCH:
                continue
            single = TextManifest(
                asset_id="asset-b", total_regions=1, instances=[inst],
                src_lang="en", targ_lang="es",
            )
            matches = memory.lookup(single, str(tmp_path), "es", candidates)
            if inst.id in matches:
                predicted += 1
                if matches[inst.id]["target_text"] == best_target:
                    correct += 1

        precision = correct / predicted if predicted else None
        entry: dict[str, float] = {}
        if precision is not None:
            entry["precision"] = round(precision, 4)
        entry["predicted"] = float(predicted)
        entry["correct"] = float(correct)
        results[fixture] = entry
        print(f"  memory/{fixture}: {entry}")
    return results


HARNESS_MAP = {
    "cleanse_providers": gen_cleanse,
    "detect": gen_detect,
    "projects": gen_projects,
    "savor": gen_savor,
    "tofu": gen_tofu,
    "render": gen_render,
    "font_match": gen_font_match,
    "verification_corpus": gen_verification,
    "memory": gen_memory,
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--harness", default=None, help="only generate for one harness")
    args = ap.parse_args()

    harnesses = [args.harness] if args.harness else list(HARNESS_MAP.keys())
    for name in harnesses:
        print(f"\n== generating baseline: {name} ==")
        gen_fn = HARNESS_MAP[name]
        try:
            results = gen_fn()
            update_baseline(name, results)
            print(f"  wrote {len(results)} fixture(s) to baseline")
        except Exception as e:
            print(f"  ERROR: {e}")
    print("\ndone.")


if __name__ == "__main__":
    main()
