## 🍢 eval_render — cleanse/scribe/verify quality harness
## vieuxtiful
"""
Measures render-pipeline quality (scene enrichment → cleanse → scribe →
verify) on a real image and writes:
  - the cleansed intermediate and the localized output PNGs
  - a triptych overlay (source | cleansed | localized) with per-region
    boxes colored by score
  - a JSON report: per-region OCR round-trip, ring-SSIM, ink presence,
    RESIDUAL SOURCE TEXT (was the erase actually complete?), and timing

The residual-text metric lives here first (Phase 0); it graduates into
verify.assess in Phase 5.

Run before/after any cleanse/scribe tuning; compare the reports.

usage (from repo root):
  .venv/Scripts/python scripts/eval_render.py [image] [--tag label]
      [--manifest path.json]     reuse a saved manifest (skips detection)
      [--mode identity|pseudo|file]  how target_text is filled (default identity)
      [--translations path.json] {region_id: target_text} for --mode file
      [--targ-lang code]         target language for scribe/verify (default: src)
      [--redetect]               ignore the cached manifest and re-run detection
defaults to images/gemini-street.png; outputs to scripts/eval_out/.

Detection is slow (~minutes CPU); the first run caches the manifest as
eval_out/{stem}.manifest.json and later runs reuse it, so cleanse/scribe
iteration stays fast.
"""

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

# CJK-safe console output on Windows (cp1252 default chokes on kana/han)
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from tofu.layers import cicerone, scene, cleanse, scribe, verify  # noqa: E402
from tofu.utils.manifest_store import _manifest_to_dict, _dict_to_manifest  # noqa: E402


# pseudo-localization accent map (latin scripts): same length, same
# visual bulk, instantly recognizable as machine-filled
_PSEUDO = str.maketrans(
    "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ",
    "àƀçđéƒĝĥîĵķĺɱñöþǫŕšŧûṽŵẋýžÀƁÇĐÉƑĜĤÎĴĶĹṀÑÖÞǪŔŠŦÛṼŴẊÝŽ",
)


def fill_targets(manifest, mode: str, translations: dict) -> None:
    """populate target_text per instance according to the chosen mode.

    identity: target = source text (tests erase + re-render fidelity with
    zero translation noise — the OCR round-trip judge reads the same
    language it detected). pseudo: accent-mapped source. file: explicit
    {region_id: text} map; regions absent from the map stay untranslated.
    """
    for inst in manifest.instances:
        if inst.dnt or not inst.text:
            continue
        if mode == "identity":
            inst.target_text = inst.text
        elif mode == "pseudo":
            inst.target_text = inst.text.translate(_PSEUDO)
        elif mode == "file":
            if inst.id in translations:
                inst.target_text = translations[inst.id]


def residual_source_text(cleansed_np, manifest) -> dict:
    """OCR each erased region in the CLEANSED image and compare against
    the source text. similarity ~0 = erase complete; high = text survived.

    the recognition-based erasure protocol from the scene-text-removal
    literature (EnsNet, Zhang et al. 2019; EraseNet, Liu et al. 2020):
    an eraser is judged by whether a recognizer still reads the original.
    """
    out = {}
    h, w = cleansed_np.shape[:2]
    for inst in manifest.instances:
        if inst.dnt or not inst.text or inst.bounding_box is None:
            continue
        b = inst.bounding_box
        x0, y0 = max(0, b.x - 4), max(0, b.y - 4)
        x1, y1 = min(w, b.x + b.width + 4), min(h, b.y + b.height + 4)
        if x1 - x0 < 4 or y1 - y0 < 4:
            continue
        reader = verify._get_reader(
            inst.language or inst.detected_language or manifest.src_lang or "en"
        )
        if reader is None:
            continue
        try:
            results = reader.readtext(cleansed_np[y0:y1, x0:x1])
        except Exception:
            continue
        recognized = " ".join(r[1] for r in results)
        out[inst.id] = round(verify._text_similarity(inst.text, recognized), 4)
    return out


def triptych(source_path: Path, cleansed, localized, manifest,
             per_region: dict, residual: dict, out_path: Path) -> None:
    """side-by-side (source | cleansed | localized) with score-colored
    boxes on the localized panel and residual flags on the cleansed one."""
    from PIL import Image, ImageDraw, ImageFont
    src = Image.open(source_path).convert("RGB")
    cle = cleansed.convert("RGB")
    loc = localized.convert("RGB")
    w, h = src.size
    canvas = Image.new("RGB", (w * 3 + 16 * 2, h + 28), (24, 24, 27))
    for i, (img, label) in enumerate([(src, "source"), (cle, "cleansed"), (loc, "localized")]):
        canvas.paste(img, (i * (w + 16), 28))
    draw = ImageDraw.Draw(canvas)
    try:
        font = ImageFont.truetype("malgun.ttf", 16)
    except Exception:
        font = ImageFont.load_default()
    for i, label in enumerate(["source", "cleansed", "localized"]):
        draw.text((i * (w + 16) + 4, 5), label, fill=(212, 212, 216), font=font)

    def score_color(s: float):
        return (34, 197, 94) if s >= 0.8 else (245, 158, 11) if s >= 0.6 else (239, 68, 68)

    x_cle, x_loc = w + 16, 2 * (w + 16)
    for inst in manifest.instances:
        b = inst.bounding_box
        if b is None:
            continue
        res = residual.get(inst.id)
        if res is not None:
            # residual similarity: LOW is good (erase complete)
            color = (34, 197, 94) if res <= 0.3 else (245, 158, 11) if res <= 0.6 else (239, 68, 68)
            draw.rectangle([x_cle + b.x, 28 + b.y, x_cle + b.x + b.width,
                            28 + b.y + b.height], outline=color, width=2)
            draw.text((x_cle + b.x + 2, 28 + max(0, b.y - 18)),
                      f"{inst.id} res {res:.2f}", fill=color, font=font)
        score = per_region.get(inst.id)
        if score is not None:
            color = score_color(score)
            draw.rectangle([x_loc + b.x, 28 + b.y, x_loc + b.x + b.width,
                            28 + b.y + b.height], outline=color, width=2)
            draw.text((x_loc + b.x + 2, 28 + max(0, b.y - 18)),
                      f"{inst.id} {score:.2f}", fill=color, font=font)
    canvas.save(out_path)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("image", nargs="?", default=str(ROOT / "images" / "gemini-street.png"))
    ap.add_argument("--tag", default="run", help="label for output filenames")
    ap.add_argument("--manifest", default=None, help="manifest JSON to reuse (skips detection)")
    ap.add_argument("--mode", default="identity", choices=("identity", "pseudo", "file"),
                    help="how target_text is filled")
    ap.add_argument("--translations", default=None, help="{region_id: text} JSON for --mode file")
    ap.add_argument("--targ-lang", default=None, help="target language (default: inferred source)")
    ap.add_argument("--redetect", action="store_true", help="ignore the cached manifest")
    ap.add_argument("--engine", default="easyocr", choices=("easyocr", "paddleocr"))
    ap.add_argument("--fonts", default=None,
                    help="font library dir for FontRegistry-based weight/"
                         "italic face resolution in scribe (e.g. "
                         "C:/Windows/Fonts). slow to discover; opt-in only "
                         "since most iteration doesn't need it.")
    args = ap.parse_args()

    import os
    os.environ["OCR_ENGINE"] = args.engine

    image_path = Path(args.image)
    out_dir = ROOT / "scripts" / "eval_out"
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = image_path.stem
    cache_path = out_dir / f"{stem}.manifest.json"

    # -- manifest: explicit file > cache > fresh detection --------------------
    timing = {}
    if args.manifest:
        manifest = _dict_to_manifest(json.loads(Path(args.manifest).read_text(encoding="utf-8")))
        manifest_src = args.manifest
    elif cache_path.exists() and not args.redetect:
        manifest = _dict_to_manifest(json.loads(cache_path.read_text(encoding="utf-8")))
        manifest_src = f"{cache_path} (cached)"
    else:
        t0 = time.time()
        scene_regions = scene.analyze_regions(str(image_path))
        manifest = cicerone.detect(str(image_path), scene_regions=scene_regions)
        timing["detect"] = round(time.time() - t0, 1)
        cache_path.write_text(
            json.dumps(_manifest_to_dict(manifest), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        manifest_src = "fresh detection"

    translations = {}
    if args.translations:
        translations = json.loads(Path(args.translations).read_text(encoding="utf-8"))
    fill_targets(manifest, args.mode, translations)
    targ_lang = args.targ_lang or manifest.src_lang or "en"
    manifest.targ_lang = targ_lang

    # -- scene enrichment -----------------------------------------------------
    t0 = time.time()
    manifest = scene.analyze(str(image_path), manifest)
    timing["scene_enrich"] = round(time.time() - t0, 1)

    # -- cleanse --------------------------------------------------------------
    t0 = time.time()
    cleansed = cleanse.erase(str(image_path), manifest)
    timing["cleanse"] = round(time.time() - t0, 1)

    # -- scribe ---------------------------------------------------------------
    font_registry = None
    if args.fonts:
        from tofu.layers.fonts import FontRegistry
        t0 = time.time()
        font_registry = FontRegistry(args.fonts)
        timing["font_discovery"] = round(time.time() - t0, 1)
    t0 = time.time()
    localized = scribe.render(cleansed, manifest, targ_lang, font_registry=font_registry)
    timing["scribe"] = round(time.time() - t0, 1)

    # -- verify + residual ----------------------------------------------------
    import numpy as np
    t0 = time.time()
    qa = verify.assess(localized, manifest, str(image_path))
    timing["verify"] = round(time.time() - t0, 1)
    t0 = time.time()
    residual = residual_source_text(np.asarray(cleansed.convert("RGB")), manifest)
    timing["residual"] = round(time.time() - t0, 1)

    per_region = qa.per_asset_instance_score.get(manifest.asset_id, {})
    rendered = [i for i in manifest.instances if i.target_text and not i.dnt]
    residual_vals = list(residual.values())

    report = {
        "image": str(image_path),
        "tag": args.tag,
        "mode": args.mode,
        "targ_lang": targ_lang,
        "manifest_source": manifest_src,
        "region_count": manifest.total_regions,
        "rendered_regions": len(rendered),
        "qa_overall": round(qa.overall_score, 4) if qa.overall_score is not None else None,
        "qa_scorer": qa.metrics.get("scorer"),
        "mean_ring_ssim": (
            round(sum(qa.metrics["ring_ssim"].values()) / len(qa.metrics["ring_ssim"]), 4)
            if qa.metrics.get("ring_ssim") else None
        ),
        "mean_ocr_roundtrip": (
            round(sum(qa.metrics["ocr_roundtrip"].values()) / len(qa.metrics["ocr_roundtrip"]), 4)
            if qa.metrics.get("ocr_roundtrip") else None
        ),
        "mean_residual_similarity": (
            round(sum(residual_vals) / len(residual_vals), 4) if residual_vals else None
        ),
        "max_residual_similarity": (
            round(max(residual_vals), 4) if residual_vals else None
        ),
        "timing_s": timing,
        "regions": [
            {
                "id": i.id,
                "text": i.text,
                "target_text": i.target_text,
                "lang": i.detected_language,
                "bbox": [i.bounding_box.x, i.bounding_box.y,
                         i.bounding_box.width, i.bounding_box.height],
                "score": per_region.get(i.id),
                "ocr_roundtrip": qa.metrics.get("ocr_roundtrip", {}).get(i.id),
                "ring_ssim": qa.metrics.get("ring_ssim", {}).get(i.id),
                "ink_presence": qa.metrics.get("ink_presence", {}).get(i.id),
                "residual_similarity": residual.get(i.id),
            }
            for i in manifest.instances
        ],
        "recommendations": qa.recommendations,
    }

    report_path = out_dir / f"{stem}-{args.tag}.render.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    cleansed_path = out_dir / f"{stem}-{args.tag}.cleansed.png"
    localized_path = out_dir / f"{stem}-{args.tag}.localized.png"
    cleansed.convert("RGB").save(cleansed_path)
    localized.convert("RGB").save(localized_path)
    overlay_path = out_dir / f"{stem}-{args.tag}.render.png"
    triptych(image_path, cleansed, localized, manifest, per_region, residual, overlay_path)

    print(f"== {stem} [{args.tag}] mode={args.mode} → {targ_lang} ==")
    print(f"manifest: {manifest_src}")
    print(f"regions: {report['region_count']} ({report['rendered_regions']} rendered)")
    print(f"QA overall: {report['qa_overall']}  (scorer: {report['qa_scorer']})")
    print(f"mean ring-SSIM: {report['mean_ring_ssim']}  |  mean OCR round-trip: {report['mean_ocr_roundtrip']}")
    print(f"residual source text: mean {report['mean_residual_similarity']}  max {report['max_residual_similarity']}"
          "  (lower = cleaner erase)")
    print(f"timing: {timing}")
    print(f"report:  {report_path}")
    print(f"overlay: {overlay_path}")


if __name__ == "__main__":
    main()
