## 🍢 eval_tofu — pre-flight score distribution harness
## vieuxtiful
"""
Measures what ToFU's own scores actually DO across the project's fixtures.

Two of ToFU's warning gates were set by hand and never checked against a
real distribution:

  - ToFU_003 (glyph segmentation) compares a CONSTANT 0.65 for complex
    scripts against a `< 0.7` gate, so it fires on every ja/zh/ko/ar/hi/th
    run regardless of the asset. A warning that is always on carries no
    information.
  - ToFU_004 (render quality) compares `_deterministic_render_score`
    against `< 0.6`, but with full script coverage that score sits at
    ~0.94 and with partial coverage at ~0.87. Anything low enough to trip
    the gate has already failed ToFU_001 as an ERROR, so the check never
    fires alone.

This harness dumps both scores — plus the per-asset evidence that a
better score could be derived FROM (region height, measured stroke
ratio) — across every fixture and a spread of target languages, so the
gates can be set from the observed range instead of a guess.

Manifests are detected once and cached; re-runs are near-instant, which
is what makes a before/after calibration sweep practical.

usage (from repo root):
  .venv/Scripts/python scripts/eval_tofu.py --tag calibration-baseline
  .venv/Scripts/python scripts/eval_tofu.py --tag calibration-applied
  .venv/Scripts/python scripts/eval_tofu.py --refresh          # re-detect
"""

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tofu.layers import cicerone, scene  # noqa: E402
from tofu.layers.fonts import FontRegistry, pantry  # noqa: E402
from tofu.layers.tofu import ToFU, lang_to_script  # noqa: E402
from tofu.utils.manifest_store import _dict_to_manifest, _manifest_to_dict  # noqa: E402

OUT_DIR = ROOT / "scripts" / "eval_out"
CACHE_DIR = OUT_DIR / "tofu_manifests"

# the real street scenes (dense CJK + latin, with ground truth) plus the
# synthetic single-surface fixtures the unit suite renders against
DEFAULT_ASSETS = (
    ROOT / "images" / "japan-street.jpeg",
    ROOT / "images" / "china-street.png",
    ROOT / "images" / "gemini-street.png",
    ROOT / "tests" / "fixtures" / "cjk-vertical.png",
    ROOT / "tests" / "fixtures" / "expansion-en.png",
    ROOT / "tests" / "fixtures" / "flat-sign.png",
    ROOT / "tests" / "fixtures" / "gradient-banner.png",
    ROOT / "tests" / "fixtures" / "stylized-italic.png",
    ROOT / "tests" / "fixtures" / "textured-wall.png",
)

# a spread that exercises every support tier in the static map and every
# expansion direction: latin FULL, cyrillic/greek FULL, CJK FULL,
# arabic/devanagari/thai PARTIAL, and both growth (de) and shrink (ja)
DEFAULT_LANGS = (
    "en", "de", "fr", "ru", "el", "ja", "zh-cn", "ko", "ar", "he", "hi", "th",
)


def _plate(path: Path, registry, refresh: bool):
    """Serve up a scene-enriched manifest for an asset, detecting only when
    the cache is cold.

    Enrichment matters here: the evidence a better glyph-segmentation
    score would be derived from (`characteristics.size`, the measured
    `stroke_ratio`) is written by scene.analyze, not by detection.
    """
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cached = CACHE_DIR / f"{path.stem}.json"
    if cached.exists() and not refresh:
        return _dict_to_manifest(json.loads(cached.read_text(encoding="utf-8"))), 0.0

    t0 = time.time()
    try:
        surfaces = scene.analyze_regions(str(path))
    except Exception:
        surfaces = []
    manifest = cicerone.detect(
        str(path), scene_regions=surfaces, font_registry=registry
    )
    try:
        manifest = scene.analyze(str(path), manifest)
    except Exception:
        pass  # unenriched manifest still exercises the script/coverage paths
    elapsed = time.time() - t0

    cached.write_text(
        json.dumps(_manifest_to_dict(manifest), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return manifest, elapsed


def _mise_en_place(manifest) -> dict:
    """Per-asset source-side evidence, laid out before any scoring runs.

    These are the measurable quantities a segmentation-feasibility score
    SHOULD be derived from — small text and thin strokes are what
    actually break binarization, not the target language's script.
    """
    heights, sizes, strokes = [], [], []
    for inst in manifest.instances:
        if inst.bounding_box is not None:
            heights.append(inst.bounding_box.height)
        ch = inst.characteristics
        if ch is None:
            continue
        if ch.size:
            sizes.append(ch.size)
        ratio = (ch.positioning or {}).get("stroke_ratio")
        if isinstance(ratio, (int, float)) and ratio > 0:
            strokes.append(float(ratio))

    def _mid(values):
        return round(statistics.median(values), 4) if values else None

    return {
        "regions": len(manifest.instances),
        "src_lang": manifest.src_lang,
        "median_region_height_px": _mid(heights),
        "min_region_height_px": min(heights) if heights else None,
        "median_font_px": _mid(sizes),
        "median_stroke_ratio": _mid(strokes),
        "enriched_regions": len(sizes),
    }


def _taste(validator, path: Path, manifest, lang: str, font: str | None) -> dict:
    """Run one preflight exactly the way the pipeline runs it.

    The context is built by TofuPipeline's own helper rather than a copy
    of it, so this harness cannot quietly drift from the thing it is
    supposed to be measuring. That matters: an asset-level call with no
    font size in context always lands on the render score's
    no-information default, which is precisely how ToFU_004 came to look
    unreachable.
    """
    from tofu.core.pipeline import TofuPipeline
    context = TofuPipeline._tofu_context(font, manifest)
    report = validator.validate(str(path), lang, context, manifest)
    fits = list(report.expansion_fit.values())
    codes = sorted({i.code for i in report.issues})
    return {
        "lang": lang,
        "passed": report.passed,
        "codes": codes,
        "glyph_segmentation_score": report.glyph_segmentation_score,
        "render_quality_score": report.render_quality_score,
        "script_support": {k: v.value for k, v in report.scrpt_spprt.items()},
        "expansion": {
            "n": len(fits),
            "min": round(min(fits), 3) if fits else None,
            "median": round(statistics.median(fits), 3) if fits else None,
            "max": round(max(fits), 3) if fits else None,
        },
    }


def _per_region_render_scores(validator, path: Path, manifest, lang: str,
                              font: str | None) -> list:
    """Render-quality scores under REAL per-region context.

    `_deterministic_render_score` weights font size and effect count, but
    the asset-level call supplies neither, so it always lands on the same
    optimistic default. This mirrors the per-region context the server
    already builds (server/main.py's region re-validation) to find the
    score's true low end.
    """
    scores = []
    for inst in manifest.instances:
        ctx: dict = {"font": font} if font else {}
        if inst.characteristics and inst.characteristics.size:
            ctx["font_px"] = inst.characteristics.size
        sp = inst.style_profile
        effects = []
        if sp:
            if sp.shadow:
                effects.append("shadow")
            if sp.stroke_width:
                effects.append("stroke")
            if sp.italic:
                effects.append("italic")
        if effects:
            ctx["effects"] = effects
        report = validator.validate(str(path), lang, ctx or None)
        if report.render_quality_score is not None:
            scores.append(report.render_quality_score)
    return scores


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="run", help="label for the output filename")
    ap.add_argument("--assets", nargs="*", default=None, help="override the fixture set")
    ap.add_argument("--langs", nargs="*", default=None, help="override the target languages")
    ap.add_argument("--refresh", action="store_true", help="re-detect instead of using cached manifests")
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    font_dir = pantry()
    registry = FontRegistry(font_dir) if font_dir else None
    validator = ToFU(font_library_path=font_dir)
    langs = tuple(args.langs) if args.langs else DEFAULT_LANGS
    assets = [Path(a) for a in args.assets] if args.assets else list(DEFAULT_ASSETS)

    results = []
    all_glyph, all_render, all_region_render = [], [], []

    for path in assets:
        if not path.exists():
            print(f"  skip (missing): {path}")
            continue
        manifest, detect_s = _plate(path, registry, args.refresh)
        evidence = _mise_en_place(manifest)
        print(f"\n{path.name}: {evidence['regions']} region(s), "
              f"src={evidence['src_lang']}, "
              f"median h={evidence['median_region_height_px']}px"
              + (f", detected in {detect_s:.0f}s" if detect_s else " (cached)"))

        per_lang = []
        for lang in langs:
            script = lang_to_script.get(lang)
            font = None
            if registry and script:
                ranked = registry.recommend(script, lang, limit=1)
                if ranked and ranked[0][1] > 0:
                    font = ranked[0][0]
            row = _taste(validator, path, manifest, lang, font)
            row["font"] = Path(font).name if font else None
            row["region_render_scores"] = _per_region_render_scores(
                validator, path, manifest, lang, font
            )
            per_lang.append(row)

            if row["glyph_segmentation_score"] is not None:
                all_glyph.append(row["glyph_segmentation_score"])
            if row["render_quality_score"] is not None:
                all_render.append(row["render_quality_score"])
            all_region_render.extend(row["region_render_scores"])

            rr = row["region_render_scores"]
            print(f"  {lang:<6} glyph={row['glyph_segmentation_score']} "
                  f"render={row['render_quality_score']} "
                  f"region_render_min={min(rr) if rr else None} "
                  f"codes={','.join(row['codes']) or '-'}")

        results.append({
            "asset": str(path),
            "evidence": evidence,
            "per_lang": per_lang,
        })

    def _spread(values, label):
        if not values:
            return {"label": label, "n": 0}
        ordered = sorted(values)
        return {
            "label": label,
            "n": len(ordered),
            "min": round(ordered[0], 4),
            "p10": round(ordered[max(0, int(len(ordered) * 0.10) - 1)], 4),
            "median": round(statistics.median(ordered), 4),
            "p90": round(ordered[min(len(ordered) - 1, int(len(ordered) * 0.90))], 4),
            "max": round(ordered[-1], 4),
            "distinct": sorted({round(v, 4) for v in ordered})[:12],
        }

    distribution = [
        _spread(all_glyph, "glyph_segmentation_score"),
        _spread(all_render, "render_quality_score (pipeline context)"),
        _spread(all_region_render, "render_quality_score (per-region context)"),
    ]

    report = {
        "tag": args.tag,
        "font_dir": font_dir,
        "fonts_loaded": len(registry.fonts) if registry else 0,
        "langs": list(langs),
        "distribution": distribution,
        "assets": results,
    }
    out = OUT_DIR / f"tofu-{args.tag}.json"
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n--- score distributions ---")
    for row in distribution:
        if not row["n"]:
            continue
        print(f"{row['label']}: n={row['n']} min={row['min']} p10={row['p10']} "
              f"median={row['median']} p90={row['p90']} max={row['max']}")
        print(f"    distinct values: {row['distinct']}")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
