## 🍢 eval_font_match — visual font retrieval accuracy harness
## vieuxtiful
"""
Measures whether font_matching actually names the face a sign was set in.

Every other scoring constant in the project was eventually checked against
a real distribution; this one never was. `_visual_score`'s weights, the
class-mismatch penalty, and the evidence floors all decide which typeface
a localiser is offered, and until this harness existed there was no way to
tell whether a change to them helped or hurt.

Two kinds of asset, because they answer different questions:

  serif-vs-sans fixture   ground truth is EXACT — the generator recorded
                          which face rendered each region, so "did the
                          matcher return the right family, and at what
                          rank" is directly answerable.

  real photographs        no ground truth, but the la-rue-sans-nom plaque
                          is a single sign in a single face, so its two
                          regions must AGREE. Disagreement is a defect
                          whether or not the answer is nameable.

The headline number is the rank of the correct face, not its score: scores
move whenever weights move, ranks only move when the ordering genuinely
changes.

usage (from repo root):
  .venv/Scripts/python scripts/eval_font_match.py --tag baseline
  .venv/Scripts/python scripts/eval_font_match.py --tag serif-features
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tofu.core.types import BBox, InstText, TextManifest  # noqa: E402
from tofu.layers import basil, font_matching, scene  # noqa: E402
from tofu.layers.fonts import FontRegistry, faces_of, pantry  # noqa: E402
from tofu.utils.manifest_store import load_manifest  # noqa: E402

OUT_DIR = ROOT / "scripts" / "eval_out"
FIXTURES = ROOT / "tests" / "fixtures"


def _family_of(path: str, registry) -> str:
    face = faces_of(registry).get(path)
    return face.family if face else Path(path).stem


def _rank_of(family: str, ranked: list) -> int | None:
    """1-based rank of a family among scored candidates, or None."""
    for index, candidate in enumerate(ranked):
        if (candidate.get("family") or "").lower() == family.lower():
            return index + 1
    return None


def eval_fixture(registry, name: str = "serif-vs-sans") -> dict:
    """Exact ground truth: we rendered these, so we know the answer."""
    image_path = FIXTURES / f"{name}.png"
    gt_path = FIXTURES / f"{name}.gt.json"
    if not image_path.exists() or not gt_path.exists():
        return {"asset": name, "skipped": "fixture not generated; run scripts/make_fixtures.py"}

    img = np.asarray(Image.open(image_path).convert("RGB"))
    regions = json.loads(gt_path.read_text(encoding="utf-8"))["regions"]
    rows = []
    correct_class = 0
    for index, region in enumerate(regions):
        x, y, w, h = region["bbox"]
        pad = 4  # capture boxes are padded past the ink; match that here
        inst = InstText(
            id=f"r{index + 1}",
            bounding_box=BBox(x - pad, y - pad, w + 2 * pad, h + 2 * pad),
            text=region["text"],
        )
        result = font_matching.local_match(img, inst, registry)
        style = region.get("style") or {}
        expected_file = style.get("font_file")
        expected_family = _family_of(
            str(Path(pantry() or "") / expected_file) if expected_file else "", registry
        )
        if result is None:
            rows.append({"region": inst.id, "text": region["text"], "matched": False})
            continue
        candidates = result["candidates"]
        top = candidates[0]
        # Was the ANSWER at least the right kind of face?  A serif sign set
        # in some other serif is a different quality of error from a serif
        # sign set in a grotesk.
        top_is_serif = _looks_serif(top.get("family", ""), registry)
        expected_serif = bool(style.get("serif"))
        if top_is_serif is not None and top_is_serif == expected_serif:
            correct_class += 1
        rows.append({
            "region": inst.id,
            "text": region["text"],
            "expected_family": expected_family,
            "expected_serif": expected_serif,
            "top": top.get("family"),
            "top_score": top.get("score"),
            "top_is_serif": top_is_serif,
            "expected_rank": _rank_of(expected_family, candidates),
            "confidence": result.get("confidence"),
            "status": result.get("status"),
            "candidates": [(c.get("family"), c.get("score")) for c in candidates],
        })
    scored = [r for r in rows if r.get("top") and r.get("top_is_serif") is not None]
    return {
        "asset": name,
        "regions": len(rows),
        "class_correct": correct_class,
        "class_accuracy": round(correct_class / max(1, len(scored)), 3),
        "class_scorable": len(scored),
        "detail": rows,
    }


# Serif/sans oracle for scoring THIS harness, read from each face's own
# PANOSE classification (PANOSE 1.0, OS/2 table byte 1 "Serif Style"): 11
# Normal Sans, 12 Obtuse Sans and 13 Perpendicular Sans are the sans
# values; 2-10 are the serif variants (Cove through Triangle).  Using the
# foundry's own declaration beats a hand-kept family list, which scored
# Modern No. 20 (a Didone) and Berlin Sans FB as unknown purely because
# nobody had typed them in.
#
# This is a TEST oracle over the installed library.  font_matching itself
# must never consult it -- its whole job is to reach the verdict from pixels.
_PANOSE_SANS = {11, 12, 13}
_PANOSE_SERIF = set(range(2, 11))
_SERIF_CACHE: dict = {}


def _face_is_serif(font_path: str) -> bool | None:
    if font_path in _SERIF_CACHE:
        return _SERIF_CACHE[font_path]
    verdict = None
    try:
        from fontTools.ttLib import TTCollection, TTFont

        base, _, index = font_path.rpartition("#")
        if base and index.isdigit():
            face = TTCollection(base, lazy=True).fonts[int(index)]
        else:
            face = TTFont(font_path, lazy=True)
        panose = face["OS/2"].panose
        # byte 0 selects the family kind; only Latin Text (2) defines the
        # serif byte the way we are reading it
        if panose.bFamilyType in (0, 2):
            style = panose.bSerifStyle
            if style in _PANOSE_SANS:
                verdict = False
            elif style in _PANOSE_SERIF:
                verdict = True
    except Exception:
        verdict = None
    _SERIF_CACHE[font_path] = verdict
    return verdict


def _looks_serif(family: str, registry=None) -> bool | None:
    if not family or registry is None:
        return None
    for face in faces_of(registry).values():
        if (face.family or "").lower() == family.strip().lower():
            return _face_is_serif(face.font_path)
    return None


def eval_asset(registry, asset_id: str, image: Path, label: str) -> dict:
    """Real photograph: no ground truth, but one sign means one face."""
    manifest = load_manifest(ROOT / "server" / "uploads", asset_id)
    if manifest is None or not manifest.instances:
        return {"asset": label, "skipped": f"no stored manifest for {asset_id}"}
    img = np.asarray(Image.open(image).convert("RGB"))
    started = time.perf_counter()
    font_matching.identify_manifest_fonts(img, manifest, registry)
    elapsed = time.perf_counter() - started

    rows, recommended = [], set()
    for inst in manifest.instances:
        evidence = inst.font_match or {}
        substitute = evidence.get("recommended_substitute") or {}
        own = evidence.get("region_substitute") or substitute
        if substitute.get("font_path"):
            recommended.add(substitute["font_path"])
        rows.append({
            "region": inst.id,
            "text": inst.text,
            "own_top": own.get("family"),
            "recommended": substitute.get("family"),
            "recommended_is_serif": _looks_serif(substitute.get("family", ""), registry),
            "cohort": (evidence.get("cohort") or {}).get("region_ids"),
            "agreement": (evidence.get("cohort") or {}).get("agreement"),
            "candidates": [(c.get("family"), c.get("score")) for c in (evidence.get("candidates") or [])],
        })
    return {
        "asset": label,
        "asset_id": asset_id,
        "regions": len(rows),
        "seconds": round(elapsed, 2),
        "seconds_per_region": round(elapsed / max(1, len(rows)), 2),
        "distinct_recommended_faces": len(recommended),
        "bouquets": basil.bouquet(manifest),
        "detail": rows,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="run", help="label for the output filename")
    ap.add_argument("--skip-assets", action="store_true", help="fixture only, no photographs")
    ap.add_argument("--silhouette-only", action="store_true",
                    help="score with Dice/Chamfer/projection/aspect alone, as the kernel "
                         "did before the typographic terms — the before half of a sweep")
    args = ap.parse_args()

    if args.silhouette_only:
        original = font_matching._visual_score

        def silhouette(source, candidate, **kwargs):
            return original(source, candidate, typographic=False)

        font_matching._visual_score = silhouette
        print("scoring with silhouette terms only")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    font_dir = pantry()
    registry = FontRegistry(font_dir) if font_dir else None
    if registry is None:
        print("no font directory; nothing to match against")
        return
    print(f"font dir: {font_dir}  ({len(registry.fonts)} faces)")

    results = [eval_fixture(registry)]
    if not args.skip_assets:
        for asset_id, image, label in (
            ("256f0913d4ad", ROOT / "server" / "uploads" / "256f0913d4ad.jpg", "la-rue-sans-nom"),
            ("bc40c1eaebfd", ROOT / "server" / "uploads" / "bc40c1eaebfd.png", "gemini-street"),
        ):
            if image.exists():
                results.append(eval_asset(registry, asset_id, image, label))

    report = {"tag": args.tag, "font_dir": font_dir,
              "fonts_loaded": len(registry.fonts), "results": results}
    out = OUT_DIR / f"font-match-{args.tag}.json"
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n--- serif / sans discrimination (exact ground truth) ---")
    fixture = results[0]
    if fixture.get("skipped"):
        print(f"  skipped: {fixture['skipped']}")
    else:
        print(f"  class accuracy: {fixture['class_accuracy']:.0%} "
              f"({fixture['class_correct']}/{fixture['class_scorable']} scorable "
              f"of {fixture['regions']} regions)")
        for row in fixture["detail"]:
            if not row.get("top"):
                continue
            # A face PANOSE does not classify is unknown, not wrong -- and
            # counting it as wrong would make two runs' accuracies
            # incomparable, since which faces win changes between them.
            if row["top_is_serif"] is None:
                mark = "?   "
            elif row["top_is_serif"] == row["expected_serif"]:
                mark = "ok  "
            else:
                mark = "MISS"
            print(f"  {mark} {row['text']!r:14} expected {'serif' if row['expected_serif'] else 'sans ':5}"
                  f" -> {row['top']:<26} ({row['top_score']:.3f})"
                  f"  correct-face rank {row['expected_rank']}")

    for entry in results[1:]:
        print(f"\n--- {entry['asset']} ---")
        if entry.get("skipped"):
            print(f"  skipped: {entry['skipped']}")
            continue
        print(f"  {entry['regions']} regions, {entry['seconds']}s "
              f"({entry['seconds_per_region']}s/region), "
              f"{entry['distinct_recommended_faces']} distinct face(s) recommended")
        for row in entry["detail"][:8]:
            print(f"    {row['region']:>4} {str(row['text'])[:16]!r:18} "
                  f"own {str(row['own_top'])[:22]:<24} -> {str(row['recommended'])[:22]}")

    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
