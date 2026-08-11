## 🍢 eval_proof_matcher — can silhouette matching pick the right GT string?
## vieuxtiful
"""Phase 1's baseline, measured before any encoder is trained.

THE QUESTION. A region has been located and misread. Ground truth offers a
handful of candidate strings. Can rendering each candidate and comparing
silhouettes recover the right one from ink the recognizer could not read?

This is the cheapest thing that could possibly work -- no training, no
weights, nothing that could memorise the evaluation set -- and it is the
number a trained encoder has to beat before it earns its place. Nine
interventions in `docs/measured-dead-ends.md` looked obviously right in the
abstract and lost on measurement; an encoder proposed without a baseline
would be the tenth.

WHAT IT MEASURES, per ground-truth region whose recognizer read was wrong:

  top1        did the highest-scoring candidate equal the true string?
  margin      support(best) - support(runner-up). Says whether the winner
              is DISTINGUISHABLE, which top-1 accuracy alone cannot: a
              corpus of coin flips can score 50% top-1 and be worthless.
  support     absolute silhouette agreement of the winner.

THE CANDIDATE SET is every distinct annotated string on the same image, plus
mined hard negatives for the true string. Deliberately not "the true string
versus noise": the discrimination that matters is against the OTHER THINGS
ON THIS SIGN, which is what a real Ground Truth pool looks like.

HONEST LIMITS, stated because they bound every number below:

  * The font is a stand-in. ToFU does not know the sign's typeface at
    capture time, so this renders in one common face. A mismatch between
    the rendered face and the painted one is charged to the matcher here,
    which makes this a LOWER bound on what a face-aware version could do.
  * Only regions the recognizer got WRONG are scored. Regions it already
    reads correctly need no rescuing, and including them would inflate
    every figure with work that was already done.

usage:
  .venv/Scripts/python scripts/eval_proof_matcher.py
  .venv/Scripts/python scripts/eval_proof_matcher.py --out evidence/proof-baseline.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tests"))

## Faces to try, in order. A Latin face and a CJK-capable one, because a
## proof that cannot draw the candidate says nothing about the candidate.
FACE_CANDIDATES = (
    ("arial.ttf", "DejaVuSans.ttf", "LiberationSans-Regular.ttf"),
    ("msgothic.ttc", "NotoSansCJK-Regular.ttc", "YuGothM.ttc"),
)


def _faces() -> List[str]:
    from conftest import system_font
    found = []
    for group in FACE_CANDIDATES:
        path = system_font(*group)
        if path is not None:
            found.append(str(path))
    return found


def _observed_mask(image_path: Path, bbox: List[int]):
    """The ink inside a ground-truth box, as a tight mask.

    Uses the same `text_mask` the pipeline's own analysis uses rather than a
    fresh threshold: a diagnostic that segments differently from the product
    measures its own segmentation.
    """
    try:
        import numpy as np
        from tofu.utils.imaging import load_rgb, text_mask
    except ImportError:
        return None
    from tofu.core.types import BBox
    img = load_rgb(str(image_path))
    if img is None:
        return None
    x, y, w, h = [int(v) for v in bbox]
    ## text_mask takes the FULL image plus a bbox, not a pre-cut crop: it
    ## chooses Otsu or Sauvola per crop from the surrounding context, and
    ## handing it a slice would discard the thing it decides on.
    mask = text_mask(img, BBox(x=x, y=y, width=w, height=h))
    if mask is None or not mask.any():
        return None
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    if not rows.any() or not cols.any():
        return None
    y0, y1 = np.where(rows)[0][[0, -1]]
    x0, x1 = np.where(cols)[0][[0, -1]]
    return mask[y0:y1 + 1, x0:x1 + 1]


def _read(image_path: Path, bbox: List[int], language: Optional[str]) -> str:
    """What the pipeline reads here — the thing being rescued."""
    from tofu.core.types import BBox
    from tofu.layers import cicerone
    langs = cicerone.expand_langset([language]) if language else ("en",)
    backend = cicerone.EasyOCRBackend(languages=tuple(langs), gpu=False)
    x, y, w, h = [int(v) for v in bbox]
    found = backend.detect_in_regions(
        str(image_path), [BBox(x=x, y=y, width=w, height=h)],
    )
    return "".join(d.text or "" for group in found for d in group).strip()


def run(fixtures: List[Path], language: Optional[str] = None) -> Dict[str, Any]:
    from tofu.layers import proof

    faces = _faces()
    if not faces:
        raise SystemExit("no usable font faces on this machine")

    rows: List[Dict[str, Any]] = []
    for gt_path in fixtures:
        ## `Path.with_suffix` replaces only the LAST suffix, so on
        ## "la-bastille-1789.gt.json" it yields "...gt.jpeg" and every image
        ## resolves to None -- which the loop then skips silently. Strip the
        ## whole ".gt.json" tail instead.
        stem = gt_path.name[: -len(".gt.json")]
        image = next(
            (gt_path.parent / f"{stem}{ext}" for ext in (".jpeg", ".jpg", ".png")
             if (gt_path.parent / f"{stem}{ext}").exists()), None,
        )
        if image is None:
            continue
        doc = json.loads(gt_path.read_text(encoding="utf-8"))
        regions = [r for r in doc["regions"] if (r.get("text") or "").strip()]
        pool = sorted({(r.get("text") or "").strip() for r in regions})
        if len(pool) < 2:
            continue

        for region in regions:
            truth = (region.get("text") or "").strip()
            read = _read(image, region["bbox"], language)
            if read == truth:
                ## Already correct; nothing to rescue. Counting it would
                ## inflate the result with work the recognizer already did.
                continue
            observed = _observed_mask(image, region["bbox"])
            if observed is None:
                continue

            negatives = [n["text"] for n in proof.hard_negatives(truth)]
            candidates = sorted(set(pool) | set(negatives))
            best_face = None
            for face in faces:
                result = proof.match(observed, candidates, face)
                if result is None:
                    continue
                if best_face is None or result["support"] > best_face["support"]:
                    best_face = result
            if best_face is None:
                continue

            rows.append({
                "image": image.name,
                "truth": truth,
                "recognizer_read": read,
                "matcher_best": best_face["best"],
                "top1": best_face["best"] == truth,
                "support": best_face["support"],
                "margin": best_face["margin"],
                "candidates": len(candidates),
            })

    top1 = sum(1 for r in rows if r["top1"])
    return {
        "schema": 1,
        "faces": faces,
        "scored_regions": len(rows),
        "top1": top1,
        "top1_rate": round(100.0 * top1 / len(rows), 1) if rows else None,
        "rows": rows,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lang", default=None)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()

    fixtures = sorted((ROOT / "images").glob("*.gt.json"))
    result = run(fixtures, args.lang)

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n",
                            encoding="utf-8")
        print(f"wrote {args.out}\n")

    print(f"faces: {', '.join(Path(f).name for f in result['faces'])}")
    print(f"regions the recognizer got wrong, with usable ink: {result['scored_regions']}")
    if not result["rows"]:
        print("nothing to score")
        return 0
    ## The console is cp1252 on a default Windows shell and cannot print
    ## CJK. The JSON is already written by this point; the table is a
    ## convenience and must never be able to fail the run.
    def safe(text: str) -> str:
        encoding = sys.stdout.encoding or "utf-8"
        return text.encode(encoding, "replace").decode(encoding, "replace")

    print(f"\n{'truth':<16}{'read':<16}{'matcher':<16}{'ok':>3}{'support':>9}{'margin':>8}")
    for row in result["rows"]:
        print(f"{safe(row['truth'])[:15]:<16}{safe(row['recognizer_read'])[:15]:<16}"
              f"{safe(row['matcher_best'])[:15]:<16}{'Y' if row['top1'] else '.':>3}"
              f"{row['support']:>9.3f}"
              f"{(row['margin'] if row['margin'] is not None else 0):>8.3f}")
    print(f"\ntop-1: {result['top1']}/{result['scored_regions']} "
          f"= {result['top1_rate']}%")
    print("\nBaseline only. An encoder proposed for Phase 1 has to beat this "
          "number, and the font is a stand-in, so this is a LOWER bound.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
