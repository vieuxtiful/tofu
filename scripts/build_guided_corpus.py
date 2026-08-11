## 🍢 build_guided_corpus — the Guided evaluation corpus (Release 5 gate)
## vieuxtiful
"""Derive Guided Blocks from existing ground truth, and freeze them.

Gate 2 asks whether Guided finds REQUESTED text that Auto misses.  Answering
that needs something the detection corpus does not carry: for each asset, the
source text a user would actually type, and where each occurrence of it is.

Rather than hand-authoring that (and inventing geometry in the process), it
is DERIVED from annotations that already exist.  Every derivation rule is
recorded per Block so a reader can tell what was authored by a human and what
was computed:

  gt_region   one annotated region's text becomes one Block
  gt_repeat   the same text annotated in N regions of one asset -- the
              duplicate-occurrence case, which is exactly where set-based
              matching quietly fails

What this deliberately does NOT do:

  * invent multi-region phrase Blocks.  Deciding that "Rue" + "des" +
    "MARTYRS" is one requested phrase is a judgement about the sign, not a
    fact in the annotation, and a heuristic guess dressed as ground truth
    would corrupt the very measurement it feeds.
  * claim human review.  `review_status` starts at `machine_derived`, and
    `eval_guided_corpus.py` REFUSES to certify Gate 2 until a human has
    reviewed and the status says so.  A corpus that certifies itself is not
    evidence.

Usage:
    .venv/Scripts/python scripts/build_guided_corpus.py
    .venv/Scripts/python scripts/build_guided_corpus.py --verify
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import unicodedata
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

OUT_PATH = ROOT / "evidence" / "guided-corpus-v1.json"
SCHEMA = 1

## The plan's release minimums.  Recorded in the artifact so the shortfall is
## visible in the file itself, not only in whoever ran the script.
MINIMUMS = {"assets": 30, "blocks": 120, "occurrences": 180, "strata": 6}

## Strata are DERIVED FROM THE ANNOTATION TEXT, not from a hand-written map.
## A hand map got three of these wrong on the first attempt: `rtl-sign`
## contains the Latin word "Welcome" (it is a sign to be localized INTO an RTL
## target, not RTL text), `indic-shaping` is Devanagari/Thai/Tamil which shape
## richly but read left-to-right, and `cjk-horizontal` is annotated "City
## Library" / "Open today".  A corpus that claims RTL coverage it does not
## have is worse than one that admits the gap, so the rule reads the content.
##
## Only `nonlinear_irregular` stays declarative: rotation and overlap are
## properties of the rendering, not of the string, and the annotations do not
## record them.
NONLINEAR: set = {
    "stylized-italic", "perspective-sign", "mixed-orientation",
    "dense-layout", "la-bastille-1789",
    # Dense real street scenes. Previously misfiled as japanese_vertical by a
    # geometry heuristic: a crowded photo has many tall boxes, which is not
    # the same thing as a stacked column. Those two supplied 22 of that
    # stratum's 24 occurrences, so its recall figure was never a measurement
    # of vertical text at all.
    "china-street", "gemini-street",
}

## Vertical layout is DECLARED, for the same reason nonlinear is: whether a
## sign reads top-to-bottom is a property of the rendering, and the
## annotations do not record it. Inferring it from box aspect ratio produced
## exactly the mislabel above.
VERTICAL: set = {"cjk-vertical", "cjk-vertical-menu", "cjk-vertical-banner"}


def stratum_for(stem: str, regions: list) -> str:
    """Classify an asset by the scripts its ground truth actually contains."""
    from tofu.layers.palate import scripts_in

    if stem in NONLINEAR:
        return "nonlinear_irregular"
    scripts: set = set()
    has_digits = False
    for region in regions:
        text = (region.get("text") or "")
        for name in scripts_in(text):
            if name != "Zyyy":
                scripts.add(name)
        has_digits = has_digits or any(ch.isdigit() for ch in text)

    if scripts & {"Arab", "Hebr"}:
        return "rtl_bidi"
    japanese = scripts & {"Hira", "Kana", "Hani"}
    if japanese:
        return "japanese_vertical" if stem in VERTICAL else "japanese_horizontal"
    if len(scripts) > 1 or (scripts and has_digits and scripts != {"Latn"}):
        return "mixed_script_numeric"
    if scripts == {"Latn"} and has_digits:
        return "mixed_script_numeric"
    return "latin_horizontal"


REQUIRED_STRATA = {
    "latin_horizontal", "japanese_horizontal", "japanese_vertical",
    "mixed_script_numeric", "rtl_bidi", "nonlinear_irregular",
}

SEARCH_DIRS = (ROOT / "images", ROOT / "tests" / "fixtures")
IMAGE_SUFFIXES = (".png", ".jpeg", ".jpg")
## Below this a "Block" is a glyph, not a request someone would type.
MIN_BLOCK_CHARS = 2


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", unicodedata.normalize("NFC", text or "")).strip().casefold()


def _find_asset(stem: str) -> Path | None:
    for directory in SEARCH_DIRS:
        for suffix in IMAGE_SUFFIXES:
            candidate = directory / f"{stem}{suffix}"
            if candidate.exists():
                return candidate
    return None


def _annotations() -> Dict[str, Path]:
    found: Dict[str, Path] = {}
    for directory in SEARCH_DIRS:
        for path in sorted(directory.glob("*.gt.json")):
            found.setdefault(path.name[: -len(".gt.json")], path)
    return found


def _blocks_for(regions: List[Dict[str, Any]], asset_stem: str) -> List[Dict[str, Any]]:
    """One Block per distinct annotated string; occurrences are its regions."""
    grouped: Dict[str, List[Dict[str, Any]]] = {}
    for region in regions:
        text = (region.get("text") or "").strip()
        if len(text) < MIN_BLOCK_CHARS:
            continue
        grouped.setdefault(_normalize(text), []).append(region)

    blocks: List[Dict[str, Any]] = []
    for index, (key, members) in enumerate(sorted(grouped.items()), 1):
        occurrences = [
            {
                "bbox": list(member.get("bbox") or []),
                "text": (member.get("text") or "").strip(),
                "language": member.get("lang"),
            }
            for member in members
        ]
        blocks.append({
            "block_id": f"{asset_stem}#b{index}",
            "requested_text": (members[0].get("text") or "").strip(),
            "normalized_text": key,
            "derivation": "gt_repeat" if len(members) > 1 else "gt_region",
            "expected_occurrences": len(occurrences),
            "occurrences": occurrences,
            # One region means the plate is unambiguous; anything else is a
            # grouping question this corpus does not claim to answer.
            "expected_plate": "single_region" if len(occurrences) == 1 else "unspecified",
        })
    return blocks


def build() -> Dict[str, Any]:
    annotations = _annotations()
    assets: Dict[str, Any] = {}
    unstratified: List[str] = []

    for stem in sorted(annotations):
        image = _find_asset(stem)
        if image is None:
            continue
        annotation_path = annotations[stem]
        document = json.loads(annotation_path.read_text(encoding="utf-8"))
        regions = document.get("regions") or []
        blocks = _blocks_for(regions, stem)
        if not blocks:
            continue
        stratum = stratum_for(stem, regions)
        assets[stem] = {
            "stratum": stratum,
            # A partial annotation means unlisted text is NOT a false
            # positive; the evaluator has to withhold precision for these.
            "partial_annotation": bool(document.get("partial", False)),
            "image": {"path": str(image.relative_to(ROOT)).replace("\\", "/"),
                      "sha256": _sha256(image)},
            "annotation": {"path": str(annotation_path.relative_to(ROOT)).replace("\\", "/"),
                           "sha256": _sha256(annotation_path)},
            "blocks": blocks,
        }

    # Deterministic, content-blind holdout: within each stratum, sort by name
    # and hold out the last. Chosen before any score was seen.
    by_stratum: Dict[str, List[str]] = {}
    for stem, entry in assets.items():
        by_stratum.setdefault(entry["stratum"], []).append(stem)
    for members in by_stratum.values():
        members.sort()
        for stem in members:
            assets[stem]["split"] = "dev"
        if len(members) > 1:
            assets[members[-1]]["split"] = "holdout"

    total_blocks = sum(len(a["blocks"]) for a in assets.values())
    total_occurrences = sum(
        block["expected_occurrences"] for a in assets.values() for block in a["blocks"]
    )
    repeat_blocks = sum(
        1 for a in assets.values() for b in a["blocks"] if b["derivation"] == "gt_repeat"
    )
    totals = {
        "assets": len(assets),
        "blocks": total_blocks,
        "occurrences": total_occurrences,
        "repeat_blocks": repeat_blocks,
        "strata": len(by_stratum),
    }
    shortfalls = {
        name: {"required": MINIMUMS[name], "have": totals[name]}
        for name in MINIMUMS if totals.get(name, 0) < MINIMUMS[name]
    }
    missing_strata = sorted(REQUIRED_STRATA - set(by_stratum))

    return {
        "corpus_id": "guided-corpus-v1",
        "schema": SCHEMA,
        "note": (
            "Guided Blocks derived from existing ground truth. Derivation is "
            "recorded per Block. NOT human-reviewed: see review_status."
        ),
        # The single field that stops this artifact from certifying itself.
        "review_status": "machine_derived",
        "review": {
            "required": [
                "a contributor confirms each requested_text is text a user "
                "would plausibly ask ToFU to locate",
                "a second contributor independently reviews, and disagreements "
                "are adjudicated and recorded here",
            ],
            "annotators": [],
            "adjudications": [],
        },
        "minimums": MINIMUMS,
        "totals": totals,
        "shortfalls": shortfalls,
        "missing_strata": missing_strata,
        "unstratified_annotations": sorted(unstratified),
        "strata": {
            name: {
                "assets": len(members),
                "blocks": sum(len(assets[s]["blocks"]) for s in members),
                "occurrences": sum(
                    b["expected_occurrences"] for s in members for b in assets[s]["blocks"]
                ),
            }
            for name, members in sorted(by_stratum.items())
        },
        "split_rule": (
            "Within each stratum, sort asset names and hold out the last. "
            "Deterministic and content-blind, so the split cannot have been "
            "chosen after seeing which assets scored well."
        ),
        "assets": assets,
    }


def _report(document: Dict[str, Any]) -> None:
    totals, minimums = document["totals"], document["minimums"]
    print(f"corpus         {document['corpus_id']}  (schema {document['schema']})")
    print(f"review status  {document['review_status']}")
    print()
    print(f"{'metric':14s} {'have':>6} {'required':>9}   status")
    for name in ("assets", "blocks", "occurrences", "strata"):
        have, need = totals[name], minimums[name]
        print(f"{name:14s} {have:6d} {need:9d}   {'ok' if have >= need else 'SHORT'}")
    print(f"{'repeat blocks':14s} {totals['repeat_blocks']:6d} {'-':>9}   duplicate-occurrence cases")
    print()
    print(f"{'stratum':24s} {'assets':>6} {'blocks':>7} {'occ':>5}")
    for name, counts in document["strata"].items():
        print(f"{name:24s} {counts['assets']:6d} {counts['blocks']:7d} {counts['occurrences']:5d}")
    if document["missing_strata"]:
        print("\nMISSING REQUIRED STRATA:", ", ".join(document["missing_strata"]))
    if document["unstratified_annotations"]:
        print("\nannotations with no declared stratum (excluded):",
              ", ".join(document["unstratified_annotations"]))
    if document["shortfalls"]:
        print("\nThis corpus is BELOW the release minimums. eval_guided_corpus.py")
        print("will refuse to certify Gate 2 against it.")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--verify", action="store_true",
                        help="rebuild and compare against the committed artifact")
    args = parser.parse_args()

    document = build()
    if args.verify:
        if not OUT_PATH.exists():
            print(f"no corpus at {OUT_PATH}", file=sys.stderr)
            return 1
        committed = json.loads(OUT_PATH.read_text(encoding="utf-8"))
        # review_status and review are edited by humans; everything else is derived.
        rebuilt = dict(document)
        rebuilt["review_status"] = committed.get("review_status")
        rebuilt["review"] = committed.get("review")
        if rebuilt == committed:
            print("corpus matches the committed artifact")
            return 0
        print("CORPUS HAS DRIFTED from the committed artifact", file=sys.stderr)
        return 1

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(document, indent=2, ensure_ascii=False) + "\n",
                        encoding="utf-8")
    _report(document)
    print(f"\nwrote {OUT_PATH.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
