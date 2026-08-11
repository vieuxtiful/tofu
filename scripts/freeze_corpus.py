"""Freeze the detection corpus: hashes, strata, split, provenance (roadmap P0.3a).

A corpus that can change without anyone noticing is not a measurement
instrument, it is a moving target that produces numbers. Three things have
to be pinned before any recall figure means anything across two commits:

  **Content.** Every asset and every annotation file, by hash. The existing
  baseline already carried a `gt_sha256` field -- containing a 32-character
  MD5. A hash whose name misstates its algorithm is worse than no hash: it
  invites a verification that silently never matches.

  **Strata.** Which fixture stands for which declared difficulty. Without
  this, "no stratum below 0.80 recall" is unenforceable because nothing says
  what the strata are.

  **Split.** Which fixtures thresholds may be tuned on and which they may
  not. The split rule here is deterministic and content-blind -- sort each
  stratum by name, hold out the last -- specifically so it cannot have been
  chosen after seeing which fixtures scored well. A split picked to flatter
  a result is not a holdout.

What this deliberately does NOT do is invent licences. Provenance is
recorded from evidence in the repository or marked `unknown`; see the
`provenance` block per fixture and the WARNINGS section of --verify output.

Usage:
    .venv/Scripts/python scripts/freeze_corpus.py           # write the manifest
    .venv/Scripts/python scripts/freeze_corpus.py --verify  # check, change nothing
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
IMAGES = ROOT / "images"
MANIFEST = ROOT / "evidence" / "corpus-v1.json"

CORPUS_ID = "detection-corpus-v1"

# Declared strata. A fixture belongs to exactly one; the stratum is what the
# per-stratum recall floor in the roadmap is actually measured over. Keep the
# names stable -- they are quoted in targets.
STRATA: dict[str, str] = {
    "avenue-de-la-république": "latin_plaque",
    "la-rue-sans-nom-example": "latin_plaque",
    "quai-des-orfevres": "latin_plaque",
    "rue-des-martyrs": "latin_plaque",
    "decolonisons-nos-rues": "latin_poster",
    "la-bastille-1789": "latin_poster",
    "china-street": "cjk_horizontal",
    "japan-street": "cjk_vertical",
    "russian-billboard": "cyrillic",
    "russian-billboard-2": "cyrillic",
    "gemini-street": "mixed_dense",
}

# Extensions vary across the corpus (.png/.jpg/.jpeg, inconsistently), so the
# asset is discovered rather than assumed. A hardcoded suffix table is a
# standing invitation to freeze a fixture that is not the one being measured.
IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".webp")


def resolve_image(stem: str) -> Path:
    """The one image file for a fixture stem, or an explicit failure."""
    found = [
        path
        for ext in IMAGE_EXTENSIONS
        for path in [IMAGES / f"{stem}{ext}"]
        if path.exists()
    ]
    if not found:
        raise FileNotFoundError(
            f"no asset for fixture {stem!r} in {IMAGES} "
            f"(tried {', '.join(IMAGE_EXTENSIONS)})"
        )
    if len(found) > 1:
        raise RuntimeError(
            f"fixture {stem!r} is ambiguous -- {len(found)} assets share the "
            f"stem: {[p.name for p in found]}. Which one the harness reads "
            f"would depend on extension order, so the corpus is not frozen "
            f"until one is removed."
        )
    return found[0]

# Provenance recorded ONLY from evidence present in the repository -- the GT
# notes, or text visible in the asset itself. Everything else is `unknown`,
# which is a finding rather than a formality: these files are tracked in a
# public MIT-licensed repository, so an asset whose licence nobody has
# established is being redistributed on an assumption. Resolving these is
# part of P0.3b; recording them honestly is part of P0.3a.
PROVENANCE: dict[str, dict[str, str]] = {
    "gemini-street": {
        "licence": "unknown",
        "source": "AI-generated (Google Gemini, per filename and GT note)",
        "evidence": "gt note: 'AI-generated scene'",
        "redistribution": "unresolved: generated-image terms not recorded",
    },
    "russian-billboard-2": {
        "licence": "unknown",
        "source": "third-party photograph",
        "evidence": (
            "watermark transcribed in the asset's own GT: "
            "'Valentina Ursu (RFE/RL)' -- an attributed press photograph"
        ),
        "redistribution": (
            "LIKELY RESTRICTED: a visible photographer/agency credit is "
            "positive evidence of third-party copyright. Highest-priority "
            "asset to clear or replace."
        ),
    },
}

_DEFAULT_PROVENANCE = {
    "licence": "unknown",
    "source": "unrecorded",
    "evidence": "none found in repository",
    "redistribution": "unresolved",
}


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _dimensions(path: Path) -> list[int] | None:
    try:
        from PIL import Image

        with Image.open(path) as img:
            return [img.width, img.height]
    except Exception:
        return None


def assign_split(stems: list[str]) -> dict[str, str]:
    """Deterministic, content-blind holdout: last name in each stratum.

    Strata with a single fixture cannot contribute a holdout member without
    emptying themselves, so they stay in dev and are reported as having no
    holdout representation. That is a real limitation of an 11-fixture
    corpus and is surfaced rather than smoothed over.
    """
    by_stratum: dict[str, list[str]] = {}
    for stem in stems:
        by_stratum.setdefault(STRATA[stem], []).append(stem)

    split: dict[str, str] = {stem: "dev" for stem in stems}
    for members in by_stratum.values():
        if len(members) > 1:
            split[sorted(members)[-1]] = "holdout"
    return split


def build() -> dict[str, Any]:
    stems = sorted(STRATA)
    split = assign_split(stems)
    fixtures: dict[str, Any] = {}
    warnings: list[str] = []

    for stem in stems:
        image = resolve_image(stem)
        gt = IMAGES / f"{stem}.gt.json"
        if not gt.exists():
            raise FileNotFoundError(f"annotation missing: {gt}")

        gt_data = json.loads(gt.read_text(encoding="utf-8"))
        provenance = PROVENANCE.get(stem, dict(_DEFAULT_PROVENANCE))
        if provenance["licence"] == "unknown":
            warnings.append(f"{stem}: licence unresolved ({provenance['redistribution']})")

        fixtures[stem] = {
            "stratum": STRATA[stem],
            "split": split[stem],
            "image": {
                "path": f"images/{image.name}",
                "sha256": sha256_of(image),
                "bytes": image.stat().st_size,
                "dimensions": _dimensions(image),
            },
            "annotation": {
                "path": f"images/{gt.name}",
                "sha256": sha256_of(gt),
                "regions": len(gt_data.get("regions", [])),
                "partial": bool(gt_data.get("partial", False)),
            },
            "provenance": provenance,
        }

    holdout = sorted(s for s in stems if split[s] == "holdout")
    dev = sorted(s for s in stems if split[s] == "dev")

    # The split hash covers membership AND annotation content, so moving a
    # fixture between sides or editing its GT both invalidate it. Tuning
    # against a holdout that has since been edited is the failure this
    # catches.
    split_material = "\n".join(
        f"{stem}:{fixtures[stem]['annotation']['sha256']}" for stem in holdout
    )
    split_hash = hashlib.sha256(split_material.encode("utf-8")).hexdigest()

    strata_summary: dict[str, Any] = {}
    for stem in stems:
        entry = strata_summary.setdefault(
            STRATA[stem], {"fixtures": 0, "regions": 0, "dev": 0, "holdout": 0}
        )
        entry["fixtures"] += 1
        entry["regions"] += fixtures[stem]["annotation"]["regions"]
        entry[split[stem]] += 1

    for name, entry in strata_summary.items():
        if entry["holdout"] == 0:
            warnings.append(
                f"stratum {name!r} has no holdout representation "
                f"({entry['fixtures']} fixture(s)) -- its recall floor is "
                f"measured on tuned data only"
            )

    total_regions = sum(f["annotation"]["regions"] for f in fixtures.values())
    holdout_regions = sum(fixtures[s]["annotation"]["regions"] for s in holdout)

    return {
        "corpus_id": CORPUS_ID,
        "schema": 1,
        "note": (
            "Frozen detection corpus (roadmap P0.3a). Generated by "
            "scripts/freeze_corpus.py -- do not hand-edit. Hashes are SHA-256 "
            "throughout; the superseded baseline field named `gt_sha256` held "
            "an MD5."
        ),
        "split_rule": (
            "Deterministic and content-blind: within each stratum, sort "
            "fixtures by name and hold out the last. Chosen so the split "
            "cannot have been selected after seeing which fixtures scored "
            "well. Single-fixture strata stay in dev and are reported as "
            "having no holdout representation."
        ),
        "split_hash": split_hash,
        "totals": {
            "fixtures": len(fixtures),
            "regions": total_regions,
            "dev_fixtures": len(dev),
            "holdout_fixtures": len(holdout),
            "holdout_regions": holdout_regions,
            "holdout_region_fraction": round(holdout_regions / total_regions, 3),
        },
        "dev": dev,
        "holdout": holdout,
        "strata": strata_summary,
        "fixtures": fixtures,
        "warnings": sorted(set(warnings)),
        "limitations": [
            "11 fixtures / 72 regions is far below the 100/500 release "
            "minimum. Per-stratum figures over 2-18 regions carry intervals "
            "wide enough that most differences are not resolvable.",
            "Four of six strata hold a single fixture, so they contribute no "
            "holdout evidence at all.",
            "Nine of eleven annotations are PARTIAL: unannotated text is "
            "neither credited nor penalised, so precision and F1 are not "
            "meaningful corpus-wide and only recall is comparable.",
            "Licences are unresolved for every asset. These files are "
            "tracked in a public MIT repository.",
        ],
    }


def verify() -> int:
    if not MANIFEST.exists():
        print(f"FAIL: no manifest at {MANIFEST}. Run without --verify first.")
        return 1

    recorded = json.loads(MANIFEST.read_text(encoding="utf-8"))
    current = build()
    problems: list[str] = []

    for stem, entry in recorded.get("fixtures", {}).items():
        now = current["fixtures"].get(stem)
        if now is None:
            problems.append(f"{stem}: in the manifest but no longer on disk")
            continue
        for kind in ("image", "annotation"):
            if entry[kind]["sha256"] != now[kind]["sha256"]:
                problems.append(
                    f"{stem}: {kind} changed since freeze "
                    f"({entry[kind]['sha256'][:12]} -> {now[kind]['sha256'][:12]})"
                )
        if entry["split"] != now["split"]:
            problems.append(
                f"{stem}: split moved {entry['split']} -> {now['split']}"
            )

    for stem in current["fixtures"]:
        if stem not in recorded.get("fixtures", {}):
            problems.append(f"{stem}: on disk but not in the frozen manifest")

    if recorded.get("split_hash") != current["split_hash"]:
        problems.append(
            "split hash changed: holdout membership or holdout annotations "
            "were edited. Any threshold tuned since the last freeze may have "
            "seen holdout data."
        )

    if problems:
        print("corpus verification FAILED:")
        for p in problems:
            print(f"  - {p}")
        return 1

    t = recorded["totals"]
    print(
        f"corpus OK: {t['fixtures']} fixtures, {t['regions']} regions, "
        f"holdout {t['holdout_fixtures']} fixtures / {t['holdout_regions']} "
        f"regions ({t['holdout_region_fraction']:.0%})"
    )
    print(f"split hash: {recorded['split_hash'][:16]}...")
    if recorded.get("warnings"):
        print(f"\n{len(recorded['warnings'])} WARNING(S) recorded at freeze time:")
        for w in recorded["warnings"]:
            print(f"  ! {w}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--verify", action="store_true", help="check without writing")
    args = ap.parse_args()

    if args.verify:
        return verify()

    manifest = build()
    MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    MANIFEST.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    t = manifest["totals"]
    print(f"wrote {MANIFEST}")
    print(
        f"  {t['fixtures']} fixtures, {t['regions']} regions, "
        f"holdout {t['holdout_fixtures']} / {t['holdout_regions']} regions "
        f"({t['holdout_region_fraction']:.0%})"
    )
    print(f"  holdout: {', '.join(manifest['holdout'])}")
    print(f"  split hash: {manifest['split_hash'][:16]}...")
    if manifest["warnings"]:
        print(f"\n{len(manifest['warnings'])} warning(s):")
        for w in manifest["warnings"]:
            print(f"  ! {w}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
