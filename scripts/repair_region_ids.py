## 🍢 repair: make a damaged manifest satisfy the contiguity invariant
## vieuxtiful
"""Repair a manifest whose region ids and references have come apart.

WHAT DAMAGE LOOKS LIKE. prem-sais-gt: seven regions numbered r9-r15, every
semantic unit still naming r1-r8, and a `reading_order` of 8,1,2,3,4,5,6 --
no 0 and no 7. Regions were rebuilt at some point without rewriting the
references, and nothing noticed because nothing checked.

WHY THE UNITS ARE DROPPED RATHER THAN REMAPPED. Their references are already
dangling, so there is no correct target to remap them TO. Matching them by
ordinal -- treating the old r1 as the new r1 -- would manufacture a mapping
from nothing and make invalid references look valid, which is worse than the
visible breakage. Every unit in the damaged manifest is `origin: derived`,
`analysis_provider: single_region`, so no manual grouping is lost; the plate
layer rebuilds them from the compacted regions.

ORDER OF OPERATIONS MATTERS. The stale units are removed BEFORE compaction,
so the compactor never sees a dangling reference it could be asked to
reinterpret, and its default `on_dangling="error"` stays meaningful.

DRY RUN BY DEFAULT. Nothing is written without --apply, and --apply takes a
backup first.
"""
from __future__ import annotations

import argparse
import shutil
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "server"))

from tofu.layers import couvert  # noqa: E402
from tofu.utils.manifest_store import load_manifest, save_manifest  # noqa: E402


def repair(store: Path, asset_id: str, *, apply: bool) -> int:
    manifest = load_manifest(store, asset_id)
    if manifest is None:
        print(f"no manifest for {asset_id}", file=sys.stderr)
        return 2

    live = [i.id for i in manifest.instances if not i.excluded]
    known = {i.id for i in manifest.instances}
    dangling = couvert.dangling_ids(manifest)
    orders = [i.reading_order for i in manifest.instances if not i.excluded]

    print(f"asset            : {asset_id}")
    print(f"live regions     : {live}")
    print(f"reading_order    : {orders}")
    print(f"dangling refs    : {sorted(dangling)}")

    stale_units = [
        unit for unit in (manifest.semantic_units or [])
        if set(unit.region_ids or []) - known
    ]
    print(f"stale units      : {[u.id for u in stale_units]}")

    if not apply:
        plan = couvert.plan(manifest)
        print(f"would renumber   : "
              f"{ {old: new for old, new in plan.mapping.items()} }")
        print("\n(dry run — pass --apply to write)")
        return 0

    ## UTC to the microsecond PLUS a short uuid: two repairs inside the same
    ## second would otherwise write the same backup name, and the second would
    ## overwrite the only copy of the pre-repair state.
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%f")
    backup = store / f"{asset_id}.manifest.pre-repair-{stamp}-{uuid.uuid4().hex[:8]}.json"
    shutil.copy2(store / f"{asset_id}.manifest.json", backup)
    print(f"backup           : {backup.name}")

    ## 1. Drop the stale units first, so compaction sees a clean graph.
    if stale_units:
        keep = [u for u in manifest.semantic_units if u not in stale_units]
        manifest.semantic_units = keep

    ## 2. Compact. `error` on purpose: after step 1 there should be nothing
    ##    dangling left, and if there is, this must stop rather than guess.
    mapping = couvert.compact_region_ids(manifest, on_dangling="error")
    print(f"renumbered       : {mapping}")

    ## 3. Prove it, then write.
    couvert.validate(manifest)
    manifest.total_regions = sum(1 for i in manifest.instances if not i.excluded)
    save_manifest(store, asset_id, manifest)

    reloaded = load_manifest(store, asset_id)
    couvert.validate(reloaded)
    print(f"live regions now : {[i.id for i in reloaded.instances if not i.excluded]}")
    print(f"reading_order now: {[i.reading_order for i in reloaded.instances if not i.excluded]}")
    print(f"semantic units   : {len(reloaded.semantic_units or [])}")
    print("validated after reload — OK")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("asset_id")
    parser.add_argument("--store", default=str(ROOT / "server" / "uploads"))
    parser.add_argument("--apply", action="store_true",
                        help="write the repair (a backup is taken first)")
    args = parser.parse_args()
    return repair(Path(args.store), args.asset_id, apply=args.apply)


if __name__ == "__main__":
    raise SystemExit(main())
