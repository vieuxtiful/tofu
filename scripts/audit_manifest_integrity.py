"""Read-only integrity audit for a directory of persisted manifests."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from tofu.utils.manifest_store import load_manifest
from tests.support.integrity import assert_manifest_integrity


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest_dir", nargs="?", default="server/uploads")
    args = parser.parse_args()
    root = Path(args.manifest_dir)
    paths = sorted(root.glob("*.manifest.json"))
    damaged: list[tuple[str, str]] = []

    for path in paths:
        asset_id = path.name.removesuffix(".manifest.json")
        try:
            manifest = load_manifest(root, asset_id)
            if manifest is None:
                raise AssertionError("manifest disappeared during audit")
            assert_manifest_integrity(manifest, context=path.name)
        except Exception as exc:
            damaged.append((asset_id, str(exc)))

    print(f"TOTAL={len(paths)} GOOD={len(paths) - len(damaged)} DAMAGED={len(damaged)}")
    for asset_id, defect in damaged:
        print(f"{asset_id}: {defect}")
    return 1 if damaged else 0


if __name__ == "__main__":
    raise SystemExit(main())
