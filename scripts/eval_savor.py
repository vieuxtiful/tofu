## 🍢 eval_savor — regression sweep for Savor (src/tofu/layers/savor.py)
"""
runs cicerone.detect() (savor=True, the default -- Savor's taste test
runs as the final step) over the full existing fixture set + real photos
and reports every instance with an ocr_correction verdict: applied
(chew_on confirmed and the Morsel was swallowed) or flagged unresolved
(chew_on was still chewing -- inconclusive, left on the plate) -- so a
human can eyeball each one for an unintended change, per the plan's
stated regression-sweep methodology.
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from tofu.core.types import AssetInfo, AssetType
from tofu.layers import cicerone

ROOT = Path(__file__).resolve().parent.parent
ASSETS = [
    ROOT / "tests/fixtures/flat-sign.png",
    ROOT / "tests/fixtures/gradient-banner.png",
    ROOT / "tests/fixtures/textured-wall.png",
    ROOT / "tests/fixtures/stylized-italic.png",
    ROOT / "tests/fixtures/expansion-en.png",
    ROOT / "tests/fixtures/cjk-vertical.png",
    ROOT / "images/gemini-street.png",
    ROOT / "japan-street.jpeg",
]


def detect(path):
    info = AssetInfo(asset_type=AssetType.IMAGE, frame_count=1, source=str(path))
    return cicerone.detect(str(path), info)


def main():
    t0 = time.time()
    total_instances = 0
    total_applied = 0
    total_unresolved = 0
    for path in ASSETS:
        if not path.exists():
            print(f"skip (missing): {path}")
            continue
        manifest = detect(path)
        total_instances += len(manifest.instances)
        print(f"\n== {path.name} == ({len(manifest.instances)} region(s))")
        for inst in manifest.instances:
            print(f"  {inst.id}: {inst.text!r}")
            oc = inst.ocr_correction
            if oc is None:
                continue
            if oc["applied"]:
                total_applied += 1
                print(f"    -> CORRECTED: {oc['original_text']!r} -> {oc['corrected_text']!r} ({oc['reason']})")
            else:
                total_unresolved += 1
                print(f"    -> UNRESOLVED candidate: {oc['candidate_text']!r} ({oc['reason']})")

    print(f"\n== summary ==")
    print(f"total regions: {total_instances}")
    print(f"corrections applied: {total_applied}")
    print(f"unresolved candidates flagged (not applied): {total_unresolved}")
    print(f"elapsed: {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
