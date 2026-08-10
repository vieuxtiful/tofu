## 🍢 warm_easyocr — pull the recognizer's weights at BUILD time, not first use
## vieuxtiful
"""
EasyOCR ships no weights. ``easyocr.Reader([...])`` downloads CRAFT plus one
recognition model per language group from jaided.ai on first construction and
caches them under ``~/.EasyOCR/model``.

In a container that default puts a ~250 MB download on the critical path of
the first real request a user makes, from a third party we do not control, in
a process already holding the upload. The failure mode is worse than the
latency: jaided.ai being slow or unreachable turns a working deployment into
one where detection times out, and nothing in the image says why.

So the download moves into the build, where it is allowed to be slow and
where failing is a build failure. Nothing about the runtime contract changes
-- ``cicerone.EasyOCRBackend._reader`` still constructs Readers exactly as it
did; it simply finds the cache already populated.

usage (from repo root, or inside a Docker build stage):
  .venv/Scripts/python scripts/warm_easyocr.py
  .venv/Scripts/python scripts/warm_easyocr.py --all-scripts
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

## The language sets ToFU actually constructs, taken from
## cicerone.SCRIPT_TO_EASYOCR_SET rather than restated here -- a hand-kept
## copy is how the warm set silently stops matching the runtime set.
##
## The default covers the scripts the project's own fixtures and the demo's
## advertised languages need: Latin, Cyrillic, Japanese, Hangul, Han. The
## remaining families are real but weigh another ~200 MB of recognition
## models, so they are opt-in rather than inflicted on every build.
DEFAULT_SCRIPTS = ("latin", "cyrillic", "japanese", "hangul", "han")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--all-scripts", action="store_true",
        help="also warm greek, arabic, hebrew, devanagari and thai",
    )
    args = parser.parse_args()

    import easyocr  # noqa: E402  (deferred: importing torch costs seconds)

    from tofu.layers.cicerone import SCRIPT_TO_EASYOCR_SET

    scripts = tuple(SCRIPT_TO_EASYOCR_SET) if args.all_scripts else DEFAULT_SCRIPTS

    # Distinct language SETS, not distinct scripts: latin and cyrillic share
    # ("ru", "en") members, and a Reader is cached per set, so deduplicating
    # here is what stops the same weights being fetched twice.
    wanted = []
    for script in scripts:
        languages = SCRIPT_TO_EASYOCR_SET.get(script)
        if languages and languages not in wanted:
            wanted.append(languages)

    failures = []
    for languages in wanted:
        print(f"warming easyocr {list(languages)} ...", flush=True)
        try:
            easyocr.Reader(list(languages), gpu=False)
        except Exception as exc:
            # Report every failure rather than dying on the first, so one
            # build shows the operator the whole picture.
            failures.append((languages, f"{type(exc).__name__}: {exc}"))
            print(f"  FAILED: {type(exc).__name__}: {exc}", file=sys.stderr, flush=True)

    if failures:
        print(f"\n{len(failures)} of {len(wanted)} language sets failed to warm",
              file=sys.stderr)
        return 1
    print(f"\nwarmed {len(wanted)} language sets")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
