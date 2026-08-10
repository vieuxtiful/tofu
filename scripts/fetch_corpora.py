## 🍢 fetch_corpora — the one place a corpus is allowed to come from the network
## vieuxtiful
"""
Downloads Leipzig Corpora Collection packages and renders them into the plain
one-sentence-per-line files build_ngram_models.py and build_vector_models.py
expect.

Why this is a separate, operator-run step rather than a line in the Dockerfile.
Every provider in tofu.layers.language_models promises never to download
anything, and that promise is what makes a ToFU deployment's behaviour a
function of its configuration rather than of whether some host was reachable
that morning. Folding a download into the image build would keep the letter of
that promise and lose its point: the artifact would still have arrived from
somewhere nobody recorded. So the fetch is explicit, it happens once, and it
writes down exactly what it got.

What gets written down is a lockfile, corpora.lock.json, beside the corpora.
First fetch of a package records its SHA-256; every later fetch verifies
against it and refuses a mismatch. Leipzig republishes under stable names, so
a changed digest means the corpus changed underneath a model that was
measured on the old one -- which is a thing to be told about, loudly, not a
thing to absorb silently.

Licensing: Leipzig packages are CC BY-NC or CC BY depending on the year and
source, and the per-package licence travels in the archive. The lockfile
records the package name so the deployment's provenance is answerable.

usage (from repo root):
  .venv/Scripts/python scripts/fetch_corpora.py --package fra_news_2024_100K
  .venv/Scripts/python scripts/fetch_corpora.py --package eng_news_2024_100K
  .venv/Scripts/python scripts/fetch_corpora.py --list-known

then build the models:
  .venv/Scripts/python scripts/build_ngram_models.py --family latin \\
      --corpus corpora/fra_news_2024_100K.txt --out models/ngram
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import tarfile
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASE_URL = "https://downloads.wortschatz-leipzig.de/corpora"

## Packages known to suit ToFU's two model families, with the family each
## feeds. Not a restriction -- any package name is accepted -- but naming the
## sensible defaults saves an operator a trip to the download page and stops
## a 1M-sentence package being fetched for a job a 100K one does better.
KNOWN = {
    "fra_news_2024_100K": "latin",
    "eng_news_2024_100K": "latin",
    "deu_news_2024_100K": "latin",
    "spa_news_2024_100K": "latin",
    "rus_news_2024_100K": "latin",
    "jpn_news_2024_100K": "cjk",
    "zho_news_2024_100K": "cjk",
    "kor_news_2024_100K": "cjk",
}


def _digest(path: Path) -> str:
    sha = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            sha.update(chunk)
    return sha.hexdigest()


def _load_lock(lock_path: Path) -> dict:
    if not lock_path.is_file():
        return {}
    try:
        return json.loads(lock_path.read_text(encoding="utf-8"))
    except Exception:
        print(f"warning: {lock_path} is unreadable; treating as empty", file=sys.stderr)
        return {}


def _extract_sentences(archive: Path, destination: Path) -> int:
    """Pull the *-sentences.txt member out and strip its leading id column.

    Leipzig ships 'id<TAB>sentence' per line. The id is corpus bookkeeping and
    would otherwise become a token in every model built from it -- a hundred
    thousand distinct integers in the vocabulary, each seen exactly once.
    """
    written = 0
    with tarfile.open(archive, "r:gz") as tar:
        member = next(
            (m for m in tar.getmembers() if m.name.endswith("-sentences.txt")), None
        )
        if member is None:
            raise ValueError(f"no *-sentences.txt member in {archive.name}")
        source = tar.extractfile(member)
        if source is None:
            raise ValueError(f"could not read {member.name} from {archive.name}")
        with destination.open("w", encoding="utf-8") as out:
            for raw in source:
                line = raw.decode("utf-8", errors="replace").rstrip("\n")
                _, _, sentence = line.partition("\t")
                sentence = (sentence or line).strip()
                if sentence:
                    out.write(sentence + "\n")
                    written += 1
    return written


def fetch(package: str, corpora_dir: Path, lock_path: Path, force: bool) -> int:
    url = f"{BASE_URL}/{package}.tar.gz"
    archive = corpora_dir / f"{package}.tar.gz"
    text = corpora_dir / f"{package}.txt"
    lock = _load_lock(lock_path)
    recorded = (lock.get(package) or {}).get("sha256")

    if text.is_file() and not force:
        print(f"{text.name} already present; pass --force to refetch")
        return 0

    if not archive.is_file() or force:
        print(f"downloading {url}")
        corpora_dir.mkdir(parents=True, exist_ok=True)
        try:
            urllib.request.urlretrieve(url, archive)
        except Exception as exc:
            print(f"download failed: {type(exc).__name__}: {exc}", file=sys.stderr)
            return 2

    digest = _digest(archive)
    if recorded and digest != recorded:
        print(
            f"REFUSING {package}: sha256 {digest[:16]}… does not match the recorded "
            f"{recorded[:16]}….\nThe published corpus changed. Any model measured on "
            f"the old one is no longer reproducible from this package -- decide that "
            f"deliberately, then rerun with --force.",
            file=sys.stderr,
        )
        return 3
    if not recorded:
        print(f"recording sha256 {digest[:16]}… for {package}")

    lines = _extract_sentences(archive, text)
    lock[package] = {
        "sha256": digest,
        "url": url,
        "family": KNOWN.get(package, "unknown"),
        "sentences": lines,
        "bytes": text.stat().st_size,
    }
    lock_path.write_text(json.dumps(lock, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    family = KNOWN.get(package, "latin")
    print(f"wrote {text} ({lines:,} sentences, {text.stat().st_size / 1e6:.1f} MB)")
    print(f"\nbuild the model:\n"
          f"  .venv/Scripts/python scripts/build_ngram_models.py "
          f"--family {family} --corpus {text} --out models/ngram")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package", action="append", default=[],
                        help="Leipzig package name, e.g. fra_news_2024_100K")
    parser.add_argument("--out", type=Path, default=ROOT / "corpora")
    parser.add_argument("--lock", type=Path, default=ROOT / "corpora.lock.json")
    parser.add_argument("--force", action="store_true",
                        help="refetch even when the text file is already present")
    parser.add_argument("--list-known", action="store_true")
    args = parser.parse_args()

    if args.list_known:
        print(f"{'package':<24}family")
        for name, family in sorted(KNOWN.items()):
            print(f"{name:<24}{family}")
        print("\nfull catalogue: https://wortschatz-leipzig.de/en/download")
        return 0

    if not args.package:
        parser.error("give at least one --package, or --list-known")

    for package in args.package:
        status = fetch(package, args.out, args.lock, args.force)
        if status:
            return status
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
