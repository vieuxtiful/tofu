## 🍢 build_ngram_models — n-gram artifacts for OCR rescoring, no toolchain
## vieuxtiful
"""
Builds the ``latin.ngram.json.gz`` / ``cjk.ngram.json.gz`` artifacts that
KneserNeyScoringProvider loads.

The sibling script build_kenlm_models.py produces a better model and needs
``lmplz`` and ``build_binary`` to do it -- C++ programs that ``pip install
kenlm`` does not provide. This one needs the interpreter already in hand,
and trains from tens of MB rather than gigabytes, because the model's job
is to rank a handful of candidate readings per region rather than to
generate text.

Deliberately takes a LOCAL corpus path and never downloads anything -- the
same contract every provider in language_models.py honours, and what makes
a build reproducible: the manifest written beside each model records
exactly which corpus produced it.

usage (from repo root):
  .venv/Scripts/python scripts/build_ngram_models.py \\
      --family latin --corpus D:/corpora/fra_news.txt --out models/ngram

  .venv/Scripts/python scripts/build_ngram_models.py \\
      --family cjk --corpus D:/corpora/jpn_news.txt --out models/ngram

then point ToFU at the result:
  set TOFU_NGRAM_DIR=models/ngram

## Getting a corpus

Any plain-text file, one sentence or document per line.

The Leipzig Corpora Collection publishes 100k-sentence packages per
language at roughly 10-20MB, freely licensed and citable, which is the
right size and shape for this:
https://wortschatz.uni-leipzig.de/en/download

Wikipedia extracts work equally well; see build_kenlm_models.py for the
WikiExtractor recipe. A few hundred MB is not better here in proportion to
what it costs -- this model ranks strings, it does not generate them.

## What it builds

An interpolated modified Kneser-Ney bigram: unigram continuation
probabilities (how many distinct contexts a token completes, not how often
it appeared) plus discounted bigram counts. The continuation term is the
whole reason for choosing Kneser-Ney here -- a plain backoff model ranks a
rare-but-real word below a common one, which is exactly the mistake that
matters when ranking OCR candidates.
"""
from __future__ import annotations

import argparse
import gzip
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tofu.layers.language_models import tokenize_for_lm  # noqa: E402

## Chen & Goodman's single-discount simplification. The three-discount form
## buys little on a corpus this size and costs a parameter fit nobody here
## would be able to check.
DISCOUNT = 0.75

## Tokens seen once in the whole corpus are typographic noise as often as
## they are words, and every one of them inflates the artifact.
MIN_UNIGRAM_COUNT = 2


def build(corpus: Path, family: str, max_lines: int | None) -> dict:
    unigram_counts: dict[str, int] = defaultdict(int)
    bigram_counts: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    continuation: dict[str, set] = defaultdict(set)
    lines = 0

    with corpus.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            if max_lines and lines >= max_lines:
                break
            tokens = tokenize_for_lm(line, family)
            if not tokens:
                continue
            lines += 1
            previous = "<s>"
            for token in tokens:
                unigram_counts[token] += 1
                bigram_counts[previous][token] += 1
                # Kneser-Ney's lower order counts CONTEXTS, not occurrences
                continuation[token].add(previous)
                previous = token
            if lines % 50000 == 0:
                print(f"  {lines:,} lines, {len(unigram_counts):,} types")

    keep = {t for t, c in unigram_counts.items() if c >= MIN_UNIGRAM_COUNT}
    total_contexts = sum(len(continuation[t]) for t in keep) or 1
    unigram = {t: len(continuation[t]) / total_contexts for t in keep}

    bigram: dict[str, dict] = {}
    for context, followers in bigram_counts.items():
        kept = {t: c for t, c in followers.items() if t in keep}
        if not kept:
            continue
        total = sum(kept.values())
        bigram[context] = {
            "counts": kept,
            "distinct": len(kept),   # mass this context reserves for backoff
            "total": total,
        }

    return {
        "family": family,
        "discount": DISCOUNT,
        "unigram": unigram,
        "bigram": bigram,
        "manifest": {
            "corpus": str(corpus),
            "corpus_bytes": corpus.stat().st_size,
            "lines_used": lines,
            "types": len(unigram),
            "contexts": len(bigram),
            "min_unigram_count": MIN_UNIGRAM_COUNT,
            "tokenizer": "tofu.layers.language_models.tokenize_for_lm",
            "built_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--family", choices=("latin", "cjk"), required=True)
    parser.add_argument("--corpus", type=Path, required=True,
                        help="local plain-text file; nothing is downloaded")
    parser.add_argument("--out", type=Path, default=ROOT / "models" / "ngram")
    parser.add_argument("--max-lines", type=int, default=None,
                        help="cap the corpus; useful for a first build")
    args = parser.parse_args()

    if not args.corpus.is_file():
        print(f"corpus not found: {args.corpus}", file=sys.stderr)
        return 2

    print(f"building {args.family} from {args.corpus}")
    model = build(args.corpus, args.family, args.max_lines)
    args.out.mkdir(parents=True, exist_ok=True)
    path = args.out / f"{args.family}.ngram.json.gz"
    with gzip.open(path, "wt", encoding="utf-8") as handle:
        json.dump(model, handle, ensure_ascii=False)

    manifest = model["manifest"]
    print(f"wrote {path} ({path.stat().st_size / 1e6:.1f} MB)")
    print(f"  {manifest['lines_used']:,} lines -> {manifest['types']:,} types, "
          f"{manifest['contexts']:,} contexts")
    print(f"\npoint ToFU at it:  set TOFU_NGRAM_DIR={args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
