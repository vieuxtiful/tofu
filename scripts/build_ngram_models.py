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
import math
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

## Order of the character model that scores words the word model never saw.
## Three is the shortest order that still encodes which letter sequences a
## language permits -- 'qu' followed by a vowel, 'ght' word-finally in
## English, 'ez' in French -- and the artifact grows with the order while the
## thing being ranked is a handful of readings per region.
CHAR_ORDER = 3
CHAR_BOUNDARY = "\x02"


def _char_ngrams(token: str):
    """Padded character n-grams of a single word token.

    Padding is what lets the model charge for a shape at the EDGES of a word,
    which is where OCR confusions land: a leading or trailing substitution
    produces a sequence the language does not begin or end with, and an
    unpadded model would never see that.
    """
    padded = CHAR_BOUNDARY * (CHAR_ORDER - 1) + token + CHAR_BOUNDARY
    for i in range(CHAR_ORDER - 1, len(padded)):
        yield padded[i - CHAR_ORDER + 1:i], padded[i]


def _char_logprob_per_char(token: str, counts: dict, alphabet: int) -> float:
    """Mean log10 P(character | preceding characters) across a token.

    Shared in spirit with KneserNeyScoringProvider._oov_logprob, and it has
    to stay that way: this function establishes the scale that one measures
    against, so a difference in smoothing between them would silently shift
    every OOV score relative to its own reference.
    """
    padded = CHAR_BOUNDARY * (CHAR_ORDER - 1) + token + CHAR_BOUNDARY
    total = 0.0
    observed = 0
    for i in range(CHAR_ORDER - 1, len(padded)):
        context, nxt = padded[i - CHAR_ORDER + 1:i], padded[i]
        entry = counts.get(context)
        if entry is None:
            total += math.log10(1.0 / (alphabet + 1))
        else:
            total += math.log10((entry["counts"].get(nxt, 0) + 1) / (entry["total"] + alphabet))
        observed += 1
    return total / observed if observed else 0.0


## Fraction of types withheld from the reference model. A tenth is enough to
## measure a mean and a spread on a real vocabulary while leaving the
## reference model nearly as informed as the shipped one.
HELDOUT_FRACTION = 10


def _char_reference(alphabet: int, unigram_counts: dict,
                    keep: set) -> tuple[float, float]:
    """What a per-character score looks like for a word the model HASN'T seen.

    Measured held-out, and it has to be. The obvious version -- score the
    training vocabulary against the character counts trained on it -- asks
    the model about words it has already memorised, so the spread it reports
    is the spread of its own successes. Every genuinely unseen word then
    lands many standard deviations below that mean, the graded score
    saturates at the floor, and the change is worth nothing: measured on a
    150-type corpus, all three real-word/misread pairs collapsed to an exact
    tie at OOV_LOGPROB, which is the flat cliff this exists to remove.

    So the reference is built the way the runtime case actually looks: a
    tenth of the types are withheld, the character counts are accumulated
    from the rest, and the withheld types -- unseen, by construction -- are
    what the distribution is measured over. This is ordinary held-out
    estimation (Jelinek & Mercer 1980); the deleted-interpolation refinement
    that averages over every fold buys precision this use does not need,
    since the number is a scale for a comparison and not a parameter fit.

    Types are assigned to the fold by a hash of the token, not by position,
    so the split does not track corpus order -- and is stable across rebuilds
    of the same corpus.

    Weighted by occurrence rather than by type: the question at scoring time
    is "does this look like a word of this language", and a type-weighted
    mean is dominated by the long tail of rare and often malformed strings.

    Note the one asymmetry left. The shipped character model is trained on
    ALL types, so a runtime OOV word scores marginally better against it than
    the withheld types did against the smaller reference model. That bias
    flatters unseen words slightly, which is the safe direction: it can make
    the signal weaker, never more confident than the evidence.
    """
    alphabet = max(alphabet, 1)
    heldout = {t for t in keep if hash_fold(t) == 0}
    trained = keep - heldout
    if not heldout or not trained:
        # Too small to hold anything out. Decline to publish a scale rather
        # than publish one measured on the model's own memorised vocabulary.
        return 0.0, 0.0

    reference_counts: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for token in trained:
        weight = unigram_counts[token]
        for context, nxt in _char_ngrams(token):
            reference_counts[context][nxt] += weight
    prepared = {
        context: {"counts": dict(followers), "total": sum(followers.values())}
        for context, followers in reference_counts.items()
    }

    weight_total = 0.0
    weighted_sum = 0.0
    weighted_square = 0.0
    for token in heldout:
        weight = unigram_counts[token]
        value = _char_logprob_per_char(token, prepared, alphabet)
        weight_total += weight
        weighted_sum += weight * value
        weighted_square += weight * value * value
    if weight_total <= 0:
        return 0.0, 0.0
    mean = weighted_sum / weight_total
    variance = max(weighted_square / weight_total - mean * mean, 0.0)
    ## A floor on the spread so a degenerate corpus (one word repeated)
    ## cannot produce a zero denominator and turn every comparison into a
    ## division by nothing.
    return mean, max(math.sqrt(variance), 0.05)


def hash_fold(token: str) -> int:
    """Stable fold assignment for held-out estimation.

    Python's hash() is salted per process, so a rebuild of the same corpus
    would draw a different split and produce a different artifact. md5 of the
    token is not cryptography here, it is determinism.
    """
    import hashlib

    digest = hashlib.md5(token.encode("utf-8")).digest()
    return digest[0] % HELDOUT_FRACTION


def build(corpus: Path, family: str, max_lines: int | None) -> dict:
    unigram_counts: dict[str, int] = defaultdict(int)
    bigram_counts: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    continuation: dict[str, set] = defaultdict(set)
    ## Character-level context -> next-character counts, for the OOV backoff.
    ## Accumulated only for latin: the cjk tokenizer already emits one
    ## codepoint per token, so its unigram model IS a character model and an
    ## unseen character has nothing finer to back off to.
    char_counts: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    char_alphabet: set[str] = set()
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
                if family == "latin":
                    for context, nxt in _char_ngrams(token):
                        char_counts[context][nxt] += 1
                        char_alphabet.add(nxt)
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

    ## Add-one smoothed conditional counts. The character model is only ever
    ## consulted for a token the word model already failed on, so its job is
    ## to separate "unseen but well-formed" from "unseen and impossible" --
    ## not to be a good generative model in its own right, which is what
    ## would justify a costlier smoothing here.
    charmodel: dict = {}
    if char_counts:
        counts = {
            context: {"counts": dict(followers), "total": sum(followers.values())}
            for context, followers in char_counts.items()
        }
        mean, spread = _char_reference(len(char_alphabet), unigram_counts, keep)
        charmodel = {
            "order": CHAR_ORDER,
            "boundary": CHAR_BOUNDARY,
            "alphabet_size": len(char_alphabet),
            "counts": counts,
            ## What a per-character score MEANS in this corpus. Without a
            ## reference point the scorer has a number and no scale: -1.4 per
            ## character is unremarkable for English and alarming for a
            ## language with longer words and a smaller alphabet. Recording
            ## the corpus's own distribution is what lets the scorer say
            ## "better formed than typical" rather than guessing a threshold.
            "mean_logprob_per_char": mean,
            "spread_logprob_per_char": spread,
        }

    return {
        "family": family,
        "discount": DISCOUNT,
        "unigram": unigram,
        "bigram": bigram,
        "charmodel": charmodel,
        "manifest": {
            "corpus": str(corpus),
            "corpus_bytes": corpus.stat().st_size,
            "lines_used": lines,
            "types": len(unigram),
            "contexts": len(bigram),
            "char_contexts": len(charmodel.get("counts", {})),
            "char_order": CHAR_ORDER if charmodel else None,
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
