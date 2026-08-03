## 🍢 build_kenlm_models — versioned n-gram artifacts for OCR rescoring
## vieuxtiful
"""
Builds the ``latin.klm`` / ``cjk.klm`` artifacts KenLMScoringProvider loads.

Deliberately takes a LOCAL corpus path and never downloads anything. That
is the same contract every other provider in language_models.py honours,
and it is also what makes a build reproducible: the manifest written
beside each model records exactly which corpus produced it.

usage (from repo root):
  .venv/Scripts/python scripts/build_kenlm_models.py \\
      --family latin --corpus D:/corpora/frwiki.txt --out models/kenlm

  .venv/Scripts/python scripts/build_kenlm_models.py \\
      --family cjk --corpus D:/corpora/zhwiki.txt --out models/kenlm

then point ToFU at the result:
  set TOFU_KENLM_DIR=models/kenlm

## Getting a corpus

Any plain-text file, one document or sentence per line. For Wikipedia:

  1. Download a dump: https://dumps.wikimedia.org/<lang>wiki/latest/
     <lang>wiki-latest-pages-articles.xml.bz2
     (frwiki ~4 GB, enwiki ~20 GB, zhwiki ~2.5 GB compressed)
  2. Extract plain text with WikiExtractor:
     python -m wikiextractor.WikiExtractor <dump>.xml.bz2 -o - --json \\
       | python -c "import sys,json;[print(json.loads(l)['text']) for l in sys.stdin]" > frwiki.txt

A model does NOT need the whole dump. A few hundred MB of text produces a
usable 5-gram for rescoring; the marginal value of the rest is small next
to the training cost.

## Prerequisites

``lmplz`` and ``build_binary`` from KenLM, on PATH or via --kenlm-bin.
They are C++ programs and are NOT what ``pip install kenlm`` provides --
that ships the scoring module only. Build them once:

  git clone https://github.com/kpu/kenlm && cd kenlm
  cmake -B build -S . && cmake --build build -j

On Windows this wants a C++ toolchain and Boost; WSL is usually the
shorter path. The script checks for the binaries up front and explains
rather than failing halfway through a multi-GB pass.
"""

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tofu.layers.language_models import tokenize_for_lm  # noqa: E402

FAMILIES = ("latin", "cjk")


def _resolve_tool(name: str, kenlm_bin: str | None) -> str | None:
    if kenlm_bin:
        candidate = Path(kenlm_bin) / name
        for path in (candidate, candidate.with_suffix(".exe")):
            if path.is_file():
                return str(path)
        return None
    return shutil.which(name)


def normalize(corpus: Path, family: str, out: Path, max_lines: int | None) -> dict:
    """Stream the corpus through the SAME tokenizer scoring will use.

    Streamed rather than read whole: these corpora are measured in
    gigabytes, and holding one in memory to tokenize it would put the
    build out of reach of the machine it is most useful on.
    """
    lines = tokens = 0
    started = time.time()
    with corpus.open("r", encoding="utf-8", errors="replace") as src, \
         out.open("w", encoding="utf-8") as dst:
        for raw in src:
            if max_lines is not None and lines >= max_lines:
                break
            pieces = tokenize_for_lm(raw, family)
            if not pieces:
                continue
            dst.write(" ".join(pieces))
            dst.write("\n")
            lines += 1
            tokens += len(pieces)
            if lines % 500_000 == 0:
                print(f"  normalized {lines:,} lines / {tokens:,} tokens", flush=True)
    return {
        "lines": lines, "tokens": tokens,
        "seconds": round(time.time() - started, 1),
    }


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return f"sha256:{digest.hexdigest()}"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--family", required=True, choices=FAMILIES)
    ap.add_argument("--corpus", required=True, help="plain-text corpus, one doc/sentence per line")
    ap.add_argument("--out", default=str(ROOT / "models" / "kenlm"))
    ap.add_argument("--order", type=int, default=5, help="n-gram order (default 5)")
    ap.add_argument("--prune", default="0 0 1", help="lmplz --prune (default '0 0 1')")
    ap.add_argument("--memory", default="40%", help="lmplz -S memory budget")
    ap.add_argument("--max-lines", type=int, default=None, help="cap corpus lines (for a quick model)")
    ap.add_argument("--kenlm-bin", default=None, help="directory holding lmplz/build_binary")
    ap.add_argument("--keep-arpa", action="store_true")
    args = ap.parse_args()

    lmplz = _resolve_tool("lmplz", args.kenlm_bin)
    build_binary = _resolve_tool("build_binary", args.kenlm_bin)
    if not lmplz or not build_binary:
        print(
            "lmplz/build_binary not found.\n"
            "These are KenLM's C++ tools and are NOT installed by `pip install kenlm`,\n"
            "which ships only the scoring module. Build them once:\n"
            "  git clone https://github.com/kpu/kenlm && cd kenlm\n"
            "  cmake -B build -S . && cmake --build build -j\n"
            "then re-run with --kenlm-bin <kenlm>/build/bin (or put them on PATH).",
            file=sys.stderr,
        )
        return 2

    corpus = Path(args.corpus)
    if not corpus.is_file():
        print(f"corpus not found: {corpus}", file=sys.stderr)
        return 2

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    normalized = out_dir / f"{args.family}.tokens.txt"
    arpa = out_dir / f"{args.family}.arpa"
    model = out_dir / f"{args.family}.klm"

    print(f"== normalizing {corpus} for '{args.family}' ==", flush=True)
    stats = normalize(corpus, args.family, normalized, args.max_lines)
    print(f"  {stats['lines']:,} lines, {stats['tokens']:,} tokens in {stats['seconds']}s")
    if not stats["lines"]:
        print("corpus produced no usable lines", file=sys.stderr)
        return 1

    print(f"== lmplz -o {args.order} ==", flush=True)
    # discount_fallback keeps small/skewed corpora from aborting Kneser-Ney
    # estimation, which is the usual failure on a capped or single-domain
    # corpus and is otherwise a very late, very confusing crash.
    cmd = [
        lmplz, "-o", str(args.order), "-S", args.memory,
        "--discount_fallback", "--text", str(normalized), "--arpa", str(arpa),
    ]
    if args.prune.strip():
        cmd += ["--prune", *args.prune.split()]
    completed = subprocess.run(cmd)
    if completed.returncode != 0:
        return completed.returncode

    print("== build_binary ==", flush=True)
    completed = subprocess.run([build_binary, str(arpa), str(model)])
    if completed.returncode != 0:
        return completed.returncode

    manifest = {
        "family": args.family,
        "order": args.order,
        "prune": args.prune,
        "corpus": str(corpus),
        "corpus_bytes": corpus.stat().st_size,
        "lines": stats["lines"],
        "tokens": stats["tokens"],
        "max_lines": args.max_lines,
        "tokenizer": "tofu.layers.language_models.tokenize_for_lm",
        "model_bytes": model.stat().st_size,
        "checksum": sha256(model),
        "built": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    (out_dir / f"{args.family}.manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8",
    )

    if not args.keep_arpa:
        arpa.unlink(missing_ok=True)
    normalized.unlink(missing_ok=True)

    print(f"\nwrote {model} ({manifest['model_bytes'] / 2**20:.0f} MiB)")
    print(f"      {out_dir / f'{args.family}.manifest.json'}")
    print(f"\npoint ToFU at it:  TOFU_KENLM_DIR={out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
