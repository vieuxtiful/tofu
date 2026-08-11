# Gate 2 status — does Guided locate more than Auto?

Measured 2026-08-11 on `guided-corpus-v1` (46 assets, 189 requested
occurrences). Evidence: `evidence/guided-paired-v1.json`,
`evidence/guided-gate2-mcnemar.json`.

**Gate 2 is NOT certified, for two independent reasons — one statistical, one
procedural. The underlying effect, however, is real and unusually clean.**

---

## The result

Both arms scored from **one detection pass per asset**, so every occurrence is a
matched pair and the comparison is paired by construction rather than by
assertion. Re-detecting per arm would have put OCR variance into the very
difference the gate measures.

| Split | occurrences | Auto | Guided | margin | McNemar (exact) |
|---|---:|---:|---:|---:|---|
| **FULL** | 189 | 67.2% (127) | **74.6%** (141) | **+7.4 pts** | **p = 0.00012** ✔ |
| dev | 161 | 64.6% | 72.0% | +7.5 | p = 0.00049 ✔ |
| holdout | 28 | 82.1% | 89.3% | +7.1 | p = 0.5 ✘ |

### The discordant pairs are the story

```
both found      127
neither found    48
guided only      14      <- every disagreement
auto only         0      <- without exception
```

**Guidance never lost an occurrence Auto had found.** Fourteen wins, zero
losses. That one-sidedness is why p = 0.00012 on only 14 discordant pairs: under
"guidance changes nothing" each disagreement is a coin flip, and fourteen heads
in a row is not a coin.

It also means the two single-arm percentages badly understate the evidence.
A ±15.5pp interval on a holdout point estimate says almost nothing; the paired
test on the same data says the effect is real at p < 0.001.

## Why it is still not certified

**1. The margin is 0.6 points short.** Gate 2 asks for +8.0; the full corpus
gives +7.4. The effect is significant *and* below the declared bar — those are
different questions and both have to be answered. The bar was written down in
advance, so it stands.

**2. The corpus cannot certify anything yet.** `review_status:
machine_derived`. Blocks were derived from existing annotations by a script, so
a significant margin here is evidence about *the derivation rule*, not about
guidance. `docs/guided-corpus-review-protocol.md` is the plan for fixing this;
it is calendar time, not engineering time, and should be started now.

Both harnesses refuse on their own, independently. Neither refusal should be
weakened to unblock a schedule.

## Per-stratum

| stratum | expected | Auto | Guided | margin |
|---|---:|---:|---:|---:|
| `japanese_horizontal` | 23 | 65.2% | 78.3% | **+13.0** |
| `mixed_script_numeric` | 55 | 78.2% | 87.3% | **+9.1** |
| `nonlinear_irregular` | 41 | 31.7% | 39.0% | +7.3 |
| `latin_horizontal` | 39 | 92.3% | 97.4% | +5.1 |
| `rtl_bidi` | 22 | 59.1% | 63.6% | +4.5 |
| `japanese_vertical` | 9 | 77.8% | 77.8% | **+0.0** |

Two strata already clear +8 on their own. `japanese_vertical` gains nothing —
but n=9, so that is one or two occurrences either way and should not be read as
a finding.

> **Do not compare these to the 2026-08-10 figures.** That baseline was 44
> assets / 182 occurrences and predates this branch's detection changes; the
> corpus is now 46 / 189. Stratum numbers that look like large movements
> (`japanese_vertical` in particular) are comparing two different measurements.
> A like-for-like statement needs both arms re-run on one frozen corpus, which
> is what this file is.

## What would close the 0.6-point gap

Unknown, and deliberately not guessed at. The relevant fact is the **48
occurrences neither arm found** — that is where any further margin has to come
from, and nothing here says whether they are locatable at all.

### The oracle arm: three runs, and a negative result

| arm | recall | what it measured |
|---|---:|---|
| Auto | 67.2% | |
| Guided | 74.6% | |
| oracle v1 — crops at `pad=0` | 42.3% | a padding bug |
| oracle v2 — the backend's crop path | 57.1% | the crop path, not the pipeline |
| oracle v3 — `seed_detections`, full pipeline | **64.0%** | **still not a ceiling** |

**v1 -> v2 was a real bug worth 14.8 points.** `pad=0` cropped tighter than the
pipeline ever does; `crop_legibility` in `eval_detector_evidence.py` had already
learned to use the backend's own crop path for the same reason.

**v2 -> v3 moved the seam into the layer.** `cicerone.detect(seed_detections=...)`
replaces the detector and keeps the pipeline's own recognition -- multipass,
edge rescue, polish, the correction layers. On two RTL assets the arm went
53.8% -> 76.9%, from absurd to plausible.

**And v3 is still below Auto. That is the finding.**

| stratum | expected | Auto | oracle v3 | delta |
|---|---:|---:|---:|---:|
| `japanese_vertical` | 9 | 77.8% | 44.4% | **-33.3** |
| `latin_horizontal` | 39 | 92.3% | 84.6% | -7.7 |
| `mixed_script_numeric` | 55 | 78.2% | 72.7% | -5.5 |
| `rtl_bidi` | 22 | 59.1% | 59.1% | +0.0 |
| `japanese_horizontal` | 23 | 65.2% | 69.6% | **+4.3** |
| `nonlinear_irregular` | 41 | 31.7% | 36.6% | **+4.9** |

The obvious explanation -- that disabling `zoom` and `vertical_split` crippled
vertical CJK -- was tested and is WRONG. Re-running the three vertical assets
with those stages back on changes nothing (2/4 and 0/3 either way).

The probe showed the real mechanism: on `cjk-vertical-menu`, **three seeded
boxes come out as one region.** The pipeline's merge and assembly stages
(`merge_detections`, `merge_vertical_columns`, `merge_baseline_runs`,
`split_tall_detections`) rewrite the geometry they are given. Supplying boxes
does not hold them fixed.

**So localization and recognition are not separable in this architecture.** The
stages that reshape geometry are the same stages that make text readable -- a
column of characters has to be merged before it reads as a word. Disable them
and recognition degrades; leave them on and the supplied geometry is modified
before it is scored. There is no configuration that replaces localization
alone.

That makes "what if the user drew every box perfectly?" a question this
codebase cannot currently answer, and three successive attempts to answer it
have each been better-founded than the last and still wrong. Recorded in
`docs/measured-dead-ends.md`. Answering it properly needs a geometry-preserving
recognition path -- a design change, not a harness flag -- and that should be
justified by something other than curiosity before anyone builds it.

**What survives for Gate 2:** on `nonlinear_irregular` and
`japanese_horizontal` the oracle beats Auto by only +4.9 and +4.3 while Guided
beats it by +7.3 and +13.0. Guided already exceeds what perfect localization
delivers on those strata, which is evidence that guidance is contributing
something other than better boxes -- and no evidence at all that the remaining
0.6-point gap is closable by localization.

## Reproducing

```bash
.venv/Scripts/python scripts/eval_guided_corpus.py --arm both --out evidence/guided-paired-v1.json
```

```bash
.venv/Scripts/python scripts/compare_guided_arms.py evidence/guided-paired-v1.json
```

Roughly 26 minutes, real OCR, one detection pass per asset shared by both arms.
