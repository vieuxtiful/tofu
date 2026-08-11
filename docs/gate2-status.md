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

### The oracle arm: two runs, and what they establish

| arm | recall | note |
|---|---:|---|
| Auto | 67.2% | |
| Guided | 74.6% | |
| oracle **v1** (`pad=0`) | **42.3%** | measurement bug |
| oracle **v2** (pipeline crop path) | **57.1%** | still not a ceiling |

**v1 was wrong, and the bug is worth recording.** It read annotated crops through
`detect_in_regions(..., pad=0)`, cropping tighter than the pipeline ever does --
while `crop_legibility` in `eval_detector_evidence.py` had already learned to use
the backend's own crop path (pad 4, upscale to `MIN_CROP_HEIGHT`) for exactly
this reason. Fixing it moved the arm **42.3% -> 57.1%: a 14.8-point measurement
artefact.**

**v2 is still below both real arms, and that is structural rather than a bug.**
No crop-based oracle can bound this pipeline, because the crop path *is* a weaker
recogniser than the pipeline: Auto and Guided get the multipass ladder, zoom,
surface probes, edge rescue and the correction layers, and a per-crop read gets
none of them. The remaining ~10-point gap to Auto is that machinery.

Per stratum, oracle minus Auto:

| stratum | expected | Auto | oracle v2 | delta |
|---|---:|---:|---:|---:|
| `japanese_vertical` | 9 | 77.8% | 22.2% | **-55.6** |
| `japanese_horizontal` | 23 | 65.2% | 21.7% | **-43.5** |
| `rtl_bidi` | 22 | 59.1% | 40.9% | -18.2 |
| `latin_horizontal` | 39 | 92.3% | 84.6% | -7.7 |
| `mixed_script_numeric` | 55 | 78.2% | 80.0% | **+1.8** |
| `nonlinear_irregular` | 41 | 31.7% | 36.6% | **+4.9** |

Two things follow, and they point in opposite directions:

1. **On CJK, ToFU's recognition is carried almost entirely by the multipass and
   zoom passes.** A single crop read reaches 22% where the pipeline reaches
   65-78%. Any measurement that reads CJK from a crop -- including
   `crop_legibility` -- is therefore **materially pessimistic** and must not be
   quoted as evidence about what is legible.
2. **On `mixed_script_numeric` and `nonlinear_irregular` the oracle is a valid
   lower bound**, because there the crop path is comparable to the pipeline. And
   there it sits only **+1.8 and +4.9** above Auto. On `nonlinear_irregular` --
   the worst stratum in the corpus -- **perfect localization still leaves ~63% of
   requested occurrences unfound.**

Point 2 is the one that bears on the 0.6-point gap. Where the oracle is
trustworthy, handing the pipeline perfect boxes buys very little, so the
remaining loss there is **recognition, not localization**, and no amount of
guidance addresses it.

That is a partial answer, not the whole one. A fully valid oracle still needs
ground-truth geometry injected into the pipeline's OWN recognition path after
detection -- a `cicerone` change, not a harness change -- so that only
localization is replaced. Until then CJK and RTL remain unmeasured on this
question.

## Reproducing

```bash
.venv/Scripts/python scripts/eval_guided_corpus.py --arm both --out evidence/guided-paired-v1.json
```

```bash
.venv/Scripts/python scripts/compare_guided_arms.py evidence/guided-paired-v1.json
```

Roughly 26 minutes, real OCR, one detection pass per asset shared by both arms.
