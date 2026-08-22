# Channel envelope report — stage S2

2026-08-16. Corpus `corpus-v1` (11 fixtures, 72 annotated regions, 66 with
non-empty text). Artifact: `evidence/channel-envelope-v1.json`. Harness:
`scripts/eval_channel_envelope.py`.

This report answers the gating question of
`docs/vision-2-supervised-objective-plan.md` stage S2: **does the channel
family contain a materially better measurement for any stratum?** If it does
not, the learned channel selector of stage S7 has nothing to select.

Every figure below is normalized edit distance, `NED ∈ [0,1]`, lower is
better. `D⋆ = min_c D_{e,c}` is **oracle**: it is computed from the ground
truth, is a training and benchmarking envelope, and is never an inference-time
score. The guard is executable (`flight.oracle_envelope`), not editorial.

---

## 1. What was measured

Two probes per annotated region:

| Probe | What it is | Geometry |
|---|---|---|
| `region_crop` | `detect_in_regions` — the pipeline's own fixed crop path, with its upscale to `MIN_CROP_HEIGHT` | **handed the annotated box** |
| `full_pipeline` | the shipped `detect()`, matched back to the annotation at IoU ≥ 0.5 | must find the box itself |

These are not like for like, and the asymmetry is not a defect of the harness
— it is what the two channels are. A region the pipeline never proposed scores
1.0, because from the consumer's side an unfound region and an unreadable one
are equally absent. Every per-channel figure is therefore given twice: over
all regions, and over the subset the pipeline localized, where the comparison
is recognition against recognition.

Strata are derived from the **annotated text's codepoints**
(`SCRIPT_VOCABULARY = "envelope-script-v1"`), never from the fixture name. The
corpus's own declared strata are reported alongside. This immediately earned
its keep: `gemini-street` is declared `mixed_dense` and is in fact 13 Korean
regions, which the declared label hides.

---

## 2. Result

### Per script stratum

| stratum | n | localized | crop | pipeline | D⋆ | fixed gap | **fixed gap, localized only** |
|---|---:|---:|---:|---:|---:|---:|---:|
| cyrillic | 6 | 6 | 0.261 | **0.000** | 0.000 | 0.000 | 0.000 |
| han | 8 | 7 | 0.833 | **0.292** | 0.292 | 0.000 | 0.000 |
| hangul | 13 | 6 | 0.827 | 0.662 | 0.538 | 0.123 | **0.100** |
| japanese_or_han | 4 | 1 | 1.000 | **0.813** | 0.813 | 0.000 | 0.000 |
| latin | 31 | 23 | 0.498 | 0.388 | 0.355 | 0.033 | **0.002** |
| mixed | 2 | 2 | 0.450 | **0.200** | 0.200 | 0.000 | 0.000 |
| unscripted | 2 | 2 | **0.000** | 0.125 | 0.000 | 0.000 | 0.000 |
| **all** | **66** | **47** | 0.596 | 0.407 | 0.363 | 0.044 | **0.019** |

*Fixed gap* is how much the best single fixed channel gives up against the
envelope. It is the S7 statistic: zero means one channel already achieves the
best the family can measure, and a selector has nothing to select.

### The crop path is a garbling, and the ordering is measured

Recognition against recognition, on regions the pipeline localized:

| stratum | crop | pipeline | crop penalty |
|---|---:|---:|---:|
| han | 0.810 | 0.190 | **+0.619** |
| hangul | 0.792 | 0.267 | **+0.525** |
| japanese_or_han | 1.000 | 0.250 | **+0.750** |
| cyrillic | 0.261 | 0.000 | +0.261 |
| latin | 0.367 | 0.176 | +0.191 |

The Blackwell ordering the technical paper predicts is present and large. The
crop is not merely noisier than the full pipeline; on CJK and Hangul it
destroys most of the reading. `japanese_or_han` at crop NED 1.000 means the
crop path recovered *nothing usable* from any Japanese region in the corpus.

This retroactively prices the measurement-honesty invariant: **any past figure
computed from a per-crop read understated the shipped pipeline by roughly 0.5
to 0.75 NED on CJK strata.** That is not a rounding difference, it is the
difference between "unreadable" and "read correctly".

---

## 3. The decision: S7 is not licensed by this corpus

The apparent headroom shrinks under scrutiny, and it shrinks for two different
reasons that must not be confused.

**Latin's gap is a localization artifact.** 0.033 over all regions,
**0.0015** over localized ones. Regions the pipeline never proposed score 1.0
for it, while the crop path — handed the annotated box — sometimes scores
below that, opening a gap that looks like a channel-selection opportunity. At
inference there is no annotated box, so no selector can recover it. This is
exactly the category error the channel index exists to prevent, and it was
found by the index catching itself.

**Korean's gap survives, and is underpowered.** 0.123 over all regions,
**0.100** over the 6 the pipeline localized. It does not dissolve under the
localization filter: on regions the pipeline found, the two channels genuinely
win different regions. But n = 6 is below the preregistered
`EQUIV_MIN_PAIRS = 8`, from one fixture, in one script.

> **Verdict: defer S7.** The one stratum showing selectable headroom is a
> single fixture's worth of Korean signage, below our own power floor. The
> prerequisite for revisiting it is *more Korean and CJK regions*, not more
> channels and not a selector fitted to six regions.

The corpus-wide figure says the same thing more bluntly: **0.019 NED** of
selectable headroom across 47 localized regions. A learned reliability model,
a regret target, a mixture at inference, and a new acceptance gate would be
built to recover two hundredths of a NED point.

---

## 4. Equivalence classes: none certified

Fitted on the **dev split only**, criterion preregistered before the first
report was read (`EQUIV_MARGIN = 0.05`, `EQUIV_ALPHA = 0.05`,
`EQUIV_MIN_PAIRS = 8`; registered in `docs/threshold-register.md`).

| stratum | n | mean difference | p | verdict | ground |
|---|---:|---:|---:|---|---|
| latin | 19 | −0.177 | 0.004 | not_equivalent | margin exceeded, direction detected |
| han | 8 | −0.542 | 0.063 | not_equivalent | margin exceeded |
| hangul | 13 | −0.165 | 0.453 | not_equivalent | margin exceeded |
| cyrillic | 4 | — | — | underpowered | below min pairs |
| japanese_or_han | 4 | — | — | underpowered | below min pairs |
| mixed | 2 | — | — | underpowered | below min pairs |
| unscripted | 1 | — | — | underpowered | below min pairs |

**No stratum is certified equivalent.** Note what `not_equivalent` does and
does not mean: it is the answer to "may S4 tie these two channels?", and the
answer is no. It is *not* "certified different" — on `hangul` the effect
exceeds the margin while the sign test says the direction is not even
established. The ground is recorded per verdict in the artifact because the
two failures call for different follow-up: one needs a better channel, the
other needs more regions.

**Consequence for S4.** `R_cons` has no licensed cross-channel tie at this
corpus size. The consistency regularizer must therefore be restricted to
attempt variation *within* a channel — the blur, morphological, and typeface
variants of one observation — which is what the originally primed λ₃ term was
about. Tying `region_crop` to `full_pipeline` on any stratum here would train
the encoder to agree with a channel that is a strict garbling of the other.

---

## 5. Channel coverage, and a provenance gap this exposed

**All 47 localized reads trace to a registered route.** The 19 remaining
regions are unlocalized and correctly carry no route: there is no shipped read
to trace.

| route | n |
|---|---:|
| independent_verifier | 24 |
| paddle_rescue | 9 |
| detection_pass_1/2/3 | 7 |
| zoom | 3 |
| baseline_row_merge | 2 |
| composed_crop | 1 |
| vertical_column_merge | 1 |
| **untraced (localized)** | **0** |

Getting there took two fixes, and the first measurement of this column was
what exposed both.

**The lineage fallback.** The first run traced 13 of 13 Korean and 14 of 31
Latin regions to `unregistered`. `recognition_history` is seeded from the raw
detection's provenance, and a transform mints a *fresh* detection — so a read
produced by zoom or a merge carried no entry at all and was invisible to the
history walk, even though the transform is precisely the channel that produced
it. `flight.channel_for` now falls back to the lineage graph's re-measuring
transforms (`REMEASURING_TRANSFORMS`), decided by whether each transform
re-runs recognition per its own source. `merge_detections` and
`multipass_union` are excluded: both are box algebra over reads that already
existed, and crediting them would name a channel that never looked at a pixel.

**The provenance gap itself.** The fallback left 9 localized regions untraced,
every one CJK or Hangul. Instrumenting `build_manifest` on `japan-street` gave
the cause directly rather than by inference: **9 of 21 detections arrived with
no provenance after the zoom pass, and 22 of 26 after split/assembly.** Every
transform that emits a fresh detection was dropping it. `cicerone` now records
the read in the detection's own provenance at the same points it records the
lineage node (`_tag_transform_read`, called from `_record_raw` for
re-measuring origin stages, from `_record_derived`, from
`merge_vertical_columns`, and from the Paddle rescue). Provenance is written
*before* the graph check, because whether okara happens to be recording is not
a reason for a region to lose the name of the route that read it.

Two boundaries worth stating, both of which shaped the fix:

- The transform entry carries **no `candidate_text`**. In this schema that key
  means "an alternative reading that was considered", and
  `proof_runtime.candidate_pool` enrols it as a retrieval candidate. A
  transform's adopted re-read is not an alternative to the region's reading —
  it *is* the reading, already on `det.text` — so writing it there would enrol
  the region's own answer as a candidate against itself.
- The Paddle rescue's lineage stage stays `raw_craft`, because Paddle's
  detector output genuinely is a raw proposal. Its *recognition route* is
  `paddle_rescue`, because a read from a second, differently-architected
  engine is not the same measurement channel as an EasyOCR pass. Lineage and
  channel answer different questions and are allowed different answers.

Re-running the corpus after the fix: route coverage 38/47 → **47/47**, and
**every region's NED unchanged**, which is what makes it a provenance fix
rather than a behaviour change.

---

## 6. Limitations

- **Zoom and the merge routes are not separately probed.** They are named
  channels in the paper, but they are not invocable per region without
  reimplementing their trigger conditions, and a probe that fired on different
  regions than the pipeline does would measure the harness. They appear only
  where `full_pipeline` routed through them.
- **Paddle is not probed as a separate channel here.** It appears inside
  `full_pipeline` as `independent_verifier`, which is the single most common
  route in the corpus (24 of 38 traced reads).
- **A probe is not a channel.** `region_crop` is one route; `full_pipeline` is
  a mixture over whichever routes the pipeline took. The equivalence verdicts
  are therefore between *probes*. The `route_mixture` block in the artifact
  says how much of a mixture — and where one route dominates a stratum, that
  route is what S4 could eventually tie at channel granularity.
- **Nine of eleven fixtures carry partial annotations.** These are
  per-annotated-region measurements and imply nothing about precision.
- **Small strata.** Four of seven script strata have n < 8. The corpus was
  frozen for detection recall, not for channel comparison.
- The holdout split is included in the per-stratum curves and excluded from
  the equivalence fit. No threshold was tuned on any of it.
