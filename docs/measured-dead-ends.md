# Measured dead ends

Interventions that were built, measured against the eleven ground-truth
fixtures, and rejected on the numbers. Each entry says what was tried, what it
scored, and where the detail lives.

The point of this file is to stop the same work being restarted. Several of
these look obviously correct in the abstract — that is exactly why they were
attempted, and why the measurement is worth keeping.

**Baseline they are measured against:** `detection-2026-08-09-scene-guide-only`
(`tests/regression/baselines/detection.json`) — mean best IoU 0.738, matched
51/72, 134 candidates, engine `hybrid`, per-fixture source-language hints.

Reproduce any row with `scripts/eval_detect.py`; the localization metric is
`mean_best_iou` with its matched / near-miss / undetected split.

---

## Detection

| Intervention | Result | Detail |
|---|---|---|
| `mag_ratio` 1.5–3.0 | japan-street −0.143, la-bastille −0.061, ~1.8× wall clock. 1.5 flat-to-worse, 2.0 helps only gemini-street and the curve is non-monotonic (2.5 < 2.0 and < 3.0), so the single-fixture gain is within noise. | `cicerone.py` module note |
| `rotation_info=[90,270]` | Zero benefit, recorded before this work and re-confirmed. | `cicerone.py` module note |
| `link_threshold` sweep | Mean IoU **identical to three decimals** across 0.10–0.70 end to end, while the raw detector output moves 25→49 boxes. The multipass union and column merge absorb it entirely. | `eval_detect.py --link-threshold` |
| Relaxing the column-merge gates | Monotonically worse: colour gate off 0.365, gap ×2 0.362, both 0.332, against 0.381 as shipped. `merge_vertical_columns` is not under-merging. | ablation, gemini-street |
| Removing the scene pre-pass entirely | 0.678 < 0.710. It helps 7 fixtures and hurts 3; only its **veto** was the defect. | `cicerone.SCENE_FILTER_VETOES` |
| PaddleOCR DB as a drop-in detector | 42/72 matched vs CRAFT's 51/72, near-miss bucket *grew* 17→21, ~2× slower, never better on any fixture. Not a test of DB's polygons — rectangle GT and bbox IoU cannot see them. | `eval_detect.py --engine paddleocr` |
| Zoom inflation gate (`MAX_ZOOM_INFLATION`) | 0.706 < 0.738, matched 51→48. **No threshold window exists** — decolonisons degrades at 8.0× before la-bastille improves at 3.5×. | `union_prefer_primary`, in-place note |
| Family B expansion **as a recall mechanism** | Geometric oracle 24/33 matched, score-guided oracle **19/33 — zero gain**. The five recoverable gemini columns have no CRAFT region-map mass in the bands that would reach them. Detector belief failure, not selector failure. | `eval_expansion_oracle.py` |

## Measurement

| Intervention | Result | Detail |
|---|---|---|
| A Guided **localization oracle** ("what if the user drew every box perfectly?") | **Not answerable in this architecture.** Three attempts: annotated crops at `pad=0` (42.3%), the backend's own crop path (57.1%), and `cicerone.detect(seed_detections=...)` running the full pipeline (64.0%) — all BELOW the Auto arm's 67.2%, which an oracle cannot be. The first two measured the crop path rather than the pipeline. The third measured the pipeline, and still failed: the merge and assembly stages rewrite supplied geometry (three seeded boxes come out as one region on `cjk-vertical-menu`), and those same stages are what make text readable. Disabling them degrades recognition; leaving them on modifies the geometry. **Localization and recognition are entangled**, so no configuration replaces localization alone. | `docs/gate2-status.md`, `layers/cicerone.detect(seed_detections=)` |

`seed_detections` is kept: it is the right seam, and it is what a
geometry-preserving recognition path would build on. What is rejected is the
claim that any current configuration yields a ceiling.

## Glyph identity from ground truth

| Intervention | Result | Detail |
|---|---|---|
| **Rank aggregation across faces** to recover the right string without knowing the face | **Worse than doing nothing.** Borda 10.7% and per-face z-normalisation 8.9%, against a 16.1% argmax baseline and a 46.4% face-oracle, on 56 misread regions. Most faces are the wrong face and rank strings near-arbitrarily; averaging over fourteen drowns the one or two that fit. | `docs/phase1-baseline.md` |
| **Identifying the face first, then matching** | **Circular, not merely hard.** `font_matching.local_match` ranks faces by rendering the region's KNOWN text in each one — and the text is exactly what a glyph matcher is trying to recover. | `font_matching.local_match` |

Both were tried because both are cheaper than training an encoder. Their
failure is what gives a typeface-invariant embedding a measured reason to
exist: a representation in which the face does not matter is what breaks the
circularity, and the 46.4% oracle is the target it has to approach.

## Recognition

| Intervention | Result | Detail |
|---|---|---|
| Charset allowlist re-read (`winnow`) | Premise disproven: a joint `(ru,en)` reader decodes Cyrillic **correctly**. `BYAYLIEE` came from an English-only reader, not from a joint charset picking Latin twins. Layer deleted; `textmatch.twin_share` kept. | removed; `twin_share` docstring |
| `decoder='beamsearch'` | Ten fixtures byte-identical, decolonisons NED 0.017→0.164, corpus mean 0.146→0.159. EasyOCR's beam decoder **truncates long reads** and exposes no top-k, so there is no lattice to fuse a language model into. | `EasyOCRBackend.decoder` note |
| Noisy-channel corrector ("course 5") | Not built. Step 0 showed all four "truncation" cases have the missing ink **outside the crop** — `劇場通り`'s り is twelve pixels below where the box ends — so they are extent failures, not recognition failures. LM-addressable set collapsed to one region (`7789`→`1789`). | this file; step-0 geometry |

---

## Two recurring traps

**Measuring a configuration the product does not run.** The first baseline used
`--engine easyocr` against the server's `auto`, then no language hint against
the server's project hint. Correcting the second alone moved five fixtures —
russian-billboard-2 from NED 0.642 to 0.014 — with no code change. A separate
attribution pass compared against a permissive `getDetBoxes` re-run and
reported six regions as destroyed by the pipeline; the production ancestry says
**one** is. That episode is why `okara.CandidateGraph` carries `run_kind` and
`DetectorConfig`: *raw* is not a sufficient provenance label.

**Fitting a threshold on the evaluation set.** `TWIN_FLOOR` was set from three
points and `MAX_ZOOM_INFLATION` from one fixture; the second shipped and
regressed the corpus. Any new constant needs held-out evidence or an explicit
note that it does not have any.
