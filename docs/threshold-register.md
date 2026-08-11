# Threshold register

Every hard-coded numeric constant in the detection path, with an honest label
for where its value came from.

**Why this exists.** The suspicion that hard thresholds are driving detection
outcomes is correct in the sense that matters: there are roughly forty of them,
they interact, and until now nothing recorded which ones were *measured* and
which were simply chosen. That distinction is the whole content of this file.
A number that was fitted on evidence and a number somebody picked because it
looked reasonable are indistinguishable in source, and they warrant completely
different amounts of trust when a result disappoints.

**Provenance labels**, exactly one per row:

| Label | Meaning |
|---|---|
| `measured` | A recorded sweep or ablation exists; the row cites it. |
| `reasoned` | Derived from a stated observation about the data (e.g. "every GT region is ≥500px²"), but not swept. |
| `inherited` | A library default carried forward without examination. |
| `invented` | Chosen by judgement. No sweep, no cited observation. |

**The rule this file enforces.** No constant moves without held-out evidence,
or an explicit note in the changelog that it has none. This project has already
paid for the alternative twice: `MAX_ZOOM_INFLATION` was fitted on a single
fixture, shipped, and regressed the corpus (0.738 → 0.706); `TWIN_FLOOR` was set
from three points. Both are recorded in `docs/measured-dead-ends.md`.

**What this file does not do.** It changes no value. Part A of the guided-capture
plan is instrumentation and honesty; retuning is separate work that needs the
attribution in `docs/detection-attribution-report.md` to say which constants are
actually killing candidates before anything is swept.

---

## Detection — `src/tofu/layers/cicerone.py`

| Constant | Value | Provenance | Basis / note |
|---|---:|---|---|
| `PASS_THRESHOLDS` | 3 rungs | `invented` | The multipass ladder. Never swept as a set; `link_threshold` alone was swept and found **inert** (mean IoU identical to 3 d.p. across 0.10–0.70). |
| `LATIN_POOL_MIN_CONFIDENCE` | 0.5 | `invented` | — |
| `LATIN_POOL_RESCUE_MIN_CONFIDENCE` | 0.35 | `invented` | — |
| `MIN_DIACRITIC_WORD_LEN` | 3 | `reasoned` | Short tokens carry too little evidence to accuse of a language. |
| `DECLARED_LANGUAGE_MARGIN` | 2.0 | `invented` | — |
| `SCENE_FILTER_VETOES` | off | **`measured`** | Veto removal: 0.710 → 0.738 mean IoU, 47 → 51 matched, at 99 → 134 candidates and review rate 0.444 → 0.622 on gemini-street. A deliberate recall-for-reviewer-time trade. |
| `COLUMN_MAX_ASPECT` | 1.6 | `reasoned` | Members must be char-like/tall, not wide lines. |
| `COLUMN_X_ALIGN` | 0.5 | `invented` | Relaxing the column gates was swept and was **monotonically worse** (0.365 / 0.362 / 0.332 vs 0.381 shipped), so the gate is not under-merging — but the specific value was not fitted. |
| `COLUMN_WIDTH_RATIO` | 1.7 | `invented` | as above |
| `COLUMN_MAX_GAP` | 0.8 | `invented` | as above |
| `COLUMN_COLOR_MAX_DIST` | 90.0 | `invented` | Colour gate off scored 0.365 vs 0.381; the threshold itself unswept. |
| `COLUMN_LATIN_MIN_CONF` / `_MIN_LEN` | 0.5 / 2 | `invented` | — |
| `VERTICAL_STACK_MIN_ASPECT` | 3.0 | `reasoned` | Height ≥ 3× width is too tall for one glyph. |
| `MIN_BAND_HEIGHT_PX` | 12 | `reasoned` | Below this a band cannot hold a legible glyph. |
| `ROW_BASELINE_ALIGN` | 0.5 | `invented` | — |
| `ROW_HEIGHT_RATIO` | 1.7 | `invented` | — |
| `ROW_MAX_GAP` | 0.6 | **`measured`**, narrow | Source note records the separating cases and a margin of only ~13% either side. Flagged: a layout with tighter columns or looser word spacing than the corpus defeats it. |
| `ROW_MIN_MEMBERS` | 2 | `reasoned` | A row of one is not a row. |
| `ROW_MERGE_MIN_CONFIDENCE` | 0.5 | `invented` | — |
| `MIN_DISAMBIGUATION_EVIDENCE` | 2 | `reasoned` | One hit flipped a scene on "ET"; the second hit is the guard. |
| `MIN_KANA_CONFIDENCE` | 0.3 | `reasoned` | A near-zero-confidence misread of a blurry crop can shape-match a kana glyph and must not be a definitive Japanese signal. |
| `MIN_SYMBOL_JUNK_AREA` | 450 | **`measured`** | Every GT region across the fixtures is ≥500px²; measured false positives were 100px² and 440px². Fitted on the *evaluation* corpus — no held-out evidence. |
| `SURFACE_PROBE_MAX` | 8 | `invented` | Runtime cap. |
| `PADDLE_RESCUE_CONF_FLOOR` | 0.6 | `invented` | — |
| `PADDLE_OVERLAP_WIN_FLOOR` | 0.5 | `invented` | — |
| `PADDLE_ESCALATED_DROP_SCORE` | 0.15 | `inherited` | Below DBNet's default, deliberately; the specific value is Paddle-side convention. |
| `PADDLE_ESCALATED_UNCLIP_RATIO` | 1.9 | `inherited` | Source note says DBNet's default under-clipped. |
| `SURFACE_COVERAGE_FLOOR` | 0.85 | `invented` | Truncation heuristic against a surface's dominant-axis extent. |
| `SUBDIVIDE_MAX_DIM` | 120 | `invented` | — |
| `SUBDIVIDE_OVERLAP_PX` | 10 | `reasoned` | So a sign spanning a tile edge is not cut. |
| `SUBDIVIDE_SPLIT_TOLERANCE` | 1.5 | `invented` | — |
| `LOAF_CONTAINMENT` | 0.8 | `reasoned` | Deliberately below `ZOOM_FRAGMENT_CONTAINMENT`'s 0.9: a zoom box maps back from an upscaled crop, and at 0.9 the rounding disowns a genuine crumb of its own line. |
| `LOAF_CRUMB_MIN_CONFIDENCE` | 0.3 | `invented` | — |
| `LOAF_SPAN_SIMILARITY` | 0.6 | `invented` | — |
| `LOAF_DUPLICATE_SIMILARITY` | 0.9 | `invented` | — |
| `LOAF_DUPLICATE_RESCUE_CEILING` | 0.5 | `invented` | — |
| `ZOOM_MAX_SURFACE_FRAC` | 0.5 | `reasoned` | A surface larger than half the frame *is* the frame; re-detecting buys nothing. |
| `ZOOM_MAX_SURFACES` | 14 | `invented` | Runtime cap against pathological surface counts. |
| `ZOOM_SCALE` / `ZOOM_PAD` | 2 / 8 | `invented` | — |
| `ZOOM_FRAGMENT_CONTAINMENT` | 0.9 | `invented` | — |
| `MAX_ZOOM_INFLATION` | *removed* | **`measured`** | Fitted on one fixture, shipped, **regressed the corpus** 0.738 → 0.706, matched 51 → 48. No threshold window exists. The cautionary case for this whole file. |
| `EDGE_RESCUE_PAD_RATIO` | 0.16 | `reasoned` | A blanket pad merged `AVENUE` with `de la RÉPUBLIQUE`, halving recall on that fixture; the per-region evidence gate is the fix. |
| `EDGE_RESCUE_CONF_SLACK` | 0.10 | `invented` | — |
| `FRAGMENT_OVERLAP` | 0.5 | `invented` | — |
| `mag_ratio` | 1.0 (default) | **`measured`** | 1.5–3.0 swept: japan-street −0.143, la-bastille −0.061, ~1.8× wall clock, curve non-monotonic. Rejected. |
| `rotation_info` | unset | **`measured`** | `[90,270]` gave zero benefit, twice. |

## Recognition — `src/tofu/utils/textmatch.py`

| Constant | Value | Provenance | Basis / note |
|---|---:|---|---|
| `FUZZY_MATCH_THRESHOLD` | 0.85 | `inherited` | "plan-specified minimum edit-distance ratio". Now also gates Guided reconciliation (`layers/aboyeur.py`), where it may only produce `review`, never `complete` — precisely because it is uncalibrated. |
| `TWIN_FLOOR` | — | **`measured`**, thin | Set from three points. Recorded in the dead-ends file as a fitting caution. |

## Verification — `src/tofu/layers/verify.py`

| Constant | Value | Provenance |
|---|---:|---|
| `OCR_WEIGHT` / `SSIM_WEIGHT` / `STYLE_WEIGHT` / `GARNISH_WEIGHT` | 0.7 / 0.3 / 0.2 / 0.15 | `invented` |
| `CONTENT_PASS_THRESHOLD` / `CONTENT_REVIEW_THRESHOLD` | 0.85 / 0.50 | `invented` |
| `RESIDUAL_PENALTY_THRESHOLD` | 0.3 | `invented` |
| `OCR_CONFIDENCE_REVIEW` | 0.50 | `invented` |
| `COLOR_DELTA_E_SCALE` | 40.0 | `invented` |
| `QUAD_OVERFLOW_TOLERANCE` | 0.02 | `invented` |

## Reconstruction — `src/tofu/layers/cleanse.py`

| Constant | Value | Provenance | Note |
|---|---:|---|---|
| `DILATE_ITER` | 3 | `reasoned` | ~3px, to catch anti-aliased glyph edges. |
| `FEATHER_PX` | 3.0 | `invented` | |
| `RING_PX` | 14 | `invented` | |
| `MIN_RING_PIXELS` | 20 | `reasoned` | Below this a gradient fit degrades to flat fill. |
| `MIN_MASK_PIXELS` | 6 | `reasoned` | A tiny Otsu component is not safe evidence alone. |
| `GROUP_GAP_PX` | 12 | `invented` | |
| `TELEA_RADIUS_DEFAULT` | 3 | `inherited` | OpenCV convention. |

---

## What the tally says

Counting the detection table alone: **4 `measured`**, 10 `reasoned`,
3 `inherited`, and **the clear majority `invented`**.

That is the finding, and it is worth more than any individual retune. It does
not mean the values are wrong — most were chosen by someone looking at real
output, and the corpus result is what it is *with* them. It means that when a
detection result disappoints, "a threshold is miscalibrated" is a hypothesis
with roughly thirty untested candidates behind it, and picking one to nudge is
guessing unless the lineage graph says that constant is what killed the
candidate.

Which is exactly what `GET /api/manifest/{id}/detection-attribution` and
`scripts/eval_detector_evidence.py` are for: instrument first, sweep only what
the instrument implicates, and hold out evidence before shipping the change.
