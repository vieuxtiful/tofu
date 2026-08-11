# Detection attribution — what Auto misses, and why

Measured 2026-08-11. Part A of `guided-capture-and-detection-attribution-plan.md`.

**No constant was changed to produce this.** The point is to find out where
candidates die before anything is retuned, because this project has twice paid
for the other order: `MAX_ZOOM_INFLATION` was fitted on one fixture, shipped, and
regressed the corpus; and an attribution pass that compared against a detector
configuration the product does not run reported six pipeline-destroyed regions
where the production ancestry has one.

---

## 1. The headline, stated plainly

**On the evidence available, PaddleOCR's — and CRAFT's — ability to tell text
pixels from scene elements is not the problem.** On the one fixture whose
annotation is complete enough to test the claim, the separation is total:

| `avenue-de-la-république` | median | min / max |
|---|---:|---:|
| CRAFT peak inside annotated text (2 regions) | **1.013** | min **0.995** |
| CRAFT peak inside 12 text-free scene surfaces | **0.005** | max **0.311** |

Every annotated region outscores every text-free surface, and by a factor of
three even at the worst pairing. The detector is not confusing stone, panel or
sky for glyphs on this asset. Its two failures there are `proposed-then-lost`
and `fragmented` — the text was seen and then mishandled downstream.

That is the honest answer to the question as posed, and it points the work
somewhere else.

### What this does not establish

- **One fixture, not a corpus.** Only two of the annotated assets
  (`avenue-de-la-république`, `la-rue-sans-nom-example`) are complete
  annotations. The rest are `partial`, and on a partial annotation "text-free"
  means "overlaps no *annotated* box", which is not the same thing.
  `la-bastille-1789` duly reports OVERLAPPING (text min 0.910 vs scene max
  0.923) — but its own annotation note lists three whole blocks of unannotated
  text, so a scene surface scoring like text there is at least as likely a hole
  in the annotation as a detector error. The harness now records
  `negatives_trustworthy: false` in that case and says so on stdout.
- **This measures CRAFT, not Paddle's own probability map.** The isolated
  `.venv-paddle` worker returns final detections only; DB's per-pixel
  probability surface is not exposed across the subprocess bridge. So Paddle's
  discrimination *at pixel level* remains unmeasured. What is measurable is its
  proposal count against CRAFT's on the same image, recorded on every run.

| Asset | CRAFT raw proposals | Paddle proposals |
|---|---:|---:|
| `la-bastille-1789` | 37 | 11 |
| `avenue-de-la-république` | 4 | 2 |

  Consistent with the already-recorded drop-in comparison (42/72 matched vs
  CRAFT's 51/72): Paddle proposes **less**, not more. Whatever is being lost, it
  is not being lost to Paddle over-firing on scene texture.

- **Paddle reachability is now recorded on every run**, reachable or not.
  A field that is simply absent when the isolated venv is missing cannot be
  distinguished later from one where Paddle ran and found nothing — that is the
  "measuring a configuration the product does not run" trap, and it has cost
  this project real work twice. The first run of the new arm reported
  `reachable=False` and named an `AttributeError` in *this harness*, not a
  missing venv; without the reason field that would have read as "Paddle is not
  installed" and been believed.

## 2. Where the candidates actually die

`la-bastille-1789`, the corpus's worst asset (recall 0.333), per ground-truth
region:

| region | peak region | peak affinity | raw IoU | verdict |
|---|---:|---:|---:|---|
| LA | 0.923 | 0.881 | 0.866 | proposed-then-lost |
| BASTILLE | 0.980 | 0.947 | 0.793 | proposed-then-lost |
| ET | 0.962 | 0.906 | 0.719 | proposed-then-lost |
| LA | 0.957 | 0.885 | 0.567 | proposed-then-lost |
| RUE | 0.971 | 0.948 | 0.735 | proposed-then-lost |
| St | 0.949 | 0.964 | 0.095 | boundary |
| ANTOINE | 0.971 | 0.964 | 0.333 | boundary |
| EN | 0.910 | 0.858 | 0.604 | proposed-then-lost |
| 1789 | 0.985 | 0.949 | 0.572 | proposed-then-lost |

**Nine regions out of nine, and not one is `no-activation`.** Every single one
was seen by the detector at high confidence — peak region ≥ 0.91 throughout —
and lost afterwards. Seven arrive as good raw proposals and are destroyed
downstream; two arrive already mis-bounded.

This is the finding that decides where effort goes. The three candidate remedies
cost very different amounts:

| If the failure were… | Remedy | Cost |
|---|---|---|
| no detector evidence at this scale | tiles, another detector, fine-tuning | high |
| evidence present, below proposal | score calibration, targeted retry | low |
| **evidence present, wrongly grouped** | **polygonisation, split/merge rules** | **medium** |

Nothing on this asset is in the first row. Resolution and model choice are not
what is wrong with it.

## 3. The failure ladder, now visible in the product

`GET /api/manifest/{asset_id}/detection-attribution` names which of six events
produced an empty Capture tab, from the lineage the shipping run recorded. It
re-runs nothing — a fresh permissive pass is a different run, and comparing one
against what shipped is the mistake described at the top of this file.

| Rung | Meaning | Where it points |
|---|---|---|
| `no_engine` | no OCR engine available | install/config |
| `no_proposals` | detector proposed nothing | tiles / detector / fine-tune |
| `all_suppressed` | proposed, then everything suppressed | thresholds, this file's §2 |
| `all_excluded` | captured, then excluded by the user | History |
| `regions_present` | the ordinary case | — |
| `no_lineage` | nothing recorded; cannot say | recapture |

The frontend card names the rung and shows which stage suppressed how many
candidates. Before this, all six read `no text regions detected. you can draw
them manually.`

`no_lineage` exists because absence of evidence has to be sayable. Reporting a
pre-lineage manifest as "the detector proposed nothing" would be inventing a
finding — which is precisely the error this whole workstream is a correction of.

## 4. The threshold register

`docs/threshold-register.md` inventories every hard constant in the detection
path with a provenance label. The tally for detection alone: **4 `measured`,
10 `reasoned`, 3 `inherited`, and the clear majority `invented`.**

That is the answer to "there are hard threshold values potentially contributing
to these outcomes": yes, about forty of them, and most have never been swept.
It does not follow that they are wrong — the corpus result is what it is *with*
them — but it does mean "a threshold is miscalibrated" is a hypothesis with
thirty untested candidates behind it, and nudging one is guessing unless the
lineage graph says that constant killed the candidate.

Two of the four `measured` entries are cautionary rather than supportive:
`MIN_SYMBOL_JUNK_AREA` was fitted on the evaluation corpus with no held-out
evidence, and `MAX_ZOOM_INFLATION` was fitted on a single fixture and regressed
the corpus when shipped.

## 5. What to do next, in order

1. **Re-annotate `la-bastille-1789` at line level.** Its own annotation note
   already concedes the word-level premise is false — the detector now emits
   line-scale boxes for both display lines, and the annotation is fitted to a
   defect that no longer exists. Recall there is 0.333 with the full pipeline
   **and still 0.333 with `scene_filter` and `prune_garbage` both disabled**, so
   no threshold relaxation can move it. This is measurement debt, not detection
   debt, and it is the single cheapest correction available.
2. **Complete more annotations.** The discrimination question is answerable on
   exactly two assets today. Completing annotations is what turns §1 from an
   indication into a result, and it is the same blocker as corpus-wide precision
   being undefined.
3. **Attribute the `proposed-then-lost` bucket by stage.** The lineage graph can
   now say which transform suppressed each candidate; §2 says only that
   something did. Sweep only the constants that appear there.
4. **Do not sweep `link_threshold`.** Already measured inert — mean IoU
   identical to three decimals across 0.10–0.70 while raw box count moves 25→49.
   The multipass union and column merge absorb it entirely.

## Reproducing

```bash
.venv/Scripts/python scripts/eval_detector_evidence.py images/avenue-de-la-république.jpeg --lang fr --paddle
```

Every output file records the languages, `mag_ratio`, Paddle reachability with
its reason, and `negatives_trustworthy` for the annotation it scored.
