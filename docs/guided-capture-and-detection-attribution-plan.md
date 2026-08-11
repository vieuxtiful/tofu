# Guided Capture & Detection Attribution — megaplan

Status: in progress, 2026-08-11. Branch `detection-observability`.

**Shipped: M1–M6; M7 partial.** Backend 1893 passed / 2 skipped; frontend 223
passed; `npm run build` clean.

M3 resolved E1.2 in favour of option (a): Guided commits on Enter into a chip
strip (`BlocksField.tsx`), wrapping the existing field rather than re-pointing
its tokenizer, so Auto's per-word behaviour is untouched.

M4 put reconciliation in `layers/aboyeur.py` — the brigade role that calls the
order and checks each plate against it. Ownership, not string comparison, is
what it enforces: one region has at most one Block owner, duplicate Blocks
consume distinct evidence, explicit association outranks inference, and fuzzy
evidence reaches `review` but never `complete`.

M5 wired the flow: Guided Capture routes to drawing and never runs detection,
draw mode persists between boxes behind a pointer lock, and the prompt advances
only from the server-returned assessment.

M6 measured rather than asserted. On the one fully-annotated fixture, CRAFT's
response separates text from text-free scene surfaces completely (median 1.013
vs 0.005; min 0.995 vs max 0.311), and all nine of `la-bastille`'s regions are
`proposed-then-lost` or `boundary` — none is `no-activation`. So detection's
problem there is grouping, not discrimination. `docs/threshold-register.md`
labels every constant's provenance: 4 measured, 10 reasoned, 3 inherited, the
majority **invented**. Full write-up in `docs/detection-attribution-report.md`.

M7 is PARTIAL. The paired Gate 2 comparison is built and run: Guided 74.6% vs
Auto 67.2%, **+7.4 pts, 14 discordant wins and 0 losses, McNemar p = 0.00012**
(`docs/gate2-status.md`). Not certified — the margin is 0.6 points under the
declared +8.0, and the corpus is still `machine_derived`.
`docs/guided-corpus-review-protocol.md` plans the two-annotator review that
unblocks the second.

**The oracle arm is built and its result is INVALID as a ceiling** — 42.3%,
below both arms, because it reads annotated crops through a bare backend rather
than the pipeline's own recognition path. Labelled as such in
`docs/gate2-status.md` and in the function's docstring rather than deleted, so
the plausible-looking number cannot be quoted later. The feature's premise is
therefore still untested.

Outstanding: the corpus review (calendar time, not engineering time), a valid
oracle that injects annotated boxes into the pipeline's own recognition path,
and an end-to-end Guided capture driven through the UI against a live backend —
components and API are covered by tests, the assembled journey is not.

Seven workstreams. They are ordered so that each one's evidence exists before the
next one needs it, and so that no polished surface ends up driving a data model
that has already destroyed the thing it displays.

| # | Workstream | Depends on | Ships |
|---|---|---|---|
| **0** | **Manifest persistence contract** | — | **the gate on both A and E** |
| A | Detection attribution: why Auto draws nothing | 0 | a report + a harness + one endpoint |
| B | Guided empirical baseline | A (for attribution), E2 (for the real arm) | numbers, not features |
| C | Capture-tab UX: recapture confirm, prepping gate | — | two small frontend fixes |
| D | Dictionary keycap hint (`Ctrl`+`Space`) | — | one component + one hint |
| E | Guided capture (5 parts) | 0, then E1 | the feature |
| F | Sequencing, risks, acceptance | all | — |

C and D are independent of everything and can land first as warm-up. A is
investigation and must not be blocked on the feature. Part 0 gates both A and E;
within E, E1 gates the rest.

**This is a program document, not one delivery unit.** Section F.0 splits it into
seven independently reviewable milestones. Nothing here should be implemented as
a single change, and no small UI fix should end up queued behind Paddle
instrumentation or a human corpus review.

---

## Premise corrections

Six things found while reading the code that change what the work is. Stated up
front because four of them are load-bearing, and the sixth is a blocker that
predates every feature in this document.

**1. Block-flattening happens in three places, not one.** The plan named
[App.tsx:1375](../frontend/src/App.tsx:1375). There are two more, and the third
is worse than the first:

| Site | What it does | Damage |
|---|---|---|
| [`padGroundTruth`](../frontend/src/App.tsx:292) | `terms.join(" ") + " "` | round-trips the list back into one space-joined string, so phrase boundaries are gone before the user sees the field again |
| [App.tsx:1375](../frontend/src/App.tsx:1375) / [:1391](../frontend/src/App.tsx:1391) | `.split(/\s+/)` | splits `la première saisie` into three |
| [`_normalize_ground_truth`](../server/main.py:2202) | `str(value).split()` **plus a `seen` set** | splits again server-side **and deduplicates** |

The server dedupe is the one that matters most: `PARIS PARIS` becomes `PARIS`
before it is ever stored. The Guided model's entire promise — order preserved,
duplicates independently fulfillable ([mise.make_blocks](../src/tofu/layers/mise.py:151)) —
is contradicted by the persistence layer, not just by the UI. And because
[`saveGuidedBlocks`](../frontend/src/App.tsx:1391) is handed the already-split
`terms`, **`mise` has never once seen a phrase.** Fixing App.tsx alone would
change nothing.

**2. The Blocks field is a single-line `<input>`.** [GroundTruthField](../frontend/src/GroundTruthField.tsx:37)
holds an `HTMLInputElement`, and its colour mirror tokenizes on whitespace
(`value.split(/(\s+)/)` at [:55](../frontend/src/GroundTruthField.tsx:55)) to
assign one palette colour per word. So "newline-delimited entries" is not a
parsing change — Enter cannot produce a newline in that element, and even if it
could, a phrase Block would still render as three differently-coloured words,
visually contradicting the claim that it is one thing. This needs a decided
affordance (E1.2), not a regex.

**3. Manual recapture destroys the manifest with no prompt and no snapshot.**
[`onRerunDetect`](../frontend/src/App.tsx:2593) confirms and snapshots.
[`onManualDraw`](../frontend/src/App.tsx:2609) does `setManifest([])` with
neither. Both are wired to the same `recapture` button
([App.tsx:3931](../frontend/src/App.tsx:3931)). The requested ToFU pop-up should
cover both paths, and manual recapture should snapshot too — that is a data-loss
bug the pop-up work happens to sit on top of.

**4. shadcn is not installed, in any form.** No `components.json`, no `@/` alias
in [tsconfig](../frontend/tsconfig.json) or [vite.config.ts](../frontend/vite.config.ts),
no `clsx` / `tailwind-merge` / `class-variance-authority` in
[package.json](../frontend/package.json). Tailwind is v4 with a hand-tuned
[index.css](../frontend/src/index.css) and a large bespoke
[uikit.css](../frontend/src/uikit.css). `npx shadcn@latest init` rewrites
`index.css`, `tsconfig`, and the Tailwind config. See D2 for how to get the real
component without letting the initializer near the design system. Also: `kbd` is
an npm component, so it belongs in `package.json` — `requirements.txt` is the
Python server's and takes nothing from this.

**5. The existing keycaps are already unstyled.** [App.tsx:4122](../frontend/src/App.tsx:4122)
renders `<kbd className="kbd kbd-xs">A</kbd>`. `.kbd` and `.kbd-xs` are daisyUI
classes, and daisyUI was deliberately removed
([index.css](../frontend/src/index.css) explains why at length). Neither class is
defined anywhere in the project's CSS, so those three keycaps currently render as
bare text. The new `Kbd` component should replace them in the same pass.

**6. Ordinary autosave silently deletes Guided state, and the lineage graph is
never persisted at all.** Found in review, verified, and broader than first
reported. The frontend `TextManifest`
([api.ts:577](../frontend/src/api.ts:577)) has no `guided_blocks` field;
[`autoSave`](../frontend/src/App.tsx:1530) builds a **complete** manifest object
from scratch every 1.5 s; [`put_manifest`](../server/main.py:2881) calls
`_dict_to_manifest(manifest_data)`, which defaults `guided_blocks` to `[]`
([manifest_store.py:482](../src/tofu/utils/manifest_store.py:482)); and
`save_manifest` **replaces** the stored document. So:

> Blocks saved → user drags one box → 1.5 s later, Guided progress is gone.

The same boundary is why Part A has nothing to read: `_manifest_to_dict`
([manifest_store.py:77](../src/tofu/utils/manifest_store.py:77)) has **no
`candidate_lineage` key**, so the graph `cicerone` builds is dropped by the very
first save after detection. It is not "persisted but unreachable" as this plan
originally claimed — it has never been written to disk, and every lineage number
on record comes from an in-process harness run.

This is one defect with two victims, and it is not confined to autosave: there
are **29 `save_manifest` call sites** in `server/main.py`, and every one
round-trips through the same serializer. Any server-owned field absent from
`_manifest_to_dict`/`_dict_to_manifest` is erased by *any* endpoint that touches
the manifest — font resolution, merges, snapshot restore, exclusion. Part 0
below.

**Correct in the plan as written, confirmed:** `detection_assessment` exists on
[GuidedBlock](../src/tofu/core/types.py:433), already round-trips through
[manifest_store](../src/tofu/utils/manifest_store.py:186), and is written by
nothing. Guided capture does still launch Auto detection
([App.tsx:3917](../frontend/src/App.tsx:3917):
`captureMode === "manual" ? onManualDraw() : onAutoDetect()`).

---

## Part 0 — The manifest persistence contract

Premise correction 6 is the blocker. Everything downstream — Guided assessments,
lineage, and any provenance field added later — dies at the same boundary, so fix
the boundary rather than the two symptoms.

### 0.1 Ownership, declared

Split manifest fields into three sets and write the list into
`manifest_store.py` as a module-level constant, so the next field added has to
declare which set it joins:

| Set | Fields | Who may write |
|---|---|---|
| client-owned | `instances`, `src_lang`, `targ_lang`, `img_dim`, `semantic_units` (user-pinned plates aside) | frontend, via full PUT |
| server-owned | `guided_blocks`, `candidate_lineage`, `resolved_font_family`, `total_regions`, `prcssng_time`, plate identity fields | server only |
| derived | `total_regions`, `asset_class` | recomputed on write |

### 0.2 Merge semantics for omitted fields

`PUT /api/manifest/{id}` must preserve server-owned fields when the request does
not carry them. The reviewer is right that requiring every frontend constructor
to reproduce all backend metadata forever is the fragile option — it fails
silently, once per newly added field, and the failure looks like data loss rather
than like a bug.

One subtlety the merge has to get right: the endpoint takes a bare
`Dict[str, Any]`, so **absent and empty are currently indistinguishable**. The
contract must therefore key on *presence*:

- `"guided_blocks" not in body` → preserve what is stored.
- `"guided_blocks": []` → an explicit, intentional clear.
- The frontend must **never** send the key unless it genuinely owns that state.

Clearing Guided state deliberately (a Guided reset) goes through
`DELETE /api/assets/{id}/guided-blocks`, not through an empty array in an
autosave body. An intent that important should not be expressible as an
accidental omission's twin.

### 0.3 Prefer targeted mutations for what the client actually owns

Full-document PUT is a poor fit for single-region edits: it forces the client to
reconstruct state it does not own, and it makes every save a potential
whole-manifest regression. Where Guided is concerned, region mutation moves to
targeted endpoints (Part E2). The full PUT stays for bulk Translate-step editing,
now with merge semantics behind it.

### 0.4 Stale-write protection

Give the manifest a monotonic `revision` (or an ETag) bumped on every accepted
write. `PUT` carries `If-Match` / `expected_revision`; a mismatch returns 409
rather than overwriting. Without this, an autosave in flight when a Guided
reconciliation lands will still clobber the newer assessment even with correct
merge semantics — the payload was assembled before the newer state existed.

Mitigating factor worth knowing: `put_manifest` snapshots every accepted write to
the ledger (`db.add_snapshot(..., reason="autosave")`,
[main.py:2894](../server/main.py:2894)), so a clobber is *recoverable*. It is not
*detectable*, which is the part that matters.

### 0.5 Tests

1. `PUT /guided-blocks` → `PUT /manifest` without `guided_blocks` →
   `GET /guided-blocks` returns the Blocks unchanged.
2. The same for `candidate_lineage` across a font-resolution save, a merge, an
   exclusion, and a snapshot restore — all 29 save sites share one serializer, so
   one parametrised test over the endpoint list is the honest coverage.
3. `"guided_blocks": []` explicitly clears; omission does not.
4. A `PUT` carrying a stale revision is rejected, not applied.
5. Detection run → `save_manifest` → reload → lineage still present (this fails
   today).

**Acceptance:** no field in the server-owned set can be erased by any client
request that does not name it, proven per endpoint, not per field.

---

## Part A — Why Auto draws no boxes

The goal is an attribution, not a fix. The project has been burned twice by the
opposite order — `MAX_ZOOM_INFLATION` was fitted on one fixture, shipped, and
regressed the corpus; an attribution pass compared against a detector
configuration the product does not run and invented five phantom defects
(`docs/measured-dead-ends.md`). So: measure first, and record provenance for
every number.

### A1. The failure ladder

"No boxes are drawn" is at least six distinct events, and they have nothing in
common except the screen. Instrument them so a single run says which one happened.

| Rung | Event | Where it is decidable today |
|---|---|---|
| 0 | No OCR engine — `ev.engine === "null"` | already toasted, [App.tsx:2543](../frontend/src/App.tsx:2543) |
| 1 | Detector emitted no raw proposals at all | `okara` graph has zero `raw_craft` nodes |
| 2 | Proposals existed, every one suppressed | graph nodes exist, none `active` |
| 3 | Regions shipped, all `excluded` | [`visibleManifest`](../frontend/src/App.tsx:867) filters them out |
| 4 | Regions shipped and visible, canvas painted nothing | `imgSize` null → no overlay geometry |
| 5 | Regions shipped, drawn, invisible | bbox colour vs. background, z-order, zero-area boxes |

Rungs 3–5 are frontend and cost almost nothing to rule out. Do them first: they
are the cheapest possible falsification of a detection story, and per
`verify-paint-order-not-just-computed-styles` the Browser pane cannot be trusted
to screenshot the answer — prove it structurally.

Rungs 1 and 2 are exactly what [okara](../src/tofu/layers/okara.py) was built to
answer, and the graph reaches 100% coverage on all 11 fixtures — but it never
reaches disk. `cicerone` attaches it to the **in-memory** manifest object at
[cicerone.py:4553](../src/tofu/layers/cicerone.py:4553), the eval harnesses read
it in-process (`eval_detect_corpus.py:112`, `forage.py:188`, `vtm.py:252`), and
then [`_manifest_to_dict`](../src/tofu/utils/manifest_store.py:77) — which has no
`candidate_lineage` key at all — drops it at the first `save_manifest`. Every
recorded lineage number in this repository comes from an in-process run. See
Part 0: this is the same serializer boundary that erases Guided state, and it is
A1.1's prerequisite, not a detail.

**A1.0 — Persist the graph** (blocked on Part 0). Add `candidate_lineage` to the
serializer, or write it to a per-run sidecar (`lineage/{asset_id}/{run_id}.json`)
if manifest size becomes a concern — the graph is append-only and can be large on
dense scenes. Sidecar is the safer default: it keeps a diagnostic artifact out of
the document that ships, and it makes run-over-run comparison natural.

**A1.1 — Expose the lineage.** `GET /api/manifest/{asset_id}/detection-attribution`
returning the persisted `candidate_lineage` plus a derived summary:
`{raw_nodes, active, merged, pruned, replaced, suppressed_by_stage: {...},
scene_verdicts: {...}, engine, detector_configs: [...]}`. Read-only, no
recomputation — a re-run would be a different run, which is the trap A exists to
avoid.

**A1.2 — Surface it.** When Capture has zero visible regions, replace the
existing bare `"no text regions detected. you can draw them manually."` toast
with a card that names the rung: *"the detector proposed 41 candidates; all 41
were suppressed — 33 at the merge stage, 8 below the confidence floor"* versus
*"the detector proposed nothing at this resolution."* Those two sentences send
the user to completely different remedies, and today they read identically.

### A2. The threshold register

The suspicion that hard-coded constants are driving outcomes is correct, and the
constants are numerous. An inventory from `cicerone.py` alone:

| Constant | Value | Line |
|---|---|---|
| `PASS_THRESHOLDS` | 3 rungs | [2443](../src/tofu/layers/cicerone.py:2443) |
| `LATIN_POOL_MIN_CONFIDENCE` / `_RESCUE_` | 0.5 / 0.35 | [1107](../src/tofu/layers/cicerone.py:1107) |
| `DECLARED_LANGUAGE_MARGIN` | 2.0 | [1143](../src/tofu/layers/cicerone.py:1143) |
| `SCENE_FILTER_VETOES` | off (env) | [1378](../src/tofu/layers/cicerone.py:1378) |
| `COLUMN_MAX_ASPECT` / `_X_ALIGN` / `_WIDTH_RATIO` / `_MAX_GAP` / `_COLOR_MAX_DIST` / `_LATIN_MIN_CONF` | 1.6 / 0.5 / 1.7 / 0.8 / 90.0 / 0.5 | [1638](../src/tofu/layers/cicerone.py:1638)–[1664](../src/tofu/layers/cicerone.py:1664) |
| `VERTICAL_STACK_MIN_ASPECT` / `MIN_BAND_HEIGHT_PX` | 3.0 / 12 | [1877](../src/tofu/layers/cicerone.py:1877) |
| `ROW_*` (align, ratio, gap, members, conf) | 0.5 / 1.7 / 0.6 / 2 / 0.5 | [2069](../src/tofu/layers/cicerone.py:2069)–[2098](../src/tofu/layers/cicerone.py:2098) |
| `MIN_DISAMBIGUATION_EVIDENCE` / `MIN_KANA_CONFIDENCE` / `MIN_SYMBOL_JUNK_AREA` | 2 / 0.3 / 450 | [2480](../src/tofu/layers/cicerone.py:2480)–[2495](../src/tofu/layers/cicerone.py:2495) |
| `PADDLE_RESCUE_CONF_FLOOR` / `PADDLE_OVERLAP_WIN_FLOOR` | 0.6 / 0.5 | [2802](../src/tofu/layers/cicerone.py:2802)–[2806](../src/tofu/layers/cicerone.py:2806) |
| `PADDLE_ESCALATED_DROP_SCORE` / `_UNCLIP_RATIO` | 0.15 / 1.9 | [2973](../src/tofu/layers/cicerone.py:2973) |
| `SURFACE_COVERAGE_FLOOR` / `SUBDIVIDE_*` | 0.85 / 120 / 10 / 1.5 | [2977](../src/tofu/layers/cicerone.py:2977)–[3002](../src/tofu/layers/cicerone.py:3002) |
| `LOAF_*` (containment, crumb conf, span, duplicate, rescue ceiling) | 0.8 / 0.3 / 0.6 / 0.9 / 0.5 | [3407](../src/tofu/layers/cicerone.py:3407)–[3441](../src/tofu/layers/cicerone.py:3441) |
| `ZOOM_MAX_SURFACE_FRAC` / `_MAX_SURFACES` / `_SCALE` / `_PAD` / `_FRAGMENT_CONTAINMENT` | 0.5 / 14 / 2 / 8 / 0.9 | [3763](../src/tofu/layers/cicerone.py:3763)–[3854](../src/tofu/layers/cicerone.py:3854) |
| `EDGE_RESCUE_PAD_RATIO` / `_CONF_SLACK` | 0.16 / 0.10 | [3954](../src/tofu/layers/cicerone.py:3954) |
| `FRAGMENT_OVERLAP` | 0.5 | [4956](../src/tofu/layers/cicerone.py:4956) |

**Deliverable: `docs/threshold-register.md`** — one row per constant with a
`provenance` column taking exactly one of:

- `measured` — a recorded sweep exists; cite it,
- `inherited` — a library default carried forward unexamined,
- `invented` — chosen by judgement, never swept.

Expect most rows to read `invented`. That is the honest finding, and writing it
down is more valuable than changing any single value. Then one rule, enforced in
review: **no constant moves without held-out evidence, or an explicit note that
it has none.**

Do not sweep everything. The dead-ends file already records that `link_threshold`
moves raw box count 25→49 while mean IoU is identical to three decimals — the
multipass union and column merge absorb it entirely. Sweeps of absorbed
parameters produce flat lines and false confidence. Prioritise constants that
`okara` shows are actually killing candidates on the failing assets: instrument
first (A1.1), then sweep only what the instrument implicates.

### A3. PaddleOCR vs. scene pixels — the honest statement

The request is to address this honestly, so here is what the evidence in the
repository already supports, and what it does not.

**Supported today:**
- As a *drop-in detector*, PaddleOCR's DB scores **42/72 matched against CRAFT's
  51/72**, with the near-miss bucket growing 17→21, at roughly 2× the wall clock,
  and it was never better on any fixture (`docs/measured-dead-ends.md`). It is
  not currently a text/scene discriminator that outperforms what ships.
- That comparison is **not a test of DB's polygons.** Ground truth is
  axis-aligned and scoring is box-IoU, so a polygon detector is being graded
  through a rectangle. This is stated in the dead-ends file and must be repeated
  in any write-up that cites 42/72 — otherwise the number reads as a verdict on
  the model when it is partly a verdict on the annotation format.
- Paddle is a **rescue path**, gated by [`should_paddle_rescue`](../src/tofu/layers/cicerone.py:2934)
  on CJK-dominant scenes and run out-of-process under an isolated `.venv-paddle`
  ([cicerone.py:20](../src/tofu/layers/cicerone.py:20)). If that venv is absent
  the path is a silent no-op. **Any claim about Paddle's behaviour must first
  record whether the worker was even reachable during the run** — this is the
  "measuring a configuration the product does not run" trap, and it has already
  cost this project real work twice.

**Not supported, and must not be asserted without measurement:** that Paddle
specifically confuses scene texture for glyphs, or that its detection failures
are text/non-text discrimination failures rather than scale, contrast, or
grouping failures. The pipeline does not currently separate those.

**A3.1 — Make the claim decidable.** `scripts/eval_detector_evidence.py` already
reads CRAFT's own region/affinity maps at ground-truth coordinates and
distinguishes *never proposed* from *proposed and discarded*. Extend it with a
Paddle arm reading DB's probability map at the same coordinates, plus a
**negative** arm: sample the DB/CRAFT response inside `scene_regions` that
contain no annotated text. Two numbers come out:

- `peak_prob` inside annotated text (does it see text?),
- `peak_prob` inside text-free scene surfaces (does it hallucinate text?).

Their separation *is* the discrimination claim, measured. If the distributions
overlap, "cannot discern text pixels from scene elements" is established rather
than asserted; if they separate cleanly, the failure is elsewhere and the honest
write-up says so.

**A3.2 — Report the scene filter's real standing.** [`_scene_verdict`](../src/tofu/layers/cicerone.py:1384)
computes what the filter *would* decide whether or not the veto is enabled, and
the veto ships **off** (0.710 → 0.738 mean IoU, 47 → 51 matched, at the cost of
99 → 134 candidates and a rise in review rate). That is a deliberate
recall-for-reviewer-time trade, and it belongs in the write-up as a decision with
a price, not as a bug.

### A4. Part A deliverables

1. `docs/threshold-register.md` — inventory with provenance.
2. `docs/detection-attribution-report.md` — the ladder, the per-asset rung, the
   Paddle/scene separation measurement, and an explicit "what we still cannot
   say" section.
3. `GET /api/manifest/{id}/detection-attribution` + the empty-state card (A1.1, A1.2).
4. `eval_detector_evidence.py` Paddle + negative-surface arms, with worker
   reachability recorded in every output file.

**Acceptance:** for every asset that draws no boxes, the report names its rung
and cites the lineage node ids. No threshold is changed in Part A.

---

## Part B — Guided empirical baseline

Distinct from Part A: A explains Auto's misses, B establishes what Guided is
worth. Two blockers are already known and neither is a coding problem.

**B1. The corpus cannot certify itself.** `evidence/guided-corpus-v1.json` is
`review_status: machine_derived`, and `eval_guided_corpus.py` correctly refuses
Gate 2 until two annotators have reviewed it. That review has not happened.
Schedule it as its own task — it is the gate, and no amount of implementation
substitutes for it.

**B2. `--arm guided` raises rather than silently reporting the Auto number.**
That refusal is correct and must stay until E2's reconciliation exists. Which
means: **B cannot produce its headline number before E2 lands.** Say so now
rather than discovering it at measurement time.

**B3. What can be measured before E2 — the oracle arm.** Add
`--arm guided_oracle`: feed the pipeline the annotated boxes as if a user had
drawn every one perfectly, and score recognition and downstream only. This
separates two things the Auto number fuses:

- localization failure (the box was never found), from
- recognition failure (the box was found and read wrong).

The oracle arm is the **ceiling** Guided can reach. If it sits near the Auto
baseline, Guided's value proposition is not localization and the plan should know
that before shipping five parts of it. This is measurable today and is the single
highest-information run available before E2.

**B4. Certification design, unchanged from the recorded decision.** Gate 2 is
paired: both arms score the same occurrences, so **McNemar on discordant
occurrences** is the analysis, on the FULL corpus, with dev/holdout reported
separately for transparency. n=40 holdout gives ±15.5pp on a single-arm point
estimate and cannot certify an 8pp margin alone. Do not certify on holdout-only.

Baselines to beat, already recorded: FULL 67.0% (122/182), DEV 71.8%, HOLDOUT
50.0%; worst strata `japanese_vertical` 29.2%, `nonlinear_irregular` 42.1%.

---

## Part C — Capture-tab UX

**C1. Recapture confirm → ToFU pop-up.** There are already three near-identical
copies of the same overlay in App.tsx —
[`TitleConfirmOverlay`](../frontend/src/App.tsx:344),
[`AssetDeleteConfirmOverlay`](../frontend/src/App.tsx:372),
[`UnsavedChangesOverlay`](../frontend/src/App.tsx:402) — differing only in their
message and one subtext line. Generalize into one `ConfirmOverlay({message,
detail?, onYes, onNo})` keeping `title-confirm-backdrop` / `title-confirm-card` /
`title-confirm-btn` and the 300 ms `leaving` exit exactly as they are, then
collapse the three call sites onto it.

Route recapture through it: **"begin recapture?"** with the detail line carrying
the consequence — `"replaces the current N region(s); a snapshot is saved first."`
Both buttons at [App.tsx:3931](../frontend/src/App.tsx:3931) (the `recapture`
FlipButton) and [:3986](../frontend/src/App.tsx:3986) (the Capture-tab AI-scan
button) go through it, and — per premise correction 3 — **the manual path must
snapshot before clearing**, which it does not do today.

Because this is async-confirm rather than blocking `confirm()`, the state has to
move to a pending action: `const [pendingRecapture, setPendingRecapture] =
useState<null | "auto" | "manual" | "guided">(null)`. Keep the native `confirm()`
at the other five call sites for now; converting them is separate work and the
pop-up component this creates makes it a mechanical follow-up.

**C2. `(prepping...)` must not lock the Blocks field.** One line:
[App.tsx:3727](../frontend/src/App.tsx:3727) passes
`capturing={busy === "detecting" || scan?.status === "scanning"}`. The scan is
the pre-flight sniff, not the capture; drop the second clause so it reads
`capturing={busy === "detecting"}`.

The lock itself is well-built and stays — [GroundTruthField](../frontend/src/GroundTruthField.tsx:230)
sets `readOnly` and paints travelling stripes so the state reads as "working" not
"disabled", and its rationale (editing Blocks mid-capture means the run no longer
matches the request that started it) applies to detection only. Add a test that
pins the distinction, because the two states look similar and the regression
would be invisible.

---

## Part D — Dictionary keycap hint

**D1. Where it goes and what it says.** The IME suggestion popover already prints
`User dictionary` / `Ctrl+Space` as plain text at
[AnimatedCaretTextarea.tsx:402](../frontend/src/AnimatedCaretTextarea.tsx:402),
and the binding is [`isRecommendationActivationKey`](../frontend/src/AnimatedCaretTextarea.tsx:34)
(`Ctrl+Space`, or `Alt+ArrowDown`). The hint should be a dismissable strip in the
same register as the fonts hint at [App.tsx:3788](../frontend/src/App.tsx:3788):
`bezier-impression subtext ... text-xs`, `MdTipsAndUpdates` leading icon, plus a
dismiss control matching the tiny-upload advisory's
([App.tsx:4047](../frontend/src/App.tsx:4047)).

Content: `Dictionary = ⌨ Ctrl + Space`, keyboard glyph followed by
`<KbdGroup><Kbd>Ctrl</Kbd> + <Kbd>Space</Kbd></KbdGroup>` as specified.

Two honest notes on the copy: the binding also accepts `Alt+↓`, and on macOS
`Ctrl+Space` collides with the OS input-source switcher. Mention the alternate
in the `aria-label` at minimum; a hint that names a shortcut the platform steals
is worse than no hint.

Dismissal persists in `localStorage` under a versioned key
(`tofu.hint.dictionary.v1`) — session-only dismissal means it returns on every
reload, which reads as a bug.

**D2. Getting `kbd` without letting the initializer near the design system.**
`npx shadcn@latest add kbd` requires `components.json`, which requires
`npx shadcn@latest init`, which rewrites `index.css`, `tsconfig.json`, and the
Tailwind config. [index.css](../frontend/src/index.css) carries a long, specific
rationale for its current state (the daisyUI removal, the v4 border-colour
compatibility layer, the Vietnamese-diacritic font stack) and
[uikit.css](../frontend/src/uikit.css) is ~6.5k lines of bespoke system. Letting
init rewrite them to obtain one keycap component is a bad trade.

Recommended sequence:

1. Run the real CLI in the scratchpad, against a throwaway Vite+Tailwind-v4 app:
   `npx shadcn@latest init` then `npx shadcn@latest add kbd`. This yields the
   canonical, current source rather than a reconstruction from memory.
2. Read the generated `components/ui/kbd.tsx` and its actual dependencies. `Kbd`
   is presentational — expect `clsx` + `tailwind-merge` (the `cn` helper) and
   possibly `class-variance-authority`, and **no Radix**.
3. Vendor it into `frontend/src/components/ui/kbd.tsx`, add only the dependencies
   it genuinely uses to `package.json`, and add a two-line `src/lib/utils.ts`
   exporting `cn`.
4. Add the `@/` alias in **both** `tsconfig.json` (`compilerOptions.paths`) and
   `vite.config.ts` (`resolve.alias`) — the import in the spec is `@/components/ui/kbd`
   and Vitest resolves through the Vite config, so missing either breaks tests
   rather than the build.
5. Record in the file header that it is vendored from shadcn, at which version,
   and why init was not run in-tree.

If the project would rather not take the alias and the two utility deps for one
component, the fallback is a ~30-line local `Kbd`/`KbdGroup` styled from
`uikit.css` tokens. Recommend the vendored route: it matches what was asked for,
and the `cn` helper pays for itself the next time a component is added.

**D3. The keyboard glyph.** The two supplied React components are byte-identical
except for the `color` default (`#000000` vs `#FFFFFF`). Shipping two components
that differ only in a default that the `<path fill="currentColor">` already
overrides would be dead code — the path fills with `currentColor` regardless, so
the `color` prop only ever reaches `stroke`, and the outline has no stroked
geometry. Ship **one** `KeyboardIcon`, let it inherit `currentColor`, and let the
existing dark-mode class chain pick the colour. This is what every other icon in
the app already does.

**D4. Fix the dead keycaps while here.** [App.tsx:4122](../frontend/src/App.tsx:4122)'s
`A` / `Del` / `Esc` hints use daisyUI's `.kbd .kbd-xs`, which is defined nowhere
since daisyUI was removed. Move them to the new `Kbd`. Small, visible, and
already broken.

---

## Part E — Guided capture

The proposed interaction, state machine, bubble, and progress UI are adopted as
written. What follows adjusts for what the code actually does.

**Part 0 comes first.** Preserving Blocks through the API (E1) is pointless while
the next autosave deletes them from the manifest — E1 without Part 0 produces a
Guided flow that works for 1.5 seconds at a time.

### E1 — Preserve Blocks as strings *(strict prerequisite, after Part 0)*

Nothing else in Part E can be correct before this. Four changes, and the third is
the one the original plan missed.

**E1.1 — Stop routing Guided through `ground_truth`.** Guided Blocks go to
`PUT /api/assets/{id}/guided-blocks` **directly from the raw field entries**, not
from `terms` derived by `.split(/\s+/)`. Legacy `ground_truth` keeps its existing
whitespace-and-dedupe semantics untouched, because
[`_ground_truth_pool`](../server/main.py:2214) feeds it to the recognizer as a
term pool where dedupe is correct behaviour. Two vocabularies, cleanly separated,
each right for its own consumer. Do **not** try to make `ground_truth` carry
phrases — that would change OCR biasing as a side effect of a UI change.

**E1.2 — Decide the input affordance.** Given premise correction 2, one of:

- **(a) Enter-commits-a-chip**, keeping the `<input>`: Enter commits the current
  buffer as one Block, chips render above/below with drag-reorder and per-chip
  delete. Order and duplicates are explicit and visible; the mirror keeps its
  current per-word colouring *within* a chip, which now reads as atoms of one
  Block rather than as separate Blocks. **Recommended** — it matches the Block
  model visually, and the chip strip already has precedent in
  [SemanticSubstitutionPanel](../frontend/src/SemanticSubstitutionPanel.tsx:316).
- **(b) Swap to a textarea, one Block per line.** Cheaper, but it inherits the
  mirror's whitespace tokenizer, loses the delimiter-settled colour behaviour the
  field was built for, and makes a duplicate-line Block look like a typo.

Under (a), Auto mode keeps the current single-line whitespace field verbatim —
`captureMode` already gates the label and placeholder
([App.tsx:3724](../frontend/src/App.tsx:3724)), so the branch exists.

**E1.3 — Load Guided values from `/guided-blocks`, not from asset ground truth.**
Extend `GuidedBlockRecord` and the GET response to expose `normalized_text`,
`atoms`, `find_all`, and `detection_assessment` — the PUT response already
returns atoms ([main.py:1898](../server/main.py:1898)) but the GET does not
([main.py:1916](../server/main.py:1916)), so a reload sees less than a save did.
That asymmetry is why reload cannot currently restore Guided state.

**E1.4 — `mise` stays authoritative.** The UI never tokenizes a phrase. Atoms
arrive from the server or they do not exist. This is the invariant that keeps
`saisie` from ever being promoted to a standalone Block, and it should be
asserted in a test, not just in prose.

### E2 — Durable reconciliation

`detection_assessment` shape as proposed, with two additions:

```jsonc
{
  "status": "pending | partial | review | complete | skipped",
  "region_ids": ["r5", "r6"],
  "matched_atom_ids": ["g3a1", "g3a2"],
  "remaining_atom_ids": ["g3a3"],
  "matched_text": "la première",
  "remaining_text": "saisie",
  "evidence": [                       // why each atom was considered matched
    {"atom_id": "g3a1", "region_id": "r5", "basis": "exact|fuzzy|dictionary|override|user"}
  ],
  "resolution": null,                 // "user_complete" | "user_skipped", set only by explicit override
  "updated_at": 0
}
```

`evidence` exists so a wrong match is debuggable without re-deriving it — the
same reason `okara` keeps merge parents. `resolution` is separate from `status`
so that an explicitly skipped Block can never be confused with one the matcher
believes it completed. `review` exists so that fuzzy-only evidence has somewhere
to land: without it, an uncertain match must be forced into either `partial`
(which strands the user) or `complete` (which lies), and the numeric match scores
that would justify `complete` are explicitly not calibrated yet.

#### E2.1 — Ownership invariants

The matcher needs these decided before it is written, not discovered during it.

| Question | Rule |
|---|---|
| One region satisfying atoms from several Blocks? | **No.** Each active region has at most one Block owner. |
| One atom supported by several regions? | **Yes** — a phrase can span regions; that is the point. Each atom is fulfilled **once** within its Block. |
| Manual reassignment to another Block? | **Yes**, explicitly, and it updates coverage on both the old and the new Block in one transaction. |
| Two identical Blocks, one matching region? | Duplicate Blocks consume **distinct** region evidence. A region already owned cannot also satisfy its twin. First-drawn wins; the second Block stays `pending`. |
| User edits a region's text? | Revokes every `fuzzy` and `dictionary` evidence row for that region and re-reconciles. `user` and `override` rows survive. |
| Automatic reconciliation stealing an explicitly associated region? | **Never.** Explicit user association outranks inferred assignment, permanently. |
| Punctuation-only atoms? | Not required for completion. They are `kind: "punctuation"` from [mise](../src/tofu/layers/mise.py:46) and carry weight 0 (E4.1). |
| What fuzzy score completes a Block? | Nothing, initially. Fuzzy-only evidence produces `review`, never `complete`, until a threshold is calibrated against the corpus. `exact` after normalization completes. |
| User completion overwritten later? | **Never.** `resolution` is recorded separately and no automatic pass rewrites it. |

The bias throughout: automatic reconciliation may *propose*, and may complete only
on exact evidence; anything softer asks. That is the same posture the scene filter
took when its veto was removed — suppression belongs downstream of recall.

#### E2.2 — One canonical, idempotent transaction

The reviewer is right that `POST /api/detect/refine`
([api.ts:1436](../frontend/src/api.ts:1436)) is the wrong place for `block_id`:
it owns no manifest instance, so an association recorded there is not durable,
and having two endpoints that can both start a reconciliation invites two
contradictory ones. **Drop `block_id` from refine.**

One endpoint persists the region and its reconciliation together:

```jsonc
POST /api/manifest/{asset_id}/regions
{
  "x": .., "y": .., "width": .., "height": ..,
  "text": "…",                       // from the stateless refine precursor
  "guided_block_id": "g3",           // optional; absent for Auto/Manual
  "client_operation_id": "uuid-v4",  // idempotency key
  "expected_revision": 41            // Part 0.4
}
→ { region, guided_assessment, guided_progress, revision }
```

Server-side, atomically: add the region → associate → reconcile the Block →
persist once → return all three. One write, one revision bump, one source of
truth for what the bubble shows next.

`client_operation_id` is not optional ceremony. Pointer-locking (E3) prevents a
*second deliberate draw*; it does not prevent a retried request, a replayed
double-click, or a response landing after the user has navigated away. A repeat
of the same operation id returns the original result without adding a second
region.

**Alternative worth considering:** a single `POST /api/guided/{asset_id}/draw`
that refines *and* persists *and* reconciles server-side. It costs one round trip
instead of two, which removes the window E3's pointer lock exists to cover, and
it keeps refinement policy on the server where the OCR engine is. Recommended if
the refine step's UI feedback (currently none — the user just waits) is not
missed.

Auto and Manual paths omit the Guided fields entirely and their responses stay
byte-identical to today, pinned by test.

Matching rules, as proposed and endorsed: normalized exact first, then existing
fuzzy/text-span utilities; atoms matched in order with duplicates preserved;
geometry proximity as supporting evidence only, never a requirement (a phrase
split across a corner or two faces of a sign must still complete); dictionary and
`source_override` hits corroborate but never redefine Block boundaries; never
promote a substring to a Block; return unresolved atoms without rewriting
`raw_text`.

Re-run reconciliation and persist after **every** mutation: add, OCR re-read,
text edit, exclusion, merge, undo-restored save, delete. Soft-deleted (`excluded`)
regions are excluded from coverage — note that
[`onDeleteRegion`](../frontend/src/App.tsx:2721) marks `excluded: true` rather
than splicing, precisely so the autosave PUT does not undo the server's flag, so
reconciliation must read the flag rather than the array length.

**Escape hatches** (re-read, edit source, undo/remove, mark-complete, skip) are
not optional polish. Without them, one unreadable sign traps the user in a state
machine that will not advance. `resolution` persists the decision so a reload
does not re-trap them.

### E3 — The state machine

Adopted as written, extracted into `useGuidedCapture`. Additions:

- Guided auto-routing sits **beside** the Manual routing effect
  ([App.tsx:2622](../frontend/src/App.tsx:2622)), with the same
  `routedRef`-per-asset guard — that guard exists because re-running on every
  render fights a user who deliberately navigated back to Upload.
- `onGuidedDraw`, not `onAutoDetect`, from the Capture button
  ([App.tsx:3917](../frontend/src/App.tsx:3917)).
- Snapshot before clearing on recapture (shared with C1).
- [`onAddRegion`](../frontend/src/App.tsx:2632) currently calls `setDrawMode(false)`
  as its first statement. Under Guided it must not — but note that the refine →
  add sequence is two awaited round-trips, so "keep draw mode on" without a lock
  invites a second box mid-flight. Lock pointer-down for the duration and show it
  (`status: "saving"` on the bubble), rather than relying on the user's restraint.
- Advance only from the server-returned assessment. Never from an optimistic
  local guess — an optimistic advance that the server then contradicts leaves the
  prompt and the persisted state disagreeing, and the user believing the wrong one.
- Auto and Manual behaviour unchanged, pinned by test.

### E4 — Bubble and progress

Presentation-only props as specified; the parent owns workflow state. On
placement: put the bubble inside the canvas overlay so it survives expand/resize
— the canvas already manages its own height, zoom, and scroll
([BBoxCanvas.tsx:97](../frontend/src/BBoxCanvas.tsx:97) onward), and anything
positioned from outside it will drift. Do not use `IntersectionObserver` for any
of the attach/reposition logic: it never fires in the Browser pane, silently, so
the behaviour cannot be verified locally — use `getBoundingClientRect` plus a
debounced scroll handler.

Chips use `var(--bbox-color)` (already threaded through
[bbox.css](../frontend/src/bbox.css:79) and settable by the user), and newly-hit
terms reuse the `new` marker treatment. Accessibility: `aria-live="polite"` on
prompt transitions, and the **full** Block in the accessible name even when only
the remaining atoms are visually emphasized — a screen-reader user hearing
`Draw box for "saisie"` has been told something false.

Progress sits immediately left of the source-language indicator at
[App.tsx:4124](../frontend/src/App.tsx:4124), inside the existing
`flex items-center justify-between` row.

#### E4.1 — The exact progress formula

"Weighted atom coverage internally, completed Blocks displayed" as originally
written is two measures pretending to be one, and they visibly disagree: a
five-atom Block at four atoms shows a nearly-full bar labelled `0/1`. Ship two
values with two jobs, and never derive one from the other:

```ts
// the BAR: continuous, so drawing a box always moves something
progressFraction = matchedAtomWeight / totalAtomWeight

// the LABEL: discrete, so the number means what a user thinks it means
completedLabel  = `${resolvedBlocks}/${totalBlocks}`
```

Atom weights, from [`GuidedAtom.kind`](../src/tofu/core/types.py:411):

| kind | weight | why |
|---|---:|---|
| `word`, `phrase`, `numeric` | 1 | the things a user is actually asked to find |
| `punctuation` | 0 | never independently locatable; must not gate completion |

`resolvedBlocks` counts `complete` + `skipped` + `user_complete`, because those
are the Blocks that no longer ask anything of the user. `review` is **not**
resolved — it still wants an answer. Skipped Blocks therefore let the workflow
finish while **never** counting as successfully located: Gate 2's recall
numerator (B) reads `status == "complete"` only, so a user skipping their way to a
green bar cannot inflate the measured recall. Those are different questions and
they get different counters.

The accessible description reports both plus the skipped count:
*"4 of 8 Blocks resolved, 1 skipped, 62% of terms located."*

The completion animation fires on `resolvedBlocks === totalBlocks`, and reads
differently when any Block was skipped — "capture finished, 1 Block skipped"
rather than an unqualified success state.

### E5 — Tests

The backend and frontend test lists in the original plan are adopted verbatim.
Add these, which the premise corrections and the review make necessary:

**Persistence (Part 0 — these fail today):**

- Autosave does not erase Blocks: `PUT /guided-blocks` → full-manifest `PUT`
  without the key → Blocks intact.
- A stale-revision `PUT` does not regress Guided progress; it is rejected.
- `candidate_lineage` survives a detect → save → reload cycle.

**Reconciliation ownership (E2.1):**

- Duplicate Blocks require distinct region evidence; one region cannot satisfy
  both.
- A region cannot silently satisfy two different Blocks.
- Manual reassignment updates coverage on both the old and the new Block.
- Fuzzy-only evidence yields `review`, never `complete`.
- Editing a region's text revokes its fuzzy/dictionary evidence and re-reconciles.
- An explicit user association is not stolen by a later automatic pass.

**Transaction (E2.2):**

- Repeating a `client_operation_id` returns the original region; no duplicate is
  created.
- The response's `guided_progress` matches a subsequent `GET /guided-blocks`.

**Progress (E4.1):**

- A skipped Block completes the workflow without entering the recall numerator.
- Punctuation-only atoms never block completion.
- Bar fraction and `n/m` label are computed independently and both appear in the
  accessible description.

**And the original five:**

1. `_normalize_ground_truth` is **not** applied to Guided Blocks — a phrase with
   internal spaces survives a `PUT /guided-blocks` → `GET /guided-blocks`
   round-trip with its spaces intact.
2. Two identical Blocks survive that round-trip as two Blocks (the dedupe path is
   genuinely not reachable from Guided).
3. `GET /guided-blocks` returns everything `PUT` returned — atoms,
   `normalized_text`, `find_all`, `detection_assessment` — so reload restores
   state rather than a subset.
4. A region marked `excluded` (not spliced) reopens its Block.
5. `POST /regions` without the Guided fields, and `POST /detect/refine` (which no
   longer takes them at all), produce responses identical to today's, byte for
   byte.

Run backend tests via `.venv/Scripts/python -m pytest` — bare `python` is the
system 3.14 here and invents phantom failures.

---

## Part F — Sequencing, risks, acceptance

### F.0 — Seven milestones, each independently reviewable

This document is a program, not a change. Implemented as one unit it would be
unreviewable, and a two-line UI fix would sit behind a human corpus review. Split
it:

| M | Milestone | Contains | Blocked by | Ships alone? |
|---|---|---|---|---|
| M1 | **Incidental Capture fixes** ✅ | C1, C2, D1–D4 | — | yes |
| M2 | **Manifest persistence contract** ✅ | Part 0 | — | yes, and fixes live data loss |
| M3 | **Block persistence** ✅ | E1.1–E1.4 | M2 | yes (Blocks survive; no Guided flow yet) |
| M4 | **Guided association + reconciliation API** ✅ | E2, E2.1, E2.2 | M3 | yes (API-only, testable without UI) |
| M5 | **Guided state machine + presentation** ✅ | E3, E4, E4.1 | M4 | yes — the feature |
| M6 | **Detection attribution** ✅ | A1.0–A1.2, A2, A3 | M2 (for A1.0) | yes, entirely parallel |
| M7 | **Empirical certification** ◐ | B1–B4 | M4 (real arm), M6 (attribution) | yes |

M1 and M2 have no dependencies and both fix live defects — start there. M6 runs
in parallel with M3–M5 throughout and shares only Part 0 with them.

**M7 does not gate M5.** Part B validates release claims; it should not hold the
core interaction hostage to a two-annotator review that is calendar time rather
than engineering time. The one exception is B3: if the oracle arm shows Guided's
ceiling sitting at the Auto baseline, the feature's premise is disproven and M5
should not be built. That is the only Part B result with veto power, and it is
runnable before M4.

### Order within that

1. **M1** — warm-up; C2 and D4 fix live defects.
2. **M2** — the blocker. Everything durable depends on it.
3. **M3** — Blocks survive persistence. Nothing in E is correct before this.
4. **M6 in parallel from here** — start with A1 rungs 3–5 (cheapest falsification
   available: it says whether the detection story is even a detection story),
   then A1.0–A1.2, then A2 + A3.
5. **B3 (oracle arm)** — Guided's ceiling, before three more milestones assume it
   is high.
6. **M4 → M5**.
7. **B1 (corpus human review)** — schedule at step 3; it is calendar time and it
   gates Gate 2 entirely.
8. **M7** — the real Guided arm and McNemar certification, once M4 exists.

### Risks

| Risk | Mitigation |
|---|---|
| **Autosave erases server-owned state** (live today, for `guided_blocks` and `candidate_lineage`) | Part 0: presence-based merge semantics + revision guard, proven per endpoint. M2 ships before anything durable is built on the manifest. |
| A newly added server-owned field repeats the same erasure | The ownership sets in 0.1 live in `manifest_store.py`, so a new field must declare which set it joins. |
| Two endpoints start contradictory reconciliations | E2.2: one canonical transaction; `block_id` removed from `/detect/refine`. |
| A retried or replayed draw creates a duplicate region | `client_operation_id` idempotency key; pointer-locking alone does not cover network replay. |
| Fuzzy matching silently completes a Block it should not | `review` status; only exact-after-normalization completes until a threshold is calibrated. |
| Skipping inflates measured recall | Recall numerator counts `complete` only; `skipped` resolves the workflow, never the metric. |
| Fitting a threshold on the eval set (has happened, shipped, regressed) | Part A changes no constants. Register records provenance; any later change needs held-out evidence. |
| Measuring a configuration the product does not run (has happened twice) | Every attribution output records engine, detector config, and Paddle-worker reachability. `run_kind` already exists for this. |
| shadcn init rewrites the design system | Vendor from a scratch checkout (D2); never run init in-tree. |
| Guided ships on a flattened Block list | E1 is a hard gate, with the round-trip tests as the check. |
| Gate 2 certified on an unreviewed corpus | `eval_guided_corpus.py` already refuses. Do not weaken the refusal to unblock a schedule. |
| Optimistic UI advance desyncs from persisted state | Advance only from server-returned assessment. |
| Guided's ceiling is lower than assumed | B3 measures it before E2–E4 are built. |

### Acceptance

- **0**: no server-owned field can be erased by a client request that does not
  name it, proven per endpoint; `candidate_lineage` survives a save/reload;
  a stale-revision write is rejected rather than applied.
- **A**: every no-box asset is assigned a rung with cited lineage node ids; the
  threshold register exists with a provenance value on every row; the
  Paddle/scene separation is a measured distribution, not an assertion. No
  constant changed.
- **B**: oracle-arm ceiling recorded; corpus review scheduled with two named
  annotators; Gate 2 run paired on the full corpus, dev/holdout reported
  separately, only after `review_status: human_reviewed`.
- **C**: recapture confirms through the ToFU overlay on all three capture modes
  and snapshots first in every one; the Blocks field is editable during
  `(prepping...)` and locked during detection, both pinned by test.
- **D**: `Kbd` vendored with its origin recorded; hint dismissal persists across
  reload; the `A`/`Del`/`Esc` keycaps render styled.
- **E**: `la première saisie` is one Block end to end; two identical Blocks are
  independently fulfillable and consume distinct regions; a phrase split across
  non-adjacent regions completes; matching `saisie` alone leaves the Block
  partial; a repeated `client_operation_id` adds no second region; an autosave
  between two draws loses nothing; reload restores the active Block and progress;
  Auto and Manual are byte-identical to today.

### What this plan does not do

It does not raise detection recall. Part A is instrumentation and honesty; the
remedies it points at — polygon-aware scoring, tiling, a different detector,
fine-tuning — are separate, larger work, and choosing between them from an
outcome metric is the guessing this project has already paid for twice.
