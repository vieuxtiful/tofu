# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added — ToFU Vision 2, first increment
- **`cicerone.detect(seed_detections=...)`** — replace the detector with
  supplied geometry and run the normal recognition pipeline over it. Built for
  the Guided oracle. It is the right seam — crop-based attempts measured the
  crop path rather than the pipeline, and on CJK that machinery is nearly the
  whole read (22% from a crop vs 65–78% from the pipeline) — though the oracle
  itself turned out not to be answerable; see "Measured and rejected".
- **`layers/decant.py`** — evidence survival, kept separate from candidate
  ranking. Answers "is there enough here to justify choosing *any* reading?",
  which a ranking cannot: a candidate can lead because every alternative was
  eliminated. Returns a state (`absent` / `weak` / `partial` / `present` /
  `unknown`) plus the measurements behind it, and **deliberately no score** —
  weighting is the ranker's job, and `layers/ticket.py` records why the ranker
  cannot be priced until reviewer outcomes exist. `unknown` is never a synonym
  for `absent`.
- **`scene.substrate()`** — samples the surface a text region sits on with
  every glyph mask excluded, using `SceneRegion.polygon` and each
  `segmentation_mask`. `cleanse` currently estimates fills from a fixed 14px
  ring, whose own source notes it can sample an adjacent sign; on dense signage
  it also samples neighbouring glyphs. Surface-minus-glyphs is both larger and
  cleaner, and it is the "surrounding surface" a degradation model needs.
  Observational only — it changes no repair, and returns `None` rather than a
  fabricated estimate when no sample survives.
- `docs/vision-2-assessment.md` — the program assessed against measured
  evidence, including a correction to its own §1 after the oracle re-run showed
  crop-read legibility is materially pessimistic for CJK.

### Measured and rejected
- **A Guided localization oracle is not answerable in this architecture.**
  Three attempts — crops at `pad=0` (42.3%), the backend's crop path (57.1%),
  and the full pipeline via `seed_detections` (64.0%) — all scored below the
  Auto arm's 67.2%, which an oracle cannot. The pipeline's merge and assembly
  stages rewrite supplied geometry, and those same stages are what make text
  readable, so localization cannot be replaced on its own. Indexed in
  `docs/measured-dead-ends.md`.

### Fixed
- **Guided Blocks are no longer flattened before `mise` sees them.** Entries
  were split on whitespace by the frontend and split *and deduplicated* again
  by the server's `_normalize_ground_truth`, so `la première saisie` reached
  the Block model as three things and `PARIS PARIS` as one. Guided now writes
  entries verbatim to `/guided-blocks`; `ground_truth` keeps its
  split-and-dedupe semantics, which are correct for the recogniser's term
  pool and are still populated from the same entries.
- `GET /api/assets/{id}/guided-blocks` returned strictly less than `PUT` did
  (no atoms, `normalized_text`, `find_all`, or `detection_assessment`), so a
  reload could not restore Guided state. Both routes now share one payload.
- **Manifest autosave no longer deletes server-owned state.** `PUT
  /api/manifest/{id}` replaces the whole document, and the frontend rebuilds
  it from a type that models only the fields the UI edits — so `guided_blocks`
  was erased ~1.5s after any region edit. Server-owned fields declared in
  `manifest_store.SERVER_OWNED_FIELDS` are now carried forward when a payload
  does not *mention* them; an explicit `[]`/`null` still clears, and
  `DELETE /api/assets/{id}/guided-blocks` clears deliberately.
- **Eight per-region fields were dropped on every save.** `_inst_to_dict`
  omitted `scene_eligibility`, `review_features`, `lineage_candidate_id` and
  the five video-identity fields, so `ticket.py`'s output never reached disk,
  the scene filter's "on the record either way" verdict was not, and — worst —
  the stamp joining a shipped region to its okara node was lost, leaving the
  newly-persisted lineage graph unattributable after a reload. A test now
  parametrises over the dataclass, so a new field fails a test rather than
  losing a user's work.
- **`candidate_lineage` is persisted.** okara's proposal graph had no key in
  `_manifest_to_dict`, so it was dropped by the first `save_manifest` after
  detection and had never reached disk; every recorded lineage figure came
  from an in-process harness run.
- Recapture in **Manual** mode cleared the manifest with no confirmation and
  no snapshot. All capture modes now snapshot before clearing.
- The Ground Truth/Blocks field was locked during the `(prepping...)`
  pre-flight scan, which is the slowest step between upload and Capture. It
  now locks during detection only.
- The `A` / `Del` / `Esc` keycaps in Capture used daisyUI's `.kbd` classes,
  which have been undefined since daisyUI was removed — they rendered as bare
  text.

### Added
- **`GET /api/manifest/{id}/detection-attribution`** — names which of six
  events left a Capture tab empty (no engine / no proposals / all suppressed /
  all excluded / regions present / no lineage), from the lineage the shipping
  run recorded. Recomputes nothing: a fresh detector pass is a different run.
  The Capture tab shows the rung and which stage suppressed how many
  candidates, replacing one sentence that covered all six cases.
- **`docs/threshold-register.md`** — every hard constant in the detection path
  with a provenance label (`measured` / `reasoned` / `inherited` / `invented`).
  Detection tally: 4 measured, 10 reasoned, 3 inherited, majority invented.
- **`docs/detection-attribution-report.md`** — on the one fully-annotated
  fixture, CRAFT separates text from text-free scene surfaces completely
  (median 1.013 vs 0.005, min 0.995 vs max 0.311); all nine of `la-bastille`'s
  regions are `proposed-then-lost` or `boundary`, none `no-activation`. The
  failure there is grouping, not text/scene discrimination.
- `eval_detector_evidence.py` gains a **negative arm** (detector response over
  text-free scene surfaces — the other half of any discrimination claim), a
  **Paddle probe** that records reachability *and its reason* on every run, and
  a `negatives_trustworthy` flag that withholds the claim on partially-annotated
  fixtures where unlisted text is real text.
- `eval_guided_corpus.py` gains `--arm both` (both arms scored from **one**
  detection pass, so Gate 2's pairing is exact by construction) and
  `--arm guided_oracle` (annotated boxes handed over, only the read scored —
  intended as a ceiling; see the caveat below). Per-occurrence outcomes are now
  recorded, not just totals.
- **`scripts/compare_guided_arms.py`** — Gate 2 as a paired McNemar test with an
  exact binomial, no scipy. First result: Guided 74.6% vs Auto 67.2%, **+7.4
  pts, 14 discordant wins and 0 losses, p = 0.00012**. Not certified: the margin
  is 0.6 pts under the declared +8.0 and the corpus is still `machine_derived`.
  See `docs/gate2-status.md`, and "Measured and rejected" above for the oracle
  arm's three runs and why it does not yield a ceiling.
- **`docs/guided-corpus-review-protocol.md`** — the two-annotator review that
  unblocks certification, with the occurrence budget that constrains it (9 to
  spare before the corpus falls below its own declared minimums).
- **Guided capture flow.** Pressing Capture in Guided mode now routes straight
  to drawing instead of launching automatic detection — the behaviour the mode
  was missing, which left entered Blocks unused. Draw mode stays armed between
  boxes, with a pointer lock over the refine→add round trips so a second box
  cannot start mid-flight; the prompt shows "reading…" while it holds. The
  active Block, coverage and status all come from the server's persisted
  assessment, so a reload resumes where the user was.
- **`GuidedPromptBubble`** — pink prompt inside the canvas card: `Draw box for
  “釁”` / `Draw box(es) for “la première saisie”`. A partial phrase strikes
  through what was found and highlights what remains in `--bbox-color`, but
  never re-words the prompt down to the remaining atoms — `Draw box for
  “saisie”` would assert that `saisie` is a thing to find in its own right.
  The accessible name always carries the whole Block, and carries "mark found"
  and "skip" as escape hatches.
- **`GuidedProgress`**, immediately left of the source-language indicator. The
  bar moves on atom coverage and the label counts Blocks, computed
  independently: deriving either from the other produces the nearly-full bar
  labelled `0/1` that a five-atom Block would show at four atoms. Skipped
  Blocks are named in the accessible label.
- `frontend` dev server takes `PORT` (default 5173) and `.claude/launch.json`
  sets `autoPort`, so two checkouts can run at once without one silently
  attaching to the other's server.
- **`layers/aboyeur.py`** — reconciles requested Blocks against captured
  regions. Enforces ownership rather than mere string matching: one region
  has at most one Block owner, duplicate Blocks consume distinct evidence,
  explicit user association outranks inference permanently, soft-excluded
  regions withdraw their coverage, and fuzzy evidence reaches a new `review`
  status but never `complete` (the score that would justify completing on a
  similarity figure is not calibrated). Geometry is supporting evidence, never
  a requirement, so a phrase spanning two faces of a sign still completes.
- Guided draws are **one transaction**: `POST /api/manifest/{id}/regions`
  takes `guided_block_id` and `client_operation_id`, then adds, associates,
  reconciles and persists in a single write, returning the region plus the
  updated Guided progress and active Block. Repeating an operation id returns
  the original region. Auto and Manual responses are unchanged — the `guided`
  envelope is present only when the asset has Blocks.
- `POST /api/assets/{id}/guided-blocks/{block}/resolve` — the escape hatch.
  A Block can be marked complete or skipped by hand, recorded separately from
  `status` so no later automatic pass overturns it; skipping resolves the
  workflow without counting as a successful location.
- Reconciliation re-runs on every route that can change coverage: add, text
  edit, exclusion, merge, and full-manifest save (undo/redo/snapshot restore).
- **`BlocksField`**: the Guided entry surface. Blocks commit on Enter into an
  ordered chip strip — duplicates preserved, phrases shown whole — instead of
  reusing Auto's whitespace-delimited field, whose per-word colouring would
  announce that a phrase had already been broken apart. Enter never commits
  mid-IME-composition, where it confirms a candidate rather than ending the
  entry. Re-saving the list carries each Block's `detection_assessment` with
  it by text, as a per-text queue so duplicates cannot inherit each other's
  progress.
- `ConfirmOverlay` replaces three near-identical confirm components, and
  recapture now asks "begin recapture?" through it instead of a native
  `confirm()`.
- Dismissable **Dictionary = ⌨ Ctrl + Space** hint in the Translate step,
  with the dismissal persisted; names the `Alt+↓` alternate in its accessible
  label, since macOS intercepts Ctrl+Space.
- `Kbd`/`KbdGroup` vendored from shadcn/ui (registry `kbd`, new-york), plus a
  `cn` helper and the `@/` path alias in tsconfig *and* vite config.
- Opt-in stale-write guard on `PUT /api/manifest/{id}`: send
  `expected_revision` and a mismatch is a 409 instead of a silent overwrite.
  The response now returns `revision`.
- Docker deployment: multi-stage Dockerfile, Dockerfile.paddle,
  Dockerfile.inpaint, Dockerfile.frontend, docker-compose.yml with
  profiles for CJK and GPU inpainting sidecars.

### Changed
- Version bumped to 1.0.0 (Production/Stable).
- FastAPI app version synchronized with library version.

## [1.0.0] - 2026-08-04

First production release. All Phase 1-5 work is included here.

### Added
- **Video pipeline**: temporal tracking edge cases — occlusion handling
  (`MAX_GAP_FRAMES`), rapid text change splitting
  (`TEXT_CHANGE_SPLIT_THRESHOLD`), birth/death smoothing
  (`MIN_CONFIRM_OBSERVATIONS`).
- **VideoWorkspace frontend**: timeline scrubber with keyframe/issue markers,
  track list with confidence and keyframe count, per-track style editing
  (opacity, color), side-by-side preview comparison, progress indicator with
  cancel/resume.
- **Video integration tests**: 13 end-to-end tests covering the full job
  lifecycle (create → timeline → track update → keyframe → cancel/resume →
  error paths).
- **API reference**: `docs/api.md` generated from FastAPI OpenAPI schema via
  `scripts/generate_api_docs.py`. 80 endpoints grouped by 24 tags.
- **Architecture diagram**: `docs/architecture.md` with Mermaid diagrams for
  system overview, image pipeline, video pipeline, temporal tracking states,
  three-venv architecture, and data flow.
- **CONTRIBUTING.md**: development setup, code style, test conventions, PR
  checklist, release process.
- **CI**: API docs freshness check added to `ci.yml` typecheck job.
- **Benchmark CI**: nightly `benchmark.yml` workflow for regression metric
  drift detection.
- **Frontend CI**: `frontend-test` job added to `ci.yml` for vitest.
- **E2E server tests**: 15 tests covering full lifecycle and error paths.
- **Frontend component tests**: 27 tests for Stepper, RegionTable,
  SemanticSubstitutionPanel.
- **Video regression baselines**: 4 SSIM-based tests with 27 baseline PNGs.

### Changed
- FastAPI app enriched with description and route tags for OpenAPI grouping.
- `detect_text_change` threshold raised from 0.35 to 0.6 to correctly split
  short-text swaps ("SALE" → "CLOSED") without splitting OCR noise.
- `detect_text_change` returns False for None inputs (missing readings are
  not content changes).
- `MISSED_KEYFRAMES_BEFORE_RETIREMENT` kept at 2 to preserve resume test
  compatibility.

### Fixed
- `drain()` now correctly manages `dormant` and `pending` tracks, recreating
  tracks from dormant state and preserving consensus history across
  occlusion gaps.
- TypeScript `TS2322` error in `SemanticSubstitutionPanel.test.tsx` resolved
  by explicitly typing `baseProps`.

## [1.0.0] - 2026-01-15

### Added
- **7-layer image pipeline**: ToFU (pre-flight validation), Scene (semantic
  context), Cicerone (text detection + OCR), Cleanse (text erasure +
  inpainting), Scribe (style-aware text rendering), Verify (quality
  verification), Memory (visual translation memory).
- **Auxiliary modules**: Savor (post-recognition glyph correction), Wasabi
  (CJK glyph normalization), Menu (gazetteer-assisted place-name recovery).
- **Three-venv architecture**: main venv (torch + EasyOCR + FastAPI),
  PaddleOCR venv (CJK detection), inpaint venv (LaMa neural repair).
- **Video pipeline**: Braise adaptive keyframe OCR tracker, compositor with
  preview/export parity, resumable checkpoints.
- **FastAPI server**: 81 REST endpoints for asset management, detection,
  translation, rendering, verification, video jobs, projects, snapshots,
  translation memory, glossary, and semantic units.
- **React frontend**: Vite + MUI + Tailwind, 5-step workflow (Upload →
  Capture → Translate → Render → Verify), VideoWorkspace for video editing.
- **Open VTM format**: open visual translation memory format v1.0 (MIT).
- **Evaluation harnesses**: `scripts/eval_*.py` for detection, render,
  cleanse, font match, memory, savor, paddle, verification corpus.
- **XLIFF/VTM interchange**: import/export for translation memory and
  industry-standard XLIFF format.
- **Regression baselines**: per-image metric baselines for detection,
  rendering, and verification quality.
- **Manual bounding-box snap**: click-to-snap region boundaries to text edges.
- **LLM comparison**: side-by-side comparison with Gemini and ChatGPT output.
- **Semantic substitution**: Basil semantic units for context-aware
  translation of signs, menus, and recurring text.
- **Region merging**: manual region merge with OCR fallback for fragmented
  detections.
- **Perspective quad**: perspective-aware text erasure and rendering for
  non-planar surfaces.
- **Font matching**: visual font matching with weight/italic/color profiling.
- **Multi-candidate OCR**: multiple OCR candidates per region with
  confidence scoring and arbitration.
- **Glossary**: project-level glossary for consistent terminology.
- **Project management**: multi-project support with asset organization.
- **Snapshot system**: save and restore translation session state.

### Known Limitations
- Video analysis requires FFmpeg; degrades gracefully when absent.
- PaddleOCR CJK detection requires separate `.venv-paddle`; degrades to
  reduced CJK recall when absent.
- Neural inpainting (LaMa) requires separate `.venv-inpaint` and CUDA;
  falls back to non-neural inpainting when absent.
