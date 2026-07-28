# Static-image release readiness — feasibility assessment and revised plan

Assessment of the "prove and harden the static-image product" proposal, checked against the
working tree on 2026-07-26.

**Verdict: adopt the strategy, reject the stated baseline, rescope Phases 0/3/4.**

The strategic core — measure before building, classify each gap before scheduling it, refuse
learned models without demonstrated necessity, and split video off — is right and should be
kept verbatim. But four of the proposal's factual premises are wrong, and roughly a third of
the work it schedules already exists. Executing it as written would spend its first two
phases re-deriving things that are in the repository now.

---

## 1. Baseline claims, verified

| Claim | Status | Evidence |
|---|---|---|
| "Python tests could not run because `pytest` is unavailable" | **False** | `pytest` resolves in *both* interpreters — `.venv` 9.1.1 and system 8.0.1. `pyproject.toml:58` declares `dev = ["pytest>=8.0"]`; `server/requirements.txt:73` pins 9.1.1. The full suite runs: **649 passed** in ~131s. |
| "Frontend production build passes" | **True** | `vite build` succeeds in 863ms. |
| "Bundle emits a non-blocking size warning" | **True** | 832.03 kB (244.05 kB gzip), over the 500 kB advisory. |
| "34 modified tracked files plus new evaluation artifacts" | **Substantially true** | 35 modified tracked, 28 untracked — of which only **7 are source**; the rest are eval artifacts. |

The pytest claim is the consequential one. It is the stated blocker for Phase 0's exit
criterion, and it does not exist. **The baseline is already green.** Phase 0 is not a phase;
it is an afternoon.

## 2. Work the proposal schedules that is already built

| Proposal item | Reality |
|---|---|
| Phase 4.5 "implement windowed SSIM ... first" | Already implemented. `verify.py:172` `_ssim()` — single-window SSIM, Wang et al. 2004 eq. 13, applied over a `RING_PX = 12` background ring, weighted `SSIM_WEIGHT = 0.3`. |
| Phase 4.5 "avoid relying exclusively on the same OCR family" | Partly addressed. Verify already runs **two independent families**: OCR round-trip (SRNet/STEFANN protocol, `verify.py:13-15`) *and* structural SSIM + edge-density/chroma (`verify.py:272`). The residual — detector and verifier sharing one OCR engine — is real but narrower than stated. |
| Phase 0.5 "record backend availability" | Endpoint pattern exists: `/api/semantic/providers` and `/api/inpainting/providers`, both over `provider_statuses()`. Needs extending to Paddle/SAM/shaping, not inventing. |
| Phase 0.4 "VTM conformance tests" | Exist and pass: **49 tests**, with `spec/conformance/{valid,invalid,index.json}`. |
| Phase 3 "build a compact golden corpus" | Substantially exists: 7 synthetic fixtures with exact `.gt.json`, 3 real street images with detection ground truth, and **8 evaluation harnesses** (`scripts/eval_{detect,tofu,render,savor,memory,paddle,cleanse_providers,font_match}.py`). |
| Phase 6 "video is scaffolding" | Correct, and already explicit: `pipeline.py:121` rejects video outright. |

## 3. What the proposal missed

**The completeness report is itself uncommitted.** `docs/cross-layer-architecture.md` is
untracked. The "~75% complete" figure being treated as a dated inventory is not in git
history at all — and the document *already says so about itself*: §2.1 (ToFU) and §2.2
(Cicerone) were re-audited against the implementation; §2.3–2.12 and §3 explicitly were
not, and carry provisional asterisks. The "report is partially stale" hypothesis is
confirmed by the report.

**The real risk is loss, not staleness.** 35 modified tracked files and 7 untracked source
files are uncommitted, in a repository that has already lost untracked files to `git clean`
once. The proposal says "preserve the current dirty worktree" as a caveat. It is step one.

**No fixture stores a regression tolerance.** The harnesses write timestamped JSON reports
and comparison is manual, by eye, across `--tag` pairs. Nothing fails when a metric moves.
This — not corpus size — is the actual measurement gap.

---

## Revised plan

### Slice 0 — Secure the work (hours)

1. Commit the worktree on a branch, source separated from eval artifacts (28 untracked files,
   only 7 of them source — the split is mechanical).
2. Decide whether `scripts/eval_out/**` belongs in git or in `.gitignore` with a retained
   index. It is currently ~190 JSON reports plus PNG overlays.
3. Record the green baseline: 649 pytest, 27 vitest, 49 VTM conformance, `vite build` clean,
   dependency versions, and which optional backends resolved.

*Exit:* nothing is only on disk.

### Slice 1 — Traceability matrix (the primary deliverable)

Keep the proposal's schema and disposition taxonomy unchanged — they are good. Two additions:

- A column for **"already addressed by uncommitted work"**, which is the axis on which this
  report is actually stale.
- Scope it: 12 layer sections plus 9 auxiliary sections in the architecture doc — 21 rows,
  not an open-ended audit.

Replace percentages with the proposal's status vocabulary. Start from the document's own
admission: §2.1 and §2.2 are audited; everything else is provisional.

*Exit:* every observation has an evidence-backed disposition.

### Slice 2 — Static-image release contract

Adopt as proposed, with one constraint: **every clause names the test that proves it.** A
contract clause with no test id is a wish. This is the highest-value net-new document in the
plan; nothing resembling it exists.

### Slice 3 — Close corpus gaps and add tolerances (rescoped)

Do **not** build a corpus. Extend the one that exists.

Missing coverage, against the proposal's list: RTL, Indic/SEA shaping, perspective
distortion, multicolour text, mixed orientation, low-confidence/hallucinated OCR. Six
fixtures, generated deterministically by `scripts/make_fixtures.py` in the established
pattern (image + exact `.gt.json`).

The infrastructure gap is **stored tolerances**: a checked-in expected-metrics file per
fixture and a test that fails when a harness result drifts outside it. Small, and it is what
converts eight report generators into a regression suite.

*Exit:* a metric regression fails a test rather than requiring someone to notice.

### Slice 4 — High-leverage static gaps

The proposal's ordering is sound. Amended:

1. **Shared mask reliability** — endorse as #1 without reservation. `imaging.text_mask()`
   bottoms out six layers (§4.5); its own doc grades it 82% and names multicolour and
   degraded ink as the weakness. `tests/test_imaging_threshold.py` already has the
   IoU-against-a-baseline test pattern to extend.
2. **Geometry and orientation** — endorse.
3. **Shaping** — smaller than stated. `knead.py` already wraps HarfBuzz + FreeType, and
   scribe already does UAX #9 bidi via python-bidi. This is *verify and define failure
   paths*, not build.
4. **Cleanse self-assessment** — endorse. Genuinely missing, and the natural consumer of
   Verify's existing metrics before the render rather than after.
5. **Independent QA** — restate: SSIM is done. The remaining item is detector/verifier
   engine independence.
6. **Language and domain correction** — endorse, including versioned data files behind the
   existing evidence gates.
7. **Memory operations** — endorse as last.

### Slice 5 — Learned components, gated

Adopt verbatim. The requirement to produce baseline failure fixtures *before* proposing a
model is the single most valuable rule in the proposal.

### Slice 6 — Video as a separate epic

Adopt verbatim. `pipeline.py` already rejects video, so there is no ambiguity to resolve.

---

## Recommended first slice

Narrower than the proposal's, because its verification step is already satisfied:

1. Commit the worktree (source and artifacts separated).
2. Record the green baseline — it exists today; capture it before anything moves.
3. Produce the traceability matrix, marking items already closed by the committed work.
4. Draft the release contract's table of contents only, to expose which clauses have no test.

No corpus work, no learned models, no video, and no `text_mask` changes until the matrix says
where the measured failures are.
