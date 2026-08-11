# ToFU Vision 2 — assessment against measured evidence

Reviewed 2026-08-11, against the M6/M7 measurements taken this week.

**Verdict: the program's priority ordering is correct, and I can now say so from
measurement rather than taste.** Track A is aimed at ToFU's actual residual
bottleneck. There is one mathematical error in the unifying model, one
sequencing hazard that would stop Phase 0 from ever exiting, one gate-design
flaw that has already bitten this project once this week, and five integration
constraints that are only visible from inside the code that just shipped.

---

## 1. The premise check — and a correction to my own prior

I expected to argue that Track A targets a failure mode this corpus does not
exhibit. M6 had just found that on `la-bastille-1789` **all nine ground-truth
regions are `proposed-then-lost` or `boundary` and none is `no-activation`** —
the detector sees everything at peak ≥ 0.91 and the pipeline loses it. That
reads as a grouping problem, not a degradation problem.

**That inference was wrong, and the data correcting it was already in the repo.**
`crop_legibility` in `eval_detector_evidence.py` reads each ground-truth box
through the backend's own crop path — pad 4, upscale to `MIN_CROP_HEIGHT` —
and scores edit distance to the known answer. Across 34 GT regions in four
fixtures:

| fixture | regions | unreadable (NED ≥ 0.8) | median NED |
|---|---:|---:|---:|
| `avenue-de-la-république` | 2 | 0 | 0.188 |
| `gemini-street` | 16 | **13** | 1.0 |
| `japan-street` | 7 | **6** | 1.0 |
| `la-bastille-1789` | 9 | **5** | 0.875 |
| **total** | **34** | **24 (71%)** | — |

Detection verdicts over the same 34: `proposed-then-lost` 19, `boundary` 9,
`fragmented` 4, `weak-activation` 2, `no-activation` 2.

So: **detection sees ~94% of it, and recognition fails on ~71% of it even when
handed a perfect box.** The residual bottleneck is recognition from hard and
degraded glyphs, which is precisely what Track A addresses. The program's
"recommended priority" — degradation-invariant matcher first, evidence-survival
second, Scene material third — is supported.

Caveats, stated because they bound the claim: n = 34; two of the four evidence
files (`gemini-street`, `japan-street`) date from 2026-08-09 and predate this
branch's `cicerone.py` changes; and one asset (`avenue`) shows clean
recognition, so this is a property of the *hard* strata, not of the pipeline
everywhere.

### Correction: the CJK part of this table is confounded

The Guided-oracle re-run (below) measured how a per-crop read compares with the
full pipeline, per stratum. On CJK the crop path reaches **22%** where the
pipeline reaches **65–78%** — the multipass and zoom passes carry almost the
entire read. `crop_legibility` is a per-crop read.

So `japan-street`'s 6-of-7 "unreadable" is substantially an artefact of the
measurement, not a statement about the image, and **the 71% figure above is
inflated.** The trustworthy part of the table is the Latin-ish material, where
the oracle showed the crop path is comparable to the pipeline (`latin_horizontal`
−7.7, `mixed_script_numeric` +1.8, `nonlinear_irregular` +4.9): there,
`gemini-street` 13/16 and `la-bastille` 5/9 stand — **18 of 25 unreadable from a
perfect box.**

The conclusion survives the correction, on narrower evidence: recognition, not
localization, is the residual bottleneck on the Latin and mixed strata. For CJK
it is currently unmeasured, and any Vision 2 evaluation that reads CJK from a
crop will understate the baseline it is trying to beat.

### The same finding condemns the oracle-arm design — and it did

Re-run with the pipeline's crop path instead of `pad=0`, the Guided-oracle arm
moved **42.3% → 57.1%** (a 14.8-point measurement artefact) and is **still below
both real arms**. That settles the structural question: **there is no way to
isolate localization by re-reading crops, because the crop path is itself the
weak recogniser.**

Where the oracle *is* trustworthy — `mixed_script_numeric`,
`nonlinear_irregular` — it beats Auto by only +1.8 and +4.9 points. On
`nonlinear_irregular`, the corpus's worst stratum, perfect localization still
leaves ~63% of requested occurrences unfound. That is independent support for
Track A: on the strata where the measurement can be trusted, the residual loss
is recognition.

That injection was then built (`cicerone.detect(seed_detections=...)`) and run:
**64.0%, still below Auto.** The cause is not padding and not a disabled stage —
both were tested. The pipeline's merge and assembly stages *rewrite supplied
geometry* (three seeded boxes emerge as one region on `cjk-vertical-menu`), and
those stages are also what make text readable.

**So localization and recognition are entangled in this architecture, and no
configuration replaces localization alone.** That is a finding Vision 2 should
absorb directly: the program's clean split — Cicerone for identity, Scene for
process, fused at the end — assumes the two can be reasoned about separately.
At least for CJK they currently cannot, and a Track A evaluation that holds
geometry fixed will not behave the way the design diagram implies. Full numbers
in `docs/gate2-status.md`; the negative result is indexed in
`docs/measured-dead-ends.md`.

## 2. The unifying model double-counts the observation

§1 proposes

> P(G | X, GT) ∝ P_Cicerone(G | X, GT) · P_Scene(X | G, M, S, 𝒫, D, L, V) · P(G | GT, language)

This is not Bayes. `P_C(G | X)` is already a posterior: it conditions on X and
already contains a prior over G. Multiplying it by the generative likelihood
`P_S(X | G, …)` uses the observation twice, and multiplying by `P(G | GT, lang)`
puts the prior in twice as well. The product is systematically overconfident,
and overconfidence is the one failure this program says it exists to prevent.

Two clean repairs:

1. **Divide the prior out.** Use Cicerone's likelihood ratio
   `P_C(G | X) / P(G)` in place of the posterior, which recovers a proper
   product with the Scene likelihood — at the cost of needing `P(G)` explicitly.
2. **Stop calling it a posterior.** §C3's log-linear form with weights
   `w_c, w_s, w_l, w_o` is a *product of experts* / calibrated scorer, which is
   a perfectly respectable thing and is what the program actually builds. Weights
   fitted by calibration absorb the double-counting.

**Recommendation: take (2), and delete the §1 equation's claim to be a
posterior.** It costs nothing — C3 is already the operative formula — and it
matters because the entire safety argument rests on calibration. A calibration
target derived from a mis-specified posterior is not a calibration target.

Related: `S(g)` in C3 subtracts `w_n·C(g)` for contradiction. Contradiction
evidence is not independent of the support terms — the same pixels generate
both — so `w_n` must be fitted jointly, not set. See §5(e).

## 3. Phase 0 would not exit as written

Phase 0 adds material, process and degradation labels to the fixtures. The
existing annotation debt makes that unreachable in 2–3 weeks:

- **Nine of eleven detection fixtures are `partial: true`.** Only
  `avenue-de-la-république` and `la-rue-sans-nom-example` are complete. Corpus
  precision is *undefined* for this reason, and the discrimination measurement in
  `docs/detection-attribution-report.md` is answerable on exactly two assets.
- **The Guided corpus is `machine_derived`** and needs the two-annotator review
  in `docs/guided-corpus-review-protocol.md`, which has only **9 occurrences of
  slack** before the corpus drops below its own declared minimums.
- **`la-bastille-1789`'s own annotation note concedes its word-level premise is
  false** — the detector now emits line-scale boxes and the annotation is fitted
  to a defect that no longer exists.

Layering three new label dimensions on top of that produces a benchmark whose
text ground truth cannot support the claims the new labels are for.

**Recommendation: split Phase 0 in two.** Phase 0a clears the existing text
debt — complete the partial annotations, re-annotate `la-bastille` at line
level, run the Guided corpus review. Phase 0b adds material/process/degradation
labels *to the completed set*. Phase 0a is already on the critical path for two
other workstreams, so this is sequencing rather than new work.

One cost saving: material and process are properties of a **surface**, not of a
region. They belong on `scene_regions`, where there are far fewer of them and
where `scene.analyze_regions` already produces the objects to hang them on.

## 4. The ±8-point gate is the wrong instrument, and it just proved it

H₁C asks for "≥8 absolute points". Gate 2 asked for +8pp and measured **+7.4pp
with McNemar p = 0.00012 on 14 discordant wins and 0 losses** — an effect that
is real beyond reasonable doubt and misses the bar by 0.6 points. The bar was
pre-registered so it stands, but repeating the same instrument will repeat the
same outcome.

**Recommendation:** pre-register the *test*, not only the margin.

- State the minimum detectable effect for the corpus size before running.
- Report "is it real" (paired test) and "is it enough" (margin) as two separate
  verdicts. `scripts/compare_guided_arms.py` already does exactly this and is
  the template to reuse — it prints significance and margin as independent
  blockers.
- Prefer paired designs throughout. Both arms scoring one detection pass is what
  made Gate 2's 14-0 discordance visible at all; two independent runs would have
  shown ±15.5pp intervals and concluded nothing.

Also: "green precision ≥ 98%" presumes today's green means something.
`CONTENT_PASS_THRESHOLD = 0.85` and its siblings are labelled **`invented`** in
`docs/threshold-register.md`. Calibrate green before writing a precision target
onto it.

## 5. Integration constraints from the code that just shipped

**(a) New evidence objects must be declared server-owned, or autosave deletes
them.** `MaterialEvidence` and `GlyphMatchEvidence` living "inside the
manifest's per-instance evidence/provenance fields" hits a boundary that was
losing data until this week: the frontend rebuilds the whole manifest from a
type that models only what the UI edits, and anything absent from that type was
erased ~1.5s later. Both `guided_blocks` and `candidate_lineage` were victims.
Any new field must be added to `manifest_store.SERVER_OWNED_FIELDS`; the
round-trip test parametrises over that tuple, so the omission fails a test
instead of losing a user's work.

**(b) Hang the evidence off the lineage graph, not beside it.** `okara`'s
`CandidateGraph` is append-only, records `operation_features` per node, and is
built on the rule that *a merge may create a candidate but must never destroy
its inputs*. That is the same discipline B5 asks for ("restoration is evidence,
not source truth"). Re-inventing a parallel evidence store next to it would give
the same run two provenance vocabularies — which is exactly the failure the
`run_kind` field exists to prevent.

**(c) `evidence_survival` (A4's q_visible) is already half-built.**
`eval_detector_evidence.classify()` returns `no-activation` / `weak-activation`
/ `proposed-then-lost` / `fragmented` / `boundary` from CRAFT's peak response,
and `crop_ned` measures whether the text is legible from a perfect box. Those
two are the empirical basis for q_visible, and §1 above is the first table of
what they say. Start from them.

**(d) There are already two abstention vocabularies. Do not add a third.**
`ocr_quality.state ∈ {review_required, unresolvable}` and — new this week —
`aboyeur`'s `review` status, which exists precisely so uncertain evidence has
somewhere to land that is neither "incomplete" nor a false claim of completion.
Vision 2's green/amber/red/indeterminate should map onto those, and it should
inherit the rule already established there: **uncalibrated similarity may
produce `review`, never `complete`.**

**(e) Every new constant goes in the threshold register.** The program
introduces λ₁–λ₅, `w_c/w_s/w_l/w_o/w_n`, an OT ε, and decision thresholds —
roughly a dozen numbers. The register's current tally for detection is 4
`measured`, 10 `reasoned`, 3 `inherited`, **majority `invented`**. A program
whose safety story is calibration cannot add twelve hand-set weights. Fit them,
and record provenance per constant.

## 6. What to defer

- **B3/B4, surface-conditioned forward rendering.** Highest research risk in the
  program, and §1 says its value is as a *recognition* prior — which is Track A's
  job and much cheaper. Gate it behind Phase 1 showing that degradation-invariant
  matching moves the 24-of-34 unreadable figure. If Track A closes most of that,
  B4's marginal value is small; if it does not, B4 has a measured target.
- **Hyperbolic embeddings.** Already correctly scoped as a fifth ablation. Keep
  it there.
- **The Word2World transfer.** The document already concedes ToFU needs a
  physical generative model rather than a social hierarchy, and that the fusion
  weights must not be inherited. Worth keeping that explicit, because a citation
  next to a weighting scheme invites exactly that inheritance later. (I have not
  read that paper — it is outside this repository — so this is a note about how
  the citation reads, not about its content.)

## 7. Revised order

| # | Work | Why here |
|---|---|---|
| 0a | Clear the text-annotation debt: complete partial annotations, `la-bastille` at line level, Guided corpus review | Already on the critical path for Gate 2 and for corpus precision; everything below inherits it |
| 0b | Material / process / degradation labels **on the completed set**, attached to `scene_regions` | Cheap once 0a exists; unreachable before |
| 1 | **A4 evidence-survival + calibration** — promoted from second to first | It is the safety mechanism, it is half-built (§5c), and it is what makes every later claim checkable |
| 2 | A1/A2 degradation-invariant matcher | The measured bottleneck (§1) |
| 3 | B1/B2 material + process taxonomy, observational only | World context without hallucination |
| 4 | A5 entropic component alignment | Targeted at segmentation damage; narrower |
| 5 | B3/B4 forward model | Gated on Phase 1's result |
| — | A valid localization oracle (§1) | Needed before anyone claims a ceiling |

## 8. What is right, and worth saying plainly

- The separation of duties — Cicerone discriminative, Scene generative, Verify
  adjudicating — is clean and maps onto layers that already exist.
- **B5 is the most important paragraph in the document.** "A diffusion or
  super-resolution model must not silently feed hallucinated strokes back as
  observed evidence" is the correct constraint, it matches the existing
  `repair_provenance` discipline, and it is the thing most likely to be quietly
  dropped under schedule pressure.
- Recording contradiction alongside support (A3) is a genuine improvement over a
  single score, and it is the right shape for a system that has to abstain.
- Insisting a candidate "cannot be green unless sufficient positive evidence
  survives" is exactly the invariant that stops elimination-by-exhaustion from
  manufacturing certainty. It is also the invariant `aboyeur` adopted this week
  for a much simpler problem, which is mild evidence it generalises.
