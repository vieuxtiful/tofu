# Pre-implementation review: MT, TM/history, and video

Reviewed source: `.windsurf/plans/ocr-inpainting-enhancements.md`  
Purpose: identify required plan amendments before implementation

## Recommendation

Proceed with Part 1 and Part 2 after the blocking amendments below are added.
Treat Part 3 Phase 1 as an analysis/prototype pipeline, not a production
localized-video release. Independent per-frame inpainting is expected to
flicker, and a sampled frame set cannot simply be concatenated at the source
frame rate.

## P0 — Amend before implementation

### 1. Machine-translation confidence is not a shared probability

DeepL, Google, LLM, and TM scores do not expose a common calibrated confidence
scale. Some providers expose no segment confidence at all. Consequently:

- rename provider output to `raw_score` and allow `None`;
- add `quality_estimate` only when ToFU has a separately calibrated estimator;
- never auto-accept solely from a provider-reported or heuristic score;
- use deterministic review rules: exact approved TM match, provider
  disagreement, glossary/placeholder violations, language mismatch, length/fit
  risk, and semantic-unit inconsistency;
- record the calibration/policy revision used by a decision.

Until a held-out bilingual evaluation exists, external MT should default to
`review_required`. An exact, approved TM match may follow a distinct policy.

### 2. Separate translation attempts from the accepted target

A single mutable `translation_provenance` object cannot faithfully represent
fallbacks, rejection, editing, retranslation, or CAT import. Use:

```text
translation_attempts[]  append-only provider/TM proposals
translation_decision    current selection and review state
target_text             current accepted/editable value
translation_history[]   state transitions with timestamp and actor/source
```

Required states should distinguish:

- `suggested`
- `review_required`
- `accepted_automatic`
- `accepted_human`
- `rejected`
- `edited_human`
- `superseded`
- `stale`

“Manually rejected and entered their own translation” is not one state. The
provider attempt is rejected; the new target is a human-authored decision.

### 3. Add stale-result and concurrency protection

Every proposal and apply request needs:

- `source_text_hash`;
- source/target language;
- manifest revision or ETag;
- glossary revision;
- semantic-context revision/hash;
- provider/model/prompt-template revision;
- stable attempt ID; and
- idempotency key for paid provider requests.

`POST /api/translate/apply` must reject a stale proposal with HTTP 409 if the
source text, target locale, DNT/excluded status, glossary, or manifest revision
changed after proposal generation. This prevents late network responses from
overwriting newer human work.

### 4. Cloud MT must require explicit consent

Do not infer consent from the presence of an API key and do not call cloud MT
automatically after detection. API keys can exist on a shared server.

Required policy:

- explicit project-level cloud-MT enablement;
- clear disclosure that source text and selected context leave the machine;
- per-provider allowlist;
- local/TM-only mode;
- request preview for LLM context;
- no source pixels, local paths, project names, or unnecessary adjacent text;
- configurable retention/redaction policy.

Post-detect should attach local TM suggestions. External MT should run
on-demand or through an explicitly enabled project automation.

### 5. Define provider reliability and cost controls

The registry requires:

- provider-specific language-code mapping and capability checks;
- maximum batch size and byte/character count;
- request timeout, retry policy, and retryable-status allowlist;
- exponential backoff with `Retry-After`;
- quota/cost ceiling per request and project;
- cancellation and progress reporting;
- circuit breaker after repeated provider failures;
- response schema validation and output count/ID reconciliation;
- secret redaction from exceptions and logs.

Fallback and comparison are different modes. Fallback stops when an eligible
proposal exists; comparison intentionally calls multiple paid providers. Make
that distinction explicit.

### 6. Protect translation structure

Before a result is eligible, validate:

- placeholders, numbers, URLs, product codes, and inline tags;
- required glossary terms and forbidden translations;
- leading/trailing whitespace and line-break contract;
- empty or source-identical output where inappropriate;
- target-language/script plausibility;
- expansion and layout-fit risk;
- DNT/excluded regions;
- per-region target-language overrides; and
- many-region Basil semantic units.

LLM calls should use a versioned structured-output schema. Treat source and
glossary text as untrusted prompt content and delimit it from instructions.

### 7. Do not expose `qa_score` as an ordinary editable field

`qa_score` is currently the gate that prevents poisoning visual TM. Allowing a
user to edit it directly defeats that contract. Store separately:

- immutable verifier score and verifier revision;
- human approval/rejection;
- optional human rating;
- edit provenance.

Changing source or target text must invalidate the old verifier score until the
record is reverified or explicitly human-approved under a documented policy.

### 8. TM deduplication key is too destructive

`(normalized_text, source_lang, target_lang)` can have multiple correct targets
because of domain, locale, gender, register, context, customer terminology, or
visual constraints. Do not delete all but the highest-QA row.

Use duplicate classes:

- exact duplicate: same normalized source, locale pair, target, context/domain,
  and relevant provenance;
- variant: same source key, different valid target;
- conflict: incompatible targets without enough context to resolve.

Exact duplicates may be consolidated transactionally. Variants must be
preserved and ranked. Conflicts require review. Add `domain`, locale (not just
base language), approval state, usage count, `last_used_at`, and optional
canonical/superseded relationships.

### 9. Cross-project TM search needs a sharing boundary

Before global search, define ownership/tenant and archive rules:

- private project by default;
- explicit shared/global TM scope;
- archived projects excluded unless requested;
- project and record authorization on every result;
- no thumbnail/path leakage across unauthorized projects.

Even in a single-user build, encode scope now so a later multi-user deployment
does not inherit an unsafe global index.

### 10. Snapshot IDs are not stable identities

Region IDs are reassigned after OCR pruning/re-detection. Comparing only by
region ID will report false add/remove changes.

Diff matching should use:

1. exact stable region ID when snapshot lineage proves continuity;
2. track/source identity when available;
3. one-to-one geometry assignment using polygon/bbox overlap;
4. normalized source-text similarity as supporting evidence; and
5. an ambiguity result rather than forced matching.

Diff output should be per-region with a field-change map, not one row per
field. Include manifest schema version and reject cross-asset comparisons by
default.

### 11. Video Phase 1 cannot safely promise end-to-end production output

The current acceptance criteria conflict:

- frames may be evenly sampled;
- reassembly uses original FPS;
- no interpolation/propagation exists;
- independent per-frame Cleanse is expected to flicker;
- audio is stripped.

Revise Phase 1 to one of:

1. analysis-only/keyframe proof of concept with frame manifests and preview
   frames; or
2. full-frame experimental render with original timestamps, audio remux, and a
   visible experimental-quality status.

Do not call a silent, flickering, timebase-altered file a completed localized
video.

## P1 — Required architecture amendments

### MT stage placement

- Keep local TM suggestion attachment after detection.
- Run semantic-context MT only after Basil semantic units and glossary state
  are finalized.
- Translate a semantic unit once, then assign target spans to immutable visual
  regions. Do not independently translate fragments that Basil later joins.
- Do not place MT inside the low-level static rendering pipeline by default;
  rendering must remain deterministic from an accepted manifest.

### Translation API contract

Prefer:

```text
POST /api/assets/{asset_id}/translation-runs
GET  /api/translation-runs/{run_id}
POST /api/assets/{asset_id}/translation-decisions
```

A run-oriented API supports batching, cancellation, cost reporting, partial
failure, retries, and idempotency better than a synchronous `/api/translate`.
The decision endpoint should accept attempt IDs, not client-supplied provider
claims.

### TM storage and audit

Add:

- `updated_at`, `deleted_at`, `approved_at`;
- origin (`render`, `tmx_import`, `human`, `mt`);
- immutable verifier evidence/revision;
- edit history or a `tm_record_events` table;
- usage counters;
- canonical/superseded relationships;
- normalized-source algorithm revision.

Bulk delete should be soft-delete by default and atomic. A purge operation
should be separate and deliberately destructive.

### Search implementation

SQLite indexes do not accelerate arbitrary `%LIKE%` searches. Use FTS5 when
available, with a documented fallback. Add cursor pagination, deterministic
sort order, query-length limits, and a maximum page size. Normalize query
language consistently with TM insertion.

### TMX handling

The existing manifest TMX parser is intentionally simple and loses TM-specific
metadata. The TM-record importer needs:

- namespace-aware TMX 1.4/1.4b parsing;
- source/target locale selection from `tuv`, not “last tuv is target”;
- segment text including nested inline content;
- stable record IDs/properties/notes where safe;
- size, unit-count, and nesting limits;
- malformed-unit isolation and per-unit errors;
- import preview and dry-run;
- duplicate/variant/conflict report;
- one database transaction per accepted import.

Do not reuse manifest-ID import semantics unchanged.

### Project archiving

Prefer `archived_at REAL NULL` over a boolean. Define effects on:

- default project listing;
- active asset/session;
- new writes and renders;
- TM lookup and global search;
- duplicate-upload detection;
- snapshots/events;
- unarchive;
- existing hard-delete endpoint.

Archiving must not silently change into deletion, and hard deletion should
require a separate confirmation path.

### Structured events

Current events are human-readable strings. Filtering will be more reliable with
structured event data:

- stable event kind enum;
- `payload_json`;
- actor/source;
- asset/record/snapshot IDs;
- request/correlation ID;
- timestamp.

Keep a rendered human-readable summary for the UI.

## Video project amendments

### Correct pipeline order

Running the complete static pipeline independently on every frame performs OCR,
translation, Cleanse, Scribe, Verify, and Memory before track identity exists.
That creates inconsistent translations and duplicate TM writes.

Use staged video processing:

1. probe/decode frames and timestamps;
2. detect/OCR selected frames;
3. associate observations into tracks;
4. vote/resolve source text per track;
5. translate once per track;
6. propagate approved text/style to frame observations;
7. Cleanse/Scribe frames;
8. perform temporal QA;
9. encode and remux audio/subtitle/data streams as policy permits;
10. write TM once per approved track, not once per frame.

Static layer functions may be reused, but `_process_static()` is too coarse as
the unit of reuse.

### Tracking

Greedy adjacent-frame IoU is not one-to-one and fails under camera motion,
occlusion, cuts, scale changes, and crossing regions. At minimum:

- enforce bipartite one-to-one assignment;
- combine IoU, text similarity, appearance, and scene-cut boundaries;
- permit missed-frame gaps;
- version tracking policy;
- record ambiguity and split/merge events;
- key mappings by `(frame_index, instance_id)`, since `r1` repeats per frame.

### Media correctness

Preserve:

- presentation timestamps and variable frame rate;
- rotation metadata;
- pixel/sample aspect ratio;
- color range, matrix, transfer, and primaries;
- resolution and alpha where applicable;
- audio and other streams;
- container/codec compatibility.

OpenCV `VideoWriter` is insufficient as the primary fidelity path. Treat
FFmpeg/ffprobe as the supported production backend and OpenCV as a constrained
fallback. Use argument arrays, fixed server configuration, timeouts, and
validated paths.

### Resource and recovery model

Add:

- streaming decode and bounded queues;
- disk-space estimates and quotas;
- temporary-work cleanup after success, failure, and cancellation;
- resumable per-stage checkpoints;
- per-frame/track error policy;
- cancellation;
- GPU/provider concurrency limits;
- deterministic cache keys including frame timestamp and model revisions;
- progress based on stages and decoded duration, not only frame count.

### Video manifest/storage

A flat list of all frame instances inside `TextManifest.instances` will become
large and expensive to snapshot. Define:

- track-level manifest;
- paged/chunked frame observations;
- schema version;
- baseline plus deltas;
- snapshot semantics;
- asset/content hash;
- timebase and timestamp fields.

Do not defer this storage decision until after full manifests are persisted.

### Temporal acceptance criteria

Even before temporal inpainting, measure:

- track precision/recall and ID switches;
- OCR consistency within a track;
- translation consistency within a track;
- bounding-box and baseline jitter;
- frame-to-frame fill difference outside expected motion;
- flicker/temporal perceptual metric;
- A/V duration and sync drift;
- dropped/duplicated frames;
- output timestamp preservation;
- cancellation/resume behavior;
- p50/p95 throughput and peak memory/disk use.

## Revised implementation order

### Phase 0 — Contracts and migrations

- translation attempt/decision state machine;
- manifest revision and stale-result rules;
- provider capability/security/cost policy;
- TM provenance and soft-delete schema;
- snapshot matching contract;
- video manifest/timebase design;
- migration and rollback tests.

### Phase A1 — Local TM and translation review foundation

- exact/fuzzy/visual TM proposals with distinct acceptance policy;
- append-only translation attempts;
- human accept/reject/edit workflow;
- stale-result protection;
- glossary/placeholder/language/fit validators.

### Phase A2 — One external MT provider

- explicit project consent;
- asynchronous translation runs;
- provider capability mapping, batching, quotas, retries, cancellation;
- default all external results to review-required;
- evaluation corpus and policy calibration.

Add a second provider only after the registry contract is proven. Add LLM
context after semantic-unit translation is stable.

### Phase B1 — TM correctness

- immutable verifier score plus human approval;
- edit history, soft deletion, variants/conflicts;
- FTS/pagination;
- safe TMX preview/import/export.

### Phase B2 — History and archive

- lineage-aware snapshot diff;
- structured event filtering;
- project archive semantics.

### Phase C0 — Video feasibility

- FFmpeg capability probe;
- media metadata/timebase round-trip;
- streaming decode/encode and audio remux;
- track-level schema and storage estimates.

### Phase C1 — Video analysis prototype

- detection observations;
- one-to-one tracking;
- track OCR voting;
- track-level translation;
- frame/timeline UI;
- no claim of production temporal inpainting.

### Phase C2 — Experimental full-frame localization

- full-frame Cleanse/Scribe;
- audio remux;
- temporal QA and explicit experimental status;
- bounded resume/cancel workflow.

Temporal inpainting remains a later phase.

## Additional acceptance gates

- Existing static pipeline remains behaviorally unchanged when MT/video are
  disabled.
- No network request occurs from detection alone without explicit project
  consent.
- No provider score is treated as calibrated confidence without an evaluation
  revision.
- A stale translation response cannot overwrite human edits.
- TM variants are preserved; only proven exact duplicates are consolidated.
- Editing a TM record cannot preserve an obsolete verifier score.
- Cross-project search respects explicit scope and archived-state policy.
- Snapshot diff handles re-detected/renumbered regions and reports ambiguity.
- TMX import is bounded, previewable, transactional, and per-unit auditable.
- Video output preserves timebase and audio or is explicitly labeled a preview.
- Sampled frames are never concatenated as if they represented every source
  frame.
- Track-level translation and TM storage occur once per track.
- Video failures are resumable and temporary artifacts are cleaned safely.

