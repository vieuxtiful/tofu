# ToFU API Reference

**Version:** 1.0.0

**Base URL:** `http://localhost:8000`

**Total endpoints:** 80

---

## Table of Contents

- [assets](#assets) (7 endpoints)
- [capabilities](#capabilities) (1 endpoints)
- [detection](#detection) (3 endpoints)
- [export](#export) (1 endpoints)
- [font-file](#font-file) (1 endpoints)
- [font-match](#font-match) (1 endpoints)
- [fonts](#fonts) (5 endpoints)
- [glossary](#glossary) (3 endpoints)
- [import](#import) (1 endpoints)
- [inpainting](#inpainting) (5 endpoints)
- [languages](#languages) (1 endpoints)
- [localized-baseline](#localized-baseline) (2 endpoints)
- [manifest](#manifest) (6 endpoints)
- [memory](#memory) (1 endpoints)
- [ocr-region](#ocr-region) (1 endpoints)
- [preview](#preview) (2 endpoints)
- [process](#process) (1 endpoints)
- [projects](#projects) (12 endpoints)
- [render](#render) (3 endpoints)
- [semantic](#semantic) (6 endpoints)
- [snapshots](#snapshots) (2 endpoints)
- [treatment](#treatment) (2 endpoints)
- [validation](#validation) (1 endpoints)
- [video](#video) (12 endpoints)

---

## assets

### POST `/api/assets`

**Upload Asset**

#### Parameters

| Name | In | Type | Required | Description |
|------|-----|------|----------|-------------|
| `project_id` | query | any | No |  |

#### Request Body

*No request body*

#### Responses

- **200**: Successful Response → *no schema*
- **422**: Validation Error → `HTTPValidationError`

---

### GET `/api/assets/check-duplicate`

**Check Duplicate Asset**

#### Parameters

| Name | In | Type | Required | Description |
|------|-----|------|----------|-------------|
| `hash` | query | string | Yes |  |

#### Responses

- **200**: Successful Response → *no schema*
- **422**: Validation Error → `HTTPValidationError`

---

### PATCH `/api/assets/{asset_id}/ground-truth`

**Update Asset Ground Truth**

#### Parameters

| Name | In | Type | Required | Description |
|------|-----|------|----------|-------------|
| `asset_id` | path | string | Yes |  |

#### Request Body

```json
`GroundTruthUpdate`
``` *(required)*

#### Responses

- **200**: Successful Response → *no schema*
- **422**: Validation Error → `HTTPValidationError`

---

### POST `/api/assets/{asset_id}/ground-truth/import`

**Import Asset Ground Truth**

#### Parameters

| Name | In | Type | Required | Description |
|------|-----|------|----------|-------------|
| `asset_id` | path | string | Yes |  |

#### Request Body

*No request body*

#### Responses

- **200**: Successful Response → *no schema*
- **422**: Validation Error → `HTTPValidationError`

---

### POST `/api/assets/{asset_id}/scan-language`

**Scan Language**

background language scan (lexiq-parity upload guard): detect the
asset's dominant language via cicerone (read-only — the manifest on
disk is untouched) and compare it against the project's source
language. when the project source is unset ('auto'), the first
successful scan locks it.

#### Parameters

| Name | In | Type | Required | Description |
|------|-----|------|----------|-------------|
| `asset_id` | path | string | Yes |  |

#### Responses

- **200**: Successful Response → *no schema*
- **422**: Validation Error → `HTTPValidationError`

---

### POST `/api/assets/{asset_id}/translation-decisions`

**Apply Translation Decisions**

Apply server-owned attempts with optimistic concurrency protection.

#### Parameters

| Name | In | Type | Required | Description |
|------|-----|------|----------|-------------|
| `asset_id` | path | string | Yes |  |

#### Request Body

```json
`TranslationDecisionRequest`
``` *(required)*

#### Responses

- **200**: Successful Response → *no schema*
- **422**: Validation Error → `HTTPValidationError`

---

### POST `/api/assets/{asset_id}/translation-runs`

**Create Translation Run**

Create local TM proposals. No network provider is invoked here.

#### Parameters

| Name | In | Type | Required | Description |
|------|-----|------|----------|-------------|
| `asset_id` | path | string | Yes |  |

#### Request Body

```json
`TranslationRunRequest`
``` *(required)*

#### Responses

- **200**: Successful Response → *no schema*
- **422**: Validation Error → `HTTPValidationError`

---


## capabilities

### GET `/api/capabilities`

**Capabilities**

One side-effect-free runtime contract for frontend feature gating.

#### Responses

- **200**: Successful Response → *no schema*

---


## detection

### POST `/api/detect`

**Detect**

#### Request Body

```json
`DetectRequest`
``` *(required)*

#### Responses

- **200**: Successful Response → *no schema*
- **422**: Validation Error → `HTTPValidationError`

---

### POST `/api/detect/refine`

**Refine Region**

detect and recognize text inside a user-drawn bbox; return the
detected sub-boxes in original-image coordinates so the UI can snap
a rough rectangle to precise text polygons.

#### Request Body

```json
`RefineRegionRequest`
``` *(required)*

#### Responses

- **200**: Successful Response → *no schema*
- **422**: Validation Error → `HTTPValidationError`

---

### GET `/api/detect/stream`

**Detect Stream**

SSE variant of /api/detect: emits progress events per stage/pass.

GET (not POST) so the browser's native EventSource can consume it;
languages is a comma-separated list of tofu codes. The engine
parameter accepts 'easyocr' (default) or 'paddleocr'. shares
merge_detections + build_manifest with /api/detect, so both produce
identical manifests from identical inputs.

#### Parameters

| Name | In | Type | Required | Description |
|------|-----|------|----------|-------------|
| `asset_id` | query | string | Yes |  |
| `languages` | query | any | No |  |
| `gpu` | query | boolean | No |  |
| `engine` | query | any | No |  |

#### Responses

- **200**: Successful Response → *no schema*
- **422**: Validation Error → `HTTPValidationError`

---


## export

### POST `/api/export`

**Export**

#### Request Body

```json
`ExportRequest`
``` *(required)*

#### Responses

- **200**: Successful Response → *no schema*
- **422**: Validation Error → `HTTPValidationError`

---


## font-file

### GET `/api/font-file`

**Serve Font File**

serve a font file for @font-face preview in the frontend.

FontRegistry keys the Nth face of a collection as "<file>#<n>"
(fonts.py's load_font), and those keys are exactly what reaches the
client on style_profile.font_family / resolved_font_family. Path(...)
on such a key is not a file, so every collection face used to 404 --
and CJK families on Windows are overwhelmingly .ttc, which is
precisely where the preview most needs the real face.

A browser @font-face cannot select a face INSIDE a collection, so
serving the whole .ttc would silently preview face 0: msgothic.ttc#1
(MS UI Gothic) drawn as face 0 (MS Gothic) is a different typeface at
different metrics, which desynchronizes both the fit measurement and
the paint from what scribe draws. Extract the requested face into a
standalone font instead. Cached in-process because the extraction is
pure CPU over an unchanging file and the panel preloads whole
families at once.

#### Parameters

| Name | In | Type | Required | Description |
|------|-----|------|----------|-------------|
| `path` | query | string | Yes |  |

#### Responses

- **200**: Successful Response → *no schema*
- **422**: Validation Error → `HTTPValidationError`

---


## font-match

### POST `/api/font-match/{asset_id}`

**Font Match**

Refresh local font evidence and, only with explicit consent, query a
configured commercial font catalog for unavailable/licensed candidates.

#### Parameters

| Name | In | Type | Required | Description |
|------|-----|------|----------|-------------|
| `asset_id` | path | string | Yes |  |

#### Request Body

```json
`FontMatchRequest`
``` *(required)*

#### Responses

- **200**: Successful Response → *no schema*
- **422**: Validation Error → `HTTPValidationError`

---


## fonts

### GET `/api/fonts`

**Fonts**

Ranked fonts for one target language.

Two modes, deliberately distinct.  The default is the dropdown's short
ranked list, capped at ``limit`` families.  ``full=true`` is the Font
Manager's complete catalog: every family, each tagged with a browsing
category, and no ``fonts`` face list (the manager picks per family, so
computing a flat ranked face list for it would be pure waste).

#### Parameters

| Name | In | Type | Required | Description |
|------|-----|------|----------|-------------|
| `lang` | query | string | Yes |  |
| `limit` | query | integer | No |  |
| `full` | query | boolean | No |  |

#### Responses

- **200**: Successful Response → *no schema*
- **422**: Validation Error → `HTTPValidationError`

---

### GET `/api/fonts/packs`

**List Font Packs**

#### Responses

- **200**: Successful Response → *no schema*

---

### POST `/api/fonts/packs/install`

**Install Font Pack**

Install a zip of fonts plus an optional pack.json.

#### Request Body

*No request body*

#### Responses

- **200**: Successful Response → *no schema*
- **422**: Validation Error → `HTTPValidationError`

---

### DELETE `/api/fonts/packs/{name}`

**Remove Font Pack**

#### Parameters

| Name | In | Type | Required | Description |
|------|-----|------|----------|-------------|
| `name` | path | string | Yes |  |

#### Responses

- **200**: Successful Response → *no schema*
- **422**: Validation Error → `HTTPValidationError`

---

### POST `/api/fonts/upload`

**Upload Font**

Install one font file for this server, available immediately.

No OS-level installation is involved, and none is needed: scribe
renders through ImageFont.truetype(path), the preview is served over
/api/font-file, and FontRegistry keys on absolute paths. The system
font directory was only ever a discovery convenience.

#### Request Body

*No request body*

#### Responses

- **200**: Successful Response → *no schema*
- **422**: Validation Error → `HTTPValidationError`

---


## glossary

### GET `/api/glossary/status`

**Glossary Status**

#### Parameters

| Name | In | Type | Required | Description |
|------|-----|------|----------|-------------|
| `project_id` | query | any | No |  |

#### Responses

- **200**: Successful Response → *no schema*
- **422**: Validation Error → `HTTPValidationError`

---

### POST `/api/glossary/upload`

**Upload Glossary**

#### Parameters

| Name | In | Type | Required | Description |
|------|-----|------|----------|-------------|
| `scope` | query | string | No |  |
| `project_id` | query | any | No |  |
| `mode` | query | string | No |  |
| `src_lang` | query | any | No |  |
| `targ_lang` | query | any | No |  |

#### Request Body

*No request body*

#### Responses

- **200**: Successful Response → *no schema*
- **422**: Validation Error → `HTTPValidationError`

---

### DELETE `/api/glossary/{scope}`

**Delete Glossary**

#### Parameters

| Name | In | Type | Required | Description |
|------|-----|------|----------|-------------|
| `scope` | path | string | Yes |  |
| `project_id` | query | any | No |  |

#### Responses

- **200**: Successful Response → *no schema*
- **422**: Validation Error → `HTTPValidationError`

---


## import

### POST `/api/import`

**Import File**

#### Parameters

| Name | In | Type | Required | Description |
|------|-----|------|----------|-------------|
| `asset_id` | query | string | No |  |

#### Request Body

*No request body*

#### Responses

- **200**: Successful Response → *no schema*
- **422**: Validation Error → `HTTPValidationError`

---


## inpainting

### POST `/api/inpaint`

**Inpaint**

#### Request Body

```json
`InpaintRequest`
``` *(required)*

#### Responses

- **200**: Successful Response → *no schema*
- **422**: Validation Error → `HTTPValidationError`

---

### POST `/api/inpaint/candidate/{asset_id}/{candidate_id}`

**Apply Repair Candidate**

Apply an editor-approved neural candidate as an undoable patch.

Approval is explicit and affects only the treatment layer.  It never
flips a provider promotion flag or rewrites the deterministic Cleanse
cache, so later benchmark evidence remains attributable to the model.

#### Parameters

| Name | In | Type | Required | Description |
|------|-----|------|----------|-------------|
| `asset_id` | path | string | Yes |  |
| `candidate_id` | path | string | Yes |  |

#### Request Body

```json
any
``` *(optional)*

#### Responses

- **200**: Successful Response → *no schema*
- **422**: Validation Error → `HTTPValidationError`

---

### DELETE `/api/inpaint/{asset_id}`

**Undo Inpaint**

#### Parameters

| Name | In | Type | Required | Description |
|------|-----|------|----------|-------------|
| `asset_id` | path | string | Yes |  |
| `patch_id` | query | any | No |  |

#### Responses

- **200**: Successful Response → *no schema*
- **422**: Validation Error → `HTTPValidationError`

---

### DELETE `/api/inpaint/{asset_id}/{patch_id}`

**Undo Inpaint**

#### Parameters

| Name | In | Type | Required | Description |
|------|-----|------|----------|-------------|
| `asset_id` | path | string | Yes |  |
| `patch_id` | path | any | Yes |  |

#### Responses

- **200**: Successful Response → *no schema*
- **422**: Validation Error → `HTTPValidationError`

---

### GET `/api/inpainting/providers`

**Inpainting Providers**

Local provider capabilities; never probes or downloads model weights.

#### Responses

- **200**: Successful Response → *no schema*

---


## languages

### GET `/api/languages`

**Languages**

#### Responses

- **200**: Successful Response → *no schema*

---


## localized-baseline

### GET `/api/localized-baseline/{asset_id}`

**Get Localized Baseline**

#### Parameters

| Name | In | Type | Required | Description |
|------|-----|------|----------|-------------|
| `asset_id` | path | string | Yes |  |

#### Responses

- **200**: Successful Response → *no schema*
- **422**: Validation Error → `HTTPValidationError`

---

### PUT `/api/localized-baseline/{asset_id}`

**Capture Localized Baseline**

Persist the first Render-entry state; manual edits never overwrite it.

#### Parameters

| Name | In | Type | Required | Description |
|------|-----|------|----------|-------------|
| `asset_id` | path | string | Yes |  |

#### Request Body

```json
`LocalizedBaselineRequest`
``` *(required)*

#### Responses

- **200**: Successful Response → *no schema*
- **422**: Validation Error → `HTTPValidationError`

---


## manifest

### GET `/api/manifest/{asset_id}`

**Get Manifest**

#### Parameters

| Name | In | Type | Required | Description |
|------|-----|------|----------|-------------|
| `asset_id` | path | string | Yes |  |

#### Responses

- **200**: Successful Response → *no schema*
- **422**: Validation Error → `HTTPValidationError`

---

### PUT `/api/manifest/{asset_id}`

**Put Manifest**

#### Parameters

| Name | In | Type | Required | Description |
|------|-----|------|----------|-------------|
| `asset_id` | path | string | Yes |  |

#### Request Body

```json
object
``` *(required)*

#### Responses

- **200**: Successful Response → *no schema*
- **422**: Validation Error → `HTTPValidationError`

---

### POST `/api/manifest/{asset_id}/regions`

**Add Region**

#### Parameters

| Name | In | Type | Required | Description |
|------|-----|------|----------|-------------|
| `asset_id` | path | string | Yes |  |

#### Request Body

```json
`RegionCreate`
``` *(required)*

#### Responses

- **200**: Successful Response → *no schema*
- **422**: Validation Error → `HTTPValidationError`

---

### POST `/api/manifest/{asset_id}/regions/merge`

**Merge Regions**

Fold several regions into one.

merge_baseline_runs already joins the words of a line automatically,
and declines wherever the geometry is ambiguous -- across a column
gutter, over a gap wider than a word space. This is the manual door
for those, and for any grouping only a person knows is one unit.

The survivor is the FIRST member in reading order, which keeps its id
and its correction/provenance history; the rest are marked excluded
exactly as delete_region marks them, so cleanse() still erases their
pixels even though the merged box already covers them.

#### Parameters

| Name | In | Type | Required | Description |
|------|-----|------|----------|-------------|
| `asset_id` | path | string | Yes |  |

#### Request Body

```json
`RegionMerge`
``` *(required)*

#### Responses

- **200**: Successful Response → *no schema*
- **422**: Validation Error → `HTTPValidationError`

---

### DELETE `/api/manifest/{asset_id}/regions/{rid}`

**Delete Region**

"remove" a region from the workspace. the instance is marked
excluded rather than actually dropped from the manifest: cleanse()
still erases its source pixels (nothing is left half-translated on
the canvas), but scribe() never renders it and the UI never lists
it -- matches every other "delete" in this app while still letting
a user keep a region out of the export without leaving its source
text sitting untouched in the final image.

#### Parameters

| Name | In | Type | Required | Description |
|------|-----|------|----------|-------------|
| `asset_id` | path | string | Yes |  |
| `rid` | path | string | Yes |  |

#### Responses

- **200**: Successful Response → *no schema*
- **422**: Validation Error → `HTTPValidationError`

---

### PATCH `/api/manifest/{asset_id}/regions/{rid}`

**Update Region**

#### Parameters

| Name | In | Type | Required | Description |
|------|-----|------|----------|-------------|
| `asset_id` | path | string | Yes |  |
| `rid` | path | string | Yes |  |

#### Request Body

```json
`RegionUpdate`
``` *(required)*

#### Responses

- **200**: Successful Response → *no schema*
- **422**: Validation Error → `HTTPValidationError`

---


## memory

### DELETE `/api/memory/{record_id}`

**Delete Memory Record**

#### Parameters

| Name | In | Type | Required | Description |
|------|-----|------|----------|-------------|
| `record_id` | path | integer | Yes |  |

#### Responses

- **200**: Successful Response → *no schema*
- **422**: Validation Error → `HTTPValidationError`

---


## ocr-region

### POST `/api/ocr-region`

**Ocr Region**

#### Request Body

```json
`OcrRegionRequest`
``` *(required)*

#### Responses

- **200**: Successful Response → *no schema*
- **422**: Validation Error → `HTTPValidationError`

---


## preview

### POST `/api/preview/candidate/{asset_id}/{candidate_id}`

**Preview Candidate Localized**

Preview a background-only repair with current translated text layered on top.

#### Parameters

| Name | In | Type | Required | Description |
|------|-----|------|----------|-------------|
| `asset_id` | path | string | Yes |  |
| `candidate_id` | path | string | Yes |  |

#### Request Body

```json
`CandidatePreviewRequest`
``` *(required)*

#### Responses

- **200**: Successful Response → *no schema*
- **422**: Validation Error → `HTTPValidationError`

---

### POST `/api/preview/render`

**Preview Render**

Build an isolated preview from an optional unsaved manifest.

#### Request Body

```json
`PreviewRenderRequest`
``` *(required)*

#### Responses

- **200**: Successful Response → *no schema*
- **422**: Validation Error → `HTTPValidationError`

---


## process

### POST `/api/process`

**Process**

#### Request Body

```json
`ProcessRequest`
``` *(required)*

#### Responses

- **200**: Successful Response → *no schema*
- **422**: Validation Error → `HTTPValidationError`

---


## projects

### GET `/api/projects`

**List Projects**

#### Parameters

| Name | In | Type | Required | Description |
|------|-----|------|----------|-------------|
| `archived` | query | any | No |  |
| `query` | query | any | No |  |
| `sort` | query | string | No |  |

#### Responses

- **200**: Successful Response → *no schema*
- **422**: Validation Error → `HTTPValidationError`

---

### POST `/api/projects`

**Create Project**

#### Request Body

```json
`ProjectCreate`
``` *(required)*

#### Responses

- **200**: Successful Response → *no schema*
- **422**: Validation Error → `HTTPValidationError`

---

### DELETE `/api/projects/{pid}`

**Delete Project**

#### Parameters

| Name | In | Type | Required | Description |
|------|-----|------|----------|-------------|
| `pid` | path | string | Yes |  |

#### Responses

- **200**: Successful Response → *no schema*
- **422**: Validation Error → `HTTPValidationError`

---

### GET `/api/projects/{pid}`

**Get Project**

#### Parameters

| Name | In | Type | Required | Description |
|------|-----|------|----------|-------------|
| `pid` | path | string | Yes |  |

#### Responses

- **200**: Successful Response → *no schema*
- **422**: Validation Error → `HTTPValidationError`

---

### PATCH `/api/projects/{pid}`

**Update Project**

#### Parameters

| Name | In | Type | Required | Description |
|------|-----|------|----------|-------------|
| `pid` | path | string | Yes |  |

#### Request Body

```json
`ProjectUpdate`
``` *(required)*

#### Responses

- **200**: Successful Response → *no schema*
- **422**: Validation Error → `HTTPValidationError`

---

### POST `/api/projects/{pid}/archive`

**Archive Project**

#### Parameters

| Name | In | Type | Required | Description |
|------|-----|------|----------|-------------|
| `pid` | path | string | Yes |  |

#### Responses

- **200**: Successful Response → *no schema*
- **422**: Validation Error → `HTTPValidationError`

---

### DELETE `/api/projects/{pid}/assets/{asset_id}`

**Delete Project Asset**

unlink an asset from its project. the snapshot ledger and the
uploaded file are retained — deletion never destroys session data.

#### Parameters

| Name | In | Type | Required | Description |
|------|-----|------|----------|-------------|
| `pid` | path | string | Yes |  |
| `asset_id` | path | string | Yes |  |

#### Responses

- **200**: Successful Response → *no schema*
- **422**: Validation Error → `HTTPValidationError`

---

### POST `/api/projects/{pid}/assets/{asset_id}/activate`

**Activate Asset**

switch the project's active session to a previously uploaded asset.

#### Parameters

| Name | In | Type | Required | Description |
|------|-----|------|----------|-------------|
| `pid` | path | string | Yes |  |
| `asset_id` | path | string | Yes |  |

#### Responses

- **200**: Successful Response → *no schema*
- **422**: Validation Error → `HTTPValidationError`

---

### GET `/api/projects/{pid}/history`

**Project History**

#### Parameters

| Name | In | Type | Required | Description |
|------|-----|------|----------|-------------|
| `pid` | path | string | Yes |  |
| `asset_id` | query | any | No |  |

#### Responses

- **200**: Successful Response → *no schema*
- **422**: Validation Error → `HTTPValidationError`

---

### GET `/api/projects/{pid}/memory`

**Project Memory**

translation-memory browser panel: every stored record for this
project, newest first, with a thumb_url for the crop preview.

#### Parameters

| Name | In | Type | Required | Description |
|------|-----|------|----------|-------------|
| `pid` | path | string | Yes |  |

#### Responses

- **200**: Successful Response → *no schema*
- **422**: Validation Error → `HTTPValidationError`

---

### POST `/api/projects/{pid}/restore`

**Restore Project**

#### Parameters

| Name | In | Type | Required | Description |
|------|-----|------|----------|-------------|
| `pid` | path | string | Yes |  |

#### Responses

- **200**: Successful Response → *no schema*
- **422**: Validation Error → `HTTPValidationError`

---

### POST `/api/projects/{pid}/snapshots`

**Create Snapshot**

snapshot the asset's current manifest on demand — called before any
destructive step (session replacement, restore) so data is never lost.

#### Parameters

| Name | In | Type | Required | Description |
|------|-----|------|----------|-------------|
| `pid` | path | string | Yes |  |

#### Request Body

```json
`SnapshotCreate`
``` *(required)*

#### Responses

- **200**: Successful Response → *no schema*
- **422**: Validation Error → `HTTPValidationError`

---


## render

### POST `/api/render`

**Render**

render via TofuPipeline (MANUAL cicerone seeded with the stored
manifest): tofu re-validation, scene enrichment, cleanse, scribe,
verify, and memory all run through the one orchestrator, which also
produces the structured logs/errors this endpoint returns.

#### Request Body

```json
`RenderRequest`
``` *(required)*

#### Responses

- **200**: Successful Response → *no schema*
- **422**: Validation Error → `HTTPValidationError`

---

### POST `/api/render/approve`

**Approve Render**

QA Inspector sign-off: records the coverage/score snapshot the user
approved into project history. A human decision, not a derived fact —
logged as its own event kind so it's distinguishable from an ordinary
render in History.

#### Request Body

```json
`ApproveRequest`
``` *(required)*

#### Responses

- **200**: Successful Response → *no schema*
- **422**: Validation Error → `HTTPValidationError`

---

### GET `/api/render/stream`

**Render Stream**

SSE variant of /api/render: emits progress per layer (tofu, scene,
tofu_regions, cleanse, scribe, verify) + a final payload shaped like
/api/render's response. GET (not POST) so the browser's native
EventSource can consume it, mirroring /api/detect/stream. Full renders
consume the canonical TofuPipeline event observer; partial region renders
retain their specialized prior-output composition path below.

also closes the Phase-4-deferred item: after scene enrichment, every
DISTINCT effective target language across regions (target_language
overrides included) gets its own ToFU.validate() pass with the
region's real font_px (Phase 1 typography) and effects context —
the original pre-flight only ever validated the single request-level
target language, never a region override, and never with the size/
effects context that makes the render-quality prediction meaningful.

region_ids (comma-separated, optional): the QA Inspector's per-region
re-render action. When given, cleanse+scribe run ONLY on that instance
subset, and the starting image is the asset's existing localized output
(if one is on disk) rather than the raw source — so untouched regions
keep their prior render instead of reverting to source text. Falls back
to the raw source when no prior output exists yet (nothing to composite
onto).

#### Parameters

| Name | In | Type | Required | Description |
|------|-----|------|----------|-------------|
| `asset_id` | query | string | Yes |  |
| `targ_lang` | query | string | Yes |  |
| `font` | query | any | No |  |
| `qa_threshold` | query | any | No |  |
| `region_ids` | query | any | No |  |

#### Responses

- **200**: Successful Response → *no schema*
- **422**: Validation Error → `HTTPValidationError`

---


## semantic

### POST `/api/semantic-units/{asset_id}/create`

**Semantic Create Unit**

Create a new, user-authored semantic unit (a 'plate').

The unit starts empty or with the given region IDs.  Region boxes,
ids and OCR text are untouched.

#### Parameters

| Name | In | Type | Required | Description |
|------|-----|------|----------|-------------|
| `asset_id` | path | string | Yes |  |

#### Request Body

```json
`SemanticCreateUnitRequest`
``` *(required)*

#### Responses

- **200**: Successful Response → *no schema*
- **422**: Validation Error → `HTTPValidationError`

---

### DELETE `/api/semantic-units/{asset_id}/{unit_id}`

**Semantic Delete Unit**

Permanently remove a semantic unit from the manifest.

Region boxes, ids and OCR text are untouched; only the unit entry is
dropped, so its former members can be re-grouped by a later re-detection
or re-assigned by the user.

#### Parameters

| Name | In | Type | Required | Description |
|------|-----|------|----------|-------------|
| `asset_id` | path | string | Yes |  |
| `unit_id` | path | string | Yes |  |

#### Responses

- **200**: Successful Response → *no schema*
- **422**: Validation Error → `HTTPValidationError`

---

### POST `/api/semantic-units/{asset_id}/{unit_id}/members`

**Semantic Modify Members**

Add or remove a single region from a semantic unit's membership.

Only the unit's own ``region_ids``, ``source_text``, ``bbox`` and
``suggestion`` change.  Region boxes, ids and OCR text are untouched,
so a later re-detection can still re-group them.

#### Parameters

| Name | In | Type | Required | Description |
|------|-----|------|----------|-------------|
| `asset_id` | path | string | Yes |  |
| `unit_id` | path | string | Yes |  |

#### Request Body

```json
`SemanticModifyMembersRequest`
``` *(required)*

#### Responses

- **200**: Successful Response → *no schema*
- **422**: Validation Error → `HTTPValidationError`

---

### POST `/api/semantic-units/{asset_id}/{unit_id}/repair`

**Semantic Repair**

Accept or reject Basil's proposed cross-region source correction.

The decision changes only the semantic unit's own source text. Region
ids, boxes and OCR text stay exactly as Cicerone left them, so a
rejected proposal costs nothing and a later re-detection loses nothing.

#### Parameters

| Name | In | Type | Required | Description |
|------|-----|------|----------|-------------|
| `asset_id` | path | string | Yes |  |
| `unit_id` | path | string | Yes |  |

#### Request Body

```json
`SemanticRepairRequest`
``` *(required)*

#### Responses

- **200**: Successful Response → *no schema*
- **422**: Validation Error → `HTTPValidationError`

---

### POST `/api/semantic-units/{asset_id}/{unit_id}/substitution`

**Semantic Substitution**

Plan or explicitly apply target spans to immutable source anchors.

#### Parameters

| Name | In | Type | Required | Description |
|------|-----|------|----------|-------------|
| `asset_id` | path | string | Yes |  |
| `unit_id` | path | string | Yes |  |

#### Request Body

```json
`SemanticSubstitutionRequest`
``` *(required)*

#### Responses

- **200**: Successful Response → *no schema*
- **422**: Validation Error → `HTTPValidationError`

---

### GET `/api/semantic/providers`

**Semantic Providers**

Report only locally provisioned semantic/translation seams.

It is intentionally not an installer: language models are large and must
pass ToFU's benchmark gate before a deployment enables them.

#### Parameters

| Name | In | Type | Required | Description |
|------|-----|------|----------|-------------|
| `project_id` | query | any | No |  |

#### Responses

- **200**: Successful Response → *no schema*
- **422**: Validation Error → `HTTPValidationError`

---


## snapshots

### DELETE `/api/snapshots/{sid}`

**Delete Snapshot**

#### Parameters

| Name | In | Type | Required | Description |
|------|-----|------|----------|-------------|
| `sid` | path | integer | Yes |  |

#### Responses

- **200**: Successful Response → *no schema*
- **422**: Validation Error → `HTTPValidationError`

---

### POST `/api/snapshots/{sid}/restore`

**Restore Snapshot**

write a snapshot's manifest back as the asset's live manifest.
The pre-restore state is itself snapshotted first (restore-backup).

#### Parameters

| Name | In | Type | Required | Description |
|------|-----|------|----------|-------------|
| `sid` | path | integer | Yes |  |

#### Responses

- **200**: Successful Response → *no schema*
- **422**: Validation Error → `HTTPValidationError`

---


## treatment

### GET `/api/treatment/{asset_id}`

**Treatment State**

#### Parameters

| Name | In | Type | Required | Description |
|------|-----|------|----------|-------------|
| `asset_id` | path | string | Yes |  |

#### Responses

- **200**: Successful Response → *no schema*
- **422**: Validation Error → `HTTPValidationError`

---

### PUT `/api/treatment/{asset_id}`

**Restore Treatment**

Restore an ordered treatment layer for unified localized undo/redo.

#### Parameters

| Name | In | Type | Required | Description |
|------|-----|------|----------|-------------|
| `asset_id` | path | string | Yes |  |

#### Request Body

```json
`TreatmentRestoreRequest`
``` *(required)*

#### Responses

- **200**: Successful Response → *no schema*
- **422**: Validation Error → `HTTPValidationError`

---


## validation

### POST `/api/validate`

**Validate**

#### Request Body

```json
`ValidateRequest`
``` *(required)*

#### Responses

- **200**: Successful Response → *no schema*
- **422**: Validation Error → `HTTPValidationError`

---


## video

### GET `/api/video/assets/{asset_id}/latest-job`

**Get Latest Video Job**

#### Parameters

| Name | In | Type | Required | Description |
|------|-----|------|----------|-------------|
| `asset_id` | path | string | Yes |  |

#### Responses

- **200**: Successful Response → *no schema*
- **422**: Validation Error → `HTTPValidationError`

---

### POST `/api/video/jobs`

**Create Video Job**

#### Request Body

```json
`VideoJobCreate`
``` *(required)*

#### Responses

- **200**: Successful Response → *no schema*
- **422**: Validation Error → `HTTPValidationError`

---

### GET `/api/video/jobs/{job_id}`

**Get Video Job**

#### Parameters

| Name | In | Type | Required | Description |
|------|-----|------|----------|-------------|
| `job_id` | path | string | Yes |  |

#### Responses

- **200**: Successful Response → *no schema*
- **422**: Validation Error → `HTTPValidationError`

---

### POST `/api/video/jobs/{job_id}/cancel`

**Cancel Video Job**

#### Parameters

| Name | In | Type | Required | Description |
|------|-----|------|----------|-------------|
| `job_id` | path | string | Yes |  |

#### Responses

- **200**: Successful Response → *no schema*
- **422**: Validation Error → `HTTPValidationError`

---

### GET `/api/video/jobs/{job_id}/events`

**Stream Video Job**

#### Parameters

| Name | In | Type | Required | Description |
|------|-----|------|----------|-------------|
| `job_id` | path | string | Yes |  |

#### Responses

- **200**: Successful Response → *no schema*
- **422**: Validation Error → `HTTPValidationError`

---

### POST `/api/video/jobs/{job_id}/export`

**Export Video**

#### Parameters

| Name | In | Type | Required | Description |
|------|-----|------|----------|-------------|
| `job_id` | path | string | Yes |  |

#### Responses

- **200**: Successful Response → *no schema*
- **422**: Validation Error → `HTTPValidationError`

---

### PUT `/api/video/jobs/{job_id}/keyframes`

**Put Video Keyframe**

#### Parameters

| Name | In | Type | Required | Description |
|------|-----|------|----------|-------------|
| `job_id` | path | string | Yes |  |

#### Request Body

```json
`VideoKeyframeRequest`
``` *(required)*

#### Responses

- **200**: Successful Response → *no schema*
- **422**: Validation Error → `HTTPValidationError`

---

### POST `/api/video/jobs/{job_id}/preview`

**Render Video Preview**

#### Parameters

| Name | In | Type | Required | Description |
|------|-----|------|----------|-------------|
| `job_id` | path | string | Yes |  |

#### Request Body

```json
`VideoRenderRequest`
``` *(required)*

#### Responses

- **200**: Successful Response → *no schema*
- **422**: Validation Error → `HTTPValidationError`

---

### POST `/api/video/jobs/{job_id}/resume`

**Resume Video Job**

#### Parameters

| Name | In | Type | Required | Description |
|------|-----|------|----------|-------------|
| `job_id` | path | string | Yes |  |

#### Responses

- **200**: Successful Response → *no schema*
- **422**: Validation Error → `HTTPValidationError`

---

### GET `/api/video/jobs/{job_id}/timeline`

**Get Video Timeline**

#### Parameters

| Name | In | Type | Required | Description |
|------|-----|------|----------|-------------|
| `job_id` | path | string | Yes |  |
| `start` | query | integer | No |  |
| `end` | query | integer | No |  |
| `limit` | query | integer | No |  |
| `cursor` | query | integer | No |  |

#### Responses

- **200**: Successful Response → *no schema*
- **422**: Validation Error → `HTTPValidationError`

---

### PATCH `/api/video/jobs/{job_id}/tracks/{track_id}`

**Patch Video Track**

#### Parameters

| Name | In | Type | Required | Description |
|------|-----|------|----------|-------------|
| `job_id` | path | string | Yes |  |
| `track_id` | path | string | Yes |  |

#### Request Body

```json
`VideoTrackUpdate`
``` *(required)*

#### Responses

- **200**: Successful Response → *no schema*
- **422**: Validation Error → `HTTPValidationError`

---

### POST `/api/video/jobs/{job_id}/upgrade`

**Upgrade Video Job**

#### Parameters

| Name | In | Type | Required | Description |
|------|-----|------|----------|-------------|
| `job_id` | path | string | Yes |  |

#### Responses

- **200**: Successful Response → *no schema*
- **422**: Validation Error → `HTTPValidationError`

---


## Schema Reference

### `ApproveRequest`

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `asset_id` | string | Yes |  |
| `targ_lang` | string | Yes |  |
| `overall_score` | any | No |  |
| `regions_total` | any | No |  |
| `rendered` | any | No |  |
| `dnt` | any | No |  |

### `Body_import_asset_ground_truth_api_assets__asset_id__ground_truth_import_post`

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `file` | string | Yes |  |

### `Body_import_file_api_import_post`

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `file` | string | Yes |  |

### `Body_install_font_pack_api_fonts_packs_install_post`

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `file` | string | Yes |  |

### `Body_upload_asset_api_assets_post`

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `file` | string | Yes |  |

### `Body_upload_font_api_fonts_upload_post`

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `file` | string | Yes |  |

### `Body_upload_glossary_api_glossary_upload_post`

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `file` | string | Yes |  |

### `CandidateApplyRequest`

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `cache_key` | any | No |  |

### `CandidatePreviewRequest`

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `manifest` | any | No |  |
| `targ_lang` | any | No |  |

### `DetectRequest`

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `asset_id` | string | Yes |  |
| `gpu` | any | No |  |
| `languages` | any | No |  |

### `ExportRequest`

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `asset_id` | string | Yes |  |
| `format` | string | Yes |  |
| `variant` | any | No |  |
| `targ_lang` | any | No |  |

### `FontMatchRequest`

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `allow_external` | boolean | No |  |

### `GroundTruthUpdate`

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `ground_truth` | array | No |  |

### `HTTPValidationError`

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `detail` | array | No |  |

### `InpaintRequest`

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `asset_id` | string | Yes |  |
| `polygon` | array | No |  |
| `points` | array | No |  |
| `mode` | string | No |  |
| `radius` | integer | No |  |
| `hardness` | number | No |  |
| `blur_strength` | number | No |  |
| `opacity` | number | No |  |
| `clone_source` | any | No |  |
| `manifest` | any | No |  |

### `LocalizedBaselineRequest`

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `manifest` | object | Yes |  |
| `patch_ids` | array | No |  |

### `OcrRegionRequest`

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `asset_id` | string | Yes |  |
| `bbox` | object | Yes |  |

### `PreviewRenderRequest`

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `asset_id` | string | Yes |  |
| `targ_lang` | string | Yes |  |
| `manifest` | any | No |  |
| `fast_path` | boolean | No |  |
| `show_localized_text` | boolean | No |  |

### `ProcessRequest`

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `asset_id` | string | Yes |  |
| `targ_lang` | string | Yes |  |
| `font` | any | No |  |
| `qa_threshold` | any | No |  |
| `translations` | any | No |  |

### `ProjectCreate`

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `name` | string | Yes |  |
| `target_lang` | string | Yes |  |
| `asset_kind` | string | No |  |

### `ProjectUpdate`

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `name` | any | No |  |
| `target_lang` | any | No |  |
| `source_lang` | any | No |  |
| `ground_truth` | any | No |  |
| `archived` | any | No |  |

### `RefineRegionRequest`

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `asset_id` | string | Yes |  |
| `bbox` | object | Yes |  |
| `engine` | any | No |  |
| `scale` | any | No |  |

### `RegionCreate`

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `x` | integer | Yes |  |
| `y` | integer | Yes |  |
| `width` | integer | Yes |  |
| `height` | integer | Yes |  |
| `text` | any | No |  |
| `target_text` | any | No |  |

### `RegionMerge`

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `region_ids` | array | Yes |  |
| `texts` | any | No |  |
| `reread` | boolean | No |  |

### `RegionUpdate`

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `x` | any | No |  |
| `y` | any | No |  |
| `width` | any | No |  |
| `height` | any | No |  |
| `text` | any | No |  |
| `target_text` | any | No |  |
| `dnt` | any | No |  |
| `target_language` | any | No |  |
| `language` | any | No |  |
| `font` | any | No |  |
| `excluded` | any | No |  |
| `target_orientation` | any | No |  |
| `word_order` | any | No |  |
| `segmentation_mask` | any | No |  |

### `RenderRequest`

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `asset_id` | string | Yes |  |
| `targ_lang` | string | Yes |  |
| `font` | any | No |  |
| `qa_threshold` | any | No |  |

### `SemanticCreateUnitRequest`

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `region_ids` | any | No |  |

### `SemanticModifyMembersRequest`

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `add_region_id` | any | No |  |
| `remove_region_id` | any | No |  |

### `SemanticRepairRequest`

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `accepted` | boolean | Yes |  |

### `SemanticSubstitutionRequest`

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `target_text` | string | Yes |  |
| `targ_lang` | any | No |  |
| `apply` | boolean | No |  |

### `SnapshotCreate`

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `asset_id` | string | Yes |  |
| `reason` | string | No |  |

### `TranslationDecisionItem`

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `region_id` | string | Yes |  |
| `action` | string | Yes |  |
| `attempt_id` | any | No |  |
| `text` | any | No |  |

### `TranslationDecisionRequest`

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `manifest_revision` | string | Yes |  |
| `decisions` | array | Yes |  |

### `TranslationRunRequest`

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `region_ids` | any | No |  |
| `target_lang` | any | No |  |

### `TreatmentRestoreRequest`

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `patch_ids` | array | No |  |

### `ValidateRequest`

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `asset_id` | string | Yes |  |
| `targ_lang` | string | Yes |  |
| `font` | any | No |  |

### `ValidationError`

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `loc` | array | Yes |  |
| `msg` | string | Yes |  |
| `type` | string | Yes |  |
| `input` | any | No |  |
| `ctx` | object | No |  |

### `VideoJobCreate`

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `asset_id` | string | Yes |  |
| `project_id` | any | No |  |
| `chunk_size` | integer | No |  |

### `VideoKeyframeRequest`

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `id` | any | No |  |
| `track_id` | string | Yes |  |
| `frame_index` | integer | Yes |  |
| `scope` | string | No |  |
| `end_frame` | any | No |  |
| `bbox` | any | No |  |
| `quad` | any | No |  |
| `opacity` | any | No |  |
| `style` | any | No |  |
| `effects` | any | No |  |
| `mask_path` | any | No |  |
| `expected_job_revision` | any | No |  |

### `VideoRenderRequest`

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `start_frame` | any | No |  |
| `end_frame` | any | No |  |

### `VideoTrackUpdate`

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `source_text` | any | No |  |
| `target_text` | any | No |  |
| `target_language` | any | No |  |
| `status` | any | No |  |
| `style` | any | No |  |
| `expected_revision` | any | No |  |
