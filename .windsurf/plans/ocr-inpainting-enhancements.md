# OCR & Inpainting Enhancements — Megaplan

## Three Enhancement Areas

1. **Real MT Backend** — Multi-provider pluggable machine translation with provenance and review states
2. **Historical Project Browsing & Memory Operations** — Full TM suite, snapshot diffing, cross-project search, TMX export/import, merge/dedup, project archiving
3. **Video as a Separate Project** — Phased: frame extraction + static reuse now, temporal features later

This plan extends the existing `multi-candidate-ocr-inpainting-plan.md` (OCR assessment, residual glyph detection, multi-candidate inpainting). Those items are assumed complete or in progress; this plan covers the three new enhancement areas only.

---

## Part 1: Multi-Provider Pluggable MT Backend

### 1.1 Current State

- `src/tofu/utils/translate.py` defines `TranslatorBackend` ABC with `translate(segments, src_lang, targ_lang) -> Dict[str, str]` and `NullTranslator` (returns `{}`).
- No real provider is wired. Translation is entirely TMS round-trip (export XLIFF/TMX → translate externally → re-import).
- `InstText` has `target_text`, `target_language`, `tm_suggestion` fields.
- `server/main.py` has `_attach_tm_suggestions()` for post-detect TM lookup, and import/export endpoints for XLIFF/TMX.
- `PipelineCfg` has no MT-related configuration.
- No provenance or review state for machine translations exists.

### 1.2 Design

#### 1.2.1 Provider Architecture

Build a multi-provider system behind the existing `TranslatorBackend` ABC, mirroring the OCR backend pattern in `cicerone.py`.

**New providers** (each in `src/tofu/utils/translate.py` or a new `src/tofu/utils/mt_providers/` package):

| Provider | Class | Config | Notes |
|----------|-------|--------|-------|
| DeepL | `DeepLTranslator` | `TOFU_DEEPL_API_KEY` env var | Best quality for European + CJK; REST API |
| Google | `GoogleTranslator` | `TOFU_GOOGLE_API_KEY` env var | Broad language coverage |
| LLM | `LLMTranslator` | `TOFU_LLM_PROVIDER`, `TOFU_LLM_API_KEY`, `TOFU_LLM_MODEL` | Context-aware; supports system prompts with scene/glossary context |
| TM Lookup | `TMLookupTranslator` | Uses existing `memory.lookup()` | Already partially exists via `_attach_tm_suggestions`; formalize as a backend |

**Provider registry & routing:**

```python
class TranslatorRegistry:
    """Manages ordered providers with fallback."""
    providers: List[TranslatorBackend]  # ordered by priority
    
    def translate(self, segments, src_lang, targ_lang, context=None) -> Dict[str, TranslationResult]:
        # Try providers in order; collect results
        # Untranslated segments fall through to next provider
        # TM Lookup always runs first (free, local)
```

**New return type** — extend `translate()` return to carry provenance:

```python
@dataclass
class TranslationResult:
    text: str
    provider: str           # "deepl", "google", "llm", "tm_lookup"
    confidence: float       # provider-reported or heuristic
    review_state: str       # "auto_accepted" | "review_required" | "rejected"
    provenance: Dict[str, Any]  # {provider, model, timestamp, tm_record_id?, context_hash}
    alternatives: List[Dict[str, Any]]  # other providers' outputs for comparison
```

#### 1.2.2 Provenance & Review States

**Translation provenance** on `InstText` — new field `translation_provenance`:

```python
translation_provenance: Optional[Dict[str, Any]] = None
# {
#   "provider": "deepl",
#   "model": "deepl-pro",
#   "timestamp": 1234567890,
#   "src_lang": "en", "targ_lang": "es",
#   "confidence": 0.95,
#   "review_state": "auto_accepted" | "review_required" | "manually_approved" | "manually_rejected",
#   "alternatives": [{"provider": "google", "text": "...", "confidence": 0.91}],
#   "tm_record_id": 42,           # if matched from TM
#   "context_hash": "abc123",     # hash of context sent to LLM (if applicable)
#   "edited": False,              # user modified the MT output
# }
```

**Review states:**
- `auto_accepted`: MT confidence ≥ threshold, no TM conflict
- `review_required`: MT confidence < threshold, or multiple providers disagree, or TM has a different translation for the same source
- `manually_approved`: user accepted an MT suggestion
- `manually_rejected`: user rejected MT output and entered their own

**Review workflow:**
1. MT runs after detection (post-Cicerone) or on-demand from the Translate tab
2. Results populate `target_text` with `translation_provenance`
3. `review_required` instances surface in the UI with a review badge
4. User can accept, reject, or edit — state transitions recorded in provenance
5. QA gate in Verify checks translation provenance completeness

#### 1.2.3 Pipeline Integration

**Where MT runs:**
- **Post-detect** (after Cicerone, before Translate tab): TM lookup first (existing `_attach_tm_suggestions`), then MT for untranslated regions
- **On-demand**: user clicks "Auto-translate" in the Translate tab
- **Batch**: user selects multiple regions and applies MT

**New pipeline config fields:**

```python
@dataclass
class PipelineCfg:
    # ... existing fields ...
    mt_enabled: bool = False                    # opt-in; NullTranslator when False
    mt_provider_priority: List[str] = field(default_factory=lambda: ["tm_lookup", "deepl", "google", "llm"])
    mt_confidence_threshold: float = 0.85       # below this → review_required
    mt_review_on_disagreement: bool = True      # review when providers disagree
    mt_context_aware: bool = True               # send scene/glossary context to LLM
```

**New server endpoint:**

```
POST /api/translate
  body: { asset_id, region_ids: ["r1", "r2"], targ_lang, src_lang? }
  → { translations: { "r1": { text, provider, confidence, review_state, alternatives } } }

POST /api/translate/apply
  body: { asset_id, translations: { "r1": { text, provider } }, apply: bool }
  → persists target_text + translation_provenance to manifest
```

#### 1.2.4 Context-Aware LLM Translation

When `mt_context_aware = True` and the LLM provider is used:
- Build a context string from: manifest source language, Basil's semantic units, glossary entries, scene region descriptions, and adjacent text for multi-region signs
- Send as system prompt: "You are translating text on a {sign_type} in {source_language} to {target_language}. Context: {glossary_terms}. Adjacent text: {neighbor_texts}."
- Hash the context for provenance (reproducibility without storing full prompt)

#### 1.2.5 Round-Trip Workflow

The existing TMS round-trip (export → translate → import) remains the primary workflow. MT augments it:

1. **Pre-fill**: After detection, MT populates `target_text` for all non-DNT regions
2. **Export**: XLIFF/TMX export includes MT pre-translations as initial target segments
3. **External review**: Translator reviews/edits in their CAT tool
4. **Import**: Re-imported translations overwrite MT with `manually_approved` provenance
5. **TM storage**: Final QA-passed translations stored in TM (existing flow, unchanged)

### 1.3 Files to Modify

| File | Change |
|------|--------|
| `src/tofu/utils/translate.py` | Add `TranslationResult`, `TranslatorRegistry`, `DeepLTranslator`, `GoogleTranslator`, `LLMTranslator`, `TMLookupTranslator`; extend `translate()` signature with optional context param |
| `src/tofu/core/types.py` | Add `translation_provenance` field to `InstText`; add MT config fields to `PipelineCfg` |
| `src/tofu/core/pipeline.py` | Integrate MT post-detect (after Cicerone TM lookup); add MT stage logging |
| `server/main.py` | Add `POST /api/translate`, `POST /api/translate/apply` endpoints; wire MT provider registry |
| `server/requirements.txt` | Add `httpx` (for DeepL/Google API calls) — optional, degrades gracefully |
| `frontend/src/api.ts` | Add `translateRegions()`, `applyTranslations()` API functions |
| `frontend/src/App.tsx` | Add "Auto-translate" button in Translate step; show review badges for `review_required` regions |
| `frontend/src/RegionTable.tsx` (or equivalent) | Show provider icon, confidence, review state per region |
| `tests/test_translate.py` (new) | Test provider registry, fallback, provenance, review states, TM-first routing |

### 1.4 Acceptance Criteria

- `NullTranslator` remains the default; no MT runs unless `mt_enabled = True` or an API key is set
- TM lookup always runs before external MT providers
- Every MT-translated region carries `translation_provenance` with provider, confidence, and review state
- `review_required` state set when: confidence < threshold, providers disagree, or TM conflict
- User can accept/reject/edit MT suggestions; provenance records the transition
- Existing TMS round-trip (export/import) works unchanged
- API keys read from environment variables, never hardcoded
- All providers degrade gracefully when unavailable (return empty, log warning)
- LLM context includes glossary and semantic unit info when `mt_context_aware = True`

### 1.5 Constraints

- **Backward compatibility**: `NullTranslator` default; existing `translate()` callers work unchanged (new return type is opt-in via new method)
- **No new heavy dependencies**: `httpx` is lightweight and already common; LLM providers use standard HTTP
- **No cloud for source images**: Only text segments are sent to MT APIs, never image pixels
- **API key security**: Keys via env vars only; never logged or persisted in DB

---

## Part 2: Full TM Suite + Snapshot Diff

### 2.1 Current State

**Already implemented:**
- `server/db.py`: `store_tm_record`, `find_tm_candidates`, `list_tm_records`, `delete_tm_record`, `tm_count`
- `server/main.py`: `GET /api/projects/{pid}/memory`, `DELETE /api/memory/{record_id}`
- `frontend/src/MemoryPanel.tsx`: TM browsing with thumbnails, delete
- `frontend/src/HistoryPanel.tsx`: Snapshot list, restore, delete; event timeline
- `server/db.py`: `add_snapshot`, `list_snapshots`, `get_snapshot`, `delete_snapshot`
- `server/main.py`: `GET /api/projects/{pid}/history`, `POST /api/snapshots/{sid}/restore`
- `src/tofu/layers/memory.py`: `update()` (draft records), `lookup()` (three-tier matching)
- `src/tofu/utils/interchange.py`: XLIFF/TMX/TSV/CSV/TXT export + import

**Missing:**
- TM record editing (update target_text, qa_score)
- TM search/filter (by source text, target text, language pair, QA score range, date range)
- Cross-project TM search (search across all projects' TM records)
- Snapshot diffing (compare two snapshots' manifests)
- Event filtering (by kind, date range)
- Batch TM operations (bulk delete, bulk export)
- TM export/import (TMX format specifically for TM records, distinct from manifest export)
- TM merge/deduplicate (find and merge duplicate records)
- Snapshot comparison view (visual diff in frontend)
- Project archiving (soft-delete with data retention)

### 2.2 Design

#### 2.2.1 TM Record Editing

**New endpoint:**
```
PATCH /api/memory/{record_id}
  body: { target_text?, qa_score?, source_text? }
  → updated record
```

**DB function:** `update_tm_record(rid, **fields) -> Optional[Dict]`

**Frontend:** MemoryPanel gains inline edit mode per record — click to edit target_text, save/cancel buttons.

#### 2.2.2 TM Search & Filter

**New endpoint:**
```
GET /api/projects/{pid}/memory?query=street&source_lang=en&target_lang=es&min_qa=0.8&max_qa=1.0&since=1234567890&until=1234567890
  → { records: [...], count: N }
```

**DB function:** `search_tm_records(pid, query, filters) -> List[Dict]` with SQL LIKE on source_text/target_text/normalized_text, range filters on qa_score and created_at.

**Cross-project search:**
```
GET /api/memory/search?query=street&source_lang=en&target_lang=es
  → { records: [...], count: N, projects: { pid: pname } }
```

**DB function:** `search_tm_records_global(query, filters) -> List[Dict]` — searches across all projects, includes project name in results.

**Frontend:** MemoryPanel gains search bar, language pair filter dropdowns, QA score range slider, date range picker. Cross-project toggle.

#### 2.2.3 Batch TM Operations

**New endpoints:**
```
POST /api/memory/bulk-delete
  body: { record_ids: [1, 2, 3] }
  → { deleted: 3 }

POST /api/memory/bulk-export
  body: { record_ids: [1, 2, 3], format: "tmx" | "csv" | "json" }
  → file download
```

**Frontend:** Checkbox selection in MemoryPanel, bulk action bar appears when ≥1 selected.

#### 2.2.4 TMX Export/Import for TM Records

Distinct from the existing manifest-level TMX export (which exports current manifest segments), this exports TM records directly:

**New endpoint:**
```
POST /api/projects/{pid}/memory/export?format=tmx
  body: { record_ids?: [1, 2] }  // optional; all if omitted
  → TMX file download

POST /api/projects/{pid}/memory/import
  body: file upload (TMX)
  → { imported: N, skipped: M, errors: [...] }
```

**Logic:** Uses `interchange.export_tmx()` with TM records as pairs. Import parses TMX and inserts via `store_tm_record()`, deduplicating by (project_id, normalized_text, source_lang, target_lang).

#### 2.2.5 TM Merge/Deduplicate

**New endpoint:**
```
POST /api/projects/{pid}/memory/deduplicate
  → { merged: N, duplicates: [{ kept: 1, removed: [2, 3], reason: "identical normalized_text" }] }
```

**DB function:** `find_tm_duplicates(pid) -> List[Dict]` — groups by (normalized_text, source_lang, target_lang), returns groups with >1 record.

**Merge logic:** Keep highest qa_score record in each group, delete others. Log merge as project event.

#### 2.2.6 Snapshot Diffing

**New endpoint:**
```
GET /api/snapshots/diff?left={sid1}&right={sid2}
  → {
      added: [{ region_id, text, target_text }],
      removed: [{ region_id, text, target_text }],
      modified: [{ region_id, field, old_value, new_value }],
      unchanged_count: N,
    }
```

**Logic:** Load both snapshots via `get_snapshot()`, deserialize manifests, compare instance lists by region ID. Detect: new regions, removed regions, text changes, target_text changes, bbox changes, DNT/excluded flag changes.

**Frontend:** HistoryPanel gains "Compare" mode — select two snapshots, show diff table with color-coded additions/removals/modifications. Optional visual diff (overlay bbox changes on asset thumbnail).

#### 2.2.7 Event Filtering

**New endpoint:**
```
GET /api/projects/{pid}/events?kind=render&since=1234567890&until=1234567890
  → { events: [...] }
```

**DB function:** `list_events_filtered(pid, kind, since, until) -> List[Dict]`

**Frontend:** HistoryPanel events tab gains filter dropdown (by kind) and date range.

#### 2.2.8 Project Archiving

**Schema change:** Add `archived` column to `projects` table:
```sql
ALTER TABLE projects ADD COLUMN archived INTEGER NOT NULL DEFAULT 0;
```

**Endpoints:**
```
PATCH /api/projects/{pid}
  body: { archived: true }
  → soft-archives project (excluded from default list, data retained)

GET /api/projects?include_archived=true
  → includes archived projects with visual indicator
```

**Frontend:** ProjectGate shows archived projects in a separate section with unarchive button.

### 2.3 Files to Modify

| File | Change |
|------|--------|
| `server/db.py` | Add `update_tm_record`, `search_tm_records`, `search_tm_records_global`, `find_tm_duplicates`, `merge_tm_duplicates`, `list_events_filtered`; add `archived` column migration |
| `server/main.py` | Add all new endpoints (TM edit, search, bulk ops, TMX export/import, dedup, snapshot diff, event filter, archive) |
| `src/tofu/utils/interchange.py` | Add `export_tmx_from_records()`, `import_tmx_to_records()` for TM-level TMX |
| `frontend/src/api.ts` | Add all new API functions |
| `frontend/src/MemoryPanel.tsx` | Add search, filter, inline edit, bulk select, cross-project toggle, export/import buttons |
| `frontend/src/HistoryPanel.tsx` | Add snapshot compare mode, event filtering |
| `frontend/src/ProjectGate.tsx` | Add archived section, archive/unarchive buttons |
| `tests/test_tm_operations.py` (new) | Test TM edit, search, dedup, TMX round-trip, cross-project search |
| `tests/test_snapshot_diff.py` (new) | Test snapshot diff logic: add, remove, modify detection |

### 2.4 Acceptance Criteria

- TM records can be edited (target_text, qa_score) with provenance of the edit
- TM search supports text query, language pair, QA range, date range, cross-project
- Batch delete and batch export work on selected records
- TMX export of TM records round-trips: export → import → same records (deduplicated)
- Deduplicate finds and merges records with identical (normalized_text, source_lang, target_lang), keeping highest QA
- Snapshot diff correctly identifies added/removed/modified regions between any two snapshots
- Event filtering by kind and date range works
- Project archiving soft-deletes (hidden from default list, data retained, unarchiveable)
- All new endpoints return proper error codes (404 for missing project/record, 400 for bad params)
- Existing TM operations (store, list, delete, lookup) work unchanged

### 2.5 Constraints

- **No schema breaking changes**: New columns added via migration in `init_db()`, existing data preserved
- **Performance**: TM search uses SQL LIKE with indexes; cross-project search limited to 500 results
- **TMX compliance**: TMX export follows TMX 1.4b standard (existing `interchange.export_tmx` already does)
- **Snapshot diff**: O(n) comparison by region ID; no full-text diff, field-level only

---

## Part 3: Video as a Separate Project (Phased)

### 3.1 Current State

- `AssetType.VIDEO` enum exists; `AssetInfo` has `frame_count`, `fps`, `duration`
- `TextManifest` has `asset_type`, `frame_count`, `fps`, `duration` fields
- `InstText` has `frame_index`, `temporal_span`, `track_id` fields for video
- `RenderParams` has `trajectory`, `start_frame`, `end_frame`, `fps`, `duration`
- `infer_asset_info()` classifies by file extension
- `TofuPipeline.process()` detects video and returns `FAILED` with "video pipeline not yet implemented"
- `ProjectGate.tsx` shows video as an asset kind option with "full support is in progress" label
- `projects` table has `asset_kind` column supporting "video"
- No frame extraction, no video processing logic, no temporal features

### 3.2 Design — Phase 1: Frame Extraction + Static Reuse

#### 3.2.1 Video Pipeline Module

New file: `src/tofu/core/video_pipeline.py`

```python
class VideoPipeline:
    """Frame-based video processing pipeline.
    
    Phase 1: Extract frames → run static pipeline per-frame → reassemble.
    Phase 2 (future): Temporal-coherent inpainting, optical-flow tracking.
    """
    
    def process(self, video_path, targ_lang, asset_info, ...) -> PipelineResult:
        1. Probe video metadata (fps, frame_count, duration, resolution)
        2. Extract frames (all or sampled — configurable)
        3. For each frame: run TofuPipeline._process_static()
        4. Collect per-frame manifests, merge into a video-level TextManifest
        5. Reassemble frames into output video
        6. Return PipelineResult with video-level manifest
```

#### 3.2.2 Frame Extraction

**New utility:** `src/tofu/utils/video.py`

```python
def probe_video(path) -> VideoMetadata:
    """Use ffprobe or OpenCV VideoCapture to get fps, frame_count, duration, resolution."""

def extract_frames(path, max_frames=None, sample_fps=None) -> List[np.ndarray]:
    """Extract frames from video.
    - max_frames: cap total frames (sample evenly if exceeded)
    - sample_fps: extract at reduced rate (e.g., 2fps instead of 30fps)
    Returns list of RGB frames.
    """

def reassemble_frames(frames, fps, output_path) -> str:
    """Write frames back to a video file using OpenCV VideoWriter or ffmpeg."""
```

**Dependency:** OpenCV (`cv2.VideoCapture` / `cv2.VideoWriter`) — already a dependency. No ffmpeg required (but optional for better codec support).

#### 3.2.3 Per-Frame Static Pipeline

For each extracted frame:
1. Run `TofuPipeline._process_static(frame, targ_lang, AssetInfo(asset_type=IMAGE), ...)`
2. Capture the per-frame `TextManifest` with `frame_index` set on each `InstText`
3. Capture the rendered output frame

**Frame-level manifest merging:**

```python
def merge_frame_manifests(frame_manifests: List[TextManifest]) -> TextManifest:
    """Merge per-frame manifests into a video-level manifest.
    
    - Set asset_type = VIDEO
    - Aggregate all instances across frames
    - Set frame_index on each instance
    - Assign track_id for text that appears in consecutive frames
      (Phase 1: simple bbox IoU matching between adjacent frames)
    - Set temporal_span = (first_frame, last_frame) for tracked instances
    - Set frame_count, fps, duration from video metadata
    """
```

#### 3.2.4 Simple Text Tracking (Phase 1)

**New function in `src/tofu/utils/video.py`:**

```python
def track_text_across_frames(frame_manifests: List[TextManifest], iou_threshold=0.5) -> Dict[str, str]:
    """Assign track_id to instances across frames.
    
    Phase 1 approach: greedy IoU matching between consecutive frames.
    - For each instance in frame N, find the best IoU match in frame N+1
    - If IoU > threshold, assign same track_id
    - If no match, assign new track_id
    - If multiple matches, pick highest IoU
    
    Returns {instance_id: track_id} mapping.
    """
```

This is deliberately simple — no optical flow, no Kalman filtering. It exists so the video manifest has `track_id` and `temporal_span` populated for Phase 2 to build on.

#### 3.2.5 Pipeline Integration

**`TofuPipeline.process()` update:**

```python
def process(self, asset, targ_lang, asset_info=None, ...):
    asset_info = asset_info or infer_asset_info(asset)
    if asset_info.asset_type == AssetType.VIDEO:
        from tofu.core.video_pipeline import VideoPipeline
        vp = VideoPipeline(self.cfg)
        return vp.process(asset, targ_lang, asset_info, ...)
    return self._process_static(asset, targ_lang, asset_info, ...)
```

**New `PipelineCfg` fields:**

```python
@dataclass
class PipelineCfg:
    # ... existing ...
    video_max_frames: int = 300          # cap extracted frames
    video_sample_fps: Optional[float] = None  # None = native fps; 2.0 = sample at 2fps
    video_tracking_iou: float = 0.5      # Phase 1 tracking threshold
```

#### 3.2.6 Server & Frontend

**Server changes:**
- `POST /api/process` — detect video asset type, route to video pipeline
- `POST /api/render` — for video assets, render all frames (or tracked region subset)
- Video upload already works (asset_kind = "video" in project creation)
- Video output served from `/uploads/` like images

**Frontend changes:**
- `App.tsx` — video assets show frame timeline scrubber instead of single image
- Frame navigation: prev/next, play/pause, jump to frame
- Region table shows frame index column for video assets
- Render preview shows output video or frame-by-frame

#### 3.2.7 Video Manifest Serialization

Video manifests are larger (N frames × M regions). Storage considerations:
- Snapshots store full manifest JSON (existing) — may be large for video
- Consider: snapshot only the current frame's manifest, or a diff from the video-level baseline
- Phase 1: store full manifest; optimize later if needed

### 3.3 Phase 2 Extension Points (Future, Not Implemented Now)

Design the Phase 1 architecture so Phase 2 can plug in without rework:

| Extension Point | Phase 1 | Phase 2 |
|----------------|---------|---------|
| Frame extraction | All frames (or sampled) | Keyframe detection + interpolation |
| Text tracking | Greedy IoU | Optical flow + Kalman filter |
| Inpainting | Per-frame independent | Temporal-coherent (frame N-1 and N+1 as context) |
| OCR | Per-frame independent | Temporal voting (aggregate reads across frames for same track_id) |
| Rendering | Per-frame independent | Temporal-coherent style (consistent font/color across track) |
| Reassembly | Simple frame concat | ffmpeg with original codec/audio stream |

**Clear seams in code:**
- `track_text_across_frames()` — replace with optical-flow-based tracker
- `VideoPipeline.process()` — add temporal inpainting stage between per-frame cleanse and reassembly
- `merge_frame_manifests()` — add temporal voting for OCR text per track_id

### 3.4 Files to Modify

| File | Change |
|------|--------|
| `src/tofu/core/video_pipeline.py` (new) | `VideoPipeline` class with `process()`, frame iteration, manifest merging |
| `src/tofu/utils/video.py` (new) | `probe_video()`, `extract_frames()`, `reassemble_frames()`, `track_text_across_frames()` |
| `src/tofu/core/pipeline.py` | Update `process()` to route video to `VideoPipeline` instead of returning FAILED |
| `src/tofu/core/types.py` | Add `video_max_frames`, `video_sample_fps`, `video_tracking_iou` to `PipelineCfg` |
| `server/main.py` | Handle video asset processing/rendering; video output serving |
| `frontend/src/App.tsx` | Video frame timeline, frame navigation, video render preview |
| `frontend/src/api.ts` | Video-specific types (VideoMetadata, FrameManifest) |
| `tests/test_video_pipeline.py` (new) | Test frame extraction, tracking, manifest merge, reassembly |

### 3.5 Acceptance Criteria

**Phase 1:**
- Video assets (mp4, mov, avi, mkv, webm, m4v) are processed end-to-end: detect → cleanse → scribe → verify → memory
- Frame extraction produces RGB frames at native or sampled fps
- Each frame runs through the existing static pipeline without modification
- Per-frame manifests are merged into a video-level manifest with correct `frame_index`, `track_id`, `temporal_span`
- Output video is reassembled from rendered frames at original fps
- `video_max_frames` caps total frames; excess frames are evenly sampled
- Video processing logs per-frame progress in pipeline logs
- Existing static image pipeline is completely unchanged
- Video projects show in ProjectGate with video icon (already partially done)

**Phase 2 readiness:**
- `track_text_across_frames()` is a single function replaceable with optical-flow tracker
- `VideoPipeline.process()` has clear stage boundaries for temporal inpainting insertion
- `merge_frame_manifests()` supports temporal OCR voting extension

### 3.6 Constraints

- **No static pipeline changes**: Video pipeline calls `_process_static()` as-is; no modifications to existing layers
- **OpenCV only for I/O**: Frame extraction/reassembly via `cv2.VideoCapture`/`cv2.VideoWriter`; no ffmpeg dependency required (optional for better codecs)
- **Memory**: Frame extraction loads frames one at a time (generator pattern), not all into memory
- **Performance**: Video processing is inherently slower; progress logging and frontend progress bar required
- **Audio**: Phase 1 strips audio (VideoWriter doesn't carry audio); document this limitation. Phase 2 can mux original audio back via ffmpeg
- **No temporal inpainting in Phase 1**: Each frame is inpainted independently; visible flicker is expected and documented

---

## Implementation Order

### Phase A: MT Backend (Part 1)
1. **Types** — `TranslationResult`, `translation_provenance` on `InstText`, MT config on `PipelineCfg`
2. **Providers** — `DeepLTranslator`, `GoogleTranslator`, `LLMTranslator`, `TMLookupTranslator`
3. **Registry** — `TranslatorRegistry` with priority routing and fallback
4. **Server endpoints** — `POST /api/translate`, `POST /api/translate/apply`
5. **Pipeline integration** — Post-detect MT stage, provenance recording
6. **Frontend** — Auto-translate button, review badges, provider display
7. **Tests** — Provider registry, fallback, provenance, review states

### Phase B: TM Suite + Snapshot Diff (Part 2)
1. **DB schema** — `archived` column migration
2. **DB functions** — `update_tm_record`, `search_tm_records`, `search_tm_records_global`, `find_tm_duplicates`, `merge_tm_duplicates`, `list_events_filtered`
3. **Server endpoints** — All new TM, snapshot diff, event filter, archive endpoints
4. **Interchange** — TMX export/import for TM records
5. **Frontend** — MemoryPanel search/filter/edit/bulk, HistoryPanel compare/filter, ProjectGate archive
6. **Tests** — TM operations, snapshot diff, TMX round-trip, dedup

### Phase C: Video Pipeline (Part 3)
1. **Video utils** — `probe_video()`, `extract_frames()`, `reassemble_frames()`, `track_text_across_frames()`
2. **VideoPipeline** — `process()`, frame iteration, manifest merging
3. **Pipeline integration** — Route video in `TofuPipeline.process()`
4. **Server** — Video processing/rendering endpoints
5. **Frontend** — Frame timeline, navigation, video preview
6. **Tests** — Frame extraction, tracking, manifest merge, end-to-end video process

### Phase D: Integration Testing
1. MT + TM interaction (MT pre-fill → TM storage on QA pass)
2. Video + MT (auto-translate across all frames)
3. Video + TM (TM lookup across frame manifests)
4. Full regression: all existing tests pass unchanged

---

## Cross-Cutting Concerns

### Provenance Chain

With MT provenance added, the full provenance chain per region becomes:

```
OCR detection → ocr_provenance (multi-candidate, verification)
    ↓
Translation → translation_provenance (provider, confidence, review state)
    ↓
Cleanse → repair_provenance (provider, quality gate, residual check)
    ↓
Verify → QA scores (per-instance metrics)
    ↓
Memory → TM record (qa_score, style_fingerprint, phash)
```

Every stage's provenance is independent and inspectable. The user can trace any rendered pixel back through the full chain.

### Testing Strategy

- **Unit tests**: Each new function tested in isolation with mocks
- **Integration tests**: API endpoints tested with test client + temp DB
- **Regression**: All existing tests must pass unchanged
- **Degradation tests**: Every optional feature (MT providers, PaddleOCR, video) tested in unavailable state
- **Round-trip tests**: TMX export → import → same data; snapshot → restore → same manifest

### Dependency Management

| Dependency | Required? | Purpose |
|-----------|-----------|---------|
| `httpx` | Optional | MT provider HTTP calls (DeepL, Google, LLM) |
| `openai` or `anthropic` | Optional | LLM provider SDKs (if using hosted LLM) |
| OpenCV `cv2` | Already required | Video frame I/O |
| ffmpeg | Optional | Better video codec support, audio muxing (Phase 2) |

All optional dependencies degrade gracefully: `ImportError` → feature unavailable, log warning, continue without.

### Forbidden

- No mixing video pipeline with static-image pipeline (video calls static per-frame, static never calls video)
- No implementation without plan approval
- No changes to existing test assertions (only additions)
- No hardcoded API keys
- No source image pixels sent to MT APIs (text only)
- No Tesseract integration (low accuracy, per existing plan)
