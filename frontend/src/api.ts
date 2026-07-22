// 🍢 ToFU — API client (mirrors server/main.py contracts)

export interface AssetInfo {
  asset_type: string;
  frame_count: number;
  fps: number | null;
  duration: number | null;
  source: string | null;
}

export interface UploadResponse {
  asset_id: string;
  filename: string;
  asset_info: AssetInfo;
  asset_url?: string;
}

// --- projects (localization management) ---

export interface ProjectAsset {
  asset_id: string;
  project_id: string;
  filename: string | null;
  uploaded_at: number;
  is_active: number;
  asset_url?: string | null;
  has_manifest?: boolean;
}

export type AssetKind = "image" | "video";

export interface Project {
  id: string;
  name: string;
  target_lang: string;
  source_lang: string | null;
  asset_kind: AssetKind;
  created_at: number;
  updated_at: number;
  asset_count?: number;
  snapshot_count?: number;
  assets?: ProjectAsset[];
  active_asset?: ProjectAsset | null;
}

export interface SnapshotMeta {
  id: number;
  project_id: string;
  asset_id: string;
  reason: string;
  region_count: number;
  translated: number;
  created_at: number;
}

export interface ProjectEvent {
  id: number;
  project_id: string;
  kind: string;
  detail: string;
  created_at: number;
}

export interface ProjectHistory {
  snapshots: SnapshotMeta[];
  events: ProjectEvent[];
}

export interface FontOption {
  path: string;
  coverage: number;
}

export interface FontWeight {
  path: string;
  subfamily: string;
  weight_class: number;
  coverage: number;
}

export interface FontFamily {
  family: string;
  best_path: string;
  best_coverage: number;
  weights: FontWeight[];
}

export interface LanguageOption {
  code: string;
  name: string;
}

export interface BBox {
  x: number;
  y: number;
  width: number;
  height: number;
}

export interface TMSuggestion {
  target_text: string;
  score: number;
  method: "exact" | "fuzzy" | "visual";
  source_asset_id: string;
  record_id: number | null;
}

export interface TMRecord {
  id: number;
  project_id: string;
  asset_id: string;
  region_id: string;
  source_text: string;
  normalized_text: string;
  source_lang: string | null;
  target_lang: string;
  target_text: string;
  style_fingerprint: string | null;
  phash: string | null;
  thumb_path: string | null;
  thumb_url: string | null;
  qa_score: number;
  created_at: number;
}

export interface InstText {
  id: string;
  bounding_box: BBox;
  text: string | null;
  target_text: string | null;
  confidence: number | null;
  detected_language: string | null;
  language?: string | null;  // user-confirmed per-region source language
  reading_order: number | null;
  dnt: boolean;
  excluded?: boolean;  // removed from the workspace UI/export; still erased on render, unlike dnt
  target_language: string | null;
  glyph_fallback?: boolean | null;  // scribe swapped fonts: the requested face lacked codepoints for this text
  tm_suggestion?: TMSuggestion | null;  // translation-memory match from a prior approved render
  recognition_history?: Array<{ stage: string; engine: string; candidate_text?: string; candidate_confidence?: number; primary_text?: string; primary_confidence?: number; accepted: boolean; reason: string }> | null;
  repair_provenance?: {
    requested_provider: string;
    executed_provider?: string;
    strategy: string;
    confidence: number;
    auto_accepted: boolean;
    review_required: boolean;
    reason: string;
    candidates?: Array<{
      provider: string;
      accepted: boolean;
      decision: string;
      group_ids: string[];
      evidence?: {
        artifact?: { id: string; provider: string; decision: string; bbox: BBox; url: string; cache_key?: string } | null;
      };
    }>;
  } | null;
  resolved_font_family?: string | null;  // what "auto" (style_profile.font_family unset) currently renders with — display hint only, never an override
  font_match?: FontMatch | null;
  semantic_assignment?: {
    schema: number;
    unit_id: string;
    source_text: string;
    target_text: string;
    text: string;
    semantic_region_id?: string;
    anchor_id?: string;
    target_positions: number[];
    method: string;
    confidence: number;
    geometry_unchanged: boolean;
  } | null;
  garnish_override?: GarnishProfile | null;
  garnish_enabled?: boolean | null;
  garnish_scope?: "whole_selection" | "per_region";
  garnish_regions?: GarnishRegion[];
  segmentation_mask?: { polygon: number[][]; confidence: number; holes?: number[][][] | null } | null;
  style_profile?: {
    font_family: string | null;
    font_weight: string | null;
    color: string | null;
    font_size: number | null;
    italic: boolean | null;
    underline: boolean | null;
    underline_offset?: number | null;
    underline_width?: number | null;
    subscript: boolean | null;
    superscript: boolean | null;
    align_h: string | null;
    align_v: string | null;
    justification: string | null;
    indent: number | null;
    tracking: number | null;
    kerning: number | null;
    leading: number | null;
    baseline_shift: number | null;
    tab_width: number | null;
    tsume: number | null;
    stroke_color: string | null;
    stroke_width: number | null;
    target_orientation: "horizontal" | "vertical" | null;
    word_order: "ltr" | "rtl" | null;
    transform?: { skew_x?: number; skew_y?: number; arc?: number; preset?: string; amount?: number; scale_x?: number; scale_y?: number; offset_x?: number; offset_y?: number; truncate_offset?: boolean; truncate_offset_x?: boolean; truncate_offset_y?: boolean; wrap_text?: boolean } | null;
  } | null;
  background_profile?: {
    semantic_label: string | null;  // containing scene surface: "panel" | "bordered_region" | ...
    texture: string | null;         // "flat" | "textured"
    material?: string | null;       // user-facing: "brick / masonry", "painted sign", etc.
    gradients: string[] | null;
    patterns: string[] | null;
    dominant_color: string | null;  // hex
    surface_texture?: string | null;
    cleanse_strategy?: "flat" | "smooth_gradient" | "telea" | "neural" | null;
  } | null;
  characteristics?: {
    font_style: string | null;   // detected descriptor: "bold" | "italic" | "bold italic" | "regular"
    color: string | null;
    size: number | null;         // detected text height in px (ascender-to-descender)
    positioning?: {              // detected geometry (typography layer)
      rotation_deg?: number | null;
      slant_deg?: number | null;
      stroke_ratio?: number | null;
    } | null;
  } | null;
}

export interface SceneRegion {
  bbox: BBox;
  semantic_label: string;  // "panel" | "bordered_region" | "surface" | backend-specific
  confidence: number;
  background_color: string | null;
  border_detected: boolean;
  polygon: number[][] | null;
  texture: string | null;
  material?: string | null;
  garnish_profile?: GarnishProfile | null;
}

export interface GarnishProfile {
  edge_blur_px: number; erosion_px: number; dilation_px: number;
  grain_strength: number; gamma_shift: number; smudge_strength: number;
  smudge_angle_deg: number; source_confidence: number;
}

export interface GarnishRegion {
  id: string;
  polygon: number[][];
  enabled?: boolean | null;
  profile?: GarnishProfile | null;
  source?: string;
}

export interface TextManifest {
  asset_id: string;
  total_regions: number;
  src_lang: string | null;
  targ_lang: string | null;
  img_dim: [number, number] | null;  // original (width, height): the coordinate space of all bboxes
  scene_regions: SceneRegion[];
  semantic_units?: SemanticTextUnit[];
  asset_type: string;
  frame_count: number;
  fps: number | null;
  duration: number | null;
  prcssng_time: number | null;
  instances: InstText[];
}

export interface SemanticTextUnit {
  id: string;
  region_ids: string[]; // source visual reading order; rN identity remains immutable
  source_text: string;
  bbox: BBox;
  entity_type: string;
  confidence: number;
  analysis_provider: string;
  semantic_roles: Record<string, string>;
  review_required: boolean;
  substitution?: {
    applied?: boolean;
    target_text?: string;
    source_region_order?: string[];
    spatial_anchor_order?: string[];
    target_region_order?: string[];
    assignments?: SemanticAssignment[];
    method?: string;
    confidence?: number;
  } | null;
}

export interface SemanticAssignment {
  region_id: string;
  anchor_id?: string;
  text: string;
  target_positions: number[];
  method: string;
  confidence: number;
}

export interface SemanticSubstitutionPlan {
  schema: number;
  unit_id: string;
  source_text: string;
  target_text: string;
  source_region_order: string[];
  spatial_anchor_order?: string[];
  target_region_order: string[];
  assignments: SemanticAssignment[];
  method: string;
  confidence: number;
  review_required: boolean;
  warnings: string[];
}

export interface GlossaryMeta {
  mode: string;
  source_format: string;
  filename: string;
  uploaded_at: string;
  entry_count: number;
  language_pairs: string[][];
}

export interface GlossaryStatus {
  global: GlossaryMeta | null;
  project: GlossaryMeta | null;
  effective_mode: string | null;
  effective_entry_count: number;
  effective_language_pairs: string[][];
}

export interface GlossaryUploadResult {
  entry_count: number;
  language_pairs: string[][];
  mode: string;
  source_format: string;
  filename: string;
}

export interface ValidationIssue {
  severity: string;
  code: string;
  message: string;
  suggestion: string | null;
  region_id: string | null;
}

export interface PreflightInsight {
  key: string;
  kind: "font_substitute" | "style_reference" | "licensed_font_candidate" | string;
  title: string;
  detail: string;
  severity: "info" | "review" | "warning" | string;
  region_id: string | null;
  region_ids: string[];
  confidence: number | null;
  visual_score: number | null;
  family: string | null;
  subfamily: string | null;
  font_path: string | null;
  license: string | null;
  foundry: string | null;
  url: string | null;
  source: string | null;
}

export interface ValidationReport {
  passed: boolean;
  issues: ValidationIssue[];
  scrpt_spprt: Record<string, string>;
  glyph_segmentation_score: number | null;
  render_quality_score: number | null;
  expansion_fit: Record<string, number>;
  suggested_actions: string[];
  insights: PreflightInsight[];
}

export interface ProcessResult {
  success: boolean;
  status: string;
  validation_report: ValidationReport | null;
  qa_report: { overall_score: number | null } | null;
  memory_updates: unknown[] | null;
  logs: string[];
  output_url: string | null;
  text_manifest: TextManifest | null;
}

export interface RenderLogEntry {
  ts: string;
  stage: string;
  level: "info" | "warning" | "error";
  message: string;
  duration_ms?: number;
}

export interface QACoverage {
  instances_assessed: number;
  regions_total: number;
  dnt: number;
  translated: number;
  untranslated: number;
  rendered: number;
  fallback_font: number;
}

export interface QAReport {
  overall_score: number | null;
  per_asset_instance_score?: Record<string, Record<string, number>>;
  progress?: QACoverage;
  metrics?: Record<string, unknown>;
  recommendations?: string[];
}

export interface RenderResult {
  output_url: string | null;
  qa_report: QAReport | null;
  qa_passed: boolean;
  qa_threshold: number;
  validation_report: ValidationReport | null;
  text_manifest: TextManifest | null;
  logs: RenderLogEntry[];
  errors: string[];
}

export interface DetectStreamEvent {
  stage: "scene" | "cicerone" | "finalize" | "refine" | "zoom" | "vertical_split" | "paddle_rescue" | "polish" | "savor" | "wasabi" | "menu" | "enrich" | "memory" | "complete" | "error";
  status?: "running" | "complete";
  pass?: number;
  regions?: SceneRegion[] | number;  // scene: region list; cicerone/refine: running count
  langset?: string[];               // refine: the language-tuned charset in use
  manifest?: TextManifest;
  message?: string;
  engine?: "easyocr" | "paddleocr" | "null";  // which OCR engine actually ran
  corrected?: number;  // savor: complete -- regions Savor's taste test actually rewrote
  matched?: number;    // memory: complete -- regions with a TM suggestion this pass
  tm_matched?: number; // complete -- same count, mirrored onto the terminal event
}

export interface OcrRegionResult {
  text: string;
  confidence: number;
  detected_language: string | null;
  engine_missing?: boolean;  // OCR engine absent on the server (config issue)
}

export interface RefineRegionResult {
  regions: {
    polygon: [number, number][];
    bbox: BBox;
    text: string;
    confidence: number;
  }[];
}

export interface LanguageScanResult {
  engine: "easyocr" | "null";
  detected_lang: string | null;
  match: boolean | null;  // null: indeterminate (no text / no project source)
  project_source_lang: string | null;
  locked: boolean;        // this scan just locked the project's source language
}

export interface ImportResult {
  imported: number;
  missing: string[];
  extra: string[];
  translations: Record<string, string>;
  format?: string;
  matched_by?: Record<string, number>;
  unresolved?: Array<{ id: string | null; source: string }>;
  empty_targets?: number;
}

export interface FontMatchCandidate {
  family: string;
  subfamily?: string | null;
  font_path?: string | null;
  license: "installed" | "commercial" | "free" | "unknown";
  available: boolean;
  score?: number | null;
  url?: string | null;
  preview_url?: string | null;
  foundry?: string | null;
  source?: string | null;
  reason?: string | null;
}

export interface FontMatch {
  schema: number;
  status: "matched" | "review" | "unavailable";
  provider: string;
  confidence: number;
  margin: number;
  source_text: string;
  candidates: FontMatchCandidate[];
  contextual_candidates?: string[];
  recommended_substitute?: { font_path: string; family: string; subfamily?: string; score: number } | null;
  external_candidates?: FontMatchCandidate[];
  external_provider?: { name: string; enabled: boolean; reason?: string; error?: string };
}

async function json<T>(res: Response): Promise<T> {
  if (!res.ok) {
    const text = await res.text();
    // the vite dev proxy reports a dead backend as a bare 500 — surface
    // the real condition instead of a mystery "500: Internal Server Error"
    if (res.status === 500 && (!text || /ECONNREFUSED|ECONNRESET|socket hang up|proxy/i.test(text))) {
      throw new Error("Cannot reach the ToFU backend (port 8000). Start the server, then retry.");
    }
    let detail = text;
    try {
      const parsed = JSON.parse(text) as { detail?: unknown };
      if (typeof parsed.detail === "string") detail = parsed.detail;
    } catch {
      // not JSON — keep raw text
    }
    throw new Error(`${res.status}: ${detail}`);
  }
  return res.json() as Promise<T>;
}

// --- upload + languages + fonts ---

export async function uploadAsset(file: File, projectId?: string): Promise<UploadResponse> {
  const form = new FormData();
  form.append("file", file);
  const qs = projectId ? `?project_id=${encodeURIComponent(projectId)}` : "";
  return json(await fetch(`/api/assets${qs}`, { method: "POST", body: form }));
}

export async function sha256File(file: File): Promise<string> {
  const buf = await crypto.subtle.digest("SHA-256", await file.arrayBuffer());
  return Array.from(new Uint8Array(buf)).map((b) => b.toString(16).padStart(2, "0")).join("");
}

export interface DuplicateCheckResult {
  duplicate: boolean;
  project_id: string | null;
  project_name: string | null;
}

export async function checkDuplicateAsset(hash: string): Promise<DuplicateCheckResult> {
  return json(await fetch(`/api/assets/check-duplicate?hash=${encodeURIComponent(hash)}`));
}

// --- projects ---

export async function createProject(
  name: string,
  targetLang: string,
  assetKind: AssetKind = "image"
): Promise<Project> {
  return json(
    await fetch("/api/projects", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ name, target_lang: targetLang, asset_kind: assetKind }),
    })
  );
}

export async function listProjects(): Promise<Project[]> {
  const data = await json<{ projects: Project[] }>(await fetch("/api/projects"));
  return data.projects;
}

export async function getProject(id: string): Promise<Project> {
  return json(await fetch(`/api/projects/${encodeURIComponent(id)}`));
}

export async function updateProject(
  id: string,
  updates: Partial<{ name: string; target_lang: string; source_lang: string }>
): Promise<Project> {
  return json(
    await fetch(`/api/projects/${encodeURIComponent(id)}`, {
      method: "PATCH",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(updates),
    })
  );
}

export async function deleteProject(id: string): Promise<{ ok: boolean }> {
  return json(await fetch(`/api/projects/${encodeURIComponent(id)}`, { method: "DELETE" }));
}

export async function getProjectHistory(id: string, assetId?: string): Promise<ProjectHistory> {
  const qs = assetId ? `?asset_id=${encodeURIComponent(assetId)}` : "";
  return json(await fetch(`/api/projects/${encodeURIComponent(id)}/history${qs}`));
}

export async function getProjectMemory(id: string): Promise<{ records: TMRecord[]; count: number }> {
  return json(await fetch(`/api/projects/${encodeURIComponent(id)}/memory`));
}

export async function deleteMemoryRecord(recordId: number): Promise<{ ok: boolean }> {
  return json(await fetch(`/api/memory/${recordId}`, { method: "DELETE" }));
}

export async function snapshotAsset(
  projectId: string,
  assetId: string,
  reason: string
): Promise<{ snapshot_id: number | null }> {
  return json(
    await fetch(`/api/projects/${encodeURIComponent(projectId)}/snapshots`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ asset_id: assetId, reason }),
    })
  );
}

export async function restoreSnapshot(snapshotId: number): Promise<TextManifest> {
  return json(await fetch(`/api/snapshots/${snapshotId}/restore`, { method: "POST" }));
}

export async function deleteSnapshot(snapshotId: number): Promise<{ ok: boolean }> {
  return json(await fetch(`/api/snapshots/${snapshotId}`, { method: "DELETE" }));
}

export async function scanAssetLanguage(assetId: string): Promise<LanguageScanResult> {
  return json(
    await fetch(`/api/assets/${encodeURIComponent(assetId)}/scan-language`, { method: "POST" })
  );
}

export async function deleteProjectAsset(
  projectId: string,
  assetId: string
): Promise<{ ok: boolean }> {
  return json(
    await fetch(
      `/api/projects/${encodeURIComponent(projectId)}/assets/${encodeURIComponent(assetId)}`,
      { method: "DELETE" }
    )
  );
}

export async function activateAsset(
  projectId: string,
  assetId: string
): Promise<{ asset_id: string; asset_url: string; manifest: TextManifest | null }> {
  return json(
    await fetch(
      `/api/projects/${encodeURIComponent(projectId)}/assets/${encodeURIComponent(assetId)}/activate`,
      { method: "POST" }
    )
  );
}

export async function fetchLanguages(): Promise<LanguageOption[]> {
  const data = await json<{ languages: LanguageOption[] }>(await fetch("/api/languages"));
  return data.languages;
}

export async function fetchFonts(lang: string): Promise<{ script: string; fonts: FontOption[]; families: FontFamily[] }> {
  return json(await fetch(`/api/fonts?lang=${encodeURIComponent(lang)}`));
}

export async function validateAsset(
  assetId: string,
  targLang: string,
  font?: string
): Promise<ValidationReport> {
  return json(
    await fetch("/api/validate", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ asset_id: assetId, targ_lang: targLang, font }),
    })
  );
}

// --- detect + manifest CRUD ---

export async function detectAsset(
  assetId: string,
  gpu?: boolean,
  languages?: string[]
): Promise<TextManifest> {
  return json(
    await fetch("/api/detect", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ asset_id: assetId, gpu, languages }),
    })
  );
}

export function detectAssetStream(
  assetId: string,
  languages: string[] | undefined,
  onEvent: (ev: DetectStreamEvent) => void,
  onFailure: (message: string) => void
): () => void {
  const qs = new URLSearchParams({ asset_id: assetId });
  if (languages?.length) qs.set("languages", languages.join(","));
  const es = new EventSource(`/api/detect/stream?${qs.toString()}`);
  es.onmessage = (e) => {
    const ev = JSON.parse(e.data) as DetectStreamEvent;
    // terminal events must close the socket BEFORE the handler runs, or
    // EventSource auto-reconnects and re-triggers detection
    if (ev.stage === "complete" || ev.stage === "error") es.close();
    onEvent(ev);
  };
  es.onerror = () => {
    es.close();
    onFailure("detection stream failed");
  };
  return () => es.close();
}

export async function getManifest(assetId: string): Promise<TextManifest> {
  return json(await fetch(`/api/manifest/${assetId}`));
}

export async function putManifest(
  assetId: string, manifest: TextManifest
): Promise<{ ok: boolean; total_regions: number; resolved_fonts: Record<string, string> }> {
  return json(
    await fetch(`/api/manifest/${assetId}`, {
      method: "PUT",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(manifest),
    })
  );
}

export async function semanticSubstitution(
  assetId: string,
  unitId: string,
  targetText: string,
  targLang: string,
  apply = false,
): Promise<{ manifest: TextManifest; plan: SemanticSubstitutionPlan; applied: boolean }> {
  return json(
    await fetch(`/api/semantic-units/${encodeURIComponent(assetId)}/${encodeURIComponent(unitId)}/substitution`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ target_text: targetText, targ_lang: targLang, apply }),
    })
  );
}

export async function uploadGlossary(
  file: File,
  scope: "global" | "project",
  mode: "auxiliary" | "merge" | "replace",
  projectId?: string,
  srcLang?: string,
  targLang?: string,
): Promise<GlossaryUploadResult> {
  const form = new FormData();
  form.append("file", file);
  const params = new URLSearchParams({ scope, mode });
  if (projectId) params.set("project_id", projectId);
  if (srcLang) params.set("src_lang", srcLang);
  if (targLang) params.set("targ_lang", targLang);
  return json(await fetch(`/api/glossary/upload?${params.toString()}`, { method: "POST", body: form }));
}

export async function fetchGlossaryStatus(projectId?: string): Promise<GlossaryStatus> {
  const query = projectId ? `?project_id=${encodeURIComponent(projectId)}` : "";
  return json(await fetch(`/api/glossary/status${query}`));
}

export async function deleteGlossary(scope: "global" | "project", projectId?: string): Promise<{ ok: boolean }> {
  const params = new URLSearchParams({ scope });
  if (projectId) params.set("project_id", projectId);
  return json(await fetch(`/api/glossary/${scope}?${params.toString()}`, { method: "DELETE" }));
}

export async function addRegion(
  assetId: string,
  bbox: BBox,
  text?: string,
  targetText?: string
): Promise<InstText> {
  return json(
    await fetch(`/api/manifest/${assetId}/regions`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ ...bbox, text, target_text: targetText }),
    })
  );
}

export async function deleteRegion(assetId: string, regionId: string): Promise<{ ok: boolean; total_regions: number }> {
  return json(
    await fetch(`/api/manifest/${assetId}/regions/${regionId}`, { method: "DELETE" })
  );
}

export async function updateRegion(
  assetId: string,
  regionId: string,
  updates: Partial<{ x: number; y: number; width: number; height: number; text: string; target_text: string; dnt: boolean; target_language: string; language: string; font: string }>
): Promise<InstText> {
  return json(
    await fetch(`/api/manifest/${assetId}/regions/${regionId}`, {
      method: "PATCH",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(updates),
    })
  );
}

// --- OCR on a specific region ---

export async function ocrRegion(assetId: string, bbox: BBox): Promise<OcrRegionResult> {
  return json(
    await fetch("/api/ocr-region", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ asset_id: assetId, bbox }),
    })
  );
}

export async function refineRegion(
  assetId: string,
  bbox: BBox,
  engine?: "easyocr" | "paddleocr"
): Promise<RefineRegionResult> {
  return json(
    await fetch("/api/detect/refine", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ asset_id: assetId, bbox, engine }),
    })
  );
}

// --- export + import ---

export async function exportFile(
  assetId: string,
  format: string,
  variant?: string,
  targLang?: string
): Promise<Blob> {
  const res = await fetch("/api/export", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ asset_id: assetId, format, variant, targ_lang: targLang }),
  });
  if (!res.ok) throw new Error(`${res.status}: ${await res.text()}`);
  return res.blob();
}

export async function importFile(assetId: string, file: File): Promise<ImportResult> {
  const form = new FormData();
  form.append("file", file);
  const res = await fetch(`/api/import?asset_id=${encodeURIComponent(assetId)}`, {
    method: "POST",
    body: form,
  });
  return json(res);
}

export async function matchFonts(assetId: string, allowExternal = false): Promise<{ manifest: TextManifest; local_matched: number; external_matched: number }> {
  return json(await fetch(`/api/font-match/${encodeURIComponent(assetId)}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ allow_external: allowExternal }),
  }));
}

// --- render ---

export async function renderAsset(
  assetId: string,
  targLang: string,
  font?: string,
  qaThreshold?: number
): Promise<RenderResult> {
  return json(
    await fetch("/api/render", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ asset_id: assetId, targ_lang: targLang, font, qa_threshold: qaThreshold }),
    })
  );
}

export async function renderPreview(
  assetId: string, targLang: string, manifest: TextManifest, signal?: AbortSignal
): Promise<{ output_url: string; text_manifest: TextManifest; cleanse_cache_key: string }> {
  return json(await fetch("/api/preview/render", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ asset_id: assetId, targ_lang: targLang, manifest }),
    signal,
  }));
}
export async function previewCandidateLocalized(assetId: string, candidateId: string, manifest?: TextManifest, targLang?: string): Promise<string> {
  const result = await json<{ preview_url: string }>(await fetch(`/api/preview/candidate/${encodeURIComponent(assetId)}/${encodeURIComponent(candidateId)}`, {
    method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ manifest, targ_lang: targLang }),
  }));
  return result.preview_url;
}

export interface InpaintPatch { id: string; bbox: BBox; polygon: number[][]; points?: number[][]; mode: string; strategy?: string; radius?: number; hardness?: number; candidate_id?: string; }
export interface TreatmentRequest { polygon?: number[][]; points?: number[][]; mode?: "auto" | "content_aware" | "texture"; radius?: number; hardness?: number; }
export async function createInpaintPatch(assetId: string, treatment: TreatmentRequest, manifest?: TextManifest) {
  return json(await fetch("/api/inpaint", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ asset_id: assetId, ...treatment, manifest }) })) as Promise<{ id: string; bbox: BBox; patches: InpaintPatch[]; revision: string }>;
}
export async function applyRepairCandidate(assetId: string, candidateId: string, cacheKey?: string) {
  return json(await fetch(`/api/inpaint/candidate/${encodeURIComponent(assetId)}/${encodeURIComponent(candidateId)}`, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ cache_key: cacheKey }) })) as Promise<{ id: string; bbox: BBox; patches: InpaintPatch[]; revision: string; already_applied: boolean }>;
}
export async function undoInpaint(assetId: string, patchId?: string) {
  return json(await fetch(`/api/inpaint/${encodeURIComponent(assetId)}${patchId ? `/${encodeURIComponent(patchId)}` : ""}`, { method: "DELETE" })) as Promise<{ ok: boolean; patches: InpaintPatch[]; revision: string }>;
}
export async function getTreatment(assetId: string) {
  return json(await fetch(`/api/treatment/${encodeURIComponent(assetId)}`)) as Promise<{ patches: InpaintPatch[]; revision: string }>;
}
export async function restoreTreatment(assetId: string, patchIds: string[]) {
  return json(await fetch(`/api/treatment/${encodeURIComponent(assetId)}`, {
    method: "PUT", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ patch_ids: patchIds }),
  })) as Promise<{ patches: InpaintPatch[]; revision: string }>;
}
export interface LocalizedBaseline { schema: number; manifest: TextManifest; patch_ids: string[]; }
export async function getLocalizedBaseline(assetId: string) {
  return json(await fetch(`/api/localized-baseline/${encodeURIComponent(assetId)}`)) as Promise<LocalizedBaseline>;
}
export async function captureLocalizedBaseline(assetId: string, manifest: TextManifest, patchIds: string[]) {
  return json(await fetch(`/api/localized-baseline/${encodeURIComponent(assetId)}`, {
    method: "PUT", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ manifest, patch_ids: patchIds }),
  })) as Promise<LocalizedBaseline>;
}

export interface InpaintingProvider {
  id: string;
  available: boolean;
  enabled?: boolean;
  promoted: boolean;
  review_required: boolean;
  kind: "deterministic" | "self_hosted" | "experimental" | "external_disabled" | "editor";
  runtime?: string | null;
  detail: string;
  revision?: string;
}

export async function getInpaintingProviders() {
  return json(await fetch("/api/inpainting/providers")) as Promise<{ providers: InpaintingProvider[] }>;
}

export interface RenderStreamEvent {
  stage: "tofu" | "scene" | "tofu_regions" | "cleanse" | "scribe" | "verify" | "save" | "complete" | "error";
  status?: "running" | "complete";
  passed?: boolean;          // tofu: complete
  issues?: number;           // tofu_regions: complete
  languages?: string[];      // tofu_regions: complete
  score?: number | null;     // verify: complete
  message?: string;          // error
  // "complete" stage mirrors RenderResult
  output_url?: string | null;
  qa_report?: QAReport | null;
  qa_passed?: boolean;
  qa_threshold?: number;
  validation_report?: ValidationReport | null;
  text_manifest?: TextManifest | null;
  logs?: RenderLogEntry[];
  errors?: string[];
}

/** SSE render, mirroring detectAssetStream. `regionIds`, when given, scopes
 * cleanse+scribe to that subset (per-region re-render from the QA Inspector)
 * — the server composites onto the asset's existing localized output rather
 * than reverting untouched regions to source text. */
export function renderAssetStream(
  assetId: string,
  targLang: string,
  onEvent: (ev: RenderStreamEvent) => void,
  onFailure: (message: string) => void,
  opts?: { font?: string; qaThreshold?: number; regionIds?: string[] }
): () => void {
  const qs = new URLSearchParams({ asset_id: assetId, targ_lang: targLang });
  if (opts?.font) qs.set("font", opts.font);
  if (opts?.qaThreshold != null) qs.set("qa_threshold", String(opts.qaThreshold));
  if (opts?.regionIds?.length) qs.set("region_ids", opts.regionIds.join(","));
  const es = new EventSource(`/api/render/stream?${qs.toString()}`);
  es.onmessage = (e) => {
    const ev = JSON.parse(e.data) as RenderStreamEvent;
    if (ev.stage === "complete" || ev.stage === "error") es.close();
    onEvent(ev);
  };
  es.onerror = () => {
    es.close();
    onFailure("render stream failed");
  };
  return () => es.close();
}

export async function approveRender(
  assetId: string,
  targLang: string,
  coverage?: { overall_score?: number | null; regions_total?: number; rendered?: number; dnt?: number }
): Promise<{ ok: boolean }> {
  return json(
    await fetch("/api/render/approve", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        asset_id: assetId, targ_lang: targLang,
        overall_score: coverage?.overall_score,
        regions_total: coverage?.regions_total,
        rendered: coverage?.rendered,
        dnt: coverage?.dnt,
      }),
    })
  );
}

// --- legacy process ---

export async function processAsset(
  assetId: string,
  targLang: string,
  font?: string
): Promise<ProcessResult> {
  return json(
    await fetch("/api/process", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ asset_id: assetId, targ_lang: targLang, font }),
    })
  );
}
