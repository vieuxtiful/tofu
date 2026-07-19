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
  target_language: string | null;
  segmentation_mask?: { polygon: number[][]; confidence: number } | null;
  style_profile?: {
    font_family: string | null;
    font_weight: string | null;
    color: string | null;
    font_size: number | null;
    italic: boolean | null;
    underline: boolean | null;
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
}

export interface TextManifest {
  asset_id: string;
  total_regions: number;
  src_lang: string | null;
  targ_lang: string | null;
  img_dim: [number, number] | null;  // original (width, height): the coordinate space of all bboxes
  scene_regions: SceneRegion[];
  asset_type: string;
  frame_count: number;
  fps: number | null;
  duration: number | null;
  prcssng_time: number | null;
  instances: InstText[];
}

export interface ValidationIssue {
  severity: string;
  code: string;
  message: string;
  suggestion: string | null;
  region_id: string | null;
}

export interface ValidationReport {
  passed: boolean;
  issues: ValidationIssue[];
  scrpt_spprt: Record<string, string>;
  glyph_segmentation_score: number | null;
  render_quality_score: number | null;
  expansion_fit: Record<string, number>;
  suggested_actions: string[];
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

export interface QAReport {
  overall_score: number | null;
  per_asset_instance_score?: Record<string, Record<string, number>>;
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
  stage: "scene" | "cicerone" | "finalize" | "refine" | "zoom" | "polish" | "enrich" | "complete" | "error";
  status?: "running" | "complete";
  pass?: number;
  regions?: SceneRegion[] | number;  // scene: region list; cicerone/refine: running count
  langset?: string[];               // refine: the language-tuned charset in use
  manifest?: TextManifest;
  message?: string;
  engine?: "easyocr" | "null";  // which OCR engine actually ran
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

export async function putManifest(assetId: string, manifest: TextManifest): Promise<{ ok: boolean; total_regions: number }> {
  return json(
    await fetch(`/api/manifest/${assetId}`, {
      method: "PUT",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(manifest),
    })
  );
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
