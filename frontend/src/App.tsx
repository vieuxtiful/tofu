import { lazy, Suspense, useCallback, useEffect, useMemo, useRef, useState } from "react";
import { createPortal } from "react-dom";
import InputAdornment from "@mui/material/InputAdornment";
import TextField from "@mui/material/TextField";
import {
  AlertTriangle, AlignCenter, AlignEndHorizontal, AlignEndVertical, AlignJustify, AlignLeft, AlignRight, AlignStartHorizontal, AlignStartVertical, ArrowLeft, ArrowLeftRight, ArrowUpFromLine, Baseline, Bold, BookmarkCheck, Box, Check, ChevronDown, Circle, CircleDashed, CircleDot, CircleOff, FileImage, FolderOpen, Hexagon, History, Home, Italic, Languages, Loader2,
  Play, Plus, RotateCcw, ScanText, ShieldAlert, Sparkles, SquareStack, Subscript, Superscript, Trash2, Type, Underline, VectorSquare, X, Cpu,
} from "lucide-react";
import {
  BBox, FontFamily, FontOption, FontWeight, ImportResult, InpaintPatch, InstText, LanguageOption, Project, VideoJob,
  RenderResult, RenderStreamEvent, SceneRegion, SemanticSubstitutionPlan, SemanticTextUnit, TextManifest, UploadResponse, ValidationReport, GlossaryStatus,
  addRegion, approveRender, checkDuplicateAsset, deleteProjectAsset, deleteRegion, detectAssetStream, mergeRegions,
  fetchFonts, fetchLanguages, getManifest, getProject, importFile, matchFonts, ocrRegion, putManifest,
  applyRepairCandidate, captureLocalizedBaseline, createInpaintPatch, getLocalizedBaseline, getTreatment, previewCandidateLocalized, refineRegion, renderAsset, renderAssetStream, renderPreview, restoreTreatment, scanAssetLanguage, sha256File, snapshotAsset, undoInpaint,
  semanticSubstitution, semanticRepair, updateProject, uploadAsset, validateAsset, fetchGlossaryStatus, uploadGlossary, deleteGlossary, createVideoJob, getVideoJob, getLatestVideoJob, cancelVideoJob, resumeVideoJob, upgradeVideoJob, putVideoKeyframe, updateVideoTrack, watchVideoJob, renderVideoPreview, exportVideo,
} from "./api";
import VideoWorkspace from "./VideoWorkspace";
import { FcCollapse } from "react-icons/fc";
import { LiaSpellCheckSolid } from "react-icons/lia";
import { RiCheckboxFill } from "react-icons/ri";
import { TbCubePlus, TbPhoto, TbPhotoEdit, TbPhotoScan, TbScanCube, TbCircleDashedPlus, TbCircleDashedMinus } from "react-icons/tb";
import { FaBoxOpen, FaLink, FaUnlink, FaEyeDropper } from "react-icons/fa";
import { FaFileImport } from "react-icons/fa6";
import { PiWarningCircleFill, PiHandGrabbingFill, PiHandGrabbingBold, PiArrowsMergeBold } from "react-icons/pi";
import { MdFontDownload, MdOutlineCompare, MdOutlineFontDownload, MdTipsAndUpdates } from "react-icons/md";
import { BiAbacus, BiSolidErrorCircle } from "react-icons/bi";
import { TiWarning } from "react-icons/ti";
import { HiCubeTransparent } from "react-icons/hi2";
import { HiLockClosed, HiLockOpen } from "react-icons/hi";
import { LuRedo2, LuSquareArrowDown, LuSquareArrowUp, LuUndo2 } from "react-icons/lu";
import { BsArrowDownSquareFill, BsArrowUpSquareFill } from "react-icons/bs";
import { VscDebugRestart } from "react-icons/vsc";
import { GiCoolSpices } from "react-icons/gi";
import { GrSelect } from "react-icons/gr";
import { providerLabel } from "./repairLabels";
import type { LocalizedCandidatePreview, RepairCandidate, RepairReview } from "./localizedCanvasTypes";
import SmartFillReview from "./SmartFillReview";
import { langDisplayName, langFlag, LANGUAGE_REGIONS, REGION_ORDER } from "./languageData";
import LanguageCombobox from "./LanguageCombobox";
import FontCombobox, { loadFontPreview, fontNameForPath, weightLabel } from "./FontCombobox";
const FontManager = lazy(() => import("./FontManager"));
import { RecentFont, mergeFamily, pushRecent, readRecent } from "./fontCatalog";
import { fontIdentity } from "./doppelganger";
import HexColorInput from "./HexColorInput";
import RotationDial from "./RotationDial";
import GarnishSliderField from "./GarnishSlider";
import BBoxCanvas from "./BBoxCanvas";
import TargetPreviewCanvas from "./TargetPreviewCanvas";
import PerspectiveHandles from "./PerspectiveHandles";
import {
  IDENTITY_QUAD, Quad, denormaliseQuad, isIdentityQuad, isUsableQuad,
  normaliseQuad, parseQuad, quadFromPolygon,
} from "./perspective";
import RegionTable from "./RegionTable";
const SemanticSubstitutionPanel = lazy(() => import("./SemanticSubstitutionPanel"));
import { attestedFromManifest } from "./targetGuard";
import ExportPanel from "./ExportPanel";
import ProjectGate from "./ProjectGate";
import HistoryPanel from "./HistoryPanel";
import MemoryPanel from "./MemoryPanel";
import SplashScreen from "./SplashScreen";
import TitleScreen from "./TitleScreen";
import ThemeToggle from "./ThemeToggle";
import BBoxColorDropdown from "./BBoxColorDropdown";
import { FlipButton, PenumbraSwitch, PressButton, ThemeSwitch } from "./Buttons";
import { SquareLoader } from "./Loaders";
import ToastSystem, { useToasts, useNotifications, type ToastType, type ToastAction } from "./ToastSystem";
import NotificationBell from "./NotificationBell";
import Stepper, { Step } from "./Stepper";
import { logoSrc, useTheme } from "./theme";
const SystemCapabilitiesPanel = lazy(() => import("./SystemCapabilitiesPanel"));

const PROJECT_KEY = "tofu.projectId";

type Screen = "splash" | "title" | "pantry" | "main";

function hasActivePerspective(inst: InstText | null | undefined): boolean {
  if (!inst) return false;
  const quad = parseQuad(inst.style_profile?.transform?.quad);
  return Boolean(
    quad
    && !isIdentityQuad(quad)
    && isUsableQuad(denormaliseQuad(quad, inst.bounding_box))
  );
}

interface ScanState {
  assetId: string;
  status: "scanning" | "passed" | "mismatch";
  detected?: string | null;
  projectSrc?: string | null;
}

type BrushStroke = {
  id: string;
  points: [number, number][];
};

type PreviewCause = "garnish" | "text" | "style" | "treatment" | "initial";
type PreviewTransaction = {
  phase: "idle" | "debouncing" | "rendering" | "failed";
  requestId: number;
  cause: PreviewCause;
  error?: string;
};

type LocalizedSnapshot = {
  manifest: InstText[];
  patchIds: string[];
};

const DEFAULT_GARNISH_PROFILE = {
  edge_blur_px: 0, edge_smoothing: false, edge_smoothing_strength: 0, erosion_px: 0, dilation_px: 0, grain_strength: 0,
  gamma_shift: 1, smudge_strength: 0, smudge_angle_deg: 0, source_confidence: 1,
};

const WARP_PRESETS: Array<{ value: string; label: string }> = [
  { value: "none", label: "None" },
  { value: "arc", label: "Arc" },
  { value: "arc_lower", label: "Arc Lower" },
  { value: "arc_upper", label: "Arc Upper" },
  { value: "arch", label: "Arch" },
  { value: "bulge", label: "Bulge" },
  { value: "shell_lower", label: "Shell Lower" },
  { value: "shell_upper", label: "Shell Upper" },
  { value: "flag", label: "Flag" },
  { value: "wave", label: "Wave" },
  { value: "fish", label: "Fish" },
  { value: "rise", label: "Rise" },
  { value: "fisheye", label: "Fisheye" },
  { value: "inflate", label: "Inflate" },
  { value: "squeeze", label: "Squeeze" },
  { value: "twist", label: "Twist" },
  { value: "custom", label: "Custom" },
];

const WARP_PATHS: Record<string, string> = {
  arc: "M0,18 Q25,8 50,18",
  arc_lower: "M0,12 Q25,22 50,12",
  arc_upper: "M0,20 Q25,4 50,20",
  arch: "M0,18 Q25,1 50,18",
  bulge: "M0,16 Q25,9 50,16",
  shell_lower: "M0,14 Q25,21 50,14",
  shell_upper: "M0,17 Q25,7 50,17",
  flag: "M0,15 Q8,9 16,15 Q24,21 32,15 Q40,9 50,15",
  wave: "M0,15 Q8,7 16,15 Q24,23 32,15 Q40,7 50,15",
  fish: "M0,18 Q25,11 50,18",
  rise: "M0,21 L50,7",
  fisheye: "M0,18 Q25,5 50,18",
  inflate: "M0,17 Q25,8 50,17",
  squeeze: "M0,13 Q25,19 50,13",
  twist: "M0,15 Q8,5 16,15 Q24,25 32,15 Q40,5 50,15",
};

function WarpPreview({ preset }: { preset: string }) {
  const d = WARP_PATHS[preset];
  if (!d) return null;
  const id = `warp-prev-${preset}`;
  return (
    <svg width="44" height="20" viewBox="0 0 50 24" className="shrink-0 overflow-visible">
      <defs><path id={id} d={d} fill="none" /></defs>
      <text fontSize="8.5" fill="currentColor" className="text-zinc-400 dark:text-zinc-500">
        <textPath href={`#${id}`} startOffset="50%" textAnchor="middle">sample</textPath>
      </text>
    </svg>
  );
}

function Badge({ ok, children }: { ok: boolean; children: React.ReactNode }) {
  return (
    <span
      className={`inline-flex items-center gap-1 rounded-full px-2 py-0.5 text-xs font-medium ${
        ok ? "bg-[#0f2600]/15 text-[#0f2600] dark:text-[#4f9f00]" : "bg-red-500/15 text-red-600 dark:text-red-400"
      }`}
    >
      {ok ? <Check size={12} /> : <X size={12} />}
      {children}
    </span>
  );
}

function Section({ title, icon, children, className = "", rightSideHandle, bottomHandle, sectionStyle, localized, flat, headerExtra }: { title: string; icon?: React.ReactNode; children: React.ReactNode; className?: string; rightSideHandle?: React.ReactNode; bottomHandle?: React.ReactNode; sectionStyle?: React.CSSProperties; localized?: boolean; flat?: boolean; headerExtra?: React.ReactNode }) {
  return (
    <section data-render-localized={localized ? "true" : undefined} className={flat ? `relative ${className}` : `bezier-card soft-shadow relative rounded-xl bg-white/60 p-5 dark:bg-zinc-900/60 ${className}`} style={sectionStyle}>
      <h2 className="subtext mb-3 flex items-center gap-2 text-sm font-semibold uppercase tracking-wider text-zinc-500 dark:text-zinc-400">
        {icon} {title}{headerExtra}
      </h2>
      {children}
      {rightSideHandle}
      {bottomHandle}
    </section>
  );
}

function NumberField({ label, value, onChange }: { label: string; value: number | null | undefined; onChange: (v: number | null) => void }) {
  return (
    <div>
      <label className="subtext mb-1 block text-xs text-zinc-500">{label}</label>
      <input
        type="number"
        step={0.5}
        value={value ?? ""}
        onChange={(e) => onChange(e.target.value ? Number(e.target.value) : null)}
        placeholder="—"
        className="w-full rounded-sm border border-zinc-300 bg-white px-2 py-1 text-xs text-zinc-800 dark:border-zinc-700 dark:bg-zinc-900 dark:text-zinc-200"
      />
    </div>
  );
}

function PixelField({ label, value, onChange, min, step = 0.5, placeholder, width = "7rem" }: { label: string; value: number | null | undefined; onChange: (value: number | null) => void; min?: number; step?: number; placeholder?: string; width?: string }) {
  return <TextField label={label} type="number" size="small" variant="filled" value={value ?? ""} onChange={(event) => onChange(event.target.value === "" ? null : Number(event.target.value))} placeholder={placeholder} slotProps={{ input: { startAdornment: <InputAdornment position="start" disableTypography><span className="text-zinc-400 dark:text-zinc-500">px</span></InputAdornment> }, htmlInput: { min, step } }} sx={{ width, maxWidth: "100%", flexShrink: 0, "& .MuiInputBase-input": { minWidth: 0, color: "#a1a1aa !important" }, "& .MuiInputLabel-root": { color: "#a1a1aa !important" }, "& .MuiFilledInput-root": { backgroundColor: "transparent" }, "& .MuiFilledInput-root::before": { borderColor: "rgba(113,113,122,0.3)" }, "& .MuiFilledInput-root::after": { borderColor: "#22d3ee" }, "& .MuiFilledInput-root:hover::before": { borderColor: "rgba(113,113,122,0.5)" } }} />;
}

/** total run time of the bbox arrival shimmer. Must stay in step with
 * `animation: bbox-shimmer 0.5s ... 2` in bbox.css — two passes, one second. */
const SHIMMER_MS = 1000;

const rangePulseTimers = new WeakMap<HTMLInputElement, number>();

function pulseRangeStep(input: HTMLInputElement) {
  const prior = rangePulseTimers.get(input);
  if (prior !== undefined) window.clearTimeout(prior);
  input.classList.remove("text-warp-step-pulse");
  void input.offsetWidth;
  input.classList.add("text-warp-step-pulse");
  rangePulseTimers.set(input, window.setTimeout(() => {
    input.classList.remove("text-warp-step-pulse");
    rangePulseTimers.delete(input);
  }, 150));
}

function snapRangePointer(event: React.PointerEvent<HTMLInputElement>, onValue: (value: number) => void) {
  if (event.button !== 0 || event.currentTarget.disabled) return;
  const input = event.currentTarget;
  const rect = input.getBoundingClientRect();
  const min = Number(input.min);
  const max = Number(input.max);
  const step = Number(input.step) || 1;
  if (!rect.width || !Number.isFinite(min) || !Number.isFinite(max) || max <= min) return;
  const ratio = Math.max(0, Math.min(1, (event.clientX - rect.left) / rect.width));
  const steps = Math.round(((min + ratio * (max - min)) - min) / step);
  const value = Math.max(min, Math.min(max, Number((min + steps * step).toFixed(8))));
  input.value = String(value);
  pulseRangeStep(input);
  onValue(value);
}

/** Merge a style patch onto a region, preserving every field the patch does
 * not mention.
 *
 * Never rebuild `style_profile` from a hand-written field list. StyleProfil
 * gains properties over time and nothing makes such a list follow, so the
 * omitted ones are silently discarded — which is how the Translate tab's
 * orientation, word-order and font controls came to erase the shadow,
 * effects and text-warp treatments set in Render's Text & Appearance panel. */
function withStyle(
  inst: InstText,
  patch: Partial<NonNullable<InstText["style_profile"]>>,
): InstText {
  return {
    ...inst,
    style_profile: { ...(inst.style_profile ?? {}), ...patch } as NonNullable<InstText["style_profile"]>,
  };
}

function TitleConfirmOverlay({ onYes, onNo }: { onYes: () => void; onNo: () => void }) {
  const [leaving, setLeaving] = useState(false);

  const handleNo = () => {
    setLeaving(true);
    setTimeout(onNo, 300);
  };

  const handleYes = () => {
    setLeaving(true);
    setTimeout(onYes, 300);
  };

  return (
    <div
      className={`title-confirm-backdrop${leaving ? " leaving" : ""}`}
      onClick={handleNo}
    >
      <div className="title-confirm-card" onClick={(e) => e.stopPropagation()}>
        <p className="title-confirm-message">return to title?</p>
        <div className="title-confirm-actions">
          <button className="title-confirm-btn yes" onClick={handleYes}>Yes</button>
          <button className="title-confirm-btn no" onClick={handleNo}>No</button>
        </div>
      </div>
    </div>
  );
}

function AssetDeleteConfirmOverlay({ filename, onYes, onNo }: { filename: string; onYes: () => void; onNo: () => void }) {
  const [leaving, setLeaving] = useState(false);

  const handleNo = () => {
    setLeaving(true);
    setTimeout(onNo, 300);
  };

  const handleYes = () => {
    setLeaving(true);
    setTimeout(onYes, 300);
  };

  return (
    <div
      className={`title-confirm-backdrop${leaving ? " leaving" : ""}`}
      onClick={handleNo}
    >
      <div className="title-confirm-card" onClick={(e) => e.stopPropagation()}>
        <p className="title-confirm-message">remove this asset?</p>
        <p className="subtext mt-1 text-center text-xs text-zinc-500">Snapshots captured—nothing is permanently lost.</p>
        <div className="title-confirm-actions">
          <button className="title-confirm-btn yes" onClick={handleYes}>Yes</button>
          <button className="title-confirm-btn no" onClick={handleNo}>No</button>
        </div>
      </div>
    </div>
  );
}

function UnsavedChangesOverlay({ onYes, onNo }: { onYes: () => void; onNo: () => void }) {
  const [leaving, setLeaving] = useState(false);

  const handleNo = () => {
    setLeaving(true);
    setTimeout(onNo, 300);
  };

  const handleYes = () => {
    setLeaving(true);
    setTimeout(onYes, 300);
  };

  return (
    <div
      className={`title-confirm-backdrop${leaving ? " leaving" : ""}`}
      onClick={handleNo}
    >
      <div className="title-confirm-card" onClick={(e) => e.stopPropagation()}>
        <p className="title-confirm-message">proceed with unsaved changes?</p>
        <div className="title-confirm-actions">
          <button className="title-confirm-btn yes" onClick={handleYes}>Yes</button>
          <button className="title-confirm-btn no" onClick={handleNo}>No</button>
        </div>
      </div>
    </div>
  );
}

const SCREEN_KEY = "tofu.screen";

export default function App() {
  const { theme, toggle: toggleTheme } = useTheme();
  // on reload, restore the saved screen — but always play splash for title
  const [screen, setScreen] = useState<Screen>(() => {
    const saved = sessionStorage.getItem(SCREEN_KEY);
    if (saved === "main" || saved === "pantry") return saved as Screen;
    return "splash";
  });
  const [displayedScreen, setDisplayedScreen] = useState<Screen>(() => {
    const saved = sessionStorage.getItem(SCREEN_KEY);
    if (saved === "main" || saved === "pantry") return saved as Screen;
    return "splash";
  });
  const [leaving, setLeaving] = useState(false);
  const [pantryLeaving, setPantryLeaving] = useState(false);
  const [pantryMode, setPantryMode] = useState<"full" | "pantry" | "create">("full");
  const prevScreen = useRef<Screen>("title");
  const transitionTimer = useRef<ReturnType<typeof setTimeout> | null>(null);
  const [pendingBackNav, setPendingBackNav] = useState(false);
  const navGuardRef = useRef(false);

  // persist current screen so reload restores it (except title always gets splash)
  useEffect(() => {
    if (screen === "splash") return;
    sessionStorage.setItem(SCREEN_KEY, screen);
  }, [screen]);

  const transitionTo = useCallback((next: Screen) => {
    if (next === displayedScreen) return;
    if (transitionTimer.current) clearTimeout(transitionTimer.current);
    setLeaving(true);
    transitionTimer.current = setTimeout(() => {
      setDisplayedScreen(next);
      setLeaving(false);
    }, 150);
  }, [displayedScreen]);

  const handleSplashDone = useCallback(() => setScreen("title"), []);

  useEffect(() => {
    if (screen !== displayedScreen) transitionTo(screen);
  }, [screen, displayedScreen, transitionTo]);

  const [step, setStep] = useState<Step>(0);
  const [stackAnimDisabled, setStackAnimDisabled] = useState<boolean>(
    () => localStorage.getItem("tofu.stackAnimDisabled") === "true"
  );
  const [stackReveal, setStackReveal] = useState(0);
  const [stackRevealStep, setStackRevealStep] = useState<Step>(0);
  const stackTimers = useRef<ReturnType<typeof setTimeout>[]>([]);
  const [asset, setAsset] = useState<UploadResponse | null>(null);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const [videoJob, setVideoJob] = useState<VideoJob | null>(null);
  const [preRenderUrl, setPreRenderUrl] = useState<string | null>(null);
  const [previewRenderError, setPreviewRenderError] = useState<string | null>(null);
  const [previewCacheKey, setPreviewCacheKey] = useState<string | null>(null);
  const [previewTransaction, setPreviewTransaction] = useState<PreviewTransaction>({ phase: "idle", requestId: 0, cause: "initial" });
  const [previewRetryRevision, setPreviewRetryRevision] = useState(0);
  const [repairReviews, setRepairReviews] = useState<RepairReview[]>([]);
  const [smartFillHoverId, setSmartFillHoverId] = useState<string | null>(null);
  const [localizedCandidatePreviews, setLocalizedCandidatePreviews] = useState<Record<string, LocalizedCandidatePreview>>({});
  const [candidatePreviewRevision, setCandidatePreviewRevision] = useState(0);
  const [repairFallbackIds, setRepairFallbackIds] = useState<string[]>([]);
  const previewRenderSeq = useRef(0);
  const previewCauseRef = useRef<PreviewCause>("initial");
  // Keep the exact post-edit snapshot that requested a preview.  This avoids
  // a garnish slider update being replaced by unrelated state work before the
  // debounced renderer obtains its manifest.
  const queuedPreviewManifest = useRef<TextManifest | null>(null);
  const candidatePreviewSeq = useRef(0);
  const previewSyncing = previewTransaction.phase === "rendering";
  const previewPending = previewTransaction.phase === "debouncing" || previewTransaction.phase === "rendering";
  const garnishPreviewSyncing = previewSyncing && previewTransaction.cause === "garnish";
  const [lassoMode, setLassoMode] = useState(false);
  const [lassoPoints, setLassoPoints] = useState<[number, number][]>([]);
  const [nearFirstPoint, setNearFirstPoint] = useState(false);
  const [brushMode, setBrushMode] = useState(false);
  const [brushStrokes, setBrushStrokes] = useState<BrushStroke[]>([]);
  const [activeBrushStroke, setActiveBrushStroke] = useState<BrushStroke | null>(null);
  const [brushCursor, setBrushCursor] = useState<[number, number] | null>(null);
  const [brushRadius, setBrushRadius] = useState(18);
  const [brushIntensity, setBrushIntensity] = useState(50);
  const [localizedZoom, setLocalizedZoom] = useState(1);
  const [localizedDragMode, setLocalizedDragMode] = useState(false);
  const [perspectiveMode, setPerspectiveMode] = useState(false);
  const [localizedDisplayScale, setLocalizedDisplayScale] = useState(1);
  const localizedPanRef = useRef<{ startX: number; startY: number; scrollLeft: number; scrollTop: number } | null>(null);
  const localizedScrollRef = useRef<HTMLDivElement>(null);
  const localizedImageRef = useRef<HTMLImageElement>(null);
  const quadGestureManifest = useRef<InstText[] | null>(null);
  const brushDrawing = useRef<{ pointerId: number; stroke: BrushStroke } | null>(null);
  const [brushApplying, setBrushApplying] = useState(false);
  const [inpaintPatchIds, setInpaintPatchIds] = useState<string[]>([]);
  const [appliedCandidateIds, setAppliedCandidateIds] = useState<string[]>([]);
  const [patchRevision, setPatchRevision] = useState(0);
  const localizedUndoStack = useRef<LocalizedSnapshot[]>([]);
  const localizedRedoStack = useRef<LocalizedSnapshot[]>([]);
  const localizedBaseline = useRef<LocalizedSnapshot | null>(null);
  const [canLocalizedUndo, setCanLocalizedUndo] = useState(false);
  const [canLocalizedRedo, setCanLocalizedRedo] = useState(false);
  // A shared canvas-picking mode: either canvas can supply the sampled colour.
  // A named sentinel prevents the old boolean/null mismatch at reset time.
  const [colorPickMode, setColorPickMode] = useState<"active" | null>(null);
  const [languages, setLanguages] = useState<LanguageOption[]>([]);
  const [targLang, setTargLang] = useState("es");
  const [formerTargLang, setFormerTargLang] = useState<string | null>(null);
  const [showLangChangePopup, setShowLangChangePopup] = useState(false);
  const [fontsByLang, setFontsByLang] = useState<Record<string, FontOption[]>>({});
  const [familiesByLang, setFamiliesByLang] = useState<Record<string, FontFamily[]>>({});
  const [script, setScript] = useState<string>("");
  const fontFetches = useRef<Set<string>>(new Set());
  // The Font Manager's complete catalog, kept apart from familiesByLang:
  // that one stays the ranked 24 that feeds the dropdown, the table and the
  // preflight cards. Fetched lazily per language the first time the panel
  // opens, because it is the whole installed library.
  const [fullFamiliesByLang, setFullFamiliesByLang] = useState<Record<string, FontFamily[]>>({});
  const fullFontFetches = useRef<Set<string>>(new Set());
  const [fullFontsLoading, setFullFontsLoading] = useState(false);
  const [showFontManager, setShowFontManager] = useState(false);
  const [recentFonts, setRecentFonts] = useState<RecentFont[]>(() => readRecent(localStorage));
  const [report, setReport] = useState<ValidationReport | null>(null);
  const [renderResult, setRenderResult] = useState<RenderResult | null>(null);
  const [renderSelId, setRenderSelId] = useState<string | null>(null);
  const [prevSelId, setPrevSelId] = useState<string | null>(null);

  // A Smart Fill card can disappear while its preview refreshes.  Selection
  // changes are therefore a second, independent stale-hover backstop.
  useEffect(() => { setSmartFillHoverId(null); }, [renderSelId]);
  const [styleCollapsed, setStyleCollapsed] = useState(false);
  const [canvasExpandedH, setCanvasExpandedH] = useState(false);
  const [canvasCardOrder, setCanvasCardOrder] = useState<"source-first" | "localized-first">("localized-first");
  const sourceCanvasFirst = canvasCardOrder === "source-first";
  const canSwapCanvasCards = Boolean(previewUrl);
  const [styleExpandedH, setStyleExpandedH] = useState(false);
  // vertical expand for Text & Appearance card (half-width only)
  const [styleExpandedV, setStyleExpandedV] = useState(false);
  const [styleCardH, setStyleCardH] = useState<number | null>(null);
  const styleCardRef = useRef<HTMLDivElement | null>(null);
  const styleDragStartY = useRef(0);
  const styleDragStartH = useRef(0);
  const preExpandStyleH = useRef<number | null>(null);
  const styleExpandTimer = useRef<ReturnType<typeof setTimeout> | null>(null);
  const onStyleExpandClick = useCallback(() => {
    if (styleExpandTimer.current) {
      clearTimeout(styleExpandTimer.current);
      styleExpandTimer.current = null;
      setStyleExpandedH((v) => {
        const nv = !v;
        if (nv) { setStyleCardH(null); setStyleExpandedV(false); }
        return nv;
      });
      return;
    }
    styleExpandTimer.current = setTimeout(() => { styleExpandTimer.current = null; }, 250);
  }, []);
  const onStyleDragStart = useCallback((e: React.MouseEvent) => {
    e.stopPropagation();
    e.preventDefault();
    styleDragStartY.current = e.clientY;
    styleDragStartH.current = styleCardRef.current?.offsetHeight ?? 400;
    // compute max height: localized card bottom - style card top - gap
    const styleEl = styleCardRef.current;
    const localizedEl = document.querySelector('[data-render-localized]');
    let maxH = 9999;
    if (styleEl && localizedEl) {
      const styleRect = styleEl.getBoundingClientRect();
      const localizedRect = localizedEl.getBoundingClientRect();
      maxH = localizedRect.bottom - styleRect.top - 16;
    }
    const onMove = (ev: MouseEvent) => {
      const delta = ev.clientY - styleDragStartY.current;
      const newH = Math.max(200, Math.min(maxH, styleDragStartH.current + delta));
      setStyleExpandedV(false);
      preExpandStyleH.current = null;
      setStyleCardH(newH);
    };
    const onUp = () => {
      document.removeEventListener("mousemove", onMove);
      document.removeEventListener("mouseup", onUp);
    };
    document.addEventListener("mousemove", onMove);
    document.addEventListener("mouseup", onUp);
  }, []);
  const onStyleDoubleClick = useCallback(() => {
    if (styleExpandedV) {
      // collapse back to standard
      setStyleExpandedV(false);
      setStyleCardH(null);
    } else {
      // expand to max: align with localized asset card bottom
      const styleEl = styleCardRef.current;
      if (!styleEl) return;
      const localizedEl = document.querySelector('[data-render-localized]');
      if (!localizedEl) return;
      const styleRect = styleEl.getBoundingClientRect();
      const localizedRect = localizedEl.getBoundingClientRect();
      const gap = 16; // space-y-4 gap
      const maxH = localizedRect.bottom - styleRect.top - gap;
      preExpandStyleH.current = styleEl.offsetHeight;
      setStyleCardH(Math.max(200, maxH));
      setStyleExpandedV(true);
    }
  }, [styleExpandedV]);

  // Garnish card expansion state (mirrors Text & Appearance card)
  const [garnishCardCollapsed, setGarnishCardCollapsed] = useState(false);
  const [garnishCardExpandedH, setGarnishCardExpandedH] = useState(false);
  const [garnishCardExpandedV, setGarnishCardExpandedV] = useState(false);
  const [garnishCardH, setGarnishCardH] = useState<number | null>(null);
  const garnishCardRef = useRef<HTMLDivElement | null>(null);
  const garnishDragStartY = useRef(0);
  const garnishDragStartH = useRef(0);
  const preExpandGarnishH = useRef<number | null>(null);
  const garnishExpandTimer = useRef<ReturnType<typeof setTimeout> | null>(null);
  const onGarnishExpandClick = useCallback(() => {
    if (garnishExpandTimer.current) {
      clearTimeout(garnishExpandTimer.current);
      garnishExpandTimer.current = null;
      setGarnishCardExpandedH((v) => {
        const nv = !v;
        if (nv) { setGarnishCardH(null); setGarnishCardExpandedV(false); }
        return nv;
      });
      return;
    }
    garnishExpandTimer.current = setTimeout(() => { garnishExpandTimer.current = null; }, 250);
  }, []);
  const onGarnishDragStart = useCallback((e: React.MouseEvent) => {
    e.stopPropagation();
    e.preventDefault();
    garnishDragStartY.current = e.clientY;
    garnishDragStartH.current = garnishCardRef.current?.offsetHeight ?? 400;
    const garnishEl = garnishCardRef.current;
    const localizedEl = document.querySelector('[data-render-localized]');
    let maxH = 9999;
    if (garnishEl && localizedEl) {
      const garnishRect = garnishEl.getBoundingClientRect();
      const localizedRect = localizedEl.getBoundingClientRect();
      maxH = localizedRect.bottom - garnishRect.top - 16;
    }
    const onMove = (ev: MouseEvent) => {
      const delta = ev.clientY - garnishDragStartY.current;
      const newH = Math.max(200, Math.min(maxH, garnishDragStartH.current + delta));
      setGarnishCardExpandedV(false);
      preExpandGarnishH.current = null;
      setGarnishCardH(newH);
    };
    const onUp = () => {
      document.removeEventListener("mousemove", onMove);
      document.removeEventListener("mouseup", onUp);
    };
    document.addEventListener("mousemove", onMove);
    document.addEventListener("mouseup", onUp);
  }, []);
  const onGarnishDoubleClick = useCallback(() => {
    if (garnishCardExpandedV) {
      setGarnishCardExpandedV(false);
      setGarnishCardH(null);
    } else {
      const garnishEl = garnishCardRef.current;
      if (!garnishEl) return;
      const localizedEl = document.querySelector('[data-render-localized]');
      if (!localizedEl) return;
      const garnishRect = garnishEl.getBoundingClientRect();
      const localizedRect = localizedEl.getBoundingClientRect();
      const maxH = localizedRect.bottom - garnishRect.top - 16;
      preExpandGarnishH.current = garnishEl.offsetHeight;
      setGarnishCardH(Math.max(200, maxH));
      setGarnishCardExpandedV(true);
    }
  }, [garnishCardExpandedV]);

  // reset render-tab expansion state when leaving step 3
  useEffect(() => {
    if (step !== 3) { setStyleCardH(null); setStyleExpandedV(false); setStyleExpandedH(false); setGarnishCardH(null); setGarnishCardExpandedV(false); setGarnishCardExpandedH(false); }
  }, [step]);

  // stacking animation: reset reveal counter on step change, then increment
  useEffect(() => {
    stackTimers.current.forEach(clearTimeout);
    stackTimers.current = [];
    setStackReveal(0);
    setStackRevealStep(step);
    if (stackAnimDisabled) return;
    // reveal cards one by one at 100ms intervals (up to 8 cards max)
    for (let i = 1; i <= 8; i++) {
      stackTimers.current.push(setTimeout(() => setStackReveal(i), i * 100));
    }
    return () => { stackTimers.current.forEach(clearTimeout); };
  }, [step, stackAnimDisabled]);

  // persist stacking animation preference
  useEffect(() => {
    localStorage.setItem("tofu.stackAnimDisabled", String(stackAnimDisabled));
  }, [stackAnimDisabled]);

  // helper: returns class for a card at given stack index
  const stackClass = useCallback((index: number): string => {
    if (stackAnimDisabled) return "";
    if (stackRevealStep !== step) return "stack-hidden";
    return index < stackReveal ? "stack-in" : "stack-hidden";
  }, [stackAnimDisabled, stackReveal, stackRevealStep, step]);

  // --- image size (shared across steps) ---
  const [imgSize, setImgSize] = useState<{ width: number; height: number } | null>(null);

  // --- linked canvas state (translate step) ---
  const [canvasesLinked, setCanvasesLinked] = useState(true);
  const [sharedZoom, setSharedZoom] = useState(1);
  const [sharedScroll, setSharedScroll] = useState({ x: 0, y: 0 });
  // shared canvas height: null = standard, number = expanded (px)
  const STANDARD_CANVAS_H = 300;
  const [sharedCanvasH, setSharedCanvasH] = useState<number | null>(null);
  const lastExpandedH = useRef<number | null>(null);
  const computeExpandedHeight = useCallback(() => {
    // max height before images truncate: viewport minus header, stepper, toolbar, padding
    const reserved = 96 /* logo */ + 60 /* stepper */ + 50 /* toolbar */ + 64 /* padding */ + 80 /* link button + gaps */;
    const viewportMax = Math.max(STANDARD_CANVAS_H, window.innerHeight - reserved);
    return Math.min(800, viewportMax); // cap at MAX_CANVAS_H
  }, []);
  const toggleLinkedCanvasHeight = useCallback(() => {
    setSharedCanvasH((h) => {
      if (h !== null) {
        lastExpandedH.current = h; // remember height before collapsing
        return null; // collapse to standard
      }
      // restore last expanded height, or compute a default
      return lastExpandedH.current ?? computeExpandedHeight();
    });
  }, [computeExpandedHeight]);

  // --- verify step (QA inspector) ---
  const [verifyBusy, setVerifyBusy] = useState<string | null>(null); // stage label while streaming
  const [verifyStage, setVerifyStage] = useState<string | null>(null);
  const [verifySelId, setVerifySelId] = useState<string | null>(null);
  const [renderLogsOpen, setRenderLogsOpen] = useState(false);
  const [recommendationsOpen, setRecommendationsOpen] = useState(false);
  const [recommendationsSeen, setRecommendationsSeen] = useState(false);
  const [verifyCardOrder, setVerifyCardOrder] = useState<"compare-first" | "qa-first">("compare-first");
  const [reRenderingId, setReRenderingId] = useState<string | null>(null);
  const [approved, setApproved] = useState(false);
  const cancelVerifyRef = useRef<(() => void) | null>(null);

  // auto-expand the style panel when a region is selected
  useEffect(() => {
    if (renderSelId) setStyleCollapsed(false);
  }, [renderSelId]);

  // keep prevSelId alive during collapse so content doesn't vanish before animation
  useEffect(() => {
    if (renderSelId) {
      setPrevSelId(renderSelId);
    } else {
      const t = setTimeout(() => setPrevSelId(null), 400);
      return () => clearTimeout(t);
    }
  }, [renderSelId]);
  const [busy, setBusy] = useState<string | null>(null);
  const [fontMatchingId, setFontMatchingId] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const cancelDetectRef = useRef<(() => void) | null>(null);

  const [manifest, setManifestRaw] = useState<InstText[]>([]);
  // undo/redo history for manifest edits
  const manifestUndoStack = useRef<InstText[][]>([]);
  const manifestRedoStack = useRef<InstText[][]>([]);
  const manifestSkipHistory = useRef(false);
  // Batch undo: during a drag or text-typing session, suppress per-pixel /
  // per-keystroke undo entries and push a single pre-interaction snapshot
  // when the batch ends.  One undo then returns to the state before the
  // interaction started, not to each intermediate position.
  const manifestBatching = useRef(false);
  const manifestBatchAnchor = useRef<InstText[] | null>(null);
  const [canUndo, setCanUndo] = useState(false);
  const [canRedo, setCanRedo] = useState(false);
  const syncUndoRedo = useCallback(() => {
    setCanUndo(manifestUndoStack.current.length > 0);
    setCanRedo(manifestRedoStack.current.length > 0);
  }, []);
  const setManifest = useCallback((updater: InstText[] | ((prev: InstText[]) => InstText[])) => {
    setManifestRaw((prev) => {
      const next = typeof updater === "function" ? (updater as (p: InstText[]) => InstText[])(prev) : updater;
      if (!manifestSkipHistory.current && !manifestBatching.current) {
        manifestUndoStack.current.push(prev);
        if (manifestUndoStack.current.length > 50) manifestUndoStack.current.shift();
        manifestRedoStack.current = [];
      }
      manifestSkipHistory.current = false;
      syncUndoRedo();
      return next;
    });
  }, [syncUndoRedo]);
  const beginManifestBatch = useCallback(() => {
    if (manifestBatching.current) return;
    manifestBatchAnchor.current = manifest;
    manifestBatching.current = true;
  }, [manifest]);
  const endManifestBatch = useCallback(() => {
    if (!manifestBatching.current) return;
    const anchor = manifestBatchAnchor.current;
    if (anchor !== null) {
      manifestUndoStack.current.push(anchor);
      if (manifestUndoStack.current.length > 50) manifestUndoStack.current.shift();
      manifestRedoStack.current = [];
    }
    manifestBatching.current = false;
    manifestBatchAnchor.current = null;
    syncUndoRedo();
  }, [syncUndoRedo]);
  const undoManifest = useCallback(() => {
    setManifestRaw((prev) => {
      const stack = manifestUndoStack.current;
      if (stack.length === 0) return prev;
      const previous = stack.pop()!;
      manifestRedoStack.current.push(prev);
      syncUndoRedo();
      return previous;
    });
  }, [syncUndoRedo]);
  const redoManifest = useCallback(() => {
    setManifestRaw((prev) => {
      const stack = manifestRedoStack.current;
      if (stack.length === 0) return prev;
      const next = stack.pop()!;
      manifestUndoStack.current.push(prev);
      syncUndoRedo();
      return next;
    });
  }, [syncUndoRedo]);
  // excluded regions stay in `manifest` (cleanse still needs to erase them),
  // but the canvas and table must never show a "deleted" region -- everything
  // rendered to the user reads from this filtered view instead of `manifest`
  // directly.
  const [manualOrder, setManualOrder] = useState<string[]>([]);
  // `reading_order` is independent of the detected bounding boxes.  It is
  // the durable order shown in the text manifest and exported to XLIFF.
  const orderedManifest = [...manifest].sort((a, b) => {
    const ai = manualOrder.indexOf(a.id);
    const bi = manualOrder.indexOf(b.id);
    if (ai !== -1 || bi !== -1) {
      if (ai === -1) return 1;
      if (bi === -1) return -1;
      return ai - bi;
    }
    const order = (a.reading_order ?? Number.MAX_SAFE_INTEGER) - (b.reading_order ?? Number.MAX_SAFE_INTEGER);
    return order || manifest.indexOf(a) - manifest.indexOf(b);
  });
  const visibleManifest = orderedManifest.filter((i) => !i.excluded);
  // UI-only dismissal of the per-region OCR review banner.  Does not mutate
  // the manifest's ocr_quality — re-detection or a page refresh restores
  // dismissed regions.  Cleared wholesale whenever a fresh scan completes.
  // Declared here rather than with the other useState calls below because
  // qualityReviewRegions reads it during render.
  const [dismissedOcrReview, setDismissedOcrReview] = useState<Set<string>>(new Set());
  const qualityReviewRegions = visibleManifest.filter((inst) => {
    const state = inst.ocr_quality?.state;
    return (state === "review_required" || state === "unresolvable")
      && !dismissedOcrReview.has(inst.id);
  });
  // Advisory only: usable text resolution is measured from detected crops,
  // never inferred solely from this whole-image dimension.
  const tinyUploadAdvisory = Boolean(imgSize && Math.min(imgSize.width, imgSize.height) < 360);
  const [imgDim, setImgDim] = useState<[number, number] | null>(null);
  useEffect(() => {
    const image = localizedImageRef.current;
    if (!image || !imgDim?.[0]) return;
    const update = () => {
      const width = image.getBoundingClientRect().width;
      if (width > 0) setLocalizedDisplayScale(width / imgDim[0]);
    };
    update();
    const observer = new ResizeObserver(update);
    observer.observe(image);
    return () => observer.disconnect();
  }, [localizedZoom, imgDim, preRenderUrl, previewUrl]);

  useEffect(() => {
    if (localizedDragMode || brushMode || lassoMode || colorPickMode) {
      setPerspectiveMode(false);
    }
  }, [localizedDragMode, brushMode, lassoMode, colorPickMode]);
  const [sceneRegions, setSceneRegions] = useState<SceneRegion[]>([]);
  // Render treatment has one selected-instance model.  The Garnish card and
  // localized canvas consume this same derivation so scope/profile changes
  // cannot disagree between the two surfaces.
  const selectedRenderInst = manifest.find((item) => item.id === renderSelId) ?? null;
  // the region a style edit actually lands on: updateSelectedStyle falls back
  // to prevSelId during the style panel's collapse animation, so anything
  // that reports or previews the target must use the same resolution
  const styleTargetInst = manifest.find((item) => item.id === (renderSelId ?? prevSelId)) ?? null;
  const selectedGarnishSurface = selectedRenderInst ? (() => {
    const b = selectedRenderInst.bounding_box;
    return sceneRegions.find((region) => b.x + b.width / 2 >= region.bbox.x && b.x + b.width / 2 <= region.bbox.x + region.bbox.width && b.y + b.height / 2 >= region.bbox.y && b.y + b.height / 2 <= region.bbox.y + region.bbox.height) ?? null;
  })() : null;
  const selectedGarnishRecommended = selectedGarnishSurface?.garnish_profile;
  const selectedGarnishProfile = selectedRenderInst?.garnish_override ?? selectedGarnishRecommended ?? DEFAULT_GARNISH_PROFILE;
  const selectedGarnishEnabled = selectedRenderInst?.garnish_enabled !== false;
  // Basil owns semantic reading units separately from the immutable region
  // list.  This lets target-language order differ from visual box order
  // without making `rN` identity or geometry mutable in the editor.
  const [semanticUnits, setSemanticUnits] = useState<SemanticTextUnit[]>([]);
  const [semanticDrafts, setSemanticDrafts] = useState<Record<string, string>>({});
  const [semanticPlans, setSemanticPlans] = useState<Record<string, SemanticSubstitutionPlan>>({});
  const [semanticBusyId, setSemanticBusyId] = useState<string | null>(null);
  const [glossaryStatus, setGlossaryStatus] = useState<GlossaryStatus | null>(null);
  const [glossaryUploading, setGlossaryUploading] = useState(false);
  const [glossaryUploadStep, setGlossaryUploadStep] = useState<"marinate" | "ferment" | "set" | "done" | null>(null);
  const [glossaryUploadError, setGlossaryUploadError] = useState<string | null>(null);
  const [srcLang, setSrcLang] = useState<string | null>(null);
  const [selectedId, setSelectedId] = useState<string | null>(null);
  const [drawMode, setDrawMode] = useState(false);
  const [ocrLoading, setOcrLoading] = useState<string | null>(null);
  const [mergeLoading, setMergeLoading] = useState(false);
  const [ocrReviewOpen, setOcrReviewOpen] = useState(false);
  const [saveStatus, setSaveStatus] = useState<"idle" | "saving" | "saved">("idle");
  const [srcLangLocked, setSrcLangLocked] = useState(true);
  const [importedHash, setImportedHash] = useState<string | null>(null);
  const [hasEditsAfterImport, setHasEditsAfterImport] = useState(false);
  const [importLoading, setImportLoading] = useState(false);
  const [dragOverTranslate, setDragOverTranslate] = useState(false);
  // The asset dropzone promised "drop or click" but only click was ever wired:
  // it is a <label> around a display:none input, and a hidden input cannot
  // receive a drop. This drives the drop affordance the copy already claims.
  const [dragOverAsset, setDragOverAsset] = useState(false);
  const saveTimer = useRef<ReturnType<typeof setTimeout> | null>(null);

  // push history state on screen transitions so browser back works
  useEffect(() => {
    if (displayedScreen === "splash") return;
    if (navGuardRef.current) {
      navGuardRef.current = false;
      return;
    }
    window.history.pushState({ screen: displayedScreen }, "");
  }, [displayedScreen]);

  // listen for browser back; if unsaved changes, prompt before navigating
  useEffect(() => {
    const onPopState = () => {
      if (displayedScreen === "main" && (saveStatus === "saving" || hasEditsAfterImport)) {
        setPendingBackNav(true);
      } else {
        navGuardRef.current = true;
        const prev = prevScreen.current;
        if (prev && prev !== displayedScreen) {
          setScreen(prev);
        } else {
          setScreen("title");
        }
      }
    };
    window.addEventListener("popstate", onPopState);
    return () => window.removeEventListener("popstate", onPopState);
  }, [displayedScreen, saveStatus, hasEditsAfterImport]);

  const [hoveredId, setHoveredId] = useState<string | null>(null);
  const [nameTyped, setNameTyped] = useState(0);
  const [nameDone, setNameDone] = useState(false);
  const [showCapturePrompt, setShowCapturePrompt] = useState(false);
  const [showLangConfirm, setShowLangConfirm] = useState(false);
  const [showLangSelect, setShowLangSelect] = useState(false);
  const [langRegionFilter, setLangRegionFilter] = useState<string | null>(null);
  const [detectProgress, setDetectProgress] = useState<string | null>(null);
  const [detectStage, setDetectStage] = useState<"idle" | "scanning" | "lang" | "drawing" | "done">("idle");
  const [lockedLangs, setLockedLangs] = useState<Set<string>>(new Set());

  // project-first session management
  const [project, setProject] = useState<Project | null>(null);
  const [showHistory, setShowHistory] = useState(false);
  const [showCapabilities, setShowCapabilities] = useState(false);
  const [historyLeaving, setHistoryLeaving] = useState(false);
  const [showMemory, setShowMemory] = useState(false);
  const [memoryLeaving, setMemoryLeaving] = useState(false);
  const [showSettings, setShowSettings] = useState(false);
  const [bboxColor, setBboxColor] = useState<string>(
    () => localStorage.getItem("tofu.bboxColor") || "#22d3ee"
  );
  useEffect(() => {
    localStorage.setItem("tofu.bboxColor", bboxColor);
  }, [bboxColor]);
  // off by default: a canvas of boxes that all pulse forever is a setting
  // some people want and most do not
  const [bboxBlink, setBboxBlink] = useState(
    () => localStorage.getItem("tofu.bboxBlink") === "true"
  );
  useEffect(() => {
    localStorage.setItem("tofu.bboxBlink", String(bboxBlink));
  }, [bboxBlink]);
  const [hideRegionCounter, setHideRegionCounter] = useState(
    () => localStorage.getItem("tofu.hideRegionCounter") === "true"
  );
  useEffect(() => {
    localStorage.setItem("tofu.hideRegionCounter", String(hideRegionCounter));
  }, [hideRegionCounter]);

  // --- arrival shimmer ---------------------------------------------------
  // Regions the SERVER just handed us get a one-off two-pass shimmer, so
  // "the machine found these" is visually distinct from "these were already
  // here".  Membership is deliberately transient: the ids are dropped on a
  // timer once the animation is over, which is what keeps a drag, a text
  // edit, a reorder or a tab change from replaying it.
  const [newRegionIds, setNewRegionIds] = useState<Set<string> | null>(null);
  const shimmerTimer = useRef<number | null>(null);
  const markRegionsNew = useCallback((instances: InstText[]) => {
    if (shimmerTimer.current !== null) window.clearTimeout(shimmerTimer.current);
    setNewRegionIds(new Set(instances.map((i) => i.id)));
    shimmerTimer.current = window.setTimeout(() => {
      setNewRegionIds(null);
      shimmerTimer.current = null;
    }, SHIMMER_MS + 200);
  }, []);
  useEffect(() => () => {
    if (shimmerTimer.current !== null) window.clearTimeout(shimmerTimer.current);
  }, []);
  const [showTitleConfirm, setShowTitleConfirm] = useState(false);
  const [pendingAssetDelete, setPendingAssetDelete] = useState<{ assetId: string; filename: string | null } | null>(null);
  const [menuOpen, setMenuOpen] = useState(false);
  const menuRef = useRef<HTMLDivElement>(null);
  const [warpOpen, setWarpOpen] = useState(false);
  const [warpCollapsed, setWarpCollapsed] = useState(false);
  const warpRef = useRef<HTMLDivElement>(null);
  const [textWarpHost, setTextWarpHost] = useState<HTMLDivElement | null>(null);
  const [garnishScopeOpen, setGarnishScopeOpen] = useState(false);
  const garnishScopeRef = useRef<HTMLDivElement>(null);
  const [pendingFile, setPendingFile] = useState<File | null>(null);
  const [pendingUploadFile, setPendingUploadFile] = useState<File | null>(null);
  const [pendingDuplicateFile, setPendingDuplicateFile] = useState<{ file: File; projectName: string } | null>(null);
  const [scan, setScan] = useState<ScanState | null>(null);

  const { toasts, addToast: rawAddToast, dismissToast } = useToasts();
  const { notifications, addNotification, clearNotifications, dismissNotification } = useNotifications();
  const toastNotifMap = useRef<Map<string, string>>(new Map());

  const addToast = useCallback((type: ToastType, message: string, autoDismiss = true, action?: ToastAction, actions?: ToastAction[]) => {
    const notifId = `notif-${Date.now()}-${Math.random().toString(36).slice(2, 7)}`;
    addNotification(type, message, notifId);
    const toastId = rawAddToast(type, message, autoDismiss, action, actions);
    toastNotifMap.current.set(toastId, notifId);
    return toastId;
  }, [rawAddToast, addNotification]);

  const dismissToastWithNotif = useCallback((id: string) => {
    const notifId = toastNotifMap.current.get(id);
    if (notifId) {
      dismissNotification(notifId);
      toastNotifMap.current.delete(id);
    }
    dismissToast(id);
  }, [dismissToast, dismissNotification]);

  const setErrorWithNotif = useCallback((msg: string | null) => {
    setError(msg);
    if (msg) addNotification("error", msg);
  }, [addNotification]);

  useEffect(() => {
    fetchLanguages().then(setLanguages).catch(() =>
      setLanguages(["es", "fr", "de", "ja", "ko", "zh-cn"].map((c) => ({ code: c, name: c })))
    );
  }, []);

  useEffect(() => {
    fetchFonts(targLang)
      .then((r) => {
        setScript(r.script);
        setFontsByLang((p) => ({ ...p, [targLang]: r.fonts }));
        setFamiliesByLang((p) => ({ ...p, [targLang]: r.families ?? [] }));
      })
      .catch(() => setScript(""));
  }, [targLang]);

  // lazy per-language font lists for the region table's Font column
  const onNeedFonts = useCallback((lang: string) => {
    if (fontFetches.current.has(lang)) return;
    fontFetches.current.add(lang);
    fetchFonts(lang)
      .then((r) => {
        setFontsByLang((p) => ({ ...p, [lang]: r.fonts }));
        setFamiliesByLang((p) => ({ ...p, [lang]: r.families ?? [] }));
      })
      .catch(() => {});
  }, []);

  /** pull the full font catalog for one language, once. Only the Font
   * Manager needs it, so nothing pays for it until the panel is opened. */
  /** The font library changed on disk, so every cached catalog is stale.
   *
   * Both caches have to go: the per-language memo here, and the server's
   * own _CATALOG_CACHE (which the install endpoints clear). Dropping only
   * one leaves a newly installed font invisible until a restart. */
  const onFontLibraryChanged = useCallback(() => {
    const languages = Array.from(fullFontFetches.current);
    fullFontFetches.current.clear();
    setFullFamiliesByLang({});
    languages.forEach((lang) => onNeedFullFontsRef.current?.(lang));
  }, []);

  const onNeedFullFontsRef = useRef<((lang: string) => void) | null>(null);

  const onNeedFullFonts = useCallback((lang: string) => {
    if (fullFontFetches.current.has(lang)) return;
    fullFontFetches.current.add(lang);
    setFullFontsLoading(true);
    fetchFonts(lang, { full: true })
      .then((r) => setFullFamiliesByLang((p) => ({ ...p, [lang]: r.families ?? [] })))
      .catch(() => { fullFontFetches.current.delete(lang); })  // retryable
      .finally(() => setFullFontsLoading(false));
  }, []);
  // onFontLibraryChanged re-fetches the languages already loaded, and is
  // declared above this; the ref breaks the declaration cycle without
  // making either depend on the other's identity.
  onNeedFullFontsRef.current = onNeedFullFonts;

  // --- session lifecycle -------------------------------------------------

  /** wipe every piece of per-asset state, including the pending autosave
   * timer — stale state from a previous asset is what used to surface as
   * "failed to fetch manifest" and cross-asset data bleed */
  const resetSession = useCallback(() => {
    if (saveTimer.current) {
      clearTimeout(saveTimer.current);
      saveTimer.current = null;
    }
    setAsset(null);
    setPreviewUrl(null);
    setVideoJob(null);
    setPreRenderUrl(null);
    setPreviewRenderError(null);
    setPreviewCacheKey(null);
    setPreviewTransaction({ phase: "idle", requestId: previewRenderSeq.current, cause: "initial" });
    setRepairReviews([]);
    setRepairFallbackIds([]);
    brushDrawing.current = null;
    setBrushStrokes([]);
    setActiveBrushStroke(null);
    setBrushCursor(null);
    setBrushApplying(false);
    setAppliedCandidateIds([]);
    setInpaintPatchIds([]);
    localizedUndoStack.current = [];
    localizedRedoStack.current = [];
    localizedBaseline.current = null;
    setCanLocalizedUndo(false);
    setCanLocalizedRedo(false);
    setColorPickMode(null);
    manifestSkipHistory.current = true;
    setManifest([]);
    if (shimmerTimer.current !== null) {
      window.clearTimeout(shimmerTimer.current);
      shimmerTimer.current = null;
    }
    setNewRegionIds(null);
    setManualOrder([]);
    manifestUndoStack.current = [];
    manifestRedoStack.current = [];
    syncUndoRedo();
    setImgDim(null);
    setSceneRegions([]);
    setSemanticUnits([]);
    setSemanticDrafts({});
    setSemanticPlans({});
    setSemanticBusyId(null);
    setSelectedId(null);
    setHoveredId(null);
    setDrawMode(false);
    setImgSize(null);
    setReport(null);
    setRenderResult(null);
    setVerifyBusy(null);
    setVerifyStage(null);
    setVerifySelId(null);
    setReRenderingId(null);
    setApproved(false);
    setImportedHash(null);
    setHasEditsAfterImport(false);
    setSaveStatus("idle");
    setScan(null);
    setError(null);
    setLockedLangs(new Set());
    setStep(0);
  }, [syncUndoRedo]);

  /** pull a project's active asset + manifest from the server and hydrate
   * the editor — this is how sessions are resumed after a reload */
  /** true while a project's manifest is still being fetched */
  const [sessionLoading, setSessionLoading] = useState(false);

  const loadProjectSession = useCallback(async (projectId: string) => {
    const p = await getProject(projectId);
    setProject(p);
    setTargLang(p.target_lang);
    setSrcLang(p.source_lang);
    const active = p.active_asset;
    if (!active?.asset_url || !active.has_manifest) return p;
    const m = await getManifest(active.asset_id);
    setAsset({
      asset_id: active.asset_id,
      filename: active.filename ?? active.asset_id,
      asset_info: {
        asset_type: m.asset_type, frame_count: m.frame_count,
        fps: m.fps, duration: m.duration, source: null,
      },
      asset_url: active.asset_url,
    });
    setPreviewUrl(active.asset_url);
    if (m.asset_type === "video") {
      try {
        const job = await getLatestVideoJob(active.asset_id);
        setVideoJob(job);
        if (job.proxy_url) setPreviewUrl(job.proxy_url);
      } catch {
        setError("This video has no resumable processing job. Re-upload it to begin analysis.");
      }
    }
    manifestUndoStack.current = [];
    manifestRedoStack.current = [];
    syncUndoRedo();
    manifestSkipHistory.current = true;
    setManifest(m.instances);
    markRegionsNew(m.instances);
    setImgDim(m.img_dim);
    setSceneRegions(m.scene_regions ?? []);
    setSemanticUnits(m.semantic_units ?? []);
    if (m.src_lang) setSrcLang(m.src_lang);
    if (m.instances.length > 0) setStep(1);
    return p;
  }, [syncUndoRedo, markRegionsNew]);

  const goPantry = useCallback((mode: "full" | "pantry" | "create" = "full") => {
    setPantryMode(mode);
    prevScreen.current = displayedScreen;
    setDisplayedScreen("pantry");
    setScreen("pantry");
  }, [displayedScreen]);

  const openProject = useCallback((p: Project) => {
    resetSession();
    localStorage.setItem(PROJECT_KEY, p.id);
    setProject(p);
    setTargLang(p.target_lang);
    setSrcLang(p.source_lang);
    prevScreen.current = displayedScreen;
    setScreen("main");
    // The workspace is shown before the manifest arrives -- loadProjectSession
    // makes two round trips -- so without this a project WITH regions looks
    // for a moment exactly like a project with none.
    setSessionLoading(true);
    loadProjectSession(p.id)
      .then((full) => {
        if (full.active_asset?.has_manifest) {
          addToast("success", `resumed session: "${p.name}"`);
        }
      })
      .catch(() => {})
      .finally(() => setSessionLoading(false));
  }, [resetSession, loadProjectSession, addToast, displayedScreen]);

  // auto-resume project session when reloading directly into main screen
  const sessionRestoredRef = useRef(false);
  useEffect(() => {
    if (screen !== "main" || sessionRestoredRef.current) return;
    const pid = localStorage.getItem(PROJECT_KEY);
    if (!pid || project) return;
    sessionRestoredRef.current = true;
    setSessionLoading(true);
    loadProjectSession(pid)
      .catch(() => {
        sessionStorage.removeItem(SCREEN_KEY);
        setScreen("title");
      })
      .finally(() => setSessionLoading(false));
  }, [screen, project, loadProjectSession]);

  const refreshProject = useCallback(() => {
    if (!project) return;
    getProject(project.id).then(setProject).catch(() => {});
  }, [project]);

  useEffect(() => {
    fetchGlossaryStatus(project?.id).then(setGlossaryStatus).catch(() => setGlossaryStatus(null));
  }, [project?.id]);

  const currentManifest = useCallback((instances: InstText[] = manifest): TextManifest | null => {
    if (!asset) return null;
    return {
      asset_id: asset.asset_id,
      total_regions: instances.length,
      src_lang: srcLang,
      targ_lang: targLang,
      img_dim: imgDim ?? (imgSize ? [imgSize.width, imgSize.height] : null),
      scene_regions: sceneRegions,
      semantic_units: semanticUnits,
      asset_type: asset.asset_info.asset_type,
      frame_count: asset.asset_info.frame_count,
      fps: asset.asset_info.fps,
      duration: asset.asset_info.duration,
      prcssng_time: null,
      instances,
    };
  }, [asset, manifest, srcLang, targLang, imgDim, imgSize, sceneRegions, semanticUnits]);

  const queueLocalizedPreview = useCallback((instances: InstText[], cause: PreviewCause) => {
    const snapshot = currentManifest(instances);
    if (snapshot) queuedPreviewManifest.current = snapshot;
    previewCauseRef.current = cause;
    setPreviewRetryRevision((revision) => revision + 1);
  }, [currentManifest]);

  useEffect(() => {
    const candidates = repairReviews.flatMap((review) => review.candidates);
    if (!asset || candidates.length === 0) {
      candidatePreviewSeq.current += 1; // invalidate any earlier in-flight batch
      setLocalizedCandidatePreviews({});
      return;
    }
    // Snapshot once per batch: a preview must represent the same unsaved
    // manifest revision that the localized canvas is currently rendering.
    const snapshot = currentManifest();
    const seq = ++candidatePreviewSeq.current;
    if (!snapshot) {
      console.warn("localized candidate preview skipped: no in-memory manifest");
      setLocalizedCandidatePreviews(Object.fromEntries(candidates.map((candidate) => [candidate.id, {
        status: "error" as const, error: "No current localization manifest",
      }])));
      return;
    }
    setLocalizedCandidatePreviews(Object.fromEntries(candidates.map((candidate) => [candidate.id, {
      status: "loading" as const,
    }])));
    Promise.allSettled(candidates.map(async (candidate) => ({
      id: candidate.id,
      url: await previewCandidateLocalized(asset.asset_id, candidate.id, snapshot, targLang),
    }))).then((results) => {
      if (seq !== candidatePreviewSeq.current) return;
      const next: Record<string, LocalizedCandidatePreview> = {};
      results.forEach((result, index) => {
        const candidate = candidates[index];
        if (result.status === "fulfilled") next[candidate.id] = { status: "ready", url: result.value.url };
        else {
          console.warn("localized candidate preview failed", candidate.id, result.reason);
          next[candidate.id] = { status: "error", error: "Localized preview unavailable" };
        }
      });
      setLocalizedCandidatePreviews(next);
    });
  }, [asset, repairReviews, currentManifest, targLang, candidatePreviewRevision]);

  const syncTreatmentPatches = useCallback((patches: InpaintPatch[]) => {
    setInpaintPatchIds(patches.map((patch) => patch.id));
    setAppliedCandidateIds(patches.flatMap((patch) => patch.candidate_id ? [patch.candidate_id] : []));
  }, []);

  const flushCurrentManifest = useCallback(async (): Promise<TextManifest | null> => {
    const snapshot = currentManifest();
    if (!snapshot || !asset) return null;
    if (saveTimer.current) {
      clearTimeout(saveTimer.current);
      saveTimer.current = null;
    }
    setSaveStatus("saving");
    try {
      await putManifest(asset.asset_id, snapshot);
      setSaveStatus("saved");
      return snapshot;
    } catch (error) {
      setSaveStatus("idle");
      throw error;
    }
  }, [asset, currentManifest]);

  const autoSave = useCallback((instances: InstText[]) => {
    if (!asset) return;
    setSaveStatus("saving");
    if (saveTimer.current) clearTimeout(saveTimer.current);
    saveTimer.current = setTimeout(async () => {
      try {
        const m: TextManifest = {
          asset_id: asset.asset_id,
          total_regions: instances.length,
          src_lang: srcLang,
          targ_lang: targLang,
          img_dim: imgDim ?? (imgSize ? [imgSize.width, imgSize.height] : null),
          scene_regions: sceneRegions,
          semantic_units: semanticUnits,
          asset_type: asset.asset_info.asset_type,
          frame_count: asset.asset_info.frame_count,
          fps: asset.asset_info.fps,
          duration: asset.asset_info.duration,
          prcssng_time: null,
          instances,
        };
        const { resolved_fonts } = await putManifest(asset.asset_id, m);
        setSaveStatus("saved");
        // "auto" font resolution depends on target_text/target_language,
        // exactly what changes during Translate-step editing — the
        // server recomputes it on every save; merge back in so the Font
        // column and preview stay current without a second fetch
        if (resolved_fonts && Object.keys(resolved_fonts).length > 0) {
          setManifest((prev) => prev.map((i) =>
            resolved_fonts[i.id] !== undefined && resolved_fonts[i.id] !== i.resolved_font_family
              ? { ...i, resolved_font_family: resolved_fonts[i.id] }
              : i
          ));
        }
      } catch {
        setSaveStatus("idle");
      }
    }, 1500);
  }, [asset, srcLang, targLang, imgDim, imgSize, sceneRegions, semanticUnits]);

  const cloneLocalizedSnapshot = useCallback((instances: InstText[] = manifest, patchIds: string[] = inpaintPatchIds): LocalizedSnapshot => ({
    manifest: JSON.parse(JSON.stringify(instances)) as InstText[],
    patchIds: [...patchIds],
  }), [manifest, inpaintPatchIds]);

  const syncLocalizedUndoRedo = useCallback(() => {
    setCanLocalizedUndo(localizedUndoStack.current.length > 0);
    setCanLocalizedRedo(localizedRedoStack.current.length > 0);
  }, []);

  const recordLocalizedChange = useCallback(() => {
    if (step !== 3) return;
    localizedUndoStack.current.push(cloneLocalizedSnapshot());
    if (localizedUndoStack.current.length > 80) localizedUndoStack.current.shift();
    localizedRedoStack.current = [];
    syncLocalizedUndoRedo();
  }, [step, cloneLocalizedSnapshot, syncLocalizedUndoRedo]);

  const restoreLocalizedSnapshot = useCallback(async (snapshot: LocalizedSnapshot) => {
    if (!asset) return;
    const result = await restoreTreatment(asset.asset_id, snapshot.patchIds);
    manifestSkipHistory.current = true;
    setManifestRaw(snapshot.manifest);
    syncTreatmentPatches(result.patches);
    setPatchRevision((revision) => revision + 1);
    autoSave(snapshot.manifest);
  }, [asset, autoSave, syncTreatmentPatches]);

  const undoLocalized = useCallback(async () => {
    const previous = localizedUndoStack.current.pop();
    if (!previous) return;
    localizedRedoStack.current.push(cloneLocalizedSnapshot());
    try {
      await restoreLocalizedSnapshot(previous);
      addToast("info", "undid localized canvas change");
    } catch (error) {
      localizedUndoStack.current.push(previous);
      localizedRedoStack.current.pop();
      setErrorWithNotif(String(error));
    } finally { syncLocalizedUndoRedo(); }
  }, [addToast, cloneLocalizedSnapshot, restoreLocalizedSnapshot, syncLocalizedUndoRedo]);

  const redoLocalized = useCallback(async () => {
    const next = localizedRedoStack.current.pop();
    if (!next) return;
    localizedUndoStack.current.push(cloneLocalizedSnapshot());
    try {
      await restoreLocalizedSnapshot(next);
      addToast("info", "redid localized canvas change");
    } catch (error) {
      localizedRedoStack.current.push(next);
      localizedUndoStack.current.pop();
      setErrorWithNotif(String(error));
    } finally { syncLocalizedUndoRedo(); }
  }, [addToast, cloneLocalizedSnapshot, restoreLocalizedSnapshot, syncLocalizedUndoRedo]);

  const updateSelectedStyle = useCallback((patch: Partial<NonNullable<InstText["style_profile"]>>) => {
    const selected = renderSelId ?? prevSelId;
    if (!selected) return;
    recordLocalizedChange();
    const next = manifest.map((inst): InstText => inst.id === selected ? {
      ...inst,
      style_profile: { ...(inst.style_profile ?? {}), ...patch } as NonNullable<InstText["style_profile"]>,
    } : inst);
    setManifest(next);
    queueLocalizedPreview(next, "style");
    autoSave(next);
  }, [renderSelId, prevSelId, recordLocalizedChange, manifest, autoSave, setManifest, queueLocalizedPreview]);

  const manifestWithSelectedQuad = useCallback((instances: InstText[], quad: Quad): InstText[] => {
    const selected = renderSelId ?? prevSelId;
    if (!selected) return instances;
    return instances.map((inst): InstText => {
      if (inst.id !== selected) return inst;
      const transform = { ...(inst.style_profile?.transform ?? {}), quad, preset: "custom" };
      return {
        ...inst,
        style_profile: { ...(inst.style_profile ?? {}), transform } as NonNullable<InstText["style_profile"]>,
      };
    });
  }, [renderSelId, prevSelId]);

  const beginQuadGesture = useCallback(() => {
    recordLocalizedChange();
    quadGestureManifest.current = manifest;
  }, [manifest, recordLocalizedChange]);

  const previewQuadGesture = useCallback((quad: Quad) => {
    const base = quadGestureManifest.current ?? manifest;
    const next = manifestWithSelectedQuad(base, quad);
    manifestSkipHistory.current = true;
    setManifestRaw(next);
  }, [manifest, manifestWithSelectedQuad]);

  const finishQuadGesture = useCallback((quad: Quad, cancelled = false) => {
    const base = quadGestureManifest.current ?? manifest;
    quadGestureManifest.current = null;
    const next = manifestWithSelectedQuad(base, quad);
    manifestSkipHistory.current = true;
    setManifest(next);
    if (cancelled) {
      localizedUndoStack.current.pop();
      syncLocalizedUndoRedo();
    } else {
      queueLocalizedPreview(next, "style");
      autoSave(next);
    }
  }, [manifest, manifestWithSelectedQuad, setManifest, queueLocalizedPreview, autoSave, syncLocalizedUndoRedo]);

  const clearSelectedQuad = useCallback(() => {
    const selected = renderSelId ?? prevSelId;
    if (!selected) return;
    recordLocalizedChange();
    const next = manifest.map((inst): InstText => {
      if (inst.id !== selected) return inst;
      const transform = { ...(inst.style_profile?.transform ?? {}) };
      delete transform.quad;
      return {
        ...inst,
        style_profile: { ...(inst.style_profile ?? {}), transform } as NonNullable<InstText["style_profile"]>,
      };
    });
    manifestSkipHistory.current = true;
    setManifest(next);
    queueLocalizedPreview(next, "style");
    autoSave(next);
  }, [renderSelId, prevSelId, manifest, recordLocalizedChange, setManifest, queueLocalizedPreview, autoSave]);

  /** apply a Font Manager pick to the selected Render region.
   *
   * Goes through updateSelectedStyle rather than onFontChange so the pick
   * is undoable, refreshes the localized preview, and sets font_weight —
   * onFontChange does none of the three. */
  const onFontManagerPick = useCallback((fam: FontFamily, weight: FontWeight) => {
    updateSelectedStyle({ font_family: weight.path, font_weight: weight.subfamily });
    // a font outside the ranked 24 has to join the session's dropdown list
    // too, or the combobox cannot resolve the path back to a family name
    const lang = styleTargetInst?.target_language ?? targLang;
    setFamiliesByLang((prev) => ({ ...prev, [lang]: mergeFamily(prev[lang] ?? [], fam) }));
    setRecentFonts(pushRecent(localStorage, { family: fam.family, path: weight.path }));
  }, [updateSelectedStyle, styleTargetInst, targLang]);

  const updateSelectedGarnish = useCallback((patch: Partial<NonNullable<InstText["garnish_override"]>>) => {
    const selected = renderSelId ?? prevSelId;
    if (!selected) return;
    const selectedInst = manifest.find((inst) => inst.id === selected);
    if (!selectedInst) return;
    const allRegions = selectedInst.garnish_scope !== "per_region";
    const b = selectedInst.bounding_box;
    const surface = sceneRegions.find((region) => b.x + b.width / 2 >= region.bbox.x && b.x + b.width / 2 <= region.bbox.x + region.bbox.width && b.y + b.height / 2 >= region.bbox.y && b.y + b.height / 2 <= region.bbox.y + region.bbox.height);
    const baseProfile = selectedInst.garnish_override ?? surface?.garnish_profile ?? DEFAULT_GARNISH_PROFILE;
    const profile = { ...baseProfile, source_confidence: 1, ...patch };
    recordLocalizedChange();
    const next = manifest.map((inst) => {
      if (allRegions) {
        if (inst.excluded || !inst.target_text || inst.dnt) return inst;
        return { ...inst, garnish_override: { ...profile, ...(inst.garnish_override ?? {}), ...patch } };
      }
      if (inst.id !== selected) return inst;
      return { ...inst, garnish_override: profile };
    });
    setManifest(next);
    queueLocalizedPreview(next, "garnish");
    autoSave(next);
  }, [renderSelId, prevSelId, recordLocalizedChange, manifest, sceneRegions, autoSave, setManifest, queueLocalizedPreview]);

  const setSelectedGarnishEnabled = useCallback((enabled: boolean) => {
    const selected = renderSelId ?? prevSelId;
    if (!selected) return;
    const inst = manifest.find((item) => item.id === selected);
    if (!inst) return;
    const allRegions = inst.garnish_scope !== "per_region";
    recordLocalizedChange();
    const next = manifest.map((item) => {
      if (allRegions) {
        if (item.excluded || !item.target_text || item.dnt) return item;
        return { ...item, garnish_enabled: enabled };
      }
      if (item.id !== selected) return item;
      return { ...item, garnish_enabled: enabled };
    });
    setManifest(next);
    queueLocalizedPreview(next, "garnish");
    autoSave(next);
  }, [renderSelId, prevSelId, recordLocalizedChange, manifest, autoSave, setManifest, queueLocalizedPreview]);

  const useSelectedSceneGarnish = useCallback(() => {
    const selected = renderSelId ?? prevSelId;
    if (!selected) return;
    const inst = manifest.find((item) => item.id === selected);
    if (!inst) return;
    const allRegions = inst.garnish_scope !== "per_region";
    recordLocalizedChange();
    const next = manifest.map((item) => {
      if (allRegions) {
        if (item.excluded || !item.target_text || item.dnt) return item;
        return { ...item, garnish_override: null, garnish_enabled: null };
      }
      if (item.id !== selected) return item;
      return { ...item, garnish_override: null, garnish_enabled: null };
    });
    setManifest(next);
    queueLocalizedPreview(next, "garnish");
    autoSave(next);
  }, [renderSelId, prevSelId, recordLocalizedChange, manifest, autoSave, setManifest, queueLocalizedPreview]);

  const setSelectedGarnishScope = useCallback((scope: "whole_selection" | "per_region") => {
    const selected = renderSelId ?? prevSelId;
    if (!selected) return;
    const inst = manifest.find((item) => item.id === selected);
    if (!inst) {
      addToast("error", "No text selected to apply garnish scope.");
      return;
    }
    recordLocalizedChange();
    const next = manifest.map((item) => item.id === selected ? { ...item, garnish_scope: scope } : item);
    setManifest(next);
    queueLocalizedPreview(next, "garnish");
    autoSave(next);
  }, [renderSelId, prevSelId, manifest, recordLocalizedChange, autoSave, addToast, queueLocalizedPreview]);

  useEffect(() => {
    if (step !== 3 || !asset || localizedBaseline.current) return;
    const snapshot = currentManifest();
    if (!snapshot) return;
    let disposed = false;
    (async () => {
      try {
        const baseline = await getLocalizedBaseline(asset.asset_id);
        if (!disposed) localizedBaseline.current = { manifest: baseline.manifest.instances, patchIds: baseline.patch_ids };
      } catch {
        try {
          const baseline = await captureLocalizedBaseline(asset.asset_id, snapshot, inpaintPatchIds);
          if (!disposed) localizedBaseline.current = { manifest: baseline.manifest.instances, patchIds: baseline.patch_ids };
        } catch {
          // The in-memory snapshot remains a safe session-only reset fallback.
          if (!disposed) localizedBaseline.current = cloneLocalizedSnapshot();
        }
      }
    })();
    return () => { disposed = true; };
  }, [step, asset, currentManifest, inpaintPatchIds, cloneLocalizedSnapshot]);

  const resetLocalizedCanvas = useCallback(async () => {
    const baseline = localizedBaseline.current;
    if (!baseline) return;
    if (!confirm("Reset text placement, appearance, transforms, and treatment edits to the Render-entry baseline?")) return;
    recordLocalizedChange();
    try {
      await restoreLocalizedSnapshot(baseline);
      addToast("info", "restored the localized canvas baseline");
    } catch (error) { setErrorWithNotif(String(error)); }
  }, [addToast, recordLocalizedChange, restoreLocalizedSnapshot]);

  const sampleCanvasFill = useCallback((event: React.PointerEvent<HTMLImageElement>, surface: "source" | "localized"): boolean => {
    if (!colorPickMode || !renderSelId) return false;
    const image = event.currentTarget;
    const rect = image.getBoundingClientRect();
    if (!image.naturalWidth || !image.naturalHeight || !rect.width || !rect.height) return false;
    const x = Math.max(0, Math.min(image.naturalWidth - 1, Math.floor((event.clientX - rect.left) * image.naturalWidth / rect.width)));
    const y = Math.max(0, Math.min(image.naturalHeight - 1, Math.floor((event.clientY - rect.top) * image.naturalHeight / rect.height)));
    try {
      const canvas = document.createElement("canvas");
      canvas.width = image.naturalWidth;
      canvas.height = image.naturalHeight;
      const context = canvas.getContext("2d", { willReadFrequently: true });
      if (!context) return false;
      context.drawImage(image, 0, 0);
      const [r, g, b] = context.getImageData(x, y, 1, 1).data;
      const color = `#${[r, g, b].map((value) => value.toString(16).padStart(2, "0")).join("")}`;
      event.preventDefault();
      updateSelectedStyle({ color });
      setColorPickMode(null);
      addToast("success", `sampled ${color} from ${surface === "source" ? "source" : "localized"} canvas`);
      return true;
    } catch {
      addToast("warning", "the selected canvas cannot be sampled yet");
      return false;
    }
  }, [colorPickMode, renderSelId, updateSelectedStyle, addToast]);

  // Render tab is a server-backed doppelganger of the final compositor.
  // It accepts the current in-memory manifest, so the autosave debounce can
  // never make the canvas lag behind a text/style edit.
  useEffect(() => {
    if (step !== 3 || !asset) return;
    const seq = ++previewRenderSeq.current;
    const controller = new AbortController();
    const cause = previewCauseRef.current;
    previewCauseRef.current = "style";
    setPreviewTransaction({ phase: "debouncing", requestId: seq, cause });
    const timer = setTimeout(async () => {
      if (controller.signal.aborted) return;
      try {
        const queued = queuedPreviewManifest.current;
        const snapshot = queued ?? currentManifest();
        if (queuedPreviewManifest.current === queued) queuedPreviewManifest.current = null;
        if (!snapshot) {
          if (seq === previewRenderSeq.current) setPreviewTransaction({ phase: "idle", requestId: seq, cause });
          return;
        }
        if (seq === previewRenderSeq.current) setPreviewTransaction({ phase: "rendering", requestId: seq, cause });
        // Style, text, and transform edits cannot change Scene or the
        // cleansed base.  Fast-path them while retaining the full path for an
        // initial/incomplete manifest; the server still re-cleanses on a real
        // geometry cache miss.
        const fastPreview = cause === "garnish" || cause === "treatment" || cause === "style" || cause === "text";
        const result = await renderPreview(asset.asset_id, targLang, snapshot, controller.signal, fastPreview);
        if (!controller.signal.aborted && seq === previewRenderSeq.current) {
          setPreRenderUrl(result.output_url);
          setPreviewRenderError(null);
          setPreviewCacheKey(result.cleanse_cache_key);
          setPreviewTransaction({ phase: "idle", requestId: seq, cause });
          // Scene enriches the current in-memory manifest during preview.
          // Keep its surface profiles in sync with the panel that exposes
          // the corresponding Garnish recommendations.
          if (Array.isArray(result.text_manifest.scene_regions)) {
            setSceneRegions((previous) => JSON.stringify(previous) === JSON.stringify(result.text_manifest.scene_regions)
              ? previous : result.text_manifest.scene_regions);
          }
          const repairItems = result.text_manifest.instances.flatMap((inst) => {
            const repair = inst.repair_provenance;
            const candidates = (repair?.candidates ?? []).flatMap((candidate) => {
              const artifact = candidate.evidence?.artifact;
              return artifact?.url ? [{
                id: artifact.id, provider: artifact.provider, decision: artifact.decision,
                url: artifact.url, bbox: artifact.bbox,
                cacheKey: artifact.cache_key ?? result.cleanse_cache_key,
              }] : [];
            });
            return repair?.review_required ? [{
              id: inst.id, provider: repair.executed_provider ?? repair.requested_provider,
              reason: repair.reason, candidates,
            }] : [];
          });
          // An unavailable optional model is an asset-level setup state, not
          // four independent failed repairs.  Keep it visible and require a
          // final-render acknowledgement, but reserve the review queue for a
          // real generated candidate that needs an editor's judgement.
          const unavailable = repairItems.filter((repair) => repair.provider === "telea_fallback" && repair.reason.startsWith("no configured local neural provider"));
          const nextFallbackIds = unavailable.map((repair) => repair.id);
          const nextReviews = repairItems.filter((repair) => !unavailable.includes(repair));
          setRepairFallbackIds((previous) => JSON.stringify(previous) === JSON.stringify(nextFallbackIds) ? previous : nextFallbackIds);
          // Do not rebuild Smart Fill thumbnails after a garnish-only render:
          // the review evidence is unchanged, and regenerating each candidate
          // makes a one-control edit feel like a full Cleanse pass.
          setRepairReviews((previous) => JSON.stringify(previous) === JSON.stringify(nextReviews) ? previous : nextReviews);
        }
      } catch (e) {
        if (!controller.signal.aborted && seq === previewRenderSeq.current) {
          const error = String(e);
          setPreviewRenderError(error);
          setPreviewTransaction({ phase: "failed", requestId: seq, cause, error });
        }
        // The last good canvas remains visible while a transient preview
        // request fails; final Render still reports actionable errors.
      }
    // Garnish is an exact mask/filter pass and can remain responsive without
    // a visually misleading client-side approximation.
    }, cause === "garnish" || cause === "treatment" ? 150 : 350);
    return () => { controller.abort(); clearTimeout(timer); };
    // Scene's returned garnish metadata is presentation data.  It must not
    // be a dependency here: accepting it after a successful response would
    // otherwise schedule a second, latent canvas render.
  }, [step, asset, targLang, manifest, patchRevision, previewRetryRevision]);

  useEffect(() => {
    if (!asset) return;
    getTreatment(asset.asset_id)
      .then((state) => syncTreatmentPatches(state.patches))
      .catch(() => { setInpaintPatchIds([]); setAppliedCandidateIds([]); });
  }, [asset, syncTreatmentPatches]);






  const applyReviewCandidate = useCallback(async (candidate: RepairCandidate) => {
    if (!asset) return;
    try {
      if (previewPending || previewCacheKey !== candidate.cacheKey) {
        addToast("warning", "waiting for the current treatment preview before applying this repair");
        return;
      }
      recordLocalizedChange();
      const result = await applyRepairCandidate(asset.asset_id, candidate.id, candidate.cacheKey);
      syncTreatmentPatches(result.patches);
      setPatchRevision((revision) => revision + 1);
      addToast("success", result.already_applied ? `${providerLabel(candidate.provider)} is already in the treatment layer` : `applied ${providerLabel(candidate.provider)} as an undoable treatment`);
    } catch (e) { setErrorWithNotif(String(e)); }
  }, [asset, addToast, previewPending, previewCacheKey, syncTreatmentPatches, recordLocalizedChange]);

  const applyAllReviewCandidates = useCallback(async () => {
    if (!asset || previewPending) return;
    const candidates = Array.from(new Map(
      repairReviews.flatMap((review) => review.candidates.map((candidate) => [candidate.id, candidate]))
    ).values()).filter((candidate) => localizedCandidatePreviews[candidate.id]?.status === "ready" && !appliedCandidateIds.includes(candidate.id));
    for (const candidate of candidates) {
      if (previewCacheKey !== candidate.cacheKey) {
        addToast("warning", `skipping ${providerLabel(candidate.provider)} — preview is no longer current`);
        continue;
      }
      try {
        recordLocalizedChange();
        const result = await applyRepairCandidate(asset.asset_id, candidate.id, candidate.cacheKey);
        syncTreatmentPatches(result.patches);
        setPatchRevision((revision) => revision + 1);
        addToast("success", result.already_applied ? `${providerLabel(candidate.provider)} is already in the treatment layer` : `applied ${providerLabel(candidate.provider)} as an undoable treatment`);
      } catch (e) {
        setErrorWithNotif(String(e));
        break;
      }
    }
  }, [asset, repairReviews, localizedCandidatePreviews, appliedCandidateIds, previewPending, previewCacheKey, addToast, syncTreatmentPatches, recordLocalizedChange]);

  const pointOnLocalizedCanvas = useCallback((event: React.PointerEvent<HTMLImageElement>): [number, number] | null => {
    if (!imgDim) return null;
    const rect = event.currentTarget.getBoundingClientRect();
    if (rect.width <= 0 || rect.height <= 0) return null;
    return [
      Math.max(0, Math.min(imgDim[0] - 1, Math.round((event.clientX - rect.left) * imgDim[0] / rect.width))),
      Math.max(0, Math.min(imgDim[1] - 1, Math.round((event.clientY - rect.top) * imgDim[1] / rect.height))),
    ];
  }, [imgDim]);

  const beginBrushStroke = useCallback((event: React.PointerEvent<HTMLImageElement>) => {
    if (!brushMode || brushApplying) return;
    const point = pointOnLocalizedCanvas(event);
    if (!point) return;
    event.preventDefault();
    const stroke = { id: `${Date.now()}-${event.pointerId}-${Math.random().toString(36).slice(2, 7)}`, points: [point] as [number, number][] };
    brushDrawing.current = { pointerId: event.pointerId, stroke };
    setActiveBrushStroke(stroke);
    setBrushCursor(point);
    event.currentTarget.setPointerCapture(event.pointerId);
  }, [brushMode, brushApplying, pointOnLocalizedCanvas]);

  const extendBrushStroke = useCallback((event: React.PointerEvent<HTMLImageElement>) => {
    if (!brushMode) return;
    const point = pointOnLocalizedCanvas(event);
    if (!point) return;
    // Track the pointer whether or not a stroke is live, so the radius ring
    // previews the brush on hover. It used to move only while dragging, which
    // meant the one moment you needed to see how big the brush was -- before
    // committing a stroke -- was the one moment it was not shown.
    setBrushCursor(point);
    const drawing = brushDrawing.current;
    if (!drawing || drawing.pointerId !== event.pointerId) return;
    event.preventDefault();
    const prior = drawing.stroke.points[drawing.stroke.points.length - 1];
    // Pointer events can arrive several times with the same mapped pixel.
    // Ignore those duplicates while retaining every meaningful held-click path.
    if (prior && prior[0] === point[0] && prior[1] === point[1]) return;
    drawing.stroke = { ...drawing.stroke, points: [...drawing.stroke.points, point] };
    brushDrawing.current = drawing;
    setActiveBrushStroke(drawing.stroke);
  }, [brushMode, pointOnLocalizedCanvas]);

  const finishBrushStroke = useCallback((event: React.PointerEvent<HTMLImageElement>, cancelled = false) => {
    const drawing = brushDrawing.current;
    if (!drawing || drawing.pointerId !== event.pointerId) return;
    event.preventDefault();
    // Clear first: releasePointerCapture can synchronously emit
    // lostpointercapture in some browsers.
    brushDrawing.current = null;
    try {
      if (event.currentTarget.hasPointerCapture(event.pointerId)) event.currentTarget.releasePointerCapture(event.pointerId);
    } catch { /* capture can already be released after a browser cancellation */ }
    setActiveBrushStroke(null);
    if (!cancelled && drawing.stroke.points.length) {
      setBrushStrokes((strokes) => [...strokes, drawing.stroke]);
    }
  }, []);

  const applyBrush = useCallback(async () => {
    if (!asset || !brushStrokes.length || brushApplying) return;
    setBrushApplying(true);
    try {
      recordLocalizedChange();
      const snapshot = await flushCurrentManifest();
      let remaining = [...brushStrokes];
      for (const stroke of brushStrokes) {
        const result = await createInpaintPatch(asset.asset_id, { points: stroke.points, mode: "blur", radius: brushRadius, blur_strength: brushIntensity / 100 }, snapshot ?? undefined);
        syncTreatmentPatches(result.patches);
        remaining = remaining.filter((candidate) => candidate.id !== stroke.id);
        setBrushStrokes(remaining);
      }
      setPatchRevision((v) => v + 1);
      addToast("success", "text blur applied");
    } catch (e) { setErrorWithNotif(String(e)); }
    finally { setBrushApplying(false); }
  }, [asset, brushStrokes, brushRadius, brushIntensity, brushApplying, addToast, flushCurrentManifest, syncTreatmentPatches, recordLocalizedChange]);

  type CanvasTransformKey = "skew_x" | "skew_y" | "arc" | "scale_x" | "scale_y" | "offset_x" | "offset_y" | "rotation";
  const LOCKABLE_TRANSFORM_KEYS: CanvasTransformKey[] = ["skew_x", "skew_y", "arc", "rotation", "scale_x", "scale_y", "offset_x", "offset_y"];

  const isTransformLocked = useCallback((key: CanvasTransformKey): boolean => {
    const inst = manifest.find((item) => item.id === (renderSelId ?? prevSelId));
    if (inst && ["skew_x", "skew_y", "scale_x", "scale_y"].includes(key) && hasActivePerspective(inst)) {
      return true;
    }
    const transform = inst?.style_profile?.transform;
    return Boolean(transform?.locked_fields?.includes(key));
  }, [manifest, renderSelId, prevSelId]);

  const updateCanvasTransform = useCallback((key: CanvasTransformKey, value: number) => {
    const transform = manifest.find((inst) => inst.id === (renderSelId ?? prevSelId))?.style_profile?.transform ?? {};
    if (transform.locked_fields?.includes(key)) return;
    // A manual slider is intentionally a Custom transform.  It must not
    // retain a named preset's hidden amount and give the user a result they
    // cannot read back from the controls.
    const { amount: _amount, ...manualTransform } = transform;
    updateSelectedStyle({ transform: { ...manualTransform, preset: "custom", [key]: value } });
  }, [renderSelId, prevSelId, manifest, updateSelectedStyle]);

  const transformHistoryRef = useRef<Record<CanvasTransformKey, { undoStack: number[]; redoStack: number[] }>>({
    skew_x: { undoStack: [], redoStack: [] }, skew_y: { undoStack: [], redoStack: [] },
    arc: { undoStack: [], redoStack: [] },
    scale_x: { undoStack: [], redoStack: [] }, scale_y: { undoStack: [], redoStack: [] },
    offset_x: { undoStack: [], redoStack: [] }, offset_y: { undoStack: [], redoStack: [] },
    rotation: { undoStack: [], redoStack: [] },
  });
  const trackTransformChange = useCallback((key: CanvasTransformKey, previous: number, next: number) => {
    if (!Number.isFinite(next) || previous === next) return;
    const history = transformHistoryRef.current[key];
    if (history.undoStack[history.undoStack.length - 1] !== previous) history.undoStack.push(previous);
    if (history.undoStack.length > 50) history.undoStack.shift();
    history.redoStack = [];
  }, []);
  const toggleTransformLock = useCallback((key: CanvasTransformKey) => {
    const transform = manifest.find((inst) => inst.id === (renderSelId ?? prevSelId))?.style_profile?.transform ?? {};
    const locked = new Set(transform.locked_fields ?? []);
    if (locked.has(key)) locked.delete(key); else locked.add(key);
    updateSelectedStyle({ transform: { ...transform, locked_fields: Array.from(locked) } });
  }, [manifest, renderSelId, prevSelId, updateSelectedStyle]);
  const toggleAllTransformLocks = useCallback(() => {
    const transform = manifest.find((inst) => inst.id === (renderSelId ?? prevSelId))?.style_profile?.transform ?? {};
    const allLocked = LOCKABLE_TRANSFORM_KEYS.every((key) => transform.locked_fields?.includes(key));
    updateSelectedStyle({ transform: { ...transform, locked_fields: allLocked ? [] : [...LOCKABLE_TRANSFORM_KEYS] } });
  }, [manifest, renderSelId, prevSelId, updateSelectedStyle]);
  const undoTransformKey = useCallback((key: CanvasTransformKey, fallback: number) => {
    if (isTransformLocked(key)) return;
    const history = transformHistoryRef.current[key];
    const previous = history.undoStack.pop();
    if (previous === undefined) return;
    const current = manifest.find((inst) => inst.id === (renderSelId ?? prevSelId))?.style_profile?.transform?.[key] ?? fallback;
    history.redoStack.push(Number(current));
    updateCanvasTransform(key, previous);
  }, [manifest, renderSelId, prevSelId, updateCanvasTransform, isTransformLocked]);
  const redoTransformKey = useCallback((key: CanvasTransformKey) => {
    if (isTransformLocked(key)) return;
    const history = transformHistoryRef.current[key];
    const next = history.redoStack.pop();
    if (next === undefined) return;
    const current = manifest.find((inst) => inst.id === (renderSelId ?? prevSelId))?.style_profile?.transform?.[key] ?? next;
    history.undoStack.push(Number(current));
    updateCanvasTransform(key, next);
  }, [manifest, renderSelId, prevSelId, updateCanvasTransform, isTransformLocked]);

  const applyCanvasWarpPreset = useCallback((preset: string) => {
    const transform = manifest.find((inst) => inst.id === (renderSelId ?? prevSelId))?.style_profile?.transform ?? {};
    if (preset === "none") {
      const locked = new Set(transform.locked_fields ?? []);
      const reset: Record<string, number | string> = { preset: "none", amount: 0, skew_anchor: "center" };
      if (!locked.has("arc")) reset.arc = 0;
      if (!locked.has("skew_x")) reset.skew_x = 0;
      if (!locked.has("skew_y")) reset.skew_y = 0;
      if (!locked.has("scale_x")) reset.scale_x = 1;
      if (!locked.has("scale_y")) reset.scale_y = 1;
      if (!locked.has("offset_x")) reset.offset_x = 0;
      if (!locked.has("offset_y")) reset.offset_y = 0;
      if (!locked.has("rotation")) reset.rotation = 0;
      const cleared = { ...transform, ...reset };
      delete cleared.quad;
      updateSelectedStyle({ transform: cleared });
      return;
    }
    updateSelectedStyle({ transform: { ...transform, preset, amount: transform.amount && transform.amount !== 0 ? transform.amount : 12 } });
  }, [renderSelId, prevSelId, manifest, updateSelectedStyle]);

  /** Push the selected region's transform onto every other live region.
   *
   *  Regions on one sign share a plane: the same rotation, the same skew,
   *  the same stretch. Dialling that in per region is the tedious part of
   *  this step, and it was the only way to do it.
   *
   *  The perspective QUAD is deliberately not copied. Its corners describe
   *  one region's own outline, so pushing them onto a region of a different
   *  shape distorts it rather than matching it — the affine values are the
   *  ones that describe a shared plane. Each target's own locked fields are
   *  honoured, which is what the lock is for. */
  const applyTransformToAllRegions = useCallback(() => {
    const selected = renderSelId ?? prevSelId;
    if (!selected) return;
    const source = manifest.find((inst) => inst.id === selected)?.style_profile?.transform;
    if (!source) return;
    const { locked_fields: _locks, quad: _quad, ...shared } = source;
    let applied = 0;
    const next = manifest.map((inst): InstText => {
      if (inst.id === selected || inst.excluded) return inst;
      const current = inst.style_profile?.transform ?? {};
      const locked = new Set(current.locked_fields ?? []);
      const merged: Record<string, unknown> = { ...current };
      let touched = false;
      for (const [key, value] of Object.entries(shared)) {
        if (locked.has(key)) continue;
        merged[key] = value;
        touched = true;
      }
      if (!touched) return inst;
      applied += 1;
      return {
        ...inst,
        style_profile: {
          ...(inst.style_profile ?? {}),
          transform: merged,
        } as NonNullable<InstText["style_profile"]>,
      };
    });
    if (!applied) {
      addToast("info", "No other region could take this transform — they are all locked or excluded");
      return;
    }
    recordLocalizedChange();
    setManifest(next);
    queueLocalizedPreview(next, "style");
    autoSave(next);
    addToast("success", `Transform applied to ${applied} other region${applied === 1 ? "" : "s"}`);
  }, [renderSelId, prevSelId, manifest, recordLocalizedChange, setManifest,
      queueLocalizedPreview, autoSave, addToast]);

  const undoLastInpaint = useCallback(async () => {
    if (!asset || !inpaintPatchIds.length) return;
    try {
      const result = await undoInpaint(asset.asset_id, inpaintPatchIds[inpaintPatchIds.length - 1]);
      syncTreatmentPatches(result.patches);
      setPatchRevision((v) => v + 1);
    } catch (e) { setErrorWithNotif(String(e)); }
  }, [asset, inpaintPatchIds, addToast, syncTreatmentPatches]);

  // --- upload flow + language guard --------------------------------------

  /** background language scan (lexiq-parity): spinner on the asset row →
   * check mark (temporary, fades) → active badge. mismatches block the
   * session pending an explicit user decision. */
  const runLanguageScan = useCallback(async (uploaded: UploadResponse, projectId: string) => {
    setScan({ assetId: uploaded.asset_id, status: "scanning" });
    try {
      const r = await scanAssetLanguage(uploaded.asset_id);
      if (r.engine === "null") {
        setScan(null);
        addToast("warning", "language scan skipped — OCR engine not installed on the server.");
        return;
      }
      if (r.locked && r.detected_lang) {
        setSrcLang(r.detected_lang);
        addToast("success", `asset source language locked: ${langDisplayName(r.detected_lang)}.`);
        getProject(projectId).then(setProject).catch(() => {});
      }
      if (r.match === false) {
        setScan({
          assetId: uploaded.asset_id, status: "mismatch",
          detected: r.detected_lang, projectSrc: r.project_source_lang,
        });
        // State what was observed and what was chosen; nothing else. The
        // remediation lives in the mismatch card below, which offers both
        // options -- a toast that says only "change project source language"
        // presents one of them as the sole way forward, and it is wrong
        // whenever the scan is the thing that got it wrong.
        addToast(
          "error",
          `ToFU detected "${uploaded.filename}" is in ${langDisplayName(r.detected_lang ?? "?")}. ` +
          `You selected ${langDisplayName(r.project_source_lang ?? "?")} as the source language.`,
          false
        );
      } else {
        setScan({ assetId: uploaded.asset_id, status: "passed" });
        if (r.detected_lang) {
          setSrcLang(r.detected_lang);
          setShowLangConfirm(true);
        }
        setTimeout(() => {
          setScan((s) => (s?.assetId === uploaded.asset_id && s.status === "passed" ? null : s));
        }, 2600);
      }
    } catch {
      setScan(null);
      // Language scanning is advisory. Upload remains usable when OCR is
      // unavailable or a difficult image cannot be classified.
    }
  }, [addToast]);

  const doUpload = useCallback(async (file: File) => {
    if (!project) return;
    setError(null);
    setBusy("uploading");
    try {
      // guard the outgoing session: snapshot before anything is erased
      if (asset && manifest.length > 0) {
        try {
          await snapshotAsset(project.id, asset.asset_id, "pre-erase");
        } catch {
          // snapshot failure must not block the upload; autosaves remain
        }
      }
      resetSession();
      setPreviewUrl(URL.createObjectURL(file));
      const uploaded = await uploadAsset(file, project.id);
      setAsset(uploaded);
      getProject(project.id).then(setProject).catch(() => {});
      if (uploaded.asset_info.asset_type === "video") {
        const job = await createVideoJob(uploaded.asset_id, project.id);
        setVideoJob(job);
      } else {
        void runLanguageScan(uploaded, project.id);
      }
    } catch (e) {
      setErrorWithNotif(String(e));
    } finally {
      setBusy(null);
    }
  }, [project, asset, manifest.length, resetSession, rawAddToast, runLanguageScan]);

  useEffect(() => {
    if (!videoJob || ["ready", "completed", "failed", "cancelled"].includes(videoJob.status)) return;
    return watchVideoJob(videoJob.id, update => {
      setVideoJob(current => current && current.id === videoJob.id ? { ...current, ...update } : current);
      if (["ready", "completed", "failed", "cancelled"].includes(update.status || "")) getVideoJob(videoJob.id).then(setVideoJob).catch(() => {});
    }, () => getVideoJob(videoJob.id).then(setVideoJob).catch(() => {}));
  }, [videoJob?.id, videoJob?.status]);

  const proceedWithFile = useCallback((file: File) => {
    if (!project) {
      // no project yet: hold the file and go straight to project creation
      // rather than the full browser -- the drop itself is "start something
      // new", not "go find an existing project"
      setPendingUploadFile(file);
      goPantry("create");
      return;
    }
    // ongoing session? confirm before erasing capture data
    if (asset && manifest.length > 0) {
      setPendingFile(file);
    } else {
      void doUpload(file);
    }
  }, [project, asset, manifest.length, doUpload, goPantry]);

  const onFile = useCallback((file: File) => {
    void (async () => {
      try {
        const hash = await sha256File(file);
        const dup = await checkDuplicateAsset(hash);
        if (dup.duplicate && dup.project_name && dup.project_id !== project?.id) {
          setPendingDuplicateFile({ file, projectName: dup.project_name });
          return;
        }
      } catch {
        // duplicate check failing must never block an upload
      }
      proceedWithFile(file);
    })();
  }, [project, proceedWithFile]);

  // once project creation completes, upload the file that was held pending
  useEffect(() => {
    if (project && pendingUploadFile) {
      const file = pendingUploadFile;
      setPendingUploadFile(null);
      void doUpload(file);
    }
  }, [project, pendingUploadFile, doUpload]);

  /** mismatch resolution: remove the offending asset entirely */
  const onMismatchRemove = useCallback(async () => {
    if (!project || !asset) return;
    try {
      await deleteProjectAsset(project.id, asset.asset_id);
      resetSession();
      refreshProject();
      addToast("info", "asset removed. upload an asset in the project's source language.");
    } catch (e) {
      setErrorWithNotif(String(e));
    }
  }, [project, asset, resetSession, refreshProject, addToast]);

  /** mismatch resolution: the model was right — adopt the asset's language
   * as the project source ("change your project input language") */
  const onMismatchSwitch = useCallback(async () => {
    if (!project || !scan?.detected) return;
    try {
      await updateProject(project.id, { source_lang: scan.detected });
      setSrcLang(scan.detected);
      refreshProject();
      setScan((s) => (s ? { ...s, status: "passed" } : s));
      addToast("success", `project source language changed: ${langDisplayName(scan.detected)}.`);
    } catch (e) {
      setErrorWithNotif(String(e));
    }
  }, [project, scan, refreshProject, addToast]);

  /** mismatch resolution: protect — the detection was wrong, keep asset */
  const onMismatchKeep = useCallback(() => {
    setScan((s) => (s ? { ...s, status: "passed" } : s));
    addToast("info", `language override applied: ${langDisplayName(scan?.projectSrc ?? "?")}.`);
  }, [addToast, scan]);

  // streaming detection with live per-stage progress; when langs is given
  // the OCR reader charset is language-tuned, which materially improves
  // recognition of non-latin scripts (used by Re-detect after the user
  // confirms the source language)
  const runDetect = useCallback((langs?: string[]) => {
    if (!asset) return;
    setShowCapturePrompt(false);
    setShowLangConfirm(false);
    setShowLangSelect(false);
    setError(null);
    setBusy("detecting");
    setDetectStage("scanning");
    setDetectProgress("analyzing surfaces…");
    cancelDetectRef.current = detectAssetStream(asset.asset_id, langs, (ev) => {
      // ev.regions is SceneRegion[] for the scene stage and a number for the
      // counting stages; normalise to a count for pluralisation.
      const regionCount = typeof ev.regions === "number" ? ev.regions : 0;
      const regionWord = (n: number) => n <= 1 ? "region" : "regions";
      if (ev.stage === "scene") {
        setDetectProgress(
          ev.status === "running"
            ? "analyzing surfaces…"
            : `${Array.isArray(ev.regions) ? ev.regions.length : 0} candidate surface(s)`
        );
      } else if (ev.stage === "cicerone") {
        setDetectStage("drawing");
        const passLabels = ["draining", "pressing", "slicing"];
        const label = passLabels[(ev.pass ?? 1) - 1] ?? "processing";
        setDetectProgress(
          ev.status === "running"
            ? `${label}… ${ev.pass}/3`
            : `${ev.pass}/3: ${regionCount} ${regionWord(regionCount)}`
        );
      } else if (ev.stage === "finalize") {
        setDetectStage("lang");
        setDetectProgress("confirming recipe…");
      } else if (ev.stage === "refine") {
        setDetectStage("drawing");
        setDetectProgress(
          ev.status === "running"
            ? `re-pressing with ${(ev.langset ?? []).map((l) => langDisplayName(l)).join("+")}-tuned recognition…`
            : `refined: ${regionCount} ${regionWord(regionCount)}`
        );
      } else if (ev.stage === "zoom") {
        setDetectStage("drawing");
        setDetectProgress(
          ev.status === "running"
            ? "inspecting surfaces" /* zooming into surfaces (fine-grain pass...) */
            : `fine-grain: ${regionCount} ${regionWord(regionCount)}`
        );
      } else if (ev.stage === "paddle_rescue") {
        setDetectStage("drawing");
        setDetectProgress(
          ev.status === "running"
            ? "retasting to ensure flavor consistency…" /* previously: trying a second engine on weak/missed regions… */
            : `rescue pass: ${regionCount} ${regionWord(regionCount)}`
        );
      } else if (ev.stage === "polish") {
        setDetectStage("lang");
        setDetectProgress(
          ev.status === "running"
            ? "second-look recognition on weak regions…"
            : `polished ${regionCount} ${regionWord(regionCount)}`
        );
      } else if (ev.stage === "savor") {
        setDetectStage("lang");
        setDetectProgress(
          ev.status === "running"
            ? "taste-testing recognized text…"
            : (ev.corrected ?? 0) > 0 ? `savored: ${ev.corrected} correction(s)` : "tastes right"
        );
      } else if (ev.stage === "wasabi") {
        setDetectStage("lang");
        setDetectProgress(
          ev.status === "running"
            ? "checking for the right regional flavor…"
            : (ev.corrected ?? 0) > 0 ? `wasabi: ${ev.corrected} correction(s)` : "flavor's right"
        );
      } else if (ev.stage === "menu") {
        setDetectStage("lang");
        setDetectProgress(
          ev.status === "running"
            ? "checking the menu for known place names…"
            : (ev.corrected ?? 0) > 0 ? `menu: ${ev.corrected} correction(s)` : "no menu matches"
        );
      } else if (ev.stage === "enrich") {
        setDetectStage("lang");
        setDetectProgress(
          ev.status === "running"
            ? "seasoning… (style & background analysis)"
            : `enriched ${regionCount} ${regionWord(regionCount)}`
        );
      } else if (ev.stage === "memory") {
        setDetectStage("lang");
        setDetectProgress(
          ev.status === "running"
            ? "checking translation memory…"
            : (ev.matched ?? 0) > 0 ? `seen before: ${ev.matched} ${(ev.matched ?? 0) <= 1 ? "region" : "regions"}` : "no memory matches"
        );
      } else if (ev.stage === "complete" && ev.manifest) {
        const m = ev.manifest;
        const detected = langs?.[0] ?? m.src_lang;
        manifestSkipHistory.current = true;
        setManifest(m.instances);
        markRegionsNew(m.instances);
        setDismissedOcrReview(new Set());
        setImgDim(m.img_dim);
        setSceneRegions(m.scene_regions ?? []);
        setSemanticUnits(m.semantic_units ?? []);
        setSrcLang(detected);
        setBusy(null);
        setDetectStage("lang");
        const name = detected ? langDisplayName(detected) : "unknown";
        setDetectProgress(`language: ${name} (${detected ?? "?"})`);
        setTimeout(() => {
          setDetectStage("done");
          setDetectProgress("upload successful");
        }, 800);
        if (ev.engine === "null") {
          addToast(
            "error",
            "ocr engine unavailable on the server — automatic capture and text recognition are disabled. install server requirements (easyocr).",
            false
          );
        } else if (m.instances.length > 0) {
          addToast("success", `detected ${m.instances.length} ${m.instances.length <= 1 ? "region" : "regions"}`);
          if ((ev.tm_matched ?? 0) > 0) {
            addToast("info", `seen before: ${ev.tm_matched} ${(ev.tm_matched ?? 0) <= 1 ? "region" : "regions"} matched translation memory — suggestions ready in Translate.`);
          }
        } else {
          addToast("info", "no text regions detected. you can draw them manually.");
        }
        setTimeout(() => {
          setStep(1);
          setDetectProgress(null);
          setDetectStage("idle");
        }, 2000);
      } else if (ev.stage === "error") {
        setBusy(null);
        setDetectProgress(null);
        setDetectStage("idle");
        setErrorWithNotif(ev.message ?? "detection failed");
      }
    }, (message) => {
      setBusy(null);
      setDetectProgress(null);
      setDetectStage("idle");
      setErrorWithNotif(message);
    });
  }, [asset, addToast, markRegionsNew]);

  const cancelDetect = useCallback(() => {
    if (cancelDetectRef.current) {
      cancelDetectRef.current();
      cancelDetectRef.current = null;
    }
    setBusy(null);
    setDetectProgress(null);
    setDetectStage("idle");
    addToast("info", "detection cancelled.");
  }, [addToast]);

  // source-language priority: a known project source language tunes the
  // reader charset from the first pass (the server enforces this too)
  const onAutoDetect = useCallback(
    () => runDetect(srcLang ? [srcLang] : undefined),
    [runDetect, srcLang]
  );

  /** re-run detection from the Capture tab; existing regions are
   * snapshotted first because detection replaces the manifest */
  const onRerunDetect = useCallback(async () => {
    if (!asset) return;
    if (manifest.length > 0) {
      const ok = confirm(
        `Re-running detection replaces the current ${manifest.length} region(s). ` +
        "A snapshot is saved first, so you can restore from History. Continue?"
      );
      if (!ok) return;
      if (project) {
        try { await snapshotAsset(project.id, asset.asset_id, "pre-erase"); } catch {}
      }
    }
    runDetect(srcLang ? [srcLang] : undefined);
  }, [asset, manifest.length, project, runDetect, srcLang]);

  const onManualDraw = useCallback(() => {
    setShowCapturePrompt(false);
    manifestSkipHistory.current = true;
    setManifest([]);
    setStep(1);
    setDrawMode(true);
    addToast("info", "draw bounding boxes by clicking and dragging on the image.");
  }, [addToast]);

  const onAddRegion = useCallback(async (bbox: BBox) => {
    if (!asset) return;
    setDrawMode(false);
    let final = bbox;
    let finalText: string | undefined;
    try {
      const refined = await refineRegion(asset.asset_id, bbox);
      const best = refined.regions[0];
      if (best && best.bbox.width > 0 && best.bbox.height > 0) {
        final = {
          x: best.bbox.x, y: best.bbox.y,
          width: best.bbox.width, height: best.bbox.height,
        };
        finalText = best.text;
      }
    } catch {
      // Refinement is a convenience, not a prerequisite for manual capture.
      // A missing OCR engine or a hard crop must never discard the user's box.
      addToast("info", "Region added without OCR refinement.");
    }
    try {
      const inst = await addRegion(asset.asset_id, final, finalText);
      setManifest((prev) => {
        const next = [...prev, inst];
        autoSave(next);
        return next;
      });
      setSelectedId(inst.id);
    } catch (e) {
      setErrorWithNotif(String(e));
    }
  }, [asset, autoSave, addToast]);

  const onUpdateRegion = useCallback((id: string, bbox: BBox) => {
    setManifest((prev) => {
      const next = prev.map((i) => i.id === id ? { ...i, bounding_box: bbox } : i);
      autoSave(next);
      return next;
    });
  }, [autoSave]);

  const onDeleteRegion = useCallback(async (id: string) => {
    if (!asset) return;
    try {
      await deleteRegion(asset.asset_id, id);
      // the instance stays in `manifest` (marked excluded) rather than
      // being removed outright -- autoSave below PUTs this array back to
      // the server, and dropping it here would silently undo the
      // excluded flag deleteRegion() just persisted, turning the soft
      // exclude back into a hard delete on the very next autosave.
      setManifest((prev) => {
        const next = prev.map((i) => i.id === id ? { ...i, excluded: true } : i);
        autoSave(next);
        return next;
      });
      if (selectedId === id) setSelectedId(null);
    } catch (e) {
      setErrorWithNotif(String(e));
    }
  }, [asset, selectedId, autoSave]);

  const onMergeRegions = useCallback(async (ids: string[]) => {
    if (!asset || ids.length < 2) return;
    setMergeLoading(true);
    try {
      const res = await mergeRegions(asset.asset_id, ids);
      // Same shape as onDeleteRegion: the absorbed regions stay in the
      // array marked excluded rather than being spliced out, or the next
      // autosave PUT would undo the excluded flag the server just wrote.
      setManifest((prev) => {
        const merged = new Set(res.merged_ids);
        const next = prev.map((i) =>
          i.id === res.region.id ? { ...i, ...res.region }
          : merged.has(i.id) ? { ...i, excluded: true }
          : i
        );
        autoSave(next);
        return next;
      });
      setSelectedId(res.region.id);
      addToast("success", res.source === "reread"
        ? `Merged ${ids.length} regions and re-read them as one`
        : `Merged ${ids.length} regions — kept the existing text, the re-read did not improve on it`);
    } catch (e) {
      setErrorWithNotif(String(e));
    } finally {
      setMergeLoading(false);
    }
  }, [asset, autoSave, addToast, setErrorWithNotif]);

  const onReorder = useCallback((fromId: string, toId: string) => {
    // Always start from the list the user can currently see.  A cached manual
    // list may predate detection, exclusion, or an imported manifest and omit
    // an otherwise draggable rN row.
    const base = visibleManifest.map((i) => i.id);
    const fromIdx = base.indexOf(fromId);
    const toIdx = base.indexOf(toId);
    if (fromIdx === -1 || toIdx === -1 || fromIdx === toIdx) return;
    const next = [...base];
    next.splice(fromIdx, 1);
    next.splice(toIdx, 0, fromId);
    // Keep hidden/excluded regions in the saved manifest as well.  They are
    // placed after the visible Text Manifest rows so every region retains a
    // unique, stable order for round-trip exports.
    const allIds = [...next, ...orderedManifest.filter((inst) => inst.excluded).map((inst) => inst.id)];
    const rank = new Map(allIds.map((id, index) => [id, index]));
    setManualOrder(next);
    setManifest((prev) => {
      const updated = prev.map((inst) => {
        const reading_order = rank.get(inst.id);
        return reading_order === undefined ? inst : { ...inst, reading_order };
      });
      autoSave(updated);
      return updated;
    });
  }, [visibleManifest, orderedManifest, autoSave, setManifest]);

  const onTextChange = useCallback((id: string, text: string) => {
    setManifest((prev) => {
      const next = prev.map((i) => i.id === id ? { ...i, text } : i);
      autoSave(next);
      if (importedHash) setHasEditsAfterImport(true);
      return next;
    });
  }, [autoSave, importedHash]);

  const onTargetChange = useCallback((id: string, target: string) => {
    setManifest((prev) => {
      const next = prev.map((i) => i.id === id ? { ...i, target_text: target, target_language: i.target_language ?? targLang } : i);
      autoSave(next);
      if (importedHash) setHasEditsAfterImport(true);
      return next;
    });
  }, [autoSave, importedHash, targLang]);

  const onSemanticDraftChange = useCallback((unitId: string, text: string) => {
    setSemanticDrafts((previous) => ({ ...previous, [unitId]: text }));
    // A changed phrase invalidates only that unit's prior plan. The existing
    // per-region targets remain untouched until an explicit Apply.
    setSemanticPlans((previous) => {
      if (!(unitId in previous)) return previous;
      const { [unitId]: _discarded, ...rest } = previous;
      return rest;
    });
  }, []);

  // Basil colours its plating field per source region and checks the entry
  // against the source manifest, so it needs both the regions themselves and
  // the Latin strings Cicerone actually attested in this asset.
  const regionsById = useMemo(
    () => Object.fromEntries(manifest.map((inst) => [inst.id, inst])),
    [manifest],
  );
  const attestedLatin = useMemo(() => attestedFromManifest(manifest), [manifest]);

  const onSemanticRepair = useCallback(async (unitId: string, accepted: boolean) => {
    if (!asset) return;
    setSemanticBusyId(unitId);
    try {
      const result = await semanticRepair(asset.asset_id, unitId, accepted);
      setSemanticUnits(result.manifest.semantic_units ?? []);
      addToast("info", accepted ? "source reading corrected; regions untouched" : "proposal rejected; the original reading stands");
    } catch (error) {
      addToast("error", `could not record that decision: ${error}`);
    } finally {
      setSemanticBusyId(null);
    }
  }, [asset, addToast]);

  const onSemanticPlan = useCallback(async (unitId: string, apply: boolean) => {
    if (!asset) return;
    const unit = semanticUnits.find((item) => item.id === unitId);
    if (!unit) return;
    const target = (semanticDrafts[unitId] ?? unit.substitution?.target_text ?? unit.suggestion?.target_text ?? "").trim();
    if (!target) {
      addToast("warning", "enter a complete target phrase before planning placement");
      return;
    }
    setSemanticBusyId(unitId);
    try {
      const result = await semanticSubstitution(asset.asset_id, unitId, target, targLang, apply);
      setSemanticUnits(result.manifest.semantic_units ?? []);
      setSemanticPlans((previous) => ({ ...previous, [unitId]: result.plan }));
      if (result.applied) {
        // This is a normal manifest edit, so the existing Capture/Localized
        // undo machinery can still return to the pre-substitution text.
        setManifest(result.manifest.instances);
        setImgDim(result.manifest.img_dim);
        setSceneRegions(result.manifest.scene_regions ?? []);
        setSemanticDrafts((previous) => ({ ...previous, [unitId]: target }));
        if (importedHash) setHasEditsAfterImport(true);
      } else if (result.plan.review_required) {
        addToast("warning", "Basil kept this phrase in review: no verified alignment evidence was available.");
      } else {
        addToast("info", "plate ready; review cube placement, then apply."); /* prev.: placement plan is ready; review the spatial anchors, then apply it. */
      }
    } catch (error) {
      addToast("error", `plating failed: ${error}`); /* prev.: translation substitut */
    } finally {
      setSemanticBusyId(null);
    }
  }, [asset, semanticUnits, semanticDrafts, targLang, importedHash, addToast, setManifest]);

  const onGlossaryUpload = useCallback(async (file: File, scope: "global" | "project", mode: "auxiliary" | "merge" | "replace") => {
    setGlossaryUploading(true);
    setGlossaryUploadError(null);
    setGlossaryUploadStep("marinate");
    try {
      await new Promise((resolve) => setTimeout(resolve, 150));
      setGlossaryUploadStep("ferment");
      await uploadGlossary(file, scope, mode, project?.id, srcLang ?? undefined, targLang ?? undefined);
      setGlossaryUploadStep("set");
      setGlossaryStatus(await fetchGlossaryStatus(project?.id));
      setGlossaryUploadStep("done");
      addToast("success", `Basil glossary set from ${file.name}`);
      setTimeout(() => setGlossaryUploadStep(null), 1800);
    } catch (error) {
      const message = String(error);
      setGlossaryUploadError(message);
      setGlossaryUploadStep(null);
      addToast("error", `Basil glossary upload failed: ${message}`);
    } finally {
      setGlossaryUploading(false);
    }
  }, [project?.id, srcLang, targLang, addToast]);

  const onGlossaryDelete = useCallback(async (scope: "global" | "project") => {
    try {
      await deleteGlossary(scope, project?.id);
      setGlossaryStatus(await fetchGlossaryStatus(project?.id));
      addToast("success", `${scope} Basil glossary cleared`);
    } catch (error) {
      addToast("error", `Basil glossary could not be cleared: ${error}`);
    }
  }, [project?.id, addToast]);

  const onOcr = useCallback(async (id: string) => {
    if (!asset) return;
    const inst = manifest.find((i) => i.id === id);
    if (!inst) return;
    setOcrLoading(id);
    try {
      const result = await ocrRegion(asset.asset_id, inst.bounding_box);
      if (result.engine_missing) {
        addToast("error", "ocr engine unavailable on the server — install server requirements (easyocr) to enable text recognition.", false);
        return;
      }
      setManifest((prev) => {
        const next = prev.map((i) => i.id === id ? {
          ...i, text: result.text, confidence: result.confidence,
          detected_language: result.detected_language,
        } : i);
        autoSave(next);
        return next;
      });
      if (result.text) {
        addToast("success", `ocr: "${result.text}" (${(result.confidence * 100).toFixed(0)}%)`);
      } else {
        addToast("warning", "no text found in this region. try adjusting the box or drawing it manually.");
      }
    } catch (e) {
      addToast("error", `ocr failed: ${e}`);
    } finally {
      setOcrLoading(null);
    }
  }, [asset, manifest, autoSave, addToast]);

  const onToggleDnt = useCallback((id: string) => {
    setManifest((prev) => {
      const next = prev.map((i) => i.id === id ? { ...i, dnt: !i.dnt } : i);
      autoSave(next);
      return next;
    });
  }, [autoSave]);

  const onTargetLangChange = useCallback((id: string, lang: string) => {
    setManifest((prev) => {
      const next = prev.map((i) => i.id === id ? { ...i, target_language: lang } : i);
      autoSave(next);
      return next;
    });
  }, [autoSave]);

  const onSrcLangChange = useCallback((id: string, lang: string) => {
    setManifest((prev) => {
      const next = prev.map((i) => i.id === id ? { ...i, language: lang } : i);
      autoSave(next);
      return next;
    });
  }, [autoSave]);

  const toggleLangLock = useCallback((id: string) => {
    setLockedLangs((prev) => {
      const next = new Set(prev);
      if (next.has(id)) next.delete(id);
      else next.add(id);
      return next;
    });
  }, []);

  const onFontChange = useCallback((id: string, fontPath: string) => {
    setManifest((prev) => {
      const next = prev.map((i) => i.id === id ? withStyle(i, { font_family: fontPath }) : i);
      autoSave(next);
      return next;
    });
  }, [autoSave]);

  const onApplyTargetLang = useCallback((ids: string[], lang: string) => {
    const idSet = new Set(ids);
    setManifest((prev) => {
      const next = prev.map((i) => idSet.has(i.id) ? { ...i, target_language: lang } : i);
      autoSave(next);
      return next;
    });
    addToast("success", `applied ${langDisplayName(lang)} to ${ids.length} region(s)`);
  }, [autoSave, addToast]);

  const onOrientationToggle = useCallback((id: string) => {
    setManifest((prev) => {
      const next = prev.map((i) => {
        if (i.id !== id) return i;
        const current = i.style_profile?.target_orientation ?? "horizontal";
        return withStyle(i, { target_orientation: current === "vertical" ? "horizontal" : "vertical" });
      });
      autoSave(next);
      return next;
    });
  }, [autoSave]);

  const onWordOrderToggle = useCallback((id: string) => {
    setManifest((prev) => {
      const next = prev.map((i) => {
        if (i.id !== id) return i;
        const current = i.style_profile?.word_order ?? null;
        return withStyle(i, { word_order: current === "rtl" ? null : "rtl" });
      });
      autoSave(next);
      return next;
    });
  }, [autoSave]);

  const onImported = useCallback((result: ImportResult) => {
    if (asset) {
      getManifest(asset.asset_id).then((m) => {
        manifestSkipHistory.current = true;
        setManifest(m.instances);
        setImgDim(m.img_dim);
        setSceneRegions(m.scene_regions ?? []);
        setSemanticUnits(m.semantic_units ?? []);
        setImportedHash(JSON.stringify(m.instances.map((i) => i.target_text)));
        setHasEditsAfterImport(false);
      });
    }
    const methods = result.matched_by
      ? Object.entries(result.matched_by).filter(([, count]) => count > 0).map(([method, count]) => `${count} by ${method}`).join(", ")
      : "";
    if ((result.unresolved?.length ?? 0) > 0) {
      addToast("warning", `imported ${result.imported} ${result.format?.toUpperCase() ?? ""} segment(s) (${methods || "ID mapping"}); ${result.unresolved!.length} ambiguous segment(s) were left unchanged.`);
      return;
    }
    if (result.missing.length > 0) {
      addToast("warning", `imported ${result.imported} segment(s)${methods ? ` (${methods})` : ""}. Missing: ${result.missing.join(", ")}`);
    } else if (methods) {
      addToast("success", `imported ${result.imported} translation(s) (${methods}) — targets populated below`);
    } else {
      addToast("success", `imported ${result.imported} translations — targets populated below`);
    }
  }, [asset, addToast]);

  const onFontMatch = useCallback(async (id: string, allowExternal = false) => {
    if (!asset) return;
    if (allowExternal) {
      const approved = window.confirm(
        "Search a commercial font catalog? ToFU will send cropped source-text images to the configured catalog provider. " +
        "Your selected font will not change automatically."
      );
      if (!approved) return;
    }
    setFontMatchingId(id);
    try {
      const result = await matchFonts(asset.asset_id, allowExternal);
      manifestSkipHistory.current = true;
      setManifest(result.manifest.instances);
      setImgDim(result.manifest.img_dim);
      setSceneRegions(result.manifest.scene_regions ?? []);
      setSemanticUnits(result.manifest.semantic_units ?? []);
      if (allowExternal && result.external_matched === 0) {
        addToast("warning", "catalog matching is not configured or returned no candidates; local font evidence is still available.");
      } else {
        addToast("success", `${result.local_matched} region(s) compared against installed glyph shapes${allowExternal ? `; ${result.external_matched} catalog result(s) added` : ""}.`);
      }
    } catch (e) {
      addToast("error", `font analysis failed: ${e}`);
    } finally {
      setFontMatchingId(null);
    }
  }, [asset, addToast]);

  const onImportTranslation = useCallback(async (file: File) => {
    if (!asset) return;
    setImportLoading(true);
    try {
      const result = await importFile(asset.asset_id, file);
      onImported(result);
    } catch (e) {
      addToast("error", `import failed: ${e}`);
    } finally {
      setImportLoading(false);
    }
  }, [asset, onImported, addToast]);

  const onTargetLangSelect = useCallback((code: string) => {
    if (code === targLang) return;
    const hasTranslations = manifest.some((i) => i.target_text && i.target_text.trim());
    if (hasTranslations) {
      setFormerTargLang(targLang);
    }
    setTargLang(code);
    if (project) {
      updateProject(project.id, { target_lang: code })
        .then((p) => setProject((prev) => prev ? { ...prev, ...p, assets: prev.assets, active_asset: prev.active_asset } : p))
        .catch(() => {});
    }
  }, [targLang, manifest, project]);

  useEffect(() => {
    if (step === 2 && formerTargLang) {
      setShowLangChangePopup(true);
    }
  }, [step, formerTargLang]);

  const onLangChangeProceed = useCallback(() => {
    setShowLangChangePopup(false);
  }, []);

  const onLangChangeClear = useCallback(() => {
    setManifest((prev) => {
      const next = prev.map((i) =>
        i.target_language === formerTargLang ? { ...i, target_text: null } : i
      );
      autoSave(next);
      return next;
    });
    setFormerTargLang(null);
    setShowLangChangePopup(false);
  }, [formerTargLang, autoSave]);

  /** project hero: remove an asset from the project (CRUD delete). the
   * snapshot ledger and file on disk survive. */
  const onDeleteAsset = useCallback(async (assetId: string, filename: string | null) => {
    if (!project) return;
    setPendingAssetDelete({ assetId, filename });
  }, [project]);

  const confirmDeleteAsset = useCallback(async () => {
    if (!project || !pendingAssetDelete) return;
    const { assetId, filename } = pendingAssetDelete;
    try {
      await deleteProjectAsset(project.id, assetId);
      if (asset?.asset_id === assetId) resetSession();
      refreshProject();
      addToast("info", `removed ${filename ?? assetId} from the project.`);
    } catch (e) {
      setErrorWithNotif(String(e));
    }
  }, [project, pendingAssetDelete, asset, resetSession, refreshProject, addToast]);

  const onHistoryRestored = useCallback(() => {
    setShowHistory(false);
    if (!project) return;
    resetSession();
    loadProjectSession(project.id)
      .then(() => addToast("success", "snapshot restored"))
      .catch((e) => setErrorWithNotif(String(e)));
  }, [project, resetSession, loadProjectSession, addToast]);

  // lock all source languages when entering capture with detected regions
  useEffect(() => {
    if (step === 1 && manifest.length > 0 && lockedLangs.size === 0) {
      setLockedLangs(new Set(manifest.map((r) => r.id)));
    }
  }, [step, manifest, lockedLangs.size]);

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (step !== 1 || screen !== "main") return;
      const t = e.target as HTMLElement | null;
      if (t && (t.tagName === "INPUT" || t.tagName === "TEXTAREA")) return;
      if (e.key === "a" || e.key === "A") { setDrawMode((d) => !d); }
      if (e.key === "Delete" && selectedId) { onDeleteRegion(selectedId); }
      if (e.key === "Escape") { setDrawMode(false); setSelectedId(null); }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [step, screen, selectedId, onDeleteRegion]);

  // Style toggle shortcuts (step 3, region selected)
  useEffect(() => {
    if (step !== 3) return;
    const onKey = (e: KeyboardEvent) => {
      const selected = renderSelId ?? prevSelId;
      if (!selected) return;
      const t = e.target as HTMLElement | null;
      if (t && (t.tagName === "INPUT" || t.tagName === "TEXTAREA" || t.isContentEditable)) return;
      if (!(e.ctrlKey || e.metaKey)) return;
      const inst = manifest.find((i) => i.id === selected);
      if (!inst) return;
      const sp = inst.style_profile;
      if (e.key === "b" || e.key === "B") {
        e.preventDefault();
        const isBold = (sp?.font_weight ?? "").toLowerCase().includes("bold");
        updateSelectedStyle({ font_weight: isBold ? null : "Bold" });
      } else if (e.key === "i" || e.key === "I") {
        e.preventDefault();
        updateSelectedStyle({ italic: !sp?.italic ? true : null });
      } else if (e.key === "u" || e.key === "U") {
        e.preventDefault();
        updateSelectedStyle({ underline: !sp?.underline ? true : null });
      } else if (e.key === "+" || e.key === "=") {
        e.preventDefault();
        if (e.shiftKey) {
          updateSelectedStyle({ superscript: !sp?.superscript ? true : null });
        } else {
          updateSelectedStyle({ subscript: !sp?.subscript ? true : null });
        }
      }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [step, renderSelId, prevSelId, manifest, updateSelectedStyle]);

  useEffect(() => {
    if (!menuOpen) return;
    const onDown = (e: MouseEvent) => {
      if (menuRef.current && !menuRef.current.contains(e.target as Node)) {
        setMenuOpen(false);
      }
    };
    document.addEventListener("mousedown", onDown);
    return () => document.removeEventListener("mousedown", onDown);
  }, [menuOpen]);

  useEffect(() => {
    if (!warpOpen) return;
    const onDown = (e: MouseEvent) => {
      if (warpRef.current && !warpRef.current.contains(e.target as Node)) {
        setWarpOpen(false);
      }
    };
    document.addEventListener("mousedown", onDown);
    return () => document.removeEventListener("mousedown", onDown);
  }, [warpOpen]);

  useEffect(() => {
    if (!garnishScopeOpen) return;
    const onDown = (e: MouseEvent) => {
      if (garnishScopeRef.current && !garnishScopeRef.current.contains(e.target as Node)) {
        setGarnishScopeOpen(false);
      }
    };
    document.addEventListener("mousedown", onDown);
    return () => document.removeEventListener("mousedown", onDown);
  }, [garnishScopeOpen]);


  useEffect(() => {
    if (hasEditsAfterImport) {
      addToast("warning", "unsaved changes: live edits differ from imported file. re-export recommended.", false);
    }
  }, [hasEditsAfterImport, addToast]);

  useEffect(() => {
    setNameTyped(0);
    setNameDone(false);
    if (!project) return;
    const name = project.name;
    const timers: ReturnType<typeof setTimeout>[] = [];
    for (let i = 1; i <= name.length; i++) {
      timers.push(setTimeout(() => setNameTyped(i), 200 + i * 80));
    }
    timers.push(setTimeout(() => setNameDone(true), 200 + name.length * 80 + 200));
    return () => timers.forEach(clearTimeout);
  }, [project]);

  const onValidate = useCallback(async () => {
    if (!asset) return;
    setError(null);
    setBusy("validating");
    try {
      setReport(await validateAsset(asset.asset_id, targLang));
    } catch (e) {
      setErrorWithNotif(String(e));
    } finally {
      setBusy(null);
    }
  }, [asset, targLang]);

  const onRender = useCallback(async () => {
    if (!asset) return;
    if (repairReviews.length) {
      const ok = confirm("One or more repairs are provisional. Review the localized canvas or confirm that this render should use the shown candidate.");
      if (!ok) return;
    }
    if (repairFallbackIds.length) {
      const ok = confirm(`No local neural inpainting provider is configured for ${repairFallbackIds.length} textured region(s). Render with the editable deterministic fallback, or cancel to configure a provider first?`);
      if (!ok) return;
    }
    if (hasEditsAfterImport) {
      const ok = confirm("discrepancy detected: live edits differ from the imported file. proceed with current state?");
      if (!ok) return;
    }
    setError(null);
    setBusy("rendering");
    try {
      // Final Render deliberately commits the same manifest snapshot that
      // generated the localized preview.  This closes the autosave-debounce
      // gap that could otherwise make final pixels diverge from the canvas.
      await flushCurrentManifest();
      // per-region fonts (style_profile.font_family) drive scribe now;
      // no request-level font override
      const result = await renderAsset(asset.asset_id, targLang);
      setRenderResult(result);
      setApproved(false);
      addToast("success", "render complete");
    } catch (e) {
      setErrorWithNotif(String(e));
    } finally {
      setBusy(null);
    }
  }, [asset, targLang, hasEditsAfterImport, repairReviews.length, repairFallbackIds.length, addToast, flushCurrentManifest]);

  const renderResultFromEvent = (ev: RenderStreamEvent): RenderResult => ({
    output_url: ev.output_url ?? null,
    verification_report: ev.verification_report ?? null,
    qa_report: ev.qa_report ?? null,
    qa_passed: !!ev.qa_passed,
    qa_threshold: ev.qa_threshold ?? 0.8,
    validation_report: ev.validation_report ?? null,
    text_manifest: ev.text_manifest ?? null,
    logs: ev.logs ?? [],
    errors: ev.errors ?? [],
  });

  const STAGE_LABEL: Record<string, string> = {
    tofu: "pre-flight check…",
    scene: "re-reading the recipe…", /* re-reading scene context… */
    tofu_regions: "validating per-region fit…",
    cleanse: "rinsing…", /* erasing source text… */
    scribe: "seasoning…", /* rendering target text... */
    verify: "scoring quality…", /* scoring quality */
    save: "saving output…",
  };

  /** the QA Inspector's streamed render: same pipeline as onRender, but
   * over SSE so per-stage progress and recommendations surface live
   * instead of behind one spinner */
  const runVerifyRender = useCallback(async () => {
    if (!asset) return;
    setError(null);
    setApproved(false);
    setVerifyBusy("rendering");
    setVerifyStage("tofu");
    try {
      await flushCurrentManifest();
    } catch (error) {
      setVerifyBusy(null);
      setVerifyStage(null);
      setErrorWithNotif(String(error));
      return;
    }
    cancelVerifyRef.current = renderAssetStream(asset.asset_id, targLang, (ev) => {
      if (ev.stage === "complete") {
        cancelVerifyRef.current = null;
        setVerifyBusy(null);
        setVerifyStage(null);
        const result = renderResultFromEvent(ev);
        setRenderResult(result);
        if (result.text_manifest) { manifestSkipHistory.current = true; setManifest(result.text_manifest.instances); }
        const recs = result.qa_report?.recommendations ?? [];
        recs.slice(0, 3).forEach((r) => addToast("warning", r, false));
        if (result.errors.length > 0) {
          result.errors.forEach((e) => addToast("error", e, false));
        } else if (recs.length === 0) {
          addToast("success", "verification complete — no issues flagged");
        }
      } else if (ev.stage === "error") {
        cancelVerifyRef.current = null;
        setVerifyBusy(null);
        setVerifyStage(null);
        setErrorWithNotif(ev.message ?? "render failed");
      } else {
        setVerifyStage(ev.stage);
      }
    }, (message) => {
      cancelVerifyRef.current = null;
      setVerifyBusy(null);
      setVerifyStage(null);
      setErrorWithNotif(message);
    });
  }, [asset, targLang, addToast, flushCurrentManifest]);

  const cancelVerify = useCallback(() => {
    if (cancelVerifyRef.current) {
      cancelVerifyRef.current();
      cancelVerifyRef.current = null;
    }
    setVerifyBusy(null);
    setVerifyStage(null);
    addToast("info", "verification cancelled.");
  }, [addToast]);

  /** per-region re-render: composites onto the existing output instead of
   * reverting untouched regions to source text (server: region_ids) */
  const onReRenderRegion = useCallback(async (id: string) => {
    if (!asset) return;
    setReRenderingId(id);
    setApproved(false);
    try {
      await flushCurrentManifest();
    } catch (error) {
      setReRenderingId(null);
      setErrorWithNotif(String(error));
      return;
    }
    renderAssetStream(asset.asset_id, targLang, (ev) => {
      if (ev.stage === "complete") {
        setReRenderingId(null);
        const result = renderResultFromEvent(ev);
        setRenderResult(result);
        if (result.text_manifest) { manifestSkipHistory.current = true; setManifest(result.text_manifest.instances); }
        addToast("success", `region ${id} re-rendered`);
      } else if (ev.stage === "error") {
        setReRenderingId(null);
        setErrorWithNotif(ev.message ?? "re-render failed");
      }
    }, (message) => {
      setReRenderingId(null);
      setErrorWithNotif(message);
    }, { regionIds: [id] });
  }, [asset, targLang, addToast, flushCurrentManifest]);

  const onApprove = useCallback(async () => {
    if (!asset || !renderResult?.qa_report) return;
    try {
      await approveRender(asset.asset_id, targLang, {
        overall_score: renderResult.qa_report.overall_score,
        regions_total: renderResult.qa_report.progress?.regions_total,
        rendered: renderResult.qa_report.progress?.rendered,
        dnt: renderResult.qa_report.progress?.dnt,
      });
      setApproved(true);
      addToast("success", "QA sign-off recorded in project history");
      refreshProject();
    } catch (e) {
      setErrorWithNotif(String(e));
    }
  }, [asset, targLang, renderResult, addToast, refreshProject]);

  const translatableCount = manifest.filter((i) => !i.dnt).length;
  const translatedCount = manifest.filter((i) => !i.dnt && i.target_text).length;
  const scanBlocked = scan !== null && scan.status !== "passed";

  const saveIndicator = (
    <span className={`subtext text-xs ${saveStatus === "saving" ? "text-amber-500 dark:text-amber-400" : saveStatus === "saved" ? "text-[#0f2600] dark:text-[#4f9f00]" : "text-zinc-500"}`}>
      {saveStatus === "saving" ? "saving…" : saveStatus === "saved" ? "saved" : ""}
    </span>
  );

  // --- screens ------------------------------------------------------------

  if (displayedScreen === "splash") {
    return (
      <>
        <div className={`screen-fade screen-fade-fixed${leaving ? " leaving" : ""}`}>
          <SplashScreen onDone={handleSplashDone} />
        </div>
        <ThemeToggle theme={theme} onToggle={toggleTheme} />
      </>
    );
  }

  if (displayedScreen === "title") {
    return (
      <>
        <div className={`screen-fade screen-fade-fixed${leaving ? " leaving" : ""}`}>
          <TitleScreen
            onEnter={() => {
              prevScreen.current = "title";
              resetSession();
              setProject(null);
              localStorage.removeItem(PROJECT_KEY);
              setScreen("main");
            }}
            onSelectProject={openProject}
            onCreateProject={() => goPantry("create")}
            theme={theme}
            onToggleTheme={toggleTheme}
            currentProjectId={project?.id ?? null}
            onProjectRenamed={setProject}
          />
        </div>
      </>
    );
  }

  return (
    <div className={`screen-fade${leaving ? " leaving" : ""}`}>
    <div className="mx-auto max-w-7xl space-y-6 p-8 pb-12">
      <header className="relative z-200 flex items-center gap-4">
        <img
          src={logoSrc(theme)}
          alt="ToFU"
          className="h-[96px] w-auto cursor-pointer transition hover:opacity-80"
          onClick={() => setShowTitleConfirm(true)}
          title="Return to title"
        />
        <span className="title-typewriter text-zinc-500 dark:text-zinc-400" style={{ fontSize: "0.75rem" }}>v0.1.0</span>
        <div className="flex-1" />
        <div className="bezier-card flex items-center rounded-lg bg-white/60 px-3 py-2 dark:bg-zinc-900/60">
          <span style={{ transform: "scale(0.75)", transformOrigin: "center", display: "inline-block" }}>
            <ThemeToggle theme={theme} onToggle={toggleTheme} inline />
          </span>
        </div>
        {project && (
          <div className="bezier-card flex items-center gap-2 rounded-lg bg-white/60 px-3 py-2 dark:bg-zinc-900/60">
            <Box size={14} className="text-cyan-600 dark:text-cyan-400" />
            <span className="relative inline-block max-w-[220px] truncate text-sm font-medium text-zinc-800 dark:text-zinc-200">
              <span style={{ visibility: "hidden" }}>{project.name}</span>
              <span style={{ position: "absolute", left: 0, top: 0, whiteSpace: "pre" }}>
                {project.name.slice(0, nameTyped)}
                {!nameDone && <span className="name-typewriter-block" />}
              </span>
            </span>
            <span className="subtext text-xs text-zinc-500">
              {srcLang ?? "auto"} → {targLang}
            </span>
          </div>
        )}
        <NotificationBell notifications={notifications} onClear={clearNotifications} onDismiss={dismissNotification} />
        <button onClick={() => setShowCapabilities(true)} title="System capabilities" className="bezier-card rounded-lg bg-white/60 p-2 text-zinc-600 hover:bg-zinc-100 dark:bg-zinc-900/60 dark:text-zinc-300 dark:hover:bg-zinc-800"><Cpu size={16} /></button>
        <div className="relative" ref={menuRef}>
          <button
            onClick={() => setMenuOpen((v) => !v)}
            className="bezier-card flex items-center gap-2 rounded-lg bg-white/60 px-3 py-2 text-sm text-zinc-700 transition hover:bg-zinc-100 dark:bg-zinc-900/60 dark:text-zinc-300 dark:hover:bg-zinc-800"
          >
            <Home size={14} />
            <ChevronDown size={14} className={`transition ${menuOpen ? "rotate-180" : ""}`} />
          </button>
          {/* Dropdown Morph — reusable expand/collapse animation (see .dropdown-morph in uikit.css) */}
          <div className={`dropdown-morph bezier-card absolute right-0 top-full z-200 mt-2 w-44 rounded-lg bg-white p-1.5 dark:bg-zinc-900${menuOpen ? " expanded" : ""}`}
               style={menuOpen ? { boxShadow: "4px 4px 0 var(--bc-shadow), 8px 8px 16px rgba(0,0,0,0.18)" } : undefined}>
              <button
                onClick={() => { setMenuOpen(false); goPantry("create"); }}
                className="flex w-full items-center gap-2 rounded-md px-3 py-2 text-sm text-zinc-700 transition hover:bg-zinc-100 dark:text-zinc-300 dark:hover:bg-zinc-800"
              >
                <TbCubePlus size={14} className="text-cyan-600 dark:text-cyan-400" />
                new project
              </button>
              <button
                onClick={() => { setMenuOpen(false); goPantry("pantry"); }}
                className="flex w-full items-center gap-2 rounded-md px-3 py-2 text-sm text-zinc-700 transition hover:bg-zinc-100 dark:text-zinc-300 dark:hover:bg-zinc-800"
              >
                <FaBoxOpen size={14} className="text-cyan-600 dark:text-cyan-400" />
                my pantry
              </button>
              <button
                onClick={() => { setMenuOpen(false); setShowHistory(true); }}
                disabled={!project}
                className="flex w-full items-center gap-2 rounded-md px-3 py-2 text-sm text-zinc-700 transition hover:bg-zinc-100 disabled:opacity-40 dark:text-zinc-300 dark:hover:bg-zinc-800"
              >
                <SquareStack size={14} className="text-cyan-600 dark:text-cyan-400" />
                history
              </button>
              <button
                onClick={() => { setMenuOpen(false); setShowMemory(true); }}
                disabled={!project}
                className="flex w-full items-center gap-2 rounded-md px-3 py-2 text-sm text-zinc-700 transition hover:bg-zinc-100 disabled:opacity-40 dark:text-zinc-300 dark:hover:bg-zinc-800"
              >
                <BookmarkCheck size={14} className="text-cyan-600 dark:text-cyan-400" />
                memory
              </button>
              <button
                onClick={() => { setMenuOpen(false); setShowSettings(true); }}
                className="flex w-full items-center gap-2 rounded-md px-3 py-2 text-sm text-zinc-700 transition hover:bg-zinc-100 dark:text-zinc-300 dark:hover:bg-zinc-800"
              >
                <Hexagon size={14} className="text-cyan-600 dark:text-cyan-400" />
                settings
              </button>
          </div>
        </div>
      </header>

      <div className="flex justify-center">
        <Stepper
          current={step}
          onStep={setStep}
          canCapture={!!asset && asset.asset_info.asset_type !== "video" && !scanBlocked}
          canTranslate={!!asset && asset.asset_info.asset_type !== "video" && manifest.length > 0 && !scanBlocked}
          canRender={asset?.asset_info.asset_type !== "video" && translatedCount > 0 && !scanBlocked}
          canVerify={asset?.asset_info.asset_type !== "video" && !!renderResult && !scanBlocked}
          theme={theme}
          flagStep={formerTargLang ? 2 : null}
        />
      </div>

      {error && (
        <div className="subtext rounded-lg border border-red-300 bg-red-100 px-4 py-2 text-sm text-red-700 dark:border-red-800 dark:bg-red-950/50 dark:text-red-300">
          {error}
        </div>
      )}

      {/* STEP 0: Upload — asset intake + project overview */}
      {step === 0 && asset?.asset_info.asset_type === "video" && videoJob && previewUrl && (
        <div key="video-workspace" className="step-fade">
          <Section title="Video localization workspace" className={stackClass(0)}>
            <div className="mb-3 flex items-center justify-between gap-3">
              <div><p className="text-sm font-medium">{asset.filename}</p><p className="subtext text-xs">{videoJob.stage} · {Math.round(videoJob.progress * 100)}% · {videoJob.status}</p></div>
              <button onClick={() => project && onDeleteAsset(asset.asset_id, asset.filename)} className="rounded-md border border-zinc-300 px-3 py-1.5 text-xs dark:border-zinc-700">Replace video</button>
            </div>
            <VideoWorkspace job={videoJob} videoUrl={videoJob.proxy_url || previewUrl}
              onCancel={async () => setVideoJob(await cancelVideoJob(videoJob.id))}
              onResume={async () => setVideoJob(await resumeVideoJob(videoJob.id))}
              onUpgrade={async () => setVideoJob(await upgradeVideoJob(videoJob.id))}
              onTrackUpdate={async (trackId, changes) => { await updateVideoTrack(videoJob.id, trackId, changes); }}
              onPreview={async (start, end) => setVideoJob(await renderVideoPreview(videoJob.id, start, end))}
              onExport={async () => setVideoJob(await exportVideo(videoJob.id))}
              onKeyframe={async (key) => { await putVideoKeyframe(videoJob.id, {...key, expected_job_revision: videoJob.dependency_revision}); setVideoJob(await getVideoJob(videoJob.id)); }} />
          </Section>
        </div>
      )}
      {step === 0 && asset?.asset_info.asset_type !== "video" && (
        <div key="step-0" className="step-fade grid gap-6 md:grid-cols-[minmax(0,1fr)_360px]">
          <Section title="Asset" className={stackClass(0)}>
            <label
              onDragOver={(e) => { e.preventDefault(); setDragOverAsset(true); }}
              onDragEnter={(e) => { e.preventDefault(); setDragOverAsset(true); }}
              // only leave when the pointer left the ZONE, not when it crossed
              // onto the preview image or the caption inside it
              onDragLeave={(e) => { if (e.currentTarget === e.target) setDragOverAsset(false); }}
              onDrop={(e) => {
                e.preventDefault();
                setDragOverAsset(false);
                const file = e.dataTransfer.files?.[0];
                if (!file) return;
                const wantsVideo = project?.asset_kind === "video";
                const isVideo = file.type.startsWith("video/");
                // The click path filters by `accept`; a drop bypasses it
                // entirely, so the same rule has to be enforced here or a
                // video lands in an image project and fails downstream.
                if (wantsVideo !== isVideo) {
                  addToast("error", `this project takes ${wantsVideo ? "video" : "image"} assets; "${file.name}" is ${isVideo ? "a video" : "not a video"}.`);
                  return;
                }
                onFile(file);
              }}
              className={`flex min-h-64 cursor-pointer flex-col items-center justify-center gap-2 rounded-lg border-2 border-dashed p-4 transition dark:hover:border-zinc-500 dark:hover:text-zinc-300 ${
                dragOverAsset
                  ? "border-cyan-500 bg-cyan-50 text-cyan-700 dark:bg-cyan-950/30 dark:text-cyan-300"
                  : "border-zinc-400 text-zinc-500 hover:border-zinc-600 hover:text-zinc-700 dark:border-zinc-700"
              }`}
            >
              {previewUrl && asset?.asset_info.asset_type !== "video" ? (
                <img src={previewUrl} alt="preview" className="max-h-72 rounded-sm object-contain" />
              ) : (
                <>
                  <ArrowUpFromLine size={28} />
                  <span className="text-sm">
                    {dragOverAsset ? "release to upload" : "drop or click to upload asset"}
                  </span>
                  {/* <span className="subtext text-xs text-zinc-500 dark:text-zinc-600">uploading a new image starts a fresh capture session</span> */}
                </>
              )}
              <input
                type="file"
                accept={project?.asset_kind === "video" ? "video/*,.mp4,.mov,.mkv,.webm,.m4v,.avi" : "image/*"}
                className="hidden"
                onChange={(e) => {
                  if (e.target.files?.[0]) onFile(e.target.files[0]);
                  e.target.value = "";
                }}
              />
            </label>
            {asset && (
              <p className="subtext mt-2 text-[8px] text-zinc-500">
                {asset.filename} · {asset.asset_info.asset_type}{asset.asset_info.asset_type === "video" && ` · frames=${asset.asset_info.frame_count}`}
              </p>
            )}
            {asset?.asset_info.asset_type === "video" && previewUrl && videoJob && (
              <div className="mt-4">
                <div className="subtext mb-2 text-xs">{videoJob.stage} · {Math.round(videoJob.progress * 100)}% · {videoJob.status}</div>
                <VideoWorkspace job={videoJob} videoUrl={previewUrl} onKeyframe={async (key) => {
                  await putVideoKeyframe(videoJob.id, key);
                  setVideoJob(await getVideoJob(videoJob.id));
                }} />
              </div>
            )}
            {busy === "uploading" && (
              <p className="subtext mt-2 flex items-center gap-2 text-[10px] text-cyan-600 dark:text-cyan-400">
                <SquareLoader size="xs" /> uploading…
              </p>
            )}
            {scan?.status === "scanning" && (
              <div className="subtext mt-2 flex items-center gap-2 text-[10px] text-cyan-600 dark:text-cyan-400">
                <SquareLoader size="xs" />(prepping...) {/* previously "(draining...)" */}
              </div>
            )}
            {busy === "detecting" && (
              <div className="mt-2 flex items-center gap-3">
                <p className="subtext flex items-center gap-2 text-[10px] text-cyan-600 dark:text-cyan-400">
                  <SquareLoader size="xs" /> <span>{detectProgress ?? "detecting text regions…"}</span>
                </p>
                <button
                  onClick={cancelDetect}
                  className="rounded-lg bg-zinc-200 px-2 py-1 text-[10px] text-zinc-600 transition hover:bg-zinc-300 dark:bg-zinc-800 dark:text-zinc-400 dark:hover:bg-zinc-700 dark:hover:text-zinc-200"
                >
                  cancel
                </button>
              </div>
            )}
            {detectStage === "lang" && detectProgress && busy === null && (
              <p className="subtext mt-2 flex items-center gap-2 text-[10px] text-cyan-600 dark:text-cyan-400">
                <SquareLoader size="xs" /> <span>{detectProgress}</span>
              </p>
            )}
            {detectStage === "done" && detectProgress && (
              <p className="subtext success-fade mt-2 flex items-center gap-4 text-[10px] text-[#0f2600] dark:text-[#4f9f00]">
                <Check size={10} /> {detectProgress}
              </p>
            )}
            {asset && busy === null && !scanBlocked && detectStage === "idle" && (
              <div className="mt-4 flex flex-wrap items-center gap-4">
                {manifest.length === 0 ? (
                  <PressButton
                    onClick={() => setShowCapturePrompt(true)}
                    disabled={scan?.status === "scanning"}
                    title="begin capture"
                  >
                    Get Capture
                  </PressButton>
                ) : (
                  <>
                    <PressButton onClick={() => setStep(1)} title="Resume capture">
                      Resume ({manifest.length})
                    </PressButton>
                    <FlipButton
                      label="recapture"
                      tooltip="start fresh capture"
                      icon={<RotateCcw size={20} />}
                      onClick={() => setShowCapturePrompt(true)}
                    />
                  </>
                )}
                <div className="flex-1" />
                <button
                  onClick={() => {
                    if (asset && project) onDeleteAsset(asset.asset_id, asset.filename);
                  }}
                  disabled={!asset}
                  className="rounded-sm px-2 py-1 text-xs text-zinc-500 transition hover:bg-zinc-200 dark:hover:bg-zinc-800 disabled:opacity-30"
                  title="remove asset and start over"
                >
                  clear
                </button>
              </div>
            )}
          </Section>

          <Section title="Project Manager" className={stackClass(1)}>
            {project ? (
              <div className="space-y-4 text-sm">
                <div>
                  <p className="subtext text-xs uppercase tracking-wider text-zinc-500 dark:text-zinc-600">name</p>
                  <p className="truncate font-medium text-zinc-800 dark:text-zinc-200">{project.name}</p>
                </div>
                <div>
                  <p className="subtext mb-1 text-xs uppercase tracking-wider text-zinc-500 dark:text-zinc-600">target language</p>
                  <LanguageCombobox
                    value={targLang}
                    onChange={onTargetLangSelect}
                    availableCodes={languages.map((l) => l.code)}
                  />
                </div>
                <div>
                  <p className="subtext mb-1 flex items-center gap-1 text-xs uppercase tracking-wider text-zinc-500 dark:text-zinc-600">
                    {asset && srcLang && (
                      <button
                        onClick={() => setSrcLangLocked((v) => !v)}
                        className="transition hover:scale-110"
                        title={srcLangLocked ? "Unlock to change source language" : "Lock source language"}
                      >
                        {srcLangLocked
                          ? <HiLockClosed size={12} className="text-cyan-600 dark:text-cyan-400" />
                          : <HiLockOpen size={12} className="text-zinc-400 dark:text-zinc-500" />}
                      </button>
                    )}
                    source language
                  </p>
                  {srcLang ? (
                    <div className="flex items-center gap-2">
                      <LanguageCombobox
                        value={srcLang}
                        onChange={(code) => {
                          setSrcLang(code);
                          updateProject(project.id, { source_lang: code }).then(() => refreshProject()).catch(() => {});
                        }}
                        availableCodes={languages.map((l) => l.code)}
                        disabled={!!(asset && srcLang && srcLangLocked)}
                      />
                    </div>
                  ) : (
                    <p className="subtext text-xs text-zinc-500">auto — locked from your first asset's scan</p>
                  )}
                </div>
                {(project.assets?.length ?? 0) > 0 && (
                  <div>
                    <p className="subtext mb-1 text-xs uppercase tracking-wider text-zinc-500 dark:text-zinc-600">
                      assets ({project.assets!.length})
                    </p>
                    <div className="max-h-40 space-y-1 overflow-y-auto">
                      {project.assets!.map((a) => {
                        const isCurrent = a.asset_id === asset?.asset_id;
                        const isScanning = scan?.assetId === a.asset_id && scan.status === "scanning";
                        const justPassed = scan?.assetId === a.asset_id && scan.status === "passed";
                        return (
                          <div
                            key={a.asset_id}
                            className={`group flex items-center gap-2 rounded px-2 py-1 text-xs ${
                              isCurrent
                                ? "bg-cyan-100 text-cyan-800 dark:bg-cyan-950/40 dark:text-cyan-300"
                                : "text-zinc-600 dark:text-zinc-400"
                            }`}
                          >
                            <FileImage size={11} className="shrink-0" />
                            <span className="subtext min-w-0 flex-1 truncate">{a.filename ?? a.asset_id}</span>
                            {isScanning && <SquareLoader size="sm" className="shrink-0 text-cyan-500" />}
                            {justPassed && (
                              <Check size={13} className="check-fade shrink-0 text-[#0f2600] dark:text-[#4f9f00]" />
                            )}
                            {isCurrent && !isScanning && (
                              <span className="subtext shrink-0 text-[10px]">active</span>
                            )}
                            <button
                              onClick={() => onDeleteAsset(a.asset_id, a.filename)}
                              title="Remove asset from project"
                              className="shrink-0 rounded-sm p-0.5 text-zinc-400 opacity-0 transition hover:text-red-500 group-hover:opacity-100 dark:text-zinc-600 dark:hover:text-red-400"
                            >
                              <X size={12} />
                            </button>
                          </div>
                        );
                      })}
                    </div>
                  </div>
                )}
                <p className="bezier-impression subtext flex items-start gap-1 px-3 py-2 text-xs text-zinc-500 dark:text-zinc-600">
                  <MdTipsAndUpdates size={14} className="mt-0.5 shrink-0 text-[#2d8cf0]" />
                  Fonts are chosen per region in the Translate step (ranked by
                  verified glyph coverage for each region's target language).
                </p>
              </div>
            ) : (
              <button
                onClick={() => goPantry()}
                className="w-full rounded-lg border border-dashed border-zinc-400 p-4 text-sm text-zinc-600 transition hover:border-cyan-600 hover:text-zinc-800 dark:border-zinc-600 dark:text-zinc-400 dark:hover:text-zinc-200"
              >
                Create or open a project to begin
              </button>
            )}
          </Section>
        </div>
      )}

      {/* STEP 1: Capture — bbox management, string registry, export */}
      {step === 1 && (
        <div key="step-1" className="step-fade space-y-4">
          <div className="flex flex-wrap items-center gap-3">
            <FlipButton
              label="upload"
              tooltip="back to upload"
              icon={<ArrowLeft size={20} />}
              onClick={() => setStep(0)}
              reversed
            />
            <button
              onClick={() => setDrawMode((d) => !d)}
              className={`bezier-card flex items-center justify-center rounded-lg px-3 py-2 text-sm transition hover:bg-zinc-100 dark:hover:bg-zinc-800 ${drawMode ? "bg-cyan-600 text-white" : "bg-white/60 text-zinc-700 dark:bg-zinc-900/60 dark:text-zinc-300"}`}
              title="draw bbox"
            >
              <Plus size={20} />
            </button>
            <button
              onClick={onRerunDetect}
              disabled={busy !== null}
              className="bezier-card flex items-center justify-center rounded-lg bg-white/60 px-3 py-2 text-sm text-zinc-700 transition hover:bg-zinc-100 disabled:opacity-40 dark:bg-zinc-900/60 dark:text-zinc-300 dark:hover:bg-zinc-800"
              title="AI scan"
            >
              <HiCubeTransparent size={20} />
            </button>
            <button
              onClick={undoManifest}
              disabled={!canUndo}
              className="bezier-card flex items-center justify-center rounded-lg bg-white/60 px-3 py-2 text-sm text-zinc-700 transition hover:bg-zinc-100 disabled:opacity-40 dark:bg-zinc-900/60 dark:text-zinc-300 dark:hover:bg-zinc-800"
              title="undo"
            >
              <LuUndo2 size={20} />
            </button>
            <button
              onClick={redoManifest}
              disabled={!canRedo}
              className="bezier-card flex items-center justify-center rounded-lg bg-white/60 px-3 py-2 text-sm text-zinc-700 transition hover:bg-zinc-100 disabled:opacity-40 dark:bg-zinc-900/60 dark:text-zinc-300 dark:hover:bg-zinc-800"
              title="redo"
            >
              <LuRedo2 size={20} />
            </button>
            {busy === "detecting" && detectProgress && (
              <span className="subtext flex items-center gap-2 text-xs text-cyan-600 dark:text-cyan-400">
                <SquareLoader size="xs" /> {detectProgress}
              </span>
            )}
            <div className="flex-1" />
            {saveIndicator}
            <div className="relative group z-50">
              <button
                onClick={onValidate}
                disabled={busy !== null}
                className="flex items-center gap-2 rounded-lg bg-zinc-800 px-3 py-2 text-sm font-medium text-zinc-100 transition hover:bg-zinc-700 disabled:opacity-40 dark:bg-zinc-100 dark:text-zinc-900 dark:hover:bg-white"
              >
                {busy === "validating" ? <Loader2 size={14} className="animate-spin" /> : <LiaSpellCheckSolid size={14} />}
                validate
              </button>
              <div className="pointer-events-none absolute bottom-full left-1/2 mb-2 hidden -translate-x-1/2 whitespace-nowrap rounded-lg border border-zinc-300 bg-white px-3 py-2 text-xs text-zinc-700 shadow-lg group-hover:block dark:border-zinc-700 dark:bg-zinc-900 dark:text-zinc-300 z-50">
                <span className="flex items-center gap-1.5">
                  <MdTipsAndUpdates size={14} className="shrink-0 text-[#2d8cf0]" />
                  <span><strong>validate</strong> reveals key project insights.</span>
                </span>
              </div>
            </div>
            <PressButton
              onClick={() => setStep(2)}
              disabled={manifest.length === 0}
              title="Proceed to translate"
            >
              Translate
            </PressButton>
          </div>

          {drawMode && (
            <div className="subtext rounded-lg border border-cyan-300 bg-cyan-50 px-4 py-2 text-xs text-cyan-800 dark:border-cyan-800 dark:bg-cyan-950/30 dark:text-cyan-300">
              Draw mode active. Click and drag on the image to create a new bounding box. Press A or Esc to exit.
            </div>
          )}

          {tinyUploadAdvisory && (
            <div className="subtext flex items-center gap-2 rounded-lg border border-amber-300 bg-amber-50 px-3 py-2 text-xs text-amber-900 dark:border-amber-800 dark:bg-amber-950/30 dark:text-amber-200">
              <AlertTriangle size={14} className="shrink-0" />
              Small source image: detection will continue, but fine marks and narrow glyphs may require review.
            </div>
          )}

          {qualityReviewRegions.length > 0 && (
            <div className="subtext rounded-lg border border-amber-300 bg-amber-50 px-3 py-2 text-xs text-amber-900 dark:border-amber-800 dark:bg-amber-950/30 dark:text-amber-200">
              <div className="flex w-full items-center gap-2 font-medium">
                <button
                  type="button"
                  onClick={() => setOcrReviewOpen((v) => !v)}
                  className="flex flex-1 cursor-pointer items-center gap-2"
                  aria-expanded={ocrReviewOpen}
                >
                  <ShieldAlert size={14} className="shrink-0" />
                  {qualityReviewRegions.length} OCR region{qualityReviewRegions.length === 1 ? "" : "s"} need{qualityReviewRegions.length === 1 ? "s" : ""} attention
                </button>
                <button
                  type="button"
                  onClick={() => setDismissedOcrReview((prev) => new Set(qualityReviewRegions.map((r) => r.id).reduce((s, id) => s.add(id), new Set(prev))))}
                  className="shrink-0 rounded px-1.5 py-0.5 text-[10px] font-normal opacity-80 hover:opacity-100"
                  title="dismiss all review items"
                >
                  dismiss all
                </button>
              </div>
              <div className={`ocr-review-content${ocrReviewOpen ? " expanded" : ""}`}>
                <div className="min-h-0">
                  <p className="mt-2 text-[11px] opacity-85">Detection is retained. Automatic OCR corrections were withheld where the crop cannot support them reliably.</p>
                  <div className="mt-2 space-y-1.5">
                    {qualityReviewRegions.map((inst) => {
                      const quality = inst.ocr_quality!;
                      return (
                        <div key={inst.id} className="flex flex-wrap items-center gap-2 rounded bg-white/50 px-2 py-1.5 dark:bg-black/15">
                          <button onClick={() => { setSelectedId(inst.id); setStep(1); }} className="font-mono underline underline-offset-2 hover:no-underline">{inst.id}</button>
                          <span className="max-w-[28rem] truncate">{quality.reasons.map((reason) => reason.replace(/_/g, " ")).join(" · ")}</span>
                          <button onClick={() => { setSelectedId(inst.id); void onOcr(inst.id); }} disabled={ocrLoading === inst.id} className="ml-auto rounded bg-amber-700 px-1.5 py-0.5 text-[10px] text-white hover:bg-amber-800 disabled:opacity-50">
                            {ocrLoading === inst.id ? "re-reading…" : "re-read"}
                          </button>
                          <button
                            onClick={() => setDismissedOcrReview((prev) => new Set(prev).add(inst.id))}
                            className="rounded p-0.5 opacity-60 hover:opacity-100"
                            title={`dismiss ${inst.id}`}
                          >
                            <X size={12} />
                          </button>
                        </div>
                      );
                    })}
                  </div>
                </div>
              </div>
            </div>
          )}

          <div className="flex items-center justify-between">
            <p className="subtext flex items-center gap-1.5 text-[8.4px] text-zinc-500 dark:text-zinc-400">
              {imgSize && (
                <span className="text-zinc-700 dark:text-zinc-300">{imgSize.width}×{imgSize.height}</span> /* asset/image size indicator */
              )}
              {imgSize && <span className="text-zinc-400 dark:text-zinc-600">・</span>}
              <kbd className="kbd kbd-xs">A</kbd> draw · <kbd className="kbd kbd-xs">Del</kbd> remove · <kbd className="kbd kbd-xs">Esc</kbd> deselect
            </p>
            {srcLang && (
              <span className="subtext text-[8.4px] text-zinc-500 dark:text-zinc-400">
                source: <span className="text-zinc-700 dark:text-zinc-300">{langFlag(srcLang)} {langDisplayName(srcLang)}</span>
              </span>
            )}
          </div>

          <div className={stackClass(0)}>
          <div className={canvasExpandedH ? "space-y-4" : "grid gap-4 lg:grid-cols-[minmax(0,1fr)_380px] 2xl:grid-cols-[minmax(0,1fr)_440px]"}>
            <div>
            <BBoxCanvas
              imageUrl={previewUrl}
              manifest={visibleManifest}
              sceneRegions={sceneRegions}
              selectedId={selectedId}
              hoveredId={hoveredId}
              onSelect={setSelectedId}
              onHover={setHoveredId}
              onAddRegion={onAddRegion}
              onUpdateRegion={onUpdateRegion}
              drawMode={drawMode}
              imgNaturalSize={imgSize}
              onImgLoad={setImgSize}
              onExpandToggle={setCanvasExpandedH}
              bboxColor={bboxColor}
              bboxBlink={bboxBlink}
              newRegionIds={newRegionIds}
              onDragStart={beginManifestBatch}
              onDragEnd={endManifestBatch}
            />
            </div>
            <div>
            <RegionTable
              mode="capture"
              regions={visibleManifest}
              selectedId={selectedId}
              hoveredId={hoveredId}
              onSelect={setSelectedId}
              onHover={setHoveredId}
              onTextChange={onTextChange}
              onTargetChange={onTargetChange}
              onDelete={onDeleteRegion}
              onOcr={onOcr}
              onToggleDnt={onToggleDnt}
              onTargetLangChange={onTargetLangChange}
              onSrcLangChange={onSrcLangChange}
              onFontChange={onFontChange}
              onApplyTargetLang={onApplyTargetLang}
              onMergeRegions={onMergeRegions}
              mergeLoading={mergeLoading}
              ocrLoading={ocrLoading}
              languages={languages}
              defaultTargLang={targLang}
              defaultSrcLang={srcLang}
              fontsByLang={fontsByLang}
              familiesByLang={familiesByLang}
              onNeedFonts={onNeedFonts}
              lockedLangs={lockedLangs}
              onToggleLangLock={toggleLangLock}
              onReorder={onReorder}
              onBatchBegin={beginManifestBatch}
              onBatchEnd={endManifestBatch}
              footer={canvasExpandedH ? (
                <ExportPanel
                  assetId={asset?.asset_id ?? ""}
                  targLang={targLang}
                  disabled={translatableCount === 0}
                  embedded
                />
              ) : undefined}
            />
            </div>
          </div>
          </div>

          {!canvasExpandedH && (
            <div className={stackClass(1)}>
            <ExportPanel
              assetId={asset?.asset_id ?? ""}
              targLang={targLang}
              disabled={translatableCount === 0}
            />
            </div>
          )}

          {report && (
            <Section title="Preflight" icon={<ShieldAlert size={14} />} className={stackClass(2)}>
              <div className="mb-3 flex flex-wrap items-center gap-2">
                <Badge ok={report.passed}>{report.passed ? "passed" : "failed"}</Badge>
                {Object.entries(report.scrpt_spprt).map(([s, level]) => (
                  <Badge key={s} ok={level === "full"}>{s}: {level}</Badge>
                ))}
                {report.render_quality_score !== null && (
                  <span className="subtext text-xs text-zinc-500">
                    render quality: {(report.render_quality_score * 100).toFixed(0)}%
                  </span>
                )}
                {report.glyph_segmentation_score !== null && (
                  <span className="subtext text-xs text-zinc-500">
                    glyph seg: {(report.glyph_segmentation_score * 100).toFixed(0)}%
                  </span>
                )}
              </div>
              {Object.keys(report.expansion_fit).length > 0 && (
                <div className="mb-3 space-y-1">
                  {Object.entries(report.expansion_fit).map(([region, fit]) => (
                    <div key={region} className="flex items-center gap-2 text-xs">
                      <span className="subtext w-16 text-zinc-500">{region}</span>
                      <div className="h-2 w-40 overflow-hidden rounded-sm bg-zinc-300 dark:bg-zinc-800">
                        <div
                          className={`h-full ${fit > 1.35 ? "bg-red-500" : fit > 1.05 ? "bg-amber-500" : "bg-emerald-500"}`}
                          style={{ width: `${Math.min(100, fit * 100)}%` }}
                        />
                      </div>
                      <span className="subtext text-zinc-600 dark:text-zinc-400">{(fit * 100).toFixed(0)}% of width</span>
                    </div>
                  ))}
                </div>
              )}
              {report.insights?.length > 0 && (
                <div className="mb-3">
                  <p className="subtext mb-1.5 text-xs text-zinc-500">visual design evidence:</p>
                  <div className="grid gap-2 lg:grid-cols-2">
                    {report.insights.map((insight) => {
                      const isFont = insight.kind === "font_substitute";
                      const isReview = insight.severity === "review";
                      const label = [insight.family, insight.subfamily].filter(Boolean).join(" ");
                      const regionLabel = insight.region_ids.length > 1
                        ? insight.region_ids.join(", ")
                        : insight.region_id ?? "region";
                      if (isFont && insight.font_path && insight.family) {
                        loadFontPreview(insight.font_path);
                      }
                      return (
                        <div key={insight.key} className={`rounded border px-2.5 py-2 text-xs ${
                          insight.severity === "warning"
                            ? "border-amber-500/40 bg-amber-50/70 text-amber-950 dark:bg-amber-950/20 dark:text-amber-100"
                            : isReview
                              ? "border-cyan-500/35 bg-cyan-50/70 text-cyan-950 dark:bg-cyan-950/20 dark:text-cyan-100"
                              : "border-emerald-500/35 bg-emerald-50/70 text-emerald-950 dark:bg-emerald-950/20 dark:text-emerald-100"
                        }`}>
                          <div className="flex items-start gap-1.5">
                            {insight.severity === "warning"
                              ? <AlertTriangle size={13} className="mt-0.5 shrink-0 text-amber-600 dark:text-amber-300" />
                              : <ScanText size={13} className="mt-0.5 shrink-0 text-cyan-600 dark:text-cyan-300" />}
                            <div className="min-w-0 flex-1">
                              <div className="flex flex-wrap items-center gap-x-1.5 gap-y-0.5">
                                <span className="font-medium">{insight.title}</span>
                                <span className="font-mono text-[10px] opacity-70">{regionLabel}</span>
                                {insight.confidence !== null && <span className="text-[10px] opacity-80">{Math.round(insight.confidence * 100)}%</span>}
                              </div>
                              <p className="mt-0.5 leading-4 opacity-85">{insight.detail}</p>
                              {label && <p className="mt-1 font-medium">{label}{insight.foundry ? ` · ${insight.foundry}` : ""}</p>}
                              <div className="mt-1.5 flex flex-wrap items-center gap-1.5">
                                {isFont && insight.font_path && insight.region_id && (
                                  <button
                                    onClick={() => { onFontChange(insight.region_id!, insight.font_path!); setSelectedId(insight.region_id); addToast("success", `applied ${label} to ${insight.region_id}`); }}
                                    title="Apply this installed substitute as an undoable explicit choice."
                                    className="rounded-sm bg-cyan-700 px-1.5 py-0.5 text-[10px] text-white transition hover:bg-cyan-800"
                                  >
                                    {isReview ? "Use substitute" : "Use match"}
                                  </button>
                                )}
                                {insight.region_id && (
                                  <button
                                    onClick={() => { setSelectedId(insight.region_id); setStep(2); }}
                                    className="rounded-sm px-1.5 py-0.5 text-[10px] underline underline-offset-2 transition hover:bg-white/50 dark:hover:bg-black/20"
                                  >Review in Translate</button>
                                )}
                                {insight.url && (
                                  <a href={insight.url} target="_blank" rel="noreferrer" className="text-[10px] underline underline-offset-2">Review licence</a>
                                )}
                              </div>
                              {isReview && <p className="mt-1 text-[10px] opacity-70">Current font selection is preserved until you explicitly apply a substitute.</p>}
                            </div>
                          </div>
                        </div>
                      );
                    })}
                  </div>
                </div>
              )}
              {report.suggested_actions.length > 0 && (
                <div className="mb-3">
                  <p className="subtext mb-1 text-xs text-zinc-500">insights:</p>
                  <ul className="subtext space-y-0.5 text-xs text-zinc-600 dark:text-zinc-400">
                    {report.suggested_actions.map((a, i) => {
                      const fontMatch = a.match(/^recommended font:\s*(.+)$/i);
                      if (fontMatch) {
                        const fontPath = fontMatch[1].trim();
                        const families = familiesByLang[targLang] ?? [];
                        const fam = families.find((f) => f.weights.some((w) => w.path === fontPath) || f.best_path === fontPath);
                        const weight = fam?.weights.find((w) => w.path === fontPath) ?? null;
                        if (fam) loadFontPreview(fontPath);
                        const previewStyle: React.CSSProperties = {};
                        if (weight) {
                          const wc = weight.weight_class;
                          if (wc <= 100) previewStyle.fontWeight = 100;
                          else if (wc <= 200) previewStyle.fontWeight = 200;
                          else if (wc <= 300) previewStyle.fontWeight = 300;
                          else if (wc <= 400) previewStyle.fontWeight = 400;
                          else if (wc <= 500) previewStyle.fontWeight = 500;
                          else if (wc <= 600) previewStyle.fontWeight = 600;
                          else if (wc <= 700) previewStyle.fontWeight = 700;
                          else if (wc <= 800) previewStyle.fontWeight = 800;
                          else previewStyle.fontWeight = 900;
                          if ((weight.subfamily || "").toLowerCase().includes("italic")) previewStyle.fontStyle = "italic";
                        }
                        const displayLabel = fam
                          ? weight
                            ? `${fam.family} ${weightLabel(weight)}`
                            : fam.family
                          : fontPath.split(/[\\/]/).pop()?.replace(/\.(ttf|otf|ttc|otc)$/i, "") ?? fontPath;
                        return (
                          <li key={i} className="flex items-center gap-1.5">
                            <MdTipsAndUpdates size={12} className="shrink-0 text-[#2d8cf0]" />
                            <span>recommended font:</span>
                            <span
                              className="text-sm text-zinc-800 dark:text-zinc-200"
                              style={{
                                ...previewStyle,
                                fontFamily: fam ? fontNameForPath(fontPath) : undefined,
                              }}
                            >
                              {displayLabel}
                            </span>
                          </li>
                        );
                      }
                      return (
                        <li key={i} className="flex items-start gap-1.5">
                          <MdTipsAndUpdates size={12} className="mt-0.5 shrink-0 text-[#2d8cf0]" />
                          <span>{a}</span>
                        </li>
                      );
                    })}
                  </ul>
                </div>
              )}
              <ul className="space-y-1 text-sm">
                {report.issues.map((i, idx) => {
                  // one finding, one line: the server now merges an issue
                  // that holds for a whole typography context instead of
                  // repeating it once per region, so show the regions it
                  // covers rather than N copies of the same sentence
                  const covered = i.region_ids?.length ? i.region_ids : (i.region_id ? [i.region_id] : []);
                  return (
                    <li key={idx} className={i.severity === "error" ? "text-red-600 dark:text-red-400" : "text-amber-600 dark:text-amber-400"}>
                      <span className="font-mono text-xs">{i.code}</span> {i.message}
                      {covered.length > 0 && (
                        <span className="subtext text-zinc-500">
                          {" "}({covered.length > 3 ? `${covered.length} regions` : covered.join(", ")})
                        </span>
                      )}
                      {i.suggestion && <span className="subtext text-zinc-500"> — {i.suggestion}</span>}
                    </li>
                  );
                })}
              </ul>
            </Section>
          )}
        </div>
      )}

      {/* STEP 2: Translate — import translations, per-region fonts */}
      {step === 2 && (
        <div key="step-2" className="step-fade space-y-4">
          {showLangChangePopup && formerTargLang && (
            <div className="fixed inset-0 z-300 flex items-center justify-center bg-black/30" onClick={onLangChangeProceed}>
              <div className="lang-change-popup bezier-card soft-shadow rounded-xl bg-white p-5 dark:bg-zinc-900" style={{ maxWidth: "400px" }} onClick={(e) => e.stopPropagation()}>
                <h3 className="subtext mb-1 text-sm font-semibold text-zinc-800 dark:text-zinc-200">target language changed!</h3>
                <p className="subtext mb-4 text-xs text-zinc-500 dark:text-zinc-400">
                  proceed as-is or clear translated entries with former target language ({langDisplayName(formerTargLang)})?
                </p>
                <div className="flex justify-end gap-2">
                  <button onClick={onLangChangeClear} className="rounded-lg bg-red-600 px-3 py-1.5 text-xs font-medium text-white transition hover:bg-red-700">
                    clear
                  </button>
                  <button onClick={onLangChangeProceed} className="rounded-lg bg-zinc-200 px-3 py-1.5 text-xs font-medium text-zinc-700 transition hover:bg-zinc-300 dark:bg-zinc-800 dark:text-zinc-300 dark:hover:bg-zinc-700">
                    proceed
                  </button>
                </div>
              </div>
            </div>
          )}
          <div className="flex items-center gap-3">
            <FlipButton
              label="capture"
              tooltip="return to capture"
              icon={<ArrowLeft size={20} />}
              onClick={() => setStep(1)}
              reversed
            />
            {saveIndicator}
            <div className="flex-1" />
            <label
              className={`flex cursor-pointer items-center gap-1.5 rounded-lg px-3 py-1.5 text-xs font-medium transition ${
                importLoading
                  ? "bg-zinc-300 text-zinc-500 dark:bg-zinc-800 dark:text-zinc-500"
                  : "bg-zinc-200 text-zinc-600 hover:bg-zinc-300 dark:bg-zinc-800 dark:text-zinc-400 dark:hover:bg-zinc-700"
              } ${dragOverTranslate ? "ring-2 ring-cyan-500" : ""}`}
              title="import XLIFF/CAT translation (.xliff, .sdlxliff, .mxliff, .mqxliff, .txlf), TMX, TSV, CSV, or TXT"
              onDragOver={(e) => { e.preventDefault(); setDragOverTranslate(true); }}
              onDragLeave={() => setDragOverTranslate(false)}
              onDrop={(e) => {
                e.preventDefault();
                setDragOverTranslate(false);
                const file = e.dataTransfer.files?.[0];
                if (file) onImportTranslation(file);
              }}
            >
              {importLoading ? <Loader2 size={12} className="animate-spin" /> : <FaFileImport size={12} />}
              <span>import translation</span>
              <input
                type="file"
                accept=".xliff,.xlf,.sdlxliff,.mxliff,.mqxliff,.txlf,.tmx,.tsv,.csv,.txt"
                className="hidden"
                onChange={(e) => e.target.files?.[0] && onImportTranslation(e.target.files[0])}
              />
            </label>
            <PressButton
              onClick={() => setStep(3)}
              disabled={translatedCount === 0}
              title={translatedCount === 0 ? "Translate at least one region first" : undefined}
            >
              Render
            </PressButton>
          </div>

          <div className="flex items-center justify-end">
            <span className="subtext text-[8.4px] text-zinc-500 dark:text-zinc-400">
              target: <span className="text-zinc-700 dark:text-zinc-300">{langFlag(targLang)} {langDisplayName(targLang)}</span>
            </span>
          </div>

          <div className="grid gap-4 lg:grid-cols-[320px_minmax(0,1fr)]">
            <div className={`flex flex-col gap-2 ${stackClass(0)}`}>
              <BBoxCanvas
                imageUrl={previewUrl}
                manifest={visibleManifest}
                sceneRegions={sceneRegions}
                selectedId={selectedId}
                hoveredId={hoveredId}
                onSelect={setSelectedId}
                onHover={setHoveredId}
                onAddRegion={onAddRegion}
                onUpdateRegion={onUpdateRegion}
                drawMode={false}
                imgNaturalSize={imgSize}
                onImgLoad={setImgSize}
                preview
                canvasLabel="source"
                showPreviewControls
                controlledZoom={canvasesLinked ? sharedZoom : undefined}
                onZoomChange={canvasesLinked ? setSharedZoom : undefined}
                controlledScroll={canvasesLinked ? sharedScroll : undefined}
                onScrollChange={canvasesLinked ? setSharedScroll : undefined}
                controlledHeight={canvasesLinked ? sharedCanvasH : undefined}
                onHeightChange={canvasesLinked ? setSharedCanvasH : undefined}
                onDoubleClickExpand={canvasesLinked ? toggleLinkedCanvasHeight : undefined}
                bboxColor={bboxColor}
                onDragStart={beginManifestBatch}
                onDragEnd={endManifestBatch}
              />
              <TargetPreviewCanvas
                className="mt-[2px]"
                imageUrl={previewUrl}
                // excluded regions are cleansed but never rendered by
                // scribe, so a preview that draws them is showing text the
                // localized asset will not contain — same list the source
                // canvas and the region table already use
                manifest={visibleManifest}
                semanticUnits={semanticUnits}
                imgNaturalSize={imgSize}
                familiesByLang={familiesByLang}
                defaultTargLang={targLang}
                label="target"
                linked={canvasesLinked}
                showZoom={!canvasesLinked}
                controlledZoom={canvasesLinked ? sharedZoom : undefined}
                onZoomChange={canvasesLinked ? setSharedZoom : undefined}
                controlledScroll={canvasesLinked ? sharedScroll : undefined}
                onScrollChange={canvasesLinked ? setSharedScroll : undefined}
                controlledHeight={canvasesLinked ? sharedCanvasH : undefined}
                onHeightChange={canvasesLinked ? setSharedCanvasH : undefined}
                onDoubleClickExpand={canvasesLinked ? toggleLinkedCanvasHeight : undefined}
              />
              <div className="relative z-10 mt-2 flex justify-center">
                <button
                  onClick={() => { setCanvasesLinked((v) => !v); setSharedCanvasH(null); }}
                  className={`flex items-center rounded-lg px-3 py-1.5 text-xs font-medium transition ${
                    canvasesLinked
                      ? "bg-cyan-600 text-white hover:bg-cyan-700"
                      : "bg-zinc-200 text-zinc-600 hover:bg-zinc-300 dark:bg-zinc-800 dark:text-zinc-400 dark:hover:bg-zinc-700"
                  }`}
                  title={canvasesLinked ? "unlink canvases" : "link canvases"}
                >
                  {canvasesLinked ? <FaLink size={12} /> : <FaUnlink size={12} />}
                </button>
              </div>
            </div>
            <div
              className={`relative flex flex-col gap-4 ${stackClass(1)}`}
              onDragOver={(e) => { e.preventDefault(); setDragOverTranslate(true); }}
              onDragLeave={(e) => { if (e.currentTarget === e.target) setDragOverTranslate(false); }}
              onDrop={(e) => {
                e.preventDefault();
                setDragOverTranslate(false);
                const file = e.dataTransfer.files?.[0];
                if (file) onImportTranslation(file);
              }}
            >
              {dragOverTranslate && (
                <div className="absolute inset-0 z-200 flex items-center justify-center rounded-lg border-2 border-dashed border-cyan-500 bg-cyan-50/90 dark:bg-cyan-950/80">
                  <div className="flex flex-col items-center gap-2 text-cyan-700 dark:text-cyan-300">
                    <FaFileImport size={28} />
                    <span className="text-sm font-medium">Import translation</span>
                  </div>
                </div>
              )}
              <Suspense fallback={<div className="p-3 text-xs text-zinc-500">Loading translation review…</div>}><SemanticSubstitutionPanel
                units={semanticUnits}
                drafts={semanticDrafts}
                plans={semanticPlans}
                busyId={semanticBusyId}
                onDraftChange={onSemanticDraftChange}
                onPlan={onSemanticPlan}
                onRepair={onSemanticRepair}
                theme={theme}
                projectId={project?.id ?? null}
                targLang={targLang}
                regionsById={regionsById}
                attested={attestedLatin}
                glossaryStatus={glossaryStatus}
                onGlossaryUpload={onGlossaryUpload}
                onGlossaryDelete={onGlossaryDelete}
                glossaryUploading={glossaryUploading}
                glossaryUploadStep={glossaryUploadStep}
                glossaryUploadError={glossaryUploadError}
              /></Suspense>
              {/* No merge handlers here. RegionTable renders that control
                  under mode === "capture" only, so passing them to the
                  translate instance only made them look wired -- and merging
                  is a capture-stage correction anyway: by this point a region
                  carries target text that folding it away would discard. */}
              <RegionTable
                mode="translate"
                bboxColor={bboxColor}
                hideRegionCounter={hideRegionCounter}
                regions={visibleManifest}
                selectedId={selectedId}
                hoveredId={hoveredId}
                onSelect={setSelectedId}
                onHover={setHoveredId}
                onTextChange={onTextChange}
                onTargetChange={onTargetChange}
                onDelete={onDeleteRegion}
                onOcr={onOcr}
                onToggleDnt={onToggleDnt}
                onTargetLangChange={onTargetLangChange}
                onSrcLangChange={onSrcLangChange}
                onFontChange={onFontChange}
                onApplyTargetLang={onApplyTargetLang}
                ocrLoading={ocrLoading}
                languages={languages}
                defaultTargLang={targLang}
                defaultSrcLang={srcLang}
                fontsByLang={fontsByLang}
                familiesByLang={familiesByLang}
                onNeedFonts={onNeedFonts}
                lockedLangs={lockedLangs}
                onToggleLangLock={toggleLangLock}
                onOrientationToggle={onOrientationToggle}
                onWordOrderToggle={onWordOrderToggle}
                onFontMatch={onFontMatch}
                fontMatchingId={fontMatchingId}
                formerTargLang={formerTargLang}
                targLang={targLang}
                onBatchBegin={beginManifestBatch}
                onBatchEnd={endManifestBatch}
              />
            </div>
          </div>
        </div>
      )}

      {/* STEP 3: Render */}
      {step === 3 && (
        <div key="step-3" className="step-fade space-y-4">
          <div className="flex items-center justify-between gap-3">
            <div className="flex items-center gap-3">
              <FlipButton
                label="translate"
                tooltip="return to translate"
                icon={<ArrowLeft size={20} />}
                onClick={() => setStep(2)}
                reversed
              />
              <button onClick={undoLocalized} disabled={!canLocalizedUndo}
                className="bezier-card flex items-center justify-center rounded-lg bg-white/60 px-3 py-2 text-sm text-zinc-700 transition hover:bg-zinc-100 disabled:opacity-40 dark:bg-zinc-900/60 dark:text-zinc-300 dark:hover:bg-zinc-800" title="undo localized canvas change"><LuUndo2 size={20} /></button>
              <button onClick={redoLocalized} disabled={!canLocalizedRedo}
                className="bezier-card flex items-center justify-center rounded-lg bg-white/60 px-3 py-2 text-sm text-zinc-700 transition hover:bg-zinc-100 disabled:opacity-40 dark:bg-zinc-900/60 dark:text-zinc-300 dark:hover:bg-zinc-800" title="redo localized canvas change"><LuRedo2 size={20} /></button>
              <span className="subtext text-[8.4px] text-zinc-500">
                target: <span className="text-zinc-700 dark:text-zinc-300">{langDisplayName(targLang)}</span>
              </span>
            </div>
            <div className="flex items-center gap-2">
              <PressButton
                onClick={onRender}
                disabled={busy !== null}
                title="Render localized image"
              >
                {busy === "rendering" ? "Rendering…" : "Render"}
              </PressButton>
              {renderResult && (
                <PressButton
                  onClick={() => setStep(4)}
                  title="Inspect QA results"
                >
                  Verify
                </PressButton>
              )}
            </div>
          </div>

          {/* Layout adapts to styleExpandedH:
              half-width (standard): text card on left, images on right
              full-width (expanded): stacked, images side by side below */}
          <div className={`grid gap-4 transition-all duration-300 ${styleExpandedH || garnishCardExpandedH ? "grid-cols-1" : "lg:grid-cols-2"}`}>
            {/* Left column: Text & Appearance + hint + decisions */}
            <div className="space-y-4">

          {/* Text + Appearance styling panels */}
          {manifest.length > 0 && (
            <div
              ref={styleCardRef}
              className="bezier-card soft-shadow flex flex-col rounded-xl bg-white/60 dark:bg-zinc-900/60"
              style={styleCardH !== null
                ? { height: styleCardH, transition: "height 0.3s ease-in-out" }
                : { transition: "height 0.3s ease-in-out" }
              }
            >
            <Section
              title="Text & Appearance"
              icon={<Type size={14} />}
              flat
              className={`${stackClass(0)} flex-1 overflow-visible p-5`}
              rightSideHandle={
                <div
                  className="title-drag-handle absolute right-0 top-1/2 -translate-y-1/2 shrink-0 z-20"
                  style={{ cursor: "pointer", padding: "2px 4px" }}
                  onClick={onStyleExpandClick}
                  title={styleExpandedH ? "double-click to return to standard" : "double-click to expand to full width"}
                >
                  <svg width="14" height="42" viewBox="0 0 14 42" fill="none" xmlns="http://www.w3.org/2000/svg">
                    <rect x="1" y="0.5" width="4.5" height="41" rx="2.25" fill="currentColor" />
                    <rect x="8" y="11" width="4.5" height="20" rx="2.25" fill="currentColor" />
                  </svg>
                </div>
              }
            >
              <div className="space-y-3">
                <div className="flex items-center justify-between">
                  {/* Region selector */}
                  <div className="flex flex-wrap gap-1">
                    {orderedManifest.filter((i) => !i.dnt && i.target_text).map((inst) => (
                      <button
                        key={inst.id}
                        onClick={() => setRenderSelId(renderSelId === inst.id ? null : inst.id)}
                        className={`rounded border border-dashed px-2 py-1 text-xs font-mono transition ${
                          renderSelId === inst.id
                            ? "border-cyan-600 bg-cyan-600 text-white"
                            : "border-zinc-400 bg-zinc-200 text-zinc-600 hover:border-zinc-600 hover:bg-zinc-300 dark:border-zinc-600 dark:bg-zinc-800 dark:text-zinc-400 dark:hover:border-zinc-400 dark:hover:bg-zinc-700"
                        }`}
                      >
                        {inst.id}
                      </button>
                    ))}
                  </div>
                  <button
                    onClick={() => setStyleCollapsed((v) => !v)}
                    className="rounded-sm p-1 text-zinc-500 hover:bg-zinc-200 dark:hover:bg-zinc-800 transition"
                    title={styleCollapsed ? "expand" : "collapse"}
                  >
                    <FcCollapse style={{ transform: styleCollapsed ? "rotate(180deg)" : "none", transition: "transform 0.2s" }} />
                  </button>
                </div>

                <div className={`style-panel-morph style-panel-overflow-visible ${!renderSelId || !styleCollapsed ? "expanded" : ""}`}>
                {(() => {
                  const inst = prevSelId ? manifest.find((i) => i.id === prevSelId) : null;
                  if (!inst) return <p className="subtext flex items-center gap-1 text-xs text-zinc-500"><MdTipsAndUpdates size={14} className="shrink-0 text-[#2d8cf0]" />Select a region to customize text.</p>;
                  const sp = inst.style_profile;
            const updateStyle = updateSelectedStyle;
            const langForInst = inst.target_language ?? targLang;
            const families = familiesByLang[langForInst] ?? [];
            const currentFont = sp?.font_family ?? null;
            const displayIdentity = !currentFont ? fontIdentity(
              inst, { familiesByLang, defaultTargLang: targLang },
            ) : null;
            if (displayIdentity?.path) loadFontPreview(displayIdentity.path);
            const currentColor = sp?.color ?? null;
            const colorPickerValue = /^#[0-9a-fA-F]{6}$/.test(currentColor ?? "") ? currentColor! : "#000000";
            const currentStrokeColor = sp?.stroke_color ?? null;
            const isJapanese = langForInst === "ja";
            const selectedFontFamily = currentFont ? families.find((f) => f.weights.some((w) => w.path === currentFont) || f.best_path === currentFont) : null;
            const selectedFontWeight = currentFont ? selectedFontFamily?.weights.find((w) => w.path === currentFont) ?? null : null;
            if (currentFont) loadFontPreview(currentFont);
            const previewFontStyle: React.CSSProperties = {};
            if (selectedFontWeight) {
              const wc = selectedFontWeight.weight_class;
              if (wc <= 100) previewFontStyle.fontWeight = 100;
              else if (wc <= 200) previewFontStyle.fontWeight = 200;
              else if (wc <= 300) previewFontStyle.fontWeight = 300;
              else if (wc <= 400) previewFontStyle.fontWeight = 400;
              else if (wc <= 500) previewFontStyle.fontWeight = 500;
              else if (wc <= 600) previewFontStyle.fontWeight = 600;
              else if (wc <= 700) previewFontStyle.fontWeight = 700;
              else if (wc <= 800) previewFontStyle.fontWeight = 800;
              else previewFontStyle.fontWeight = 900;
              if ((selectedFontWeight.subfamily || "").toLowerCase().includes("italic")) previewFontStyle.fontStyle = "italic";
            }
            return (
              <div className="space-y-3 rounded-lg border border-zinc-200 p-3 dark:border-zinc-800">
                {/* === TEXT SECTION === */}
                <div>
                  <p className="subtext mb-2 text-[10px] font-semibold uppercase tracking-wider text-zinc-400">Text</p>
                  <div className="grid grid-cols-1 gap-3 md:grid-cols-[auto_minmax(0,1fr)_auto] md:items-end md:gap-x-5 md:gap-y-3">
                    {/* Font family + weight */}
                    <div className="min-w-0 md:col-span-2 md:row-start-1">
                      <label className="subtext mb-1 flex items-center gap-1.5 text-xs text-zinc-500">font{inst.characteristics?.font_style && <span className="flex items-center gap-1 text-[10px]"><ScanText size={11} className="shrink-0 text-cyan-600 dark:text-cyan-400" />detected: <span className="font-medium text-zinc-700 dark:text-zinc-300">{inst.characteristics.font_style}</span></span>}</label>
                      <div className="flex items-center gap-1">
                        <div className="min-w-0 flex-1">
                          <FontCombobox
                            value={currentFont}
                            families={families}
                            onChange={(path, _family, weight) => {
                              updateStyle({ font_family: path || null, font_weight: weight?.subfamily ?? null });
                            }}
                            placeholder="auto (best coverage)"
                            displayIdentity={displayIdentity}
                          />
                        </div>
                        <button
                          onClick={() => { onNeedFullFonts(langForInst); setShowFontManager(true); }}
                          title="browse every installed font"
                          aria-label="open font manager"
                          className="shrink-0 rounded-sm border border-transparent p-1 text-zinc-500 transition hover:border-zinc-300 hover:bg-zinc-200 hover:text-zinc-700 dark:hover:border-zinc-700 dark:hover:bg-zinc-800 dark:hover:text-zinc-300"
                        >
                          {theme === "dark" ? <MdFontDownload size={13} /> : <MdOutlineFontDownload size={13} />}
                        </button>
                      </div>
                    </div>

                    {/* Font size */}
                    <div className="md:col-span-2 md:row-start-2">
                      <div className="flex items-center gap-2">
                        <PixelField label="size" min={6} value={sp?.font_size} onChange={(value) => updateStyle({ font_size: value })} placeholder={inst.characteristics?.size != null ? `Detected: ${inst.characteristics.size}` : "Auto-fit"} />
                        {inst.characteristics?.size != null && sp?.font_size == null && (
                          <span className="subtext text-[10px] text-cyan-600/70 dark:text-cyan-400/70">detected · auto-fit</span>
                        )}
                      </div>
                    </div>

                    {/* Detected typography (from capture-time analysis) */}
                    {/* legacy detected-style block moved beside the Font label
                      <p className="subtext flex items-center gap-1.5 text-[10px] text-zinc-500 md:col-span-3 md:row-start-5">
                        <ScanText size={11} className="shrink-0 text-cyan-600 dark:text-cyan-400" />
                        detected style: <span className="font-medium text-zinc-700 dark:text-zinc-300">{inst.characteristics.font_style}</span>
                        {inst.characteristics.positioning?.rotation_deg ? (
                          <span> · rotated {inst.characteristics.positioning.rotation_deg}°</span>
                        ) : null}
                      </p>
                    */}
                    {inst.recognition_history?.length ? (
                      <p className="subtext text-[10px] text-zinc-500 md:col-span-3 md:row-start-6" title={inst.recognition_history.map((h) => `${h.engine}: ${h.reason}`).join("\n")}>
                        OCR audit: {inst.recognition_history[inst.recognition_history.length - 1]?.accepted ? "Paddle evidence accepted" : "candidate retained for review"}
                      </p>
                    ) : null}

                    {/* Text alignment: horizontal */}
                    <div className="md:col-start-3 md:row-start-1 md:justify-self-end">
                      <label className="subtext mb-1 block text-right text-xs text-zinc-500">horizontal alignment</label>
                      <div className="flex justify-end gap-1">
                        {([["left", <AlignLeft size={14} key="l" />], ["center", <AlignCenter size={14} key="c" />], ["right", <AlignRight size={14} key="r" />]] as const).map(([val, icon]) => (
                          <button
                            key={val}
                            onClick={() => updateStyle({ align_h: sp?.align_h === val ? null : val })}
                            className={`rounded-sm p-1.5 transition ${sp?.align_h === val ? "bg-cyan-600 text-white" : "bg-zinc-200 text-zinc-600 hover:bg-zinc-300 dark:bg-zinc-800 dark:text-zinc-400 dark:hover:bg-zinc-700"}`}
                            title={val}
                          >{icon}</button>
                        ))}
                      </div>
                    </div>

                    {/* Text alignment: vertical */}
                    <div className="md:col-start-3 md:row-start-2 md:justify-self-end">
                      <label className="subtext mb-1 block text-right text-xs text-zinc-500">vertical alignment</label>
                      <div className="flex justify-end gap-1">
                        {([["top", <AlignStartVertical size={14} key="t" />], ["middle", <AlignCenter size={14} key="m" />], ["bottom", <AlignEndVertical size={14} key="b" />]] as const).map(([val, icon]) => (
                          <button
                            key={val}
                            onClick={() => updateStyle({ align_v: sp?.align_v === val ? null : val })}
                            className={`rounded-sm p-1.5 transition ${sp?.align_v === val ? "bg-cyan-600 text-white" : "bg-zinc-200 text-zinc-600 hover:bg-zinc-300 dark:bg-zinc-800 dark:text-zinc-400 dark:hover:bg-zinc-700"}`}
                            title={val}
                          >{icon}</button>
                        ))}
                      </div>
                    </div>

                    {/* Word order reversal (vertical text only) */}
                   <div className="self-start md:col-start-1 md:row-start-3">
                    <label className="subtext mb-1 flex h-4 items-center text-xs text-zinc-500">word order</label>
                      <button
                        onClick={() => updateStyle({ word_order: sp?.word_order === "rtl" ? null : "rtl" })}
                        disabled={sp?.target_orientation !== "vertical"}
                        className={`flex h-8 items-center gap-1.5 rounded px-2 text-xs transition ${
                          sp?.target_orientation !== "vertical"
                            ? "cursor-not-allowed bg-zinc-100 text-zinc-300 dark:bg-zinc-900 dark:text-zinc-700"
                            : sp?.word_order === "rtl"
                              ? "bg-cyan-600 text-white"
                              : "bg-zinc-200 text-zinc-600 hover:bg-zinc-300 dark:bg-zinc-800 dark:text-zinc-400 dark:hover:bg-zinc-700"
                        }`}
                        title={sp?.target_orientation !== "vertical" ? "only available for vertical text" : sp?.word_order === "rtl" ? "right-to-left word order — click to reset" : "reverse word order (right-to-left)"}
                      >
                        <ArrowLeftRight size={14} />
                        {sp?.word_order === "rtl" ? "RTL" : "LTR"}
                      </button>
                    </div>

                    <div className="min-w-68 self-start md:col-start-2 md:row-start-3">
                      <label className="subtext mb-1 flex h-4 items-center text-xs text-zinc-500">shape (degrees / arc)</label>
                      <div className="flex">
                        {([['skew_x', 'X'], ['skew_y', 'Y'], ['arc', 'Arc']] as const).map(([key, label], index) => (
                          <div key={key} className={`grid w-20 shrink-0 grid-cols-[1.25rem_2.25rem_1rem] items-center gap-1 text-[10px] text-zinc-500${index === 1 ? " ml-2" : index === 2 ? " ml-3" : ""}`}><span className="justify-self-end text-right">{label}</span><input aria-label={`${label} shape value`} type="number" step="1" min="-25" max="25"
                              value={sp?.transform?.[key] ?? 0}
                              disabled={isTransformLocked(key)}
                              onChange={(e) => { const value = Number(e.target.value || 0); if (!isTransformLocked(key)) { trackTransformChange(key, sp?.transform?.[key] ?? 0, value); updateCanvasTransform(key, value); } }}
                              onDoubleClick={() => { if (!isTransformLocked(key)) { trackTransformChange(key, sp?.transform?.[key] ?? 0, 0); updateCanvasTransform(key, 0); } }}
                              title="double-click to reset to 0"
                              className="h-8 w-9 rounded-sm border border-zinc-300 bg-white px-1 text-xs disabled:cursor-not-allowed disabled:opacity-40 dark:border-zinc-700 dark:bg-zinc-900" /><button type="button" onClick={() => toggleTransformLock(key)} className={`flex h-4 w-4 items-center justify-center rounded-sm ${isTransformLocked(key) ? "text-cyan-600 dark:text-cyan-400" : "text-zinc-400 hover:text-zinc-700 dark:hover:text-zinc-200"}`} title={isTransformLocked(key) ? "Unlock" : "Lock"}>{isTransformLocked(key) ? <HiLockClosed size={9} /> : <HiLockOpen size={9} />}</button>
                          </div>
                        ))}
                      </div>
                    </div>

                    {/* Justification */}
                    <div className="md:col-start-3 md:row-start-3 md:justify-self-end">
                      <label className="subtext mb-1 block text-right text-xs text-zinc-500">justification</label>
                      <div className="flex justify-end gap-1">
                        {([["last_left", <AlignJustify size={14} key="ll" />], ["last_right", <AlignJustify size={14} key="lr" style={{ transform: "scaleX(-1)" }} />], ["justify", <AlignJustify size={14} key="j" />], ["justify_center", <AlignCenter size={14} key="jc" />]] as const).map(([val, icon]) => (
                          <button
                            key={val}
                            onClick={() => updateStyle({ justification: sp?.justification === val ? null : val })}
                            className={`rounded-sm p-1.5 transition ${sp?.justification === val ? "bg-cyan-600 text-white" : "bg-zinc-200 text-zinc-600 hover:bg-zinc-300 dark:bg-zinc-800 dark:text-zinc-400 dark:hover:bg-zinc-700"}`}
                            title={val}
                          >{icon}</button>
                        ))}
                      </div>
                    </div>

                    {/* Compact pixel fields */}
                    <div className="md:col-span-2 md:row-start-4">
                      <label className="subtext mb-1 block text-xs text-zinc-500">position (px)</label>
                      <div className="flex flex-wrap gap-1">
                        {([
                          ["indent", "indent"],
                          ["baseline_shift", "base"],
                        ] as const).map(([key, label]) => (
                          <div key={key} className="flex items-center gap-0.5">
                            <span className="subtext text-[10px] text-zinc-500">{label}</span>
                            <input
                              type="number"
                              step={0.5}
                              value={(sp?.[key] as number | null | undefined) ?? ""}
                              onChange={(e) => updateStyle({ [key]: e.target.value ? Number(e.target.value) : null } as Partial<NonNullable<InstText["style_profile"]>>)}
                              placeholder="—"
                              className="w-12 rounded-sm border border-zinc-300 bg-white px-1 py-0.5 text-xs text-zinc-800 dark:border-zinc-700 dark:bg-zinc-900 dark:text-zinc-200"
                            />
                          </div>
                        ))}
                      </div>
                    </div>

                    <div className="md:col-start-3 md:row-start-4 md:justify-self-end">
                      <label className="subtext mb-1 block text-xs text-zinc-500">spacing (px)</label>
                      <div className="flex flex-wrap gap-1">
                        {([["tracking", "track"], ["kerning", "kern"], ["leading", "lead"], ["tab_width", "tab"]] as const).map(([key, label]) => (
                          <div key={key} className="flex items-center gap-0.5">
                            <span className="subtext text-[10px] text-zinc-500">{label}</span>
                            <input type="number" step={0.5} value={(sp?.[key] as number | null | undefined) ?? ""} onChange={(e) => updateStyle({ [key]: e.target.value ? Number(e.target.value) : null } as Partial<NonNullable<InstText["style_profile"]>>)} placeholder="-" className="w-12 rounded-sm border border-zinc-300 bg-white px-1 py-0.5 text-xs text-zinc-800 dark:border-zinc-700 dark:bg-zinc-900 dark:text-zinc-200" />
                          </div>
                        ))}
                      </div>
                    </div>

                    {/* Tsume (Japanese only) */}
                    {isJapanese && (
                      <div>
                        <label className="subtext mb-1 block text-xs text-zinc-500">tsume (CJK compression 0.0–1.0)</label>
                        <input
                          type="number"
                          min={0}
                          max={1}
                          step={0.1}
                          value={sp?.tsume ?? ""}
                          onChange={(e) => updateStyle({ tsume: e.target.value ? Number(e.target.value) : null })}
                          placeholder="0"
                          className="w-full rounded-sm border border-zinc-300 bg-white px-2 py-1 text-xs text-zinc-800 dark:border-zinc-700 dark:bg-zinc-900 dark:text-zinc-200"
                        />
                      </div>
                    )}
                  </div>
                </div>

                {/* === APPEARANCE SECTION === */}
                <div className="border-t border-zinc-200 pt-3 dark:border-zinc-800">
                  <p className="subtext mb-2 text-[10px] font-semibold uppercase tracking-wider text-zinc-400">Appearance</p>
                  <div className="grid grid-cols-1 gap-2 md:grid-cols-2">
                    {/* Fill */}
                    <div className="grid grid-rows-[auto_1fr]">
                      <label className="subtext mb-1 block text-xs text-zinc-500">fill</label>
                      <div className="flex min-h-14 items-center gap-2">
                        <input
                          type="color"
                          value={colorPickerValue}
                          onChange={(e) => updateStyle({ color: e.target.value })}
                          className="h-7 w-10 rounded-sm border border-zinc-300 dark:border-zinc-700"
                        />
                        <button
                          onClick={() => updateStyle({ color: null })}
                          className="subtext rounded-sm px-2 py-0.5 text-xs text-zinc-500 hover:bg-zinc-200 dark:hover:bg-zinc-800"
                        >auto</button>
                        <button onClick={() => setColorPickMode((mode) => mode ? null : "active")}
                          className={`subtext flex items-center gap-1 rounded-sm px-2 py-0.5 text-xs ${colorPickMode ? "bg-cyan-600 text-white" : "text-zinc-500 hover:bg-zinc-200 dark:hover:bg-zinc-800"}`}
                          title="pick color" aria-label="pick color"><FaEyeDropper size={11} /></button>
                        <HexColorInput value={currentColor} onChange={(color) => updateStyle({ color })} />
                      </div>
                    </div>

                    {/* Stroke */}
                    <div className="grid grid-rows-[auto_1fr]">
                      <label className="subtext mb-1 block text-xs text-zinc-500">stroke</label>
                      <div className="grid min-h-14 grid-cols-[2.5rem_6rem_minmax(0,7rem)_1fr] items-center gap-2">
                        <input
                          type="color"
                          value={currentStrokeColor ?? "#000000"}
                          onChange={(e) => updateStyle({ stroke_color: e.target.value })}
                          className="h-7 w-10 rounded-sm border border-zinc-300 dark:border-zinc-700"
                        />
                        <HexColorInput value={currentStrokeColor} onChange={(color) => updateStyle({ stroke_color: color })} />
                        <PixelField label="weight" min={0} value={sp?.stroke_width} onChange={(value) => updateStyle({ stroke_width: value })} width="7rem" />
                        <div className="ml-auto grid grid-cols-2 gap-1 self-center justify-self-end" role="group" aria-label="stroke position">
                          {([
                            { value: "none", label: "no stroke", icon: CircleOff },
                            { value: "outer", label: "outer stroke", icon: Circle },
                            { value: "center", label: "center stroke", icon: CircleDashed },
                            { value: "inner", label: "inner stroke", icon: CircleDot },
                          ] as const).map(({ value, label, icon: Icon }) => {
                            const strokeEnabled = Boolean(sp?.stroke_color && sp?.stroke_width && sp.stroke_width > 0);
                            const active = value === "none"
                              ? !strokeEnabled
                              : strokeEnabled && (sp?.stroke_position ?? "outer") === value;
                            return <button
                              key={value}
                              type="button"
                              onClick={() => value === "none"
                                ? updateStyle({ stroke_color: null, stroke_width: null, stroke_position: null })
                                : updateStyle({
                                    stroke_color: sp?.stroke_color ?? currentStrokeColor ?? "#000000",
                                    stroke_width: sp?.stroke_width && sp.stroke_width > 0 ? sp.stroke_width : 1,
                                    stroke_position: value,
                                  })}
                              className={`flex items-center justify-center rounded-sm p-1.5 transition ${active ? "bg-cyan-600 text-white" : "bg-zinc-200 text-zinc-600 hover:bg-zinc-300 dark:bg-zinc-800 dark:text-zinc-400 dark:hover:bg-zinc-700"}`}
                              title={label}
                              aria-label={label}
                              aria-pressed={active}
                            ><Icon size={14} /></button>;
                          })}
                        </div>
                      </div>
                    </div>

                    {/* Toggle buttons: underline, italic, subscript, superscript */}
                    <div className="flex flex-wrap gap-1">
                      <button
                        onClick={() => updateStyle({ underline: !sp?.underline ? true : null })}
                        className={`flex items-center gap-1 rounded-sm px-2 py-1 text-xs transition ${sp?.underline ? "bg-cyan-600 text-white" : "bg-zinc-200 text-zinc-600 hover:bg-zinc-300 dark:bg-zinc-800 dark:text-zinc-400 dark:hover:bg-zinc-700"}`}
                        title="underline"
                      aria-label="underline"><Underline size={14} /></button>
                      <button
                        onClick={() => updateStyle({ italic: !sp?.italic ? true : null })}
                        className={`flex items-center gap-1 rounded-sm px-2 py-1 text-xs transition ${sp?.italic ? "bg-cyan-600 text-white" : "bg-zinc-200 text-zinc-600 hover:bg-zinc-300 dark:bg-zinc-800 dark:text-zinc-400 dark:hover:bg-zinc-700"}`}
                        title="italic"
                      aria-label="italic"><Italic size={14} /></button>
                      <button
                        onClick={() => updateStyle({ subscript: !sp?.subscript ? true : null })}
                        className={`flex items-center gap-1 rounded-sm px-2 py-1 text-xs transition ${sp?.subscript ? "bg-cyan-600 text-white" : "bg-zinc-200 text-zinc-600 hover:bg-zinc-300 dark:bg-zinc-800 dark:text-zinc-400 dark:hover:bg-zinc-700"}`}
                        title="subscript"
                      aria-label="subscript"><Subscript size={14} /></button>
                      <button
                        onClick={() => updateStyle({ superscript: !sp?.superscript ? true : null })}
                        className={`flex items-center gap-1 rounded-sm px-2 py-1 text-xs transition ${sp?.superscript ? "bg-cyan-600 text-white" : "bg-zinc-200 text-zinc-600 hover:bg-zinc-300 dark:bg-zinc-800 dark:text-zinc-400 dark:hover:bg-zinc-700"}`}
                        title="superscript"
                      aria-label="superscript"><Superscript size={14} /></button>
                    </div>
                    {sp?.underline && <div className="mt-2 flex flex-wrap items-center gap-2">
                      <PixelField label="underline offset" value={sp.underline_offset} onChange={(value) => updateStyle({ underline_offset: value })} placeholder="detected" width="9rem" />
                      <PixelField label="underline weight" min={0.5} value={sp.underline_width} onChange={(value) => updateStyle({ underline_width: value })} placeholder="detected" width="9rem" />
                    </div>}
                  </div>
                </div>

                {!brushMode && selectedRenderInst && (preRenderUrl || previewUrl || previewPending || previewRenderError) && <div className="border-t border-zinc-200 pt-3 dark:border-zinc-800">
                  <p className="subtext mb-2 text-[10px] font-semibold uppercase tracking-wider text-zinc-400">Text Warp</p>
                  <div ref={setTextWarpHost} />
                </div>}

                {/* Preview text */}
                <div className="rounded-sm bg-zinc-100 p-2 dark:bg-zinc-950">
                  <p className="subtext mb-0.5 text-[10px] uppercase tracking-wider text-zinc-500">preview</p>
                  <p className="text-sm text-zinc-800 dark:text-zinc-200" style={{
                    ...previewFontStyle,
                    fontFamily: currentFont ? fontNameForPath(currentFont) : undefined,
                    color: currentColor ?? undefined,
                    fontStyle: sp?.italic ? "italic" : previewFontStyle.fontStyle,
                    textDecoration: sp?.underline ? "underline" : undefined,
                  }}>
                    {inst.target_text || "(no translation)"}
                  </p>
                </div>
              </div>
            );
          })()}
                </div>
              </div>
            </Section>
            {!styleExpandedH && (
              <div
                className="title-drag-handle shrink-0 flex items-center justify-center"
                onMouseDown={onStyleDragStart}
                onDoubleClick={onStyleDoubleClick}
                title="drag to resize · double-click to toggle height"
                style={{ position: "relative" }}
              >
                <svg width="42" height="14" viewBox="0 0 42 14" fill="none" xmlns="http://www.w3.org/2000/svg">
                  <rect x="0.5" y="1" width="41" height="4.5" rx="2.25" fill="currentColor" />
                  <rect x="11" y="8" width="20" height="4.5" rx="2.25" fill="currentColor" />
                </svg>
              </div>
            )}
            </div>
          )}

          {manifest.length > 0 && (
            <div
              ref={garnishCardRef}
              className="bezier-card soft-shadow flex flex-col rounded-xl bg-white/60 dark:bg-zinc-900/60"
              style={garnishCardH !== null ? { height: garnishCardH, transition: "height 0.3s ease-in-out" } : { transition: "height 0.3s ease-in-out" }}
            >
              <Section
                title="Garnish"
                icon={<GiCoolSpices size={14} />}
                flat
                className={`${stackClass(1)} flex-1 overflow-auto p-5`}
                headerExtra={selectedRenderInst && <input type="checkbox" checked={selectedGarnishEnabled} onChange={(event) => setSelectedGarnishEnabled(event.target.checked)} aria-label="enable garnish" title={selectedGarnishEnabled ? "disable garnish" : "enable garnish"} className="toggle garnish-activation-toggle h-4 w-7 shrink-0 border-violet-500 bg-violet-400 checked:border-violet-500 checked:bg-violet-800 checked:text-violet-900" />}
                rightSideHandle={
                  <div className="title-drag-handle absolute right-0 top-1/2 z-20 shrink-0 -translate-y-1/2" style={{ cursor: "pointer", padding: "2px 4px" }} onClick={onGarnishExpandClick} title={garnishCardExpandedH ? "double-click to return to standard" : "double-click to expand to full width"}>
                    <svg width="14" height="42" viewBox="0 0 14 42" fill="none" xmlns="http://www.w3.org/2000/svg"><rect x="1" y="0.5" width="4.5" height="41" rx="2.25" fill="currentColor" /><rect x="8" y="11" width="4.5" height="20" rx="2.25" fill="currentColor" /></svg>
                  </div>
                }
              >
                {(() => {
                  if (!selectedRenderInst) return <p className="subtext flex items-center gap-1 text-xs text-zinc-500"><GrSelect size={14} className="text-[#2d8cf0]" />Select a region to customize treatment.</p>;
                  const perRegion = selectedRenderInst.garnish_scope === "per_region";
                  const g = selectedGarnishProfile;
                  const recommended = selectedGarnishRecommended;
                  const slider = (label: string, field: keyof NonNullable<InstText["garnish_override"]>, min: number, max: number, step: number, suffix = "", digits = 1) => {
                    const marker = recommended ? Math.max(0, Math.min(100, (Number(recommended[field]) - min) * 100 / (max - min))) : null;
                    return <GarnishSliderField key={field} label={label} value={Number(g[field])} min={min} max={max} step={step} suffix={suffix} digits={digits} marker={marker} markerLabel={recommended ? `scene recommendation: ${Number(recommended[field]).toFixed(digits)}${suffix}` : undefined} sceneLabel={recommended ? `scene: ${Number(recommended[field]).toFixed(digits)}${suffix}` : undefined} disabled={!selectedGarnishEnabled} onChange={(value) => updateSelectedGarnish({ [field]: value })} />;
                  };
                  return <div className="space-y-2">
                    <div className="relative flex flex-wrap items-center gap-2 pr-7">
                      <div className={`garnish-scope-morph relative order-1 flex items-center gap-2${!garnishCardCollapsed ? " expanded" : ""}`} ref={garnishScopeRef}>
                        <button type="button" onClick={() => setGarnishScopeOpen((value) => !value)} className="bezier-card flex w-24 items-center justify-between gap-1.5 rounded-md bg-white/60 px-2 py-1 text-[10px] text-violet-700 transition hover:bg-violet-100 dark:bg-zinc-900/60 dark:text-violet-300 dark:hover:bg-zinc-800">{perRegion ? "per region" : "all regions"}<ChevronDown size={10} className={`transition ${garnishScopeOpen ? "rotate-180" : ""}`} /></button>
                        <div className={`dropdown-morph bezier-card absolute left-0 top-full z-200 mt-1 w-24 rounded-lg bg-white p-1 dark:bg-zinc-900${garnishScopeOpen ? " expanded" : ""}`} style={garnishScopeOpen ? { boxShadow: "1px 1px 0 var(--bc-shadow), 2px 2px 6px rgba(0,0,0,0.06)" } : undefined}>
                          <button type="button" onClick={() => { setSelectedGarnishScope("whole_selection"); setGarnishScopeOpen(false); }} className={`flex w-full rounded-md px-2 py-1.5 text-[10px] transition hover:bg-zinc-100 dark:hover:bg-zinc-800 ${!perRegion ? "bg-zinc-100 font-medium text-zinc-900 dark:bg-zinc-800 dark:text-zinc-100" : "text-zinc-600 dark:text-zinc-400"}`}>all regions</button>
                          <button type="button" onClick={() => { setSelectedGarnishScope("per_region"); setGarnishScopeOpen(false); }} className={`flex w-full rounded-md px-2 py-1.5 text-[10px] transition hover:bg-zinc-100 dark:hover:bg-zinc-800 ${perRegion ? "bg-zinc-100 font-medium text-zinc-900 dark:bg-zinc-800 dark:text-zinc-100" : "text-zinc-600 dark:text-zinc-400"}`}>per region</button>
                        </div>
                        {recommended && <button type="button" onClick={useSelectedSceneGarnish} className="shrink-0 rounded-sm bg-violet-700 px-1.5 py-0.5 text-[10px] text-white">AI Preset</button>}
                      </div>
                      <div className={`garnish-controls-morph order-3 basis-full${!garnishCardCollapsed ? " expanded mt-1" : ""}`}>
                        <fieldset disabled={!selectedGarnishEnabled} className="grid w-full grid-cols-2 gap-x-8 gap-y-1 rounded-lg border border-zinc-300/80 bg-white/30 px-6 py-3 disabled:opacity-45 dark:border-zinc-700/80 dark:bg-zinc-950/20 [&>label:nth-child(odd)]:pl-2">
                          {slider("edge blur", "edge_blur_px", 0, 10, 0.1, "px")}
                          {slider("feather", "edge_smoothing_strength", 0, 1, 0.05, "", 2)}
                          {slider("wear", "erosion_px", 0, 5, 0.1, "px")}{slider("thicken", "dilation_px", 0, 5, 0.1, "px")}{slider("grain", "grain_strength", 0, 1, 0.02, "", 2)}{slider("gamma", "gamma_shift", 0.5, 2, 0.05, "", 2)}{slider("smudge", "smudge_strength", 0, 1, 0.02, "", 2)}{slider("angle", "smudge_angle_deg", 0, 360, 1, "°", 0)}
                        </fieldset>
                      </div>
                      {/* Last child, pushed right by ml-auto, so it sits at the
                          card's right edge exactly as Text & Appearance's does
                          and does not move when the controls open beside it. */}
                      <button type="button" onClick={() => setGarnishCardCollapsed((value) => { const next = !value; if (next) setGarnishScopeOpen(false); return next; })} className="absolute right-0 top-0 shrink-0 rounded-sm p-1 text-zinc-500 transition-colors hover:bg-zinc-200 dark:hover:bg-zinc-800" title={garnishCardCollapsed ? "expand garnish controls" : "collapse garnish controls"}><FcCollapse style={{ transform: garnishCardCollapsed ? "rotate(180deg)" : "none", transition: "transform 0.2s ease-in-out" }} /></button>
                    </div>
                    <div className="flex flex-wrap items-center gap-2">
                      {!recommended && <span className="text-[10px] text-violet-700/75 dark:text-violet-300/75">manual baseline</span>}
                      {garnishPreviewSyncing && <span className="text-[10px] text-violet-700 dark:text-violet-300">updating treatment…</span>}
                    </div>
                    {/* Smart Fill belongs to the Garnish body, so it discloses
                        with the rest of it rather than sitting under a
                        collapsed card. */}
                    <div className={`style-panel-morph${!garnishCardCollapsed ? " expanded" : ""}`}>
                      <SmartFillReview reviews={repairReviews} fallbackIds={repairFallbackIds} previews={localizedCandidatePreviews} previewPending={previewPending} appliedIds={appliedCandidateIds} onRetryPreview={() => setCandidatePreviewRevision((value) => value + 1)} onApply={applyReviewCandidate} onApplyAll={applyAllReviewCandidates} onHoverRegion={setSmartFillHoverId} />
                    </div>
                  </div>;
                })()}
              </Section>
              {!garnishCardExpandedH && <div className="title-drag-handle flex shrink-0 items-center justify-center" onMouseDown={onGarnishDragStart} onDoubleClick={onGarnishDoubleClick} title="drag to resize · double-click to toggle height" style={{ position: "relative" }}><svg width="42" height="14" viewBox="0 0 42 14" fill="none" xmlns="http://www.w3.org/2000/svg"><rect x="0.5" y="1" width="41" height="4.5" rx="2.25" fill="currentColor" /><rect x="11" y="8" width="20" height="4.5" rx="2.25" fill="currentColor" /></svg></div>}
            </div>
          )}
            </div>

            {/* Source reference + the single editable localized canvas. */}
            <div className={styleExpandedH || garnishCardExpandedH ? "grid grid-cols-2 gap-4" : "flex flex-col gap-4"}>
              {previewUrl && <Section title="Source Reference" icon={<TbPhoto size={14} />} className={`${stackClass(2)} ${sourceCanvasFirst ? "order-1" : "order-2"}`}
                rightSideHandle={canSwapCanvasCards ? <div className="absolute right-5 top-4 flex items-center gap-1"><button type="button" onClick={() => setCanvasCardOrder((order) => order === "source-first" ? "localized-first" : "source-first")} title={sourceCanvasFirst ? "Move source reference below the localized canvas" : "Move source reference above the localized canvas"} aria-label={sourceCanvasFirst ? "move source reference down" : "move source reference up"} className="rounded-sm p-1 text-zinc-500 hover:bg-zinc-200 dark:hover:bg-zinc-800">{theme === "dark" ? (sourceCanvasFirst ? <BsArrowDownSquareFill size={16} /> : <BsArrowUpSquareFill size={16} />) : (sourceCanvasFirst ? <LuSquareArrowDown size={16} /> : <LuSquareArrowUp size={16} />)}</button></div> : undefined}>
                <div className="relative inline-block max-w-full">
                  <img src={previewUrl} alt="source reference" className={`block max-w-full rounded-lg border border-zinc-300 dark:border-zinc-800 ${colorPickMode ? "cursor-crosshair" : ""}`}
                    onPointerDown={(event) => { sampleCanvasFill(event, "source"); }} />
                  {imgDim && (renderSelId || smartFillHoverId) && <svg className="pointer-events-none absolute inset-0 h-full w-full" viewBox={`0 0 ${imgDim[0]} ${imgDim[1]}`} preserveAspectRatio="none">
                    {renderSelId && (() => {
                      const inst = visibleManifest.find((item) => item.id === renderSelId);
                      if (!inst) return null;
                      const b = inst.bounding_box;
                      return <rect x={b.x} y={b.y} width={b.width} height={b.height} fill="none" stroke="#06b6d4" strokeWidth="2" pointerEvents="none" />;
                    })()}
                    {smartFillHoverId && (() => {
                      const inst = visibleManifest.find((item) => item.id === smartFillHoverId);
                      if (!inst) return null;
                      const b = inst.bounding_box;
                      const perimeter = Math.max(1, 2 * (b.width + b.height));
                      return <rect x={b.x} y={b.y} width={b.width} height={b.height} fill="none" stroke="#f59e0b" strokeWidth="2" strokeDasharray="4 2" pointerEvents="none">
                        <animate attributeName="stroke-dashoffset" from="0" to={-perimeter} dur="15s" repeatCount="indefinite" />
                      </rect>;
                    })()}
                  </svg>}
                </div>
              </Section>}

              {(preRenderUrl || previewUrl || previewPending || previewRenderError) && <Section title="Localized Asset" icon={<TbPhotoEdit size={14} />} className={`${stackClass(3)} ${sourceCanvasFirst ? "order-2" : "order-1"}`} localized
                headerExtra={previewSyncing && <span className="ml-2 flex items-center gap-1 normal-case text-xs text-cyan-600 dark:text-cyan-400"><SquareLoader size="xs" /> trimming...</span>}
                rightSideHandle={<div className="absolute right-5 top-4 flex items-center gap-1">{canSwapCanvasCards && <button type="button" onClick={() => setCanvasCardOrder((order) => order === "source-first" ? "localized-first" : "source-first")} title={sourceCanvasFirst ? "Move localized canvas above the source reference" : "Move localized canvas below the source reference"} aria-label={sourceCanvasFirst ? "move localized canvas up" : "move localized canvas down"} className="rounded-sm p-1 text-zinc-500 hover:bg-zinc-200 dark:hover:bg-zinc-800">{theme === "dark" ? (sourceCanvasFirst ? <BsArrowUpSquareFill size={16} /> : <BsArrowDownSquareFill size={16} />) : (sourceCanvasFirst ? <LuSquareArrowUp size={16} /> : <LuSquareArrowDown size={16} />)}</button>}<button onClick={resetLocalizedCanvas} title="Reset all localized canvas edits to the Render-entry baseline" className="flex items-center gap-1 rounded-sm px-2 py-1 text-xs text-zinc-500 hover:bg-zinc-200 dark:hover:bg-zinc-800"><VscDebugRestart size={16} /> reset</button></div>}>
                <div className="mb-2 flex items-center gap-2">
                  <p className="text-xs text-zinc-500">Modify, place, and warp text.</p>
                </div>
                <div className="relative">
                <div ref={localizedScrollRef} className="max-h-[68vh] max-w-full overflow-auto rounded-lg" style={{ cursor: localizedDragMode ? (localizedPanRef.current ? "grabbing" : "grab") : undefined }} onMouseDown={(e) => { if (!localizedDragMode) return; const el = localizedScrollRef.current; if (!el) return; localizedPanRef.current = { startX: e.clientX, startY: e.clientY, scrollLeft: el.scrollLeft, scrollTop: el.scrollTop }; e.preventDefault(); }} onMouseMove={(e) => { if (!localizedPanRef.current) return; const el = localizedScrollRef.current; if (!el) return; el.scrollLeft = localizedPanRef.current.scrollLeft - (e.clientX - localizedPanRef.current.startX); el.scrollTop = localizedPanRef.current.scrollTop - (e.clientY - localizedPanRef.current.startY); }} onMouseUp={() => { localizedPanRef.current = null; }} onMouseLeave={() => { localizedPanRef.current = null; }}>
                <div className="relative inline-block min-w-full touch-none select-none" style={{ width: `${localizedZoom * 100}%` }}>
                  <img ref={localizedImageRef} src={preRenderUrl || previewUrl || "data:image/gif;base64,R0lGODlhAQABAIAAAAAAAP///ywAAAAAAQABAAACAUwAOw=="} alt="localized treatment canvas" draggable={false} className={`block w-full max-w-none touch-none select-none rounded-lg border border-zinc-300 dark:border-zinc-800 ${localizedDragMode ? "pointer-events-none" : nearFirstPoint ? "cursor-pointer" : (brushMode || lassoMode || colorPickMode) ? "cursor-crosshair" : ""}`}
                    style={{ touchAction: "none", WebkitUserDrag: "none" } as React.CSSProperties}
                    onDragStart={(event) => event.preventDefault()}
                    onPointerDown={(event) => {
                      if (sampleCanvasFill(event, "localized")) return;
                      if (brushMode) beginBrushStroke(event);
                      else if (lassoMode) {
                        if (nearFirstPoint && lassoPoints.length >= 3) { event.preventDefault(); return; }
                        const point = pointOnLocalizedCanvas(event);
                        if (point) { event.preventDefault(); setLassoPoints((points) => [...points, point]); }
                      }
                    }}
                    onPointerMove={(event) => {
                      if (brushMode) extendBrushStroke(event);
                      else if (lassoMode) {
                        const point = pointOnLocalizedCanvas(event);
                        if (point) {
                          setBrushCursor(point);
                          if (lassoPoints.length >= 3) {
                            const rect = event.currentTarget.getBoundingClientRect();
                            const scale = imgDim ? imgDim[0] / rect.width : 1;
                            const threshold = 15 * scale;
                            const dx = point[0] - lassoPoints[0][0];
                            const dy = point[1] - lassoPoints[0][1];
                            setNearFirstPoint(Math.sqrt(dx * dx + dy * dy) < threshold);
                          } else setNearFirstPoint(false);
                        }
                      }
                    }}
                    onPointerUp={(event) => finishBrushStroke(event)}
                    onPointerCancel={(event) => finishBrushStroke(event, true)}
                    onLostPointerCapture={(event) => finishBrushStroke(event, true)}
                    onPointerLeave={() => { if (!brushDrawing.current) setBrushCursor(null); setNearFirstPoint(false); }} />
                  {imgDim && <svg data-perspective-overlay className={`absolute inset-0 h-full w-full ${brushMode || lassoMode ? "pointer-events-none" : ""}`} viewBox={`0 0 ${imgDim[0]} ${imgDim[1]}`} preserveAspectRatio="none">
                    {!brushMode && !lassoMode && <>
                      {/* Hit-testing layer stays invisible so UI outlines never mask glyph pixels. */}
                      {!perspectiveMode && visibleManifest.map((inst) => { const b = inst.bounding_box; return <rect key={inst.id} x={b.x} y={b.y} width={b.width} height={b.height} fill="transparent" stroke="none" onClick={() => setRenderSelId(inst.id)} className="cursor-pointer" />; })}
                      {renderSelId && (() => {
                        const inst = visibleManifest.find((item) => item.id === renderSelId);
                        if (!inst) return null;
                        if (perspectiveMode) {
                          const quad = parseQuad(inst.style_profile?.transform?.quad);
                          return <PerspectiveHandles
                            bbox={inst.bounding_box}
                            quad={quad}
                            displayScale={localizedDisplayScale}
                            imageRect={() => localizedImageRef.current?.getBoundingClientRect() ?? null}
                            onGestureStart={beginQuadGesture}
                            onGestureChange={previewQuadGesture}
                            onGestureEnd={finishQuadGesture}
                          />;
                        }
                        const b = inst.bounding_box;
                        const corners: Array<{ cx: number; cy: number; d: string }> = [
                          { cx: b.x, cy: b.y, d: "M0,4 L0,0 L4,0" },
                          { cx: b.x + b.width, cy: b.y, d: "M0,0 L-4,0 L0,4" },
                          { cx: b.x, cy: b.y + b.height, d: "M0,0 L0,-4 L4,0" },
                          { cx: b.x + b.width, cy: b.y + b.height, d: "M0,0 L-4,0 L0,-4" },
                        ];
                        return <>
                          <rect x={b.x} y={b.y} width={b.width} height={b.height} fill="rgba(6,182,212,.06)" stroke="#06b6d4" strokeWidth="1.5" pointerEvents="none" />
                          {corners.map((corner, index) => <path key={index} d={corner.d} transform={`translate(${corner.cx},${corner.cy})`} stroke="#06b6d4" strokeWidth="2" fill="none" pointerEvents="none" />)}
                        </>;
                      })()}
                      {smartFillHoverId && (() => {
                        const inst = visibleManifest.find((item) => item.id === smartFillHoverId);
                        if (!inst) return null;
                        const b = inst.bounding_box;
                        const perimeter = Math.max(1, 2 * (b.width + b.height));
                        return <rect x={b.x} y={b.y} width={b.width} height={b.height} fill="rgba(245,158,11,.12)" stroke="#f59e0b" strokeWidth="2" strokeDasharray="4 2" pointerEvents="none">
                          <animate attributeName="stroke-dashoffset" from="0" to={-perimeter} dur="15s" repeatCount="indefinite" />
                        </rect>;
                      })()}
                    </>}
                    {brushStrokes.map((stroke) => <polyline key={stroke.id} points={stroke.points.map((p) => p.join(",")).join(" ")} fill="none" stroke="#06b6d4" strokeWidth={brushRadius * 2} strokeLinecap="round" strokeLinejoin="round" opacity=".45" />)}
                    {activeBrushStroke && <polyline points={activeBrushStroke.points.map((p) => p.join(",")).join(" ")} fill="none" stroke="#06b6d4" strokeWidth={brushRadius * 2} strokeLinecap="round" strokeLinejoin="round" opacity=".70" />}
                    {brushMode && brushCursor && <circle cx={brushCursor[0]} cy={brushCursor[1]} r={brushRadius} fill="rgba(6,182,212,.10)" stroke="#06b6d4" strokeWidth="1.5" />}
                    {lassoPoints.length > 0 && <>
                      <polyline points={[...lassoPoints, ...(lassoMode && brushCursor ? [brushCursor] : [])].map((p) => p.join(",")).join(" ")} fill="rgba(6,182,212,.15)" stroke="#06b6d4" strokeWidth="2" strokeDasharray={lassoMode && brushCursor ? "5 3" : undefined} />
                      {lassoPoints.map((point, index) => <circle key={`${point[0]}-${point[1]}-${index}`} cx={point[0]} cy={point[1]} r="3" fill="#06b6d4" stroke="white" strokeWidth="1" />)}
                      {nearFirstPoint && lassoPoints.length >= 3 && <circle cx={lassoPoints[0][0]} cy={lassoPoints[0][1]} r="8" fill="none" stroke="#06b6d4" strokeWidth="2" strokeDasharray="3 2"><animate attributeName="r" values="6;10;6" dur="1s" repeatCount="indefinite" /></circle>}
                    </>}
                  </svg>}
                </div>
                </div>
                <div className="absolute bottom-2 left-2 flex gap-1 z-20 w-fit">
                  <button onClick={() => {
                    const el = localizedScrollRef.current;
                    if (!el) { setLocalizedZoom((z) => Math.max(0.25, Number((z - 0.25).toFixed(2)))); return; }
                    const fracX = el.scrollLeft / Math.max(1, el.scrollWidth);
                    const fracY = el.scrollTop / Math.max(1, el.scrollHeight);
                    setLocalizedZoom((z) => Math.max(0.25, Number((z - 0.25).toFixed(2))));
                    requestAnimationFrame(() => { const ne = localizedScrollRef.current; if (ne) { ne.scrollLeft = fracX * ne.scrollWidth; ne.scrollTop = fracY * ne.scrollHeight; } });
                  }} className="flex items-center rounded-sm bg-white px-2 py-1 text-xs text-zinc-700 hover:bg-zinc-200 dark:bg-zinc-800 dark:text-zinc-300 dark:hover:bg-zinc-700">
                    <TbCircleDashedMinus size={12} />
                  </button>
                  <span className="rounded-sm bg-white px-2 py-1 text-xs text-zinc-600 dark:bg-zinc-800 dark:text-zinc-400">{(localizedZoom * 100).toFixed(0)}%</span>
                  <button onClick={() => {
                    const el = localizedScrollRef.current;
                    if (!el) { setLocalizedZoom((z) => Math.min(4, Number((z + 0.25).toFixed(2)))); return; }
                    const fracX = el.scrollLeft / Math.max(1, el.scrollWidth);
                    const fracY = el.scrollTop / Math.max(1, el.scrollHeight);
                    setLocalizedZoom((z) => Math.min(4, Number((z + 0.25).toFixed(2))));
                    requestAnimationFrame(() => { const ne = localizedScrollRef.current; if (ne) { ne.scrollLeft = fracX * ne.scrollWidth; ne.scrollTop = fracY * ne.scrollHeight; } });
                  }} className="flex items-center rounded-sm bg-white px-2 py-1 text-xs text-zinc-700 hover:bg-zinc-200 dark:bg-zinc-800 dark:text-zinc-300 dark:hover:bg-zinc-700">
                    <TbCircleDashedPlus size={12} />
                  </button>
                  <button onClick={() => setLocalizedZoom(1)} className="rounded-sm bg-white px-2 py-1 text-xs text-zinc-700 hover:bg-zinc-200 dark:bg-zinc-800 dark:text-zinc-300 dark:hover:bg-zinc-700">fit</button>
                  <button
                    onClick={() => setLocalizedDragMode((v) => !v)}
                    disabled={localizedZoom <= 1}
                    className={`flex items-center rounded-sm px-2 py-1 text-xs ${localizedDragMode ? "bg-cyan-900/70 text-cyan-300" : localizedZoom <= 1 ? "bg-white text-zinc-300 dark:bg-zinc-800 dark:text-zinc-600" : "bg-white text-zinc-500 hover:bg-zinc-200 dark:bg-zinc-800 dark:text-zinc-400 dark:hover:bg-zinc-700"}`}
                    title={localizedZoom <= 1 ? "Zoom in first to enable canvas panning" : localizedDragMode ? "Drag mode active — click to deactivate" : "Activate drag mode to pan the canvas"}
                  >
                    {localizedDragMode ? <PiHandGrabbingFill size={12} /> : <PiHandGrabbingBold size={12} />}
                  </button>
                  <button
                    type="button"
                    onClick={() => setPerspectiveMode((value) => !value)}
                    disabled={!renderSelId || localizedDragMode || brushMode || lassoMode || Boolean(colorPickMode)}
                    className={`flex items-center gap-1 rounded-sm px-2 py-1 text-xs ${
                      perspectiveMode
                        ? "bg-cyan-900/70 text-cyan-300"
                        : !renderSelId || localizedDragMode || brushMode || lassoMode || colorPickMode
                          ? "bg-white text-zinc-300 dark:bg-zinc-800 dark:text-zinc-600"
                          : "bg-white text-zinc-500 hover:bg-zinc-200 dark:bg-zinc-800 dark:text-zinc-400 dark:hover:bg-zinc-700"
                    }`}
                    title={!renderSelId
                      ? "Select a region to edit its perspective"
                      : localizedDragMode || brushMode || lassoMode || colorPickMode
                        ? "Finish the active canvas tool first"
                        : perspectiveMode ? "Exit four-corner perspective editing" : "Edit four-corner perspective"}
                  >
                    <GrSelect size={12} /> perspective
                  </button>
                </div>
                {/* render metadata indicators — bottom-right of localized canvas */}
                <div className="absolute bottom-2 right-2 z-20 flex max-w-[60%] items-center gap-2 rounded-lg bg-white/80 px-2.5 py-1 text-[10px] backdrop-blur-sm dark:bg-zinc-900/80">
                  <span className={`font-medium ${translatedCount === translatableCount && translatableCount > 0 ? "text-emerald-600 dark:text-emerald-400" : "text-red-500 dark:text-red-400"}`}>
                    {translatedCount}/{translatableCount}
                  </span>
                  <span className="text-zinc-300 dark:text-zinc-600">·</span>
                  <span className="flex items-center gap-1 text-cyan-700 dark:text-cyan-300">
                    <ScanText size={11} className="shrink-0" />
                    {[...new Set(manifest.filter((i) => !i.dnt).map((i) => i.target_language ?? targLang))]
                      .map((l) => langDisplayName(l)).join(", ") || "—"}
                  </span>
                  {/* No font list here. It printed "family (id)" for every
                      region, so a sign with a dozen regions pushed a dozen
                      family names across the canvas and the two readings that
                      matter -- how many are translated, and into what -- were
                      lost in it. Per-region font evidence belongs in the
                      Translate table, which has room to show it properly. */}
                  {manifest.some((i) => i.dnt) && (
                    <>
                      <span className="text-zinc-300 dark:text-zinc-600">·</span>
                      <span className="text-amber-600 dark:text-amber-400">{manifest.filter((i) => i.dnt).length} DNT</span>
                    </>
                  )}
                </div>
                </div>
                {!brushMode && selectedRenderInst && (() => {
                  const transform = selectedRenderInst.style_profile?.transform ?? {};
                  const parsedQuad = parseQuad(transform.quad);
                  const perspectiveActive = Boolean(
                    parsedQuad
                    && !isIdentityQuad(parsedQuad)
                    && isUsableQuad(denormaliseQuad(parsedQuad, selectedRenderInst.bounding_box))
                  );
                  const detectedQuad = Number(selectedRenderInst.segmentation_mask?.confidence ?? 0) >= 0.5
                    ? quadFromPolygon(selectedRenderInst.segmentation_mask?.polygon, selectedRenderInst.bounding_box)
                    : null;
                  const allTransformLocksActive = LOCKABLE_TRANSFORM_KEYS.every((key) => isTransformLocked(key));
                  // Nine numeric controls wrap into one row, so "has this
                  // region been transformed at all, and how much of it is
                  // mine?" was not answerable without reading every value.
                  const changedCount = ([
                    ["skew_x", 0], ["skew_y", 0], ["scale_x", 1], ["scale_y", 1],
                    ["offset_x", 0], ["offset_y", 0], ["rotation", 0], ["arc", 0],
                  ] as const).filter(([key, base]) => Number(transform?.[key] ?? base) !== base).length
                    + (perspectiveActive ? 1 : 0);
                  return textWarpHost ? createPortal(<div className="text-warp-controls space-y-2 text-xs text-cyan-900 dark:text-cyan-100" onInputCapture={(event) => { const target = event.target; if (target instanceof HTMLInputElement && target.type === "range") pulseRangeStep(target); }}>
                    <div className="flex flex-wrap items-center gap-2">
                      <div className="relative shrink-0" ref={warpRef}><button onClick={() => setWarpOpen((value) => !value)} className="bezier-card flex items-center gap-1.5 rounded-md bg-white/60 px-2 py-1 text-xs text-zinc-700 transition hover:bg-zinc-100 dark:bg-zinc-900/60 dark:text-zinc-300 dark:hover:bg-zinc-800">{WARP_PRESETS.find((preset) => preset.value === (transform.preset ?? "custom"))?.label ?? "Custom"}<ChevronDown size={12} className={`transition ${warpOpen ? "rotate-180" : ""}`} /></button><div className={`dropdown-morph bezier-card absolute left-0 top-full z-200 mt-1 w-40 rounded-lg bg-white p-1 dark:bg-zinc-900${warpOpen ? " expanded" : ""}`} style={warpOpen ? { boxShadow: "1px 1px 0 var(--bc-shadow), 2px 2px 6px rgba(0,0,0,0.06)" } : undefined}>{WARP_PRESETS.map((preset) => <button key={preset.value} onClick={() => { applyCanvasWarpPreset(preset.value); setWarpOpen(false); }} className={`flex w-full items-center justify-between gap-2 rounded-md px-2 py-1.5 text-xs transition hover:bg-zinc-100 dark:hover:bg-zinc-800 ${(transform.preset ?? "custom") === preset.value ? "bg-zinc-100 font-medium text-zinc-900 dark:bg-zinc-800 dark:text-zinc-100" : "text-zinc-600 dark:text-zinc-400"}`}><span>{preset.label}</span>{preset.value !== "none" && preset.value !== "custom" && <WarpPreview preset={preset.value} />}</button>)}</div></div>
                      <button type="button" onClick={toggleAllTransformLocks} className="rounded-sm p-0.5 transition hover:scale-110" title={allTransformLocksActive ? "Unlock all transform values" : "Lock all transform values"}>{allTransformLocksActive ? <HiLockClosed size={12} className="text-cyan-600 dark:text-cyan-400" /> : <HiLockOpen size={12} className="text-zinc-400 dark:text-zinc-500" />}</button>
                      <button type="button" disabled={!detectedQuad} onClick={() => detectedQuad && updateSelectedStyle({ transform: { ...transform, quad: detectedQuad, preset: "custom" } })} className="rounded-sm border border-cyan-300 px-1.5 py-0.5 text-[10px] text-cyan-700 disabled:cursor-not-allowed disabled:opacity-35 dark:border-cyan-800 dark:text-cyan-300" title={detectedQuad ? "Seed perspective from the detected text outline" : "No sufficiently confident, usable text outline is available"}>from detection</button>
                      <button type="button" disabled={!parsedQuad || isIdentityQuad(parsedQuad)} onClick={clearSelectedQuad} className="rounded-sm border border-zinc-300 px-1.5 py-0.5 text-[10px] text-zinc-600 disabled:cursor-not-allowed disabled:opacity-35 dark:border-zinc-700 dark:text-zinc-400">reset perspective</button>
                      <button
                        type="button"
                        disabled={changedCount === 0 || visibleManifest.length < 2}
                        onClick={applyTransformToAllRegions}
                        className="flex items-center gap-1 rounded-sm border border-cyan-300 px-1.5 py-0.5 text-[10px] text-cyan-700 disabled:cursor-not-allowed disabled:opacity-35 dark:border-cyan-800 dark:text-cyan-300"
                        title={changedCount === 0
                          ? "Set a skew, rotation, stretch or offset first"
                          : visibleManifest.length < 2
                            ? "There is no other region to apply it to"
                            : "Give every other region this skew, rotation, stretch and offset. Perspective corners stay per-region, and locked values are left alone."}
                      >
                        <PiArrowsMergeBold size={11} /> apply to all
                      </button>
                      {changedCount > 0 && (
                        <span
                          className="rounded-full bg-cyan-100 px-1.5 py-0.5 text-[10px] font-medium text-cyan-800 dark:bg-cyan-950 dark:text-cyan-300"
                          title={`${changedCount} transform value${changedCount === 1 ? "" : "s"} differ${changedCount === 1 ? "s" : ""} from default on this region`}
                        >
                          {changedCount} changed
                        </span>
                      )}
                      <button onClick={() => setWarpCollapsed((value) => !value)} className="ml-auto shrink-0 rounded-sm p-1 text-zinc-500 transition hover:bg-zinc-200 dark:hover:bg-zinc-800" title={warpCollapsed ? "expand" : "collapse"}><FcCollapse style={{ transform: warpCollapsed ? "rotate(180deg)" : "none", transition: "transform 0.2s" }} /></button>
                      <div className={`style-panel-morph style-panel-overflow-visible flex-1 ${!warpCollapsed ? "expanded" : ""}`}>
                        <div className="flex flex-wrap items-center gap-2 pt-1">
                      {perspectiveActive && <span className="basis-full text-[10px] text-cyan-700 dark:text-cyan-300">Perspective corners replace skew and stretch.</span>}
                      {/* Corner coordinates. Every other control here pairs a
                        * slider with a number you can read and type into; the
                        * quad had neither, so a perspective could only be set
                        * by dragging and could not be read back, matched to
                        * another region, or nudged to a round number.
                        *
                        * Shown in IMAGE PIXELS, which is what the canvas shows
                        * and the handles move through, though the manifest
                        * stores the corners normalised to the bbox so they
                        * survive the region moving or resizing. An edit that
                        * would fold the quad (isUsableQuad rejects a
                        * self-intersecting or degenerate one) is ignored --
                        * the same guard the drag path answers to. */}
                      {perspectiveActive && parsedQuad && (() => {
                        const bbox = selectedRenderInst.bounding_box;
                        const pixels = denormaliseQuad(parsedQuad, bbox);
                        const setCorner = (index: number, axis: 0 | 1, raw: string) => {
                          const value = Number(raw);
                          if (!Number.isFinite(value)) return;
                          const next = pixels.map((point, i) =>
                            i === index
                              ? (axis === 0 ? [value, point[1]] : [point[0], value])
                              : point) as Quad;
                          if (!isUsableQuad(next)) return;
                          updateSelectedStyle({
                            transform: { ...transform, quad: normaliseQuad(next, bbox), preset: "custom" },
                          });
                        };
                        return (
                          <span className="basis-full flex flex-wrap items-center gap-x-3 gap-y-1 text-[10px] text-zinc-500">
                            {(["TL", "TR", "BR", "BL"] as const).map((corner, index) => (
                              <span key={corner} className="flex items-center gap-1">
                                <span className="w-5 font-mono text-cyan-700 dark:text-cyan-300">{corner}</span>
                                {([0, 1] as const).map((axis) => (
                                  <input
                                    key={axis}
                                    type="number"
                                    step={1}
                                    aria-label={`perspective corner ${corner} ${axis === 0 ? "x" : "y"}`}
                                    value={Math.round(pixels[index][axis])}
                                    onChange={(event) => setCorner(index, axis, event.target.value)}
                                    className="w-14 rounded-sm border border-zinc-300 bg-white px-1 py-0.5 text-right font-mono text-[10px] dark:border-zinc-700 dark:bg-zinc-900"
                                  />
                                ))}
                              </span>
                            ))}
                          </span>
                        );
                      })()}
                      {transform?.preset && transform.preset !== "none" && transform.preset !== "custom" && <label className="subtext flex items-center gap-1 text-xs text-zinc-500">amount<span className="warp-slider" style={{ "--warp-default": 0.74 } as React.CSSProperties}><input aria-label="warp amount" type="range" min="-25" max="25" step="0.5" value={transform.amount ?? 12} onPointerDown={(e) => snapRangePointer(e, (value) => updateSelectedStyle({ transform: { ...transform, amount: value } }))} onChange={(e) => { pulseRangeStep(e.currentTarget); updateSelectedStyle({ transform: { ...transform, amount: Number(e.target.value) } }); }} onDoubleClick={() => updateSelectedStyle({ transform: { ...transform, amount: 12 } })} title="double-click to return to the preset baseline" /></span><span className="min-w-9 text-right font-mono text-[10px]">{Number(transform.amount ?? 12).toFixed(1)}</span></label>}
                      {([['skew_x', 'X'], ['skew_y', 'Y']] as const).map(([key, label]) => {
                        const hist = transformHistoryRef.current[key];
                        const locked = isTransformLocked(key);
                        const canUndoT = !!(hist && hist.undoStack.length > 0);
                        const canRedoT = !!(hist && hist.redoStack.length > 0);
                        return <label key={key} className="subtext flex items-center gap-1 text-xs text-zinc-500">{label}<span className="text-[10px]">−45</span><span className="warp-slider" style={{ "--warp-default": 0.5 } as React.CSSProperties}><input aria-label={`${label} warp`} type="range" min="-45" max="45" step="0.5" value={transform?.[key] ?? 0} disabled={locked} onPointerDown={(e) => snapRangePointer(e, (v) => { trackTransformChange(key, transform?.[key] ?? 0, v); updateCanvasTransform(key, v); })} onChange={(e) => { pulseRangeStep(e.currentTarget); const v = Number(e.target.value); trackTransformChange(key, transform?.[key] ?? 0, v); updateCanvasTransform(key, v); }} onDoubleClick={() => { trackTransformChange(key, transform?.[key] ?? 0, 0); updateCanvasTransform(key, 0); }} title="double-click to reset to 0" /></span><input type="number" min="-45" max="45" step="0.5" value={transform?.[key] ?? 0} disabled={locked} onChange={(e) => { const v = e.target.value === "" ? 0 : Number(e.target.value); trackTransformChange(key, transform?.[key] ?? 0, v); updateCanvasTransform(key, v); }} className="w-12 rounded-sm border border-zinc-300 bg-white px-1 py-0.5 text-right font-mono text-[10px] disabled:opacity-40 dark:border-zinc-700 dark:bg-zinc-900" /><span className="text-[10px]">°</span><button type="button" onClick={() => undoTransformKey(key, 0)} disabled={!canUndoT || locked} className="rounded-sm p-0.5 text-zinc-400 hover:text-zinc-700 disabled:opacity-20 dark:hover:text-zinc-200" title="undo"><LuUndo2 size={10} /></button><button type="button" onClick={() => redoTransformKey(key)} disabled={!canRedoT || locked} className="rounded-sm p-0.5 text-zinc-400 hover:text-zinc-700 disabled:opacity-20 dark:hover:text-zinc-200" title="redo"><LuRedo2 size={10} /></button><button type="button" onClick={() => toggleTransformLock(key)} className={`rounded-sm p-0.5 transition ${locked ? "text-cyan-600 dark:text-cyan-400" : "text-zinc-400 hover:text-zinc-700 dark:hover:text-zinc-200"}`} title={locked ? "Unlock" : "Lock"}>{locked ? <HiLockClosed size={10} /> : <HiLockOpen size={10} />}</button></label>;
                      })}
                      {([['scale_x', 'width'], ['scale_y', 'height']] as const).map(([key, label]) => { const hist = transformHistoryRef.current[key]; const locked = isTransformLocked(key); const canUndoT = !!(hist && hist.undoStack.length > 0); const canRedoT = !!(hist && hist.redoStack.length > 0); return <label key={key} className="subtext flex items-center gap-1 text-xs text-zinc-500">{label}<span className="text-[10px]">0.5x</span><span className="warp-slider" style={{ "--warp-default": 0.5 } as React.CSSProperties}><input aria-label={`${label} stretch`} type="range" min="0.5" max="1.5" step="0.01" value={transform?.[key] ?? 1} disabled={locked} onPointerDown={(e) => snapRangePointer(e, (v) => { trackTransformChange(key, transform?.[key] ?? 1, v); updateCanvasTransform(key, v); })} onChange={(e) => { const v = Number(e.target.value); trackTransformChange(key, transform?.[key] ?? 1, v); updateCanvasTransform(key, v); }} onDoubleClick={() => { trackTransformChange(key, transform?.[key] ?? 1, 1); updateCanvasTransform(key, 1); }} title="double-click to reset to 1.00x" /></span><input type="number" min="0.5" max="1.5" step="0.01" value={transform?.[key] ?? 1} disabled={locked} onChange={(e) => { const v = e.target.value === "" ? 1 : Number(e.target.value); trackTransformChange(key, transform?.[key] ?? 1, v); updateCanvasTransform(key, v); }} className="w-12 rounded-sm border border-zinc-300 bg-white px-1 py-0.5 text-right font-mono text-[10px] disabled:opacity-40 dark:border-zinc-700 dark:bg-zinc-900" /><span className="text-[10px]">x</span><button type="button" onClick={() => undoTransformKey(key, 1)} disabled={!canUndoT || locked} className="rounded-sm p-0.5 text-zinc-400 hover:text-zinc-700 disabled:opacity-20 dark:hover:text-zinc-200" title="undo"><LuUndo2 size={10} /></button><button type="button" onClick={() => redoTransformKey(key)} disabled={!canRedoT || locked} className="rounded-sm p-0.5 text-zinc-400 hover:text-zinc-700 disabled:opacity-20 dark:hover:text-zinc-200" title="redo"><LuRedo2 size={10} /></button><button type="button" onClick={() => toggleTransformLock(key)} className={`rounded-sm p-0.5 transition ${locked ? "text-cyan-600 dark:text-cyan-400" : "text-zinc-400 hover:text-zinc-700 dark:hover:text-zinc-200"}`} title={locked ? "Unlock" : "Lock"}>{locked ? <HiLockClosed size={10} /> : <HiLockOpen size={10} />}</button></label>; })}
                      {([['offset_x', 'pos X'], ['offset_y', 'pos Y']] as const).map(([key, label]) => { const hist = transformHistoryRef.current[key]; const locked = isTransformLocked(key); const canUndoT = !!(hist && hist.undoStack.length > 0); const canRedoT = !!(hist && hist.redoStack.length > 0); return <span key={key} className="subtext flex items-center gap-1 text-xs text-zinc-500"><label className="flex items-center gap-1">{label}<span className="text-[10px]">−50</span><span className="warp-slider" style={{ "--warp-default": 0.5 } as React.CSSProperties}><input aria-label={`${label} position`} type="range" min="-50" max="50" step="1" value={transform?.[key] ?? 0} disabled={locked} onPointerDown={(e) => snapRangePointer(e, (v) => { trackTransformChange(key, transform?.[key] ?? 0, v); updateCanvasTransform(key, v); })} onChange={(e) => { const v = Number(e.target.value); trackTransformChange(key, transform?.[key] ?? 0, v); updateCanvasTransform(key, v); }} onDoubleClick={() => { trackTransformChange(key, transform?.[key] ?? 0, 0); updateCanvasTransform(key, 0); }} title="double-click to snap to original position" /></span><input type="number" min="-50" max="50" step="1" value={transform?.[key] ?? 0} disabled={locked} onChange={(e) => { const v = e.target.value === "" ? 0 : Number(e.target.value); trackTransformChange(key, transform?.[key] ?? 0, v); updateCanvasTransform(key, v); }} className="w-12 rounded-sm border border-zinc-300 bg-white px-1 py-0.5 text-right font-mono text-[10px] disabled:opacity-40 dark:border-zinc-700 dark:bg-zinc-900" /><span className="text-[10px]">px</span></label><button type="button" onClick={() => undoTransformKey(key, 0)} disabled={!canUndoT || locked} className="rounded-sm p-0.5 text-zinc-400 hover:text-zinc-700 disabled:opacity-20 dark:hover:text-zinc-200" title="undo"><LuUndo2 size={10} /></button><button type="button" onClick={() => redoTransformKey(key)} disabled={!canRedoT || locked} className="rounded-sm p-0.5 text-zinc-400 hover:text-zinc-700 disabled:opacity-20 dark:hover:text-zinc-200" title="redo"><LuRedo2 size={10} /></button><button type="button" onClick={() => toggleTransformLock(key)} className={`rounded-sm p-0.5 transition ${locked ? "text-cyan-600 dark:text-cyan-400" : "text-zinc-400 hover:text-zinc-700 dark:hover:text-zinc-200"}`} title={locked ? "Unlock" : "Lock"}>{locked ? <HiLockClosed size={10} /> : <HiLockOpen size={10} />}</button></span>; })}
                      <label className="subtext flex items-center gap-0.5 text-[10px] text-zinc-400" title="When off, keep one glyph run even when it crosses the cube edge. When on, wrap only after the measured text exceeds the cube width."><input type="checkbox" checked={transform?.wrap_text ?? false} onChange={(e) => updateSelectedStyle({ transform: { ...transform, wrap_text: e.target.checked } })} /> wrap</label>
                      {(() => { const key = "rotation" as const; const current = Number(transform?.rotation ?? 0); const hist = transformHistoryRef.current[key]; const locked = isTransformLocked(key); return <span className="subtext flex items-center gap-1 text-xs text-zinc-500" title="Drag, click, or type to rotate text. Double-click the dial or value to reset."><span>rotate</span><span className="flex items-center gap-0.5"><RotationDial value={current} disabled={locked} onChange={(value) => { trackTransformChange(key, current, value); updateCanvasTransform(key, value); }} onReset={() => { trackTransformChange(key, current, 0); updateCanvasTransform(key, 0); }} /><button type="button" onClick={() => toggleTransformLock(key)} className={`self-center rounded-sm p-0.5 transition ${locked ? "text-cyan-600 dark:text-cyan-400" : "text-zinc-400 hover:text-zinc-700 dark:hover:text-zinc-200"}`} title={locked ? "Unlock rotation" : "Lock rotation"}>{locked ? <HiLockClosed size={10} /> : <HiLockOpen size={10} />}</button></span><button type="button" onClick={() => undoTransformKey(key, 0)} disabled={!hist.undoStack.length || locked} className="rounded-sm p-0.5 text-zinc-400 hover:text-zinc-700 disabled:opacity-20 dark:hover:text-zinc-200" title="undo"><LuUndo2 size={10} /></button><button type="button" onClick={() => redoTransformKey(key)} disabled={!hist.redoStack.length || locked} className="rounded-sm p-0.5 text-zinc-400 hover:text-zinc-700 disabled:opacity-20 dark:hover:text-zinc-200" title="redo"><LuRedo2 size={10} /></button></span>; })()}
                      <span className={`subtext flex items-center gap-1 text-xs text-zinc-500 ${perspectiveActive ? "opacity-40" : ""}`} title={perspectiveActive ? "Perspective corners replace the affine anchor" : "Choose the point that stays fixed while skewing or stretching."}>
                        <span>anchor</span>
                        <span className="grid grid-cols-3 gap-px rounded-sm border border-zinc-300 p-0.5 dark:border-zinc-700">
                          {(["top_left", "top_center", "top_right", "middle_left", "center", "middle_right", "bottom_left", "bottom_center", "bottom_right"] as const).map((anchor) => (
                            <button key={anchor} type="button" disabled={perspectiveActive} aria-label={`transform anchor ${anchor.replace("_", " ")}`} onClick={() => updateSelectedStyle({ transform: { ...transform, skew_anchor: anchor } })} className={`h-2.5 w-2.5 rounded-xs disabled:cursor-not-allowed ${((transform?.skew_anchor ?? "center") === anchor) ? "bg-cyan-600" : "bg-zinc-300 hover:bg-zinc-400 dark:bg-zinc-600 dark:hover:bg-zinc-500"}`} />
                          ))}
                        </span>
                      </span>
                        </div>
                      </div>
                    </div>
                  </div>, textWarpHost) : null;
                  })()}
              </Section>}

              {/* Render result */}
              {renderResult && (
                <Section title="Render Outcome" icon={<Play size={14} />} className={stackClass(4)}>
              {(() => {
                const hasOutput = Boolean(renderResult.output_url);
                const hasWarning = hasOutput && !renderResult.qa_passed;
                const score = renderResult.qa_report?.overall_score;
                const icon = !hasOutput
                  ? <BiSolidErrorCircle size={22} className="shrink-0 text-red-600 dark:text-red-300" />
                  : hasWarning
                    ? <TiWarning size={24} className="shrink-0 text-amber-600 dark:text-amber-300" />
                    : <BiAbacus size={22} className="shrink-0 text-emerald-600 dark:text-emerald-300" />;
                const bannerClass = !hasOutput
                  ? "border-red-300 bg-red-50 dark:border-red-900/70 dark:bg-red-950/30"
                  : hasWarning
                    ? "border-amber-300 bg-amber-50 dark:border-amber-900/70 dark:bg-amber-950/30"
                    : "border-emerald-300 bg-emerald-50 dark:border-emerald-900/70 dark:bg-emerald-950/30";
                const status = !hasOutput ? "Render failed" : hasWarning ? "Render complete — QA below threshold" : "Render complete";
                return <div className={`mb-3 flex flex-wrap items-center gap-3 rounded-lg border px-4 py-3 ${bannerClass}`}>
                  {icon}
                  <div className="min-w-0">
                    <p className="text-sm font-semibold text-zinc-800 dark:text-zinc-100">{status}</p>
                    {renderResult.text_manifest && <p className="subtext text-xs text-zinc-600 dark:text-zinc-400">{renderResult.text_manifest.total_regions} region(s) assessed</p>}
                  </div>
                  <div className="flex-1" />
                  {score != null && <Badge ok={renderResult.qa_passed}>QA {(score * 100).toFixed(0)}% <span className="font-normal">(gate {(renderResult.qa_threshold * 100).toFixed(0)}%)</span></Badge>}
                </div>;
              })()}
              {renderResult.errors.length > 0 && (
                <div className="mb-3 rounded-lg border border-red-300 bg-red-100 px-4 py-2 dark:border-red-800 dark:bg-red-950/50">
                  {renderResult.errors.map((e, i) => (
                    <p key={i} className="text-sm text-red-700 dark:text-red-300">{e}</p>
                  ))}
                </div>
              )}
              {renderResult.output_url ? (
                <a href={renderResult.output_url} target="_blank" rel="noreferrer" className="inline-flex rounded-sm bg-cyan-700 px-3 py-1.5 text-xs text-white">Open finalized localized image</a>
              ) : (
                <p className="text-sm text-amber-600 dark:text-amber-400">Render did not produce an output image. Check logs below.</p>
              )}
              {(() => {
                const per = renderResult.qa_report?.per_asset_instance_score;
                const entries = per ? Object.entries(Object.values(per)[0] ?? {}) : [];
                if (entries.length === 0) return null;
                const fallbackIds = new Set(
                  (renderResult.text_manifest?.instances ?? [])
                    .filter((i) => i.glyph_fallback)
                    .map((i) => i.id)
                );
                return (
                  <div className="mt-3">
                    <p className="subtext mb-1 text-xs text-zinc-500">per-region QA</p>
                    <div className="flex flex-wrap gap-1">
                      {entries.map(([rid, score]) => (
                        <span
                          key={rid}
                          title={fallbackIds.has(rid) ? "font swapped: the requested font lacked characters for this text" : undefined}
                          className={`flex items-center gap-1 rounded px-2 py-0.5 font-mono text-xs ${
                            score >= 0.8 ? "bg-emerald-500/15 text-emerald-600 dark:text-emerald-400"
                            : score >= 0.6 ? "bg-amber-500/15 text-amber-600 dark:text-amber-400"
                            : "bg-red-500/15 text-red-600 dark:text-red-400"
                          }`}
                        >
                          {fallbackIds.has(rid) && <AlertTriangle size={10} className="text-amber-500 dark:text-amber-400" />}
                          {rid} {(score * 100).toFixed(0)}%
                        </span>
                      ))}
                    </div>
                  </div>
                );
              })()}
              {renderResult.logs.length > 0 && (
                <div className="mt-3">
                  <div className="flex justify-end">
                    <button type="button" onClick={() => setRenderLogsOpen((open) => !open)} className="flex items-center gap-1 rounded-sm p-1 text-zinc-500 transition hover:bg-zinc-200 dark:hover:bg-zinc-800" title={renderLogsOpen ? "hide logs" : "show logs"} aria-expanded={renderLogsOpen}>
                      <span className="subtext text-[10px]">show logs</span>
                      <FcCollapse style={{ transform: renderLogsOpen ? "none" : "rotate(180deg)", transition: "transform 0.2s" }} />
                    </button>
                  </div>
                  <div className={`style-panel-morph${renderLogsOpen ? " expanded" : ""}`}>
                    <div className="mt-1 max-h-56 overflow-auto rounded-lg bg-zinc-100 p-3 font-mono text-xs leading-5 dark:bg-zinc-950">
                      {renderResult.logs.map((l, i) => (
                        <div
                          key={i}
                          className={
                            l.level === "error" ? "text-red-600 dark:text-red-400"
                            : l.level === "warning" ? "text-amber-600 dark:text-amber-400"
                            : "text-zinc-500"
                          }
                        >
                          [{l.ts}] {l.stage}: {l.message}
                          {l.duration_ms != null && (
                            <span className="text-zinc-400 dark:text-zinc-600"> · {l.duration_ms}ms</span>
                          )}
                        </div>
                      ))}
                    </div>
                  </div>
                </div>
              )}
            </Section>
          )}
            </div>
          </div>
        </div>
      )}

      {/* STEP 4: Verify — QA inspector: coverage, per-region scores, recommendations, approve */}
      {step === 4 && (
        <div key="step-4" className="step-fade space-y-4">
          <div className="flex items-center justify-between gap-3">
            <div className="flex items-center gap-3">
              <FlipButton
                label="render"
                tooltip="return to render"
                icon={<ArrowLeft size={20} />}
                onClick={() => setStep(3)}
                reversed
              />
              <span className="subtext text-[8.4px] text-zinc-500">
                target: <span className="text-zinc-700 dark:text-zinc-300">{langDisplayName(targLang)}</span>
              </span>
            </div>
            <div className="flex items-center gap-2">
              {verifyBusy && (
                <button
                  onClick={cancelVerify}
                  className="rounded-lg bg-zinc-200 px-3 py-2 text-sm text-zinc-600 transition hover:bg-zinc-300 dark:bg-zinc-800 dark:text-zinc-400 dark:h(s)over:bg-zinc-700"
                >
                  cancel
                </button>
              )}
              <PressButton onClick={runVerifyRender} disabled={verifyBusy !== null} title="Run full QA render">
                {verifyBusy ? "Verifying…" : renderResult ? "Re-verify" : "Run Verification"}
              </PressButton>
            </div>
          </div>

          {verifyBusy && (
            <div className="bezier-card soft-shadow flex items-center gap-2 rounded-lg bg-white/60 px-4 py-2 text-sm text-cyan-600 dark:bg-zinc-900/60 dark:text-cyan-400">
              <SquareLoader size="xs" />
              <span>{verifyStage ? (STAGE_LABEL[verifyStage] ?? verifyStage) : "starting…"}</span>
            </div>
          )}

          {!renderResult && !verifyBusy && (
            <p className="subtext flex items-center gap-1.5 text-sm text-zinc-500">
              <PiWarningCircleFill size={15} className="text-[#2d8cf0]" />
              run verification to score this render against the source image.
            </p>
          )}

          {renderResult && (() => {
            const qa = renderResult.qa_report;
            const verification = renderResult.verification_report;
            const cov = qa?.progress;
            const per = qa?.per_asset_instance_score ? Object.values(qa.per_asset_instance_score)[0] ?? {} : {};
            const metrics = qa?.metrics ?? {};
            const asDict = (k: string): Record<string, number> => (metrics[k] as Record<string, number>) ?? {};
            const fallbackIds = new Set(
              (renderResult.text_manifest?.instances ?? []).filter((i) => i.glyph_fallback).map((i) => i.id)
            );
            const coverageComplete = cov ? cov.untranslated === 0 : false;
            return (
              <>
                {verification && (
                  <Section title="Verification summary" icon={<BiAbacus size={15} />}>
                    <div className="space-y-3">
                      <div className={`rounded-lg border px-3 py-2 ${
                        verification.project.overall_status === "pass"
                          ? "border-emerald-300 bg-emerald-50 dark:border-emerald-900/70 dark:bg-emerald-950/30"
                          : verification.project.overall_status === "review"
                            ? "border-amber-300 bg-amber-50 dark:border-amber-900/70 dark:bg-amber-950/30"
                            : "border-red-300 bg-red-50 dark:border-red-900/70 dark:bg-red-950/30"
                      }`}>
                        <div className="flex flex-wrap items-center gap-2">
                          <span className="text-sm font-semibold capitalize text-zinc-800 dark:text-zinc-100">
                            {verification.project.overall_status === "pass" ? "Ready" : verification.project.overall_status === "review" ? "Needs review" : "Blocked"}
                          </span>
                          {verification.project.overall_score != null && (
                            <span className="rounded-full bg-white/70 px-2 py-0.5 font-mono text-xs dark:bg-zinc-900/60">
                              {verification.project.overall_score.toFixed(0)}/100
                            </span>
                          )}
                        </div>
                        {verification.project.summary && <p className="mt-1 text-xs text-zinc-600 dark:text-zinc-300">{verification.project.summary}</p>}
                      </div>
                      <div className="flex flex-wrap gap-1.5">
                        {Object.entries(verification.project.component_scores).map(([name, score]) => (
                          <span
                            key={name}
                            title={score == null ? "No deterministic evidence was available for this component." : `${name.replace(/_/g, " ")} score`}
                            className="rounded bg-zinc-100 px-2 py-1 text-[10px] text-zinc-600 dark:bg-zinc-900 dark:text-zinc-300"
                          >
                            {name.replace(/_/g, " ")}: {score == null ? "not scored" : `${score.toFixed(0)}/100`}
                          </span>
                        ))}
                      </div>
                      <div className="flex flex-wrap gap-1.5 text-[10px]">
                        {Object.entries(verification.project.region_totals).map(([status, count]) => (
                          <span key={status} className="rounded-full border border-zinc-200 px-2 py-0.5 capitalize text-zinc-500 dark:border-zinc-800 dark:text-zinc-400">
                            {count} {status.replace(/_/g, " ")}
                          </span>
                        ))}
                        {verification.project.summary_flags.slice(0, 3).map((flag) => (
                          <span key={flag} className="rounded-full border border-amber-300 bg-amber-50 px-2 py-0.5 text-amber-700 dark:border-amber-900 dark:bg-amber-950/30 dark:text-amber-300">
                            {flag.replace(/_/g, " ")}
                          </span>
                        ))}
                      </div>
                      {verification.project.review_order.length > 0 && (
                        <div className="space-y-1">
                          {verification.project.review_order.map((regionId) => {
                            const region = verification.regions.find((candidate) => candidate.region_id === regionId);
                            if (!region) return null;
                            return <div key={regionId} className="flex flex-wrap items-center gap-2 rounded border border-zinc-200 px-2 py-1.5 text-xs dark:border-zinc-800">
                              <span className={`font-mono font-semibold ${region.status === "fail" ? "text-red-600 dark:text-red-400" : "text-amber-600 dark:text-amber-400"}`}>{regionId}</span>
                              <span className="capitalize text-zinc-500">{region.status}</span>
                              {region.overall_score != null && <span className="font-mono text-zinc-500">{region.overall_score.toFixed(0)}/100</span>}
                              <span className="min-w-0 flex-1 text-zinc-600 dark:text-zinc-300">{region.recommended_action ?? region.flags[0]?.replace(/_/g, " ")}</span>
                            </div>;
                          })}
                        </div>
                      )}
                    </div>
                  </Section>
                )}
                {/* recommendations checklist */}
                {qa?.recommendations && qa.recommendations.length > 0 && (
                  <Section
                    title="Recommendations"
                    icon={<span className="relative inline-flex"><MdTipsAndUpdates size={14} />{!recommendationsSeen && <span className="absolute -right-0.5 -top-0.5 h-1.5 w-1.5 rounded-full bg-violet-500 ring-1 ring-white dark:ring-zinc-900" />}</span>}
                    headerExtra={!recommendationsSeen && <span className="normal-case text-[10px] font-medium tracking-normal text-violet-600 dark:text-violet-300">new</span>}
                    rightSideHandle={<button type="button" onClick={() => setRecommendationsOpen((open) => { if (!open) setRecommendationsSeen(true); return !open; })} className="absolute right-5 top-4 rounded-sm p-1 text-zinc-500 transition hover:bg-zinc-200 dark:hover:bg-zinc-800" title={recommendationsOpen ? "collapse recommendations" : "expand recommendations"} aria-label={recommendationsOpen ? "collapse recommendations" : "expand recommendations"}><FcCollapse size={12} style={{ transform: recommendationsOpen ? "none" : "rotate(180deg)", transition: "transform 0.2s" }} /></button>}
                  >
                    <div className={`style-panel-morph${recommendationsOpen ? " expanded" : ""}`}>
                      <ul className="space-y-1.5 text-sm">
                        {qa.recommendations.map((r, i) => (
                          <li key={i} className="flex items-start gap-1.5 text-zinc-700 dark:text-zinc-300">
                            <MdTipsAndUpdates size={14} className="mt-0.5 shrink-0 text-[#2d8cf0]" />
                            <span>{r}</span>
                          </li>
                        ))}
                      </ul>
                    </div>
                  </Section>
                )}

                <div className="flex flex-col gap-4">
                {/* source <-> localized compare, with the former Coverage indicators */}
                {renderResult.output_url && (
                  <Section
                    title="Compare"
                    icon={<MdOutlineCompare size={15} />}
                    className={verifyCardOrder === "compare-first" ? "order-1" : "order-2"}
                    headerExtra={<span className="ml-1 flex min-w-0 items-center gap-1 normal-case text-[10px] font-normal tracking-normal">{qa?.overall_score != null && <Badge ok={renderResult.qa_passed}>QA {(qa.overall_score * 100).toFixed(0)}%</Badge>}{cov && <><span className="subtext whitespace-nowrap rounded-full bg-zinc-200 px-2 py-0.5 text-xs text-zinc-700 dark:bg-zinc-800 dark:text-zinc-300">{cov.rendered}/{cov.regions_total} rendered</span>{cov.dnt > 0 && <span className="subtext whitespace-nowrap rounded-full bg-zinc-200 px-2 py-0.5 text-xs text-zinc-700 dark:bg-zinc-800 dark:text-zinc-300">{cov.dnt} DNT</span>}{cov.untranslated > 0 && <span className="subtext flex whitespace-nowrap items-center gap-1 rounded-full bg-red-500/15 px-2 py-0.5 text-xs text-red-600 dark:text-red-400"><AlertTriangle size={11} /> {cov.untranslated} untranslated</span>}{cov.fallback_font > 0 && <span className="subtext flex whitespace-nowrap items-center gap-1 rounded-full bg-amber-500/15 px-2 py-0.5 text-xs text-amber-600 dark:text-amber-400"><AlertTriangle size={11} /> {cov.fallback_font} font fallback</span>}</>}</span>}
                    rightSideHandle={Object.keys(per).length > 0 ? <button type="button" onClick={() => setVerifyCardOrder((order) => order === "compare-first" ? "qa-first" : "compare-first")} title={verifyCardOrder === "compare-first" ? "Move Compare below Per-Region QA" : "Move Compare above Per-Region QA"} aria-label={verifyCardOrder === "compare-first" ? "move Compare down" : "move Compare up"} className="absolute right-5 top-4 rounded-sm p-1 text-zinc-500 hover:bg-zinc-200 dark:hover:bg-zinc-800">{theme === "dark" ? (verifyCardOrder === "compare-first" ? <BsArrowDownSquareFill size={16} /> : <BsArrowUpSquareFill size={16} />) : (verifyCardOrder === "compare-first" ? <LuSquareArrowDown size={16} /> : <LuSquareArrowUp size={16} />)}</button> : undefined}
                  >
                    <div className="grid gap-4 md:grid-cols-2">
                      <div>
                        <p className="subtext mb-1 text-xs text-zinc-500">source</p>
                        {previewUrl && <img src={previewUrl} alt="source" className="rounded-lg border border-zinc-300 dark:border-zinc-800" />}
                      </div>
                      <div>
                        <p className="subtext mb-1 text-xs text-zinc-500">localized ({langDisplayName(targLang)})</p>
                        <div className="relative">
                          <img src={renderResult.output_url} alt="localized" className="w-full rounded-lg border border-zinc-300 dark:border-zinc-800" />
                          {verification && renderResult.text_manifest?.img_dim && verification.visual_flags.map((flag) => {
                            const [width, height] = renderResult.text_manifest!.img_dim!;
                            return <div
                              key={flag.region_id}
                              className="pointer-events-none absolute border-2"
                              title={`${flag.region_id}: ${flag.label ?? flag.severity}`}
                              style={{
                                left: `${100 * flag.bounds.x / width}%`,
                                top: `${100 * flag.bounds.y / height}%`,
                                width: `${100 * flag.bounds.width / width}%`,
                                height: `${100 * flag.bounds.height / height}%`,
                                borderColor: flag.color,
                                boxShadow: `0 0 0 1px ${flag.color}55`,
                              }}
                            >
                              <span
                                className="absolute -top-5 left-0 whitespace-nowrap rounded px-1 py-0.5 font-mono text-[9px] font-semibold text-white"
                                style={{ backgroundColor: flag.color }}
                              >
                                {flag.region_id} · {flag.severity}
                              </span>
                            </div>;
                          })}
                        </div>
                      </div>
                    </div>
                  </Section>
                )}

                {/* per-region scores + re-render */}
                {Object.keys(per).length > 0 && (
                  <Section title="Per-Region QA" icon={<TbScanCube size={14} />} className={`${stackClass(3)} ${verifyCardOrder === "compare-first" ? "order-2" : "order-1"}`} rightSideHandle={renderResult.output_url ? <button type="button" onClick={() => setVerifyCardOrder((order) => order === "compare-first" ? "qa-first" : "compare-first")} title={verifyCardOrder === "compare-first" ? "Move Per-Region QA above Compare" : "Move Per-Region QA below Compare"} aria-label={verifyCardOrder === "compare-first" ? "move Per-Region QA up" : "move Per-Region QA down"} className="absolute right-5 top-4 rounded-sm p-1 text-zinc-500 hover:bg-zinc-200 dark:hover:bg-zinc-800">{theme === "dark" ? (verifyCardOrder === "compare-first" ? <BsArrowUpSquareFill size={16} /> : <BsArrowDownSquareFill size={16} />) : (verifyCardOrder === "compare-first" ? <LuSquareArrowUp size={16} /> : <LuSquareArrowDown size={16} />)}</button> : undefined}>
                    <div className="space-y-1.5">
                      {Object.entries(per).map(([rid, score]) => {
                        const isSel = verifySelId === rid;
                        return (
                          <div key={rid} className="rounded-lg border border-zinc-200 dark:border-zinc-800">
                            <div
                              role="button"
                              tabIndex={0}
                              aria-expanded={isSel}
                              onClick={() => setVerifySelId(isSel ? null : rid)}
                              onKeyDown={(e) => { if (e.key === "Enter" || e.key === " ") setVerifySelId(isSel ? null : rid); }}
                              className="flex w-full cursor-pointer items-center gap-2 px-3 py-2 text-left"
                            >
                              <span
                                className={`flex items-center gap-1 rounded px-2 py-0.5 font-mono text-xs ${
                                  score >= 0.8 ? "bg-emerald-500/15 text-emerald-600 dark:text-emerald-400"
                                  : score >= 0.6 ? "bg-amber-500/15 text-amber-600 dark:text-amber-400"
                                  : "bg-red-500/15 text-red-600 dark:text-red-400"
                                }`}
                              >
                                {fallbackIds.has(rid) && <AlertTriangle size={10} className="text-amber-500 dark:text-amber-400" />}
                                {rid} {(score * 100).toFixed(0)}%
                              </span>
                              <span className="flex-1" />
                              <button
                                onClick={(e) => { e.stopPropagation(); onReRenderRegion(rid); }}
                                disabled={reRenderingId !== null || verifyBusy !== null}
                                title="re-render just this region"
                                className="flex items-center gap-1 rounded-sm px-2 py-1 text-xs text-zinc-500 transition hover:bg-zinc-200 disabled:opacity-40 dark:hover:bg-zinc-800"
                              >
                                {reRenderingId === rid ? <Loader2 size={12} className="animate-spin" /> : <RotateCcw size={12} />}
                                re-render
                              </button>
                            </div>
                            <div className={`smart-fill-content${isSel ? " expanded" : ""}`}>
                              <div className="min-h-0">
                                <div className="subtext space-y-0.5 border-t border-zinc-200 px-3 py-2 text-xs text-zinc-600 dark:border-zinc-800 dark:text-zinc-400">
                                  {asDict("ocr_roundtrip")[rid] != null && <p>OCR round-trip: {(asDict("ocr_roundtrip")[rid] * 100).toFixed(0)}%</p>}
                                  {asDict("ring_ssim")[rid] != null && <p>ring SSIM: {(asDict("ring_ssim")[rid] * 100).toFixed(0)}%</p>}
                                  {asDict("residual_text")[rid] != null && <p>residual source text: {(asDict("residual_text")[rid] * 100).toFixed(0)}%</p>}
                                  {asDict("style_color")[rid] != null && <p>style color match: {(asDict("style_color")[rid] * 100).toFixed(0)}%</p>}
                                  {asDict("style_size")[rid] != null && <p>style size match: {(asDict("style_size")[rid] * 100).toFixed(0)}%</p>}
                                  {fallbackIds.has(rid) && <p className="text-amber-600 dark:text-amber-400">font swapped: the requested font lacked characters for this text.</p>}
                                </div>
                              </div>
                            </div>
                          </div>
                        );
                      })}
                    </div>
                  </Section>
                )}
                </div>

                {/* double-confirmation approve gate */}
                <div className={`bezier-card soft-shadow flex flex-wrap items-center gap-3 rounded-lg bg-white/60 px-4 py-3 dark:bg-zinc-900/60 ${stackClass(4)}`}>
                  <span className="subtext text-sm text-zinc-600 dark:text-zinc-400">
                    {cov ? `${cov.regions_total}/${cov.regions_total} region(s) addressed — ${cov.rendered} rendered, ${cov.dnt} DNT` : "coverage unavailable"}
                    {!coverageComplete && cov && cov.untranslated > 0 && (
                      <span className="text-red-600 dark:text-red-400"> ({cov.untranslated} still untranslated)</span>
                    )}
                  </span>
                  <div className="flex-1" />
                  {approved ? (
                    <Badge ok>approved</Badge>
                  ) : (
                    <PressButton
                      onClick={onApprove}
                      disabled={!coverageComplete}
                      title={coverageComplete ? "Sign off on this render" : "All non-DNT regions must be translated and rendered first"}
                    >
                      Approve
                    </PressButton>
                  )}
                </div>
              </>
            );
          })()}
        </div>
      )}

      {/* Ongoing-session confirmation: uploading a new image erases the
          Capture data of the current one (after a pre-erase snapshot) */}
      {pendingFile && (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/60">
          <div className="bezier-card w-full max-w-md rounded-xl bg-white p-6 dark:bg-zinc-900">
            <h3 className="mb-2 flex items-center gap-2 text-lg font-semibold text-zinc-800 dark:text-zinc-200">
              <AlertTriangle size={18} className="text-amber-500 dark:text-amber-400" /> Ongoing Session
            </h3>
            <p className="mb-4 text-sm text-zinc-600 dark:text-zinc-400">
              You have an active capture session with{" "}
              <span className="text-zinc-800 dark:text-zinc-200">{manifest.length} region(s)</span>
              {translatedCount > 0 && <> ({translatedCount} translated)</>}.
              Uploading <span className="text-zinc-800 dark:text-zinc-200">{pendingFile.name}</span> will
              clear the bounding-box and translation data in this workspace.
            </p>
            <p className="subtext mb-4 rounded-lg border border-emerald-300 bg-emerald-50 px-3 py-2 text-xs text-emerald-700 dark:border-emerald-900/60 dark:bg-emerald-950/30 dark:text-emerald-300">
              A snapshot of the current session is saved first — you can restore
              it any time from History.
            </p>
            <div className="flex justify-end gap-2">
              <button
                onClick={() => setPendingFile(null)}
                className="rounded-lg px-4 py-2 text-sm text-zinc-500 hover:text-zinc-800 dark:text-zinc-400 dark:hover:text-zinc-200"
              >
                Cancel
              </button>
              <button
                onClick={() => {
                  const f = pendingFile;
                  setPendingFile(null);
                  if (f) void doUpload(f);
                }}
                className="rounded-lg bg-amber-600 px-4 py-2 text-sm font-medium text-white transition hover:bg-amber-500"
              >
                Snapshot & Continue
              </button>
            </div>
          </div>
        </div>
      )}

      {/* Duplicate-image cross-check: this exact file already exists in
          another project -- confirm before uploading a second copy */}
      {pendingDuplicateFile && (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/60">
          <div className="bezier-card w-full max-w-md rounded-xl bg-white p-6 dark:bg-zinc-900">
            <h3 className="mb-2 flex items-center gap-2 text-lg font-semibold text-zinc-800 dark:text-zinc-200">
              <AlertTriangle size={18} className="text-amber-500 dark:text-amber-400" /> Duplicate Image Detected
            </h3>
            <p className="mb-4 text-sm text-zinc-600 dark:text-zinc-400">
              Image exists in project{" "}
              <span className="text-zinc-800 dark:text-zinc-200">{pendingDuplicateFile.projectName}</span>. Proceed?
            </p>
            <div className="flex justify-end gap-2">
              <button
                onClick={() => setPendingDuplicateFile(null)}
                className="rounded-lg px-4 py-2 text-sm text-zinc-500 hover:text-zinc-800 dark:text-zinc-400 dark:hover:text-zinc-200"
              >
                Cancel
              </button>
              <button
                onClick={() => {
                  const f = pendingDuplicateFile.file;
                  setPendingDuplicateFile(null);
                  proceedWithFile(f);
                }}
                className="rounded-lg bg-amber-600 px-4 py-2 text-sm font-medium text-white transition hover:bg-amber-500"
              >
                Proceed
              </button>
            </div>
          </div>
        </div>
      )}

      {/* Language-mismatch resolver: block + protect override */}
      {scan?.status === "mismatch" && asset && (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/60">
          <div className="bezier-card w-full max-w-md rounded-xl bg-white p-6 dark:bg-zinc-900">
            <h3 className="mb-2 flex items-center gap-2 text-lg font-semibold text-zinc-800 dark:text-zinc-200">
              <AlertTriangle size={18} className="text-red-500 dark:text-red-400" /> Language Mismatch
            </h3>
            <p className="mb-2 text-sm text-zinc-600 dark:text-zinc-400">
              <span className="text-zinc-800 dark:text-zinc-200">{asset.filename}</span> appears to be in{" "}
              <span className="font-medium text-red-600 dark:text-red-400">
                {langFlag(scan.detected ?? "")} {langDisplayName(scan.detected ?? "?")}
              </span>
              , but this project's source language is{" "}
              <span className="font-medium text-cyan-700 dark:text-cyan-400">
                {langFlag(scan.projectSrc ?? "")} {langDisplayName(scan.projectSrc ?? "?")}
              </span>.
            </p>
            <p className="subtext mb-4 text-xs text-zinc-500">
              Assets must match the project's source language. Change the
              project's source language to proceed, or override if the
              detection is wrong.
            </p>
            <div className="space-y-2">
              <button
                onClick={onMismatchSwitch}
                className="w-full rounded-lg bg-cyan-600 px-4 py-2 text-left text-sm font-medium text-white transition hover:bg-cyan-500"
              >
                Switch project source to {langDisplayName(scan.detected ?? "?")} & keep asset
              </button>
              <button
                onClick={onMismatchKeep}
                className="w-full rounded-lg bg-zinc-200 px-4 py-2 text-left text-sm text-zinc-700 transition hover:bg-zinc-300 dark:bg-zinc-800 dark:text-zinc-300 dark:hover:bg-zinc-700"
              >
                Detection is wrong — keep asset as {langDisplayName(scan.projectSrc ?? "?")}
              </button>
              <button
                onClick={onMismatchRemove}
                className="w-full rounded-lg bg-red-600/90 px-4 py-2 text-left text-sm font-medium text-white transition hover:bg-red-500"
              >
                Remove asset
              </button>
            </div>
          </div>
        </div>
      )}

      {/* Capture Kickoff Modal: manual vs automatic */}
      {showCapturePrompt && asset && (
        <div className="title-confirm-backdrop" onClick={() => setShowCapturePrompt(false)}>
          <div className="bezier-card title-confirm-card" style={{ maxWidth: "420px" }} onClick={(e) => e.stopPropagation()}>
            <div className="flex items-center gap-2 mb-2">
              <VectorSquare size={18} className="text-cyan-500 dark:text-cyan-400" />
              <p className="text-sm font-semibold text-zinc-800 dark:text-zinc-200">begin capture</p>
            </div>
            <p className="subtext text-sm text-zinc-600 dark:text-zinc-400 mb-4">
              how would you like to identify the text regions in{" "}
              <span className="font-medium text-zinc-800 dark:text-zinc-200">{asset.filename}</span>?
            </p>
            <div className="space-y-3 w-full">
              <button
                onClick={onAutoDetect}
                disabled={busy !== null}
                className="flex w-full items-center gap-3 rounded-lg border border-zinc-300 bg-zinc-100 p-4 text-left transition hover:border-cyan-600 dark:border-zinc-700 dark:bg-zinc-800 dark:hover:bg-zinc-800/80"
              >
                <HiCubeTransparent size={20} className="text-cyan-500 dark:text-cyan-400" />
                <div>
                  <p className="text-sm font-medium text-zinc-800 dark:text-zinc-200">automatically</p>
                  <p className="subtext text-xs text-zinc-500">the model scans the image, draws bounding boxes around detected text, and identifies the source language.</p>
                </div>
                {busy === "detecting" && <SquareLoader size="sm" className="ml-auto text-cyan-400" />}
              </button>
              <button
                onClick={onManualDraw}
                className="flex w-full items-center gap-3 rounded-lg border border-zinc-300 bg-zinc-100 p-4 text-left transition hover:border-cyan-600 dark:border-zinc-700 dark:bg-zinc-800 dark:hover:bg-zinc-800/80"
              >
                <Plus size={20} className="text-cyan-500 dark:text-cyan-400" />
                <div>
                  <p className="text-sm font-medium text-zinc-800 dark:text-zinc-200">manually</p>
                  <p className="subtext text-xs text-zinc-500">you draw the bounding boxes yourself. useful for obscured or hard-to-detect text.</p>
                </div>
              </button>
            </div>
            <button
              onClick={() => setShowCapturePrompt(false)}
              className="subtext mt-2 w-full text-center text-xs text-zinc-500 hover:text-zinc-700 dark:hover:text-zinc-300"
            >
              cancel
            </button>
          </div>
        </div>
      )}

      {/* Detected Language Confirmation Card */}
      {showLangConfirm && srcLang && (
        <div className="title-confirm-backdrop" onClick={() => setShowLangConfirm(false)}>
          <div className="bezier-card title-confirm-card" onClick={(e) => e.stopPropagation()}>
            <div className="flex items-center gap-2 mb-3">
              <ScanText size={18} className="text-cyan-500 dark:text-cyan-400" />
              <p className="text-sm font-semibold text-zinc-800 dark:text-zinc-200">detected source language</p>
            </div>
            <p className="subtext text-sm text-zinc-600 dark:text-zinc-400 mb-4">
              the model identified the primary language as{" "}
              <span className="font-medium text-cyan-700 dark:text-cyan-400">{langFlag(srcLang)} {langDisplayName(srcLang)}</span>{" "}
              ({srcLang}). please confirm.
            </p>
            <div className="flex items-center gap-3 justify-center">
              <button
                onClick={() => {
                  setShowLangConfirm(false);
                  if (project) {
                    updateProject(project.id, { source_lang: srcLang }).then(() => refreshProject()).catch(() => {});
                  }
                  addToast("success", `source language set: ${langDisplayName(srcLang)}`);
                  runDetect(srcLang ? [srcLang] : undefined);
                }}
                className="title-mono-btn"
                style={{ width: "auto", padding: "10px 24px" }}
              >
                <Check size={16} />
                <p className="title-mono-text">confirm</p>
              </button>
              <button
                onClick={() => {
                  setShowLangConfirm(false);
                  setShowLangSelect(true);
                }}
                className="title-mono-btn"
                style={{ width: "auto", padding: "10px 24px" }}
              >
                <Languages size={16} />
                <p className="title-mono-text">other</p>
              </button>
            </div>
          </div>
        </div>
      )}

      {/* Language Select Card (after "other") */}
      {showLangSelect && (
        <div className="title-confirm-backdrop" onClick={() => { setShowLangSelect(false); setLangRegionFilter(null); }}>
          <div className="bezier-card title-confirm-card" style={{ maxWidth: "480px" }} onClick={(e) => e.stopPropagation()}>
            <p className="text-sm font-semibold text-zinc-800 dark:text-zinc-200 mb-3">pick the source language below.</p>
            <div className="flex flex-wrap gap-1 mb-3">
              {REGION_ORDER.map((region) => (
                <button
                  key={region}
                  onClick={() => {
                    setLangRegionFilter((prev) => prev === region ? null : region);
                    const regionCodes = LANGUAGE_REGIONS[region] || [];
                    const firstCode = regionCodes.find((c) => languages.map((l) => l.code).includes(c));
                    if (firstCode) setSrcLang(firstCode);
                  }}
                  className={`rounded-lg px-3 py-1.5 text-xs font-medium transition ${
                    langRegionFilter === region
                      ? "bg-cyan-600 text-white hover:bg-cyan-700"
                      : "bg-zinc-200 text-zinc-600 hover:bg-zinc-300 dark:bg-zinc-800 dark:text-zinc-400 dark:hover:bg-zinc-700"
                  }`}
                >
                  {region}
                </button>
              ))}
            </div>
            <LanguageCombobox
              value={srcLang}
              onChange={(code) => setSrcLang(code)}
              availableCodes={languages.map((l) => l.code)}
              placeholder="select language…"
              regionFilter={langRegionFilter}
            />
            <div className="flex items-center gap-3 justify-end mt-4">
              <button
                onClick={() => {
                  setShowLangSelect(false);
                  runDetect(srcLang ? [srcLang] : undefined);
                }}
                disabled={busy !== null}
                className="title-mono-btn"
                style={{ width: "auto", padding: "10px 20px" }}
              >
                {busy === "detecting" ? <SquareLoader size="sm" /> : <ScanText size={14} />}
                <p className="title-mono-text">re-detect</p>
              </button>
              <button
                onClick={() => {
                  setShowLangSelect(false);
                  if (project && srcLang) {
                    updateProject(project.id, { source_lang: srcLang }).then(() => refreshProject()).catch(() => {});
                  }
                  if (srcLang) addToast("success", `source language set to ${langDisplayName(srcLang)}`);
                  runDetect(srcLang ? [srcLang] : undefined);
                }}
                className="title-mono-btn"
                style={{ width: "auto", padding: "10px 20px" }}
              >
                <Check size={14} />
                <p className="title-mono-text">confirm</p>
              </button>
            </div>
          </div>
        </div>
      )}

      {/* Projects page (gate) */}
      {displayedScreen === "pantry" && (
        <ProjectGate
          languages={languages}
          onSelectProject={openProject}
          onProjectRenamed={setProject}
          onClose={() => {
            setPantryLeaving(true);
            setTimeout(() => {
              setPantryLeaving(false);
              setDisplayedScreen(prevScreen.current);
              setScreen(prevScreen.current);
            }, 300);
          }}
          theme={theme}
          leaving={pantryLeaving}
          initialView={pantryMode === "create" ? "create-name" : "list"}
          listOnly={pantryMode === "pantry"}
          currentProjectId={project?.id ?? null}
        />
      )}

      {showHistory && project && (
        <HistoryPanel
          project={project}
          currentAssetId={asset?.asset_id ?? null}
          onRestored={onHistoryRestored}
          onClose={() => {
            setHistoryLeaving(true);
            setTimeout(() => {
              setHistoryLeaving(false);
              setShowHistory(false);
            }, 300);
          }}
          leaving={historyLeaving}
        />
      )}

      {showMemory && project && (
        <MemoryPanel
          project={project}
          onClose={() => {
            setMemoryLeaving(true);
            setTimeout(() => {
              setMemoryLeaving(false);
              setShowMemory(false);
            }, 300);
          }}
          leaving={memoryLeaving}
        />
      )}

      <Suspense fallback={null}><FontManager
        open={showFontManager}
        onClose={() => setShowFontManager(false)}
        families={fullFamiliesByLang[styleTargetInst?.target_language ?? targLang] ?? []}
        loading={fullFontsLoading}
        value={styleTargetInst?.style_profile?.font_family ?? null}
        targLang={styleTargetInst?.target_language ?? targLang}
        recent={recentFonts}
        onPick={onFontManagerPick}
        onLibraryChanged={onFontLibraryChanged}
      /></Suspense>

      {showCapabilities && <Suspense fallback={null}><SystemCapabilitiesPanel onClose={() => setShowCapabilities(false)} /></Suspense>}

      {showSettings && (
        <div className="title-confirm-backdrop" style={{ zIndex: 400 }} onClick={() => setShowSettings(false)}>
          <div className="bezier-card title-confirm-card step-fade" style={{ maxWidth: "560px", width: "min(560px, calc(100vw - 32px))" }} onClick={(e) => e.stopPropagation()}>
            <div className="flex items-center gap-2 mb-4 w-full">
              <Hexagon size={18} className="text-cyan-500 dark:text-cyan-400" />
              <p className="text-sm font-semibold text-zinc-800 dark:text-zinc-200">settings</p>
            </div>
            <div className="space-y-4">
              <div>
                <p className="subtext mb-2 text-xs uppercase tracking-wider text-zinc-500 dark:text-zinc-600">capture</p>
                <div className="space-y-2">
                  <div className="flex items-center justify-between gap-6">
                    <p className="subtext text-xs text-zinc-600 dark:text-zinc-400">bounding boxes</p>
                    <div className="w-40 shrink-0">
                      <BBoxColorDropdown value={bboxColor} onChange={setBboxColor} />
                    </div>
                  </div>
                  <div className="flex items-center justify-between gap-6">
                    <p className="subtext text-xs text-zinc-600 dark:text-zinc-400">bbox blink</p>
                    <div className="flex w-40 shrink-0 justify-end">
                      <PenumbraSwitch checked={bboxBlink} onChange={setBboxBlink} label="" />
                    </div>
                  </div>
                </div>
              </div>
              <div>
                <p className="subtext mb-2 text-xs uppercase tracking-wider text-zinc-500 dark:text-zinc-600">translate</p>
                <div className="flex items-center justify-between gap-6">
                  <p className="subtext text-xs text-zinc-600 dark:text-zinc-400">hide region counter</p>
                  <div className="flex w-40 shrink-0 justify-end">
                    <PenumbraSwitch checked={hideRegionCounter} onChange={setHideRegionCounter} label="" />
                  </div>
                </div>
              </div>
            </div>
            <button
              onClick={() => setShowSettings(false)}
              className="subtext mt-4 w-full text-center text-xs text-zinc-500 hover:text-zinc-700 dark:hover:text-zinc-300"
            >
              close
            </button>
          </div>
        </div>
      )}

      {/* The session is still resolving: same card shell as the confirm
          overlays, so it arrives with the identical animation, but with
          nothing to choose -- clicking the scrim must not dismiss it. */}
      {sessionLoading && (
        <div className="title-confirm-backdrop">
          <div className="title-confirm-card">
            <p className="title-confirm-message flex items-center gap-3">
              <SquareLoader size="sm" />
              loading…
            </p>
          </div>
        </div>
      )}

      {showTitleConfirm && (
        <TitleConfirmOverlay
          onYes={() => { setShowTitleConfirm(false); prevScreen.current = displayedScreen; setScreen("title"); }}
          onNo={() => setShowTitleConfirm(false)}
        />
      )}

      {pendingAssetDelete && (
        <AssetDeleteConfirmOverlay
          filename={pendingAssetDelete.filename ?? pendingAssetDelete.assetId}
          onYes={() => { confirmDeleteAsset(); setPendingAssetDelete(null); }}
          onNo={() => setPendingAssetDelete(null)}
        />
      )}

      {pendingBackNav && (
        <UnsavedChangesOverlay
          onYes={() => {
            setPendingBackNav(false);
            navGuardRef.current = true;
            const prev = prevScreen.current;
            setScreen(prev && prev !== displayedScreen ? prev : "title");
          }}
          onNo={() => {
            setPendingBackNav(false);
            window.history.pushState({ screen: displayedScreen }, "");
          }}
        />
      )}

      <ToastSystem toasts={toasts} onDismiss={dismissToastWithNotif} />
    </div>
    </div>
  );
}
