import { useCallback, useEffect, useRef, useState } from "react";
import {
  AlertTriangle, AlignCenter, AlignEndHorizontal, AlignEndVertical, AlignJustify, AlignLeft, AlignRight, AlignStartHorizontal, AlignStartVertical, ArrowLeft, ArrowLeftRight, ArrowUpFromLine, Baseline, Bold, BookmarkCheck, Box, Check, ChevronDown, FileImage, FolderOpen, Hexagon, History, Home, Italic, Languages, Loader2,
  Play, Plus, RotateCcw, ScanText, ShieldAlert, Sparkles, SquareStack, Subscript, Superscript, Type, Underline, X,
} from "lucide-react";
import {
  BBox, FontFamily, FontOption, ImportResult, InpaintPatch, InpaintingProvider, InstText, LanguageOption, Project,
  RenderResult, RenderStreamEvent, SceneRegion, SemanticSubstitutionPlan, SemanticTextUnit, TextManifest, UploadResponse, ValidationReport, GlossaryStatus,
  addRegion, approveRender, checkDuplicateAsset, deleteProjectAsset, deleteRegion, detectAssetStream,
  fetchFonts, fetchLanguages, getManifest, getProject, importFile, matchFonts, ocrRegion, putManifest,
  applyRepairCandidate, captureLocalizedBaseline, createInpaintPatch, getInpaintingProviders, getLocalizedBaseline, getTreatment, previewCandidateLocalized, refineRegion, renderAsset, renderAssetStream, renderPreview, restoreTreatment, scanAssetLanguage, sha256File, snapshotAsset, undoInpaint,
  semanticSubstitution, updateProject, uploadAsset, validateAsset, fetchGlossaryStatus, uploadGlossary, deleteGlossary,
} from "./api";
import { FcCollapse } from "react-icons/fc";
import { LiaSpellCheckSolid } from "react-icons/lia";
import { RiCheckboxFill } from "react-icons/ri";
import { TbCubePlus, TbPhotoScan } from "react-icons/tb";
import { FaBoxOpen, FaLink, FaUnlink } from "react-icons/fa";
import { FaFileImport } from "react-icons/fa6";
import { PiWarningCircleFill } from "react-icons/pi";
import { MdTipsAndUpdates } from "react-icons/md";
import { HiCubeTransparent } from "react-icons/hi2";
import { HiLockClosed, HiLockOpen } from "react-icons/hi";
import { LuUndo2, LuRedo2 } from "react-icons/lu";
import { VscDebugRestart } from "react-icons/vsc";
import { GiCoolSpices } from "react-icons/gi";
import { langDisplayName, langFlag, LANGUAGE_REGIONS, REGION_ORDER } from "./languageData";
import LanguageCombobox from "./LanguageCombobox";
import FontCombobox, { loadFontPreview, fontNameForPath, weightLabel } from "./FontCombobox";
import BBoxCanvas from "./BBoxCanvas";
import TargetPreviewCanvas from "./TargetPreviewCanvas";
import RegionTable from "./RegionTable";
import SemanticSubstitutionPanel from "./SemanticSubstitutionPanel";
import ExportPanel from "./ExportPanel";
import ProjectGate from "./ProjectGate";
import HistoryPanel from "./HistoryPanel";
import MemoryPanel from "./MemoryPanel";
import SplashScreen from "./SplashScreen";
import TitleScreen from "./TitleScreen";
import ThemeToggle from "./ThemeToggle";
import BBoxColorDropdown from "./BBoxColorDropdown";
import { FlipButton, PressButton, ThemeSwitch } from "./Buttons";
import { SquareLoader } from "./Loaders";
import ToastSystem, { useToasts, useNotifications, type ToastType, type ToastAction } from "./ToastSystem";
import NotificationBell from "./NotificationBell";
import Stepper, { Step } from "./Stepper";
import { logoSrc, useTheme } from "./theme";

const PROJECT_KEY = "tofu.projectId";

type Screen = "splash" | "title" | "pantry" | "main";

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

type RepairCandidate = {
  id: string;
  provider: string;
  decision: string;
  url: string;
  bbox: BBox;
  cacheKey: string;
};

type RepairReview = {
  id: string;
  provider: string;
  reason: string;
  candidates: RepairCandidate[];
};

type LocalizedCandidatePreview = {
  status: "loading" | "ready" | "error";
  url?: string;
  error?: string;
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
  edge_blur_px: 0, erosion_px: 0, dilation_px: 0, grain_strength: 0,
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

function visualReadingOrder(instances: InstText[]): InstText[] {
  // Older manifests were persisted with raw top-edge ordering.  Repair their
  // presentation without changing durable region IDs: words whose boxes share
  // a baseline form one line, then read left-to-right within that line.
  const horizontal = instances.filter((inst) => inst.bounding_box.width >= inst.bounding_box.height * .55);
  const vertical = instances.filter((inst) => !horizontal.includes(inst));
  if (!horizontal.length) return [...instances].sort((a, b) => (a.reading_order ?? 0) - (b.reading_order ?? 0));
  const heights = horizontal.map((inst) => inst.bounding_box.height).sort((a, b) => a - b);
  const tolerance = Math.max(4, heights[Math.floor(heights.length / 2)] * .42);
  const lines: Array<{ center: number; height: number; items: InstText[] }> = [];
  [...horizontal].sort((a, b) => (a.bounding_box.y + a.bounding_box.height / 2) - (b.bounding_box.y + b.bounding_box.height / 2)).forEach((inst) => {
    const center = inst.bounding_box.y + inst.bounding_box.height / 2;
    const line = lines.find((candidate) => Math.abs(candidate.center - center) <= Math.max(tolerance, candidate.height * .42));
    if (!line) { lines.push({ center, height: inst.bounding_box.height, items: [inst] }); return; }
    const count = line.items.push(inst);
    line.center += (center - line.center) / count;
    line.height += (inst.bounding_box.height - line.height) / count;
  });
  return [
    ...lines.sort((a, b) => a.center - b.center).flatMap((line) => line.items.sort((a, b) => a.bounding_box.x - b.bounding_box.x)),
    ...vertical.sort((a, b) => (a.reading_order ?? 0) - (b.reading_order ?? 0)),
  ];
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
        className="w-full rounded border border-zinc-300 bg-white px-2 py-1 text-xs text-zinc-800 dark:border-zinc-700 dark:bg-zinc-900 dark:text-zinc-200"
      />
    </div>
  );
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
  const [preRenderUrl, setPreRenderUrl] = useState<string | null>(null);
  const [previewRenderError, setPreviewRenderError] = useState<string | null>(null);
  const [previewCacheKey, setPreviewCacheKey] = useState<string | null>(null);
  const [previewTransaction, setPreviewTransaction] = useState<PreviewTransaction>({ phase: "idle", requestId: 0, cause: "initial" });
  const [previewRetryRevision, setPreviewRetryRevision] = useState(0);
  const [inpaintingProviders, setInpaintingProviders] = useState<InpaintingProvider[]>([]);
  const [repairReviews, setRepairReviews] = useState<RepairReview[]>([]);
  const [localizedCandidatePreviews, setLocalizedCandidatePreviews] = useState<Record<string, LocalizedCandidatePreview>>({});
  const [candidatePreviewRevision, setCandidatePreviewRevision] = useState(0);
  const [repairFallbackIds, setRepairFallbackIds] = useState<string[]>([]);
  const previewRenderSeq = useRef(0);
  const previewCauseRef = useRef<PreviewCause>("initial");
  const candidatePreviewSeq = useRef(0);
  const [garnishExpanded, setGarnishExpanded] = useState(true);
  const [garnishRegionMode, setGarnishRegionMode] = useState(false);
  const [selectedGarnishRegionId, setSelectedGarnishRegionId] = useState<string | null>(null);
  const previewSyncing = previewTransaction.phase === "rendering";
  const previewPending = previewTransaction.phase === "debouncing" || previewTransaction.phase === "rendering";
  const garnishPreviewSyncing = previewSyncing && previewTransaction.cause === "garnish";
  const [lassoMode, setLassoMode] = useState(false);
  const [lassoPoints, setLassoPoints] = useState<[number, number][]>([]);
  const [brushMode, setBrushMode] = useState(false);
  const [brushStrokes, setBrushStrokes] = useState<BrushStroke[]>([]);
  const [activeBrushStroke, setActiveBrushStroke] = useState<BrushStroke | null>(null);
  const [brushCursor, setBrushCursor] = useState<[number, number] | null>(null);
  const [brushRadius, setBrushRadius] = useState(18);
  const [brushHardness, setBrushHardness] = useState(0.85);
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
  const [colorPickMode, setColorPickMode] = useState<"source" | "localized" | null>(null);
  const [languages, setLanguages] = useState<LanguageOption[]>([]);
  const [targLang, setTargLang] = useState("es");
  const [formerTargLang, setFormerTargLang] = useState<string | null>(null);
  const [showLangChangePopup, setShowLangChangePopup] = useState(false);
  const [fontsByLang, setFontsByLang] = useState<Record<string, FontOption[]>>({});
  const [familiesByLang, setFamiliesByLang] = useState<Record<string, FontFamily[]>>({});
  const [script, setScript] = useState<string>("");
  const fontFetches = useRef<Set<string>>(new Set());
  const [report, setReport] = useState<ValidationReport | null>(null);
  const [renderResult, setRenderResult] = useState<RenderResult | null>(null);
  const [renderSelId, setRenderSelId] = useState<string | null>(null);
  const [prevSelId, setPrevSelId] = useState<string | null>(null);
  const [styleCollapsed, setStyleCollapsed] = useState(false);
  const [canvasExpandedH, setCanvasExpandedH] = useState(false);
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
    // compute max height: localized card bottom - style card top - decisions summary - gap
    const styleEl = styleCardRef.current;
    const localizedEl = document.querySelector('[data-render-localized]');
    const decisionsEl = document.querySelector('[data-render-decisions]');
    let maxH = 9999;
    if (styleEl && localizedEl) {
      const styleRect = styleEl.getBoundingClientRect();
      const localizedRect = localizedEl.getBoundingClientRect();
      const decisionsH = decisionsEl ? (decisionsEl as HTMLElement).offsetHeight : 0;
      maxH = localizedRect.bottom - styleRect.top - decisionsH - 16;
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
      // expand to max: decisions summary bottom should align with localized asset card bottom
      const styleEl = styleCardRef.current;
      if (!styleEl) return;
      const localizedEl = document.querySelector('[data-render-localized]');
      if (!localizedEl) return;
      const decisionsEl = document.querySelector('[data-render-decisions]');
      const styleRect = styleEl.getBoundingClientRect();
      const localizedRect = localizedEl.getBoundingClientRect();
      // account for decisions summary card height + gap (space-y-4 = 16px)
      const decisionsH = decisionsEl ? (decisionsEl as HTMLElement).offsetHeight : 0;
      const gap = 16; // space-y-4 gap
      const maxH = localizedRect.bottom - styleRect.top - decisionsH - gap;
      preExpandStyleH.current = styleEl.offsetHeight;
      setStyleCardH(Math.max(200, maxH));
      setStyleExpandedV(true);
    }
  }, [styleExpandedV]);

  // reset render-tab expansion state when leaving step 3
  useEffect(() => {
    if (step !== 3) { setStyleCardH(null); setStyleExpandedV(false); setStyleExpandedH(false); }
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
  const [canUndo, setCanUndo] = useState(false);
  const [canRedo, setCanRedo] = useState(false);
  const syncUndoRedo = useCallback(() => {
    setCanUndo(manifestUndoStack.current.length > 0);
    setCanRedo(manifestRedoStack.current.length > 0);
  }, []);
  const setManifest = useCallback((updater: InstText[] | ((prev: InstText[]) => InstText[])) => {
    setManifestRaw((prev) => {
      const next = typeof updater === "function" ? (updater as (p: InstText[]) => InstText[])(prev) : updater;
      if (!manifestSkipHistory.current) {
        manifestUndoStack.current.push(prev);
        if (manifestUndoStack.current.length > 50) manifestUndoStack.current.shift();
        manifestRedoStack.current = [];
      }
      manifestSkipHistory.current = false;
      syncUndoRedo();
      return next;
    });
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
  const orderedManifest = manualOrder.length > 0
    ? [...manifest].sort((a, b) => {
        const ai = manualOrder.indexOf(a.id);
        const bi = manualOrder.indexOf(b.id);
        if (ai === -1 && bi === -1) return (a.reading_order ?? 0) - (b.reading_order ?? 0);
        if (ai === -1) return 1;
        if (bi === -1) return -1;
        return ai - bi;
      })
    : visualReadingOrder(manifest);
  const visibleManifest = orderedManifest.filter((i) => !i.excluded);
  const [imgDim, setImgDim] = useState<[number, number] | null>(null);
  const [sceneRegions, setSceneRegions] = useState<SceneRegion[]>([]);
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
  const [saveStatus, setSaveStatus] = useState<"idle" | "saving" | "saved">("idle");
  const [srcLangLocked, setSrcLangLocked] = useState(true);
  const [importedHash, setImportedHash] = useState<string | null>(null);
  const [hasEditsAfterImport, setHasEditsAfterImport] = useState(false);
  const [importLoading, setImportLoading] = useState(false);
  const [dragOverTranslate, setDragOverTranslate] = useState(false);
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
  const [showTitleConfirm, setShowTitleConfirm] = useState(false);
  const [pendingAssetDelete, setPendingAssetDelete] = useState<{ assetId: string; filename: string | null } | null>(null);
  const [menuOpen, setMenuOpen] = useState(false);
  const menuRef = useRef<HTMLDivElement>(null);
  const [warpOpen, setWarpOpen] = useState(false);
  const [warpCollapsed, setWarpCollapsed] = useState(false);
  const [cleanseDismissed, setCleanseDismissed] = useState(false);
  const warpRef = useRef<HTMLDivElement>(null);
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
    setPreRenderUrl(null);
    setPreviewRenderError(null);
    setPreviewCacheKey(null);
    setPreviewTransaction({ phase: "idle", requestId: previewRenderSeq.current, cause: "initial" });
    setInpaintingProviders([]);
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
    manifestUndoStack.current = [];
    manifestRedoStack.current = [];
    syncUndoRedo();
    manifestSkipHistory.current = true;
    setManifest(m.instances);
    setImgDim(m.img_dim);
    setSceneRegions(m.scene_regions ?? []);
    setSemanticUnits(m.semantic_units ?? []);
    if (m.src_lang) setSrcLang(m.src_lang);
    if (m.instances.length > 0) setStep(1);
    return p;
  }, [syncUndoRedo]);

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
    loadProjectSession(p.id)
      .then((full) => {
        if (full.active_asset?.has_manifest) {
          addToast("success", `resumed session: "${p.name}"`);
        }
      })
      .catch(() => {});
  }, [resetSession, loadProjectSession, addToast, displayedScreen]);

  // auto-resume project session when reloading directly into main screen
  const sessionRestoredRef = useRef(false);
  useEffect(() => {
    if (screen !== "main" || sessionRestoredRef.current) return;
    const pid = localStorage.getItem(PROJECT_KEY);
    if (!pid || project) return;
    sessionRestoredRef.current = true;
    loadProjectSession(pid).catch(() => {
      sessionStorage.removeItem(SCREEN_KEY);
      setScreen("title");
    });
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
    // This exact next value, not the pre-update closure, drives both the
    // manifest save and the preview revision.  It is the fix for a chosen
    // Bold/Italic face appearing in Translate but reverting in Render.
    autoSave(next);
  }, [renderSelId, prevSelId, recordLocalizedChange, manifest, autoSave, setManifest]);

  const updateSelectedGarnish = useCallback((patch: Partial<NonNullable<InstText["garnish_override"]>>) => {
    const selected = renderSelId ?? prevSelId;
    if (!selected) return;
    recordLocalizedChange();
    const next = manifest.map((inst) => {
      if (inst.id !== selected) return inst;
      const b = inst.bounding_box;
      const surface = sceneRegions.find((region) => b.x + b.width / 2 >= region.bbox.x && b.x + b.width / 2 <= region.bbox.x + region.bbox.width && b.y + b.height / 2 >= region.bbox.y && b.y + b.height / 2 <= region.bbox.y + region.bbox.height);
      const baseProfile = inst.garnish_override ?? surface?.garnish_profile ?? DEFAULT_GARNISH_PROFILE;
      const perRegion = inst.garnish_scope === "per_region" && selectedGarnishRegionId;
      if (perRegion) return { ...inst, garnish_regions: (inst.garnish_regions ?? []).map((region) => region.id === selectedGarnishRegionId ? {
        ...region, profile: { ...baseProfile, ...(region.profile ?? {}), source_confidence: 1, ...patch },
      } : region) };
      return { ...inst, garnish_override: { ...baseProfile, source_confidence: 1, ...patch } };
    });
    setManifest(next);
    previewCauseRef.current = "garnish";
    autoSave(next);
  }, [renderSelId, prevSelId, selectedGarnishRegionId, recordLocalizedChange, manifest, sceneRegions, autoSave, setManifest]);

  const setSelectedGarnishEnabled = useCallback((enabled: boolean) => {
    const selected = renderSelId ?? prevSelId;
    if (!selected) return;
    recordLocalizedChange();
    const next = manifest.map((inst) => {
      if (inst.id !== selected) return inst;
      return inst.garnish_scope === "per_region" && selectedGarnishRegionId ? { ...inst, garnish_regions: (inst.garnish_regions ?? []).map((region) => region.id === selectedGarnishRegionId ? { ...region, enabled } : region) } : { ...inst, garnish_enabled: enabled };
    });
    setManifest(next);
    previewCauseRef.current = "garnish";
    autoSave(next);
  }, [renderSelId, prevSelId, selectedGarnishRegionId, recordLocalizedChange, manifest, autoSave, setManifest]);

  const useSelectedSceneGarnish = useCallback(() => {
    const selected = renderSelId ?? prevSelId;
    if (!selected) return;
    recordLocalizedChange();
    const next = manifest.map((inst) => {
      if (inst.id !== selected) return inst;
      return inst.garnish_scope === "per_region" && selectedGarnishRegionId ? { ...inst, garnish_regions: (inst.garnish_regions ?? []).map((region) => region.id === selectedGarnishRegionId ? { ...region, profile: null, enabled: null } : region) } : { ...inst, garnish_override: null, garnish_enabled: null };
    });
    setManifest(next);
    previewCauseRef.current = "garnish";
    autoSave(next);
  }, [renderSelId, prevSelId, selectedGarnishRegionId, recordLocalizedChange, manifest, autoSave, setManifest]);

  const setSelectedGarnishScope = useCallback((scope: "whole_selection" | "per_region") => {
    const selected = renderSelId ?? prevSelId;
    if (!selected) return;
    const inst = manifest.find((item) => item.id === selected);
    if (!inst) return;
    if (scope === "per_region" && !(inst.garnish_regions ?? []).length) {
      addToast("warning", "Draw a Garnish region first, then switch to per region.");
      return;
    }
    recordLocalizedChange();
    const next = manifest.map((item) => item.id === selected ? { ...item, garnish_scope: scope, garnish_enabled: scope === "per_region" ? null : item.garnish_enabled } : item);
    setManifest(next);
    previewCauseRef.current = "garnish";
    autoSave(next);
    setSelectedGarnishRegionId(scope === "per_region" ? (inst.garnish_regions ?? [])[0]?.id ?? null : null);
  }, [renderSelId, prevSelId, manifest, recordLocalizedChange, autoSave, addToast]);

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
    if (colorPickMode !== surface || !renderSelId) return false;
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
    const cause = previewCauseRef.current;
    previewCauseRef.current = "style";
    setPreviewTransaction({ phase: "debouncing", requestId: seq, cause });
    const timer = setTimeout(async () => {
      try {
        const snapshot = currentManifest();
        if (!snapshot) {
          if (seq === previewRenderSeq.current) setPreviewTransaction({ phase: "idle", requestId: seq, cause });
          return;
        }
        if (seq === previewRenderSeq.current) setPreviewTransaction({ phase: "rendering", requestId: seq, cause });
        const result = await renderPreview(asset.asset_id, targLang, snapshot);
        if (seq === previewRenderSeq.current) {
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
          setRepairFallbackIds(unavailable.map((repair) => repair.id));
          setRepairReviews(repairItems.filter((repair) => !unavailable.includes(repair)));
        }
      } catch (e) {
        if (seq === previewRenderSeq.current) {
          const error = String(e);
          setPreviewRenderError(error);
          setPreviewTransaction({ phase: "failed", requestId: seq, cause, error });
        }
        // The last good canvas remains visible while a transient preview
        // request fails; final Render still reports actionable errors.
      }
    }, 600);
    return () => clearTimeout(timer);
    // Scene's returned garnish metadata is presentation data.  It must not
    // be a dependency here: accepting it after a successful response would
    // otherwise schedule a second, latent canvas render.
  }, [step, asset, targLang, manifest, patchRevision, previewRetryRevision]);

  useEffect(() => {
    if (step !== 3) return;
    getInpaintingProviders()
      .then(({ providers }) => setInpaintingProviders(providers))
      .catch(() => setInpaintingProviders([]));
  }, [step]);

  useEffect(() => {
    if (!asset) return;
    getTreatment(asset.asset_id)
      .then((state) => syncTreatmentPatches(state.patches))
      .catch(() => { setInpaintPatchIds([]); setAppliedCandidateIds([]); });
  }, [asset, syncTreatmentPatches]);

  const applyLasso = useCallback(async () => {
    if (!asset || lassoPoints.length < 3) return;
    try {
      recordLocalizedChange();
      const snapshot = await flushCurrentManifest();
      const result = await createInpaintPatch(asset.asset_id, { polygon: lassoPoints, mode: "auto", radius: brushRadius, hardness: brushHardness }, snapshot ?? undefined);
      syncTreatmentPatches(result.patches);
      setLassoPoints([]);
      setPatchRevision((v) => v + 1);
      addToast("success", "lasso area inpainted");
    } catch (e) { setErrorWithNotif(String(e)); }
  }, [asset, lassoPoints, addToast, brushRadius, brushHardness, flushCurrentManifest, syncTreatmentPatches, recordLocalizedChange]);

  const addGarnishRegion = useCallback(() => {
    const selected = renderSelId ?? prevSelId;
    if (!selected || lassoPoints.length < 3) return;
    const inst = manifest.find((item) => item.id === selected);
    if (!inst) return;
    const b = inst.bounding_box;
    // Clamp editor points to the immutable instance box before persistence.
    const polygon = lassoPoints.map(([x, y]) => [Math.max(b.x, Math.min(b.x + b.width, x)), Math.max(b.y, Math.min(b.y + b.height, y))]);
    const id = `${inst.id}-g${Date.now().toString(36)}`;
    recordLocalizedChange();
    const next = manifest.map((item) => item.id === selected ? { ...item, garnish_scope: "per_region" as const, garnish_regions: [...(item.garnish_regions ?? []), { id, polygon, enabled: true, profile: null, source: "manual" }] } : item);
    setManifest(next);
    previewCauseRef.current = "garnish";
    autoSave(next);
    setSelectedGarnishRegionId(id);
    setGarnishRegionMode(false);
    setLassoPoints([]);
    addToast("success", "Garnish sub-region added");
  }, [renderSelId, prevSelId, lassoPoints, manifest, recordLocalizedChange, autoSave, addToast]);

  const deleteSelectedGarnishRegion = useCallback(() => {
    const selected = renderSelId ?? prevSelId;
    if (!selected || !selectedGarnishRegionId) return;
    recordLocalizedChange();
    const next = manifest.map((item) => item.id === selected ? { ...item, garnish_regions: (item.garnish_regions ?? []).filter((region) => region.id !== selectedGarnishRegionId) } : item);
    setManifest(next);
    previewCauseRef.current = "garnish";
    autoSave(next);
    setSelectedGarnishRegionId(null);
    addToast("info", "Garnish sub-region removed");
  }, [renderSelId, prevSelId, selectedGarnishRegionId, manifest, recordLocalizedChange, autoSave, addToast]);

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
      addToast("success", result.already_applied ? `${candidate.provider} repair is already in the treatment layer` : `applied ${candidate.provider} repair as an undoable treatment`);
    } catch (e) { setErrorWithNotif(String(e)); }
  }, [asset, addToast, previewPending, previewCacheKey, syncTreatmentPatches, recordLocalizedChange]);

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
    const drawing = brushDrawing.current;
    if (!brushMode || !drawing || drawing.pointerId !== event.pointerId) return;
    const point = pointOnLocalizedCanvas(event);
    if (!point) return;
    event.preventDefault();
    const prior = drawing.stroke.points[drawing.stroke.points.length - 1];
    // Pointer events can arrive several times with the same mapped pixel.
    // Ignore those duplicates while retaining every meaningful held-click path.
    if (prior && prior[0] === point[0] && prior[1] === point[1]) return;
    drawing.stroke = { ...drawing.stroke, points: [...drawing.stroke.points, point] };
    brushDrawing.current = drawing;
    setActiveBrushStroke(drawing.stroke);
    setBrushCursor(point);
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
      // Submit one immutable stroke at a time.  Each patch is built from the
      // current treatment base on the server, so sequential edits persist and
      // an error never discards strokes that were not yet attempted.
      let remaining = [...brushStrokes];
      for (const stroke of brushStrokes) {
        const result = await createInpaintPatch(asset.asset_id, { points: stroke.points, mode: "auto", radius: brushRadius, hardness: brushHardness }, snapshot ?? undefined);
        syncTreatmentPatches(result.patches);
        remaining = remaining.filter((candidate) => candidate.id !== stroke.id);
        setBrushStrokes(remaining);
      }
      setPatchRevision((v) => v + 1);
      addToast("success", "context-aware treatment applied");
    } catch (e) { setErrorWithNotif(String(e)); }
    finally { setBrushApplying(false); }
  }, [asset, brushStrokes, brushRadius, brushHardness, brushApplying, addToast, flushCurrentManifest, syncTreatmentPatches, recordLocalizedChange]);

  const updateCanvasTransform = useCallback((key: "skew_x" | "skew_y" | "arc" | "scale_x" | "scale_y" | "offset_x" | "offset_y", value: number) => {
    const transform = manifest.find((inst) => inst.id === (renderSelId ?? prevSelId))?.style_profile?.transform ?? {};
    // A manual slider is intentionally a Custom transform.  It must not
    // retain a named preset's hidden amount and give the user a result they
    // cannot read back from the controls.
    const { amount: _amount, ...manualTransform } = transform;
    updateSelectedStyle({ transform: { ...manualTransform, preset: "custom", [key]: value } });
  }, [renderSelId, prevSelId, manifest, updateSelectedStyle]);

  const applyCanvasWarpPreset = useCallback((preset: string) => {
    const transform = manifest.find((inst) => inst.id === (renderSelId ?? prevSelId))?.style_profile?.transform ?? {};
    if (preset === "none") {
      updateSelectedStyle({ transform: { ...transform, preset: "none", amount: 0, arc: 0, skew_x: 0, skew_y: 0, scale_x: 1, scale_y: 1, offset_x: 0, offset_y: 0, truncate_offset_x: false, truncate_offset_y: false, wrap_text: false } });
      return;
    }
    updateSelectedStyle({ transform: { ...transform, preset, amount: transform.amount && transform.amount !== 0 ? transform.amount : 12 } });
  }, [renderSelId, prevSelId, manifest, updateSelectedStyle]);

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
        addToast(
          "error",
          `language mismatch: "${uploaded.filename}" is ${langDisplayName(r.detected_lang ?? "?")}, ` +
          `but the project source is ${langDisplayName(r.project_source_lang ?? "?")}. ` +
          "change project source language to proceed.",
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
    } catch (e) {
      setScan(null);
      addToast("warning", `language scan failed: ${e}`);
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
      addToast("success", `"${uploaded.filename}" scanning language…`);
      void runLanguageScan(uploaded, project.id);
    } catch (e) {
      setErrorWithNotif(String(e));
    } finally {
      setBusy(null);
    }
  }, [project, asset, manifest.length, resetSession, addToast, runLanguageScan]);

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
    addToast("info", "override applied — asset kept despite the language scan.");
  }, [addToast]);

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
            : `${ev.pass}/3: ${ev.regions} region(s)`
        );
      } else if (ev.stage === "finalize") {
        setDetectStage("lang");
        setDetectProgress("confirming recipe…");
      } else if (ev.stage === "refine") {
        setDetectStage("drawing");
        setDetectProgress(
          ev.status === "running"
            ? `re-pressing with ${(ev.langset ?? []).map((l) => langDisplayName(l)).join("+")}-tuned recognition…`
            : `refined: ${ev.regions} region(s)`
        );
      } else if (ev.stage === "zoom") {
        setDetectStage("drawing");
        setDetectProgress(
          ev.status === "running"
            ? "inspecting surfaces" /* zooming into surfaces (fine-grain pass...) */
            : `fine-grain: ${ev.regions} region(s)`
        );
      } else if (ev.stage === "paddle_rescue") {
        setDetectStage("drawing");
        setDetectProgress(
          ev.status === "running"
            ? "retasting to ensure flavor consistency…" /* previously: trying a second engine on weak/missed regions… */
            : `rescue pass: ${ev.regions} region(s)`
        );
      } else if (ev.stage === "polish") {
        setDetectStage("lang");
        setDetectProgress(
          ev.status === "running"
            ? "second-look recognition on weak regions…"
            : `polished ${ev.regions} region(s)`
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
            : `enriched ${ev.regions} region(s)`
        );
      } else if (ev.stage === "memory") {
        setDetectStage("lang");
        setDetectProgress(
          ev.status === "running"
            ? "checking translation memory…"
            : (ev.matched ?? 0) > 0 ? `seen before: ${ev.matched} region(s)` : "no memory matches"
        );
      } else if (ev.stage === "complete" && ev.manifest) {
        const m = ev.manifest;
        const detected = langs?.[0] ?? m.src_lang;
        manifestSkipHistory.current = true;
        setManifest(m.instances);
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
          addToast("success", `detected ${m.instances.length} regions`);
          if ((ev.tm_matched ?? 0) > 0) {
            addToast("info", `seen before: ${ev.tm_matched} region(s) matched translation memory — suggestions ready in Translate.`);
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
  }, [asset, addToast]);

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
    try {
      const refined = await refineRegion(asset.asset_id, bbox);
      const best = refined.regions[0];
      const final: BBox = best
        ? { x: best.bbox.x, y: best.bbox.y, width: best.bbox.width, height: best.bbox.height }
        : bbox;
      const finalText = best ? best.text : undefined;
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
  }, [asset, autoSave]);

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

  const onReorder = useCallback((fromId: string, toId: string) => {
    setManualOrder((prevOrder) => {
      const base = prevOrder.length > 0 ? prevOrder : visibleManifest.map((i) => i.id);
      const fromIdx = base.indexOf(fromId);
      const toIdx = base.indexOf(toId);
      if (fromIdx === -1 || toIdx === -1 || fromIdx === toIdx) return prevOrder;
      const next = [...base];
      next.splice(fromIdx, 1);
      next.splice(toIdx, 0, fromId);
      const updated = next.map((id, idx) => {
        const inst = manifest.find((i) => i.id === id);
        return inst ? { ...inst, reading_order: idx } : null;
      }).filter(Boolean) as InstText[];
      autoSave(updated);
      return next;
    });
  }, [visibleManifest, manifest, autoSave]);

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

  const onSemanticPlan = useCallback(async (unitId: string, apply: boolean) => {
    if (!asset) return;
    const unit = semanticUnits.find((item) => item.id === unitId);
    if (!unit) return;
    const target = (semanticDrafts[unitId] ?? unit.substitution?.target_text ?? "").trim();
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
      const next = prev.map((i) => i.id === id ? {
        ...i,
        style_profile: {
          font_family: fontPath,
          font_weight: i.style_profile?.font_weight ?? null,
          color: i.style_profile?.color ?? null,
          font_size: i.style_profile?.font_size ?? null,
          italic: i.style_profile?.italic ?? null,
          underline: i.style_profile?.underline ?? null,
          underline_offset: i.style_profile?.underline_offset ?? null,
          underline_width: i.style_profile?.underline_width ?? null,
          subscript: i.style_profile?.subscript ?? null,
          superscript: i.style_profile?.superscript ?? null,
          align_h: i.style_profile?.align_h ?? null,
          align_v: i.style_profile?.align_v ?? null,
          justification: i.style_profile?.justification ?? null,
          indent: i.style_profile?.indent ?? null,
          tracking: i.style_profile?.tracking ?? null,
          kerning: i.style_profile?.kerning ?? null,
          leading: i.style_profile?.leading ?? null,
          baseline_shift: i.style_profile?.baseline_shift ?? null,
          tab_width: i.style_profile?.tab_width ?? null,
          tsume: i.style_profile?.tsume ?? null,
          stroke_color: i.style_profile?.stroke_color ?? null,
          stroke_width: i.style_profile?.stroke_width ?? null,
          target_orientation: i.style_profile?.target_orientation ?? null,
          word_order: i.style_profile?.word_order ?? null,
          transform: i.style_profile?.transform ?? null,
        },
      } : i);
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
        return {
          ...i,
          style_profile: {
            font_family: i.style_profile?.font_family ?? null,
            font_weight: i.style_profile?.font_weight ?? null,
            color: i.style_profile?.color ?? null,
            font_size: i.style_profile?.font_size ?? null,
            italic: i.style_profile?.italic ?? null,
            underline: i.style_profile?.underline ?? null,
            underline_offset: i.style_profile?.underline_offset ?? null,
            underline_width: i.style_profile?.underline_width ?? null,
            subscript: i.style_profile?.subscript ?? null,
            superscript: i.style_profile?.superscript ?? null,
            align_h: i.style_profile?.align_h ?? null,
            align_v: i.style_profile?.align_v ?? null,
            justification: i.style_profile?.justification ?? null,
            indent: i.style_profile?.indent ?? null,
            tracking: i.style_profile?.tracking ?? null,
            kerning: i.style_profile?.kerning ?? null,
            leading: i.style_profile?.leading ?? null,
            baseline_shift: i.style_profile?.baseline_shift ?? null,
            tab_width: i.style_profile?.tab_width ?? null,
            tsume: i.style_profile?.tsume ?? null,
            stroke_color: i.style_profile?.stroke_color ?? null,
            stroke_width: i.style_profile?.stroke_width ?? null,
            target_orientation: (current === "vertical" ? "horizontal" : "vertical") as "horizontal" | "vertical",
            word_order: i.style_profile?.word_order ?? null,
          },
        };
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
        return {
          ...i,
          style_profile: {
            font_family: i.style_profile?.font_family ?? null,
            font_weight: i.style_profile?.font_weight ?? null,
            color: i.style_profile?.color ?? null,
            font_size: i.style_profile?.font_size ?? null,
            italic: i.style_profile?.italic ?? null,
            underline: i.style_profile?.underline ?? null,
            underline_offset: i.style_profile?.underline_offset ?? null,
            underline_width: i.style_profile?.underline_width ?? null,
            subscript: i.style_profile?.subscript ?? null,
            superscript: i.style_profile?.superscript ?? null,
            align_h: i.style_profile?.align_h ?? null,
            align_v: i.style_profile?.align_v ?? null,
            justification: i.style_profile?.justification ?? null,
            indent: i.style_profile?.indent ?? null,
            tracking: i.style_profile?.tracking ?? null,
            kerning: i.style_profile?.kerning ?? null,
            leading: i.style_profile?.leading ?? null,
            baseline_shift: i.style_profile?.baseline_shift ?? null,
            tab_width: i.style_profile?.tab_width ?? null,
            tsume: i.style_profile?.tsume ?? null,
            stroke_color: i.style_profile?.stroke_color ?? null,
            stroke_width: i.style_profile?.stroke_width ?? null,
            target_orientation: i.style_profile?.target_orientation ?? null,
            word_order: (current === "rtl" ? null : "rtl") as "ltr" | "rtl" | null,
          },
        };
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
  const activeNeuralProvider = inpaintingProviders.find((provider) =>
    provider.available && (provider.kind === "self_hosted" || provider.kind === "experimental")
  );
  const configuredNeuralProvider = inpaintingProviders.find((provider) =>
    provider.enabled && (provider.kind === "self_hosted" || provider.kind === "experimental")
  );

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
          />
        </div>
      </>
    );
  }

  return (
    <div className={`screen-fade${leaving ? " leaving" : ""}`}>
    <div className="mx-auto max-w-7xl space-y-6 p-8 pb-12">
      <header className="relative z-[200] flex items-center gap-4">
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
        <div className="relative" ref={menuRef}>
          <button
            onClick={() => setMenuOpen((v) => !v)}
            className="bezier-card flex items-center gap-2 rounded-lg bg-white/60 px-3 py-2 text-sm text-zinc-700 transition hover:bg-zinc-100 dark:bg-zinc-900/60 dark:text-zinc-300 dark:hover:bg-zinc-800"
          >
            <Home size={14} />
            <ChevronDown size={14} className={`transition ${menuOpen ? "rotate-180" : ""}`} />
          </button>
          {/* Dropdown Morph — reusable expand/collapse animation (see .dropdown-morph in uikit.css) */}
          <div className={`dropdown-morph bezier-card absolute right-0 top-full z-[200] mt-2 w-44 rounded-lg bg-white p-1.5 dark:bg-zinc-900${menuOpen ? " expanded" : ""}`}
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
          canCapture={!!asset && !scanBlocked}
          canTranslate={!!asset && manifest.length > 0 && !scanBlocked}
          canRender={translatedCount > 0 && !scanBlocked}
          canVerify={!!renderResult && !scanBlocked}
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
      {step === 0 && (
        <div key="step-0" className="step-fade grid gap-6 md:grid-cols-[minmax(0,1fr)_360px]">
          <Section title="Asset" className={stackClass(0)}>
            <label className="flex min-h-[16rem] cursor-pointer flex-col items-center justify-center gap-2 rounded-lg border-2 border-dashed border-zinc-400 p-4 text-zinc-500 transition hover:border-zinc-600 hover:text-zinc-700 dark:border-zinc-700 dark:hover:border-zinc-500 dark:hover:text-zinc-300">
              {previewUrl ? (
                <img src={previewUrl} alt="preview" className="max-h-72 rounded object-contain" />
              ) : (
                <>
                  <ArrowUpFromLine size={28} />
                  <span className="text-sm">drop or click to upload asset</span>
                  {/* <span className="subtext text-xs text-zinc-500 dark:text-zinc-600">uploading a new image starts a fresh capture session</span> */}
                </>
              )}
              <input
                type="file"
                accept="image/*"
                className="hidden"
                onChange={(e) => {
                  if (e.target.files?.[0]) onFile(e.target.files[0]);
                  e.target.value = "";
                }}
              />
            </label>
            {asset && (
              <p className="subtext mt-2 text-[8px] text-zinc-500">
                {asset.filename} · {asset.asset_info.asset_type} · frames={asset.asset_info.frame_count}
              </p>
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
                  className="rounded px-2 py-1 text-xs text-zinc-500 transition hover:bg-zinc-200 dark:hover:bg-zinc-800 disabled:opacity-30"
                  title="remove asset and start over"
                >
                  clear
                </button>
              </div>
            )}
          </Section>

          <Section title="Project" className={stackClass(1)}>
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
                              className="shrink-0 rounded p-0.5 text-zinc-400 opacity-0 transition hover:text-red-500 group-hover:opacity-100 dark:text-zinc-600 dark:hover:text-red-400"
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

          <div className="flex items-center justify-between">
            <p className="subtext flex items-center gap-1.5 text-[8.4px] text-zinc-500 dark:text-zinc-400">
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
                      <div className="h-2 w-40 overflow-hidden rounded bg-zinc-300 dark:bg-zinc-800">
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
                        loadFontPreview(insight.font_path, insight.family);
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
                                    className="rounded bg-cyan-700 px-1.5 py-0.5 text-[10px] text-white transition hover:bg-cyan-800"
                                  >
                                    {isReview ? "Use substitute" : "Use match"}
                                  </button>
                                )}
                                {insight.region_id && (
                                  <button
                                    onClick={() => { setSelectedId(insight.region_id); setStep(2); }}
                                    className="rounded px-1.5 py-0.5 text-[10px] underline underline-offset-2 transition hover:bg-white/50 dark:hover:bg-black/20"
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
                        if (fam) loadFontPreview(fontPath, fam.family);
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
                {report.issues.map((i, idx) => (
                  <li key={idx} className={i.severity === "error" ? "text-red-600 dark:text-red-400" : "text-amber-600 dark:text-amber-400"}>
                    <span className="font-mono text-xs">{i.code}</span> {i.message}
                    {i.suggestion && <span className="subtext text-zinc-500"> — {i.suggestion}</span>}
                  </li>
                ))}
              </ul>
            </Section>
          )}
        </div>
      )}

      {/* STEP 2: Translate — import translations, per-region fonts */}
      {step === 2 && (
        <div key="step-2" className="step-fade space-y-4">
          {showLangChangePopup && formerTargLang && (
            <div className="fixed inset-0 z-[300] flex items-center justify-center bg-black/30" onClick={onLangChangeProceed}>
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
              />
              <TargetPreviewCanvas
                className="mt-[2px]"
                imageUrl={previewUrl}
                manifest={manifest}
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
                <div className="absolute inset-0 z-[200] flex items-center justify-center rounded-lg border-2 border-dashed border-cyan-500 bg-cyan-50/90 dark:bg-cyan-950/80">
                  <div className="flex flex-col items-center gap-2 text-cyan-700 dark:text-cyan-300">
                    <FaFileImport size={28} />
                    <span className="text-sm font-medium">Import translation</span>
                  </div>
                </div>
              )}
              <SemanticSubstitutionPanel
                units={semanticUnits}
                drafts={semanticDrafts}
                plans={semanticPlans}
                busyId={semanticBusyId}
                onDraftChange={onSemanticDraftChange}
                onPlan={onSemanticPlan}
                theme={theme}
                projectId={project?.id ?? null}
                glossaryStatus={glossaryStatus}
                onGlossaryUpload={onGlossaryUpload}
                onGlossaryDelete={onGlossaryDelete}
                glossaryUploading={glossaryUploading}
                glossaryUploadStep={glossaryUploadStep}
                glossaryUploadError={glossaryUploadError}
              />
              <RegionTable
                mode="translate"
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
          <div className={`grid gap-4 transition-all duration-300 ${styleExpandedH ? "grid-cols-1" : "lg:grid-cols-2"}`}>
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
              className={`${stackClass(0)} flex-1 overflow-auto p-5`}
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
                        className={`rounded px-2 py-1 text-xs font-mono transition ${
                          renderSelId === inst.id
                            ? "bg-cyan-600 text-white"
                            : "bg-zinc-200 text-zinc-600 hover:bg-zinc-300 dark:bg-zinc-800 dark:text-zinc-400 dark:hover:bg-zinc-700"
                        }`}
                      >
                        {inst.id}
                      </button>
                    ))}
                  </div>
                  <button
                    onClick={() => setStyleCollapsed((v) => !v)}
                    className="rounded p-1 text-zinc-500 hover:bg-zinc-200 dark:hover:bg-zinc-800 transition"
                    title={styleCollapsed ? "expand" : "collapse"}
                  >
                    <FcCollapse style={{ transform: styleCollapsed ? "rotate(180deg)" : "none", transition: "transform 0.2s" }} />
                  </button>
                </div>

                <div className={`style-panel-morph ${renderSelId && !styleCollapsed ? "expanded" : ""}`}>
                {(() => {
                  const inst = prevSelId ? manifest.find((i) => i.id === prevSelId) : null;
                  if (!inst) return null;
                  const sp = inst.style_profile;
            const updateStyle = updateSelectedStyle;
            const langForInst = inst.target_language ?? targLang;
            const families = familiesByLang[langForInst] ?? [];
            const currentFont = sp?.font_family ?? null;
            const currentColor = sp?.color ?? null;
            const currentStrokeColor = sp?.stroke_color ?? null;
            const isJapanese = langForInst === "ja";
            const selectedFontFamily = currentFont ? families.find((f) => f.weights.some((w) => w.path === currentFont) || f.best_path === currentFont) : null;
            const selectedFontWeight = currentFont ? selectedFontFamily?.weights.find((w) => w.path === currentFont) ?? null : null;
            if (currentFont) loadFontPreview(currentFont, selectedFontFamily?.family ?? "");
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
                  <div className="space-y-2">
                    {/* Font family + weight */}
                    <div>
                      <label className="subtext mb-1 block text-xs text-zinc-500">font</label>
                      <FontCombobox
                        value={currentFont}
                        families={families}
                        onChange={(path, _family, weight) => {
                          updateStyle({ font_family: path || null, font_weight: weight?.subfamily ?? null });
                        }}
                        placeholder="auto (best coverage)"
                      />
                    </div>

                    {/* Font size */}
                    <div>
                      <label className="subtext mb-1 block text-xs text-zinc-500">font size (px)</label>
                      <div className="flex items-center gap-2">
                        <input
                          type="number"
                          min={6}
                          value={sp?.font_size ?? ""}
                          onChange={(e) => updateStyle({ font_size: e.target.value ? Number(e.target.value) : null })}
                          placeholder="auto-fit"
                          className="w-20 rounded border border-zinc-300 bg-white px-1 py-0.5 text-xs text-zinc-800 dark:border-zinc-700 dark:bg-zinc-900 dark:text-zinc-200"
                        />
                        {inst.characteristics?.size != null && (
                          <button
                            onClick={() => updateStyle({ font_size: inst.characteristics!.size })}
                            className="subtext rounded px-2 py-0.5 text-xs text-zinc-500 hover:bg-zinc-200 dark:hover:bg-zinc-800"
                            title="use the size detected from the source text"
                          >
                            detected ~{inst.characteristics.size}px
                          </button>
                        )}
                      </div>
                    </div>

                    {/* Detected typography (from capture-time analysis) */}
                    {inst.characteristics?.font_style && (
                      <p className="subtext flex items-center gap-1.5 text-[10px] text-zinc-500">
                        <ScanText size={11} className="shrink-0 text-cyan-600 dark:text-cyan-400" />
                        detected style: <span className="font-medium text-zinc-700 dark:text-zinc-300">{inst.characteristics.font_style}</span>
                        {inst.characteristics.positioning?.rotation_deg ? (
                          <span> · rotated {inst.characteristics.positioning.rotation_deg}°</span>
                        ) : null}
                      </p>
                    )}
                    {inst.recognition_history?.length ? (
                      <p className="subtext text-[10px] text-zinc-500" title={inst.recognition_history.map((h) => `${h.engine}: ${h.reason}`).join("\n")}>
                        OCR audit: {inst.recognition_history[inst.recognition_history.length - 1]?.accepted ? "Paddle evidence accepted" : "candidate retained for review"}
                      </p>
                    ) : null}

                    {/* Text alignment: horizontal */}
                    <div>
                      <label className="subtext mb-1 block text-xs text-zinc-500">horizontal alignment</label>
                      <div className="flex gap-1">
                        {([["left", <AlignLeft size={14} key="l" />], ["center", <AlignCenter size={14} key="c" />], ["right", <AlignRight size={14} key="r" />]] as const).map(([val, icon]) => (
                          <button
                            key={val}
                            onClick={() => updateStyle({ align_h: sp?.align_h === val ? null : val })}
                            className={`rounded p-1.5 transition ${sp?.align_h === val ? "bg-cyan-600 text-white" : "bg-zinc-200 text-zinc-600 hover:bg-zinc-300 dark:bg-zinc-800 dark:text-zinc-400 dark:hover:bg-zinc-700"}`}
                            title={val}
                          >{icon}</button>
                        ))}
                      </div>
                    </div>

                    {/* Text alignment: vertical */}
                    <div>
                      <label className="subtext mb-1 block text-xs text-zinc-500">vertical alignment</label>
                      <div className="flex gap-1">
                        {([["top", <AlignStartVertical size={14} key="t" />], ["middle", <AlignCenter size={14} key="m" />], ["bottom", <AlignEndVertical size={14} key="b" />]] as const).map(([val, icon]) => (
                          <button
                            key={val}
                            onClick={() => updateStyle({ align_v: sp?.align_v === val ? null : val })}
                            className={`rounded p-1.5 transition ${sp?.align_v === val ? "bg-cyan-600 text-white" : "bg-zinc-200 text-zinc-600 hover:bg-zinc-300 dark:bg-zinc-800 dark:text-zinc-400 dark:hover:bg-zinc-700"}`}
                            title={val}
                          >{icon}</button>
                        ))}
                      </div>
                    </div>

                    {/* Word order reversal (vertical text only) */}
                  <div>
                    <label className="subtext mb-1 block text-xs text-zinc-500">word order</label>
                      <button
                        onClick={() => updateStyle({ word_order: sp?.word_order === "rtl" ? null : "rtl" })}
                        disabled={sp?.target_orientation !== "vertical"}
                        className={`flex items-center gap-1.5 rounded px-2 py-1.5 text-xs transition ${
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

                    <div>
                      <label className="subtext mb-1 block text-xs text-zinc-500">shape (degrees / arc)</label>
                      <div className="grid grid-cols-3 gap-1">
                        {([['skew_x', 'X'], ['skew_y', 'Y'], ['arc', 'Arc']] as const).map(([key, label]) => (
                          <label key={key} className="text-[10px] text-zinc-500">{label}
                            <input type="number" step="1" min="-25" max="25"
                              value={sp?.transform?.[key] ?? 0}
                              onChange={(e) => updateStyle({ transform: { ...(sp?.transform ?? {}), [key]: Number(e.target.value || 0) } })}
                              onDoubleClick={() => updateStyle({ transform: { ...(sp?.transform ?? {}), [key]: 0 } })}
                              title="double-click to reset to 0"
                              className="mt-0.5 w-full rounded border border-zinc-300 bg-white px-1 py-0.5 text-xs dark:border-zinc-700 dark:bg-zinc-900" />
                          </label>
                        ))}
                      </div>
                    </div>

                    {/* Justification */}
                    <div>
                      <label className="subtext mb-1 block text-xs text-zinc-500">justification</label>
                      <div className="flex gap-1">
                        {([["last_left", <AlignJustify size={14} key="ll" />], ["last_right", <AlignJustify size={14} key="lr" style={{ transform: "scaleX(-1)" }} />], ["justify", <AlignJustify size={14} key="j" />], ["justify_center", <AlignCenter size={14} key="jc" />]] as const).map(([val, icon]) => (
                          <button
                            key={val}
                            onClick={() => updateStyle({ justification: sp?.justification === val ? null : val })}
                            className={`rounded p-1.5 transition ${sp?.justification === val ? "bg-cyan-600 text-white" : "bg-zinc-200 text-zinc-600 hover:bg-zinc-300 dark:bg-zinc-800 dark:text-zinc-400 dark:hover:bg-zinc-700"}`}
                            title={val}
                          >{icon}</button>
                        ))}
                      </div>
                    </div>

                    {/* Compact pixel fields: indent, tracking, kerning, leading, baseline shift, tab width */}
                    <div>
                      <label className="subtext mb-1 block text-xs text-zinc-500">spacing & position (px)</label>
                      <div className="flex flex-wrap gap-1">
                        {([
                          ["indent", "indent"],
                          ["tracking", "track"],
                          ["kerning", "kern"],
                          ["leading", "lead"],
                          ["baseline_shift", "base"],
                          ["tab_width", "tab"],
                        ] as const).map(([key, label]) => (
                          <div key={key} className="flex items-center gap-0.5">
                            <span className="subtext text-[10px] text-zinc-500">{label}</span>
                            <input
                              type="number"
                              step={0.5}
                              value={(sp?.[key] as number | null | undefined) ?? ""}
                              onChange={(e) => updateStyle({ [key]: e.target.value ? Number(e.target.value) : null } as Partial<NonNullable<InstText["style_profile"]>>)}
                              placeholder="—"
                              className="w-12 rounded border border-zinc-300 bg-white px-1 py-0.5 text-xs text-zinc-800 dark:border-zinc-700 dark:bg-zinc-900 dark:text-zinc-200"
                            />
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
                          className="w-full rounded border border-zinc-300 bg-white px-2 py-1 text-xs text-zinc-800 dark:border-zinc-700 dark:bg-zinc-900 dark:text-zinc-200"
                        />
                      </div>
                    )}
                  </div>
                </div>

                {/* === APPEARANCE SECTION === */}
                <div className="border-t border-zinc-200 pt-3 dark:border-zinc-800">
                  <p className="subtext mb-2 text-[10px] font-semibold uppercase tracking-wider text-zinc-400">Appearance</p>
                  <div className="space-y-2">
                    {/* Fill (text color) */}
                    <div>
                      <label className="subtext mb-1 block text-xs text-zinc-500">fill (text color)</label>
                      <div className="flex items-center gap-2">
                        <input
                          type="color"
                          value={currentColor ?? "#000000"}
                          onChange={(e) => updateStyle({ color: e.target.value })}
                          className="h-7 w-10 rounded border border-zinc-300 dark:border-zinc-700"
                        />
                        <button
                          onClick={() => updateStyle({ color: null })}
                          className="subtext rounded px-2 py-0.5 text-xs text-zinc-500 hover:bg-zinc-200 dark:hover:bg-zinc-800"
                        >auto (from scene)</button>
                        <button onClick={() => setColorPickMode(colorPickMode === "source" ? null : "source")}
                          className={`subtext rounded px-2 py-0.5 text-xs ${colorPickMode === "source" ? "bg-cyan-600 text-white" : "text-zinc-500 hover:bg-zinc-200 dark:hover:bg-zinc-800"}`}
                          title="sample a fill color from the source reference">pick source</button>
                        <button onClick={() => setColorPickMode(colorPickMode === "localized" ? null : "localized")}
                          className={`subtext rounded px-2 py-0.5 text-xs ${colorPickMode === "localized" ? "bg-cyan-600 text-white" : "text-zinc-500 hover:bg-zinc-200 dark:hover:bg-zinc-800"}`}
                          title="sample a fill color from the localized canvas">pick canvas</button>
                        {currentColor && (
                          <span className="subtext font-mono text-xs text-zinc-500">{currentColor}</span>
                        )}
                      </div>
                    </div>

                    {/* Stroke */}
                    <div>
                      <label className="subtext mb-1 block text-xs text-zinc-500">stroke</label>
                      <div className="flex items-center gap-2">
                        <input
                          type="color"
                          value={currentStrokeColor ?? "#000000"}
                          onChange={(e) => updateStyle({ stroke_color: e.target.value })}
                          className="h-7 w-10 rounded border border-zinc-300 dark:border-zinc-700"
                        />
                        <input
                          type="number"
                          min={0}
                          step={0.5}
                          value={sp?.stroke_width ?? ""}
                          onChange={(e) => updateStyle({ stroke_width: e.target.value ? Number(e.target.value) : null })}
                          placeholder="width px"
                          className="w-20 rounded border border-zinc-300 bg-white px-2 py-1 text-xs text-zinc-800 dark:border-zinc-700 dark:bg-zinc-900 dark:text-zinc-200"
                        />
                        <button
                          onClick={() => updateStyle({ stroke_color: null, stroke_width: null })}
                          className="subtext rounded px-2 py-0.5 text-xs text-zinc-500 hover:bg-zinc-200 dark:hover:bg-zinc-800"
                        >none</button>
                      </div>
                    </div>

                    {/* Toggle buttons: underline, italic, subscript, superscript */}
                    <div className="flex flex-wrap gap-1">
                      <button
                        onClick={() => updateStyle({ underline: !sp?.underline ? true : null })}
                        className={`flex items-center gap-1 rounded px-2 py-1 text-xs transition ${sp?.underline ? "bg-cyan-600 text-white" : "bg-zinc-200 text-zinc-600 hover:bg-zinc-300 dark:bg-zinc-800 dark:text-zinc-400 dark:hover:bg-zinc-700"}`}
                        title="underline"
                      ><Underline size={14} /> underline</button>
                      <button
                        onClick={() => updateStyle({ italic: !sp?.italic ? true : null })}
                        className={`flex items-center gap-1 rounded px-2 py-1 text-xs transition ${sp?.italic ? "bg-cyan-600 text-white" : "bg-zinc-200 text-zinc-600 hover:bg-zinc-300 dark:bg-zinc-800 dark:text-zinc-400 dark:hover:bg-zinc-700"}`}
                        title="italic"
                      ><Italic size={14} /> italic</button>
                      <button
                        onClick={() => updateStyle({ subscript: !sp?.subscript ? true : null })}
                        className={`flex items-center gap-1 rounded px-2 py-1 text-xs transition ${sp?.subscript ? "bg-cyan-600 text-white" : "bg-zinc-200 text-zinc-600 hover:bg-zinc-300 dark:bg-zinc-800 dark:text-zinc-400 dark:hover:bg-zinc-700"}`}
                        title="subscript"
                      ><Subscript size={14} /> sub</button>
                      <button
                        onClick={() => updateStyle({ superscript: !sp?.superscript ? true : null })}
                        className={`flex items-center gap-1 rounded px-2 py-1 text-xs transition ${sp?.superscript ? "bg-cyan-600 text-white" : "bg-zinc-200 text-zinc-600 hover:bg-zinc-300 dark:bg-zinc-800 dark:text-zinc-400 dark:hover:bg-zinc-700"}`}
                        title="superscript"
                      ><Superscript size={14} /> super</button>
                    </div>
                    {sp?.underline && <div className="mt-2 flex flex-wrap items-center gap-2">
                      <label className="subtext text-xs text-zinc-500">underline offset
                        <input type="number" step={0.5} value={sp.underline_offset ?? ""}
                          onChange={(e) => updateStyle({ underline_offset: e.target.value ? Number(e.target.value) : null })}
                          placeholder="detected" className="ml-1 w-16 rounded border border-zinc-300 bg-white px-1 py-0.5 text-xs dark:border-zinc-700 dark:bg-zinc-900" /> px
                      </label>
                      <label className="subtext text-xs text-zinc-500">underline weight
                        <input type="number" min={0.5} step={0.5} value={sp.underline_width ?? ""}
                          onChange={(e) => updateStyle({ underline_width: e.target.value ? Number(e.target.value) : null })}
                          placeholder="detected" className="ml-1 w-16 rounded border border-zinc-300 bg-white px-1 py-0.5 text-xs dark:border-zinc-700 dark:bg-zinc-900" /> px
                      </label>
                    </div>}
                  </div>
                </div>

                {/* Preview text */}
                <div className="rounded bg-zinc-100 p-2 dark:bg-zinc-950">
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

              {!renderSelId && (
                <p className="subtext flex items-center gap-1 text-xs text-zinc-500">
                  <PiWarningCircleFill size={15} className="text-[#2d8cf0]" />
                  select a region to customize.
                </p>
              )}

              {/* decisions summary: what this render will reflect */}
              <div data-render-decisions="true" className={`bezier-card soft-shadow subtext flex flex-wrap items-center gap-3 rounded-lg bg-white/60 px-4 py-2 text-xs text-zinc-600 dark:bg-zinc-900/60 dark:text-zinc-400 ${stackClass(1)}`}>
                <span>{translatedCount}/{translatableCount} region(s) translated</span>
                <span>
                  targets:{" "}
                  <span className="text-zinc-800 dark:text-zinc-300">
                    {[...new Set(manifest.filter((i) => !i.dnt).map((i) => i.target_language ?? targLang))]
                      .map((l) => langDisplayName(l)).join(", ") || "—"}
                  </span>
                </span>
                <span>
                  fonts:{" "}
                  <span className="text-zinc-800 dark:text-zinc-300">
                    {manifest.filter((i) => !i.dnt).map((i) => {
                      if (!i.style_profile?.font_family) return `auto (${i.id})`;
                      const path = i.style_profile.font_family;
                      const lang = i.target_language ?? targLang;
                      const families = familiesByLang[lang] ?? [];
                      const fam = families.find((f) => f.weights.some((w) => w.path === path) || f.best_path === path);
                      const weight = fam?.weights.find((w) => w.path === path);
                      const weightName = weight ? weightLabel(weight) : null;
                      const name = fam?.family ?? (path.split(/[\\/]/).pop() ?? "").replace(/\.(ttf|otf|ttc|otc)$/i, "");
                      return `${name}${weightName ? ` ${weightName}` : ""} (${i.id})`;
                    }).join("; ")}
                  </span>
                </span>
                {manifest.some((i) => i.dnt) && (
                  <span>{manifest.filter((i) => i.dnt).length} DNT (kept as-is)</span>
                )}
              </div>
            </div>

            {/* Source reference + the single editable localized canvas. */}
            <div className={`space-y-4 ${styleExpandedH ? "grid grid-cols-2 gap-4" : ""}`}>
              {previewUrl && <Section title="Source Reference" icon={<FileImage size={14} />} className={stackClass(2)}>
                <div className="relative inline-block max-w-full">
                  <img src={previewUrl} alt="source reference" className={`block max-w-full rounded-lg border border-zinc-300 dark:border-zinc-800 ${colorPickMode === "source" ? "cursor-crosshair" : ""}`}
                    onPointerDown={(event) => { sampleCanvasFill(event, "source"); }} />
                  {imgDim && renderSelId && manifest.find((inst) => inst.id === renderSelId) && (() => {
                    const b = manifest.find((inst) => inst.id === renderSelId)!.bounding_box;
                    return <svg className="pointer-events-none absolute inset-0 h-full w-full" viewBox={`0 0 ${imgDim[0]} ${imgDim[1]}`} preserveAspectRatio="none"><rect x={b.x} y={b.y} width={b.width} height={b.height} fill="none" stroke="#06b6d4" strokeWidth="2" /></svg>;
                  })()}
                </div>
              </Section>}

              {(preRenderUrl || previewUrl || previewPending || previewRenderError) && <Section title="Localized Asset Canvas" icon={<Sparkles size={14} />} className={stackClass(3)} localized
                headerExtra={previewSyncing && <span className="ml-2 flex items-center gap-1 normal-case text-xs text-cyan-600 dark:text-cyan-400"><SquareLoader size="xs" /> trimming...</span>}
                rightSideHandle={<button onClick={resetLocalizedCanvas} title="Reset all localized canvas edits to the Render-entry baseline" className="absolute right-5 top-4 flex items-center gap-1 rounded px-2 py-1 text-xs text-zinc-500 hover:bg-zinc-200 dark:hover:bg-zinc-800"><VscDebugRestart size={16} /> reset</button>}>
                <div className="mb-2 flex items-center gap-2">
                  <p className="text-xs text-zinc-500">Modify, place, and warp text.</p>
                  {renderSelId && <button type="button" onClick={() => { setGarnishRegionMode((value) => !value); setLassoMode(false); setBrushMode(false); setLassoPoints([]); setBrushCursor(null); }} className={`bezier-card flex items-center justify-center rounded-lg px-2 py-1 text-sm transition ${garnishRegionMode ? "bg-violet-600 text-white hover:bg-violet-700" : "bg-white/60 text-violet-700 hover:bg-violet-100 dark:bg-zinc-900/60 dark:text-violet-300 dark:hover:bg-zinc-800"}`} title="Add Garnish region" aria-label="Add Garnish region"><GiCoolSpices size={16} /></button>}
                </div>
                {!brushMode && !lassoMode && renderSelId && (() => {
                  const inst = manifest.find((item) => item.id === renderSelId);
                  const surface = sceneRegions.find((region) => {
                    const b = inst?.bounding_box; return !!b && b.x + b.width / 2 >= region.bbox.x && b.x + b.width / 2 <= region.bbox.x + region.bbox.width && b.y + b.height / 2 >= region.bbox.y && b.y + b.height / 2 <= region.bbox.y + region.bbox.height;
                  });
                  const recommended = surface?.garnish_profile;
                  const perRegion = inst?.garnish_scope === "per_region";
                  const selectedRegion = perRegion ? ((inst?.garnish_regions ?? []).find((region) => region.id === selectedGarnishRegionId) ?? null) : null;
                  const g = selectedRegion?.profile ?? inst?.garnish_override ?? recommended ?? DEFAULT_GARNISH_PROFILE;
                  if (!inst) return null;
                  const enabled = selectedRegion ? selectedRegion.enabled !== false : inst.garnish_enabled !== false;
                  const slider = (label: string, field: keyof NonNullable<InstText["garnish_override"]>, min: number, max: number, step: number, suffix = "", digits = 1) => {
                    const marker = recommended ? Math.max(0, Math.min(100, (Number(recommended[field]) - min) * 100 / (max - min))) : null;
                    return <label className="flex min-w-36 flex-1 flex-col gap-0.5 text-[10px]" key={field}>
                    <span className="flex items-baseline justify-between gap-1"><span className="font-medium">{label}</span><span className="font-mono">{Number(g[field]).toFixed(digits)}{suffix}</span></span>
                    <span className="relative flex h-4 items-center"><input className="w-full" aria-label={`garnish ${label}`} type="range" min={min} max={max} step={step} value={Number(g[field])} onChange={(e) => updateSelectedGarnish({ [field]: Number(e.target.value) })} />
                      {marker !== null && <span aria-hidden title={`scene recommendation: ${Number(recommended![field]).toFixed(digits)}${suffix}`} className="pointer-events-none absolute top-1/2 h-2.5 w-0.5 -translate-y-1/2 rounded bg-violet-700 dark:bg-violet-200" style={{ left: `${marker}%` }} />}
                    </span>
                    {recommended && <span className="text-[9px] text-violet-700/75 dark:text-violet-300/75">scene: {Number(recommended[field]).toFixed(digits)}{suffix}</span>}
                    </label>;
                  };
                  return <div className="mb-2 rounded border border-violet-400/35 bg-violet-50/60 text-xs text-violet-900 dark:bg-violet-950/20 dark:text-violet-100">
                    <div className="flex flex-wrap items-center gap-2 px-2 py-1.5">
                      <button type="button" onClick={() => setGarnishExpanded((value) => !value)} className="flex items-center gap-1 font-semibold" aria-expanded={garnishExpanded}>
                        <ChevronDown size={14} className={`transition-transform ${garnishExpanded ? "" : "-rotate-90"}`} /><GiCoolSpices size={14} /> Garnish · Natural Scene Integration
                      </button>
                      <select aria-label="Garnish application scope" value={perRegion ? "per_region" : "whole_selection"} onChange={(event) => setSelectedGarnishScope(event.target.value as "whole_selection" | "per_region")} className="rounded border border-violet-400/40 bg-white/70 px-1 py-0.5 text-[10px] dark:bg-zinc-900">
                        <option value="whole_selection">whole selection</option>
                        <option value="per_region">per region</option>
                      </select>
                      {perRegion && <select aria-label="Garnish region" value={selectedRegion?.id ?? ""} onChange={(event) => setSelectedGarnishRegionId(event.target.value || null)} className="rounded border border-violet-400/40 bg-white/70 px-1 py-0.5 text-[10px] dark:bg-zinc-900">
                        {(inst.garnish_regions ?? []).map((region, index) => <option key={region.id} value={region.id}>region {index + 1}</option>)}
                      </select>}
                      {!recommended && <span className="text-[10px] text-violet-700/75 dark:text-violet-300/75">manual baseline</span>}
                      {garnishPreviewSyncing && <span className="text-[10px] text-violet-700 dark:text-violet-300">natural integration</span>}
                      <label className="ml-auto flex items-center gap-1 text-[10px]" onClick={(event) => event.stopPropagation()}>
                        <input aria-label="enable garnish" type="checkbox" checked={enabled} onChange={(event) => setSelectedGarnishEnabled(event.target.checked)} /> enable
                      </label>
                    </div>
                    <div className={`dropdown-morph${garnishExpanded ? " expanded" : ""}`}>
                      <fieldset disabled={!enabled} className="flex flex-wrap gap-x-2 gap-y-1 border-t border-violet-400/25 px-2 py-2 disabled:opacity-45">
                        {slider("edge blur", "edge_blur_px", 0, 10, 0.1, "px")}
                        {slider("wear", "erosion_px", 0, 5, 0.1, "px")}
                        {slider("thicken", "dilation_px", 0, 5, 0.1, "px")}
                        {slider("grain", "grain_strength", 0, 1, 0.02, "", 2)}
                        {slider("gamma", "gamma_shift", 0.5, 2, 0.05, "", 2)}
                        {slider("smudge", "smudge_strength", 0, 1, 0.02, "", 2)}
                        {slider("angle", "smudge_angle_deg", 0, 360, 1, "°", 0)}
                        {recommended && <button type="button" onClick={useSelectedSceneGarnish} className="self-center rounded bg-violet-700 px-1.5 py-0.5 text-[10px] text-white">use scene recommendation</button>}
                        {selectedRegion && <button type="button" onClick={deleteSelectedGarnishRegion} className="self-center rounded bg-zinc-700 px-1.5 py-0.5 text-[10px] text-white">delete sub-region</button>}
                      </fieldset>
                    </div>
                  </div>;
                })()}
                <div className="relative inline-block max-w-full touch-none select-none">
                  <img src={preRenderUrl || previewUrl || "data:image/gif;base64,R0lGODlhAQABAIAAAAAAAP///ywAAAAAAQABAAACAUwAOw=="} alt="localized treatment canvas" draggable={false} className={`block max-w-full touch-none select-none rounded-lg border border-zinc-300 dark:border-zinc-800 ${(brushMode || lassoMode || garnishRegionMode || colorPickMode === "localized") ? "cursor-crosshair" : ""}`}
                    style={{ touchAction: "none", WebkitUserDrag: "none" } as React.CSSProperties}
                    onDragStart={(event) => event.preventDefault()}
                    onPointerDown={(event) => {
                      if (sampleCanvasFill(event, "localized")) return;
                      if (brushMode) beginBrushStroke(event);
                      else if (lassoMode || garnishRegionMode) {
                        const point = pointOnLocalizedCanvas(event);
                        if (point) { event.preventDefault(); setLassoPoints((points) => [...points, point]); }
                      }
                    }}
                    onPointerMove={(event) => {
                      if (brushMode) extendBrushStroke(event);
                      else if (lassoMode || garnishRegionMode) {
                        const point = pointOnLocalizedCanvas(event);
                        if (point) setBrushCursor(point);
                      }
                    }}
                    onPointerUp={(event) => finishBrushStroke(event)}
                    onPointerCancel={(event) => finishBrushStroke(event, true)}
                    onLostPointerCapture={(event) => finishBrushStroke(event, true)}
                    onPointerLeave={() => { if (!brushDrawing.current) setBrushCursor(null); }} />
                  {imgDim && <svg className={`absolute inset-0 h-full w-full ${brushMode || lassoMode || garnishRegionMode ? "pointer-events-none" : ""}`} viewBox={`0 0 ${imgDim[0]} ${imgDim[1]}`} preserveAspectRatio="none">
                    {!brushMode && !lassoMode && !garnishRegionMode && orderedManifest.map((inst) => { const b = inst.bounding_box; const active = inst.id === renderSelId; return <rect key={inst.id} x={b.x} y={b.y} width={b.width} height={b.height} fill={active ? "rgba(6,182,212,.12)" : "transparent"} stroke={active ? "#06b6d4" : "rgba(255,255,255,.7)"} strokeWidth={active ? 2 : 1} onClick={() => setRenderSelId(inst.id)} className="cursor-pointer" />; })}
                    {!brushMode && !lassoMode && !garnishRegionMode && manifest.find((inst) => inst.id === renderSelId)?.garnish_regions?.map((region) => <polygon key={region.id} points={region.polygon.map((point) => point.join(",")).join(" ")} fill={region.id === selectedGarnishRegionId ? "rgba(139,92,246,.20)" : "rgba(139,92,246,.08)"} stroke={region.id === selectedGarnishRegionId ? "#7c3aed" : "rgba(124,58,237,.6)"} strokeWidth="1.5" />)}
                    {brushStrokes.map((stroke) => <polyline key={stroke.id} points={stroke.points.map((p) => p.join(",")).join(" ")} fill="none" stroke="#06b6d4" strokeWidth={brushRadius * 2} strokeLinecap="round" strokeLinejoin="round" opacity=".45" />)}
                    {activeBrushStroke && <polyline points={activeBrushStroke.points.map((p) => p.join(",")).join(" ")} fill="none" stroke="#06b6d4" strokeWidth={brushRadius * 2} strokeLinecap="round" strokeLinejoin="round" opacity=".70" />}
                    {brushMode && brushCursor && <circle cx={brushCursor[0]} cy={brushCursor[1]} r={brushRadius} fill="rgba(6,182,212,.10)" stroke="#06b6d4" strokeWidth="1.5" />}
                    {lassoPoints.length > 0 && <>
                      <polyline points={[...lassoPoints, ...((lassoMode || garnishRegionMode) && brushCursor ? [brushCursor] : [])].map((p) => p.join(",")).join(" ")} fill={garnishRegionMode ? "rgba(139,92,246,.15)" : "rgba(6,182,212,.15)"} stroke={garnishRegionMode ? "#7c3aed" : "#06b6d4"} strokeWidth="2" strokeDasharray={(lassoMode || garnishRegionMode) && brushCursor ? "5 3" : undefined} />
                      {lassoPoints.map((point, index) => <circle key={`${point[0]}-${point[1]}-${index}`} cx={point[0]} cy={point[1]} r="3" fill="#06b6d4" stroke="white" strokeWidth="1" />)}
                    </>}
                  </svg>}
                </div>
                {previewRenderError && <div className="mb-2 flex items-center justify-between gap-2 rounded border border-amber-500/50 bg-amber-50 px-2 py-1 text-xs text-amber-800 dark:bg-amber-950/30 dark:text-amber-200"><span>Preview update failed; showing the last valid composition. {previewRenderError}</span><button type="button" onClick={() => { previewCauseRef.current = "style"; setPreviewRetryRevision((value) => value + 1); }} className="shrink-0 rounded bg-amber-700 px-1.5 py-0.5 text-[10px] text-white">retry</button></div>}
                {inpaintingProviders.length > 0 && !cleanseDismissed && <div className={`mb-2 flex items-center justify-between gap-2 rounded border px-2 py-1 text-xs ${activeNeuralProvider ? "border-emerald-500/40 bg-emerald-50 text-emerald-900 dark:bg-emerald-950/30 dark:text-emerald-100" : "border-zinc-300 bg-zinc-50 text-zinc-700 dark:border-zinc-700 dark:bg-zinc-900 dark:text-zinc-300"}`}>
                  <span><span className="font-medium">Cleanse routing:</span>{" "}{activeNeuralProvider ? `AI-assisted texture repair is available${activeNeuralProvider.promoted ? " for eligible automatic repairs" : " for review"}.` : configuredNeuralProvider ? "AI texture repair is configured but unavailable; deterministic reconstruction remains editable." : "deterministic reconstruction is active; optional AI texture repair is not yet available."}</span>
                  <button onClick={() => setCleanseDismissed(true)} className="shrink-0 rounded px-1.5 py-0.5 text-[10px] text-zinc-500 hover:bg-zinc-200 dark:hover:bg-zinc-800 transition">dismiss</button>
                </div>}
                {repairFallbackIds.length > 0 && <div className="mb-2 rounded border border-sky-500/40 bg-sky-50 px-2 py-2 text-xs text-sky-900 dark:bg-sky-950/30 dark:text-sky-100"><span className="font-medium">Texture treatment is ready for review.</span><span className="ml-1">{repairFallbackIds.length} region{repairFallbackIds.length === 1 ? " uses" : "s use"} the editable reconstruction base.</span></div>}
                {repairReviews.length > 0 && <div className="mb-2 rounded border border-amber-500/50 bg-amber-50 px-2 py-2 text-xs text-amber-900 dark:bg-amber-950/30 dark:text-amber-100">
                  <span className="font-medium">Suggested repairs:</span><span className="ml-1">Compare a proposed repair with the localized canvas, then apply it only if it improves the surface.</span>
                  {(() => {
                    const candidates = [...new Map(repairReviews.flatMap((review) => review.candidates.map((candidate) => [candidate.id, { review, candidate }] as const))).values()];
                    return candidates.length > 0 && <div className="mt-2 flex flex-wrap gap-2">
                      {candidates.map(({ review, candidate }) => {
                        const preview = localizedCandidatePreviews[candidate.id];
                        const ready = preview?.status === "ready";
                        const loading = preview?.status === "loading";
                        const failed = preview?.status === "error";
                        return <div key={candidate.id} className="flex w-20 flex-col items-center gap-1 rounded border border-amber-500/40 bg-white/80 p-1 dark:bg-zinc-900/70">
                          {ready && <img src={preview.url} alt={`localized suggested texture repair for ${review.id}`} className="h-12 w-16 rounded object-contain" />}
                          {loading && <div className="flex h-12 w-16 items-center justify-center rounded bg-amber-100/70 text-amber-700 dark:bg-amber-950/40"><SquareLoader size="xs" /></div>}
                          {failed && <img src={candidate.url} alt={`repair-only background suggestion for ${review.id}`} className="h-12 w-16 rounded object-contain opacity-75" />}
                          {!preview && <div className="h-12 w-16 rounded bg-zinc-100 dark:bg-zinc-800" />}
                          <span className="text-center text-[8px] leading-3 text-amber-800/75 dark:text-amber-200/75">{ready ? "localized preview" : loading ? "building preview" : failed ? "repair-only background" : "preview pending"}</span>
                          {failed ? <button onClick={() => setCandidatePreviewRevision((value) => value + 1)} className="rounded bg-amber-700 px-1.5 py-0.5 text-[10px] text-white">retry</button> :
                            <button disabled={!ready || previewPending || appliedCandidateIds.includes(candidate.id)} onClick={() => applyReviewCandidate(candidate)} className="rounded bg-amber-700 px-1.5 py-0.5 text-[10px] text-white disabled:cursor-not-allowed disabled:opacity-50">{appliedCandidateIds.includes(candidate.id) ? "Applied" : previewPending ? "Updating…" : ready ? "apply" : "waiting…"}</button>}
                        </div>;
                      })}
                    </div>;
                  })()}
                </div>}
                <div className="mb-2 flex flex-wrap items-center gap-2">
                  <button onClick={() => { setBrushMode((v) => !v); setLassoMode(false); setBrushCursor(null); }} className={`rounded px-2 py-1 text-xs ${brushMode ? "bg-cyan-600 text-white" : "bg-zinc-200 text-zinc-700 dark:bg-zinc-800 dark:text-zinc-200"}`}>{brushMode ? "Stop brush" : "Healing brush"}</button>
                  <button onClick={() => { setLassoMode((v) => !v); setGarnishRegionMode(false); setBrushMode(false); setLassoPoints([]); setBrushCursor(null); }} className={`rounded px-2 py-1 text-xs ${lassoMode ? "bg-cyan-600 text-white" : "bg-zinc-200 text-zinc-700 dark:bg-zinc-800 dark:text-zinc-200"}`}>{lassoMode ? "Cancel lasso" : "Content-aware lasso"}</button>
                  <span className="rounded border border-zinc-300 px-2 py-1 text-xs text-zinc-500 dark:border-zinc-700">automatic repair routing</span>
                  {(brushMode || lassoMode) && <label className="text-xs text-zinc-500">radius <input type="range" min="4" max="64" value={brushRadius} onChange={(e) => setBrushRadius(Number(e.target.value))} /><span className="ml-1 font-mono">{brushRadius}px</span></label>}
                  {brushMode && <label className="text-xs text-zinc-500">softness <input type="range" min="0.2" max="1" step="0.05" value={brushHardness} onChange={(e) => setBrushHardness(Number(e.target.value))} /></label>}
                  {brushMode && brushStrokes.length > 0 && <button disabled={brushApplying} onClick={applyBrush} className="rounded bg-cyan-700 px-2 py-1 text-xs text-white disabled:opacity-40">{brushApplying ? "Applying treatmentâ€¦" : `Apply ${brushStrokes.length} stroke${brushStrokes.length === 1 ? "" : "s"}`}</button>}
                  {brushMode && brushStrokes.length > 0 && <button disabled={brushApplying} onClick={() => setBrushStrokes([])} className="rounded bg-zinc-200 px-2 py-1 text-xs text-zinc-700 disabled:opacity-40 dark:bg-zinc-800 dark:text-zinc-200">Discard pending</button>}
                  {(lassoMode || garnishRegionMode) && <span className="text-xs text-zinc-500">click to place points ({lassoPoints.length}/3)</span>}
                  {lassoMode && <button disabled={lassoPoints.length < 3} onClick={applyLasso} className="rounded bg-cyan-700 px-2 py-1 text-xs text-white disabled:opacity-40">Close &amp; repair</button>}
                  {garnishRegionMode && <button disabled={lassoPoints.length < 3} onClick={addGarnishRegion} className="rounded bg-violet-700 px-2 py-1 text-xs text-white disabled:opacity-40">Save Garnish region</button>}
                  {!brushMode && !lassoMode && renderSelId && (() => {
                    const transform = manifest.find((inst) => inst.id === renderSelId)?.style_profile?.transform;
                    return <div className="flex items-center gap-2">
                      <span className="subtext text-[10px] font-semibold uppercase tracking-wider text-zinc-400">text warp</span>
                      <div className="relative shrink-0" ref={warpRef}>
                        <button onClick={() => setWarpOpen((v) => !v)} className="bezier-card flex items-center gap-1.5 rounded-md bg-white/60 px-2 py-1 text-xs text-zinc-700 transition hover:bg-zinc-100 dark:bg-zinc-900/60 dark:text-zinc-300 dark:hover:bg-zinc-800">
                          {WARP_PRESETS.find((p) => p.value === (transform?.preset ?? "custom"))?.label ?? "Custom"}
                          <ChevronDown size={12} className={`transition ${warpOpen ? "rotate-180" : ""}`} />
                        </button>
                        <div className={`dropdown-morph bezier-card absolute left-0 top-full z-[200] mt-1 w-40 rounded-lg bg-white p-1 dark:bg-zinc-900${warpOpen ? " expanded" : ""}`} style={warpOpen ? { boxShadow: "1px 1px 0 var(--bc-shadow), 2px 2px 6px rgba(0,0,0,0.06)" } : undefined}>
                          {WARP_PRESETS.map((preset) => (
                            <button key={preset.value} onClick={() => { applyCanvasWarpPreset(preset.value); setWarpOpen(false); }} className={`flex w-full items-center justify-between gap-2 rounded-md px-2 py-1.5 text-xs transition hover:bg-zinc-100 dark:hover:bg-zinc-800 ${(transform?.preset ?? "custom") === preset.value ? "bg-zinc-100 font-medium text-zinc-900 dark:bg-zinc-800 dark:text-zinc-100" : "text-zinc-600 dark:text-zinc-400"}`}>
                              <span>{preset.label}</span>
                              {preset.value !== "none" && preset.value !== "custom" && <WarpPreview preset={preset.value} />}
                            </button>
                          ))}
                        </div>
                      </div>
                      <div className={`style-panel-morph ml-2 flex-1 ${!warpCollapsed ? "expanded" : ""}`}>
                        <div className="flex flex-wrap items-center gap-2 pt-1">
                      {transform?.preset && transform.preset !== "none" && transform.preset !== "custom" && <label className="subtext flex items-center gap-1 text-xs text-zinc-500">amount<input aria-label="warp amount" type="range" min="-25" max="25" step="0.5" value={transform.amount ?? 12} onChange={(e) => updateSelectedStyle({ transform: { ...transform, amount: Number(e.target.value) } })} onDoubleClick={() => updateSelectedStyle({ transform: { ...transform, amount: 12 } })} title="double-click to return to the preset baseline" /><span className="min-w-9 text-right font-mono text-[10px]">{Number(transform.amount ?? 12).toFixed(1)}</span></label>}
                      {([['skew_x', 'X'], ['skew_y', 'Y']] as const).map(([key, label]) => <label key={key} className="subtext flex items-center gap-1 text-xs text-zinc-500">{label}<span className="text-[10px]">−25</span><input aria-label={`${label} warp`} type="range" min="-25" max="25" step="0.5" value={transform?.[key] ?? 0} onChange={(e) => updateCanvasTransform(key, Number(e.target.value))} onDoubleClick={() => updateCanvasTransform(key, 0)} title="double-click to reset to 0" /><span className="min-w-10 text-right font-mono text-[10px]">{Number(transform?.[key] ?? 0).toFixed(1)}°</span><span className="text-[10px]">+25</span></label>)}
                      {([['scale_x', 'width'], ['scale_y', 'height']] as const).map(([key, label]) => <label key={key} className="subtext flex items-center gap-1 text-xs text-zinc-500">{label}<span className="text-[10px]">0.5x</span><input aria-label={`${label} stretch`} type="range" min="0.5" max="1.5" step="0.01" value={transform?.[key] ?? 1} onChange={(e) => updateCanvasTransform(key, Number(e.target.value))} onDoubleClick={() => updateCanvasTransform(key, 1)} title="double-click to reset to 1.00x" /><span className="min-w-9 text-right font-mono text-[10px]">{Number(transform?.[key] ?? 1).toFixed(2)}x</span><span className="text-[10px]">1.5x</span></label>)}
                      {([['offset_x', 'pos X', 'truncate_offset_x'], ['offset_y', 'pos Y', 'truncate_offset_y']] as const).map(([key, label, truncKey]) => <span key={key} className="subtext flex items-center gap-1 text-xs text-zinc-500"><label className="flex items-center gap-1">{label}<span className="text-[10px]">−50</span><input aria-label={`${label} position`} type="range" min="-50" max="50" step="1" value={transform?.[key] ?? 0} onChange={(e) => updateCanvasTransform(key, Number(e.target.value))} onDoubleClick={() => updateCanvasTransform(key, 0)} title="double-click to snap to original position" /><span className="min-w-9 text-right font-mono text-[10px]">{Number(transform?.[key] ?? 0).toFixed(0)}px</span><span className="text-[10px]">+50</span></label><label className="flex items-center gap-0.5 text-[10px] text-zinc-400"><input type="checkbox" checked={transform?.[truncKey] ?? false} onChange={(e) => updateSelectedStyle({ transform: { ...transform, [truncKey]: e.target.checked } })} /> truncate</label></span>)}
                      <label className="flex items-center gap-0.5 text-[10px] text-zinc-400" title="When off, keep one glyph run even when it crosses the cube edge. When on, wrap only after the measured text exceeds the cube width."><input type="checkbox" checked={transform?.wrap_text ?? false} onChange={(e) => updateSelectedStyle({ transform: { ...transform, wrap_text: e.target.checked } })} /> wrap</label>
                        </div>
                      </div>
                      <button
                        onClick={() => setWarpCollapsed((v) => !v)}
                        className="shrink-0 rounded p-1 text-zinc-500 hover:bg-zinc-200 dark:hover:bg-zinc-800 transition"
                        title={warpCollapsed ? "expand" : "collapse"}
                      >
                        <FcCollapse style={{ transform: warpCollapsed ? "rotate(180deg)" : "none", transition: "transform 0.2s" }} />
                      </button>
                    </div>;
                  })()}
                </div>
              </Section>}

              {/* Render result */}
              {renderResult && (
                <Section title="Render Outcome" icon={<Play size={14} />} className={stackClass(4)}>
              <div className="mb-3 flex flex-wrap items-center gap-2">
                {renderResult.qa_report?.overall_score != null && (
                  <Badge ok={renderResult.qa_passed}>
                    QA: {(renderResult.qa_report.overall_score * 100).toFixed(0)}%
                    {" "}(gate {(renderResult.qa_threshold * 100).toFixed(0)}%)
                  </Badge>
                )}
                {renderResult.text_manifest && (
                  <span className="subtext text-xs text-zinc-500">
                    {renderResult.text_manifest.total_regions} region(s)
                  </span>
                )}
              </div>
              {renderResult.errors.length > 0 && (
                <div className="mb-3 rounded-lg border border-red-300 bg-red-100 px-4 py-2 dark:border-red-800 dark:bg-red-950/50">
                  {renderResult.errors.map((e, i) => (
                    <p key={i} className="text-sm text-red-700 dark:text-red-300">{e}</p>
                  ))}
                </div>
              )}
              {renderResult.output_url ? (
                <a href={renderResult.output_url} target="_blank" rel="noreferrer" className="inline-flex rounded bg-cyan-700 px-3 py-1.5 text-xs text-white">Open finalized localized image</a>
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
                <div className="mt-3 max-h-56 overflow-auto rounded-lg bg-zinc-100 p-3 font-mono text-xs leading-5 dark:bg-zinc-950">
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
                {/* overall + coverage banner */}
                <Section title="Coverage" icon={<TbPhotoScan size={14} />}>
                  <div className="flex flex-wrap items-center gap-2">
                    {qa?.overall_score != null && (
                      <Badge ok={renderResult.qa_passed}>
                        QA: {(qa.overall_score * 100).toFixed(0)}% (gate {(renderResult.qa_threshold * 100).toFixed(0)}%)
                      </Badge>
                    )}
                    {cov && (
                      <>
                        <span className="subtext rounded-full bg-zinc-200 px-2 py-0.5 text-xs text-zinc-700 dark:bg-zinc-800 dark:text-zinc-300">
                          {cov.rendered}/{cov.regions_total} rendered
                        </span>
                        {cov.dnt > 0 && (
                          <span className="subtext rounded-full bg-zinc-200 px-2 py-0.5 text-xs text-zinc-700 dark:bg-zinc-800 dark:text-zinc-300">
                            {cov.dnt} DNT
                          </span>
                        )}
                        {cov.untranslated > 0 && (
                          <span className="subtext flex items-center gap-1 rounded-full bg-red-500/15 px-2 py-0.5 text-xs text-red-600 dark:text-red-400">
                            <AlertTriangle size={11} /> {cov.untranslated} untranslated
                          </span>
                        )}
                        {cov.fallback_font > 0 && (
                          <span className="subtext flex items-center gap-1 rounded-full bg-amber-500/15 px-2 py-0.5 text-xs text-amber-600 dark:text-amber-400">
                            <AlertTriangle size={11} /> {cov.fallback_font} font fallback
                          </span>
                        )}
                      </>
                    )}
                  </div>
                </Section>

                {/* recommendations checklist */}
                {qa?.recommendations && qa.recommendations.length > 0 && (
                  <Section title="Recommendations" icon={<MdTipsAndUpdates size={14} />}>
                    <ul className="space-y-1.5 text-sm">
                      {qa.recommendations.map((r, i) => (
                        <li key={i} className="flex items-start gap-1.5 text-zinc-700 dark:text-zinc-300">
                          <MdTipsAndUpdates size={14} className="mt-0.5 shrink-0 text-[#2d8cf0]" />
                          <span>{r}</span>
                        </li>
                      ))}
                    </ul>
                  </Section>
                )}

                {/* source <-> localized compare */}
                {renderResult.output_url && (
                  <Section title="Compare" icon={<FileImage size={14} />}>
                    <div className="grid gap-4 md:grid-cols-2">
                      <div>
                        <p className="subtext mb-1 text-xs text-zinc-500">source</p>
                        {previewUrl && <img src={previewUrl} alt="source" className="rounded-lg border border-zinc-300 dark:border-zinc-800" />}
                      </div>
                      <div>
                        <p className="subtext mb-1 text-xs text-zinc-500">localized ({langDisplayName(targLang)})</p>
                        <img src={renderResult.output_url} alt="localized" className="rounded-lg border border-zinc-300 dark:border-zinc-800" />
                      </div>
                    </div>
                  </Section>
                )}

                {/* per-region scores + re-render */}
                {Object.keys(per).length > 0 && (
                  <Section title="Per-Region QA" icon={<ScanText size={14} />} className={stackClass(3)}>
                    <div className="space-y-1.5">
                      {Object.entries(per).map(([rid, score]) => {
                        const isSel = verifySelId === rid;
                        return (
                          <div key={rid} className="rounded-lg border border-zinc-200 dark:border-zinc-800">
                            <div
                              role="button"
                              tabIndex={0}
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
                                className="flex items-center gap-1 rounded px-2 py-1 text-xs text-zinc-500 transition hover:bg-zinc-200 disabled:opacity-40 dark:hover:bg-zinc-800"
                              >
                                {reRenderingId === rid ? <Loader2 size={12} className="animate-spin" /> : <RotateCcw size={12} />}
                                re-render
                              </button>
                            </div>
                            {isSel && (
                              <div className="subtext space-y-0.5 border-t border-zinc-200 px-3 py-2 text-xs text-zinc-600 dark:border-zinc-800 dark:text-zinc-400">
                                {asDict("ocr_roundtrip")[rid] != null && <p>OCR round-trip: {(asDict("ocr_roundtrip")[rid] * 100).toFixed(0)}%</p>}
                                {asDict("ring_ssim")[rid] != null && <p>ring SSIM: {(asDict("ring_ssim")[rid] * 100).toFixed(0)}%</p>}
                                {asDict("residual_text")[rid] != null && <p>residual source text: {(asDict("residual_text")[rid] * 100).toFixed(0)}%</p>}
                                {asDict("style_color")[rid] != null && <p>style color match: {(asDict("style_color")[rid] * 100).toFixed(0)}%</p>}
                                {asDict("style_size")[rid] != null && <p>style size match: {(asDict("style_size")[rid] * 100).toFixed(0)}%</p>}
                                {fallbackIds.has(rid) && <p className="text-amber-600 dark:text-amber-400">font swapped: the requested font lacked characters for this text.</p>}
                              </div>
                            )}
                          </div>
                        );
                      })}
                    </div>
                  </Section>
                )}

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
              <Sparkles size={18} className="text-cyan-500 dark:text-cyan-400" />
              <p className="text-sm font-semibold text-zinc-800 dark:text-zinc-200">begin bounding box capture</p>
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

      {showSettings && (
        <div className="title-confirm-backdrop" onClick={() => setShowSettings(false)}>
          <div className="bezier-card title-confirm-card step-fade" style={{ maxWidth: "420px" }} onClick={(e) => e.stopPropagation()}>
            <div className="flex items-center gap-2 mb-4 w-full">
              <Hexagon size={18} className="text-cyan-500 dark:text-cyan-400" />
              <p className="text-sm font-semibold text-zinc-800 dark:text-zinc-200">settings</p>
            </div>
            <div className="space-y-4">
              <div>
                <p className="subtext mb-2 text-xs uppercase tracking-wider text-zinc-500 dark:text-zinc-600">capture</p>
                <div className="space-y-2">
                  <div>
                    <p className="subtext mb-1 text-xs text-zinc-600 dark:text-zinc-400">bounding boxes</p>
                    <BBoxColorDropdown value={bboxColor} onChange={setBboxColor} />
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
