import { useCallback, useEffect, useRef, useState } from "react";
import {
  AlertTriangle, AlignCenter, AlignEndHorizontal, AlignEndVertical, AlignJustify, AlignLeft, AlignRight, AlignStartHorizontal, AlignStartVertical, ArrowLeft, ArrowUpFromLine, Baseline, Bold, BookmarkCheck, Box, Check, ChevronDown, FileImage, FolderOpen, Hexagon, History, Home, Italic, Languages, Loader2,
  Play, Plus, RotateCcw, ScanText, ShieldAlert, Sparkles, SquareStack, Subscript, Superscript, Type, Underline, X,
} from "lucide-react";
import {
  BBox, FontFamily, FontOption, ImportResult, InstText, LanguageOption, Project,
  RenderResult, RenderStreamEvent, SceneRegion, TextManifest, UploadResponse, ValidationReport,
  addRegion, approveRender, deleteProjectAsset, deleteRegion, detectAssetStream, fetchFonts,
  fetchLanguages, getManifest, getProject, importFile, ocrRegion, putManifest, refineRegion,
  renderAsset, renderAssetStream, scanAssetLanguage, snapshotAsset, updateProject, uploadAsset,
  validateAsset,
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
import { langDisplayName, langFlag, LANGUAGE_REGIONS, REGION_ORDER } from "./languageData";
import LanguageCombobox from "./LanguageCombobox";
import FontCombobox, { loadFontPreview, fontNameForPath, weightLabel } from "./FontCombobox";
import BBoxCanvas from "./BBoxCanvas";
import TargetPreviewCanvas from "./TargetPreviewCanvas";
import RegionTable from "./RegionTable";
import ExportPanel from "./ExportPanel";
import ProjectGate from "./ProjectGate";
import HistoryPanel from "./HistoryPanel";
import MemoryPanel from "./MemoryPanel";
import SplashScreen from "./SplashScreen";
import TitleScreen from "./TitleScreen";
import ThemeToggle from "./ThemeToggle";
import { FlipButton, PressButton } from "./Buttons";
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

function Section({ title, icon, children }: { title: string; icon?: React.ReactNode; children: React.ReactNode }) {
  return (
    <section className="bezier-card soft-shadow rounded-xl bg-white/60 p-5 dark:bg-zinc-900/60">
      <h2 className="subtext mb-3 flex items-center gap-2 text-sm font-semibold uppercase tracking-wider text-zinc-500 dark:text-zinc-400">
        {icon} {title}
      </h2>
      {children}
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
        <p className="title-confirm-message">are you sure you want to remove this asset?</p>
        <p className="subtext mt-1 text-center text-xs text-zinc-500">snapshots are taken—nothing is permanently lost.</p>
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

export default function App() {
  const { theme, toggle: toggleTheme } = useTheme();
  const [screen, setScreen] = useState<Screen>("splash");
  const [displayedScreen, setDisplayedScreen] = useState<Screen>("splash");
  const [leaving, setLeaving] = useState(false);
  const [pantryLeaving, setPantryLeaving] = useState(false);
  const [pantryMode, setPantryMode] = useState<"full" | "pantry" | "create">("full");
  const prevScreen = useRef<Screen>("title");
  const transitionTimer = useRef<ReturnType<typeof setTimeout> | null>(null);
  const [pendingBackNav, setPendingBackNav] = useState(false);
  const navGuardRef = useRef(false);

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
  const [asset, setAsset] = useState<UploadResponse | null>(null);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const [languages, setLanguages] = useState<LanguageOption[]>([]);
  const [targLang, setTargLang] = useState("es");
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

  // --- linked canvas state (translate step) ---
  const [canvasesLinked, setCanvasesLinked] = useState(true);
  const [sharedZoom, setSharedZoom] = useState(1);
  const [sharedScroll, setSharedScroll] = useState({ x: 0, y: 0 });

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
  const [error, setError] = useState<string | null>(null);
  const cancelDetectRef = useRef<(() => void) | null>(null);

  const [manifest, setManifest] = useState<InstText[]>([]);
  const [imgDim, setImgDim] = useState<[number, number] | null>(null);
  const [sceneRegions, setSceneRegions] = useState<SceneRegion[]>([]);
  const [srcLang, setSrcLang] = useState<string | null>(null);
  const [selectedId, setSelectedId] = useState<string | null>(null);
  const [drawMode, setDrawMode] = useState(false);
  const [imgSize, setImgSize] = useState<{ width: number; height: number } | null>(null);
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
  const [showTitleConfirm, setShowTitleConfirm] = useState(false);
  const [pendingAssetDelete, setPendingAssetDelete] = useState<{ assetId: string; filename: string | null } | null>(null);
  const [menuOpen, setMenuOpen] = useState(false);
  const menuRef = useRef<HTMLDivElement>(null);
  const [pendingFile, setPendingFile] = useState<File | null>(null);
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
    setManifest([]);
    setImgDim(null);
    setSceneRegions([]);
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
  }, []);

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
    setManifest(m.instances);
    setImgDim(m.img_dim);
    setSceneRegions(m.scene_regions ?? []);
    if (m.src_lang) setSrcLang(m.src_lang);
    if (m.instances.length > 0) setStep(1);
    return p;
  }, []);

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
          addToast("success", `resumed session for "${p.name}"`);
        }
      })
      .catch(() => {});
  }, [resetSession, loadProjectSession, addToast, displayedScreen]);

  // on load: silently rehydrate the last project while the splash plays
  useEffect(() => {
    const saved = localStorage.getItem(PROJECT_KEY);
    if (!saved) return;
    loadProjectSession(saved).catch(() => {
      localStorage.removeItem(PROJECT_KEY);
    });
  }, [loadProjectSession]);

  const refreshProject = useCallback(() => {
    if (!project) return;
    getProject(project.id).then(setProject).catch(() => {});
  }, [project]);

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
  }, [asset, srcLang, targLang, imgDim, imgSize, sceneRegions]);

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
        addToast("warning", "language scan skipped — the ocr engine is not installed on the server.");
        return;
      }
      if (r.locked && r.detected_lang) {
        setSrcLang(r.detected_lang);
        addToast("success", `source language locked to ${langDisplayName(r.detected_lang)} from this asset.`);
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
          "Change your project source language to proceed.",
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
      addToast("success", `"${uploaded.filename}" uploaded — scanning language…`);
      void runLanguageScan(uploaded, project.id);
    } catch (e) {
      setErrorWithNotif(String(e));
    } finally {
      setBusy(null);
    }
  }, [project, asset, manifest.length, resetSession, addToast, runLanguageScan]);

  const onFile = useCallback((file: File) => {
    if (!project) {
      goPantry();
      return;
    }
    // ongoing session? confirm before erasing capture data
    if (asset && manifest.length > 0) {
      setPendingFile(file);
    } else {
      void doUpload(file);
    }
  }, [project, asset, manifest.length, doUpload]);

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
      addToast("success", `project source language changed to ${langDisplayName(scan.detected)}.`);
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
            ? "zooming into surfaces (fine-grain pass)…"
            : `fine-grain: ${ev.regions} region(s)`
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
        setManifest(m.instances);
        setImgDim(m.img_dim);
        setSceneRegions(m.scene_regions ?? []);
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
      setManifest((prev) => {
        const next = prev.filter((i) => i.id !== id);
        autoSave(next);
        return next;
      });
      if (selectedId === id) setSelectedId(null);
    } catch (e) {
      setErrorWithNotif(String(e));
    }
  }, [asset, selectedId, autoSave]);

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
      const next = prev.map((i) => i.id === id ? { ...i, target_text: target } : i);
      autoSave(next);
      if (importedHash) setHasEditsAfterImport(true);
      return next;
    });
  }, [autoSave, importedHash]);

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
        addToast("warning", "no text found in this region. try adjusting the box or entering it manually.");
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

  const onImported = useCallback((result: ImportResult) => {
    if (asset) {
      getManifest(asset.asset_id).then((m) => {
        setManifest(m.instances);
        setImgDim(m.img_dim);
        setSceneRegions(m.scene_regions ?? []);
        setImportedHash(JSON.stringify(m.instances.map((i) => i.target_text)));
        setHasEditsAfterImport(false);
      });
    }
    if (result.missing.length > 0) {
      addToast("warning", `imported ${result.imported} segments. missing: ${result.missing.join(", ")}`);
    } else {
      addToast("success", `imported ${result.imported} translations — targets populated below`);
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
    setTargLang(code);
    if (project) {
      updateProject(project.id, { target_lang: code })
        .then((p) => setProject((prev) => prev ? { ...prev, ...p, assets: prev.assets, active_asset: prev.active_asset } : p))
        .catch(() => {});
    }
  }, [project]);

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
    if (hasEditsAfterImport) {
      const ok = confirm("Discrepancy detected: your live edits differ from the imported file. Proceed with current state?");
      if (!ok) return;
    }
    setError(null);
    setBusy("rendering");
    try {
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
  }, [asset, targLang, hasEditsAfterImport, addToast]);

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
    scene: "re-reading scene context…",
    tofu_regions: "validating per-region fit…",
    cleanse: "erasing source text…",
    scribe: "rendering target text…",
    verify: "scoring quality…",
    save: "saving output…",
  };

  /** the QA Inspector's streamed render: same pipeline as onRender, but
   * over SSE so per-stage progress and recommendations surface live
   * instead of behind one spinner */
  const runVerifyRender = useCallback(() => {
    if (!asset) return;
    setError(null);
    setApproved(false);
    setVerifyBusy("rendering");
    setVerifyStage("tofu");
    cancelVerifyRef.current = renderAssetStream(asset.asset_id, targLang, (ev) => {
      if (ev.stage === "complete") {
        cancelVerifyRef.current = null;
        setVerifyBusy(null);
        setVerifyStage(null);
        const result = renderResultFromEvent(ev);
        setRenderResult(result);
        if (result.text_manifest) setManifest(result.text_manifest.instances);
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
  }, [asset, targLang, addToast]);

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
  const onReRenderRegion = useCallback((id: string) => {
    if (!asset) return;
    setReRenderingId(id);
    setApproved(false);
    renderAssetStream(asset.asset_id, targLang, (ev) => {
      if (ev.stage === "complete") {
        setReRenderingId(null);
        const result = renderResultFromEvent(ev);
        setRenderResult(result);
        if (result.text_manifest) setManifest(result.text_manifest.instances);
        addToast("success", `region ${id} re-rendered`);
      } else if (ev.stage === "error") {
        setReRenderingId(null);
        setErrorWithNotif(ev.message ?? "re-render failed");
      }
    }, (message) => {
      setReRenderingId(null);
      setErrorWithNotif(message);
    }, { regionIds: [id] });
  }, [asset, targLang, addToast]);

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
            onEnter={() => { prevScreen.current = "title"; setScreen(project ? "main" : "pantry"); }}
            onSelectProject={openProject}
            theme={theme}
            onToggleTheme={toggleTheme}
          />
        </div>
      </>
    );
  }

  return (
    <div className={`screen-fade${leaving ? " leaving" : ""}`}>
    <div className="mx-auto max-w-7xl space-y-6 p-8 pb-16">
      <header className="relative z-[200] flex items-center gap-4">
        <img
          src={logoSrc(theme)}
          alt="ToFU"
          className="h-[96px] w-auto cursor-pointer transition hover:opacity-80"
          onClick={() => setShowTitleConfirm(true)}
          title="Return to title"
        />
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
        <NotificationBell notifications={notifications} onClear={clearNotifications} />
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
          <Section title="Asset">
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
                <SquareLoader size="xs" />(draining...)
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

          <Section title="Project">
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
            {saveIndicator}
            {busy === "detecting" && detectProgress && (
              <span className="subtext flex items-center gap-2 text-xs text-cyan-600 dark:text-cyan-400">
                <SquareLoader size="xs" /> {detectProgress}
              </span>
            )}
            <div className="flex-1" />
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

          <div className={canvasExpandedH ? "space-y-4" : "grid gap-4 lg:grid-cols-[minmax(0,1fr)_380px] 2xl:grid-cols-[minmax(0,1fr)_440px]"}>
            <BBoxCanvas
              imageUrl={previewUrl}
              manifest={manifest}
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
            />
            <RegionTable
              mode="capture"
              regions={manifest}
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

          {!canvasExpandedH && (
            <ExportPanel
              assetId={asset?.asset_id ?? ""}
              targLang={targLang}
              disabled={translatableCount === 0}
            />
          )}

          {report && (
            <Section title="Preflight" icon={<ShieldAlert size={14} />}>
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
              title="import translation file (.xliff, .tmx, .tsv, .csv, .txt)"
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
                accept=".xliff,.xlf,.tmx,.tsv,.csv,.txt"
                className="hidden"
                onChange={(e) => e.target.files?.[0] && onImportTranslation(e.target.files[0])}
              />
            </label>
            <span className="subtext text-[8.4px] text-zinc-500 dark:text-zinc-400">
              target: <span className="text-zinc-700 dark:text-zinc-300">{targLang}</span>
            </span>
            <PressButton
              onClick={() => setStep(3)}
              disabled={translatedCount === 0}
              title={translatedCount === 0 ? "Translate at least one region first" : undefined}
            >
              Render
            </PressButton>
          </div>

          <div className="grid gap-4 lg:grid-cols-[320px_minmax(0,1fr)]">
            <div className="flex flex-col gap-2">
              <BBoxCanvas
                imageUrl={previewUrl}
                manifest={manifest}
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
              />
              <TargetPreviewCanvas
                imageUrl={previewUrl}
                manifest={manifest}
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
              />
              <div className="flex justify-center">
                <button
                  onClick={() => setCanvasesLinked((v) => !v)}
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
              className="relative flex flex-col gap-4"
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
              <RegionTable
                mode="translate"
                regions={manifest}
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

          {/* Text + Appearance styling panels */}
          {manifest.length > 0 && (
            <Section title="Text & Appearance" icon={<Type size={14} />}>
              <div className="space-y-3">
                <div className="flex items-center justify-between">
                  {/* Region selector */}
                  <div className="flex flex-wrap gap-1">
                    {manifest.filter((i) => !i.dnt && i.target_text).map((inst) => (
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
            const updateStyle = (patch: Partial<NonNullable<InstText["style_profile"]>>) => {
              setManifest((prev) => prev.map((i) => i.id === (renderSelId ?? prevSelId) ? {
                ...i,
                style_profile: {
                  font_family: i.style_profile?.font_family ?? null,
                  font_weight: i.style_profile?.font_weight ?? null,
                  color: i.style_profile?.color ?? null,
                  font_size: i.style_profile?.font_size ?? null,
                  italic: i.style_profile?.italic ?? null,
                  underline: i.style_profile?.underline ?? null,
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
                  ...patch,
                },
              } : i));
              autoSave(manifest);
            };
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
          )}

          {!renderSelId && (
            <p className="subtext flex items-center gap-1 text-xs text-zinc-500">
              <PiWarningCircleFill size={15} className="text-[#2d8cf0]" />
              select a region to customize.
            </p>
          )}

          {/* decisions summary: what this render will reflect */}
          <div className="bezier-card soft-shadow subtext flex flex-wrap items-center gap-3 rounded-lg bg-white/60 px-4 py-2 text-xs text-zinc-600 dark:bg-zinc-900/60 dark:text-zinc-400">
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

          {/* Always show source image (uneditable) */}
          {previewUrl && (
            <Section title="Source Image (uneditable)" icon={<FileImage size={14} />}>
              <img src={previewUrl} alt="source" className="rounded-lg border border-zinc-300 dark:border-zinc-800" />
            </Section>
          )}

          {/* Render result */}
          {renderResult && (
            <Section title="Localized Result" icon={<Play size={14} />}>
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
                  className="rounded-lg bg-zinc-200 px-3 py-2 text-sm text-zinc-600 transition hover:bg-zinc-300 dark:bg-zinc-800 dark:text-zinc-400 dark:hover:bg-zinc-700"
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
                  <Section title="Per-Region QA" icon={<ScanText size={14} />}>
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
                <div className="bezier-card soft-shadow flex flex-wrap items-center gap-3 rounded-lg bg-white/60 px-4 py-3 dark:bg-zinc-900/60">
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
                  addToast("success", `source language set to ${langDisplayName(srcLang)}`);
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
            <div className="flex items-center gap-2 mb-2 w-full">
              <Hexagon size={18} className="text-cyan-500 dark:text-cyan-400" />
              <p className="text-sm font-semibold text-zinc-800 dark:text-zinc-200">settings</p>
            </div>
            <p className="subtext text-sm text-zinc-600 dark:text-zinc-400 mb-4 w-full">
              configuration options will appear here.
            </p>
            <button
              onClick={() => setShowSettings(false)}
              className="subtext mt-2 w-full text-center text-xs text-zinc-500 hover:text-zinc-700 dark:hover:text-zinc-300"
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
