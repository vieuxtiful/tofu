import { useEffect, useRef, useState } from "react";
import { Archive, ArchiveRestore, ArrowLeft, ArrowRight, ChevronDown, ChevronLeft, ChevronRight, Loader2, Plus, Search, Trash2, X } from "lucide-react";
import { CgRename } from "react-icons/cg";
import { MdMonochromePhotos } from "react-icons/md";
import { AiFillVideoCamera } from "react-icons/ai";
import { TbCubePlus } from "react-icons/tb";
import { AssetKind, LanguageOption, Project, SystemCapabilities, archiveProject, createProject, deleteProject, fetchSystemCapabilities, listProjects, restoreProject, updateProject } from "./api";
import SourceLangPicker from "./SourceLangPicker";
import AnimatedCaretInput from "./AnimatedCaretInput";
import { SquareLoader } from "./Loaders";
import { Theme, logoSrc } from "./theme";

const SEARCH_PLACEHOLDER = "search projects";

interface ProjectGateProps {
  languages: LanguageOption[];
  onSelectProject: (project: Project) => void;
  /** when a project is already open the gate can be dismissed */
  onClose?: (() => void) | null;
  theme: Theme;
  leaving?: boolean;
  /** starting view: "list" (pantry) or "create-name" (new project flow) */
  initialView?: "list" | "create-name";
  /** pantry-only mode: just the list of projects, no create entry point */
  listOnly?: boolean;
  /** id of the project currently open in the workspace (gets the halo) */
  currentProjectId?: string | null;
  /** fired when a project is renamed, so the caller can sync its own state (e.g. the currently open project) */
  onProjectRenamed?: (project: Project) => void;
}

function timeAgo(ts: number): string {
  const s = Math.max(0, Date.now() / 1000 - ts);
  if (s < 60) return "just now";
  if (s < 3600) return `${Math.floor(s / 60)}m ago`;
  if (s < 86400) return `${Math.floor(s / 3600)}h ago`;
  return `${Math.floor(s / 86400)}d ago`;
}

type View = "list" | "create-name" | "create-kind" | "create-src" | "create-lang";

const STEP_HINT: Record<View, string> = {
  "list": "load a session, or start a new one.",
  "create-name": "step 1/4 — name your project.",
  "create-kind": "step 2/4 — select asset type.",
  "create-src": "step 3/4 — pick source language.",
  "create-lang": "step 4/4 — pick target language.",
};

/** Project-first entry: create a project (name → asset kind → source
 * language [default auto] → target language) or load an existing one.
 * The source language gates uploads: assets scanned in another language
 * are rejected unless the user overrides. */
export default function ProjectGate({ languages, onSelectProject, onClose, theme, leaving, initialView, listOnly, currentProjectId, onProjectRenamed }: ProjectGateProps) {
  const [view, setView] = useState<View>(initialView ?? "list");
  const [projects, setProjects] = useState<Project[] | null>(null);
  const [name, setName] = useState("");
  const [assetKind, setAssetKind] = useState<AssetKind | null>(null);
  const [sourceLang, setSourceLang] = useState<string | null | undefined>(undefined); // undefined = nothing selected yet; null = auto
  const [targetLang, setTargetLang] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [pendingDelete, setPendingDelete] = useState<Project | null>(null);
  const [pendingRename, setPendingRename] = useState<Project | null>(null);
  const [renameValue, setRenameValue] = useState("");
  const [renameBusy, setRenameBusy] = useState(false);
  const [wizardLeaving, setWizardLeaving] = useState(false);
  const [showExitConfirm, setShowExitConfirm] = useState(false);
  const [showArchived, setShowArchived] = useState(false);
  const [projectQuery, setProjectQuery] = useState("");
  const [projectSort, setProjectSort] = useState<"updated" | "created" | "name">("updated");
  const [capabilities, setCapabilities] = useState<SystemCapabilities | null>(null);
  const [projPage, setProjPage] = useState(0);
  const [projRowsPerPage, setProjRowsPerPage] = useState(15);
  const [projRowsOpen, setProjRowsOpen] = useState(false);
  const [sortOpen, setSortOpen] = useState(false);
  const projRowsRef = useRef<HTMLDivElement>(null);
  const sortRef = useRef<HTMLDivElement>(null);
  const pantryScrollRef = useRef<HTMLDivElement>(null);
  const viewRef = useRef<View>(view);
  viewRef.current = view;

  // outside-click / escape closes the sort dropdown — same pattern as
  // projRowsOpen above and LanguageCombobox's open state.
  useEffect(() => {
    if (!sortOpen) return;
    const onDown = (e: MouseEvent) => {
      if (sortRef.current && !sortRef.current.contains(e.target as Node)) setSortOpen(false);
    };
    const onKey = (e: KeyboardEvent) => { if (e.key === "Escape") setSortOpen(false); };
    document.addEventListener("mousedown", onDown);
    document.addEventListener("keydown", onKey);
    return () => {
      document.removeEventListener("mousedown", onDown);
      document.removeEventListener("keydown", onKey);
    };
  }, [sortOpen]);

  const transitionTo = (next: View) => {
    if (next === viewRef.current) return;
    setWizardLeaving(true);
    setTimeout(() => {
      setWizardLeaving(false);
      setView(next);
    }, 200);
  };

  const refresh = () => {
    listProjects({ archived: showArchived, query: projectQuery || undefined, sort: projectSort })
      .then((p) => {
        setProjects(p);
        if (p.length === 0 && !listOnly) setView("create-name");
      })
      .catch((e) => { setProjects([]); setError(String(e)); });
  };
  useEffect(refresh, [showArchived, projectQuery, projectSort]);
  useEffect(() => { setProjPage(0); }, [showArchived, projectQuery, projectSort]);
  useEffect(() => {
    if (pantryScrollRef.current) {
      pantryScrollRef.current.scrollTo({ top: 0, behavior: "smooth" });
    }
  }, [projPage]);
  useEffect(() => { fetchSystemCapabilities().then(setCapabilities).catch(() => setCapabilities(null)); }, []);
  useEffect(() => {
    if (!projRowsOpen) return;
    const onDown = (e: MouseEvent) => {
      if (projRowsRef.current && !projRowsRef.current.contains(e.target as Node)) setProjRowsOpen(false);
    };
    document.addEventListener("mousedown", onDown);
    return () => document.removeEventListener("mousedown", onDown);
  }, [projRowsOpen]);

  const onCreate = async () => {
    if (!assetKind || !targetLang) return;
    setBusy(true);
    setError(null);
    try {
      let project = await createProject(name.trim(), targetLang, assetKind);
      if (sourceLang) {
        project = await updateProject(project.id, { source_lang: sourceLang });
      }
      onSelectProject(project);
    } catch (e) {
      setError(String(e));
    } finally {
      setBusy(false);
    }
  };

  const onDelete = async (p: Project) => {
    setPendingDelete(p);
  };

  const confirmDelete = async () => {
    if (!pendingDelete) return;
    try {
      await deleteProject(pendingDelete.id);
      setPendingDelete(null);
      refresh();
    } catch (e) {
      setError(String(e));
      setPendingDelete(null);
    }
  };

  const onRename = (p: Project) => {
    setPendingRename(p);
    setRenameValue(p.name);
  };

  const confirmRename = async () => {
    if (!pendingRename || !renameValue.trim()) return;
    setRenameBusy(true);
    try {
      const updated = await updateProject(pendingRename.id, { name: renameValue.trim() });
      setPendingRename(null);
      setRenameValue("");
      refresh();
      if (updated.id === currentProjectId) onProjectRenamed?.(updated);
    } catch (e) {
      setError(String(e));
      setPendingRename(null);
      setRenameValue("");
    } finally {
      setRenameBusy(false);
    }
  };

  const toggleArchive = async (project: Project) => {
    try {
      if (project.archived_at) await restoreProject(project.id);
      else await archiveProject(project.id);
      refresh();
    } catch (e) { setError(String(e)); }
  };

  const kindCard = (kind: AssetKind, icon: React.ReactNode, title: string, blurb: string, accent: string) => {
    const videoBlocked = kind === "video" && capabilities?.video.project_creation_enabled === false;
    return (
    <button
      onClick={() => !videoBlocked && setAssetKind(kind)}
      disabled={videoBlocked}
      title={videoBlocked ? String(capabilities?.video.reason ?? "Video localization is unavailable") : undefined}
      className={`flex flex-col items-center gap-2 rounded-lg border p-6 text-center transition disabled:cursor-not-allowed disabled:opacity-50 ${
        assetKind === kind
          ? `${accent} `
          : "border-zinc-300 bg-zinc-100 hover:border-cyan-600 dark:border-zinc-700 dark:bg-zinc-800/50"
      }`}
    >
      {icon}
      <p className="text-sm font-medium text-zinc-800 dark:text-zinc-200">{title}</p>
      <p className="subtext text-xs text-zinc-500">{videoBlocked ? String(capabilities?.video.reason ?? blurb) : blurb}</p>
    </button>
  );
  };

  return (
    <div className={`fixed inset-0 z-400 flex items-center justify-center bg-black/70 p-6 pantry-overlay${leaving ? " leaving" : ""}`}>
      <div className={`bezier-card pantry-card${leaving ? " leaving" : ""} flex max-h-[90vh] w-full max-w-2xl flex-col rounded-xl bg-white dark:bg-zinc-900`}>
        <div className="flex items-center justify-between border-b border-zinc-200 px-6 py-4 dark:border-zinc-800">
          <div className="flex items-center gap-3">
            <img src={logoSrc(theme)} alt="ToFU" className="h-10 w-auto" />
            <div>
              <h2 className="text-lg font-semibold text-zinc-800 dark:text-zinc-200">
                {view === "list" ? "pantry" : "new project"}
              </h2>
              <p className="subtext text-xs text-zinc-500">{view === "list" && listOnly ? "load a session." : STEP_HINT[view]}</p>
            </div>
          </div>
          {view !== "list" ? (
            <button
              onClick={() => setShowExitConfirm(true)}
              className="rounded-lg p-1.5 text-zinc-400 transition hover:bg-zinc-200 hover:text-zinc-600 dark:text-zinc-500 dark:hover:bg-zinc-800 dark:hover:text-zinc-300"
              title="Cancel project creation"
            >
              <X size={18} />
            </button>
          ) : onClose ? (
            <button
              onClick={onClose}
              className="rounded-sm p-1 text-zinc-500 hover:bg-zinc-100 hover:text-zinc-700 dark:hover:bg-zinc-800 dark:hover:text-zinc-300"
              title="close"
            >
              <X size={16} />
            </button>
          ) : null}
        </div>

        <div ref={pantryScrollRef} className="pantry-scroll flex-1 overflow-y-auto p-6">
        {error && (
          <div className="subtext mb-3 rounded-lg border border-red-300 bg-red-100 px-3 py-2 text-xs text-red-700 dark:border-red-800 dark:bg-red-950/50 dark:text-red-300">
            {error}
          </div>
        )}

        {view === "list" && (
          <div className="space-y-3">
            <div className="flex flex-wrap items-center gap-2">
              {/* Search input — AnimatedCaretInput: same transparent-ink + mirror
                  + motion caret + insertion reveal + selection highlight pattern
                  as the Capture/Translate text boxes.  While the field is empty,
                  a per-character typewriter placeholder populates the same way
                  project names do in TitleScreen. */}
              <AnimatedCaretInput
                value={projectQuery}
                onChange={setProjectQuery}
                label="search projects"
                typewriterPlaceholder={SEARCH_PLACEHOLDER}
                leadingIcon={<Search size={13} className="text-zinc-500" />}
                className="min-w-45 flex-1 rounded-md border border-zinc-300 dark:border-zinc-700"
              />
              {/* Sort dropdown — dropdown-morph panel matching LanguageCombobox's
                  dropdown CSS (bezier-card + dropdown-morph + expanded). */}
              <div className="relative" ref={sortRef}>
                <button
                  onClick={() => setSortOpen((v) => !v)}
                  className="bezier-card flex items-center gap-1 rounded-lg bg-white/60 px-2 py-1 text-xs text-zinc-700 transition hover:bg-zinc-100 dark:bg-zinc-900/60 dark:text-zinc-300 dark:hover:bg-zinc-800"
                >
                  {projectSort === "updated" ? "recent" : projectSort === "created" ? "created" : "name"}
                  <ChevronDown size={10} className={`shrink-0 text-zinc-500 transition ${sortOpen ? "rotate-180" : ""}`} />
                </button>
                <div
                  className={`dropdown-morph bezier-card absolute left-0 top-full z-100 mt-1 w-28 overflow-hidden rounded-lg border border-zinc-300 bg-white dark:border-zinc-700 dark:bg-zinc-900${sortOpen ? " expanded" : ""}`}
                  style={sortOpen ? { boxShadow: "4px 4px 0 var(--bc-shadow), 8px 8px 16px rgba(0,0,0,0.18)" } : undefined}
                >
                  {(["updated", "created", "name"] as const).map((s) => (
                    <button
                      key={s}
                      onClick={() => { setProjectSort(s); setSortOpen(false); }}
                      className={`flex w-full items-center px-2 py-1.5 text-left text-xs transition hover:bg-zinc-100 dark:hover:bg-zinc-800 ${projectSort === s ? "text-cyan-600 dark:text-cyan-400" : "text-zinc-700 dark:text-zinc-300"}`}
                    >
                      {s === "updated" ? "recent" : s === "created" ? "created" : "name"}
                    </button>
                  ))}
                </div>
              </div>
              {/* Archived toggle — same bezier-card sizing as the sort dropdown
                  button above (px-2 py-1 text-xs), with active/inactive cyan state. */}
              <button
                onClick={() => setShowArchived((value) => !value)}
                className={`bezier-card flex items-center gap-1 rounded-lg px-2 py-1 text-xs transition hover:bg-zinc-100 dark:hover:bg-zinc-800 ${
                  showArchived
                    ? "bg-cyan-600 text-white"
                    : "bg-white/60 text-zinc-700 dark:bg-zinc-900/60 dark:text-zinc-300"
                }`}
              >
                {showArchived ? "active" : "archived"}
              </button>
            </div>
            {!listOnly && (
              <button
                onClick={() => { transitionTo("create-name"); setError(null); }}
                className="flex w-full items-center gap-3 rounded-lg border border-dashed border-zinc-400 p-4 text-left transition hover:border-cyan-600 hover:bg-zinc-100 dark:border-zinc-600 dark:hover:bg-zinc-800/50"
              >
                <Plus size={18} className="text-cyan-600 dark:text-cyan-400" />
                <span className="text-sm font-medium text-zinc-800 dark:text-zinc-200">new</span>
              </button>
            )}
            {projects === null && (
              <div className="subtext step-fade flex items-center gap-2 py-4 text-[10px] text-cyan-600 dark:text-cyan-400">
                <SquareLoader size="sm" /> loading pantry…
              </div>
            )}
            {projects && (() => {
              const sorted = [...projects].sort((a, b) => b.updated_at - a.updated_at);
              const projTotalPages = Math.max(1, Math.ceil(sorted.length / projRowsPerPage));
              const projClampedPage = Math.min(projPage, projTotalPages - 1);
              const projStart = projClampedPage * projRowsPerPage;
              const projEnd = Math.min(projStart + projRowsPerPage, sorted.length);
              const paginated = sorted.slice(projStart, projEnd);
              return (
                <>
                  {paginated.map((p, idx) => {
                    const globalIdx = projStart + idx;
                    const isCurrent = p.id === currentProjectId;
                    return (
                      <div
                        key={p.id}
                        className={`${isCurrent ? "pantry-row-current " : ""}pantry-project-row relative flex items-center gap-3 rounded-lg border border-zinc-300 bg-zinc-100 p-3 transition hover:border-cyan-700 dark:border-zinc-700 dark:bg-zinc-800/50`}
                      >
                        {isCurrent && (
                          <span className="pantry-halo" aria-hidden="true">
                            <span className="pantry-halo-glow" />
                            <span className="pantry-halo-particles" />
                          </span>
                        )}
                        <button
                          onClick={() => onSelectProject(p)}
                          className="relative flex min-w-0 flex-1 items-center gap-3 text-left"
                        >
                          <span className="subtext flex h-7 w-7 shrink-0 items-center justify-center rounded-md border border-zinc-300 bg-white text-xs font-semibold text-zinc-600 dark:border-zinc-600 dark:bg-zinc-800 dark:text-zinc-300">
                            {globalIdx + 1}
                          </span>
                          <div className="min-w-0">
                            <p className="flex items-center gap-2 truncate text-sm font-medium text-zinc-800 dark:text-zinc-200">
                              <span className="truncate">{p.name}</span>
                              <span className={`inline-flex shrink-0 items-center gap-1 rounded-full px-1.5 py-0.5 text-[10px] font-medium leading-none ${
                                p.asset_kind === "video"
                                  ? "bg-purple-500/15 text-purple-600 dark:text-purple-300"
                                  : "bg-cyan-500/15 text-cyan-700 dark:text-cyan-300"
                              }`}>
                                <span className="flex items-center">{p.asset_kind === "video" ? <AiFillVideoCamera size={9} /> : <MdMonochromePhotos size={9} />}</span>
                                {p.asset_kind ?? "image"}
                              </span>
                            </p>
                            <p className="subtext truncate text-xs text-zinc-500">
                              {p.source_lang ?? "auto"} → {p.target_lang}
                              {" · "}{p.asset_count ?? 0} asset(s) · {p.snapshot_count ?? 0} save(s) · {timeAgo(p.updated_at)}
                            </p>
                          </div>
                        </button>
                        <button
                          onClick={() => onRename(p)}
                          title="Rename project"
                          className="relative rounded-sm p-1.5 text-zinc-400 hover:bg-zinc-200 hover:text-cyan-600 dark:text-zinc-600 dark:hover:bg-zinc-700 dark:hover:text-cyan-400"
                        >
                          <CgRename size={14} />
                        </button>
                        <button
                          onClick={() => onDelete(p)}
                          title="Delete project"
                          className="relative rounded-sm p-1.5 text-zinc-400 hover:bg-zinc-200 hover:text-red-500 dark:text-zinc-600 dark:hover:bg-zinc-700 dark:hover:text-red-400"
                        >
                          <Trash2 size={14} />
                        </button>
                        <button onClick={() => toggleArchive(p)} title={p.archived_at ? "Restore project" : "Archive project"} className="relative rounded-sm p-1.5 text-zinc-400 hover:bg-zinc-200 hover:text-cyan-600 dark:text-zinc-600 dark:hover:bg-zinc-700">
                          {p.archived_at ? <ArchiveRestore size={14} /> : <Archive size={14} />}
                        </button>
                      </div>
                    );
                  })}
                  {sorted.length > projRowsPerPage && (
                    <div className="flex items-center justify-end gap-2 pt-1 text-xs text-zinc-500 dark:text-zinc-400">
                      <div className="relative" ref={projRowsRef}>
                        <button
                          type="button"
                          onClick={() => setProjRowsOpen((v) => !v)}
                          className="flex items-center gap-1 rounded-md border border-zinc-300 bg-white px-1.5 py-0.5 text-[10px] text-zinc-600 transition hover:border-cyan-600 dark:border-zinc-700 dark:bg-zinc-800 dark:text-zinc-400"
                          title="Rows per page"
                        >
                          {projRowsPerPage}/page
                          <ChevronDown size={10} className={`transition-transform duration-200 ${projRowsOpen ? "rotate-180" : ""}`} />
                        </button>
                        <div
                          className={`dropdown-morph absolute right-0 top-full z-200 mt-1 w-20 rounded-lg border border-zinc-300 bg-white p-1 dark:border-zinc-700 dark:bg-zinc-900${projRowsOpen ? " expanded" : ""}`}
                          style={projRowsOpen ? { boxShadow: "1px 1px 0 var(--bc-shadow), 2px 2px 6px rgba(0,0,0,0.06)" } : undefined}
                        >
                          {[15, 20, 25, 30].map((n) => (
                            <button
                              key={n}
                              type="button"
                              onClick={() => { setProjRowsPerPage(n); setProjPage(0); setProjRowsOpen(false); }}
                              className={`flex w-full rounded-md px-2 py-1 text-[10px] transition hover:bg-zinc-100 dark:hover:bg-zinc-800 ${projRowsPerPage === n ? "bg-zinc-100 font-medium text-zinc-900 dark:bg-zinc-800 dark:text-zinc-100" : "text-zinc-600 dark:text-zinc-400"}`}
                            >
                              {n}/page
                            </button>
                          ))}
                        </div>
                      </div>
                      <span className="flex items-center gap-0.5">
                        <button
                          onClick={() => setProjPage((p) => Math.max(0, p - 1))}
                          disabled={projClampedPage === 0}
                          title="Previous page"
                          className="rounded-sm p-0.5 text-zinc-500 hover:text-cyan-600 disabled:opacity-30 dark:text-zinc-400 dark:hover:text-cyan-400"
                        >
                          <ChevronLeft size={14} />
                        </button>
                        <span className="tabular-nums">{projClampedPage + 1}/{projTotalPages}</span>
                        <button
                          onClick={() => setProjPage((p) => Math.min(projTotalPages - 1, p + 1))}
                          disabled={projClampedPage >= projTotalPages - 1}
                          title="Next page"
                          className="rounded-sm p-0.5 text-zinc-500 hover:text-cyan-600 disabled:opacity-30 dark:text-zinc-400 dark:hover:text-cyan-400"
                        >
                          <ChevronRight size={14} />
                        </button>
                      </span>
                    </div>
                  )}
                </>
              );
            })()}
          </div>
        )}

        {view === "create-name" && (
          <div className={`wizard-step${wizardLeaving ? " leaving" : ""} space-y-4`}>
            <div>
              <label className="mb-1 block text-xs font-medium text-zinc-600 dark:text-zinc-400">project name</label>
              <input
                autoFocus
                value={name}
                onChange={(e) => setName(e.target.value)}
                onKeyDown={(e) => { if (e.key === "Enter" && name.trim()) transitionTo("create-kind"); }}
                placeholder="e.g. Q3 campaign — Storefront signage"
                className="w-full rounded-lg border border-zinc-300 bg-zinc-50 px-3 py-2 text-sm text-zinc-800 outline-hidden focus:border-cyan-600 dark:border-zinc-700 dark:bg-zinc-950 dark:text-zinc-200"
              />
            </div>
            <div className="flex items-center justify-between">
              {projects && projects.length > 0 ? (
                <button
                  onClick={() => transitionTo("list")}
                  className="flex items-center gap-1 rounded-lg px-3 py-2 text-xs text-zinc-500 hover:text-zinc-700 dark:hover:text-zinc-300"
                >
                  <ArrowLeft size={12} />pantry 
                </button>
              ) : <span />}
              <button
                onClick={() => transitionTo("create-kind")}
                disabled={!name.trim()}
                className="flex items-center gap-2 rounded-lg bg-cyan-600 px-4 py-2 text-sm font-medium text-white transition hover:bg-cyan-500 disabled:opacity-40"
              >
                asset type <ArrowRight size={14} />
              </button>
            </div>
          </div>
        )}

        {view === "create-kind" && (
          <div className={`wizard-step${wizardLeaving ? " leaving" : ""} space-y-4`}>
            <div className="grid gap-3 sm:grid-cols-2">
              {kindCard(
                "image",
                <MdMonochromePhotos size={28} className={assetKind === "image" ? "text-cyan-500" : "text-zinc-500"} />,
                "Image",
                "e.g., signage, posters, screenshots, packaging...", /* signage, posters, screenshots, packaging */
                "border-cyan-500 bg-cyan-50 dark:bg-cyan-950/30"
              )}
              {kindCard(
                "video",
                <AiFillVideoCamera size={28} className={assetKind === "video" ? "text-purple-500" : "text-zinc-500"} />,
                "Video",
                "frame-based. full support is in progress.",
                "border-purple-500 bg-purple-50 dark:bg-purple-950/30"
              )}
            </div>
            <div className="flex items-center justify-between">
              <button
                onClick={() => transitionTo("create-name")}
                className="flex items-center gap-1 rounded-lg px-3 py-2 text-xs text-zinc-500 hover:text-zinc-700 dark:hover:text-zinc-300"
              >
                <ArrowLeft size={12} /> back
              </button>
              <button
                onClick={() => transitionTo("create-src")}
                disabled={!assetKind}
                className="flex items-center gap-2 rounded-lg bg-cyan-600 px-4 py-2 text-sm font-medium text-white transition hover:bg-cyan-500 disabled:opacity-40"
              >
                source language <ArrowRight size={14} />
              </button>
            </div>
          </div>
        )}

        {view === "create-src" && (
          <div className={`wizard-step${wizardLeaving ? " leaving" : ""} space-y-4`}>
            <SourceLangPicker
              selected={sourceLang}
              onSelect={(code) => setSourceLang(code)}
              availableCodes={languages.map((l) => l.code)}
            />
            <div className="flex items-center justify-between">
              <button
                onClick={() => transitionTo("create-kind")}
                className="flex items-center gap-1 rounded-lg px-3 py-2 text-xs text-zinc-500 hover:text-zinc-700 dark:hover:text-zinc-300"
              >
                <ArrowLeft size={12} /> back
              </button>
              <button
                onClick={() => transitionTo("create-lang")}
                disabled={sourceLang === undefined}
                className="flex items-center gap-2 rounded-lg bg-cyan-600 px-4 py-2 text-sm font-medium text-white transition hover:bg-cyan-500 disabled:opacity-40"
              >
                target language <ArrowRight size={14} />
              </button>
            </div>
          </div>
        )}

        {view === "create-lang" && (
          <div className={`wizard-step${wizardLeaving ? " leaving" : ""} space-y-4`}>
            <SourceLangPicker
              selected={targetLang}
              onSelect={(code) => setTargetLang(code)}
              availableCodes={languages.map((l) => l.code)}
              showAuto={false}
            />
            <div className="flex items-center justify-between">
              <button
                onClick={() => transitionTo("create-src")}
                className="flex items-center gap-1 rounded-lg px-3 py-2 text-xs text-zinc-500 hover:text-zinc-700 dark:hover:text-zinc-300"
              >
                <ArrowLeft size={12} /> back
              </button>
              <button
                onClick={onCreate}
                disabled={busy || !name.trim() || !assetKind || !targetLang}
                className="flex items-center gap-2 rounded-lg bg-emerald-600 px-4 py-2 text-sm font-medium text-white transition hover:bg-emerald-500 disabled:opacity-40"
              >
                {busy ? <Loader2 size={14} className="animate-spin" /> : <TbCubePlus size={14} />}
                create
              </button>
            </div>
          </div>
        )}

        {view === "list" && onClose && (
          <div className="mt-6 flex justify-center">
            <button
              onClick={onClose}
              className="rounded-lg px-4 py-2 text-xs text-zinc-500 transition hover:text-zinc-700 dark:text-zinc-400 dark:hover:text-zinc-200"
            >
              Close
            </button>
          </div>
        )}

        {pendingDelete && (
          <div className="title-confirm-backdrop" onClick={() => setPendingDelete(null)}>
            <div className="bezier-card title-confirm-card" style={{ maxWidth: "380px" }} onClick={(e) => e.stopPropagation()}>
              <div className="flex items-center justify-between w-full mb-2">
                <h3 className="text-sm font-semibold text-zinc-800 dark:text-zinc-200">delete project</h3>
                <button
                  onClick={() => setPendingDelete(null)}
                  className="rounded-sm p-1 text-zinc-400 hover:bg-zinc-200 hover:text-zinc-600 dark:hover:bg-zinc-700 dark:hover:text-zinc-200"
                >
                  <X size={14} />
                </button>
              </div>
              <p className="subtext text-sm text-zinc-600 dark:text-zinc-400 mb-5 w-full">
                Delete "{pendingDelete.name}"? Its snapshot history will be removed. Uploaded images stay on disk.
              </p>
              <div className="flex justify-end gap-3 w-full">
                <button
                  onClick={() => setPendingDelete(null)}
                  className="rounded-lg bg-zinc-200 px-4 py-2 text-xs text-zinc-600 transition hover:bg-zinc-300 dark:bg-zinc-800 dark:text-zinc-400 dark:hover:bg-zinc-700 dark:hover:text-zinc-200"
                >
                  No
                </button>
                <button
                  onClick={confirmDelete}
                  className="rounded-lg bg-red-500 px-4 py-2 text-xs font-medium text-white transition hover:bg-red-600"
                >
                  Yes
                </button>
              </div>
            </div>
          </div>
        )}

        {pendingRename && (
          <div className="title-confirm-backdrop" onClick={() => setPendingRename(null)}>
            <div className="bezier-card title-confirm-card" style={{ maxWidth: "380px" }} onClick={(e) => e.stopPropagation()}>
              <div className="flex items-center justify-between w-full mb-2">
                <h3 className="text-sm font-semibold text-zinc-800 dark:text-zinc-200">rename project</h3>
                <button
                  onClick={() => setPendingRename(null)}
                  className="rounded-sm p-1 text-zinc-400 hover:bg-zinc-200 hover:text-zinc-600 dark:hover:bg-zinc-700 dark:hover:text-zinc-200"
                >
                  <X size={14} />
                </button>
              </div>
              <div className="mb-5 w-full">
                <input
                  autoFocus
                  value={renameValue}
                  onChange={(e) => setRenameValue(e.target.value)}
                  onKeyDown={(e) => { if (e.key === "Enter" && renameValue.trim()) confirmRename(); }}
                  placeholder="new project name"
                  className="w-full rounded-lg border border-zinc-300 bg-zinc-50 px-3 py-2 text-sm text-zinc-800 outline-hidden focus:border-cyan-600 dark:border-zinc-700 dark:bg-zinc-950 dark:text-zinc-200"
                />
              </div>
              <div className="flex justify-end gap-3 w-full">
                <button
                  onClick={() => setPendingRename(null)}
                  className="rounded-lg bg-zinc-200 px-4 py-2 text-xs text-zinc-600 transition hover:bg-zinc-300 dark:bg-zinc-800 dark:text-zinc-400 dark:hover:bg-zinc-700 dark:hover:text-zinc-200"
                >
                  Cancel
                </button>
                <button
                  onClick={confirmRename}
                  disabled={renameBusy || !renameValue.trim()}
                  className="flex items-center gap-2 rounded-lg bg-cyan-600 px-4 py-2 text-xs font-medium text-white transition hover:bg-cyan-500 disabled:opacity-40"
                >
                  {renameBusy ? <Loader2 size={12} className="animate-spin" /> : null}
                  Save
                </button>
              </div>
            </div>
          </div>
        )}

        </div>{/* end pantry-scroll */}

        {showExitConfirm && (
          <div className="title-confirm-backdrop" onClick={() => setShowExitConfirm(false)}>
            <div className="bezier-card title-confirm-card" style={{ maxWidth: "360px" }} onClick={(e) => e.stopPropagation()}>
              <p className="title-confirm-message">are you sure?</p>
              <p className="subtext mt-1 text-center text-xs text-zinc-500">Any progress will be lost.</p>
              <div className="title-confirm-actions">
                <button className="title-confirm-btn yes" onClick={() => { setShowExitConfirm(false); onClose?.(); }}>Yes</button>
                <button className="title-confirm-btn no" onClick={() => setShowExitConfirm(false)}>No</button>
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
