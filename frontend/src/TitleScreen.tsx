import { useCallback, useEffect, useRef, useState } from "react";
import { Github, Hexagon, Plus, X } from "lucide-react";
import { FaBoxOpen } from "react-icons/fa";
import { CgRename } from "react-icons/cg";
import { PiBoundingBoxDuotone, PiBoundingBoxFill } from "react-icons/pi";
import { ThemeSwitch } from "./Buttons";
import Grainient from "./Grainient";
import { SquareLoader } from "./Loaders";
import { Theme } from "./theme";
import { Project, listProjects, updateProject } from "./api";

interface TitleScreenProps {
  onEnter: () => void;
  onSelectProject: (p: Project) => void;
  onCreateProject: () => void;
  theme: Theme;
  onToggleTheme: () => void;
}

const FOOTER_TEXT = "ToFU v0.1.0.";

/** the flattened card's resting size — the same numbers the .morphing rule in
 * uikit.css inflates to. kept here because the centring maths needs them. */
const CARD_W = 448;
const CARD_H = 200;

/** work out how far a slab has to glide for the card it becomes to land dead
 * centre. the slot is never transformed, so its rect is the block's own
 * untransformed origin, and the card grows from that top-left anchor — which
 * makes this offset exact at the end of the transition and continuous through
 * it. handed to CSS as --tb-dx / --tb-dy. */
function brineOffsets(slot: HTMLElement | null): React.CSSProperties {
  if (!slot) return {};
  const r = slot.getBoundingClientRect();
  return {
    "--tb-dx": `${window.innerWidth / 2 - (r.left + CARD_W / 2)}px`,
    "--tb-dy": `${window.innerHeight / 2 - (r.top + CARD_H / 2)}px`,
  } as React.CSSProperties;
}

/** title page: the entry hub after the splash. three tofu blocks sit on a
 * tray and levitate on hover; Workspace opens the four-pane main screen,
 * while Pantry (project list) and Settings (dark-mode switch) press flat into
 * a card floating in the blurred brine. */
export default function TitleScreen({ onEnter, onSelectProject, onCreateProject, theme, onToggleTheme }: TitleScreenProps) {
  const [viewLeaving, setViewLeaving] = useState(false);
  const [footerTyped, setFooterTyped] = useState(0);
  const [projects, setProjects] = useState<Project[] | null>(null);
  const [confirmProject, setConfirmProject] = useState<Project | null>(null);
  const [renamingProject, setRenamingProject] = useState<Project | null>(null);
  const [renameValue, setRenameValue] = useState("");

  useEffect(() => {
    const timers: ReturnType<typeof setTimeout>[] = [];
    for (let i = 1; i <= FOOTER_TEXT.length; i++) {
      timers.push(setTimeout(() => setFooterTyped(i), 800 + i * 180));
    }
    return () => timers.forEach(clearTimeout);
  }, []);

  const [morphing, setMorphing] = useState(false);
  const [morphingBack, setMorphingBack] = useState(false);
  const [cardHeight, setCardHeight] = useState<number | null>(null);
  const [pantryOffset, setPantryOffset] = useState<React.CSSProperties>({});
  const dragStartY = useRef(0);
  const dragStartHeight = useRef(0);
  const morphRef = useRef<HTMLDivElement>(null);
  const pantrySlotRef = useRef<HTMLLIElement>(null);

  // settings morph — mirrors the pantry morph pattern
  const [settingsMorphing, setSettingsMorphing] = useState(false);
  const [settingsMorphingBack, setSettingsMorphingBack] = useState(false);
  const [settingsCardHeight, setSettingsCardHeight] = useState<number | null>(null);
  const [settingsOffset, setSettingsOffset] = useState<React.CSSProperties>({});
  const settingsMorphRef = useRef<HTMLDivElement>(null);
  const settingsSlotRef = useRef<HTMLLIElement>(null);

  // a card is open (or on its way back into the tray) — the brine is showing
  const cardOpen = morphing || settingsMorphing;
  const idle = !morphing && !settingsMorphing;

  const morphToPantry = () => {
    setPantryOffset(brineOffsets(pantrySlotRef.current));
    setViewLeaving(true);
    setMorphing(true);
    if (projects === null) {
      listProjects().then(setProjects).catch(() => setProjects([]));
    }
  };

  const morphToMain = () => {
    setMorphingBack(true);
    setViewLeaving(false);
    setCardHeight(null);
    setPantryOffset({});
    setTimeout(() => {
      setMorphing(false);
      setMorphingBack(false);
    }, 400);
  };

  const morphToSettings = () => {
    setSettingsOffset(brineOffsets(settingsSlotRef.current));
    setViewLeaving(true);
    setSettingsMorphing(true);
  };

  const morphSettingsToMain = () => {
    setSettingsMorphingBack(true);
    setViewLeaving(false);
    setSettingsCardHeight(null);
    setSettingsOffset({});
    setTimeout(() => {
      setSettingsMorphing(false);
      setSettingsMorphingBack(false);
    }, 400);
  };

  /** clicking the brine closes whichever card is currently open */
  const closeOpenCard = () => {
    if (morphing && !morphingBack) morphToMain();
    if (settingsMorphing && !settingsMorphingBack) morphSettingsToMain();
  };

  const onSettingsDragStart = useCallback((e: React.MouseEvent) => {
    e.stopPropagation();
    e.preventDefault();
    dragStartY.current = e.clientY;
    dragStartHeight.current = settingsMorphRef.current?.offsetHeight ?? 200;
    const onMove = (ev: MouseEvent) => {
      const delta = ev.clientY - dragStartY.current;
      const newHeight = Math.max(160, Math.min(600, dragStartHeight.current + delta));
      setSettingsCardHeight(newHeight);
    };
    const onUp = () => {
      document.removeEventListener("mousemove", onMove);
      document.removeEventListener("mouseup", onUp);
    };
    document.addEventListener("mousemove", onMove);
    document.addEventListener("mouseup", onUp);
  }, []);

  const onDragStart = useCallback((e: React.MouseEvent) => {
    e.stopPropagation();
    e.preventDefault();
    dragStartY.current = e.clientY;
    dragStartHeight.current = morphRef.current?.offsetHeight ?? 200;
    const onMove = (ev: MouseEvent) => {
      const delta = ev.clientY - dragStartY.current;
      const newHeight = Math.max(200, Math.min(600, dragStartHeight.current + delta));
      setCardHeight(newHeight);
    };
    const onUp = () => {
      document.removeEventListener("mousemove", onMove);
      document.removeEventListener("mouseup", onUp);
    };
    document.addEventListener("mousemove", onMove);
    document.addEventListener("mouseup", onUp);
  }, []);

  const handleSelectProject = (p: Project) => {
    setConfirmProject(p);
  };

  // The card animates in (tco-fade-in / tco-card-in) and the mirrored exit
  // has been sitting in uikit.css unused: unmounting on the state change
  // alone gave it no frame to play. `leaving` applies the class, and the
  // 300ms matches the keyframes -- one duration, not a second constant.
  const [confirmLeaving, setConfirmLeaving] = useState(false);

  const dismissConfirm = (after: () => void) => {
    setConfirmLeaving(true);
    setTimeout(() => {
      setConfirmLeaving(false);
      after();
    }, 300);
  };

  const confirmYes = () => {
    dismissConfirm(() => {
      if (confirmProject) onSelectProject(confirmProject);
      setConfirmProject(null);
    });
  };

  const confirmNo = () => {
    dismissConfirm(() => setConfirmProject(null));
  };

  const startRename = (e: React.MouseEvent, p: Project) => {
    e.stopPropagation();
    setRenamingProject(p);
    setRenameValue(p.name);
  };

  const cancelRename = () => {
    setRenamingProject(null);
    setRenameValue("");
  };

  const confirmRename = async () => {
    if (!renamingProject || !renameValue.trim()) return;
    try {
      const updated = await updateProject(renamingProject.id, { name: renameValue.trim() });
      setProjects((prev) => prev ? prev.map((p) => p.id === updated.id ? updated : p) : prev);
    } catch { /* ignore */ }
    setRenamingProject(null);
    setRenameValue("");
  };

  return (
    <div className="fixed inset-0 z-90 flex flex-col items-center justify-center gap-10 bg-zinc-100 dark:bg-black">
      {theme === "light" && (
        <div className="absolute inset-0 z-0 overflow-hidden">
          <Grainient
            color1="#ffffff"
            color2="#bdbdbd"
            color3="#bfbfbf"
            timeSpeed={1.25}
            colorBalance={0}
            warpStrength={2.2}
            warpFrequency={2.2}
            warpSpeed={2}
            warpAmplitude={50}
            blendAngle={0}
            blendSoftness={0.05}
            rotationAmount={500}
            noiseScale={1.75}
            grainAmount={0.06}
            grainScale={1.2}
            grainAnimated={false}
            contrast={1.5}
            gamma={1.85}
            saturation={1}
            centerX={-0.04}
            centerY={0}
            zoom={0.9}
          />
        </div>
      )}
      {theme === "dark" && (
        <div className="absolute inset-0 z-0 overflow-hidden">
          <Grainient
            color1="#212121"
            color2="#121212"
            color3="#262626"
            timeSpeed={1.25}
            colorBalance={-0.14}
            warpStrength={1.7}
            warpFrequency={2.4}
            warpSpeed={2}
            warpAmplitude={45}
            blendAngle={0}
            blendSoftness={0.05}
            rotationAmount={500}
            noiseScale={1.75}
            grainAmount={0.05}
            grainScale={5.9}
            grainAnimated={false}
            contrast={1.5}
            gamma={1.85}
            saturation={1}
            centerX={-0.04}
            centerY={0}
            zoom={0.9}
          />
        </div>
      )}
      <img src={theme === "light" ? "/tofu-blk-alt-main.png" : "/tofu-wht-alt.png"} alt="ToFU" className="relative z-10 h-36 w-auto" />

      {/* the brine: blurs everything behind an opened card */}
      {cardOpen && (
        <div
          className={`tofu-block-brine${morphingBack || settingsMorphingBack ? " leaving" : ""}`}
          onClick={closeOpenCard}
        />
      )}

      {/* the tray of tofu blocks */}
      <div className={`relative flex items-center justify-center ${cardOpen ? "tofu-rack-raised" : "z-10"}`}>
        <ul className="tofu-block-tray">
          {/* Workspace: no card — straight into the four-pane screen */}
          <li className={`tofu-block-slot${viewLeaving ? " leaving" : ""}`}>
            <button onClick={onEnter} className="tofu-block">
              <span className="tofu-block-grain" aria-hidden />
              {theme === "dark" ? <PiBoundingBoxFill size={22} /> : <PiBoundingBoxDuotone size={22} />}
              <p className="title-mono-text">Workspace</p>
            </button>
          </li>

          {/* Pantry: presses flat into a card */}
          <li
            ref={pantrySlotRef}
            className={`tofu-block-slot${morphing ? " open" : ""}${viewLeaving && !morphing ? " leaving" : ""}`}
          >
          <div
            ref={morphRef}
            className={`tofu-block title-morph${morphing ? " morphing" : ""}${morphingBack ? " morphing-back" : ""}`}
            style={
              morphing && cardHeight !== null
                ? { ...pantryOffset, minHeight: cardHeight, height: cardHeight }
                : pantryOffset
            }
            onClick={idle ? morphToPantry : undefined}
          >
            <span className="tofu-block-grain" aria-hidden />
            {/* Button content — fades out during morph */}
            <div className="title-morph-btn-content">
              <FaBoxOpen size={22} />
              <p className="title-mono-text">Pantry</p>
            </div>
            {/* Card content — fades in during morph */}
            <div className="title-morph-card-content">
              <div className="mb-4 flex items-center justify-between">
                <p className="text-sm font-bold text-zinc-800 dark:text-zinc-200">pantry</p>
                <button
                  onClick={(e) => { e.stopPropagation(); morphToMain(); }}
                  className="flex items-center justify-center rounded-sm p-1 text-zinc-500 transition hover:text-zinc-700 dark:text-zinc-400 dark:hover:text-zinc-200"
                >
                  <X size={14} />
                </button>
              </div>
              <div className="space-y-2">
                {morphing && projects === null && (
                  <div className="subtext step-fade flex items-center gap-4 py-4 text-[10px] text-cyan-600 dark:text-cyan-400">
                    <SquareLoader size="xs" /> simmering…
                  </div>
                )}
                {morphing && projects !== null && projects.length === 0 && (
                  <div className="subtext step-fade flex items-center gap-4 py-4 text-[10px] text-cyan-600 dark:text-cyan-400">
                    pantry is empty
                  </div>
                )}
                {projects && morphing && [...projects]
                  .sort((a, b) => b.updated_at - a.updated_at)
                  .map((p, idx) => {
                    const d = new Date(p.updated_at * 1000);
                    const dd = String(d.getDate()).padStart(2, "0");
                    const mm = String(d.getMonth() + 1).padStart(2, "0");
                    const yyyy = d.getFullYear();
                    const isRenaming = renamingProject?.id === p.id;
                    return (
                      <div
                        key={p.id}
                        className="project-name-row flex w-full items-center gap-1.5 rounded-lg border border-zinc-300 bg-zinc-100 px-3 py-2 text-left text-sm font-medium text-zinc-800 transition hover:border-cyan-600 hover:bg-zinc-200 dark:border-zinc-700 dark:bg-zinc-800/50 dark:text-zinc-200 dark:hover:bg-zinc-800"
                        style={{ animationDelay: `${idx * 500}ms` }}
                      >
                        {isRenaming ? (
                          <div className="flex w-full items-center gap-1.5">
                            <input
                              autoFocus
                              value={renameValue}
                              onChange={(e) => setRenameValue(e.target.value)}
                              onKeyDown={(e) => {
                                if (e.key === "Enter") confirmRename();
                                if (e.key === "Escape") cancelRename();
                              }}
                              onClick={(e) => e.stopPropagation()}
                              className="flex-1 rounded-sm border border-cyan-500 bg-white px-2 py-1 text-sm text-zinc-800 outline-hidden dark:bg-zinc-900 dark:text-zinc-200"
                              placeholder="new name"
                            />
                            <button
                              onClick={(e) => { e.stopPropagation(); confirmRename(); }}
                              className="rounded-sm p-1 text-cyan-600 transition hover:bg-cyan-100 dark:text-cyan-400 dark:hover:bg-cyan-900/30"
                              title="confirm rename"
                            >
                              <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round"><polyline points="20 6 9 17 4 12" /></svg>
                            </button>
                            <button
                              onClick={(e) => { e.stopPropagation(); cancelRename(); }}
                              className="rounded-sm p-1 text-zinc-400 transition hover:bg-zinc-200 dark:hover:bg-zinc-700"
                              title="cancel"
                            >
                              <X size={14} />
                            </button>
                          </div>
                        ) : (
                          <>
                            <button
                              onClick={(e) => { e.stopPropagation(); handleSelectProject(p); }}
                              className="flex flex-1 flex-col items-start gap-0.5"
                            >
                              <span className="project-name-text" style={{ animationDelay: `${idx * 500}ms` }}>
                                {p.name}
                              </span>
                              <span
                                className="project-name-text project-recency"
                                style={{ animationDelay: `${idx * 500 + 500}ms` }}
                              >
                                last: {dd}.{mm}.{yyyy}
                              </span>
                            </button>
                            <button
                              onClick={(e) => startRename(e, p)}
                              className="shrink-0 rounded-sm p-1 text-zinc-400 transition hover:bg-zinc-200 hover:text-cyan-600 dark:text-zinc-500 dark:hover:bg-zinc-700 dark:hover:text-cyan-400"
                              title="rename"
                            >
                              <CgRename size={14} />
                            </button>
                          </>
                        )}
                      </div>
                    );
                  })}
              </div>
              {/* New project button with + icon */}
              {morphing && projects !== null && (
                <button
                  onClick={(e) => { e.stopPropagation(); onCreateProject(); }}
                  className="project-name-row mt-2 flex w-full items-center gap-2 rounded-lg border border-dashed border-zinc-400 px-3 py-2 text-left text-sm font-medium text-zinc-700 transition hover:border-cyan-600 hover:bg-zinc-100 dark:border-zinc-600 dark:text-zinc-300 dark:hover:bg-zinc-800/50"
                >
                  <Plus size={16} className="text-cyan-600 dark:text-cyan-400" />
                  <span className="project-name-text">new project</span>
                </button>
              )}
              {/* Drag handle to expand/collapse card */}
              {morphing && projects !== null && (
                <div className="title-drag-handle" onMouseDown={onDragStart}>
                  <svg width="42" height="14" viewBox="0 0 42 14" fill="none" xmlns="http://www.w3.org/2000/svg">
                    <rect x="0.5" y="1" width="41" height="4.5" rx="2.25" fill="currentColor" />
                    <rect x="11" y="8" width="20" height="4.5" rx="2.25" fill="currentColor" />
                  </svg>
                </div>
              )}
            </div>
          </div>
          </li>

          {/* Settings: presses flat into a card */}
          <li
            ref={settingsSlotRef}
            className={`tofu-block-slot${settingsMorphing ? " open" : ""}${viewLeaving && !settingsMorphing ? " settings-leaving" : ""}`}
          >
          <div
            ref={settingsMorphRef}
            className={`tofu-block title-morph${settingsMorphing ? " morphing" : ""}${settingsMorphingBack ? " morphing-back" : ""}`}
            style={
              settingsMorphing && settingsCardHeight !== null
                ? { ...settingsOffset, minHeight: settingsCardHeight, height: settingsCardHeight }
                : settingsOffset
            }
            onClick={idle ? morphToSettings : undefined}
          >
            <span className="tofu-block-grain" aria-hidden />
            {/* Button content — fades out during morph */}
            <div className="title-morph-btn-content">
              <Hexagon size={22} />
              <p className="title-mono-text">Settings</p>
            </div>
            {/* Card content — fades in during morph */}
            <div className="title-morph-card-content">
              <div className="mb-4 flex items-center justify-between">
                <p className="text-sm font-bold text-zinc-800 dark:text-zinc-200">settings</p>
                <button
                  onClick={(e) => { e.stopPropagation(); morphSettingsToMain(); }}
                  className="flex items-center justify-center rounded-sm p-1 text-zinc-500 transition hover:text-zinc-700 dark:text-zinc-400 dark:hover:text-zinc-200"
                >
                  <X size={14} />
                </button>
              </div>
              <div className="space-y-2">
                {settingsMorphing && (
                  <div
                    className="project-name-row flex w-full items-center justify-between rounded-lg border border-zinc-300 bg-zinc-100 px-3 py-2 text-sm text-zinc-800 dark:border-zinc-700 dark:bg-zinc-800/50 dark:text-zinc-200"
                    style={{ animationDelay: "0ms" }}
                  >
                    <span className="project-name-text pr-1" style={{ animationDelay: "0ms" }}>
                      {theme === "dark" ? "Dark mode" : "Light mode"}
                    </span>
                    <span className="project-name-text flex items-center" style={{ animationDelay: "500ms" }}>
                      <span style={{ transform: "scale(0.75)", transformOrigin: "center", display: "flex", alignItems: "center" }}>
                        <ThemeSwitch
                          checked={theme === "dark"}
                          onChange={onToggleTheme}
                          title={theme === "dark" ? "Switch to light mode" : "Switch to dark mode"}
                        />
                      </span>
                    </span>
                  </div>
                )}
              </div>
              {/* Drag handle to expand/collapse card */}
              {settingsMorphing && (
                <div className="title-drag-handle" onMouseDown={onSettingsDragStart}>
                  <svg width="42" height="14" viewBox="0 0 42 14" fill="none" xmlns="http://www.w3.org/2000/svg">
                    <rect x="0.5" y="1" width="41" height="4.5" rx="2.25" fill="currentColor" />
                    <rect x="11" y="8" width="20" height="4.5" rx="2.25" fill="currentColor" />
                  </svg>
                </div>
              )}
            </div>
          </div>
          </li>
        </ul>
      </div>

      {/* CONFIRM: open project? */}
      {confirmProject && (
        <div
          className={`title-confirm-backdrop${confirmLeaving ? " leaving" : ""}`}
          onClick={confirmNo}
        >
          <div className="bezier-card title-confirm-card" onClick={(e) => e.stopPropagation()}>
            <p className="title-confirm-message">open project?</p>
            <div className="title-confirm-actions">
              <button className="title-confirm-btn yes" onClick={confirmYes}>Yes</button>
              <button className="title-confirm-btn no" onClick={confirmNo}>No</button>
            </div>
          </div>
        </div>
      )}

      <div className="title-typewriter relative z-10 text-zinc-500 dark:text-zinc-400">
        <span style={{ position: "relative", display: "inline-block" }}>
          <span style={{ visibility: "hidden" }}>{FOOTER_TEXT}</span>
          <span style={{ position: "absolute", left: 0, top: 0, whiteSpace: "pre" }}>
            {FOOTER_TEXT.slice(0, footerTyped)}
          </span>
          <span
            className="title-typewriter-cursor"
            style={{ position: "absolute", left: `${footerTyped}ch`, top: 0 }}
          />
        </span>
        <p style={{ margin: 0 }}>
          Copyright 2026. MIT License.
        </p>
      </div>
      <a
        href="https://www.github.com/vieuxtiful/ToFU/"
        target="_blank"
        rel="noopener noreferrer"
        className="relative z-10 text-zinc-500 transition hover:text-zinc-700 dark:text-zinc-400 dark:hover:text-zinc-200"
      >
        <Github size={18} />
      </a>
      <span style={{ transform: "scale(0.75)", transformOrigin: "center", display: "inline-block", position: "relative", zIndex: 10 }}>
        <ThemeSwitch
          checked={theme === "dark"}
          onChange={onToggleTheme}
          title={theme === "dark" ? "Switch to light mode" : "Switch to dark mode"}
        />
      </span>
    </div>
  );
}
