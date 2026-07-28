import { useEffect, useState } from "react";
import { Check, ChevronDown, Loader2, X } from "lucide-react";
import { RiBook2Fill, RiBook2Line, RiBookOpenFill, RiBookOpenLine } from "react-icons/ri";
import { TbReplace, TbReplaceFilled } from "react-icons/tb";
import { VscReplaceAll } from "react-icons/vsc";
import { PiPuzzlePieceBold } from "react-icons/pi";
import type { GlossaryStatus } from "./api";
import type { Theme } from "./theme";

type Props = {
  theme: Theme;
  projectId: string | null;
  status: GlossaryStatus | null;
  onUpload: (file: File, scope: "global" | "project", mode: "auxiliary" | "merge" | "replace") => void;
  onDelete: (scope: "global" | "project") => void;
  uploading: boolean;
  uploadStep: "marinate" | "ferment" | "set" | "done" | null;
  uploadError: string | null;
};

export default function GlossaryPanel({ theme, projectId, status, onUpload, onDelete, uploading, uploadStep, uploadError }: Props) {
  const [open, setOpen] = useState(false);
  const [selectedFileName, setSelectedFileName] = useState("");
  const [typedFileName, setTypedFileName] = useState(0);
  const [fileNameDone, setFileNameDone] = useState(true);
  const [scope, setScope] = useState<"global" | "project">(projectId ? "project" : "global");
  const [mode, setMode] = useState<"auxiliary" | "merge" | "replace">("auxiliary");
  const active = scope === "project" ? status?.project : status?.global;
  const Icon = theme === "dark" ? RiBook2Fill : RiBook2Line;
  const OpenBookIcon = theme === "dark" ? RiBookOpenFill : RiBookOpenLine;
  const stepLabel = uploadStep === "marinate" ? "Marinating" : uploadStep === "ferment" ? "Fermenting" : uploadStep === "set" ? "Setting" : uploadStep === "done" ? "Glossary set" : "";
  useEffect(() => {
    setTypedFileName(0);
    setFileNameDone(!selectedFileName);
    if (!selectedFileName) return;
    const timers: ReturnType<typeof setTimeout>[] = [];
    for (let index = 1; index <= selectedFileName.length; index++) {
      timers.push(setTimeout(() => setTypedFileName(index), 200 + index * 80));
    }
    timers.push(setTimeout(() => setFileNameDone(true), 200 + selectedFileName.length * 80 + 200));
    return () => timers.forEach(clearTimeout);
  }, [selectedFileName]);
  return (
    <div className="mt-3 rounded-lg border border-emerald-200/80 bg-emerald-50/50 p-2 dark:border-emerald-900/80 dark:bg-emerald-950/15">
      <button type="button" onClick={() => setOpen((value) => !value)} className="flex w-full items-center gap-2 text-left text-xs font-semibold text-emerald-900 dark:text-emerald-100">
        <Icon size={15} /> glossary {/* prev. "add glossary" */}
        {status && status.effective_entry_count > 0 && <span className="rounded-full bg-emerald-200 px-1.5 py-0.5 text-[10px] dark:bg-emerald-900">{status.effective_entry_count} terms</span>}
        <ChevronDown size={12} className={`ml-auto shrink-0 text-emerald-700 transition-transform dark:text-emerald-300 ${open ? "rotate-180" : ""}`} />
      </button>
      <div className={`dropdown-morph${open ? " expanded" : ""}`}>
        <div className="mt-2 space-y-2 text-xs">
          <div className="flex gap-1 px-1">
            {(["global", "project"] as const).map((value) => (
              <button key={value} type="button" aria-pressed={scope === value} disabled={value === "project" && !projectId} onClick={() => setScope(value)} className={`subtext inline-flex items-center rounded-md border-2 border-white/50 bg-white/10 px-[3px] py-0.5 text-[9px] text-zinc-500 backdrop-blur-md transition-all ${scope === value ? "shadow-[inset_0_1px_3px_rgba(255,255,255,0.45),0_0_0_2px_rgb(167,243,208)]" : "shadow-inner hover:bg-white/20"} disabled:cursor-not-allowed disabled:opacity-40`}>{value}</button>
            ))}
          </div>
          <div className="flex flex-wrap gap-1">
            {(["auxiliary", "merge", "replace"] as const).map((value) => {
              const selected = mode === value;
              const ModeIcon = value === "auxiliary" ? PiPuzzlePieceBold : value === "merge" ? VscReplaceAll : (selected ? TbReplaceFilled : TbReplace);
              return <button key={value} type="button" onClick={() => setMode(value)} className={`inline-flex items-center gap-1 rounded-sm px-2 py-1 ${selected ? "bg-emerald-700 text-white" : "bg-white text-zinc-600 dark:bg-zinc-800 dark:text-zinc-300"}`}><ModeIcon size={13} /> {value}</button>;
            })}
          </div>
          <div className="space-y-1">
            <div className="flex min-w-0">
              <label className="subtext inline-flex cursor-pointer items-center border border-emerald-700 bg-emerald-700 px-4 py-2 text-xs text-white transition hover:bg-emerald-600 active:bg-emerald-800 focus-within:outline-hidden focus-within:ring-3 focus-within:ring-emerald-300 disabled:opacity-25">
                click to browse
                <input className="hidden" type="file" accept=".csv,.tsv,.tab,.tbx,.utx,.xls,.xlsx,.txt,.xliff,.xlf" disabled={uploading} onChange={(event) => { const file = event.target.files?.[0]; if (file) { setSelectedFileName(file.name); onUpload(file, scope, mode); } event.currentTarget.value = ""; }} />
              </label>
              <div className="flex min-w-0 flex-1 items-center justify-between rounded-r-md border border-emerald-300 bg-white/70 text-zinc-600 dark:border-emerald-800 dark:bg-zinc-900/40 dark:text-zinc-300">
                <span className="subtext relative block min-w-0 flex-1 p-2 text-xs text-zinc-500 dark:text-zinc-400">
                  {selectedFileName ? <><span className="block truncate" style={{ visibility: "hidden" }}>{selectedFileName}</span><span className="absolute inset-x-2 top-2 overflow-hidden whitespace-nowrap">{selectedFileName.slice(0, typedFileName)}{!fileNameDone && <span className="name-typewriter-block" />}</span></> : <span className="text-[10px] text-zinc-400 dark:text-zinc-600">upload CSV, TSV, TAB, TBX, UTX, XLS/XLSX, TXT, or XLIFF</span>}
                </span>
                {selectedFileName && <button type="button" onClick={() => setSelectedFileName("")} title="clear selected file name" className="p-2 text-red-700 hover:text-red-900 dark:text-red-400 dark:hover:text-red-300"><X size={12} /></button>}
              </div>
            </div>
          </div>
          {uploading && <div className="flex items-center gap-1.5 text-emerald-700 dark:text-emerald-300"><Loader2 size={13} className="animate-spin" /> {stepLabel}</div>}
          {uploadStep === "done" && <div className="flex items-center gap-1.5 text-emerald-700 dark:text-emerald-300"><Check size={13} /> Glossary set</div>}
          {uploadError && <div className="text-rose-700 dark:text-rose-300">{uploadError}</div>}
          {active && <div className="flex items-center justify-between rounded-sm bg-white/70 px-2 py-1.5 dark:bg-zinc-900/50"><span className="flex items-center gap-1"><OpenBookIcon size={13} /> {active.entry_count} terms · {active.mode}</span><button type="button" title="Clear glossary" onClick={() => onDelete(scope)} className="rounded-sm p-1 text-zinc-500 hover:bg-zinc-200 dark:hover:bg-zinc-800"><X size={13} /></button></div>}
        </div>
      </div>
    </div>
  );
}
