import { useState } from "react";
import { Check, ChevronDown, Loader2, X } from "lucide-react";
import { RiBook2Fill, RiBook2Line } from "react-icons/ri";
import { TbLeaf, TbLeafFilled, TbReplace, TbReplaceFilled } from "react-icons/tb";
import { VscReplaceAll } from "react-icons/vsc";
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
  const [scope, setScope] = useState<"global" | "project">(projectId ? "project" : "global");
  const [mode, setMode] = useState<"auxiliary" | "merge" | "replace">("auxiliary");
  const active = scope === "project" ? status?.project : status?.global;
  const Icon = theme === "dark" ? RiBook2Fill : RiBook2Line;
  const stepLabel = uploadStep === "marinate" ? "Marinating" : uploadStep === "ferment" ? "Fermenting" : uploadStep === "set" ? "Setting" : uploadStep === "done" ? "Glossary set" : "";
  return (
    <div className="mt-3 rounded-lg border border-emerald-200/80 bg-emerald-50/50 p-2 dark:border-emerald-900/80 dark:bg-emerald-950/15">
      <button type="button" onClick={() => setOpen((value) => !value)} className="flex w-full items-center gap-2 text-left text-xs font-semibold text-emerald-900 dark:text-emerald-100">
        <Icon size={15} /> add glossary
        {status && status.effective_entry_count > 0 && <span className="rounded-full bg-emerald-200 px-1.5 py-0.5 text-[10px] dark:bg-emerald-900">{status.effective_entry_count} terms</span>}
        <ChevronDown size={12} className={`ml-auto shrink-0 text-emerald-700 transition-transform dark:text-emerald-300 ${open ? "rotate-180" : ""}`} />
      </button>
      <div className={`dropdown-morph${open ? " expanded" : ""}`}>
        <div className="mt-2 space-y-2 text-xs">
          <div className="flex gap-1">
            {(["global", "project"] as const).map((value) => (
              <button key={value} type="button" disabled={value === "project" && !projectId} onClick={() => setScope(value)} className={`rounded px-2 py-1 ${scope === value ? "bg-emerald-700 text-white" : "bg-white text-zinc-600 dark:bg-zinc-800 dark:text-zinc-300"} disabled:cursor-not-allowed disabled:opacity-40`}>{value}</button>
            ))}
          </div>
          <div className="flex flex-wrap gap-1">
            {(["auxiliary", "merge", "replace"] as const).map((value) => {
              const selected = mode === value;
              const ModeIcon = value === "auxiliary" ? (selected ? TbLeafFilled : TbLeaf) : value === "merge" ? VscReplaceAll : (selected ? TbReplaceFilled : TbReplace);
              return <button key={value} type="button" onClick={() => setMode(value)} className={`inline-flex items-center gap-1 rounded px-2 py-1 ${selected ? "bg-emerald-700 text-white" : "bg-white text-zinc-600 dark:bg-zinc-800 dark:text-zinc-300"}`}><ModeIcon size={13} /> {value}</button>;
            })}
          </div>
          <label className="flex cursor-pointer items-center justify-between rounded border border-dashed border-emerald-300 bg-white/70 px-2 py-2 dark:border-emerald-800 dark:bg-zinc-900/40">
            <span>Upload CSV, TSV, TAB, TBX, UTX, XLS/XLSX, TXT, or XLIFF</span>
            <input className="hidden" type="file" accept=".csv,.tsv,.tab,.tbx,.utx,.xls,.xlsx,.txt,.xliff,.xlf" disabled={uploading} onChange={(event) => { const file = event.target.files?.[0]; if (file) onUpload(file, scope, mode); event.currentTarget.value = ""; }} />
            <span className="rounded bg-emerald-700 px-2 py-1 font-medium text-white">Choose</span>
          </label>
          {uploading && <div className="flex items-center gap-1.5 text-emerald-700 dark:text-emerald-300"><Loader2 size={13} className="animate-spin" /> {stepLabel}</div>}
          {uploadStep === "done" && <div className="flex items-center gap-1.5 text-emerald-700 dark:text-emerald-300"><Check size={13} /> Glossary set</div>}
          {uploadError && <div className="text-rose-700 dark:text-rose-300">{uploadError}</div>}
          {active && <div className="flex items-center justify-between rounded bg-white/70 px-2 py-1.5 dark:bg-zinc-900/50"><span>📖 {active.entry_count} terms · {active.mode}</span><button type="button" title="Clear glossary" onClick={() => onDelete(scope)} className="rounded p-1 text-zinc-500 hover:bg-zinc-200 dark:hover:bg-zinc-800"><X size={13} /></button></div>}
        </div>
      </div>
    </div>
  );
}
