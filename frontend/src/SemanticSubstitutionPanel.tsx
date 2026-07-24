import { useState } from "react";
import { Check, Loader2, ScanText, ShieldAlert } from "lucide-react";
import { FcCollapse } from "react-icons/fc";
import { MdTipsAndUpdates } from "react-icons/md";
import { TbLeafFilled, TbReplace, TbReplaceFilled } from "react-icons/tb";
import type { GlossaryStatus, SemanticSubstitutionPlan, SemanticTextUnit } from "./api";
import type { Theme } from "./theme";
import GlossaryPanel from "./GlossaryPanel";
import basilHeaderUrl from "../../images/basil-header.png";

type Props = {
  units: SemanticTextUnit[];
  drafts: Record<string, string>;
  plans: Record<string, SemanticSubstitutionPlan>;
  busyId: string | null;
  onDraftChange: (unitId: string, text: string) => void;
  onPlan: (unitId: string, apply: boolean) => void;
  theme: Theme;
  projectId: string | null;
  glossaryStatus: GlossaryStatus | null;
  onGlossaryUpload: (file: File, scope: "global" | "project", mode: "auxiliary" | "merge" | "replace") => void;
  onGlossaryDelete: (scope: "global" | "project") => void;
  glossaryUploading: boolean;
  glossaryUploadStep: "marinate" | "ferment" | "set" | "done" | null;
  glossaryUploadError: string | null;
};

/**
 * Makes the semantic and spatial orders visible at the same time.  The
 * explicit Apply button is a safety boundary: entering a natural target
 * phrase never silently replaces the per-region target fields.
 */
export default function SemanticSubstitutionPanel({
  units, drafts, plans, busyId, onDraftChange, onPlan, theme, projectId, glossaryStatus,
  onGlossaryUpload, onGlossaryDelete, glossaryUploading, glossaryUploadStep, glossaryUploadError,
}: Props) {
  const multiRegion = units.filter((unit) => unit.region_ids.length > 1);
  const [open, setOpen] = useState(false);
  const backdropFade = theme === "dark" ? "rgba(8, 47, 73, 1)" : "rgba(236, 254, 255, 1)";

  if (multiRegion.length === 0) return null;
  return (
    <section className="relative isolate rounded-xl border border-cyan-200/80 bg-cyan-50/55 p-3 dark:border-cyan-900/80 dark:bg-cyan-950/20">
      <div
        aria-hidden="true"
        className={`pointer-events-none absolute inset-0 rounded-[inherit] bg-no-repeat transition-opacity duration-300 ${open ? "opacity-100" : "opacity-55"}`}
        style={{ backgroundImage: `linear-gradient(to bottom, transparent 0%, transparent 80%, ${backdropFade} 100%), url(${basilHeaderUrl})`, backgroundPosition: "center, center -32px", backgroundSize: "100% 100%, 100% auto" }}
      />
      <div className="relative z-10">
      <div className="flex items-center gap-2 pr-7 text-sm font-semibold text-cyan-950 dark:text-cyan-100">
        <TbLeafFilled size={15} className="text-emerald-600" />
        <span>Basil</span>
      </div>
      <button type="button" onClick={() => setOpen((value) => !value)} className="absolute right-0 top-0 rounded p-1 text-zinc-500 transition hover:bg-zinc-200 dark:hover:bg-zinc-800" title={open ? "collapse Basil" : "expand Basil"} aria-label={open ? "collapse Basil" : "expand Basil"}>
        <FcCollapse size={12} style={{ transform: open ? "none" : "rotate(180deg)", transition: "transform 0.2s" }} />
      </button>
      <p className="mb-1 flex items-start gap-1.5 text-xs text-cyan-900/75 dark:text-cyan-200/70">
        <MdTipsAndUpdates size={14} className="mt-0.5 shrink-0 text-[#2d8cf0]" />
        <span>Set target arrangement and review before plating.</span>
      </p>
      <div className={`style-panel-morph${open ? " expanded" : ""}`}>
      <div>
      <GlossaryPanel theme={theme} projectId={projectId} status={glossaryStatus} onUpload={onGlossaryUpload} onDelete={onGlossaryDelete} uploading={glossaryUploading} uploadStep={glossaryUploadStep} uploadError={glossaryUploadError} />
      <div className="mt-3 space-y-3">
        {multiRegion.map((unit) => {
          const plan = plans[unit.id];
          const value = drafts[unit.id] ?? unit.substitution?.target_text ?? "";
          const loading = busyId === unit.id;
          return (
            <div key={unit.id} className="rounded-lg border border-cyan-200 bg-white/80 p-3 dark:border-cyan-900 dark:bg-zinc-950/45">
              <div className="flex flex-wrap items-center gap-x-2 gap-y-1 text-xs">
                <span className="rounded bg-cyan-100 px-1.5 py-0.5 font-mono text-cyan-800 dark:bg-cyan-900/70 dark:text-cyan-100">
                  {unit.region_ids.join(" → ")}
                </span>
                <span className="font-medium text-zinc-800 dark:text-zinc-100">{unit.source_text}</span>
                <span className="flex items-center gap-1.5 text-[10px] text-zinc-500 dark:text-zinc-400">
                  <ScanText size={12} className="shrink-0 text-cyan-600 dark:text-cyan-400" />
                  <span className="font-medium text-zinc-700 dark:text-zinc-300">{unit.entity_type.replace("_", " ")}</span>
                  <span className={unit.confidence >= 0.8 ? "text-emerald-600 dark:text-emerald-300" : unit.confidence >= 0.6 ? "text-amber-600 dark:text-amber-300" : "text-red-500 dark:text-red-400"}>
                    {Math.round(unit.confidence * 100)}%
                  </span>
                </span>
              </div>
              <div className="mt-2 flex flex-col gap-2 sm:flex-row">
                <input
                  aria-label={`Target phrase for ${unit.source_text}`}
                  value={value}
                  onChange={(event) => onDraftChange(unit.id, event.target.value)}
                  placeholder="Complete target phrase"
                  className="min-w-0 flex-1 rounded-md border border-zinc-300 bg-white px-2 py-1.5 text-sm text-zinc-900 outline-none focus:border-cyan-500 focus:ring-1 focus:ring-cyan-500 dark:border-zinc-700 dark:bg-zinc-900 dark:text-zinc-100"
                />
                <button
                  type="button"
                  onClick={() => onPlan(unit.id, false)}
                  disabled={loading || !value.trim()}
                  className="inline-flex items-center justify-center gap-1 rounded-md bg-cyan-700 px-2.5 py-1.5 text-xs font-medium text-white transition hover:bg-cyan-800 disabled:cursor-not-allowed disabled:opacity-45"
                >
                  {loading ? <Loader2 size={13} className="animate-spin" /> : theme === "dark" ? <TbReplaceFilled size={14} /> : <TbReplace size={14} />}
                  arrange
                </button>
              </div>
              {plan && (
                <div className="mt-2 rounded-md bg-zinc-100/80 p-2 text-xs dark:bg-zinc-900/80">
                  {plan.assignments.length > 0 ? (
                    <>
                      <div className="flex flex-wrap gap-1.5 text-zinc-800 dark:text-zinc-100">
                        {plan.assignments.map((assignment) => (
                          <span key={assignment.region_id} className="rounded bg-white px-1.5 py-0.5 shadow-sm dark:bg-zinc-800">
                            <span className="font-mono text-cyan-700 dark:text-cyan-300">{assignment.region_id}{assignment.anchor_id && assignment.anchor_id !== assignment.region_id ? ` → ${assignment.anchor_id}` : ""}</span> {assignment.text}
                          </span>
                        ))}
                      </div>
                      <p className="mt-1 text-zinc-500 dark:text-zinc-400">
                        blocks: <span className="font-mono">{plan.target_region_order.join(" → ")}</span>
                        <span className="mx-1">·</span>
                        cubes: <span className="font-mono">{plan.source_region_order.join(" → ")}</span>
                      </p>
                      <button
                        type="button"
                        onClick={() => onPlan(unit.id, true)}
                        disabled={loading || plan.review_required}
                        className="mt-2 inline-flex items-center gap-1 rounded-md bg-emerald-700 px-2.5 py-1.5 text-xs font-medium text-white transition hover:bg-emerald-800 disabled:cursor-not-allowed disabled:opacity-45"
                      >
                        {loading ? <Loader2 size={13} className="animate-spin" /> : <Check size={13} />}
                        Plate cubes
                      </button>
                    </>
                  ) : (
                    <div className="flex gap-1.5 text-amber-700 dark:text-amber-300">
                      <ShieldAlert size={14} className="mt-0.5 shrink-0" />
                      <span>{plan.warnings.join(" ") || "This phrase needs manual placement."}</span>
                    </div>
                  )}
                </div>
              )}
            </div>
          );
        })}
      </div>
      </div>
      </div>
      </div>
    </section>
  );
}
