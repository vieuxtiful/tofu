import { useEffect, useRef, useState } from "react";
import { PiBrainFill, PiBrainBold } from "react-icons/pi";
import { MdTipsAndUpdates } from "react-icons/md";
import { SquareLoader } from "./Loaders";
import { providerLabel } from "./repairLabels";
import type { LocalizedCandidatePreview, RepairCandidate, RepairReview } from "./localizedCanvasTypes";

type Props = {
  reviews: RepairReview[];
  fallbackIds: string[];
  previews: Record<string, LocalizedCandidatePreview>;
  previewPending: boolean;
  appliedIds: string[];
  onRetryPreview: () => void;
  onApply: (candidate: RepairCandidate) => void;
  onApplyAll: () => void;
  onHoverRegion?: (id: string | null) => void;
};

/** Candidate choice remains explicit: no repair is committed without an editor action. */
export default function SmartFillReview({ reviews, fallbackIds, previews, previewPending, appliedIds, onRetryPreview, onApply, onApplyAll, onHoverRegion }: Props) {
  const [open, setOpen] = useState(false);
  const panelRef = useRef<HTMLDivElement>(null);
  const candidates = Array.from(new Map(
    reviews.flatMap((review) => review.candidates.map((candidate) => [candidate.id, { review, candidate }]))
  ).values());
  const readyCandidates = candidates.filter(({ candidate }) => previews[candidate.id]?.status === "ready" && !appliedIds.includes(candidate.id) && !previewPending);
  useEffect(() => {
    if (!open) return;
    const onPointerDown = (event: MouseEvent) => {
      if (panelRef.current && !panelRef.current.contains(event.target as Node)) setOpen(false);
    };
    document.addEventListener("mousedown", onPointerDown);
    return () => document.removeEventListener("mousedown", onPointerDown);
  }, [open]);

  if (!reviews.length && !fallbackIds.length) return null;

  return <div ref={panelRef} className="mt-2 rounded-sm border border-amber-500/40 bg-amber-50/70 px-2 py-2 text-xs text-amber-900 dark:bg-amber-950/20 dark:text-amber-100" onMouseLeave={() => onHoverRegion?.(null)}>
    <div className="flex items-center gap-2">
      <button type="button" onClick={() => { onHoverRegion?.(null); setOpen((value) => !value); }} className="flex w-full items-center gap-1 rounded-sm px-1 py-0.5 text-left font-semibold transition hover:bg-amber-100/70 dark:hover:bg-amber-950/40" aria-expanded={open}>
        <PiBrainBold size={14} className="dark:hidden" /><PiBrainFill size={14} className="hidden dark:block" /> Smart Fill
      </button>
    </div>

    <div className={`smart-fill-content${open ? " expanded" : ""}`}>
      <div className="min-h-0">
      <div className="mt-1.5 flex items-center justify-between gap-2">
        <p className="bezier-impression subtext flex items-start gap-1 text-[10px] text-amber-700/80 dark:text-amber-200/70">
          <MdTipsAndUpdates size={12} className="mt-0.5 shrink-0 text-amber-600" /><span>Smart fill inpaints the region shown. Hover to preview.</span> {/* "to improve its surface" was removed after "shown" */}
        </p>
        <button type="button" onClick={onApplyAll} disabled={readyCandidates.length === 0} className="shrink-0 rounded-sm bg-amber-700 px-1.5 py-0.5 text-[10px] text-white disabled:cursor-not-allowed disabled:opacity-50">Apply All</button>
      </div>
      {fallbackIds.length > 0 && <div className="mb-1 mt-1.5 rounded-sm border border-sky-500/30 bg-sky-50/50 px-2 py-1 text-[10px] text-sky-900 dark:bg-sky-950/20 dark:text-sky-100"><span className="font-medium">Texture treatment is ready for review.</span><span className="ml-1">{fallbackIds.length} region{fallbackIds.length === 1 ? " uses" : "s use"} the editable reconstruction base.</span></div>}
      {candidates.length > 0 && <div className="mt-1.5 flex flex-wrap justify-center gap-2">
        {candidates.map(({ review, candidate }) => {
          const preview = previews[candidate.id];
          const ready = preview?.status === "ready";
          const loading = preview?.status === "loading";
          const failed = preview?.status === "error";
          const applied = appliedIds.includes(candidate.id);
          return <div key={candidate.id} className="flex w-20 flex-col items-center gap-1 rounded-sm border border-amber-500/40 bg-white/80 p-1 dark:bg-zinc-900/70" onMouseEnter={() => onHoverRegion?.(review.id)} onMouseLeave={() => onHoverRegion?.(null)}>
            {ready && <img src={preview.url} alt={`localized suggested ${providerLabel(candidate.provider).toLowerCase()} for ${review.id}`} className="h-12 w-16 rounded-sm object-contain" />}
            {loading && <div className="flex h-12 w-16 items-center justify-center rounded-sm bg-amber-100/70 text-amber-700 dark:bg-amber-950/40"><SquareLoader size="xs" /></div>}
            {failed && <img src={candidate.url} alt={`repair-only background suggestion for ${review.id}`} className="h-12 w-16 rounded-sm object-contain opacity-75" />}
            {!preview && <div className="h-12 w-16 rounded-sm bg-zinc-100 dark:bg-zinc-800" />}
            <span className="text-center text-[8px] leading-3 text-amber-800/75 dark:text-amber-200/75">{ready ? "localized preview" : loading ? "building preview" : failed ? "repair-only background" : "preview pending"}</span>
            <span className="text-center text-[8px] text-amber-700/70 dark:text-amber-200/70">{providerLabel(candidate.provider)}</span>
            {failed ? <button onClick={onRetryPreview} className="rounded-sm bg-amber-700 px-1.5 py-0.5 text-[10px] text-white">retry</button> :
              <button disabled={!ready || previewPending || applied} onClick={() => onApply(candidate)} className="rounded-sm bg-amber-700 px-1.5 py-0.5 text-[10px] text-white disabled:cursor-not-allowed disabled:opacity-50">{applied ? "Applied" : previewPending ? "Updating…" : ready ? "apply" : "waiting…"}</button>}
          </div>;
        })}
      </div>}
      </div>
    </div>
  </div>;
}
