import { useEffect, useState } from "react";
import { MdTipsAndUpdates } from "react-icons/md";
import { getDetectionAttribution, type DetectionAttribution } from "./api";

/** What an empty Capture tab should say.
 *
 * It used to say one thing — "no text regions detected. you can draw them
 * manually." — for six different events. The two that matter most read
 * identically under that sentence and cost completely different amounts to
 * fix:
 *
 *   the detector proposed nothing        → tiles, another detector, or
 *                                          fine-tuning. Expensive.
 *   it proposed plenty and all were      → score calibration or a targeted
 *   suppressed                             retry. Cheap.
 *
 * So the card names the rung and shows where the candidates died. Nothing
 * here re-runs detection: it reports the graph the shipping run recorded.
 */
function humanStage(stage: string): string {
  return stage.replace(/_/g, " ");
}

export default function DetectionAttributionCard({ assetId }: { assetId: string }) {
  const [report, setReport] = useState<DetectionAttribution | null>(null);

  useEffect(() => {
    let cancelled = false;
    getDetectionAttribution(assetId)
      .then((result) => { if (!cancelled) setReport(result); })
      .catch(() => { /* an empty tab is not worth an error toast */ });
    return () => { cancelled = true; };
  }, [assetId]);

  if (!report || report.rung === "regions_present") return null;

  const lineage = report.lineage;
  const suppressed = lineage
    ? Object.entries(lineage.suppressed_by_stage)
        .map(([stage, states]) => ({
          stage,
          count: Object.values(states).reduce((sum, n) => sum + n, 0),
        }))
        .sort((a, b) => b.count - a.count)
    : [];

  return (
    <div
      data-testid="detection-attribution"
      data-rung={report.rung}
      className="subtext rounded-lg border border-zinc-300 bg-white/60 px-3 py-2 text-xs text-zinc-600 dark:border-zinc-700 dark:bg-zinc-900/60 dark:text-zinc-400"
    >
      <p className="flex items-start gap-1.5">
        <MdTipsAndUpdates size={14} className="mt-0.5 shrink-0 text-[#2d8cf0]" />
        <span className="flex-1">{report.explanation}.</span>
      </p>
      {lineage && lineage.raw_proposals > 0 && (
        <p className="mt-1.5 pl-5">
          {lineage.raw_proposals} detector proposal
          {lineage.raw_proposals === 1 ? "" : "s"}
          {suppressed.length > 0 && (
            <>
              {" — suppressed at "}
              {suppressed.slice(0, 3).map((entry, index) => (
                <span key={entry.stage}>
                  {index > 0 && ", "}
                  <span className="text-zinc-700 dark:text-zinc-300">
                    {humanStage(entry.stage)}
                  </span>
                  {" ("}{entry.count}{")"}
                </span>
              ))}
            </>
          )}
          .
        </p>
      )}
      {report.rung === "all_excluded" && (
        <p className="mt-1.5 pl-5">
          {report.regions.excluded} region
          {report.regions.excluded === 1 ? " was" : "s were"} captured and then
          excluded — restore from History if that was not intended.
        </p>
      )}
      {report.rung === "no_lineage" && (
        <p className="mt-1.5 pl-5 opacity-80">
          {/* Absence of evidence, said as such. Calling this "the detector
              proposed nothing" would be inventing a finding. */}
          This asset predates lineage recording, so there is nothing to
          inspect. Recapture to record it.
        </p>
      )}
    </div>
  );
}
