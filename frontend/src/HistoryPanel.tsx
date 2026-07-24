import { useCallback, useEffect, useState } from "react";
import { Clock, History, Loader2, RotateCcw, Trash2, X } from "lucide-react";
import {
  Project, ProjectHistory, SnapshotMeta, deleteSnapshot, getProjectHistory, restoreSnapshot,
} from "./api";

interface HistoryPanelProps {
  project: Project;
  currentAssetId: string | null;
  onRestored: () => void;
  onClose: () => void;
  leaving?: boolean;
}

function fmtTime(ts: number): string {
  return new Date(ts * 1000).toLocaleString([], {
    month: "short", day: "numeric", hour: "2-digit", minute: "2-digit", second: "2-digit",
  });
}

const REASON_STYLE: Record<string, string> = {
  autosave: "bg-zinc-300/60 text-zinc-600 dark:bg-zinc-700/60 dark:text-zinc-300",
  "pre-erase": "bg-amber-500/15 text-amber-600 dark:text-amber-400",
  import: "bg-cyan-500/15 text-cyan-700 dark:text-cyan-400",
  manual: "bg-emerald-500/15 text-emerald-600 dark:text-emerald-400",
  "restore-backup": "bg-purple-500/15 text-purple-600 dark:text-purple-300",
};

const PROTECTED_REASONS = new Set(["pre-erase", "import", "manual", "restore-backup"]);

const EVENT_LABEL: Record<string, string> = {
  "project-created": "project created",
  "asset-uploaded": "asset uploaded",
  "asset-removed": "asset removed",
  detection: "text detection",
  "language-scan": "language scan",
  import: "translations imported",
  render: "render",
  "qa-approved": "QA approved",
  snapshot: "snapshot",
  "snapshot-deleted": "snapshot deleted",
  restore: "restore",
  "session-loaded": "session loaded",
};

/** Session history: every autosave/import/pre-erase snapshot is listed,
 * restorable, and deletable, alongside a timeline of project events. */
export default function HistoryPanel({ project, currentAssetId, onRestored, onClose, leaving }: HistoryPanelProps) {
  const [history, setHistory] = useState<ProjectHistory | null>(null);
  const [tab, setTab] = useState<"snapshots" | "events">("snapshots");
  const [workingOn, setWorkingOn] = useState<number | null>(null);
  const [error, setError] = useState<string | null>(null);

  const refresh = useCallback(() => {
    getProjectHistory(project.id)
      .then(setHistory)
      .catch((e) => setError(String(e)));
  }, [project.id]);
  useEffect(refresh, [refresh]);

  const onRestore = async (snap: SnapshotMeta) => {
    const ok = confirm(
      `Restore the ${snap.reason} snapshot from ${fmtTime(snap.created_at)} ` +
      `(${snap.region_count} regions, ${snap.translated} translated)?\n\n` +
      "Your current state is snapshotted first, so nothing is lost."
    );
    if (!ok) return;
    setWorkingOn(snap.id);
    setError(null);
    try {
      await restoreSnapshot(snap.id);
      onRestored();
    } catch (e) {
      setError(String(e));
    } finally {
      setWorkingOn(null);
    }
  };

  const onDelete = async (snap: SnapshotMeta) => {
    if (!confirm(`Delete the ${snap.reason} snapshot from ${fmtTime(snap.created_at)}?`)) return;
    if (
      PROTECTED_REASONS.has(snap.reason) &&
      !confirm(`"${snap.reason}" snapshots are protection points. Really delete it?`)
    ) return;
    setWorkingOn(snap.id);
    setError(null);
    try {
      await deleteSnapshot(snap.id);
      refresh();
    } catch (e) {
      setError(String(e));
    } finally {
      setWorkingOn(null);
    }
  };

  return (
    <div className={`fixed inset-0 z-50 flex items-center justify-center bg-black/60 p-6 pantry-overlay${leaving ? " leaving" : ""}`}>
      <div className={`bezier-card pantry-card${leaving ? " leaving" : ""} flex max-h-[85vh] w-full max-w-2xl flex-col rounded-xl bg-white dark:bg-zinc-900`}>
        <div className="flex items-center justify-between border-b border-zinc-200 px-5 py-4 dark:border-zinc-800">
          <h3 className="flex items-center gap-2 text-lg font-semibold text-zinc-800 dark:text-zinc-200">
            <History size={18} className="text-cyan-600 dark:text-cyan-400" /> Session History
            <span className="text-sm font-normal text-zinc-500">— {project.name}</span>
          </h3>
          <button onClick={onClose} className="rounded-sm p-1 text-zinc-500 hover:bg-zinc-100 hover:text-zinc-700 dark:hover:bg-zinc-800 dark:hover:text-zinc-300">
            <X size={16} />
          </button>
        </div>

        <div className="flex gap-1 border-b border-zinc-200 px-5 pt-3 dark:border-zinc-800">
          {(["snapshots", "events"] as const).map((t) => (
            <button
              key={t}
              onClick={() => setTab(t)}
              className={`rounded-t-lg px-4 py-2 text-xs font-medium transition ${
                tab === t
                  ? "bg-zinc-100 text-cyan-700 dark:bg-zinc-800 dark:text-cyan-300"
                  : "text-zinc-500 hover:text-zinc-700 dark:hover:text-zinc-300"
              }`}
            >
              {t === "snapshots"
                ? `Saves (${history?.snapshots.length ?? "…"})`
                : `Activity (${history?.events.length ?? "…"})`}
            </button>
          ))}
        </div>

        <div className="flex-1 overflow-y-auto p-5">
          {error && (
            <div className="subtext mb-3 rounded-lg border border-red-300 bg-red-100 px-3 py-2 text-xs text-red-700 dark:border-red-800 dark:bg-red-950/50 dark:text-red-300">
              {error}
            </div>
          )}
          {history === null && (
            <div className="flex items-center gap-2 py-6 text-sm text-zinc-500">
              <Loader2 size={14} className="animate-spin" /> loading history…
            </div>
          )}

          {history && tab === "snapshots" && (
            <div className="space-y-2">
              {history.snapshots.length === 0 && (
                <p className="subtext py-6 text-center text-sm text-zinc-500">
                  No saves yet. Autosave records a snapshot whenever your regions change.
                </p>
              )}
              {history.snapshots.map((s) => (
                <div
                  key={s.id}
                  className="flex items-center gap-3 rounded-lg border border-zinc-200 bg-zinc-100 px-3 py-2 dark:border-zinc-800 dark:bg-zinc-800/40"
                >
                  <span className={`subtext shrink-0 rounded-full px-2 py-0.5 text-[10px] font-medium ${REASON_STYLE[s.reason] ?? REASON_STYLE.autosave}`}>
                    {s.reason}
                  </span>
                  <div className="min-w-0 flex-1 text-xs">
                    <p className="text-zinc-700 dark:text-zinc-300">
                      {s.region_count} region(s) · <span className="text-emerald-600 dark:text-emerald-400">{s.translated} translated</span>
                    </p>
                    <p className="subtext text-zinc-500">
                      {fmtTime(s.created_at)} · asset <span className="font-mono">{s.asset_id}</span>
                      {s.asset_id === currentAssetId && <span className="ml-1 text-cyan-600 dark:text-cyan-500">(current)</span>}
                    </p>
                  </div>
                  <button
                    onClick={() => onRestore(s)}
                    disabled={workingOn !== null}
                    className="flex shrink-0 items-center gap-1.5 rounded-lg bg-zinc-200 px-3 py-1.5 text-xs font-medium text-zinc-700 transition hover:bg-cyan-600 hover:text-white dark:bg-zinc-700 dark:text-zinc-200 dark:hover:bg-cyan-700 disabled:opacity-40"
                  >
                    {workingOn === s.id ? <Loader2 size={12} className="animate-spin" /> : <RotateCcw size={12} />}
                    Restore
                  </button>
                  <button
                    onClick={() => onDelete(s)}
                    disabled={workingOn !== null}
                    title="Delete snapshot"
                    className="shrink-0 rounded-sm p-1.5 text-zinc-400 hover:bg-zinc-200 hover:text-red-500 dark:text-zinc-600 dark:hover:bg-zinc-700 dark:hover:text-red-400 disabled:opacity-40"
                  >
                    <Trash2 size={13} />
                  </button>
                </div>
              ))}
            </div>
          )}

          {history && tab === "events" && (
            <div className="space-y-1.5">
              {history.events.length === 0 && (
                <p className="subtext py-6 text-center text-sm text-zinc-500">No activity recorded yet.</p>
              )}
              {history.events.map((e) => (
                <div key={e.id} className="flex items-start gap-2 rounded-sm px-2 py-1.5 text-xs hover:bg-zinc-100 dark:hover:bg-zinc-800/40">
                  <Clock size={12} className="mt-0.5 shrink-0 text-zinc-400 dark:text-zinc-600" />
                  <div className="min-w-0">
                    <span className="font-medium text-zinc-700 dark:text-zinc-300">{EVENT_LABEL[e.kind] ?? e.kind}</span>
                    {e.detail && <span className="subtext text-zinc-500"> — {e.detail}</span>}
                    <span className="subtext ml-2 text-zinc-400 dark:text-zinc-600">{fmtTime(e.created_at)}</span>
                  </div>
                </div>
              ))}
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
