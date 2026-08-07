import { useCallback, useEffect, useState } from "react";
import { BookmarkCheck, Loader2, Trash2, X } from "lucide-react";
import { Project, TMRecord, deleteMemoryRecord, getProjectMemory } from "./api";
import { langDisplayName } from "./languageData";

interface MemoryPanelProps {
  project: Project;
  onClose: () => void;
  leaving?: boolean;
}

function fmtTime(ts: number): string {
  return new Date(ts * 1000).toLocaleString([], {
    month: "short", day: "numeric", hour: "2-digit", minute: "2-digit",
  });
}

/** project-level translation-memory browser: every approved (source →
 * target) pair stored from a passing QA render, reusable across assets. */
export default function MemoryPanel({ project, onClose, leaving }: MemoryPanelProps) {
  const [records, setRecords] = useState<TMRecord[] | null>(null);
  const [workingOn, setWorkingOn] = useState<number | null>(null);
  const [error, setError] = useState<string | null>(null);

  const refresh = useCallback(() => {
    getProjectMemory(project.id)
      .then((r) => setRecords(r.records))
      .catch((e) => setError(String(e)));
  }, [project.id]);
  useEffect(refresh, [refresh]);

  const onDelete = async (r: TMRecord) => {
    if (!confirm(`Remove this translation-memory entry?\n\n"${r.source_text}" → "${r.target_text}"`)) return;
    setWorkingOn(r.id);
    setError(null);
    try {
      await deleteMemoryRecord(r.id);
      refresh();
    } catch (e) {
      setError(String(e));
    } finally {
      setWorkingOn(null);
    }
  };

  return (
    <div className={`fixed inset-0 z-400 flex items-center justify-center bg-black/60 p-6 pantry-overlay${leaving ? " leaving" : ""}`}>
      <div className={`bezier-card pantry-card${leaving ? " leaving" : ""} flex max-h-[85vh] w-full max-w-2xl flex-col rounded-xl bg-white dark:bg-zinc-900`}>
        <div className="flex items-center justify-between border-b border-zinc-200 px-5 py-4 dark:border-zinc-800">
          <h3 className="flex items-center gap-2 text-lg font-semibold text-zinc-800 dark:text-zinc-200">
            <BookmarkCheck size={18} className="text-cyan-600 dark:text-cyan-400" /> translation memory
            <span className="text-sm font-normal text-zinc-500">— {project.name}</span>
          </h3>
          <button onClick={onClose} className="rounded-sm p-1 text-zinc-500 hover:bg-zinc-100 hover:text-zinc-700 dark:hover:bg-zinc-800 dark:hover:text-zinc-300">
            <X size={16} />
          </button>
        </div>

        <div className="pantry-scroll flex-1 overflow-y-auto p-5">
          {error && (
            <div className="subtext mb-3 rounded-lg border border-red-300 bg-red-100 px-3 py-2 text-xs text-red-700 dark:border-red-800 dark:bg-red-950/50 dark:text-red-300">
              {error}
            </div>
          )}
          {records === null && (
            <div className="flex items-center gap-2 py-6 text-sm text-zinc-500">
              <Loader2 size={14} className="animate-spin" /> loading memory…
            </div>
          )}
          {records && records.length === 0 && (
            <p className="subtext py-6 text-center text-sm text-zinc-500">
              No approved translations stored yet. Every region rendered above
              the QA gate is remembered here, and offered as a suggestion the
              next time similar text (or the same sign) is captured.
            </p>
          )}
          {records && records.length > 0 && (
            <div className="space-y-2">
              {records.map((r) => (
                <div
                  key={r.id}
                  className="flex items-center gap-3 rounded-lg border border-zinc-200 bg-zinc-100 px-3 py-2 dark:border-zinc-800 dark:bg-zinc-800/40"
                >
                  {r.thumb_url ? (
                    <img
                      src={r.thumb_url}
                      alt=""
                      className="h-10 w-14 shrink-0 rounded-sm object-cover border border-zinc-300 dark:border-zinc-700"
                    />
                  ) : (
                    <div className="h-10 w-14 shrink-0 rounded-sm bg-zinc-200 dark:bg-zinc-800" />
                  )}
                  <div className="min-w-0 flex-1 text-xs">
                    <p className="truncate text-zinc-700 dark:text-zinc-300">
                      <span title={r.source_text}>{r.source_text}</span>
                      <span className="mx-1.5 text-zinc-400 dark:text-zinc-600">→</span>
                      <span className="text-emerald-600 dark:text-emerald-400" title={r.target_text}>{r.target_text}</span>
                    </p>
                    <p className="subtext flex flex-wrap items-center gap-1.5 text-zinc-500">
                      <span>{langDisplayName(r.source_lang ?? "?")} → {langDisplayName(r.target_lang)}</span>
                      <span>· QA {(r.qa_score * 100).toFixed(0)}%</span>
                      <span>· {fmtTime(r.created_at)}</span>
                      <span className="font-mono">· {r.asset_id}/{r.region_id}</span>
                    </p>
                  </div>
                  <button
                    onClick={() => onDelete(r)}
                    disabled={workingOn !== null}
                    title="Remove from memory"
                    className="shrink-0 rounded-sm p-1.5 text-zinc-400 hover:bg-zinc-200 hover:text-red-500 dark:text-zinc-600 dark:hover:bg-zinc-700 dark:hover:text-red-400 disabled:opacity-40"
                  >
                    {workingOn === r.id ? <Loader2 size={13} className="animate-spin" /> : <Trash2 size={13} />}
                  </button>
                </div>
              ))}
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
