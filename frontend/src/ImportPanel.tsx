import { useState } from "react";
import { CheckCircle2, Loader2, Upload, AlertTriangle, XCircle } from "lucide-react";
import { importFile, ImportResult } from "./api";

interface ImportPanelProps {
  assetId: string;
  onImported: (result: ImportResult) => void;
}

export default function ImportPanel({ assetId, onImported }: ImportPanelProps) {
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<ImportResult | null>(null);
  const [error, setError] = useState<string | null>(null);

  const onFile = async (file: File) => {
    setLoading(true);
    setError(null);
    setResult(null);
    try {
      const r = await importFile(assetId, file);
      setResult(r);
      onImported(r);
    } catch (e) {
      setError(String(e));
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="bezier-card space-y-3 rounded-lg bg-white/60 p-4 dark:bg-zinc-900/60">
      <h3 className="text-sm font-semibold text-zinc-700 dark:text-zinc-300">Import Translations</h3>
      <label className="flex h-24 cursor-pointer flex-col items-center justify-center gap-1 rounded-lg border-2 border-dashed border-zinc-400 text-zinc-500 transition hover:border-zinc-600 hover:text-zinc-700 dark:border-zinc-700 dark:hover:border-zinc-500 dark:hover:text-zinc-300">
        {loading ? (
          <Loader2 size={20} className="animate-spin" />
        ) : (
          <>
            <Upload size={18} />
            <span className="subtext text-xs">drop or click to upload .xliff, .tmx, .tsv, .csv, .txt</span>
          </>
        )}
        <input
          type="file"
          accept=".xliff,.xlf,.tmx,.tsv,.csv,.txt"
          className="hidden"
          onChange={(e) => e.target.files?.[0] && onFile(e.target.files[0])}
        />
      </label>

      {result && (
        <div className="space-y-2 text-xs">
          <div className="flex items-center gap-2 text-emerald-600 dark:text-emerald-400">
            <CheckCircle2 size={14} />
            <span>{result.imported} segments imported</span>
          </div>
          {result.missing.length > 0 && (
            <div className="flex items-center gap-2 text-amber-600 dark:text-amber-400">
              <AlertTriangle size={14} />
              <span>missing: {result.missing.join(", ")}</span>
            </div>
          )}
          {result.extra.length > 0 && (
            <div className="flex items-center gap-2 text-red-600 dark:text-red-400">
              <XCircle size={14} />
              <span>unknown IDs: {result.extra.join(", ")}</span>
            </div>
          )}
        </div>
      )}
      {error && (
        <div className="subtext text-xs text-red-600 dark:text-red-400">{error}</div>
      )}
    </div>
  );
}
