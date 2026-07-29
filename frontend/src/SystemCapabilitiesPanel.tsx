import { useEffect, useState } from "react";
import { RefreshCw, X } from "lucide-react";
import { CapabilityStatus, SystemCapabilities, fetchSystemCapabilities } from "./api";

export default function SystemCapabilitiesPanel({ onClose }: { onClose: () => void }) {
  const [data, setData] = useState<SystemCapabilities | null>(null);
  const [error, setError] = useState<string | null>(null);
  const refresh = () => {
    setError(null);
    fetchSystemCapabilities().then(setData).catch((err) => setError(String(err)));
  };
  useEffect(refresh, []);
  const rows: Array<[string, CapabilityStatus]> = data ? [
    ["GPU", data.compute.gpu], ["Fonts", data.fonts], ["Video", data.video],
    ...data.ocr.providers.map((item) => [`OCR · ${item.id}`, item] as [string, CapabilityStatus]),
    ...data.scene.providers.map((item) => [`Scene · ${item.id}`, item] as [string, CapabilityStatus]),
    ...data.shaping.providers.map((item) => [`Shaping · ${item.id}`, item] as [string, CapabilityStatus]),
    ...data.inpainting.providers.map((item) => [`Repair · ${item.id}`, item] as [string, CapabilityStatus]),
    ...data.translation.providers.map((item) => [`MT · ${item.id}`, item] as [string, CapabilityStatus]),
  ] : [];
  return <div className="fixed inset-0 z-500 flex items-center justify-center bg-black/60 p-4" role="dialog" aria-modal="true" aria-label="System capabilities">
    <section className="w-full max-w-2xl rounded-xl bg-white p-5 shadow-2xl dark:bg-zinc-900">
      <header className="mb-4 flex items-center justify-between"><div><h2 className="font-semibold">System capabilities</h2><p className="text-xs text-zinc-500">Runtime status; no models are downloaded or probed.</p></div><div className="flex gap-1"><button onClick={refresh} title="Refresh" className="rounded p-2 hover:bg-zinc-100 dark:hover:bg-zinc-800"><RefreshCw size={16} /></button><button onClick={onClose} className="rounded p-2 hover:bg-zinc-100 dark:hover:bg-zinc-800"><X size={16} /></button></div></header>
      {error && <p className="rounded bg-red-100 p-2 text-xs text-red-700">{error}</p>}
      {!data && !error && <p className="text-sm text-zinc-500">Loading capabilities…</p>}
      {data && <div className="max-h-[65vh] space-y-1 overflow-y-auto">{rows.map(([label, item]) => <div key={label} className="flex items-start justify-between gap-4 rounded border border-zinc-200 p-2 text-xs dark:border-zinc-700"><div><span className="font-medium">{label}</span>{item.version && <span className="ml-2 text-zinc-500">{item.version}</span>}<p className="mt-1 text-zinc-500">{item.reason ?? (item.ready ? "ready" : "available but not configured")}</p></div><span className={`rounded px-2 py-0.5 ${item.ready ? "bg-emerald-100 text-emerald-700" : "bg-amber-100 text-amber-800"}`}>{item.ready ? "ready" : "unavailable"}</span></div>)}</div>}
    </section>
  </div>;
}
