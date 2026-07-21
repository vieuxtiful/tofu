import { useRef, useState, useCallback } from "react";
import { ArrowDownToLine, ChevronDown, Loader2 } from "lucide-react";
import { exportFile } from "./api";
import { FlipButton } from "./Buttons";

interface ExportPanelProps {
  assetId: string;
  targLang: string;
  disabled: boolean;
  embedded?: boolean;
}

const FORMATS = [
  { id: "xliff", label: "XLIFF 1.2", variants: ["standard", "sdl", "crowdin", "smartling", "memoq"] },
  { id: "tmx", label: "TMX", variants: [] },
  { id: "tsv", label: "TSV", variants: [] },
  { id: "csv", label: "CSV", variants: [] },
  { id: "txt", label: "TXT", variants: [] },
];

const VARIANT_LABELS: Record<string, string> = {
  standard: "standard",
  sdl: "SDL Trados",
  crowdin: "Crowdin",
  smartling: "Smartling",
  memoq: "memoQ",
};

const VARIANT_ICONS: Record<string, string> = {
  sdl: "trados-icon.png",
  crowdin: "crowdin-icon.png",
  smartling: "smartling-icon.png",
  memoq: "memoq-icon.png",
};

export default function ExportPanel({ assetId, targLang, disabled, embedded }: ExportPanelProps) {
  const [format, setFormat] = useState("xliff");
  const [variant, setVariant] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const [variantOpen, setVariantOpen] = useState(false);
  const [dropUp, setDropUp] = useState(false);
  const variantRef = useRef<HTMLDivElement | null>(null);

  const currentFormat = FORMATS.find((f) => f.id === format);

  const toggleVariant = useCallback(() => {
    setVariantOpen((v) => {
      if (!v && variantRef.current) {
        const rect = variantRef.current.getBoundingClientRect();
        const dropdownHeight = 180;
        const spaceBelow = window.innerHeight - rect.bottom;
        setDropUp(spaceBelow < dropdownHeight);
      }
      return !v;
    });
  }, []);

  const onExport = async () => {
    if (currentFormat && currentFormat.variants.length > 0 && !variant) return;
    const exportVariant = variant || "standard";
    setLoading(true);
    try {
      const blob = await exportFile(assetId, format, exportVariant, targLang);
      const url = URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = url;
      const ext = format === "xliff" ? "xliff" : format;
      a.download = `${assetId}.${ext}`;
      a.click();
      URL.revokeObjectURL(url);
    } catch (e) {
      console.error(e);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className={embedded ? "space-y-2 p-3" : "bezier-card soft-shadow space-y-3 rounded-lg bg-white/60 p-4 dark:bg-zinc-900/60"}>
      <h3 className="subtext mb-1 flex items-center gap-2 text-sm font-semibold uppercase tracking-wider text-zinc-500 dark:text-zinc-400">Export</h3>
      <div className="flex flex-wrap gap-2">
        {FORMATS.map((f) => (
          <button
            key={f.id}
            onClick={() => { setFormat(f.id); setVariant(null); }}
            className={`rounded-lg px-3 py-1.5 text-xs font-medium transition ${
              format === f.id
                ? "bg-cyan-600 text-white"
                : "bg-zinc-200 text-zinc-600 hover:bg-zinc-300 dark:bg-zinc-800 dark:text-zinc-400 dark:hover:bg-zinc-700"
            }`}
          >
            {f.label}
          </button>
        ))}
      </div>
      {currentFormat && currentFormat.variants.length > 0 && (
        <div className="relative" ref={variantRef}>
          <button
            onClick={toggleVariant}
            className={`flex w-full items-center justify-between gap-1 rounded-lg px-3 py-1.5 text-xs font-medium transition ${
              variant
                ? "bg-cyan-600 text-white"
                : "bg-zinc-200 text-zinc-600 hover:bg-zinc-300 dark:bg-zinc-800 dark:text-zinc-400 dark:hover:bg-zinc-700"
            }`}
          >
            <span className="flex items-center gap-1.5">
              {variant && VARIANT_ICONS[variant] && (
                <img src={`icons/${VARIANT_ICONS[variant]}`} alt="" className={variant === "crowdin" ? "tms-icon-crowdin" : ""} style={{ height: 12, width: 12, objectFit: "contain" }} />
              )}
              {variant ? VARIANT_LABELS[variant] ?? variant : "select cat/tms tool"}
            </span>
            <ChevronDown size={12} className={`transition ${variantOpen ? (dropUp ? "" : "rotate-180") : (dropUp ? "rotate-180" : "")}`} />
          </button>
          {/* Dropdown Morph — reusable expand/collapse animation (see .dropdown-morph in uikit.css) */}
          <div className={`dropdown-morph bezier-card absolute left-0 ${dropUp ? "bottom-full mb-1" : "top-full mt-1"} z-[100] w-full overflow-hidden rounded-lg border border-zinc-300 bg-white dark:border-zinc-700 dark:bg-zinc-900${variantOpen ? " expanded" : ""}`}
               style={variantOpen ? { boxShadow: "4px 4px 0 var(--bc-shadow), 8px 8px 16px rgba(0,0,0,0.18)" } : undefined}>
              {currentFormat.variants.map((v) => (
                <button
                  key={v}
                  onClick={() => { setVariant(v); setVariantOpen(false); }}
                  className={`flex w-full items-center gap-1.5 px-3 py-1.5 text-left text-xs transition hover:bg-zinc-100 dark:hover:bg-zinc-800 ${
                    variant === v ? "text-cyan-600 dark:text-cyan-400" : "text-zinc-700 dark:text-zinc-300"
                  }`}
                >
                  {VARIANT_ICONS[v] && (
                    <img src={`icons/${VARIANT_ICONS[v]}`} alt="" className={v === "crowdin" ? "tms-icon-crowdin" : ""} style={{ height: 12, width: 12, objectFit: "contain", flexShrink: 0 }} />
                  )}
                  {VARIANT_LABELS[v] ?? v}
                </button>
              ))}
          </div>
        </div>
      )}
      <FlipButton
        label="download"
        tooltip="export file"
        icon={loading ? <Loader2 size={18} className="animate-spin" /> : <ArrowDownToLine size={18} />}
        onClick={onExport}
        disabled={disabled || loading || (currentFormat !== undefined && currentFormat.variants.length > 0 && !variant)}
      />
    </div>
  );
}
