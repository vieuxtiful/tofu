import type { CSSProperties } from "react";
import type { ProjectAsset } from "./api";
import { identityColor, regionTint } from "./regionPalette";

type Props = {
  assets: ProjectAsset[];
  activeId?: string | null;
  switchingId?: string | null;
  onActivate: (assetId: string) => void;
};

export function assetGridColumns(count: number): number {
  if (count <= 1) return 1;
  if (count <= 4) return 2;
  if (count <= 9) return 3;
  if (count <= 16) return 4;
  if (count <= 25) return 5;
  return 6;
}

export default function AssetPreviewGrid({ assets, activeId, switchingId, onActivate }: Props) {
  const columns = assetGridColumns(assets.length);
  if (assets.length === 1) {
    const item = assets[0];
    return item.asset_url ? (
      <img src={item.asset_url} alt={item.filename ?? "asset preview"} className="max-h-72 rounded-sm object-contain" />
    ) : null;
  }
  return (
    <div
      className="grid h-64 w-full gap-2"
      style={{ gridTemplateColumns: `repeat(${columns}, minmax(0, 1fr))` }}
      data-testid="asset-preview-grid"
    >
      {assets.map((item) => {
        const active = item.asset_id === activeId;
        const color = identityColor(item.asset_id);
        const activate = () => { if (!active && !switchingId) onActivate(item.asset_id); };
        return (
          <button
            type="button"
            key={item.asset_id}
            onClick={(event) => { event.preventDefault(); event.stopPropagation(); activate(); }}
            aria-label={`${item.filename ?? item.asset_id}${active ? ", active asset" : ""}`}
            aria-current={active ? "true" : undefined}
            className="group relative min-h-0 overflow-hidden rounded-md border bg-zinc-100 transition duration-300 hover:-translate-y-0.5 hover:shadow-md focus-visible:outline-2 focus-visible:outline-offset-2 dark:bg-zinc-950"
            style={{
              borderColor: color,
              backgroundColor: regionTint(color, active ? 0.14 : 0.05),
              outlineColor: color,
            } as CSSProperties}
          >
            {item.asset_url && (
              <img
                src={item.asset_url}
                alt=""
                className={`absolute inset-0 h-full w-full object-contain transition-opacity duration-300 ${active ? "opacity-100" : "opacity-65 group-hover:opacity-90"}`}
              />
            )}
            <span className="absolute inset-x-1 bottom-1 truncate rounded bg-black/60 px-1 py-0.5 text-[8px] text-white">
              {item.filename ?? item.asset_id}
            </span>
          </button>
        );
      })}
    </div>
  );
}
