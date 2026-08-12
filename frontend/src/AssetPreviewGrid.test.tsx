import { cleanup, fireEvent, render, screen } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";
import AssetPreviewGrid, { assetGridColumns } from "./AssetPreviewGrid";
import type { ProjectAsset } from "./api";

afterEach(cleanup);

const asset = (index: number): ProjectAsset => ({
  asset_id: `asset-${index}`, project_id: "p1", filename: `${index}.png`,
  uploaded_at: index, is_active: index === 0 ? 1 : 0,
  asset_url: `/uploads/${index}.png`, has_manifest: true, ground_truth: [],
});

describe("AssetPreviewGrid", () => {
  it.each([[2, 2], [6, 3], [7, 3], [18, 5], [30, 6]])("uses the responsive grid for %i assets", (count, columns) => {
    expect(assetGridColumns(count)).toBe(columns);
  });

  it("preserves the legacy full preview for one asset", () => {
    render(<AssetPreviewGrid assets={[asset(0)]} activeId="asset-0" onActivate={() => {}} />);
    expect(screen.getByRole("img").className).toContain("max-h-72");
    expect(screen.getByRole("img").className).toContain("object-contain");
    expect(screen.queryByTestId("asset-preview-grid")).toBeNull();
  });

  it("activates an inactive tile and ignores the active one", () => {
    const activate = vi.fn();
    render(<AssetPreviewGrid assets={[asset(0), asset(1)]} activeId="asset-0" onActivate={activate} />);
    fireEvent.click(screen.getByRole("button", { name: "0.png, active asset" }));
    fireEvent.click(screen.getByRole("button", { name: "1.png" }));
    expect(activate).toHaveBeenCalledOnce();
    expect(activate).toHaveBeenCalledWith("asset-1");
  });
});
