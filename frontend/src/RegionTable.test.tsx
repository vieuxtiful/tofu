import { describe, it, expect, vi } from "vitest";
import { render, screen, fireEvent, within } from "@testing-library/react";
import RegionTable, { type TableMode } from "./RegionTable";
import type { InstText, LanguageOption, FontOption } from "./api";

function makeRegion(id: string, text: string, target?: string): InstText {
  return {
    id,
    bounding_box: { x: 10, y: 10, width: 80, height: 30 },
    text,
    target_text: target ?? null,
    confidence: 0.9,
    detected_language: "en",
    reading_order: null,
    dnt: false,
    target_language: null,
  };
}

const languages: LanguageOption[] = [
  { code: "en", name: "English" },
  { code: "es", name: "Spanish" },
  { code: "fr", name: "French" },
];

const fontsByLang: Record<string, FontOption[]> = {
  es: [{ family: "Noto Sans", weight: "400", path: "/fonts/noto-sans.ttf" }],
};

const baseProps = {
  mode: "capture" as TableMode,
  regions: [] as InstText[],
  selectedId: null,
  hoveredId: null,
  onSelect: vi.fn(),
  onHover: vi.fn(),
  onTextChange: vi.fn(),
  onTargetChange: vi.fn(),
  onDelete: vi.fn(),
  onOcr: vi.fn(),
  onToggleDnt: vi.fn(),
  onTargetLangChange: vi.fn(),
  onSrcLangChange: vi.fn(),
  onFontChange: vi.fn(),
  onApplyTargetLang: vi.fn(),
  ocrLoading: null,
  languages,
  defaultTargLang: "es",
  fontsByLang,
  onNeedFonts: vi.fn(),
};

function renderTable(overrides: Partial<typeof baseProps> = {}) {
  const props = { ...baseProps, ...overrides };
  const callbacks = {
    onSelect: props.onSelect,
    onDelete: props.onDelete,
    onTargetChange: props.onTargetChange,
    onTextChange: props.onTextChange,
  };
  const utils = render(<RegionTable {...props} />);
  return { ...utils, ...callbacks };
}

/** Find a table row by region ID (rows have data-id or the ID is in a cell). */
function findRow(container: HTMLElement, regionId: string): HTMLElement | null {
  const rows = container.querySelectorAll("tr");
  return Array.from(rows).find((r) => r.textContent?.includes(regionId)) ?? null;
}

describe("RegionTable", () => {
  it("renders region text for each row", () => {
    const regions = [makeRegion("r1", "Hello"), makeRegion("r2", "World")];
    const { container } = renderTable({ regions });
    // Text may appear in multiple places (cell + tooltip), so check getAllByText
    expect(screen.getAllByText("Hello").length).toBeGreaterThanOrEqual(1);
    expect(screen.getAllByText("World").length).toBeGreaterThanOrEqual(1);
  });

  it("calls onSelect when a row is clicked", () => {
    const regions = [makeRegion("r1", "Hello")];
    const { container, onSelect } = renderTable({ regions });
    const row = findRow(container, "r1");
    expect(row).toBeTruthy();
    fireEvent.click(row!);
    expect(onSelect).toHaveBeenCalledWith("r1");
  });

  it("calls onDelete when the trash button is clicked", () => {
    const regions = [makeRegion("r1", "Hello")];
    const { container, onDelete } = renderTable({ regions });
    // Find the trash button by its svg class
    const trashBtn = container.querySelector('button:has(svg.lucide-trash-2)')
      || Array.from(container.querySelectorAll("button")).find((b) => {
        const svg = b.querySelector("svg");
        return svg && (svg.getAttribute("class")?.includes("trash") || svg.classList.contains("lucide-trash-2"));
      });
    expect(trashBtn).toBeTruthy();
    fireEvent.click(trashBtn!);
    expect(onDelete).toHaveBeenCalledWith("r1");
  });

  it("paginates: shows only rowsPerPage regions on first page", () => {
    // Create 20 regions, default capture page size is 15
    const regions = Array.from({ length: 20 }, (_, i) =>
      makeRegion(`r${i}`, `Text${i}`),
    );
    renderTable({ regions });
    // First 15 should be visible
    expect(screen.getAllByText("Text0").length).toBeGreaterThanOrEqual(1);
    expect(screen.getAllByText("Text14").length).toBeGreaterThanOrEqual(1);
    // Text15 should not be on the first page
    expect(screen.queryByText("Text15")).toBeNull();
  });

  it("paginates: next page button advances to the next page", () => {
    const regions = Array.from({ length: 20 }, (_, i) =>
      makeRegion(`r${i}`, `Text${i}`),
    );
    const { container } = renderTable({ regions });
    // Find the pagination "next" buttons (ChevronRight) — there are top and bottom bars
    const nextBtns = Array.from(container.querySelectorAll("button")).filter((b) => {
      const svg = b.querySelector("svg");
      return svg && svg.classList.contains("lucide-chevron-right");
    });
    expect(nextBtns.length).toBeGreaterThanOrEqual(1);
    fireEvent.click(nextBtns[0]);
    // Now Text15 should be visible
    expect(screen.getAllByText("Text15").length).toBeGreaterThanOrEqual(1);
  });

  it("shows merge button when 2+ regions are checked", () => {
    const regions = [makeRegion("r1", "Hello"), makeRegion("r2", "World")];
    const onMergeRegions = vi.fn();
    const { container } = renderTable({ regions, onMergeRegions });
    // Check both rows — find checkboxes in table rows (not the "select all" header)
    const checkboxes = container.querySelectorAll('tbody input[type="checkbox"]');
    expect(checkboxes.length).toBeGreaterThanOrEqual(2);
    fireEvent.click(checkboxes[0] as HTMLInputElement);
    fireEvent.click(checkboxes[1] as HTMLInputElement);
    // Merge button should appear — it has a title like "merge 2 regions into one"
    const mergeBtn = Array.from(container.querySelectorAll("button")).find(
      (b) => b.title?.startsWith("merge "),
    );
    expect(mergeBtn).toBeTruthy();
    fireEvent.click(mergeBtn!);
    expect(onMergeRegions).toHaveBeenCalledWith(["r1", "r2"]);
  });

  it("renders target text in translate mode", () => {
    const regions = [makeRegion("r1", "Hello", "Hola")];
    renderTable({ mode: "translate", regions });
    expect(screen.getAllByText("Hello").length).toBeGreaterThanOrEqual(1);
    expect(screen.getAllByText("Hola").length).toBeGreaterThanOrEqual(1);
  });

  it("renders a table element even with no regions", () => {
    const { container } = renderTable({ regions: [] });
    expect(container.querySelector("table")).toBeTruthy();
  });

  it("applies a highlight class to the selected row", () => {
    const regions = [makeRegion("r1", "Hello"), makeRegion("r2", "World")];
    const { container } = renderTable({ regions, selectedId: "r2" });
    const row = findRow(container, "r2");
    expect(row).toBeTruthy();
    // Selected rows get a distinct background class (bg-zinc-200/70)
    expect(row!.className).toMatch(/bg-zinc-200|bg-zinc-800/);
  });
});
