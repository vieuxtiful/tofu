import { describe, it, expect, vi } from "vitest";
import { render, screen, fireEvent } from "@testing-library/react";
import SemanticSubstitutionPanel from "./SemanticSubstitutionPanel";
import type { SemanticTextUnit, InstText, GlossaryStatus } from "./api";
import type { AttestedSet } from "./targetGuard";

function makeUnit(id: string, source: string, regionIds: string[]): SemanticTextUnit {
  return {
    id,
    region_ids: regionIds,
    source_text: source,
    bbox: { x: 10, y: 10, width: 200, height: 50 },
    entity_type: "phrase",
    confidence: 0.9,
    analysis_provider: "cicerone",
    semantic_roles: {},
    review_required: false,
    substitution: null,
    pairing: null,
    suggestion: null,
    ocr_repair: null,
  };
}

const emptyGlossaryStatus: GlossaryStatus = {
  global: null,
  project: null,
  effective_mode: null,
  effective_entry_count: 0,
  effective_language_pairs: [],
};

const baseProps: Parameters<typeof SemanticSubstitutionPanel>[0] = {
  units: [],
  drafts: {},
  plans: {},
  busyId: null,
  onDraftChange: vi.fn(),
  onPlan: vi.fn(),
  onRepair: vi.fn(),
  onModifyMembers: vi.fn(),
  onCreateUnit: vi.fn(),
  onDeleteUnit: vi.fn(),
  theme: "light",
  projectId: "test-project",
  targLang: "es",
  regionsById: {},
  attested: new Set() as AttestedSet,
  glossaryStatus: emptyGlossaryStatus,
  onGlossaryUpload: vi.fn(),
  onGlossaryDelete: vi.fn(),
  glossaryUploading: false,
  glossaryUploadStep: null,
  glossaryUploadError: null,
};

function renderPanel(overrides: Partial<typeof baseProps> = {}) {
  const props = { ...baseProps, ...overrides };
  const callbacks = {
    onCreateUnit: props.onCreateUnit,
    onDeleteUnit: props.onDeleteUnit,
    onModifyMembers: props.onModifyMembers,
    onDraftChange: props.onDraftChange,
    onPlan: props.onPlan,
  };
  const utils = render(<SemanticSubstitutionPanel {...props} />);
  return { ...utils, ...callbacks };
}

describe("SemanticSubstitutionPanel", () => {
  it("renders null when no multi-region units and no user-created units", () => {
    const { container } = renderPanel({
      units: [makeUnit("u1", "Hello", ["r1"])], // single-region, not user_created
    });
    expect(container.firstChild).toBeNull();
  });

  it("renders the panel when a multi-region unit exists", () => {
    const units = [makeUnit("u1", "Hello World", ["r1", "r2"])];
    const { container } = renderPanel({ units });
    expect(container.querySelector("section")).toBeTruthy();
    // The "Basil" label should be present
    expect(screen.getByText("Basil")).toBeTruthy();
  });

  it("calls onCreateUnit when add-plate button is clicked", () => {
    const units = [makeUnit("u1", "Hello World", ["r1", "r2"])];
    const { container, onCreateUnit } = renderPanel({ units });
    // Use querySelector with aria-label to find the add-plate button
    const addBtn = container.querySelector('button[aria-label="Add plate"]');
    expect(addBtn).toBeTruthy();
    fireEvent.click(addBtn!);
    expect(onCreateUnit).toHaveBeenCalled();
  });

  it("dismisses a unit and shows a restore button", () => {
    const units = [makeUnit("u1", "Hello World", ["r1", "r2"])];
    const { container } = renderPanel({ units });
    // Find the dismiss button on the BasilUnitCard — it has an X icon
    // and is NOT the panel collapse button (which has aria-label containing "Basil")
    const xButtons = Array.from(container.querySelectorAll("button")).filter((b) => {
      const svg = b.querySelector("svg");
      const label = b.getAttribute("aria-label") ?? "";
      return svg && svg.classList.contains("lucide-x") && !label.includes("Basil");
    });
    expect(xButtons.length).toBeGreaterThanOrEqual(1);
    fireEvent.click(xButtons[0]);
    // After dismiss, should show a restore button with the source text
    const restoreBtn = container.querySelector('button[aria-label^="show plate"]');
    expect(restoreBtn).toBeTruthy();
    expect(restoreBtn!.textContent).toContain("Hello World");
  });

  it("calls onDeleteUnit when delete-plate button is clicked after dismiss", () => {
    const units = [makeUnit("u1", "Hello World", ["r1", "r2"])];
    const { container, onDeleteUnit } = renderPanel({ units });
    // First dismiss the unit
    const xButtons = Array.from(container.querySelectorAll("button")).filter((b) => {
      const svg = b.querySelector("svg");
      const label = b.getAttribute("aria-label") ?? "";
      return svg && svg.classList.contains("lucide-x") && !label.includes("Basil");
    });
    fireEvent.click(xButtons[0]);
    // Now find the delete-plate button by aria-label
    const deleteBtn = container.querySelector('button[aria-label="Delete plate"]');
    expect(deleteBtn).toBeTruthy();
    fireEvent.click(deleteBtn!);
    expect(onDeleteUnit).toHaveBeenCalledWith("u1");
  });

  it("renders user-created units even with a single region", () => {
    const userUnit = makeUnit("u-user", "My Plate", ["r1"]);
    userUnit.analysis_provider = "user_created";
    const { container } = renderPanel({ units: [userUnit] });
    expect(container.querySelector("section")).toBeTruthy();
    expect(screen.getAllByText("My Plate").length).toBeGreaterThanOrEqual(1);
  });

  it("expands and collapses when the toggle button is clicked", () => {
    const units = [makeUnit("u1", "Hello World", ["r1", "r2"])];
    const { container } = renderPanel({ units });
    // The collapse/expand button has aria-label "expand Basil" or "collapse Basil"
    const expandBtn = container.querySelector('button[aria-label="Expand Basil"]');
    expect(expandBtn).toBeTruthy();
    fireEvent.click(expandBtn!);
    // After click, the aria-label should change to "collapse Basil"
    const collapseBtn = container.querySelector('button[aria-label="Collapse Basil"]');
    expect(collapseBtn).toBeTruthy();
  });

  it("renders the glossary panel section", () => {
    const units = [makeUnit("u1", "Hello World", ["r1", "r2"])];
    const { container } = renderPanel({ units });
    // The GlossaryPanel is rendered inside — check for its presence
    // by looking for glossary-related elements
    const section = container.querySelector("section");
    expect(section).toBeTruthy();
    // The panel should contain the glossary section
    expect(section!.innerHTML).toContain("glossary");
  });

  it("shows loading indicator when busyId matches a unit", () => {
    const units = [makeUnit("u1", "Hello World", ["r1", "r2"])];
    const { container } = renderPanel({ units, busyId: "u1" });
    // A loading spinner should be present — look for Loader2 or any spinner
    const spinner = container.querySelector('svg.lucide-loader-2, svg.lucide-loader-circle, .animate-spin');
    expect(spinner).toBeTruthy();
  });
});
