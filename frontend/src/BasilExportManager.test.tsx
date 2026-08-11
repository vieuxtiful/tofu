import { describe, it, expect, vi, afterEach } from "vitest";
import { render, screen, fireEvent, cleanup } from "@testing-library/react";
import BasilExportManager from "./BasilExportManager";

afterEach(() => cleanup());

vi.mock("./api", () => ({
  exportFile: vi.fn(async () => new Blob(["<xliff/>"], { type: "application/xml" })),
}));

const baseProps: Parameters<typeof BasilExportManager>[0] = {
  assetId: "asset-1",
  targLang: "fr-FR",
  disabled: false,
  projectName: "demo",
};

/** The Export Manager moved from Capture into Basil.  These assertions pin
 *  the parts a CAT/TMS user would notice if the move changed them. */
describe("BasilExportManager", () => {
  it("preserves the format order from the Capture-tab panel", () => {
    render(<BasilExportManager {...baseProps} />);
    const labels = ["XLIFF 1.2", "TMX", "TSV", "CSV", "TXT"];
    const rendered = labels.map((label) => screen.getByText(label));
    // document order must match the declared order
    for (let i = 1; i < rendered.length; i += 1) {
      const position = rendered[i - 1].compareDocumentPosition(rendered[i]);
      expect(position & Node.DOCUMENT_POSITION_FOLLOWING).toBeTruthy();
    }
  });

  it("keeps the CAT/TMS tool order and icons for XLIFF", () => {
    render(<BasilExportManager {...baseProps} />);
    fireEvent.click(screen.getByText("select cat/tms tool"));
    const order = ["standard", "SDL Trados", "Crowdin", "Smartling", "memoQ"];
    const rendered = order.map((label) => screen.getByText(label));
    for (let i = 1; i < rendered.length; i += 1) {
      const position = rendered[i - 1].compareDocumentPosition(rendered[i]);
      expect(position & Node.DOCUMENT_POSITION_FOLLOWING).toBeTruthy();
    }
    const icons = Array.from(document.querySelectorAll("img")).map((img) => img.getAttribute("src"));
    expect(icons).toContain("icons/trados-icon.png");
    expect(icons).toContain("icons/crowdin-icon.png");
    expect(icons).toContain("icons/smartling-icon.png");
    expect(icons).toContain("icons/memoq-icon.png");
  });

  it("only offers formats the Capture panel offered (VTM stays CLI-only)", () => {
    render(<BasilExportManager {...baseProps} />);
    expect(screen.queryByText("VTM")).toBeNull();
  });

  it("blocks download until a CAT/TMS variant is chosen for XLIFF", () => {
    render(<BasilExportManager {...baseProps} />);
    const download = screen.getByText("download").closest("button");
    expect(download).not.toBeNull();
    expect((download as HTMLButtonElement).disabled).toBe(true);
    fireEvent.click(screen.getByText("select cat/tms tool"));
    fireEvent.click(screen.getByText("SDL Trados"));
    expect((download as HTMLButtonElement).disabled).toBe(false);
  });

  it("needs no variant for single-variant formats", () => {
    render(<BasilExportManager {...baseProps} />);
    fireEvent.click(screen.getByText("TSV"));
    const download = screen.getByText("download").closest("button");
    expect((download as HTMLButtonElement).disabled).toBe(false);
  });

  it("stays disabled when there is nothing translatable", () => {
    render(<BasilExportManager {...baseProps} disabled />);
    fireEvent.click(screen.getByText("TXT"));
    const download = screen.getByText("download").closest("button");
    expect((download as HTMLButtonElement).disabled).toBe(true);
  });

  it("is labelled as the Translation File Manager", () => {
    render(<BasilExportManager {...baseProps} />);
    expect(screen.getByLabelText("Translation File Manager")).toBeTruthy();
  });

  it("keeps the Capture-tab card styling and heading verbatim", () => {
    // The move was a relocation, not a restyle: this panel must look exactly
    // as it did under Capture, now beneath the Translate text manifest.
    render(<BasilExportManager {...baseProps} />);
    const card = screen.getByLabelText("Translation File Manager");
    for (const cls of ["bezier-card", "soft-shadow", "space-y-3", "rounded-lg", "bg-white/60", "p-4", "dark:bg-zinc-900/60"]) {
      expect(card.classList.contains(cls)).toBe(true);
    }
    const heading = card.querySelector("h3");
    expect(heading?.textContent).toBe("Export");
    expect(heading?.className).toContain("uppercase");
  });
});
