import { readFileSync } from "node:fs";
import { resolve } from "node:path";
import { describe, it, expect, afterEach } from "vitest";
import { render, screen, cleanup } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import GuidedPromptBubble from "./GuidedPromptBubble";
import GuidedProgress from "./GuidedProgress";
import { promptFor } from "./useGuidedCapture";
import type { GuidedBlockRecord } from "./api";
import type { GuidedProgress as Progress } from "./useGuidedCapture";

afterEach(() => cleanup());

function block(over: Partial<GuidedBlockRecord> = {}): GuidedBlockRecord {
  return {
    id: "g1", raw_text: "la première saisie",
    normalized_text: "la première saisie", position: 0, scope: "asset",
    find_all: false, language_assessment: null, detection_assessment: null,
    atoms: [
      { id: "g1a1", text: "la", position: 0, kind: "word", script: "Latn", direction: "ltr" },
      { id: "g1a2", text: "première", position: 1, kind: "word", script: "Latn", direction: "ltr" },
      { id: "g1a3", text: "saisie", position: 2, kind: "word", script: "Latn", direction: "ltr" },
    ],
    ...over,
  };
}

const progress = (over: Partial<Progress> = {}): Progress => ({
  blocks_total: 8, blocks_complete: 6, blocks_skipped: 0, blocks_review: 0,
  blocks_resolved: 6, atoms_total: 10, atoms_matched: 7, fraction: 0.7, ...over,
});

describe("promptFor", () => {
  it("asks for one box when the Block is indivisible", () => {
    const prompt = promptFor(block({
      raw_text: "釁", atoms: [
        { id: "g1a1", text: "釁", position: 0, kind: "phrase", script: "Hani", direction: "ltr" },
      ],
    }), false);
    expect(prompt?.plural).toBe(false);
  });

  it("asks for boxes when the Block is a phrase", () => {
    expect(promptFor(block(), false)?.plural).toBe(true);
  });

  it("asks for boxes when find_all is set, however it divides", () => {
    const prompt = promptFor(block({
      raw_text: "釁", find_all: true, atoms: [
        { id: "g1a1", text: "釁", position: 0, kind: "phrase", script: "Hani", direction: "ltr" },
      ],
    }), false);
    expect(prompt?.plural).toBe(true);
  });

  it("keeps the whole phrase, never just what remains", () => {
    // `saisie` is an atom of the Block, not a Block. Re-wording the prompt
    // down to it asserts the opposite.
    const prompt = promptFor(block({
      detection_assessment: {
        status: "partial", matched_text: "la première", remaining_text: "saisie",
      },
    }), false);
    expect(prompt?.text).toBe("la première saisie");
    expect(prompt?.remainingText).toBe("saisie");
  });

  it("reports saving over the server status", () => {
    expect(promptFor(block(), true)?.status).toBe("saving");
  });
});

describe("GuidedPromptBubble", () => {
  it("names the whole Block to a screen reader even when partial", () => {
    render(<GuidedPromptBubble prompt={{
      blockId: "g1", text: "la première saisie", remainingText: "saisie",
      matchedText: "la première", plural: true, status: "partial",
    }} />);
    const label = screen.getByRole("status").getAttribute("aria-label") ?? "";
    expect(label).toContain("la première saisie");
    expect(label).toContain("Still needed: saisie");
  });

  it("strikes through what has been found and highlights what has not", () => {
    const { container } = render(<GuidedPromptBubble prompt={{
      blockId: "g1", text: "la première saisie", remainingText: "saisie",
      matchedText: "la première", plural: true, status: "partial",
    }} />);
    expect(container.querySelector(".line-through")?.textContent).toBe("la première");
  });

  it("follows the configured bbox colour for the remaining term", () => {
    // Asserted structurally, not by computed style: happy-dom drops
    // `color: var(...)` declarations entirely, so an inline-style assertion
    // here would pass or fail for reasons unrelated to the component.
    const { container } = render(<GuidedPromptBubble prompt={{
      blockId: "g1", text: "la première saisie", remainingText: "saisie",
      matchedText: "la première", plural: true, status: "partial",
    }} />);
    expect(container.querySelector(".guided-term")?.textContent).toBe("saisie");
    const css = readFileSync(resolve(process.cwd(), "src/bbox.css"), "utf-8");
    expect(css).toMatch(/\.guided-term\s*\{[^}]*var\(--bbox-color/);
  });

  it("says box for a single Block and box(es) for a phrase", () => {
    render(<GuidedPromptBubble prompt={{
      blockId: "g1", text: "釁", remainingText: "釁", matchedText: "",
      plural: false, status: "pending",
    }} />);
    expect(screen.getByRole("status").textContent).toContain("Draw box for");
    cleanup();
    render(<GuidedPromptBubble prompt={{
      blockId: "g1", text: "la première saisie", remainingText: "la première saisie",
      matchedText: "", plural: true, status: "pending",
    }} />);
    expect(screen.getByRole("status").textContent).toContain("Draw box(es) for");
  });

  it("announces prompt changes politely", () => {
    render(<GuidedPromptBubble prompt={{
      blockId: "g1", text: "釁", remainingText: "釁", matchedText: "",
      plural: false, status: "pending",
    }} />);
    expect(screen.getByRole("status").getAttribute("aria-live")).toBe("polite");
  });

  it("hides the escape hatches while a box is being saved", () => {
    // The lock has to be visible, not just felt as the canvas ignoring you.
    // It says "saving", not "reading": Guided runs no recogniser, and the
    // word described a refinement pass that overwrote the string the user
    // had already typed.
    render(<GuidedPromptBubble
      prompt={{
        blockId: "g1", text: "釁", remainingText: "釁", matchedText: "",
        plural: false, status: "saving",
      }}
      onSkip={() => {}}
    />);
    expect(screen.queryByText("skip")).toBeNull();
    expect(screen.getByRole("status").textContent).toContain("saving…");
  });

  it("offers a way out of an unreadable Block", () => {
    let skipped = false;
    render(<GuidedPromptBubble
      prompt={{
        blockId: "g1", text: "釁", remainingText: "釁", matchedText: "",
        plural: false, status: "pending",
      }}
      onSkip={() => { skipped = true; }}
    />);
    return userEvent.click(screen.getByText("skip")).then(() => {
      expect(skipped).toBe(true);
    });
  });
});

describe("GuidedProgress", () => {
  it("computes the bar and the label independently", () => {
    // A five-atom Block at four atoms must not show a full bar labelled 0/1,
    // nor an empty bar because no Block has finished.
    render(<GuidedProgress progress={progress({
      blocks_total: 1, blocks_complete: 0, blocks_resolved: 0,
      atoms_total: 5, atoms_matched: 4, fraction: 0.8,
    })} />);
    const label = screen.getByRole("status").getAttribute("aria-label") ?? "";
    expect(label).toContain("0 of 1 Blocks resolved");
    expect(label).toContain("80% of terms located");
  });

  it("shows completed Blocks over total", () => {
    render(<GuidedProgress progress={progress()} />);
    expect(screen.getByTestId("guided-progress").textContent).toContain("6/8");
  });

  it("says how many Blocks were skipped", () => {
    // A bar that reads full because Blocks were skipped is not the same as
    // one that reads full because everything was located.
    render(<GuidedProgress progress={progress({
      blocks_skipped: 2, blocks_resolved: 8, fraction: 1,
    })} />);
    expect(screen.getByRole("status").getAttribute("aria-label")).toContain("2 skipped");
  });

  it("renders nothing when there are no Blocks", () => {
    render(<GuidedProgress progress={progress({ blocks_total: 0 })} />);
    expect(screen.queryByTestId("guided-progress")).toBeNull();
  });
});
