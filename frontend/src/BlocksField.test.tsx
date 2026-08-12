import { describe, it, expect, vi, afterEach } from "vitest";
import { render, screen, cleanup, fireEvent } from "@testing-library/react";
import { useState } from "react";
import BlocksField from "./BlocksField";
import type { LanguageAssessment } from "./api";

afterEach(() => cleanup());

const assess = async (): Promise<LanguageAssessment> => ({
  state: "insufficient_evidence", detected_language: null, source_language: null,
  scripts: [], direction: "ltr", reasons: [], warns: false, policy_revision: "test",
});

/** The field is controlled, so the test drives it the way App does. */
function Harness({ initial = [], onChange }: { initial?: string[]; onChange?: (b: string[]) => void }) {
  const [blocks, setBlocks] = useState<string[]>(initial);
  const [draft, setDraft] = useState("");
  return (
    <BlocksField
      blocks={blocks}
      onChange={(next) => { setBlocks(next); onChange?.(next); }}
      draft={draft}
      onDraftChange={setDraft}
      sourceLanguage="fr-FR"
      assess={assess}
    />
  );
}

function type(value: string) {
  const input = screen.getByRole("textbox");
  fireEvent.change(input, { target: { value } });
  return input;
}

const chips = () => [...document.querySelectorAll("li")].map((li) => li.textContent?.replace(/\s+$/, ""));

describe("BlocksField", () => {
  it("commits a phrase as ONE Block, not three", () => {
    // The whole point: whitespace is no longer the delimiter.
    render(<Harness />);
    fireEvent.keyDown(type("la première saisie"), { key: "Enter" });
    expect(chips()).toEqual(["la première saisie"]);
  });

  it("renders the draft as ONE token, not one per word", () => {
    // The capability was always there -- spaces have never committed -- but
    // the field coloured each whitespace-separated word separately, so it
    // looked like the phrase was already being split. Users responded by
    // committing the words one at a time, which is how prem-sais-gt ended up
    // with "la", "première" and "saisie" as three Blocks.
    render(<Harness />);
    type("la première saisie");
    const tokens = document.querySelectorAll("[data-testid='single-block-token']");
    expect(tokens.length).toBe(1);
    expect(tokens[0].textContent).toBe("la première saisie");
  });

  it("commits on Shift+Enter as well, without requiring Shift for spaces", () => {
    // An alias, never a requirement: Shift is needed for capitals and much
    // punctuation, and gating spaces on it would fight every IME.
    render(<Harness />);
    fireEvent.keyDown(type("la première saisie"), { key: "Enter", shiftKey: true });
    expect(chips()).toEqual(["la première saisie"]);
  });

  it("trims a phrase's outer whitespace but keeps its inner spaces", () => {
    render(<Harness />);
    fireEvent.keyDown(type("  la première saisie  "), { key: "Enter" });
    expect(chips()).toEqual(["la première saisie"]);
  });

  it("keeps two identical Blocks as two", () => {
    render(<Harness />);
    fireEvent.keyDown(type("PARIS"), { key: "Enter" });
    fireEvent.keyDown(type("PARIS"), { key: "Enter" });
    expect(chips()).toEqual(["PARIS", "PARIS"]);
  });

  it("preserves entry order", () => {
    render(<Harness />);
    for (const entry of ["troisième", "premier", "deuxième"]) {
      fireEvent.keyDown(type(entry), { key: "Enter" });
    }
    expect(chips()).toEqual(["troisième", "premier", "deuxième"]);
  });

  it("does not commit mid-composition", () => {
    // On a Japanese IME the first Enter CONFIRMS the candidate. Committing
    // on it would store whatever had been converted so far -- and the
    // languages most likely to need Guided are the ones that compose.
    render(<Harness />);
    const input = screen.getByRole("textbox");
    fireEvent.compositionStart(input);
    fireEvent.change(input, { target: { value: "とうきょう" } });
    fireEvent.keyDown(input, { key: "Enter", isComposing: true });
    expect(chips()).toEqual([]);

    fireEvent.compositionEnd(input, { target: { value: "東京都" } });
    fireEvent.change(input, { target: { value: "東京都" } });
    fireEvent.keyDown(input, { key: "Enter" });
    expect(chips()).toEqual(["東京都"]);
  });

  it("ignores an empty entry", () => {
    render(<Harness />);
    fireEvent.keyDown(type("   "), { key: "Enter" });
    expect(chips()).toEqual([]);
  });

  it("removes one Block without disturbing its twin", () => {
    render(<Harness initial={["PARIS", "PARIS", "SORTIE"]} />);
    fireEvent.click(screen.getByLabelText("Remove Block 1: PARIS"));
    expect(chips()).toEqual(["PARIS", "SORTIE"]);
  });

  it("removes the last Block on Backspace in an empty entry", () => {
    render(<Harness initial={["SORTIE", "ENTRÉE"]} />);
    const input = screen.getByRole("textbox");
    fireEvent.keyDown(input, { key: "Backspace" });
    expect(chips()).toEqual(["SORTIE"]);
  });

  it("does not eat a Backspace that has text to delete", () => {
    render(<Harness initial={["SORTIE"]} />);
    const input = type("PAR");
    fireEvent.keyDown(input, { key: "Backspace" });
    expect(chips()).toEqual(["SORTIE"]);
  });

  it("commits an entry the user clicks away from", () => {
    render(<Harness />);
    fireEvent.blur(type("SORTIE"));
    expect(chips()).toEqual(["SORTIE"]);
  });

  it("reports the committed list to its parent", () => {
    const onChange = vi.fn();
    render(<Harness onChange={onChange} />);
    fireEvent.keyDown(type("la première saisie"), { key: "Enter" });
    expect(onChange).toHaveBeenCalledWith(["la première saisie"]);
  });

  it("never derives atoms itself", () => {
    // `mise` is authoritative. With no server record there is no atom count
    // to show, and the UI must do without rather than tokenize the phrase.
    render(<Harness initial={["la première saisie"]} />);
    expect(screen.getByTitle("Block 1")).toBeTruthy();
  });

  it("shows the server's atom count when it has one", () => {
    render(
      <BlocksField
        blocks={["la première saisie"]}
        onChange={() => {}}
        draft="" onDraftChange={() => {}}
        assess={assess}
        records={[{
          id: "g1", raw_text: "la première saisie",
          normalized_text: "la première saisie", position: 0, scope: "asset",
          find_all: false, language_assessment: null, detection_assessment: null,
          atoms: [
            { id: "g1a1", text: "la", position: 0, kind: "word", script: "Latn", direction: "ltr" },
            { id: "g1a2", text: "première", position: 1, kind: "word", script: "Latn", direction: "ltr" },
            { id: "g1a3", text: "saisie", position: 2, kind: "word", script: "Latn", direction: "ltr" },
          ],
        }]}
      />,
    );
    // The atom count is gone from the tooltip. It was an honesty check while
    // whitespace splitting was in doubt; now that a Block is committed whole
    // and located by one box, "3 atoms" only invites the reader to wonder
    // whether it was split after all. The record is still passed in, so this
    // asserts the count is not SHOWN, not that it stopped arriving.
    expect(screen.getByTitle("Block 1")).toBeTruthy();
    expect(screen.queryByTitle(/atom/)).toBeNull();
  });

  it("hides the remove control while a capture is running", () => {
    render(
      <BlocksField
        blocks={["SORTIE"]} onChange={() => {}}
        draft="" onDraftChange={() => {}} capturing assess={assess}
      />,
    );
    expect(screen.queryByLabelText("Remove Block 1: SORTIE")).toBeNull();
  });
});
