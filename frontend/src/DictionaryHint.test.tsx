import { describe, it, expect, beforeEach, afterEach } from "vitest";
import { render, screen, cleanup } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import DictionaryHint from "./DictionaryHint";

const hint = () => screen.getByTestId("dictionary-hint");

describe("DictionaryHint", () => {
  beforeEach(() => window.localStorage.clear());
  afterEach(() => cleanup());

  it("names the shortcut in real keycaps, not daisyUI class names", () => {
    render(<DictionaryHint />);
    // <kbd> elements, so the hint survives the removal of any CSS framework
    // -- which is exactly how the previous keycaps became invisible.
    const caps = document.querySelectorAll("kbd[data-slot='kbd']");
    expect([...caps].map((c) => c.textContent)).toEqual(["Ctrl", "Space"]);
  });

  it("mentions the alternate binding to a screen reader", () => {
    // Ctrl+Space is the input-source switcher on macOS, so a user for whom
    // it never fires needs to hear the shortcut that does.
    render(<DictionaryHint />);
    expect(
      screen.getByLabelText(/Control plus Space, or Alt plus Down Arrow/i),
    ).toBeTruthy();
  });

  it("still renders when localStorage is unavailable", () => {
    // Private-browsing profiles throw on access rather than returning null.
    const original = window.localStorage.getItem;
    window.localStorage.getItem = () => { throw new Error("denied"); };
    try {
      render(<DictionaryHint />);
      expect(screen.getByTestId("dictionary-hint")).toBeTruthy();
    } finally {
      window.localStorage.getItem = original;
    }
  });

  it("shows the keys before the label", () => {
    // The order the user performs it in, matching every other readout on
    // that row ("A draw", "Esc deselect").
    render(<DictionaryHint />);
    expect(hint().textContent).toBe("Ctrl+SpaceUser Dictionary");
  });

  it("collapses rather than disappearing when dismissed", async () => {
    // It used to unmount, so the only way back was clearing site data -- a
    // permanent consequence for a click that reads as provisional.
    render(<DictionaryHint />);
    await userEvent.click(screen.getByTitle("Hide the dictionary shortcut"));
    expect(screen.queryByTestId("dictionary-hint")).toBeTruthy();
    expect(hint().getAttribute("data-collapsed")).toBe("true");
  });

  it("stays collapsed across a reload", async () => {
    const first = render(<DictionaryHint />);
    await userEvent.click(screen.getByTitle("Hide the dictionary shortcut"));
    first.unmount();

    render(<DictionaryHint />);
    expect(hint().getAttribute("data-collapsed")).toBe("true");
  });

  it("offers to show it again once collapsed", async () => {
    render(<DictionaryHint />);
    await userEvent.click(screen.getByTitle("Hide the dictionary shortcut"));
    await userEvent.click(screen.getByTitle("Show the dictionary shortcut"));
    expect(hint().getAttribute("data-collapsed")).toBe("false");
  });

  it("remembers being reopened too", async () => {
    // The stored flag has to round-trip both ways, or reopening survives
    // only until the next reload.
    window.localStorage.setItem("tofu.hint.dictionary.v1", "1");
    const first = render(<DictionaryHint />);
    await userEvent.click(screen.getByTitle("Show the dictionary shortcut"));
    first.unmount();

    render(<DictionaryHint />);
    expect(hint().getAttribute("data-collapsed")).toBe("false");
  });

  it("keeps the collapsed content out of the accessibility tree", async () => {
    // The width transition leaves it in the DOM; a screen reader must not
    // read a hint that is not on screen.
    render(<DictionaryHint />);
    await userEvent.click(screen.getByTitle("Hide the dictionary shortcut"));
    expect(
      hint().querySelector(".dictionary-hint-body")?.getAttribute("aria-hidden"),
    ).toBe("true");
  });
});
