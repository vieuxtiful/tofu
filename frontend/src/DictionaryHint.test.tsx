import { describe, it, expect, beforeEach, afterEach } from "vitest";
import { render, screen, cleanup } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import DictionaryHint from "./DictionaryHint";

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

  it("disappears when dismissed", async () => {
    render(<DictionaryHint />);
    await userEvent.click(screen.getByTitle("dismiss"));
    expect(screen.queryByTestId("dictionary-hint")).toBeNull();
  });

  it("stays dismissed across a reload", async () => {
    const first = render(<DictionaryHint />);
    await userEvent.click(screen.getByTitle("dismiss"));
    first.unmount();

    render(<DictionaryHint />);
    expect(screen.queryByTestId("dictionary-hint")).toBeNull();
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
});
