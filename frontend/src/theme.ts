import { useCallback, useEffect, useState } from "react";

export type Theme = "dark" | "light";

const THEME_KEY = "tofu.theme";

function readStored(): Theme {
  const t = localStorage.getItem(THEME_KEY);
  return t === "light" ? "light" : "dark"; // ToFU defaults to dark
}

function applyToDom(theme: Theme) {
  const root = document.documentElement;
  root.classList.toggle("dark", theme === "dark");
  root.classList.toggle("light", theme === "light");
}

// apply before first paint so the app never flashes the wrong theme
applyToDom(readStored());

/** the persisted theme, without subscribing to changes.
 *
 * For components that render before the app owns any state -- KeyGate is the
 * one -- and only need to pick an asset. Calling useTheme() there would stand
 * up a second copy of the theme state that the in-app toggle never reaches. */
export function storedTheme(): Theme {
  return readStored();
}

/** persistent light/dark theme. `dark` class on <html> drives Tailwind's
 * class strategy; the `light`/`dark` classes also drive raw CSS in
 * index.css/uikit.css. */
export function useTheme(): { theme: Theme; toggle: () => void } {
  const [theme, setTheme] = useState<Theme>(readStored);

  useEffect(() => {
    applyToDom(theme);
    localStorage.setItem(THEME_KEY, theme);
  }, [theme]);

  const toggle = useCallback(() => {
    setTheme((t) => (t === "dark" ? "light" : "dark"));
  }, []);

  return { theme, toggle };
}

/** logo path for the current theme (white mark on dark, black on light) */
export function logoSrc(theme: Theme): string {
  return theme === "light" ? "/tofu-blk-alt-main.png" : "/tofu-wht-alt.png";
}
