import { ThemeSwitch } from "./Buttons";
import { Theme } from "./theme";

/** theme toggle — inline when placed in header, fixed bottom-center otherwise.
 * uses the 3D switch style (mirrors "Get Capture" button CSS). */
export default function ThemeToggle({ theme, onToggle, inline = false }: {
  theme: Theme;
  onToggle: () => void;
  inline?: boolean;
}) {
  if (inline) {
    return (
      <ThemeSwitch
        checked={theme === "dark"}
        onChange={onToggle}
        title={theme === "dark" ? "switch to light mode" : "switch to dark mode"}
      />
    );
  }
  return (
    <div className="fixed bottom-2 left-1/2 z-40 -translate-x-1/2">
      <ThemeSwitch
        checked={theme === "dark"}
        onChange={onToggle}
        title={theme === "dark" ? "Switch to light mode" : "Switch to dark mode"}
      />
    </div>
  );
}
