// 🍢 uikit buttons — user-specified styles (see uikit.css)

import { ReactNode } from "react";

/** 3D press button ("Get Capture") — .press-button in uikit.css */
export function PressButton({ children, onClick, disabled, title }: {
  children: ReactNode;
  onClick?: () => void;
  disabled?: boolean;
  title?: string;
}) {
  return (
    <button className="press-button" onClick={onClick} disabled={disabled} title={title}>
      <span>{children}</span>
    </button>
  );
}

/** flip button: label slides up revealing an icon on hover, tooltip above —
 * .flip-button in uikit.css */
export function FlipButton({ label, tooltip, icon, onClick, disabled, active, width, reversed }: {
  label: string;
  tooltip?: string;
  icon: ReactNode;
  onClick?: () => void;
  disabled?: boolean;
  active?: boolean;
  width?: number;
  reversed?: boolean;
}) {
  return (
    <button
      className={`flip-button${active ? " active" : ""}${reversed ? " reversed" : ""}`}
      data-tooltip={tooltip || undefined}
      onClick={onClick}
      disabled={disabled}
      style={width ? ({ "--width": `${width}px` } as React.CSSProperties) : undefined}
    >
      <div className="fb-wrapper">
        <div className="fb-text">{label}</div>
        <span className="fb-icon">{icon}</span>
      </div>
    </button>
  );
}

/** penumbra switch — .penumbra-switch in uikit.css */
export function PenumbraSwitch({ checked, onChange, label }: {
  checked: boolean;
  onChange: (v: boolean) => void;
  label: string;
}) {
  return (
    <label className="penumbra-switch">
      <input
        className="penumbra-switch__input"
        type="checkbox"
        checked={checked}
        onChange={(e) => onChange(e.target.checked)}
      />
      <span className="penumbra-switch__rail" aria-hidden="true">
        <span className="penumbra-switch__knob" />
      </span>
      <span className="penumbra-switch__label">{label}</span>
    </label>
  );
}

/** theme switch — .switch in uikit.css (mirrors "Get Capture" 3D style) */
export function ThemeSwitch({ checked, onChange, title }: {
  checked: boolean;
  onChange: () => void;
  title?: string;
}) {
  return (
    <label className="switch" title={title}>
      <input
        className="toggle"
        type="checkbox"
        checked={checked}
        onChange={onChange}
      />
      <span className="slider" />
      <span className="card-side" />
    </label>
  );
}
