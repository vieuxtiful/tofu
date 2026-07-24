import * as React from "react";
import Slider, { type SliderValueLabelProps } from "@mui/material/Slider";
import { styled } from "@mui/material/styles";
import Tooltip from "@mui/material/Tooltip";

function ValueLabelComponent(props: SliderValueLabelProps) {
  const { children, value } = props;
  return (
    <Tooltip enterTouchDelay={0} placement="top" title={value}>
      {children}
    </Tooltip>
  );
}

const GarnishSlider = styled(Slider)(({ theme }) => ({
  color: "#7c3aed",
  height: 3,
  // The wide rectangular thumb needs a little extra room at its minimum
  // position so its left border is never clipped.
  marginLeft: 4,
  padding: "13px 15px 13px 17px",
  "& .MuiSlider-thumb": {
    height: 16,
    width: 39,
    borderRadius: 2,
    backgroundColor: "#fff",
    border: "2px solid currentColor",
    "&:hover": {
      boxShadow: "0 0 0 1px rgba(124, 58, 237, 0.16)",
    },
    "& .garnish-bar": {
      height: 9,
      width: 1,
      backgroundColor: "currentColor",
      marginLeft: 1,
      marginRight: 1,
    },
  },
  "& .MuiSlider-track": {
    height: 7,
  },
  "& .MuiSlider-rail": {
    color: "#d8d8d8",
    opacity: 1,
    height: 5,
    ...theme.applyStyles("dark", {
      color: "#bfbfbf",
      opacity: undefined,
    }),
  },
}));

type GarnishSliderFieldProps = {
  label: string;
  value: number;
  min: number;
  max: number;
  step: number;
  suffix?: string;
  digits?: number;
  marker?: number | null;
  markerLabel?: string;
  sceneLabel?: string;
  disabled?: boolean;
  onChange: (value: number) => void;
};

export default function GarnishSliderField({
  label,
  value,
  min,
  max,
  step,
  suffix = "",
  digits = 1,
  marker = null,
  markerLabel,
  sceneLabel,
  disabled = false,
  onChange,
}: GarnishSliderFieldProps) {
  return (
    <label className="subtext flex min-w-36 flex-1 flex-col gap-0.5 text-[10px]">
      <span className="flex w-[calc(45%+8px)] items-baseline justify-between gap-1 pl-[18px]">
        <span className="font-medium">{label}</span>
        <span className="translate-x-2 font-mono">{Number(value).toFixed(digits)}{suffix}</span>
      </span>
      <span className="relative flex h-5 items-center">
        <GarnishSlider
          aria-label={`garnish ${label}`}
          min={min}
          max={max}
          step={step}
          value={value}
          disabled={disabled}
          onChange={(_, v) => onChange(Number(v))}
          slots={{ valueLabel: ValueLabelComponent }}
          valueLabelDisplay="auto"
          valueLabelFormat={`${Number(value).toFixed(digits)}${suffix}`}
          sx={{ width: "45%", "& .MuiSlider-thumb": { "& .garnish-bar": { display: "none" } } }}
        />
        {marker !== null && (
          <span
            aria-hidden
            title={markerLabel}
            className="pointer-events-none absolute top-1/2 h-2.5 w-0.5 -translate-y-1/2 rounded-sm bg-violet-700 dark:bg-violet-200"
            style={{ left: `${marker}%` }}
          />
        )}
      </span>
      {sceneLabel && <span className="text-[9px] text-violet-700/75 dark:text-violet-300/75">{sceneLabel}</span>}
    </label>
  );
}
