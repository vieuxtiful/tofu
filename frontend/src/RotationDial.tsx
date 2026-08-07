import { useRef } from "react";
import type { PointerEvent as ReactPointerEvent } from "react";

type RotationDialProps = {
  value: number;
  onChange: (degrees: number) => void;
  onReset: () => void;
  disabled?: boolean;
};

const clamp = (degrees: number) => Math.max(-360, Math.min(360, Math.round(degrees * 10) / 10));

export default function RotationDial({ value, onChange, onReset, disabled = false }: RotationDialProps) {
  const lastPointerAngle = useRef<number | null>(null);
  const accumulated = useRef(0);

  const pointerAngle = (event: ReactPointerEvent<SVGSVGElement>) => {
    const rect = event.currentTarget.getBoundingClientRect();
    return Math.atan2(-(event.clientY - (rect.top + rect.height / 2)), event.clientX - (rect.left + rect.width / 2)) * 180 / Math.PI;
  };

  const begin = (event: ReactPointerEvent<SVGSVGElement>) => {
    if (disabled) return;
    const degrees = pointerAngle(event);
    lastPointerAngle.current = degrees;
    accumulated.current = clamp(degrees);
    event.currentTarget.setPointerCapture(event.pointerId);
    onChange(accumulated.current);
  };

  const move = (event: ReactPointerEvent<SVGSVGElement>) => {
    if (disabled) return;
    if (!event.currentTarget.hasPointerCapture(event.pointerId) || lastPointerAngle.current === null) return;
    const nextAngle = pointerAngle(event);
    let delta = nextAngle - lastPointerAngle.current;
    if (delta > 180) delta -= 360;
    if (delta < -180) delta += 360;
    lastPointerAngle.current = nextAngle;
    accumulated.current = clamp(accumulated.current + delta);
    onChange(accumulated.current);
  };

  const end = (event: ReactPointerEvent<SVGSVGElement>) => {
    if (event.currentTarget.hasPointerCapture(event.pointerId)) event.currentTarget.releasePointerCapture(event.pointerId);
    lastPointerAngle.current = null;
  };

  const radians = ((value % 360) * Math.PI) / 180;
  const cx = 25;
  const cy = 25;
  const radius = 16;
  const x = cx + Math.cos(radians) * radius;
  const y = cy - Math.sin(radians) * radius;

  return (
    <span className="flex flex-col items-center gap-0.5">
      <svg
        width="50" height="50" viewBox="0 0 50 50"
        className={`rounded-full border border-zinc-300 bg-zinc-50 touch-none dark:border-zinc-700 dark:bg-zinc-900 ${disabled ? "pointer-events-none cursor-not-allowed opacity-40" : "cursor-crosshair"}`}
        role="slider" aria-label="Text rotation" aria-valuemin={-360} aria-valuemax={360} aria-valuenow={value}
        onPointerDown={begin} onPointerMove={move} onPointerUp={end} onPointerCancel={end}
        onDoubleClick={disabled ? undefined : onReset}
      >
        <circle cx={cx} cy={cy} r="1.75" className="fill-zinc-400 dark:fill-zinc-500" />
        <line x1={cx} y1={cy} x2={x} y2={y} stroke="currentColor" strokeWidth="2" strokeLinecap="round" className="text-cyan-600 dark:text-cyan-400" />
        <circle cx={x} cy={y} r="3" className="fill-cyan-600 dark:fill-cyan-400" />
      </svg>
      <input
        type="number" min={-360} max={360} step={1} value={value}
        aria-label="Text rotation degrees"
        disabled={disabled}
        onChange={(event) => { if (!disabled) onChange(clamp(Number(event.target.value || 0))); }}
        onDoubleClick={disabled ? undefined : onReset}
        className="w-12 rounded-sm border border-zinc-300 bg-white px-1 py-0.5 text-right font-mono text-[10px] text-zinc-800 dark:border-zinc-700 dark:bg-zinc-900 dark:text-zinc-200"
      />
    </span>
  );
}
