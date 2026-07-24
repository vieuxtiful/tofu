import { useLayoutEffect, useRef, useState } from "react";
import { ArrowUpFromLine, VectorSquare, Flag } from "lucide-react";
import { LiaLanguageSolid } from "react-icons/lia";
import { SiOpentofu } from "react-icons/si";
import { PiBoundingBoxFill } from "react-icons/pi";
import { RiCheckboxFill } from "react-icons/ri";
import { TbPhotoScan } from "react-icons/tb";
import type { Theme } from "./theme";

export type Step = 0 | 1 | 2 | 3 | 4;

interface StepperProps {
  current: Step;
  onStep: (step: Step) => void;
  canCapture: boolean;    // asset uploaded
  canTranslate: boolean;  // ≥1 region captured
  canRender: boolean;     // ≥1 region translated
  canVerify: boolean;     // a render has completed
  theme: Theme;
  flagStep?: Step | null;
}

// the TMS round-trip lifecycle: capture strings → export/translate/import → render → verify
const STEPS = [
  { id: 0 as Step, label: "Upload", icon: ArrowUpFromLine },
  { id: 1 as Step, label: "Capture", icon: VectorSquare, iconDark: PiBoundingBoxFill },
  { id: 2 as Step, label: "Translate", icon: LiaLanguageSolid },
  { id: 3 as Step, label: "Render", icon: SiOpentofu },
  { id: 4 as Step, label: "Verify", icon: TbPhotoScan },
];

export default function Stepper({ current, onStep, canCapture, canTranslate, canRender, canVerify, theme, flagStep }: StepperProps) {
  const containerRef = useRef<HTMLDivElement>(null);
  const [overlay, setOverlay] = useState({ left: 0, width: 0, top: 0, height: 0 });

  useLayoutEffect(() => {
    const container = containerRef.current;
    if (!container) return;
    const btn = container.querySelector<HTMLButtonElement>(`[data-step="${current}"]`);
    if (btn) {
      setOverlay({
        left: btn.offsetLeft,
        width: btn.offsetWidth,
        top: btn.offsetTop,
        height: btn.offsetHeight,
      });
    }
  }, [current]);

  const disabled = (id: Step): boolean =>
    (id === 1 && !canCapture) || (id === 2 && !canTranslate) || (id === 3 && !canRender) || (id === 4 && !canVerify);

  const disabledTooltip = (id: Step): string | undefined => {
    if (id === 1 && !canCapture) return "upload an asset first";
    if (id === 2 && !canTranslate) return "capture at least one region first";
    if (id === 3 && !canRender) return "translate at least one region first";
    if (id === 4 && !canVerify) return "complete a render first";
    return undefined;
  };

  return (
    <div ref={containerRef} className="bezier-card relative flex items-center justify-center gap-1 rounded-lg bg-white/60 p-1 dark:bg-zinc-900/60">
      <div
        className="absolute rounded-md bg-cyan-600"
        style={{ left: overlay.left, width: overlay.width, top: overlay.top, height: overlay.height, transition: "left 0.3s ease, width 0.3s ease", zIndex: 0, pointerEvents: "none" }}
      />
      {STEPS.map((step, idx) => {
        const Icon = step.icon;
        const isDone = current > step.id;
        const isCurrent = current === step.id;
        const isDisabled = disabled(step.id);
        const tooltip = disabledTooltip(step.id);
        return (
          <div key={step.id} className="flex items-center">
            <button
              data-step={step.id}
              onClick={() => !isDisabled && onStep(step.id)}
              disabled={isDisabled}
              title={tooltip}
              style={{ position: "relative", zIndex: 1 }}
              className={`flex flex-col items-center justify-center gap-1 rounded-md px-4 py-2 text-sm font-medium transition ${
                isCurrent
                  ? "text-white"
                  : isDone
                  ? "text-[#0f2600] hover:bg-zinc-200 dark:text-[#4f9f00] dark:hover:bg-zinc-800"
                  : "text-zinc-500"
              } ${isDisabled ? "opacity-40 cursor-not-allowed hover:bg-zinc-100 dark:hover:bg-zinc-800" : "cursor-pointer"}`}
            >
              {isDone ? (
                <span className="flex h-5 w-5 items-center justify-center">
                  <RiCheckboxFill size={18} />
                </span>
              ) : (
                <span className="flex h-5 w-5 items-center justify-center">
                  {step.iconDark && theme === "dark" ? (
                    <step.iconDark size={20} style={{ width: 20, height: 20 }} />
                  ) : (
                    <Icon size={16} style={{ width: 16, height: 16 }} />
                  )}
                </span>
              )}
              {step.label}
              {flagStep === step.id && (
                <span className="absolute -right-1 -top-1 flex h-4 w-4 items-center justify-center rounded-full bg-amber-500 text-white shadow-xs" title="target language changed — translations need review">
                  <Flag size={10} />
                </span>
              )}
            </button>
            {idx < STEPS.length - 1 && (
              <div className={`mx-1 h-4 w-px ${isDone ? "bg-[#0f2600] dark:bg-[#4f9f00]" : "bg-zinc-300 dark:bg-zinc-700"}`} />
            )}
          </div>
        );
      })}
    </div>
  );
}
