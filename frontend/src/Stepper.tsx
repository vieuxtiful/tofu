import { ArrowUpFromLine, ShieldCheck, VectorSquare } from "lucide-react";
import { LiaLanguageSolid } from "react-icons/lia";
import { SiOpentofu } from "react-icons/si";
import { PiBoundingBoxFill } from "react-icons/pi";
import { RiCheckboxFill } from "react-icons/ri";
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
}

// the TMS round-trip lifecycle: capture strings → export/translate/import → render → verify
const STEPS = [
  { id: 0 as Step, label: "Upload", icon: ArrowUpFromLine },
  { id: 1 as Step, label: "Capture", icon: VectorSquare, iconDark: PiBoundingBoxFill },
  { id: 2 as Step, label: "Translate", icon: LiaLanguageSolid },
  { id: 3 as Step, label: "Render", icon: SiOpentofu },
  { id: 4 as Step, label: "Verify", icon: ShieldCheck },
];

export default function Stepper({ current, onStep, canCapture, canTranslate, canRender, canVerify, theme }: StepperProps) {
  const disabled = (id: Step): boolean =>
    (id === 1 && !canCapture) || (id === 2 && !canTranslate) || (id === 3 && !canRender) || (id === 4 && !canVerify);

  return (
    <div className="bezier-card flex items-center justify-center gap-1 rounded-lg bg-white/60 p-1 dark:bg-zinc-900/60">
      {STEPS.map((step, idx) => {
        const Icon = step.icon;
        const isDone = current > step.id;
        const isCurrent = current === step.id;
        const isDisabled = disabled(step.id);
        return (
          <div key={step.id} className="flex items-center">
            <button
              onClick={() => !isDisabled && onStep(step.id)}
              disabled={isDisabled}
              className={`flex flex-col items-center justify-center gap-1 rounded-md px-4 py-2 text-sm font-medium transition ${
                isCurrent
                  ? "bg-cyan-600 text-white"
                  : isDone
                  ? "text-[#0f2600] hover:bg-zinc-200 dark:text-[#4f9f00] dark:hover:bg-zinc-800"
                  : "text-zinc-500"
              } ${isDisabled ? "opacity-30 cursor-not-allowed" : "cursor-pointer"}`}
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
