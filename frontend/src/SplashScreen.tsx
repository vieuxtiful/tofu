import { useEffect, useRef, useState } from "react";

const TITLE = "ToFU";
const VERSION = "v0.1.0";

const HOVER_ANIM_MS = 800;  // text box auto-animates to hover state
const TYPE_START_MS = 250;   // cursor blinks alone before typing begins (after hover anim)
const TYPE_INTERVAL_MS = 500; // per-character typewriter cadence
const VERSION_DELAY_MS = 1300; // after the title finishes
const DONE_DELAY_MS = 2000;   // hold the full screen, then hand off

/** startup splash: a 3D text box (matching Get Capture / Translate style)
 * auto-animates from standard to hover state, then a blinking cursor types
 * out "ToFU" character by character inside it. the version then rises in
 * beneath it. theme-aware (light/dark). click skips. */
export default function SplashScreen({ onDone }: { onDone: () => void }) {
  const [typed, setTyped] = useState(0);
  const [stage, setStage] = useState(0); // 0 typing · 1 version
  const timers = useRef<ReturnType<typeof setTimeout>[]>([]);

  useEffect(() => {
    const later = (fn: () => void, ms: number) => {
      timers.current.push(setTimeout(fn, ms));
    };
    const typeStart = HOVER_ANIM_MS + TYPE_START_MS;
    for (let i = 1; i <= TITLE.length; i++) {
      later(() => setTyped(i), typeStart + i * TYPE_INTERVAL_MS);
    }
    const typedDone = typeStart + TITLE.length * TYPE_INTERVAL_MS;
    later(() => setStage(1), typedDone + VERSION_DELAY_MS);
    later(onDone, typedDone + VERSION_DELAY_MS + DONE_DELAY_MS);
    return () => timers.current.forEach(clearTimeout);
  }, [onDone]);

  const skip = () => {
    timers.current.forEach(clearTimeout);
    onDone();
  };

  return (
    <div
      className="fixed inset-0 z-[100] flex cursor-pointer flex-col items-center justify-center gap-4 bg-zinc-100 font-mono dark:bg-zinc-950"
      onClick={skip}
      title="Click to skip"
    >
      {/* 3D text box — auto-animates to hover state, typing plays inside */}
      <div className="splash-textbox">
        <span className="splash-chevron">{">"}</span>
        <span>{TITLE.slice(0, typed)}</span>
        <span className="splash-cursor" />
      </div>

      {/* version — same font as ToFU but smaller, positioned just below textbox */}
      <p
        className={`splash-line text-zinc-900 dark:text-zinc-400 ${stage >= 1 ? "on" : ""}`}
        style={{ marginTop: "0.25rem", fontFamily: '"Courier New", monospace', fontSize: "0.875rem", fontWeight: 700 }}
      >
        {VERSION}
      </p>
    </div>
  );
}
