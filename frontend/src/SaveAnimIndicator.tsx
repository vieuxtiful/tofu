import { useEffect, useRef, useState } from "react";

/**
 * "Stinky ToFU" save indicator. Replaces the old "saving… / saved" text span.
 *
 * The ToFU logo block (sans inscription) hops in place while spinning, with
 * three squiggly stink lines rising above it. Driven by the same
 * `saveStatus` state machine the text span used ("idle" | "saving" | "saved").
 *
 * idle  — not rendered.
 * saving — block hops + spins; spin rate escalates 15% per hop for 3 stages,
 *          then stage 3 repeats until "saved". Stink lines wiggle in
 *          bottom→top, staggered (0 / +0.25s / +0.55s), each fading over 0.3s,
 *          and ride along with the hop.
 * saved  — if mid-hop, finishes the hop and lands; holds briefly, then the
 *          whole indicator fades out.
 *
 * Hop height is capped at 35% of the FlipButton to the indicator's left
 * (FlipButton --height is 35px → 12.25px); we use 12px. Tunable via
 * --save-hop-h. The hop + spin are run from a requestAnimationFrame loop so
 * the escalating spin accumulates continuously (no CSS loop seam) and the
 * "land then fade" handoff on "saved" is exact. Stink lines are pure CSS
 * keyframes (staggered delays + fade), which is the natural fit for them.
 */
type Status = "idle" | "saving" | "saved";

const HOP_MS = 900;                         // one hop == ~0.9s
// HOP_H / BASE_DPS / STAGE_MULT are gone with the hop and the spin. HOP_MS
// survives as the cadence the "saved" hand-off is paced against.
const LAND_HOLD_MS = 900;                   // rest after landing on "saved" before fade
const FADE_MS = 300;                        // fade-out duration on "saved"

// Three squiggly stink-line paths (relative coords, x centered on 0).
// Vertical wavy line, height 14, amplitude ~2, two humps. Positioned by the
// parent <g transform="translate(cx,0)"> so the CSS animation transform on the
// path itself isn't clobbered.
const STINK_PATH_D =
  "M 0,14 C -2,11.5 2,9 0,7 C -2,4.5 2,2 0,0";
const STINK_CX = [4.5, 11.5, 18.5];          // centered across ~23px block width
const STINK_DY = [0, -3, 0];                  // middle line sits higher (block shape)
const STINK_DELAYS = ["0s", "0.25s", "0.55s"];

// Preload both block images at module load so the first "saving" state doesn't
// wait on a network fetch — the block would be invisible until the image
// arrives, which can take seconds on a cold cache.
for (const src of ["/save-anim-block.png", "/save-anim-block-dark.png"]) {
  const img = new Image();
  img.src = src;
}

export default function SaveAnimIndicator({ status, theme }: { status: Status; theme: "light" | "dark" }) {
  const [mounted, setMounted] = useState(false);
  const wrapRef = useRef<HTMLDivElement>(null);   // outer: gets opacity fade
  const hopRef = useRef<HTMLDivElement>(null);    // inner: gets hop translateY
  const blockRef = useRef<HTMLImageElement>(null);
  const rafRef = useRef<number | null>(null);
  const startRef = useRef(0);
  const savedAtRef = useRef<number | null>(null);
  const fadeStartRef = useRef<number | null>(null);
  // status is read inside the rAF loop; keep a ref so the loop (set up once on
  // mount) always sees the latest value without restarting and resetting the
  // accumulated spin angle.
  const statusRef = useRef<Status>(status);
  statusRef.current = status;

  // mount on "saving", unmount immediately on "idle" (error path). The
  // "saved" → unmount handoff is driven by the rAF loop after it fades out.
  useEffect(() => {
    if (status === "saving") {
      savedAtRef.current = null;
      fadeStartRef.current = null;
      setMounted(true);
    } else if (status === "idle") {
      setMounted(false);
    }
  }, [status]);

  useEffect(() => {
    if (!mounted) return;
    // Reduced motion is handled in CSS now: the stink lines are the only
    // thing that moves, and their keyframes are the right place to stop.
    startRef.current = performance.now();
    savedAtRef.current = null;
    fadeStartRef.current = null;

    const tick = (now: number) => {
      const wrap = wrapRef.current;
      const hop = hopRef.current;
      const block = blockRef.current;
      if (!wrap || !hop || !block) {
        rafRef.current = requestAnimationFrame(tick);
        return;
      }
      const t = now - startRef.current;

      // `phase` still paces the "saved" hand-off below; it no longer moves
      // anything. The block is STILL now -- only the stink lines animate.
      const phase = (t % HOP_MS) / HOP_MS;

      // "saved" handoff: mark the moment, then wait until we're basically
      // landed (phase near 0) to schedule the fade — so a mid-hop save
      // finishes its arc and lands cleanly before fading.
      const st = statusRef.current;
      if (st === "saved" && savedAtRef.current === null) savedAtRef.current = now;
      if (st === "idle") {
        // safety net: if status flips to idle mid-flight, stop animating.
        setMounted(false);
        return;
      }
      let opacity = 1;
      if (savedAtRef.current !== null) {
        const justLanded = phase < 0.03 || phase > 0.97;
        if (justLanded && fadeStartRef.current === null) {
          fadeStartRef.current = now + LAND_HOLD_MS;
        }
        if (fadeStartRef.current !== null && now >= fadeStartRef.current) {
          const fe = now - fadeStartRef.current;
          opacity = Math.max(0, 1 - fe / FADE_MS);
        }
      }

      // The hop and the spin are gone: the block sits still and stinks. Two
      // motions plus escalating rotation made a routine autosave the loudest
      // thing on screen, and the block is a logo -- spinning it reads as a
      // loading spinner, which is a different claim about how long this takes.
      hop.style.transform = "none";
      wrap.style.opacity = String(opacity);
      block.style.transform = "none";

      if (opacity <= 0) {
        setMounted(false);
        return;
      }
      rafRef.current = requestAnimationFrame(tick);
    };
    rafRef.current = requestAnimationFrame(tick);
    return () => {
      if (rafRef.current) cancelAnimationFrame(rafRef.current);
    };
  }, [mounted]);

  if (!mounted) return null;

  const src = theme === "dark" ? "/save-anim-block-dark.png" : "/save-anim-block.png";

  return (
    <div
      ref={wrapRef}
      className="save-anim-outer"
      style={{ opacity: 1 }}
      aria-label={status === "saving" ? "saving" : "saved"}
      role="status"
    >
      <div
        ref={hopRef}
        className={`save-anim ${status === "saving" ? "saving" : "saved"}`}
      >
        <svg
          className="save-stink"
          width="23"
          height="14"
          viewBox="0 0 23 14"
          fill="none"
          aria-hidden="true"
        >
          {STINK_CX.map((cx, i) => (
            <g key={i} transform={`translate(${cx},${STINK_DY[i]})`}>
              <path
                className="stink-line"
                d={STINK_PATH_D}
                style={{ animationDelay: STINK_DELAYS[i] }}
              />
            </g>
          ))}
        </svg>
        <img ref={blockRef} className="save-block" src={src} alt="" draggable={false} />
      </div>
      <span className="save-label subtext text-xs">
        <span className={`save-label-text saving ${status === "saving" ? "on" : ""} text-amber-500 dark:text-amber-400`}>saving…</span>
        <span className={`save-label-text saved ${status === "saved" ? "on" : ""} text-[#0f2600] dark:text-[#4f9f00]`}>saved</span>
      </span>
    </div>
  );
}
