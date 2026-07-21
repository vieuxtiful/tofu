// 🍢 loader library — adapted from the user-supplied prompt-kit loaders;
// self-contained (no @/lib/utils), keyframes live in index.css.

export type LoaderSize = "xs" | "sm" | "md" | "lg";

function cn(...parts: Array<string | undefined | false>): string {
  return parts.filter(Boolean).join(" ");
}

const SIZE = { xs: "h-3 w-3", sm: "h-4 w-4", md: "h-5 w-5", lg: "h-6 w-6" };

export function CircularLoader({ className, size = "md" }: { className?: string; size?: LoaderSize }) {
  return (
    <div
      className={cn(
        "animate-spin rounded-full border-2 border-current border-t-transparent",
        SIZE[size],
        className
      )}
    >
      <span className="sr-only">Loading</span>
    </div>
  );
}

export function ClassicLoader({ className, size = "md" }: { className?: string; size?: LoaderSize }) {
  const bar = { xs: { h: "4px", w: "1px" }, sm: { h: "6px", w: "1.5px" }, md: { h: "8px", w: "2px" }, lg: { h: "10px", w: "2.5px" } }[size];
  const origin = { xs: "0.5px 7px", sm: "0.75px 10px", md: "1px 12px", lg: "1.25px 14px" }[size];
  const ml = { xs: "-0.5px", sm: "-0.75px", md: "-1px", lg: "-1.25px" }[size];
  return (
    <div className={cn("relative", SIZE[size], className)}>
      <div className="absolute h-full w-full">
        {[...Array(12)].map((_, i) => (
          <div
            key={i}
            className="absolute animate-[spinner-fade_1.2s_linear_infinite] rounded-full bg-current"
            style={{
              top: 0, left: "50%", marginLeft: ml,
              transformOrigin: origin,
              transform: `rotate(${i * 30}deg)`,
              opacity: 0,
              animationDelay: `${i * 0.1}s`,
              height: bar.h, width: bar.w,
            }}
          />
        ))}
      </div>
      <span className="sr-only">Loading</span>
    </div>
  );
}

export function DotsLoader({ className, size = "md" }: { className?: string; size?: LoaderSize }) {
  const dot = { xs: "h-1 w-1", sm: "h-1.5 w-1.5", md: "h-2 w-2", lg: "h-2.5 w-2.5" }[size];
  return (
    <div className={cn("flex items-center space-x-1", SIZE[size].split(" ")[0], className)}>
      {[...Array(3)].map((_, i) => (
        <div
          key={i}
          className={cn("animate-[bounce-dots_1.4s_ease-in-out_infinite] rounded-full bg-current", dot)}
          style={{ animationDelay: `${i * 160}ms` }}
        />
      ))}
      <span className="sr-only">Loading</span>
    </div>
  );
}

export function PulseLoader({ className, size = "md" }: { className?: string; size?: LoaderSize }) {
  return (
    <div className={cn("relative", SIZE[size], className)}>
      <div className="absolute inset-0 animate-[thin-pulse_1.5s_ease-in-out_infinite] rounded-full border-2 border-current" />
      <span className="sr-only">Loading</span>
    </div>
  );
}

export function TerminalLoader({ className, size = "md" }: { className?: string; size?: LoaderSize }) {
  const cursor = { xs: "h-2 w-1", sm: "h-3 w-1.5", md: "h-4 w-2", lg: "h-5 w-2.5" }[size];
  const text = { xs: "text-[10px]", sm: "text-xs", md: "text-sm", lg: "text-base" }[size];
  const container = { xs: "h-3", sm: "h-4", md: "h-5", lg: "h-6" }[size];
  return (
    <div className={cn("flex items-center space-x-1", container, className)}>
      <span className={cn("font-mono text-current", text)}>{">"}</span>
      <div className={cn("bg-current animate-[blink_1s_step-end_infinite]", cursor)} />
      <span className="sr-only">Loading</span>
    </div>
  );
}

export function SquareLoader({ className, size = "md" }: { className?: string; size?: LoaderSize }) {
  const sq = { xs: 3, sm: 4, md: 5, lg: 6 }[size];
  const gap = 1;
  const step = sq + gap;
  const span = 3 * sq + 2 * gap;
  const offsets = [
    { mt: -1, ml: -1, d: 0 },
    { mt: -1, ml: 0, d: 75 },
    { mt: -1, ml: 1, d: 150 },
    { mt: 0, ml: -1, d: 225 },
    { mt: 0, ml: 0, d: 300 },
    { mt: 0, ml: 1, d: 375 },
    { mt: 1, ml: -1, d: 450 },
    { mt: 1, ml: 0, d: 525 },
    { mt: 1, ml: 1, d: 600 },
  ];
  return (
    <div className={cn("relative shrink-0", className)} style={{ width: span, height: span }}>
      {offsets.map((o, i) => (
        <div
          key={i}
          className="square-loader-tile absolute bg-current"
          style={{
            width: sq,
            height: sq,
            top: "50%",
            left: "50%",
            marginTop: o.mt * step - sq / 2,
            marginLeft: o.ml * step - sq / 2,
            animationDelay: `${o.d}ms`,
          }}
        />
      ))}
      <span className="sr-only">Loading</span>
    </div>
  );
}

export function TextShimmerLoader({ text = "Loading", className, size = "md" }: {
  text?: string; className?: string; size?: LoaderSize;
}) {
  const font = { xs: "text-[10px]", sm: "text-xs", md: "text-sm", lg: "text-base" }[size];
  return (
    <div
      className={cn(
        "bg-clip-text font-medium text-transparent",
        "bg-[linear-gradient(90deg,rgba(113,113,122,0.75)_40%,#e4e4e7_60%,rgba(113,113,122,0.75)_80%)]",
        "bg-[length:200%_auto] animate-[shimmer-text_4s_infinite_linear]",
        font, className
      )}
    >
      {text}
    </div>
  );
}
