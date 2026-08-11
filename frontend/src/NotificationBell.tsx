import { useEffect, useRef, useState } from "react";
import { X } from "lucide-react";
import { RiNotification3Fill, RiNotification3Line } from "react-icons/ri";
import { isStale, timeAgo, type Notification } from "./ToastSystem";

interface NotificationBellProps {
  notifications: Notification[];
  onClear: () => void;
  onDismiss: (id: string) => void;
}

const TYPE_COLOR: Record<string, string> = {
  info: "#0AAFFF",
  success: "#2DCA73",
  warning: "#FFC212",
  error: "#ED63D2",
};

const TYPE_WORD: Record<string, string> = {
  info: "Notice",
  success: "Success",
  warning: "Attention",
  error: "Attention",
};

export default function NotificationBell({ notifications, onClear, onDismiss }: NotificationBellProps) {
  const [open, setOpen] = useState(false);
  const [tick, setTick] = useState(0);
  const ref = useRef<HTMLDivElement>(null);

  const visible = notifications.filter((n) => !isStale(n.timestamp));
  const count = visible.length;
  const allLeaving = count > 0 && visible.every((n) => n.leaving);

  useEffect(() => {
    if (!open) return;
    const onDown = (e: MouseEvent) => {
      if (ref.current && !ref.current.contains(e.target as Node)) {
        setOpen(false);
      }
    };
    document.addEventListener("mousedown", onDown);
    return () => document.removeEventListener("mousedown", onDown);
  }, [open]);

  useEffect(() => {
    const interval = setInterval(() => setTick((t) => t + 1), 30000);
    return () => clearInterval(interval);
  }, []);

  void tick;

  return (
    <div className="indicator relative" ref={ref}>
      {count > 0 && (
        <span className="indicator-item badge badge-secondary absolute -right-1.5 -top-1.5 z-10 inline-flex min-w-4 items-center justify-center rounded-full bg-cyan-600 px-1 text-[10px] font-semibold leading-4 text-white">
          {count > 99 ? "99+" : count}
        </span>
      )}
      <button
        onClick={() => setOpen((v) => !v)}
        className="btn bezier-card relative flex items-center rounded-lg bg-white/60 px-3 py-2 text-sm text-zinc-700 transition hover:bg-zinc-100 dark:bg-zinc-900/60 dark:text-zinc-300 dark:hover:bg-zinc-800"
        title="notifications"
      >
        {count > 0 ? (
          <RiNotification3Fill size={16} className="text-cyan-600 dark:text-cyan-400" />
        ) : (
          <RiNotification3Line size={16} />
        )}
      </button>
      <div
        className={`dropdown-morph bezier-card absolute right-0 top-full z-200 mt-2 overflow-hidden rounded-lg border border-zinc-300 bg-white p-1.5 dark:border-zinc-700 dark:bg-zinc-900${open ? " expanded" : ""} w-80`}
        style={open ? { boxShadow: "4px 4px 0 var(--bc-shadow), 8px 8px 16px rgba(0,0,0,0.18)" } : undefined}
      >
        <div className="max-h-80 overflow-y-auto">
          {count === 0 ? (
            <p className="subtext px-3 py-5 text-center text-xs text-zinc-500 dark:text-zinc-600 notif-empty">
              no notifications
            </p>
          ) : (
            visible.map((n) => (
              <div
                key={n.id}
                className={`notif-item${n.leaving && visible.length > 1 ? " leaving" : ""}${allLeaving ? " clearing" : ""} flex items-start gap-2 rounded-md px-3 py-2 transition hover:bg-zinc-100 dark:hover:bg-zinc-800`}
              >
                <span
                  className="mt-0.5 h-2 w-2 shrink-0 rounded-full"
                  style={{ background: TYPE_COLOR[n.type] ?? "#0AAFFF" }}
                />
                <div className="min-w-0 flex-1">
                  <p className="subtext text-[8px] uppercase tracking-wider text-zinc-500 dark:text-zinc-600">
                    {TYPE_WORD[n.type] ?? "Notice"}
                  </p>
                  <p className="text-xs text-zinc-700 dark:text-zinc-300">{n.message}</p>
                  <p className="subtext mt-0.5 text-[8px] text-zinc-500 dark:text-zinc-600">
                    {timeAgo(n.timestamp)}
                  </p>
                </div>
                <button
                  onClick={() => onDismiss(n.id)}
                  className="ml-auto shrink-0 rounded-sm p-0.5 text-zinc-400 transition hover:bg-zinc-200 hover:text-zinc-600 dark:text-zinc-500 dark:hover:bg-zinc-700 dark:hover:text-zinc-300"
                  title="dismiss"
                >
                  <X size={12} />
                </button>
              </div>
            ))
          )}
        </div>
        {count > 0 && !allLeaving && (
          <button
            onClick={onClear}
            className="subtext mt-1 w-full rounded-md px-3 py-1.5 text-center text-xs text-zinc-500 transition hover:bg-zinc-100 dark:text-zinc-400 dark:hover:bg-zinc-800"
          >
            clear
          </button>
        )}
      </div>
    </div>
  );
}
