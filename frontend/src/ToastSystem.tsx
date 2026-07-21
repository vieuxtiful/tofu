import { useCallback, useRef, useState } from "react";
import { MailOpen } from "lucide-react";

export type ToastType = "info" | "success" | "warning" | "error";

export interface Notification {
  id: string;
  type: ToastType;
  message: string;
  timestamp: number;
  leaving?: boolean;
}

export interface ToastAction {
  label: string;
  onClick: () => void;
  dismiss?: boolean;
}

export interface Toast {
  id: string;
  type: ToastType;
  message: string;
  dismissible?: boolean;
  leaving?: boolean;
  /** optional actions rendered in the card footer (e.g. yes/no confirm) */
  actions?: ToastAction[];
  /** @deprecated — use actions[] instead */
  action?: ToastAction;
}

interface ToastSystemProps {
  toasts: Toast[];
  onDismiss: (id: string) => void;
}

// card wordmark + title-icon color per type (user's card-toast spec)
const META: Record<ToastType, { word: string; color: string }> = {
  info: { word: "Notice", color: "#0AAFFF" },
  success: { word: "Success", color: "#2DCA73" },
  warning: { word: "Attention", color: "#FFC212" },
  error: { word: "Attention", color: "#ED63D2" },
};

function ToastCard({ toast, onDismiss }: { toast: Toast; onDismiss: (id: string) => void }) {
  const [leaving, setLeaving] = useState(false);
  const [flicker, setFlicker] = useState(false);
  const meta = META[toast.type];

  const markRead = () => {
    setFlicker(true);
    setLeaving(true);
    setTimeout(() => onDismiss(toast.id), 260);
  };

  return (
    <div className={`toast-card${leaving || toast.leaving ? " leaving" : ""}`} role="status">
      <div className="tc-header">
        <div className="tc-title">
          <span className="tc-title-icon" style={{ background: meta.color }} />
          <span>{meta.word}</span>
        </div>
      </div>
      <div className="tc-body">{toast.message}</div>
      <div className="tc-actions">
        {(toast.actions ?? (toast.action ? [toast.action] : [])).map((act, i) => (
          <button
            key={i}
            className="tc-action"
            onClick={() => {
              act.onClick();
              if (act.dismiss !== false) markRead();
            }}
          >
            {act.label}
          </button>
        ))}
        {toast.dismissible !== false && (
          <button className={`tc-action${flicker ? " flicker" : ""}`} onClick={markRead}>
            <MailOpen size={13} />
            mark as read
          </button>
        )}
      </div>
    </div>
  );
}

export default function ToastSystem({ toasts, onDismiss }: ToastSystemProps) {
  return (
    <div className="fixed bottom-4 right-4 z-50 flex max-w-sm flex-col gap-2">
      {toasts.map((t) => (
        <ToastCard key={t.id} toast={t} onDismiss={onDismiss} />
      ))}
    </div>
  );
}

let toastId = 0;
export function useToasts() {
  const [toasts, setToasts] = useState<Toast[]>([]);
  const timers = useRef<Map<string, ReturnType<typeof setTimeout>>>(new Map());

  const dismissToast = useCallback((id: string) => {
    const timer = timers.current.get(id);
    if (timer) {
      clearTimeout(timer);
      timers.current.delete(id);
    }
    setToasts((prev) => prev.map((t) => t.id === id ? { ...t, leaving: true } : t));
    setTimeout(() => {
      setToasts((prev) => prev.filter((t) => t.id !== id));
    }, 250);
  }, []);

  const addToast = useCallback(
    (type: ToastType, message: string, autoDismiss = true, action?: ToastAction, actions?: ToastAction[]) => {
      const id = `toast-${++toastId}`;
      setToasts((prev) => [...prev, { id, type, message, action, actions }]);
      const hasActions = !!action || !!actions?.length;
      // warnings/errors and actionable toasts persist until marked read
      if (autoDismiss && type !== "warning" && type !== "error" && !hasActions) {
        timers.current.set(
          id,
          setTimeout(() => dismissToast(id), 5000)
        );
      }
      return id;
    },
    [dismissToast]
  );

  return { toasts, addToast, dismissToast };
}

let notifId = 0;
export function useNotifications() {
  const [notifications, setNotifications] = useState<Notification[]>([]);

  const addNotification = useCallback((type: ToastType, message: string, id?: string) => {
    const nid = id ?? `notif-${++notifId}`;
    setNotifications((prev) => [{ id: nid, type, message, timestamp: Date.now() }, ...prev]);
  }, []);

  const clearNotifications = useCallback(() => {
    setNotifications((prev) => prev.map((n) => ({ ...n, leaving: true })));
    setTimeout(() => setNotifications([]), 400);
  }, []);

  const dismissNotification = useCallback((id: string) => {
    setNotifications((prev) => prev.map((n) => n.id === id ? { ...n, leaving: true } : n));
    setTimeout(() => setNotifications((prev) => prev.filter((n) => n.id !== id)), 400);
  }, []);

  return { notifications, addNotification, clearNotifications, dismissNotification };
}

export function timeAgo(ts: number): string {
  const diff = Date.now() - ts;
  const sec = Math.floor(diff / 1000);
  if (sec < 10) return "moments ago";
  if (sec < 60) return "less than one minute ago";
  const min = Math.floor(sec / 60);
  if (min === 1) return "1 minute ago";
  if (min < 60) return `${min} minutes ago`;
  const hr = Math.floor(min / 60);
  if (hr === 1) return "1 hour ago";
  if (hr < 24) return `${hr} hours ago`;
  return "";
}

export function isStale(ts: number): boolean {
  return Date.now() - ts >= 24 * 60 * 60 * 1000;
}
