// 🍢 KeyGate — the door key, asked for once
//
// A deployment with TOFU_API_KEYS configured rejects every /api call that
// arrives without one, and until this existed the frontend had no way to
// present a key at all: the app would mount, fire its first request, and show
// a wall of "401" toasts with nothing the user could do about it.
//
// Deliberately mounted OUTSIDE the screen state machine in App.tsx, ahead of
// even the splash. Everything downstream — splash, title, pantry — makes API
// calls to decide what to render, so a gate placed anywhere later would be
// answering for requests that had already failed.
//
// This is the shared-key demo door, not a login: the key identifies the
// deployment, not the person holding it. When per-user accounts land, the
// component this becomes asks for two fields instead of one and posts to the
// same /api/session route — which is why the exchange lives behind
// openSession() rather than being spelled out here.

import { useEffect, useState } from "react";
import { KeyRound, Loader2 } from "lucide-react";
import { authRequired, getApiKey, openSession, setApiKey, setUnauthorizedHandler } from "./api";
import { logoSrc, storedTheme } from "./theme";

type GateState = "checking" | "required" | "open";

interface KeyGateProps {
  /** rendered once the deployment is satisfied — unauthenticated or keyed */
  children: React.ReactNode;
}

export default function KeyGate({ children }: KeyGateProps) {
  const theme = storedTheme();
  const [state, setState] = useState<GateState>("checking");
  const [key, setKey] = useState("");
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);

  // On mount: ask the deployment whether it wants a key at all, then try the
  // one we already hold. A stored key that still works must not re-prompt --
  // the gate should be invisible on every visit after the first.
  useEffect(() => {
    let cancelled = false;
    (async () => {
      const required = await authRequired();
      if (cancelled) return;
      if (!required) {
        setState("open");
        return;
      }
      const stored = getApiKey();
      if (stored && (await openSession(stored))) {
        if (!cancelled) setState("open");
        return;
      }
      if (!cancelled) setState("required");
    })();
    return () => {
      cancelled = true;
    };
  }, []);

  // A key can stop working while the app is open — rotated on the server, or
  // the deployment restarted with a different one. api.ts clears the stored
  // key on any 401 and calls this, which puts the door back in front of the
  // user instead of leaving them clicking into failures.
  useEffect(() => {
    setUnauthorizedHandler(() => {
      setKey("");
      setError("Your key is no longer valid.");
      setState("required");
    });
    return () => setUnauthorizedHandler(null);
  }, []);

  async function submit(event: React.FormEvent) {
    event.preventDefault();
    const candidate = key.trim();
    if (!candidate || busy) return;
    setBusy(true);
    setError(null);
    try {
      if (await openSession(candidate)) {
        setApiKey(candidate);
        setState("open");
      } else {
        setError("That key was not accepted.");
      }
    } catch {
      setError("Could not reach the ToFU backend.");
    } finally {
      setBusy(false);
    }
  }

  // A blank screen rather than a spinner: the check is one call against
  // /api/health and resolves faster than a spinner reads as anything but a
  // flicker. The splash animation starts immediately afterwards.
  if (state === "checking") return <div className="fixed inset-0 bg-white dark:bg-zinc-950" />;
  if (state === "open") return <>{children}</>;

  return (
    <div className="fixed inset-0 z-500 flex items-center justify-center bg-white p-6 dark:bg-zinc-950">
      <form
        onSubmit={submit}
        className="bezier-card flex w-full max-w-sm flex-col gap-4 rounded-xl bg-white p-6 dark:bg-zinc-900"
      >
        <div className="flex items-center gap-3">
          <img src={logoSrc(theme)} alt="ToFU" className="h-10 w-auto" />
          <div>
            <h1 className="text-lg font-semibold text-zinc-800 dark:text-zinc-200">ToFU</h1>
            <p className="subtext text-xs text-zinc-500">this deployment needs a key.</p>
          </div>
        </div>

        <label className="flex flex-col gap-1.5">
          <span className="subtext text-xs text-zinc-500">access key</span>
          <div className="flex items-center gap-2 rounded-md border border-zinc-300 px-2 dark:border-zinc-700">
            <KeyRound size={14} className="shrink-0 text-zinc-500" />
            <input
              type="password"
              autoFocus
              autoComplete="current-password"
              value={key}
              onChange={(e) => {
                setKey(e.target.value);
                setError(null);
              }}
              className="w-full bg-transparent py-2 text-sm text-zinc-800 outline-none dark:text-zinc-200"
            />
          </div>
        </label>

        {error && (
          <p className="subtext rounded-lg border border-red-300 bg-red-100 px-3 py-2 text-xs text-red-700 dark:border-red-800 dark:bg-red-950/50 dark:text-red-300">
            {error}
          </p>
        )}

        <button
          type="submit"
          disabled={!key.trim() || busy}
          className="press-button flex items-center justify-center gap-2 rounded-lg bg-zinc-800 px-4 py-2 text-sm font-medium text-white transition disabled:cursor-not-allowed disabled:opacity-50 dark:bg-zinc-200 dark:text-zinc-900"
        >
          {busy && <Loader2 size={14} className="animate-spin" />}
          {busy ? "checking" : "enter"}
        </button>
      </form>
    </div>
  );
}
