## 🍢 ToFU — API authentication and rate limiting
## vieuxtiful
"""Edge controls for a publicly reachable deployment.

Implemented against Starlette's middleware contract with nothing added to
requirements.txt on purpose. The obvious choice would be slowapi, but
server/requirements.txt is pinned with `==` so the measured numbers in the
technical paper stay reproducible, and a fixed-window counter over a dict is
not the part of this system that needs a dependency.

Both controls are middleware rather than per-route dependencies because the
app declares dozens of routes; a decorator on each is a control that silently
stops covering whatever route someone adds next. Path prefix matching applies
to everything under /api by construction, so a new endpoint is protected the
moment it exists.

Scope, stated plainly: the limiter counts per process in memory. Behind
several replicas each gets its own allowance, and a shared store would be
needed for a true global limit. It stops casual hammering and accidental
runaway clients, which is what a single-host deployment needs; it is not a
defence against a distributed flood. That belongs at the reverse proxy.
"""

from __future__ import annotations

import secrets
import threading
import time
from collections import deque

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse

## Reachable without a key. The health route has to answer for container
## healthchecks and load balancers, and it deliberately returns posture only,
## never the capability inventory -- /api/capabilities enumerates installed
## models and paths, which is reconnaissance, so it stays authenticated.
PUBLIC_PATHS = frozenset({"/api/health"})

## Everything the key protects. /api is the obvious half; the three static
## mounts are the half that was missing, and they are where the user's own
## material lives -- every photograph they uploaded, every render, every
## translation-memory thumbnail. main.py mounts them with StaticFiles OUTSIDE
## /api, so a prefix test against "/api" alone left them world-readable to
## anyone who could guess or observe a path. An asset id is not a secret; it
## travels in logs, in referrers, and in the browser history.
GUARDED_PREFIXES = ("/api", "/outputs", "/uploads", "/tm_thumbs")

## Preflight carries no credentials by design, so requiring one would break
## every browser cross-origin request before the real call is ever made.
_EXEMPT_METHODS = frozenset({"OPTIONS"})

## An <img src> cannot carry an Authorization header -- the browser decides
## what a subresource load sends, and it sends cookies. Guarding the static
## mounts therefore needs a credential the markup can use, so a successful
## key presentation at POST /api/session mints this cookie and the browser
## replays it on every image the app displays. It is HttpOnly (script cannot
## read it back out), SameSite=Lax (a foreign page embedding <img
## src="…/uploads/x.png"> gets no cookie and so gets a 401), and Secure
## whenever the deployment is not plain-http localhost.
##
## Note what this cookie is NOT: it is not a session, and it carries no
## identity. It holds the same shared key the header would have carried, and
## it exists so one credential covers both transports. When per-user accounts
## land, what the cookie PROVES changes; where it is checked does not.
SESSION_COOKIE = "tofu_key"


def is_guarded(path: str) -> bool:
    """Does this path require a key? Public paths never do."""
    if path in PUBLIC_PATHS:
        return False
    return any(path.startswith(prefix) for prefix in GUARDED_PREFIXES)


def _peer_address(request: Request, trusted_proxy_hops: int) -> str:
    """The caller's address, seen past any reverse proxies we control.

    X-Forwarded-For is a client-controlled string: anyone can prepend
    whatever they like to it, so honouring the LEFTMOST entry hands every
    caller an unlimited supply of fresh identities and turns the limiter
    off. What is trustworthy is the tail -- each proxy appends the address
    it actually accepted the connection from, so with N proxies in front of
    us the (N+1)th entry from the right was written by our own edge and
    cannot be forged from outside it.

    Zero hops (the default) ignores the header entirely and uses the socket
    peer, which is correct for a directly-exposed server and is what a
    misconfiguration degrades to: everyone behind the proxy shares one
    bucket, which throttles honest users but never lets an attacker escape
    theirs. That is the right direction to fail in.
    """
    if trusted_proxy_hops > 0:
        chain = [
            part.strip()
            for part in request.headers.get("x-forwarded-for", "").split(",")
            if part.strip()
        ]
        if len(chain) >= trusted_proxy_hops:
            return chain[-trusted_proxy_hops]
    client = request.client
    return client.host if client else "unknown"


def _client_key(request: Request, trusted_proxy_hops: int = 0) -> str:
    """Identify the caller for rate-limiting purposes.

    Address first, key second -- the reverse of what an earlier version did,
    and the reversal matters for the deployment this ships into. The demo
    issues ONE shared key to every user, so bucketing by key put the entire
    user base in a single allowance: two people working at once would
    throttle each other, and the limiter would report that as the system
    behaving correctly. The address separates them.

    The key still participates, appended, so that when per-user keys exist
    two tenants behind one NAT do not share a bucket either.
    """
    address = _peer_address(request, trusted_proxy_hops)
    token = presented_key(request)
    if token:
        return f"ip:{address}|key:{token[:16]}"
    return f"ip:{address}"


def presented_key(request: Request) -> str | None:
    """The key this request carries, by whichever transport it could use.

    Header first, because every fetch the app makes sets one and an explicit
    credential should beat an ambient one. The cookie is the fallback for
    requests the application code never gets to touch -- <img>, <video>, and
    a right-click "open image in new tab".
    """
    header = request.headers.get("authorization", "")
    if header.lower().startswith("bearer "):
        token = header[7:].strip()
        if token:
            return token
    token = request.headers.get("x-api-key", "").strip()
    if token:
        return token
    return request.cookies.get(SESSION_COOKIE) or None


def key_is_valid(candidate: str | None, allowed: frozenset[str]) -> bool:
    """Constant-time membership test.

    A plain `candidate in allowed` compares byte by byte and returns early,
    which leaks the length of the shared prefix through timing. compare_digest
    against every entry costs the same regardless of how much matched.
    """
    if not candidate:
        return False
    matched = False
    for key in allowed:
        if secrets.compare_digest(candidate, key):
            matched = True
    return matched


class APIKeyMiddleware(BaseHTTPMiddleware):
    """Require a valid key on /api routes when any key is configured.

    No configured keys means no enforcement, which is what keeps local
    development and the pytest TestClient working unchanged. Production
    cannot reach that state: config.load_settings() refuses to start without
    keys, so "unenforced" is a development-only condition rather than a
    deployment that quietly forgot.
    """

    def __init__(self, app, api_keys: frozenset[str]):
        super().__init__(app)
        self.api_keys = api_keys

    async def dispatch(self, request: Request, call_next):
        if not self.api_keys:
            return await call_next(request)
        if not is_guarded(request.url.path):
            return await call_next(request)
        if request.method in _EXEMPT_METHODS:
            return await call_next(request)
        if not key_is_valid(presented_key(request), self.api_keys):
            return JSONResponse(
                {"detail": "missing or invalid API key"},
                status_code=401,
                headers={"WWW-Authenticate": "Bearer"},
            )
        return await call_next(request)


class RateLimitMiddleware(BaseHTTPMiddleware):
    """Fixed-window request cap per caller, counted in this process.

    A deque of timestamps per caller rather than a bare counter, so the window
    slides instead of resetting on a wall-clock boundary -- a plain counter
    lets a caller spend the whole allowance at the end of one window and again
    at the start of the next, which is twice the intended rate at exactly the
    moment it matters.
    """

    def __init__(self, app, *, max_requests: int, window_seconds: int,
                 trusted_proxy_hops: int = 0):
        super().__init__(app)
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self.trusted_proxy_hops = trusted_proxy_hops
        self._hits: dict[str, deque] = {}
        self._lock = threading.Lock()
        self._last_sweep = time.monotonic()

    def _sweep(self, now: float) -> None:
        """Drop callers with no recent activity so the dict cannot grow without bound."""
        if now - self._last_sweep < self.window_seconds:
            return
        cutoff = now - self.window_seconds
        for key in [k for k, v in self._hits.items() if not v or v[-1] <= cutoff]:
            del self._hits[key]
        self._last_sweep = now

    def _retry_after(self, key: str, now: float) -> float | None:
        """None when the call is allowed; otherwise seconds until it would be."""
        cutoff = now - self.window_seconds
        with self._lock:
            self._sweep(now)
            bucket = self._hits.setdefault(key, deque())
            while bucket and bucket[0] <= cutoff:
                bucket.popleft()
            if len(bucket) >= self.max_requests:
                return max(0.0, bucket[0] + self.window_seconds - now)
            bucket.append(now)
            return None

    async def dispatch(self, request: Request, call_next):
        ## Deliberately /api only, not GUARDED_PREFIXES. The static mounts
        ## need the KEY (they carry user material) but not the COUNTER: one
        ## screen of the project pantry issues a thumbnail request per
        ## project, so counting served bytes against the same budget as
        ## pipeline calls would throttle ordinary browsing while leaving the
        ## expensive routes exactly as exposed as before. Serving a file is
        ## not what a limiter here is protecting.
        path = request.url.path
        if not path.startswith("/api") or path in PUBLIC_PATHS:
            return await call_next(request)
        if request.method in _EXEMPT_METHODS:
            return await call_next(request)
        caller = _client_key(request, self.trusted_proxy_hops)
        retry_after = self._retry_after(caller, time.monotonic())
        if retry_after is not None:
            return JSONResponse(
                {
                    "detail": (
                        f"rate limit exceeded: {self.max_requests} requests per "
                        f"{self.window_seconds}s"
                    )
                },
                status_code=429,
                headers={"Retry-After": str(max(1, int(retry_after) + 1))},
            )
        return await call_next(request)
