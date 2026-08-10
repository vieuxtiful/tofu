## 🍢 ToFU — runtime configuration
## vieuxtiful
"""One place that reads the environment, so a deployment is auditable.

Before this module the server's operational posture was spread across
literals in main.py and env vars set only in docker-compose.yml, and the two
disagreed: TOFU_FRONTEND_URL was passed to the container but CORS was
hardcoded to http://localhost:5173, so the shipped image trusted an origin
that did not exist in the deployment and refused the one that did.

The rule here is that development stays convenient and production refuses to
start when it is not configured. A missing API key in production is not a
warning that scrolls past in a log -- it raises, because the alternative is
an open pipeline that spends CPU and disk for anyone who finds the port.
That is the same "fail loudly rather than degrade silently" stance the OCR
backends take, applied to the deployment instead of the model stack.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

## Deliberately not 100 GB, which is what the old inline default was. An
## upload cap is a disk-exhaustion control, and 100 GB per request is not a
## control. 2 GiB clears the sample images and short clips the app is built
## around; a host that genuinely ingests feature-length video raises this
## knowingly rather than inheriting it.
DEFAULT_MAX_UPLOAD_BYTES = 2 * 1024**3

DEV_ORIGINS = ("http://localhost:5173", "http://localhost:3000")


class ConfigError(RuntimeError):
    """Raised when production configuration is missing or unsafe."""


def _split(raw: str) -> list[str]:
    return [item.strip() for item in raw.split(",") if item.strip()]


def _int_env(name: str, default: int) -> int:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    try:
        value = int(raw)
    except ValueError:
        raise ConfigError(f"{name} must be an integer, got {raw!r}")
    if value <= 0:
        raise ConfigError(f"{name} must be positive, got {value}")
    return value


@dataclass(frozen=True)
class Settings:
    environment: str
    api_keys: frozenset[str]
    cors_origins: tuple[str, ...]
    max_upload_bytes: int
    rate_limit_requests: int
    rate_limit_window_seconds: int
    data_dir: Path
    frontend_url: str
    ngram_dir: str | None = None
    trusted_proxy_hops: int = 0

    @property
    def is_production(self) -> bool:
        return self.environment == "production"

    @property
    def cookie_secure(self) -> bool:
        """Whether the session cookie may only travel over TLS.

        Tied to the environment rather than to a knob of its own: production
        terminates TLS at the reverse proxy, and development is plain http on
        localhost where a Secure cookie would simply never be stored -- which
        looks exactly like a broken login.
        """
        return self.is_production

    @property
    def auth_enabled(self) -> bool:
        return bool(self.api_keys)

    @property
    def rate_limit_enabled(self) -> bool:
        return self.rate_limit_requests > 0

    def public_summary(self) -> dict:
        """Non-secret posture, safe to expose on an unauthenticated health route."""
        return {
            "environment": self.environment,
            "auth_enabled": self.auth_enabled,
            "rate_limit_enabled": self.rate_limit_enabled,
            "max_upload_bytes": self.max_upload_bytes,
        }


def load_settings() -> Settings:
    environment = (os.environ.get("TOFU_ENV") or "development").strip().lower()
    if environment not in {"development", "production"}:
        raise ConfigError(
            f"TOFU_ENV must be 'development' or 'production', got {environment!r}"
        )
    production = environment == "production"

    api_keys = frozenset(_split(os.environ.get("TOFU_API_KEYS", "")))
    # A short key is worse than no key: it invites brute force while implying
    # the endpoint is protected.
    for key in api_keys:
        if len(key) < 24:
            raise ConfigError(
                "each TOFU_API_KEYS entry must be at least 24 characters; "
                "generate one with: python -c \"import secrets;print(secrets.token_urlsafe(32))\""
            )

    origins = tuple(_split(os.environ.get("TOFU_CORS_ORIGINS", "")))
    if not origins and not production:
        origins = DEV_ORIGINS

    max_upload_bytes = _int_env("TOFU_MAX_UPLOAD_BYTES", DEFAULT_MAX_UPLOAD_BYTES)
    # 0 disables the limiter; anything else is a positive count, so this one
    # cannot go through _int_env's positive-only check.
    raw_limit = os.environ.get("TOFU_RATE_LIMIT_REQUESTS", "").strip()
    try:
        rate_limit_requests = int(raw_limit) if raw_limit else (120 if production else 0)
    except ValueError:
        raise ConfigError(f"TOFU_RATE_LIMIT_REQUESTS must be an integer, got {raw_limit!r}")
    if rate_limit_requests < 0:
        raise ConfigError("TOFU_RATE_LIMIT_REQUESTS must be zero or positive")
    rate_limit_window_seconds = _int_env("TOFU_RATE_LIMIT_WINDOW_SECONDS", 60)
    ## How many reverse proxies of our own sit in front of this process. The
    ## limiter counts back from the RIGHT of X-Forwarded-For by this many
    ## entries, so the value has to match the deployment exactly: one for the
    ## single Caddy in docker-compose.prod.yml, zero when the port is exposed
    ## directly. Too high reads an attacker-supplied entry as the client
    ## address; too low collapses everyone onto the proxy's own address.
    raw_hops = os.environ.get("TOFU_TRUSTED_PROXY_HOPS", "").strip()
    try:
        trusted_proxy_hops = int(raw_hops) if raw_hops else 0
    except ValueError:
        raise ConfigError(f"TOFU_TRUSTED_PROXY_HOPS must be an integer, got {raw_hops!r}")
    if trusted_proxy_hops < 0:
        raise ConfigError("TOFU_TRUSTED_PROXY_HOPS must be zero or positive")

    data_dir = Path(os.environ.get("TOFU_DATA_DIR") or (ROOT / "server")).resolve()
    frontend_url = (
        os.environ.get("TOFU_FRONTEND_URL") or "http://localhost:5173"
    ).strip()
    ngram_dir = (os.environ.get("TOFU_NGRAM_DIR") or "").strip() or None

    if production:
        problems = []
        if not api_keys:
            problems.append(
                "TOFU_API_KEYS is empty; the API would accept unauthenticated requests"
            )
        if not origins:
            problems.append(
                "TOFU_CORS_ORIGINS is empty; set the exact frontend origin(s)"
            )
        if "*" in origins:
            problems.append(
                "TOFU_CORS_ORIGINS contains '*', which cannot be combined with credentials"
            )
        if problems:
            raise ConfigError(
                "refusing to start in production:\n  - " + "\n  - ".join(problems)
            )

    return Settings(
        environment=environment,
        api_keys=api_keys,
        cors_origins=origins,
        max_upload_bytes=max_upload_bytes,
        rate_limit_requests=rate_limit_requests,
        rate_limit_window_seconds=rate_limit_window_seconds,
        data_dir=data_dir,
        frontend_url=frontend_url,
        ngram_dir=ngram_dir,
        trusted_proxy_hops=trusted_proxy_hops,
    )


settings = load_settings()
