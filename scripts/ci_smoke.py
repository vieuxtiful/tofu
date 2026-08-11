"""What CI can honestly check when the test suite is deliberately local-only.

`tests/` is gitignored by design -- the suite is not published to the remote.
CI previously ran `pytest tests/ -q` anyway, against a path that cannot exist
on a runner. That job was not testing the project; it was testing whether an
empty directory could fail, and the answer shaped nobody's confidence.

This replaces it with checks that are real on a clean clone. The matrix
(ubuntu/windows x py3.11-3.13) exists to catch syntax and stdlib drift, per
the note in ci.yml, and every check below is sensitive to exactly that:

  1. every module under src/tofu imports          -- syntax, stdlib drift,
                                                     circular imports
  2. the layer registry resolves                  -- __init__ exports match
                                                     the modules on disk
  3. capabilities build without a live model      -- the degradation path
                                                     works with nothing
                                                     installed
  4. the FastAPI app constructs and routes exist  -- decorator-time errors

None of this substitutes for the local suite. It is not meant to: the point
is that CI stops making a claim it cannot support, and starts making a
smaller one it can.

Usage:
    python scripts/ci_smoke.py
"""
from __future__ import annotations

import importlib
import pkgutil
import sys
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

# Modules that intentionally require an optional out-of-process sibling venv
# (paddle, inpaint). Importing them is still expected to succeed -- they
# degrade at call time, not import time -- so nothing is excluded here. The
# list exists to be filled in with a reason if that ever stops being true.
EXPECTED_IMPORT_FAILURES: dict[str, str] = {}

_failures: list[str] = []


def _fail(check: str, exc: BaseException) -> None:
    _failures.append(f"{check}\n{''.join(traceback.format_exception(exc)).rstrip()}")


def check_all_modules_import() -> int:
    """Import every module under src/tofu."""
    import tofu

    count = 0
    for mod in pkgutil.walk_packages(tofu.__path__, prefix="tofu."):
        if mod.name in EXPECTED_IMPORT_FAILURES:
            continue
        try:
            importlib.import_module(mod.name)
            count += 1
        except Exception as exc:  # noqa: BLE001 -- reporting, not handling
            _fail(f"import {mod.name}", exc)
    return count


def check_layer_registry() -> None:
    """`tofu.layers` must export what actually sits in the package."""
    try:
        from tofu import layers

        missing = [
            name
            for name in getattr(layers, "__all__", [])
            if not hasattr(layers, name)
        ]
        if missing:
            raise AssertionError(
                f"tofu.layers.__all__ names absent attributes: {missing}"
            )
    except Exception as exc:  # noqa: BLE001
        _fail("layer registry", exc)


def check_capabilities_build() -> None:
    """Capability reporting must work with no optional provider installed.

    This is the degradation path every clean install takes, and it is the
    one most likely to break silently -- a missing provider should produce
    an honest 'unavailable', never an exception.
    """
    try:
        sys.path.insert(0, str(ROOT / "server"))
        # Call the endpoint function itself rather than re-deriving its
        # arguments. A smoke check that assembles its own call can pass while
        # the real /api/capabilities raises -- which is the failure worth
        # catching, since this endpoint is what the frontend gates on.
        from main import capabilities

        caps = capabilities()
        if not isinstance(caps, dict):
            raise AssertionError(f"/api/capabilities returned {type(caps)!r}, want dict")
        if not caps:
            raise AssertionError("/api/capabilities returned an empty contract")
    except Exception as exc:  # noqa: BLE001
        _fail("capabilities build", exc)


def check_app_constructs() -> None:
    """The FastAPI app must import and expose routes."""
    try:
        sys.path.insert(0, str(ROOT / "server"))
        from main import app

        routes = [getattr(r, "path", None) for r in app.routes]
        if not any(routes):
            raise AssertionError("FastAPI app exposes no routes")
    except Exception as exc:  # noqa: BLE001
        _fail("app construction", exc)


def main() -> int:
    print(f"ci smoke: python {sys.version.split()[0]} on {sys.platform}")

    imported = check_all_modules_import()
    print(f"  modules imported ........ {imported}")

    check_layer_registry()
    print("  layer registry .......... checked")

    check_capabilities_build()
    print("  capabilities build ...... checked")

    check_app_constructs()
    print("  app construction ........ checked")

    if _failures:
        print(f"\n{len(_failures)} smoke check(s) failed:\n")
        for f in _failures:
            print(f"--- {f}\n")
        return 1

    print("\nall smoke checks passed")
    print(
        "note: this is NOT the test suite. tests/ is local-only by design; "
        "release test evidence comes from evidence/test-run.json."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
