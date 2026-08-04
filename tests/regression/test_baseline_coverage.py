"""The regression suite must actually guard something.

`tolerances.compare()` returns no violations when a harness has no entry in
baseline_metrics.json (`tolerances.tolerances_for` returns [] and the loop
over it never runs). A regression test for such a harness therefore does its
full, expensive work and then asserts nothing -- it passes whatever happens.

That is not hypothetical. Seven of the nine harnesses were in exactly that
state: savor, tofu, render, cleanse_providers, font_match, memory,
verification_corpus and paddle all ran and all guarded nothing, while
`tolerances.py`'s own comment records an EARLIER occurrence of the same
failure at file scope. `paddle` was worse still -- `compare("paddle", ...)`
had no generator in HARNESS_MAP at all, so no amount of running the tool
could have baselined it.

These two tests make the silence audible.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))

from tolerances import load_baseline  # noqa: E402


def _harnesses_compared() -> dict[str, list[str]]:
    """Harness name -> the regression test files that compare against it."""
    used: dict[str, list[str]] = {}
    for path in sorted(HERE.glob("test_*.py")):
        if path.name == Path(__file__).name:
            continue
        source = path.read_text(encoding="utf-8")
        for name in re.findall(r'compare\(\s*"([a-z_]+)"', source):
            used.setdefault(name, []).append(path.name)
    return used


def _baselined() -> dict[str, dict]:
    return load_baseline().get("harnesses", {})


def test_every_compared_harness_has_a_baseline():
    """A test that compares against an unbaselined harness asserts nothing."""
    baselined = _baselined()
    hollow = {
        harness: sorted(files)
        for harness, files in _harnesses_compared().items()
        if not baselined.get(harness)
    }
    assert not hollow, (
        "these regression tests run their harness and then assert nothing, "
        "because compare() finds no baseline for them:\n"
        + "\n".join(f"  {h}: {', '.join(f)}" for h, f in sorted(hollow.items()))
        + "\n\nrun: .venv/Scripts/python tests/regression/generate_baseline.py "
          "--harness <name>"
    )


def test_every_generator_is_reachable_and_used():
    """Every generator should have produced a baseline, and vice versa.

    Catches the two halves of the same drift: a harness that is compared
    against but has no generator (paddle, until it was added), and a
    generator whose output nothing ever asserts on.
    """
    sys.path.insert(0, str(ROOT / "src"))
    sys.path.insert(0, str(ROOT / "scripts"))
    from generate_baseline import HARNESS_MAP

    compared = set(_harnesses_compared())
    generators = set(HARNESS_MAP)

    missing_generator = sorted(compared - generators)
    assert not missing_generator, (
        "these harnesses are compared against but cannot be baselined -- "
        f"no entry in HARNESS_MAP: {missing_generator}"
    )

    unused = sorted(generators - compared)
    assert not unused, (
        "these generators produce baselines no test compares against: "
        f"{unused}"
    )
