"""Bounded partial component alignment for diagnostic glyph evidence.

The transport plan answers a question the global embedding cannot: do this
candidate's components actually land on the observed ink, or does it score
well as a whole shape while its parts sit in the wrong places? Split,
merged, occluded and detached marks are exactly where those two answers
come apart.

TWO THINGS ABOUT THIS LAYER'S HISTORY, both recorded because they change
how much the numbers below should be trusted.

First, its output reached no decision at all until 2026-08-17. `align()`
returned `alignment_cost` and friends while `fusion._features` read a
`support` key that nothing ever produced, so the feature was permanently
`None` and two tests asserted exactly that. The Phase 5 ablation that was
supposed to price component alignment therefore never measured it.

Second, every constant here was hand-set and none has been swept. They are
now loadable from a registered artifact (:func:`load_cost_weights`) so a
fitted set can replace them without a source edit -- but the DEFAULT
reproduces the original values bit for bit, and it stays that way until a
fit is certified.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

ALIGNMENT_SCHEMA = "partial-component-ot-v1"
COST_WEIGHTS_SCHEMA = "alignment-cost-v1"
MAX_COMPONENTS = 64
SINKHORN_ITERATIONS = 50
ENTROPY_EPSILON = 0.08
DUSTBIN_MASS = 0.25
DUSTBIN_COST = 0.35

## Cost weights (eta) over the three component features: centroid position,
## log-area, log-aspect. `invented` -- hand-set at the values below and
## never swept; registered in docs/threshold-register.md.
##
## Kept as a named default rather than inlined so that a fitted artifact can
## replace them through `load_cost_weights` without touching source, and so
## that "what did this run use" is answerable from the alignment record
## instead of from the commit history.
DEFAULT_COST_WEIGHTS = (0.55, 0.25, 0.20)

## Aspect-ratio difference is clipped before weighting: two components whose
## log-aspects differ by more than this are already maximally unlike, and
## letting the difference run unbounded would let one wildly elongated
## component dominate a whole plan.
ASPECT_CLIP = 2.0


def component_features(mask: Any, max_components: int = MAX_COMPONENTS):
    import cv2
    import numpy as np

    image = np.asarray(mask)
    if image.ndim != 2 or image.size == 0:
        return np.empty((0, 4), dtype=np.float64)
    binary = (image > 0.5).astype(np.uint8)
    count, _labels, stats, centroids = cv2.connectedComponentsWithStats(binary, connectivity=8)
    height, width = binary.shape
    rows = []
    for index in range(1, count):
        x, y, component_width, component_height, area = stats[index]
        if area < 2:
            continue
        cx, cy = centroids[index]
        rows.append((
            float(cx / max(1, width)), float(cy / max(1, height)),
            float(np.log1p(area) / np.log1p(binary.size)),
            float(np.log(max(1, component_width) / max(1, component_height))),
            int(area), int(x), int(y),
        ))
    rows.sort(key=lambda row: (-row[4], row[6], row[5]))
    rows = rows[:max_components]
    rows.sort(key=lambda row: (row[1], row[0], -row[4]))
    return np.asarray([row[:4] for row in rows], dtype=np.float64)


def load_cost_weights(path_value: str | None) -> tuple[tuple[float, float, float], str]:
    """Read a fitted eta, or fall back to the registered default.

    Returns the weights and a revision string, because a plan scored under
    one eta and a plan scored under another are not comparable and the
    record has to say which produced it.

    Any failure -- missing file, wrong schema, malformed weights -- returns
    the default rather than raising. Alignment is diagnostic evidence: a bad
    artifact must degrade it to the incumbent behaviour, never fail a
    detection and never silently produce a negative identity verdict.
    """
    if not path_value:
        return DEFAULT_COST_WEIGHTS, "alignment-cost-default"
    try:
        data = json.loads(Path(path_value).read_text(encoding="utf-8"))
        if data.get("schema") != COST_WEIGHTS_SCHEMA:
            return DEFAULT_COST_WEIGHTS, "alignment-cost-schema-mismatch"
        weights = tuple(float(value) for value in data["weights"])
        if len(weights) != 3 or any(value < 0 for value in weights):
            return DEFAULT_COST_WEIGHTS, "alignment-cost-invalid"
        return weights, str(data.get("revision") or "alignment-cost-unnamed")
    except Exception:
        return DEFAULT_COST_WEIGHTS, "alignment-cost-unavailable"


def _cost_matrix(observed, candidate, weights=DEFAULT_COST_WEIGHTS):
    import numpy as np

    position = np.linalg.norm(observed[:, None, :2] - candidate[None, :, :2], axis=2)
    area = np.abs(observed[:, None, 2] - candidate[None, :, 2])
    aspect = np.abs(observed[:, None, 3] - candidate[None, :, 3])
    return (
        weights[0] * position
        + weights[1] * area
        + weights[2] * np.minimum(aspect, ASPECT_CLIP)
    )


def partial_sinkhorn(
    observed,
    candidate,
    epsilon: float = ENTROPY_EPSILON,
    iterations: int = SINKHORN_ITERATIONS,
    weights=DEFAULT_COST_WEIGHTS,
):
    import numpy as np

    if len(observed) == 0 or len(candidate) == 0:
        return None
    real_cost = _cost_matrix(observed, candidate, weights)
    cost = np.full((len(observed) + 1, len(candidate) + 1), DUSTBIN_COST, dtype=np.float64)
    cost[:-1, :-1] = real_cost
    cost[-1, -1] = 0.0
    left = np.full(len(observed) + 1, (1.0 - DUSTBIN_MASS) / len(observed))
    right = np.full(len(candidate) + 1, (1.0 - DUSTBIN_MASS) / len(candidate))
    left[-1] = DUSTBIN_MASS
    right[-1] = DUSTBIN_MASS
    kernel = np.exp(-cost / max(epsilon, 1e-6))
    u = np.ones_like(left)
    v = np.ones_like(right)
    for _ in range(max(1, iterations)):
        u = left / np.maximum(kernel @ v, 1e-12)
        v = right / np.maximum(kernel.T @ u, 1e-12)
    transport = u[:, None] * kernel * v[None, :]
    real_transport = transport[:-1, :-1]
    matched_mass = float(real_transport.sum())
    if matched_mass <= 1e-9:
        return None
    return {
        "alignment_cost": float((real_transport * real_cost).sum() / matched_mass),
        "matched_mass": matched_mass,
        "observed_unmatched_mass": float(transport[:-1, -1].sum()),
        "candidate_unmatched_mass": float(transport[-1, :-1].sum()),
    }


def align(
    observed_mask: Any,
    candidate_mask: Any,
    weights=DEFAULT_COST_WEIGHTS,
    revision: str = "alignment-cost-default",
) -> dict[str, Any] | None:
    """Transport the candidate's components onto the observed ink.

    The record carries the eta that produced it. Two plans scored under
    different weights are not comparable, and a feature vector that cannot
    say which weights it used cannot be pooled across runs.
    """
    observed = component_features(observed_mask)
    candidate = component_features(candidate_mask)
    result = partial_sinkhorn(observed, candidate, weights=weights)
    if result is None:
        return None
    return {
        "schema": ALIGNMENT_SCHEMA,
        "cost_weights_revision": revision,
        "observed_components": len(observed),
        "candidate_components": len(candidate),
        **{key: round(value, 8) for key, value in result.items()},
    }
