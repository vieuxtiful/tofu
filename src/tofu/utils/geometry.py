## 🍢 ToFU — shared quadrilateral tests
##
## One predicate, used by every layer that turns a perspective quad into a
## homography. Scribe draws through one, Cleanse erases through one and
## Verify measures through one; if they disagree about which quads are
## usable, one of them warps an image the others never saw, and the
## verification corpus's `perspective-degenerate-falls-back` case exists
## precisely to pin that down.
from __future__ import annotations

from typing import List, Sequence, Tuple


def quad_is_usable(corners: Sequence[Tuple[float, float]]) -> bool:
    """Convex, correctly wound and non-degenerate.

    A self-intersecting or collinear quad makes getPerspectiveTransform
    singular. A singular solve does not always raise -- more often it
    yields a matrix whose sampled coordinates run off to infinity, which
    costs an unbounded warp rather than an exception. The sign of the
    cross product at each corner catches both cases: a convex polygon
    turns the same way at every vertex, and a collinear triple turns not
    at all. Callers fall back to the affine/bbox path rather than failing.
    """
    if corners is None or len(corners) != 4:
        return False
    signs: List[bool] = []
    for index in range(4):
        ax, ay = corners[index]
        bx, by = corners[(index + 1) % 4]
        cx, cy = corners[(index + 2) % 4]
        cross = (bx - ax) * (cy - by) - (by - ay) * (cx - bx)
        if abs(cross) < 1e-9:
            return False
        signs.append(cross > 0)
    return all(signs) or not any(signs)
