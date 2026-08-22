## 🍢 distance — one edit distance, and one normalized measure of it
## vieuxtiful
"""Levenshtein distance and normalized edit distance, in exactly one place.

WHY THIS MODULE EXISTS. Four implementations of the same arithmetic had
accumulated -- `verify._edit_distance`, `eval_detector_evidence._norm_ed`,
and copies in the eval scripts and tests -- and they were free to drift.
That matters now in a way it did not before: the supervised objective uses
NED for three different jobs at once (the soft training target, the
evaluation metric, and the channel-regret target), and three jobs computed
by three functions are three different measurements wearing one name.

WHAT IS AND IS NOT SHARED. The PRIMITIVE is shared; the NORMALIZATION is
not, because the two normalizations in this codebase answer different
questions and both are correct:

    CER  (verify.error_rates)   edits / len(reference)
                                "what share of the truth did we get wrong"
    NED  (here)                 edits / max(|a|, |b|)
                                "how far apart are these two strings"

CER is asymmetric on purpose -- it is a per-reference error rate, clamped
to 1.0 so a runaway hypothesis cannot score above total failure. NED is
symmetric and bounded by construction, which is what a soft target and a
regret weight both need. `verify` therefore keeps its own normalization and
borrows only :func:`levenshtein`.

THE EMPTY-STRING CASES ARE PART OF THE DEFINITION, not an edge case swept
into a guard clause. Two empty strings agree perfectly (0); one empty
string against a real reading shares no structure at all (1). Without the
explicit cases a missed detection and an empty OCR output would divide by
zero, and those are precisely the observations the measurement exists to
score.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

## The NED at or above which a region is scored unreadable *from a crop
## already known to be geometrically valid*.
##
## `preregistered`. This is an EVALUATION CONVENTION and not an acceptance
## rule: it is not a calibrated probability, it does not license a decision,
## and a production gate still requires surviving evidence and may abstain
## far below it. It sat in a source comment in `eval_detector_evidence` for
## most of a year; registered in docs/threshold-register.md on the way into
## this module.
TAU_UNREADABLE = 0.8


def levenshtein(reference: Sequence[Any], hypothesis: Sequence[Any]) -> int:
    """Levenshtein distance (Levenshtein 1966), two-row dynamic programming.

    Sequence-generic on purpose: it runs over characters for CER and NED and
    over word tokens for WER, and the algorithm is identical -- only the
    unit changes. Keeping one implementation means a fix to the recurrence
    reaches every caller instead of three of the four.
    """
    if not reference:
        return len(hypothesis)
    if not hypothesis:
        return len(reference)
    previous = list(range(len(hypothesis) + 1))
    for i, ref_item in enumerate(reference, start=1):
        current = [i]
        for j, hyp_item in enumerate(hypothesis, start=1):
            current.append(min(
                previous[j] + 1,                            # deletion
                current[j - 1] + 1,                         # insertion
                previous[j - 1] + (ref_item != hyp_item),   # substitution
            ))
        previous = current
    return previous[-1]


def normalized_edit_distance(a: str | None, b: str | None) -> float:
    """Edit distance normalized by the longer string, in [0, 1].

    0 is an exact match; 1 shares no usable structure. Symmetric, so it can
    serve as a distance between two candidate strings as readily as between
    a reading and a truth -- which the NED-graded soft target requires,
    since it scores every candidate in the pool against `t*`, not just the
    proposed one.

    Both empty-string cases are explicit; see the module docstring.
    """
    a, b = a or "", b or ""
    if not a and not b:
        return 0.0
    if not a or not b:
        return 1.0
    return levenshtein(a, b) / max(len(a), len(b))


def is_unreadable(ned: float | None, tau: float = TAU_UNREADABLE) -> bool:
    """Whether a measured NED clears the unreadability convention.

    Returns False for an unmeasured value rather than True: "we did not
    measure this" must never collapse into "this was unreadable", which is
    the same rule `decant` follows for `unknown` versus `absent`.
    """
    return ned is not None and ned >= tau
