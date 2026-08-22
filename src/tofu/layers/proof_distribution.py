## 🍢 proof_distribution — turning a ranking into a probability
## vieuxtiful
"""A distribution over the candidate pool, which the stack has never had.

WHAT WAS MISSING. `proof_runtime` scores every candidate by cosine
similarity, averages across views and typefaces, sorts, and hands the
fusion cascade the top-1 support and the top-2 margin. Everything below
rank 2 is discarded before any decision sees it, and there is no
probability object anywhere: cross-entropy is undefined at training time
AND at inference time, calibration can only act on a single scalar margin,
and "how much of the pool's mass sits on the top candidate" cannot be
asked at all.

    ranked cosine        which candidate is nearest
    p(g_i | X)           how much belief the evidence supports, per candidate

Those are different objects and only the second can be trained against,
calibrated over its whole shape, or turned into an expected risk.

WHAT THIS IS NOT. A distribution is not a confidence. Softmaxing a
similarity produces a well-formed probability object over a pool that was
itself assembled by earlier stages -- if the true reading is not in the
pool, the mass still sums to one and says nothing about that. It ranks and
it calibrates; acceptance remains with the fusion cascade and the survival
gate. Candidate elimination alone still cannot produce acceptance.

DELIBERATELY AGNOSTIC ABOUT THE SCORE. These functions take scores, not
embeddings, so the quantity being softmaxed is the caller's choice. Today
it is cosine similarity to the candidate's rendering. The technical paper's
revision 2 defines the head over a FUSED similarity s_theta that can carry
Scene-conditioned terms, and substituting that means changing what the
caller passes in, not changing anything here.

NUMPY-FREE ON PURPOSE. This runs on the inference path, where torch is
optional and every import costs. Pools are capped at 64 candidates and the
arithmetic is a few hundred flops; a dependency would buy nothing.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

DISTRIBUTION_SCHEMA = "proof-distribution-v1"

## Softmax temperature over the candidate pool.
##
## `invented`. It sets how sharply cosine similarity converts into belief:
## as tau -> 0 the distribution collapses onto the argmax and reproduces the
## ranking this replaces; as tau -> infinity it flattens to uniform and says
## nothing. 0.07 is the value the encoder's own supervised-contrastive loss
## uses for the same cosine geometry (`proof_encoder.supervised_contrastive_loss`
## defaults to 0.1) rounded toward the sharper end, on the grounds that a
## retrieval distribution should be no flatter than the loss that shaped the
## embedding space. NOT SWEPT, and it may not decide anything until it is:
## the distribution features enter fusion at zero weight.
## Registered in docs/threshold-register.md.
TEMPERATURE = 0.07


def softmax(
    scores: Sequence[float], temperature: float = TEMPERATURE,
) -> list[float]:
    """A probability distribution over the pool, from raw scores.

    Shift-invariant by construction: the maximum is subtracted before
    exponentiating, so a pool of large similar scores cannot overflow and a
    pool of very negative ones cannot underflow to all-zero. Both are
    reachable here -- cosine scores live in [-1, 1] but tau is small, so the
    logits are an order of magnitude larger than the scores.

    An empty pool returns an empty list rather than raising: "no candidates"
    is a state the retrieval path already reports, not an error.
    """
    values = [float(value) for value in scores]
    if not values:
        return []
    tau = max(float(temperature), 1e-6)
    largest = max(values)
    weights = [math.exp((value - largest) / tau) for value in values]
    total = sum(weights)
    if total <= 0.0:
        ## Every weight underflowed. Uniform is the honest answer: the
        ## scores carry no recoverable ordering at this temperature.
        return [1.0 / len(values)] * len(values)
    return [weight / total for weight in weights]


def normalized_entropy(probabilities: Sequence[float]) -> float | None:
    """Entropy in [0, 1], divided by the entropy of a uniform pool.

    Normalized because the pool size varies per region -- `candidate_pool`
    returns anything from 2 to 64 candidates -- and raw entropy is bounded
    by log K, so an unnormalized value would say "this region is more
    uncertain" when it only means "this region had more candidates".

    0 is all mass on one candidate; 1 is a pool the evidence cannot
    separate at all. Returns None for a pool too small to have a shape --
    which is decided by how many candidates there ARE, not by how many
    carry mass: a pool of three with all belief on one has entropy zero,
    and that is the sharpest reading the measure can report, not an
    unmeasurable one.
    """
    if len(probabilities) < 2:
        return None
    values = [float(value) for value in probabilities if value > 0.0]
    ## 0 log 0 is 0 in the limit, so zero-mass candidates contribute
    ## nothing and are dropped only after the pool has been sized.
    raw = -sum(value * math.log(value) for value in values)
    return raw / math.log(len(probabilities))


def top_mass(probabilities: Sequence[float], k: int = 3) -> float | None:
    """How much belief sits on the k best candidates.

    The question the top-2 margin cannot answer: a margin of 0.02 means
    something very different when the top three hold 0.95 of the mass than
    when they hold 0.3 of it.
    """
    if not probabilities or k < 1:
        return None
    return sum(sorted(probabilities, reverse=True)[:k])


def kl_divergence(
    left: Sequence[float], right: Sequence[float], floor: float = 1e-12,
) -> float | None:
    """KL(left || right), the per-attempt consistency quantity.

    This is the term the original programme primed as
    `lambda_3 * L_consistency` and never built: blur, morphological and
    typeface variants of the same ink should tell the same story, and how
    far one attempt sits from the consensus is itself evidence. Measured
    here at inference; it becomes a training signal in the supervised
    objective.

    Returns None on a length mismatch rather than a number -- two
    distributions over different pools are not comparable, and quietly
    truncating to the shorter one would produce a plausible value for an
    incoherent question.
    """
    if len(left) != len(right) or not left:
        return None
    total = 0.0
    for p, q in zip(left, right, strict=True):
        if p <= 0.0:
            continue
        total += p * math.log(max(p, floor) / max(q, floor))
    return total


def mean_distribution(attempts: Sequence[Sequence[float]]) -> list[float]:
    """The consensus across attempts, p-bar.

    Attempts over different pool sizes are refused for the same reason
    :func:`kl_divergence` refuses them.
    """
    rows = [list(row) for row in attempts if row]
    if not rows:
        return []
    width = len(rows[0])
    rows = [row for row in rows if len(row) == width]
    if not rows:
        return []
    return [sum(row[index] for row in rows) / len(rows) for index in range(width)]


def summarize(
    attempts: Sequence[Sequence[float]],
    temperature: float = TEMPERATURE,
    top_k: int = 3,
) -> dict | None:
    """The shape of the belief, and how much the attempts disagree about it.

    Returns the consensus distribution plus the scalars fusion and
    calibration can act on. Per-attempt distributions are summarized rather
    than carried whole: an attempt-by-candidate matrix is O(N*K) floats in
    every manifest, while the quantity anything downstream actually reads is
    each attempt's divergence from the consensus, which is O(N). Training
    recomputes the full matrix from the encoder anyway.
    """
    consensus = mean_distribution(attempts)
    if not consensus:
        return None
    divergences = [
        value for row in attempts
        if (value := kl_divergence(row, consensus)) is not None
    ]
    return {
        "schema": DISTRIBUTION_SCHEMA,
        "temperature": temperature,
        "attempts": len(divergences),
        "consensus": [round(value, 8) for value in consensus],
        "entropy": (
            round(value, 8)
            if (value := normalized_entropy(consensus)) is not None else None
        ),
        "top1_probability": round(max(consensus), 8),
        f"top{top_k}_mass": (
            round(value, 8)
            if (value := top_mass(consensus, top_k)) is not None else None
        ),
        ## Zero when one attempt was scored: a single view cannot disagree
        ## with itself, and reporting None would make "no disagreement
        ## measurable" indistinguishable from "measured, and none found".
        "max_attempt_divergence": round(max(divergences), 8) if divergences else 0.0,
        "mean_attempt_divergence": (
            round(sum(divergences) / len(divergences), 8) if divergences else 0.0
        ),
    }
