## 🍢 proof_objective — a loss that knows how wrong a wrong answer is
## vieuxtiful
"""The supervised, distributional objective. TRAINING ONLY.

WHAT THE INCUMBENT LOSS CANNOT EXPRESS. `supervised_contrastive_loss`
treats identity as an atomic class label, so a negative one edit away from
the truth and a negative sharing no characters with it incur the same
penalty. `0` against `O` costs what garbage costs. The gradient therefore
carries which candidate is wrong but never HOW wrong, which is exactly the
distinction a reviewer acts on and exactly the one a retrieval system
should be learning.

    contrastive     is this the right class?          atomic
    this objective  how far is this from the truth?   graded by NED

FOUR PIECES, and each closes a specific gap:

  (a) a softmax over the pool makes cross-entropy definable at all --
      `layers/proof_distribution` builds the same object at inference, and
      the two share one implementation so training and serving cannot
      drift into scoring different quantities;
  (b) NED-derived soft targets grade the error;
  (c) summation over attempts supervises the multi-view machinery that the
      runtime currently averages post hoc;
  (d) survival gating carries the fail-closed rule into the gradient.

(d) IS NOT OPTIONAL. Backpropagating cross-entropy through an instance
whose identity-bearing evidence did not survive teaches the encoder to
manufacture agreement from destroyed ink -- precisely the failure the
survival gate exists to prevent at inference. Where nothing survives there
is no gradient, only abstention. See `layers/decant`.

WHAT IS DELIBERATELY ABSENT. The transport regularizer of the technical
paper's eq. (14) is not here: it belongs with the fitted alignment cost,
and coupling an unfitted eta to the candidate distribution would train
against three frozen guesses. It arrives with that work, not before it.

CONSISTENCY IS LICENSED, NOT ASSUMED. The paper's eq. (12) ties every
attempt to the consensus. Restricting that to attempts a measurement study
has certified as equivalent is not a caveat, it is the difference between
regularizing noise and regularizing away a real signal -- see
:func:`consistency_kl`.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from tofu.layers.proof_distribution import TEMPERATURE, softmax
from tofu.utils.distance import normalized_edit_distance

OBJECTIVE_SCHEMA = "proof-objective-v1"

## Temperature of the NED-graded soft target.
##
## `invented`. It decides how much credit a near miss keeps: as sigma -> 0
## the target collapses onto the closest candidate and the loss becomes
## ordinary hard cross-entropy against it; as sigma grows every candidate
## looks equally acceptable and the graded signal this objective exists for
## disappears. 0.15 sits just under one edit in a seven-character reading,
## so a single-character confusion retains substantial mass while a
## half-wrong string does not. NOT SWEPT.
## Registered in docs/threshold-register.md.
SIGMA = 0.15

## Weight on the attempt-consistency regularizer (mu_1).
##
## `invented`, and deliberately an order of magnitude below the
## cross-entropy it accompanies: agreement between views is evidence, not
## the objective, and a consistency term strong enough to dominate would be
## satisfied perfectly by an encoder that ignored the ink and said the same
## thing every time. NOT SWEPT.
MU_CONSISTENCY = 0.1

## The survival states that may contribute a recognition gradient.
## `partial` is included for the same reason it may still reach review:
## evidence is present and incomplete, which is a truncated read rather
## than a destroyed one. `unknown` is excluded -- not measured is not
## measured-and-adequate, the distinction `decant` exists to keep.
TRAINABLE_SURVIVAL_STATES = frozenset({"present", "partial"})


def ned_soft_target(
    candidates: Sequence[str], truth: str, sigma: float = SIGMA,
) -> list[float]:
    """The graded target q_i, from normalized edit distance to the truth.

    Replaces the one-hot label with a distribution that says how wrong each
    wrong candidate is. A confusable one edit from the truth keeps most of
    the mass a perfect match would have; an unrelated string keeps almost
    none.

    Built through the same stable softmax the inference head uses, with
    -NED as the score and sigma as the temperature, so the target and the
    prediction are the same kind of object computed the same way. Small
    sigma is the interesting regime and also the one that underflows, which
    is the whole reason not to write the exponentials out by hand here.

    `truth` is read ONLY here and only at training time. Nothing in the
    candidate generator sees it, and the existing no-ground-truth-length
    leakage rule extends to this function's output.
    """
    if not candidates:
        return []
    distances = [
        -normalized_edit_distance(candidate, truth) for candidate in candidates
    ]
    return softmax(distances, temperature=sigma)


def survival_weight(state: Any) -> float:
    """omega(X): may this instance contribute a recognition gradient?

    Binary and unapologetic. A softened weight would be a policy that lets
    destroyed evidence teach the encoder a little, which is the failure
    mode in miniature rather than a compromise.
    """
    if isinstance(state, dict):
        state = state.get("state")
    return 1.0 if state in TRAINABLE_SURVIVAL_STATES else 0.0


def _equivalence_groups(
    channels: Sequence[str],
    equivalence: dict[Any, str] | None,
) -> list[list[int]]:
    """Attempt indices grouped into sets consistency may be required within.

    Attempts on one channel are always groupable: they are the same
    measurement seen through blur, morphology or typeface, and disagreement
    between them is the noise the regularizer exists to suppress.

    Attempts on DIFFERENT channels are grouped only where a measurement
    study has certified the pair equivalent for the stratum. This is the
    whole content of `docs/channel-envelope-report.md`: on that corpus no
    stratum was certified, because the fixed crop path is a strict garbling
    of the full pipeline by 0.19 to 0.62 NED depending on script. Tying
    them would train the encoder to agree with the weaker channel, and on
    Japanese the weaker channel recovers nothing usable at all.
    """
    order: list[str] = []
    members: dict[str, list[int]] = {}
    for index, channel in enumerate(channels):
        if channel not in members:
            members[channel] = []
            order.append(channel)
        members[channel].append(index)

    parent = {channel: channel for channel in order}

    def find(channel: str) -> str:
        while parent[channel] != channel:
            parent[channel] = parent[parent[channel]]
            channel = parent[channel]
        return channel

    for pair, verdict in (equivalence or {}).items():
        ## Only an explicit `equivalent` merges. `underpowered` must not:
        ## a stratum too small to measure is not one where the channels
        ## agree, and treating it as such is how an unmeasured tie becomes
        ## a training assumption.
        if verdict != "equivalent":
            continue
        left, right = tuple(pair)
        if left in parent and right in parent:
            parent[find(left)] = find(right)

    grouped: dict[str, list[int]] = {}
    for channel in order:
        grouped.setdefault(find(channel), []).extend(members[channel])
    return [sorted(indices) for indices in grouped.values()]


def attempt_cross_entropy(
    scores: Any,
    target: Any,
    *,
    temperature: float = TEMPERATURE,
    mask: Any = None,
):
    """L_CE for one instance: every attempt scored against the graded target.

    `scores` is (attempts, candidates) of raw similarity; `target` is
    (candidates,) and sums to one. Summed over attempts rather than
    averaged, because each attempt is an observation in its own right --
    averaging first is exactly what the runtime does today and is why the
    multi-view machinery has never been trainable.

    Minimizing this drives the expected NED under the model toward the
    smallest NED the candidate pool contains. It cannot drive it lower: a
    pool that does not contain the truth bounds the loss from below, which
    is a property of candidate generation and not something the encoder can
    learn its way out of.
    """
    import torch

    logits = scores / max(float(temperature), 1e-6)
    if mask is not None:
        logits = logits.masked_fill(~mask, float("-inf"))
    log_probability = torch.log_softmax(logits, dim=-1)
    return -(target.unsqueeze(0) * log_probability).sum(dim=-1).sum()


def consistency_kl(
    scores: Any,
    *,
    temperature: float = TEMPERATURE,
    channels: Sequence[str] | None = None,
    equivalence: dict[Any, str] | None = None,
    mask: Any = None,
):
    """R_cons: attempts of one measurement must tell the same story.

    The term the original programme primed as `lambda_3 * L_consistency`
    and never built. Blur, morphological and typeface variants of the same
    ink should agree; where they do not, the disagreement is evidence, and
    the runtime already exports it as `transformation_variance`.

    Computed WITHIN certified-equivalent groups only -- see
    :func:`_equivalence_groups`. A group of one attempt contributes nothing,
    which is correct: a single view cannot disagree with itself.
    """
    import torch

    attempts = scores.shape[0]
    names = list(channels) if channels is not None else ["default"] * attempts
    if len(names) != attempts:
        raise ValueError("one channel label per attempt is required")

    logits = scores / max(float(temperature), 1e-6)
    if mask is not None:
        logits = logits.masked_fill(~mask, float("-inf"))
    log_probability = torch.log_softmax(logits, dim=-1)
    probability = log_probability.exp()

    total = scores.sum() * 0.0
    for indices in _equivalence_groups(names, equivalence):
        if len(indices) < 2:
            continue
        group = probability[indices]
        group_log = log_probability[indices]
        consensus = group.mean(dim=0, keepdim=True).clamp_min(1e-12)
        total = total + (group * (group_log - consensus.log())).sum()
    return total


## Weight on the transport regularizer (mu_2).
##
## `invented`, and unused by any promoted arm. Registered in
## docs/threshold-register.md.
MU_TRANSPORT = 0.05


def partial_sinkhorn_torch(
    cost: Any,
    *,
    epsilon: float = 0.08,
    iterations: int = 50,
    dustbin_mass: float = 0.25,
):
    """The shipped partial-transport solver, differentiably.

    A twin of `proof_alignment.partial_sinkhorn` rather than a replacement:
    the numpy one runs on the inference path and must stay torch-free, and
    this one exists so gradients can reach the cost weights. The two are
    held to agreeing numerically by test, which is the only thing that
    makes the twin honest.

    Sinkhorn is iterated matrix scaling, so it is differentiable by
    construction -- the irony the diagnosis noted is that the shipped solver
    already had this property and nothing ever used it. Gradients flow by
    ordinary backpropagation through the scaling iterations; no implicit
    function theorem, no unrolled-vs-implicit choice to get wrong.

    `cost` is the dustbin-augmented matrix, so the caller owns how eta
    enters it and this stays a solver rather than a cost model.
    """
    import torch

    rows, columns = cost.shape[0] - 1, cost.shape[1] - 1
    left = torch.full(
        (rows + 1,), (1.0 - dustbin_mass) / rows, dtype=cost.dtype, device=cost.device,
    )
    right = torch.full(
        (columns + 1,), (1.0 - dustbin_mass) / columns,
        dtype=cost.dtype, device=cost.device,
    )
    left[-1] = dustbin_mass
    right[-1] = dustbin_mass
    kernel = torch.exp(-cost / max(float(epsilon), 1e-6))
    u = torch.ones_like(left)
    v = torch.ones_like(right)
    for _ in range(max(1, iterations)):
        u = left / torch.clamp(kernel @ v, min=1e-12)
        v = right / torch.clamp(kernel.T @ u, min=1e-12)
    return u[:, None] * kernel * v[None, :]


def transport_regularizer(
    probabilities: Any,
    plans: Sequence[Any],
    costs: Sequence[Any],
):
    """R_OT: expected transport cost under the model's own distribution.

    The structural effect, and the reason it is worth having: a candidate
    may no longer earn probability mass its components cannot transport
    onto the ink. A global embedding that likes the whole silhouette is not
    enough if the parts land in the wrong places -- which is exactly the
    regime (split, merged, occluded, detached marks) where the global
    encoder is weakest.

    Normalized by the transported mass, so a plan that routes most of its
    mass to the dustbin is not rewarded for having little left to pay for.
    """
    total = probabilities.sum() * 0.0
    for index, (plan, cost) in enumerate(zip(plans, costs, strict=True)):
        real = plan[:-1, :-1]
        mass = real.sum()
        if float(mass.detach()) <= 1e-9:
            continue
        total = total + probabilities[index] * (real * cost[:-1, :-1]).sum() / mass
    return total


def instance_objective(
    scores: Any,
    candidates: Sequence[str],
    truth: str,
    survival_state: Any,
    *,
    temperature: float = TEMPERATURE,
    sigma: float = SIGMA,
    mu_consistency: float = MU_CONSISTENCY,
    channels: Sequence[str] | None = None,
    equivalence: dict[Any, str] | None = None,
    mask: Any = None,
) -> dict[str, Any]:
    """The whole objective for one instance, gated by survival.

    Returns the loss alongside the pieces that produced it, because a
    combined scalar that cannot be decomposed is a number nobody can
    diagnose when it stops moving.

    The gate multiplies rather than skips so the graph stays connected and
    the returned tensor is differentiable whatever the state -- a caller
    summing over a batch must not have to special-case the gated rows.
    """
    import torch

    weight = survival_weight(survival_state)
    target = torch.tensor(
        ned_soft_target(candidates, truth, sigma),
        dtype=scores.dtype, device=scores.device,
    )
    cross_entropy = attempt_cross_entropy(
        scores, target, temperature=temperature, mask=mask,
    )
    consistency = consistency_kl(
        scores, temperature=temperature, channels=channels,
        equivalence=equivalence, mask=mask,
    )
    return {
        "loss": weight * (cross_entropy + mu_consistency * consistency),
        "cross_entropy": cross_entropy,
        "consistency": consistency,
        "survival_weight": weight,
        "target": target,
    }
