## 🍢 flight — the same pour, served several ways, side by side
## vieuxtiful
"""Which measurement channel produced this reading?

A tasting flight is the same subject poured through several glasses so the
pours can be compared. This layer names the pours.

THE DEFECT IT EXISTS TO FIX. The pipeline does not observe a reading
through one immutable channel. It applies a family of decoders -- the fixed
crop path, three multipass threshold rungs, the zoom pass, the vertical
column and baseline row merges, the Paddle rescue, the independent verifier
-- and each is a different measurement of the same pixels. Every number
downstream of OCR was nevertheless written as though there were one channel:

    NED(t_hat, t*)          what the reports say
    D_{e,c}(X)              what was actually measured

Suppressing `c` is not cosmetic. A crop is a deterministic transformation of
the available image and so cannot ADD information about the latent reading,
but the information it loses is stratum-dependent: approximately sufficient
for Latin, a strict garbling for CJK where context, zoom, or component
merging carries identity-bearing evidence. The same image therefore earns
different NED values through different channels, and a crop score reported
as a scene-level property is a category error the corpus has already been
scored under.

WHAT THIS LAYER DOES. It names the channel, fingerprints it, and stamps it.
That is all. It runs on settled instances, changes no text, no geometry, and
no decision, and every failure path leaves the instance exactly as it found
it. It is the provenance stage that has to exist before anything can be
supervised per channel -- introducing the index inside a learning objective
would frame a measurement correction as a modelling trick.

WHAT IS AND IS NOT A CHANNEL. A channel is a route by which pixels became a
reading. A stage that EDITS a settled reading without re-measuring it is not
a channel: locale respacing, glossary matching, and the language-model
correction courses all change `inst.text` and none of them looked at the
image again. Those stages are skipped when tracing back to the channel that
produced a read, deliberately -- crediting a channel with a correction it
did not measure would attribute the language model's win to the detector.

UNREGISTERED IS NOT UNKNOWN. A route this module has never heard of is
reported as `unregistered` and carried, never dropped and never silently
folded into a registered neighbour. The same rule `decant` follows for
`unknown` versus `absent`: a consumer must not be able to confuse "not in
the vocabulary" with "measured through the default route".
"""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

CHANNEL_SCHEMA = "tofu-channel-v1"

## A route this module does not recognize. Reported, never dropped.
UNREGISTERED = "unregistered"

## Recognition routes: the stage that last MEASURED the pixels behind a
## settled read. Keyed by the `stage` value cicerone writes into
## `recognition_history`, so this table is checkable against that writer
## rather than being a parallel vocabulary somebody has to remember to
## update. `detection_pass` is a family -- the pass number is appended --
## because the three multipass rungs run at different detector thresholds
## and are genuinely different measurements.
DETECTION_PASS = "detection_pass"
RECOGNITION_ROUTES: dict[str, str] = {
    DETECTION_PASS: DETECTION_PASS,        ## + f"_{pass}"
    "hybrid_audit": "paddle_rescue",
    "skim_audit": "paddle_rescue",
    "multi_candidate_ocr": "independent_verifier",
    "hypothesis_promotion": "independent_verifier",
    "second_look": "composed_crop",
    "edge_rescue": "edge_rescue",
    ## The transforms that re-recognize. They write their own provenance
    ## entry (`cicerone._tag_transform_read`) under the okara stage name, so
    ## a region can name the route that read it without inferring it back
    ## from the lineage graph -- which matters because the graph's raw_craft
    ## backstop will happily absorb a detection whose real producer was a
    ## transform, and then the ancestry says "detector" for a read the
    ## detector never made.
    ##
    ## `paddle_rescue` is here on different grounds: its lineage stage is
    ## `raw_craft` -- it is a raw detector proposal -- but a read from a
    ## second, differently-architected engine is not the same measurement
    ## channel as an EasyOCR pass, and the route is where that is said.
    "paddle_rescue": "paddle_rescue",
    "zoom": "zoom",
    "surface_probe": "surface_probe",
    "merge_vertical_columns": "vertical_column_merge",
    "merge_baseline_runs": "baseline_row_merge",
    "split_tall_detections": "vertical_split",
    "compose_crop_text": "composed_crop",
    "rescue": "edge_rescue",
}

## Stages that edit a settled reading without re-measuring it. Listed
## explicitly rather than left to fall through to `unregistered`, so that
## adding a correction course does not silently start looking like a new
## measurement channel.
EDITING_STAGES = frozenset({
    "locale_typography", "menu", "savor", "wasabi", "ordinal", "skim",
    "glossary", "correction_memory",
})

## Geometry transforms that changed WHAT THE RECOGNIZER SAW, keyed by okara
## stage. `prune_contained_fragments` is absent on purpose: it drops
## candidates without reshaping the pixels behind the survivors.
GEOMETRY_TRANSFORMS: dict[str, str] = {
    "zoom": "zoom",
    "surface_probe": "surface_probe",
    "multipass_union": "multipass_union",
    "merge_detections": "merge_detections",
    "merge_vertical_columns": "vertical_column_merge",
    "merge_baseline_runs": "baseline_row_merge",
    "split_tall_detections": "vertical_split",
    "compose_crop_text": "composed_crop",
    "rescue": "edge_rescue",
}

## Transforms that assembled one region out of several. Recorded separately
## from the transform list because "was this read taken from a merged box"
## is the single question the CJK crop confound turns on.
MERGE_TRANSFORMS = frozenset({
    "merge_detections", "vertical_column_merge", "baseline_row_merge",
    "composed_crop", "multipass_union",
})

## Transforms that RE-MEASURED the pixels rather than only reshaping the box.
## These are channels in their own right, and naming them is not a
## convenience: reads produced by a transform carry no `detection_pass`
## provenance -- `recognition_history` is seeded from the raw detection's
## provenance, and a transform mints a fresh detection -- so without this
## fallback the pipeline's own routes are untraceable. The first envelope
## run measured the cost of that: 13 of 13 Korean regions and 14 of 31 Latin
## regions came back `unregistered`, which is exactly the routes the CJK
## confound is about.
##
## Membership is decided by whether the transform re-runs recognition, per
## its own source:
##   zoom / surface_probe  re-run the detector on an upscaled or surface crop
##   vertical_column_merge "re-recognizes the merged crop" (cicerone:1655)
##   baseline_row_merge    "the merged box is then RE-RECOGNIZED" (:2118)
##   vertical_split        "re-segment and re-recognize" (:1967)
##   composed_crop         composed from a crop re-recognition (:35)
##   edge_rescue           re-reads a padded crop
## `merge_detections` and `multipass_union` are absent: both are box algebra
## over existing reads and measure nothing new.
REMEASURING_TRANSFORMS = (
    "composed_crop", "vertical_column_merge", "baseline_row_merge",
    "vertical_split", "edge_rescue", "zoom", "surface_probe",
)

## Routes reached without going through the shipped detection path at all.
REGION_CROP = "region_crop"        ## detect_in_regions: the fixed crop path
GUIDED_BLOCK = "guided_block"      ## a reviewer-drawn block
FULL_FRAME = "full_frame"          ## one detector call over the whole image

## `UNREGISTERED` is deliberately NOT a member. It is the marker for a route
## outside the vocabulary, and a marker that reported itself as registered
## would let exactly the untraceable reads this layer exists to surface
## disappear into the coverage figure.
REGISTERED_ROUTES = frozenset(
    set(RECOGNITION_ROUTES.values())
    | set(GEOMETRY_TRANSFORMS.values())
    | {REGION_CROP, GUIDED_BLOCK, FULL_FRAME}
)


def is_registered(route: str) -> bool:
    """Whether a route name belongs to the closed vocabulary.

    Handles the `detection_pass` family by prefix: the pass number varies
    with how many rungs ran, and enumerating them here would go stale the
    first time a rung is added.
    """
    if route.startswith(f"{DETECTION_PASS}_"):
        return route[len(DETECTION_PASS) + 1:].isdigit()
    return route in REGISTERED_ROUTES


@dataclass(frozen=True)
class Channel:
    """Exactly which measurement produced one reading.

    Frozen and fingerprinted for the same reason `okara.DetectorConfig` is:
    two measurements are comparable only if these match, and a record that
    cannot name its own route cannot be checked against another run's.
    """
    engine: str = "unknown"
    route: str = UNREGISTERED
    view: str = "source"                       ## source | zoom | crop
    transforms: tuple[str, ...] = ()
    merged: bool = False
    crop: tuple[int, int, int, int] | None = None

    def fingerprint(self) -> str:
        parts = [
            self.engine, self.route, self.view,
            ",".join(self.transforms), str(self.merged),
            ## Crop geometry is part of the channel -- two reads of the same
            ## region through different boxes are different measurements --
            ## but it is fingerprinted at 4px granularity so that a one-pixel
            ## box jitter between runs does not manufacture a new channel.
            "x".join(str(int(v) // 4) for v in self.crop) if self.crop else "-",
        ]
        return hashlib.sha1("|".join(parts).encode("utf-8")).hexdigest()[:10]

    @property
    def channel_id(self) -> str:
        """A stable identifier that reads as its own route."""
        return f"{self.route}-{self.fingerprint()}"

    @property
    def registered(self) -> bool:
        return is_registered(self.route)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": CHANNEL_SCHEMA,
            "channel_id": self.channel_id,
            "engine": self.engine,
            "route": self.route,
            "view": self.view,
            "transforms": list(self.transforms),
            "merged": self.merged,
            "crop": list(self.crop) if self.crop else None,
            "registered": self.registered,
        }

    @classmethod
    def from_dict(cls, data: Any) -> Channel | None:
        if not isinstance(data, dict):
            return None
        crop = data.get("crop")
        return cls(
            engine=str(data.get("engine") or "unknown"),
            route=str(data.get("route") or UNREGISTERED),
            view=str(data.get("view") or "source"),
            transforms=tuple(str(value) for value in data.get("transforms") or ()),
            merged=bool(data.get("merged", False)),
            crop=(
                tuple(int(value) for value in crop)
                if isinstance(crop, list | tuple) and len(crop) == 4 else None
            ),
        )


def route_for_stage(stage: str | None, pass_number: Any = None) -> str:
    """Map a recognition stage onto its registered route.

    Returns `unregistered` for anything not in the table, including the
    editing stages -- callers walking a history should skip those rather
    than accept an unregistered channel from one.
    """
    if not stage:
        return UNREGISTERED
    route = RECOGNITION_ROUTES.get(str(stage))
    if route is None:
        return UNREGISTERED
    if route == DETECTION_PASS:
        suffix = pass_number if pass_number is not None else 0
        return f"{DETECTION_PASS}_{suffix}"
    return route


def _measuring_entry(history: Sequence[Any] | None) -> dict[str, Any] | None:
    """The last history entry that MEASURED the text this region ships.

    Walks backwards and takes the first entry that both belongs to a
    registered recognition route and was actually taken up -- `applied`,
    `accepted`, or `selected`. Editing stages and rejected candidates are
    skipped: a Paddle candidate that was logged and refused did not produce
    this read, and saying it did would credit the wrong channel.
    """
    for entry in reversed(list(history or ())):
        if not isinstance(entry, dict):
            continue
        stage = str(entry.get("stage") or "")
        if stage in EDITING_STAGES or stage not in RECOGNITION_ROUTES:
            continue
        taken = (
            entry.get("applied") is True
            or entry.get("accepted") is True
            or entry.get("selected") is True
        )
        if taken:
            return entry
    return None


def _lineage_transforms(
    lineage_index: dict[str, dict[str, Any]],
    candidate_id: str | None,
    max_depth: int = 32,
) -> tuple[str, ...]:
    """The pixel-reshaping transforms between the detector and this region.

    Walks the okara graph upward from the shipped node. Bounded depth
    because the graph is append-only and a malformed parent cycle must cost
    a truncated list, never a hang.
    """
    if not candidate_id or not lineage_index:
        return ()
    seen: set[str] = set()
    found: list[str] = []
    frontier = [candidate_id]
    while frontier and len(seen) < max_depth:
        current = frontier.pop(0)
        if current in seen:
            continue
        seen.add(current)
        node = lineage_index.get(current)
        if node is None:
            continue
        transform = GEOMETRY_TRANSFORMS.get(str(node.get("stage") or ""))
        if transform and transform not in found:
            found.append(transform)
        frontier.extend(
            str(parent) for parent in node.get("parent_candidate_ids") or ()
        )
    ## Ordered by the registered table rather than by traversal order, so
    ## the same ancestry always fingerprints to the same channel however the
    ## graph happened to be walked.
    order = list(GEOMETRY_TRANSFORMS.values())
    return tuple(sorted(found, key=order.index))


def index_lineage(lineage: Any) -> dict[str, dict[str, Any]]:
    """Node lookup for one manifest's candidate graph, by candidate id."""
    nodes = (lineage or {}).get("nodes") if isinstance(lineage, dict) else None
    if not isinstance(nodes, list):
        return {}
    return {
        str(node["candidate_id"]): node
        for node in nodes
        if isinstance(node, dict) and node.get("candidate_id")
    }


def channel_for(
    inst: Any,
    lineage_index: dict[str, dict[str, Any]] | None = None,
) -> Channel:
    """The channel that produced this region's shipped reading.

    Never raises and never returns None: a region whose provenance cannot be
    traced gets an `unregistered` channel carrying whatever IS known, which
    is a reportable state. Silence here would put the region back into the
    single-channel fiction this layer exists to end.
    """
    history = getattr(inst, "recognition_history", None)
    entry = _measuring_entry(history)
    ## A transform's provenance entry names the route but not the engine --
    ## it re-reads through whichever backend the pipeline settled on, and
    ## threading that through every transform signature would buy a field at
    ## the cost of touching seven call sites. Inherit it from the nearest
    ## earlier entry that does name one; "unknown" only when nothing does.
    engine = str(
        (entry or {}).get("engine")
        or next(
            (
                str(item["engine"]) for item in reversed(list(history or ()))
                if isinstance(item, dict) and item.get("engine")
            ),
            "",
        )
        or "unknown"
    )
    route = route_for_stage(
        (entry or {}).get("stage"), (entry or {}).get("pass"),
    )
    transforms = _lineage_transforms(
        lineage_index or {}, getattr(inst, "lineage_candidate_id", None),
    )
    if route == UNREGISTERED:
        ## No traceable recognition stage. Before giving up, ask the lineage
        ## whether a transform re-measured these pixels: a zoom or a merge
        ## mints a fresh detection with no `detection_pass` provenance, so
        ## its read is invisible to the history walk above even though the
        ## transform is precisely the channel that produced it. Innermost
        ## match wins -- REMEASURING_TRANSFORMS is ordered latest-first, so
        ## a region that was zoomed and then column-merged is credited to
        ## the merge, which is what actually read it.
        for candidate in REMEASURING_TRANSFORMS:
            if candidate in transforms:
                route = candidate
                break
    box = getattr(inst, "bounding_box", None)
    crop = (
        (int(box.x), int(box.y), int(box.width), int(box.height))
        if box is not None else None
    )
    return Channel(
        engine=engine,
        route=route,
        view="zoom" if "zoom" in transforms else "source",
        transforms=transforms,
        merged=any(name in MERGE_TRANSFORMS for name in transforms),
        crop=crop,
    )


def measure(
    reading: str | None,
    truth: str | None,
    *,
    engine: str,
    channel_id: str,
) -> dict[str, Any]:
    """One D_{e,c}(X): a NED that cannot be emitted without its channel.

    The engine and channel are keyword-only and required precisely because
    the defect being fixed is a NED reported without them. A caller that has
    not established which channel produced a reading has not finished
    measuring it, and should say `unregistered` rather than omit the field.
    """
    from tofu.utils.distance import is_unreadable, normalized_edit_distance

    ned = normalized_edit_distance(reading, truth)
    return {
        "schema": CHANNEL_SCHEMA,
        "engine": engine,
        "channel_id": channel_id,
        "ned": round(ned, 6),
        "unreadable": is_unreadable(ned),
        "reading": reading,
    }


## ---------------------------------------------------------------------------
## The oracle half. Everything below uses t*, and therefore may never run at
## inference.
## ---------------------------------------------------------------------------

## Where an oracle measurement is permitted. The list is the enforcement:
## `oracle_envelope` takes `context` keyword-only with no default, so a
## caller cannot reach it by accident, and a production path that tried
## would have to name itself a training run in source to do it.
ORACLE_CONTEXTS = frozenset({"training", "benchmark", "error_analysis"})


class OracleLeak(RuntimeError):
    """An oracle measurement was requested outside training or benchmarking.

    D*(X) = min_c D_{e,c}(X) is computed FROM the ground truth. It is a
    training reference and a benchmark envelope. It is not a confidence
    score, and a system that selected a channel by it at inference would be
    reporting its own answer key back to itself. This programme has already
    published one invalid oracle arm; the guard is executable so the next
    one fails loudly instead.
    """


def oracle_envelope(
    measurements: Sequence[dict[str, Any]],
    *,
    context: str,
) -> dict[str, Any] | None:
    """D*(X) and c*(X) over one region's channel measurements.

    TRAINING, BENCHMARKING AND ERROR ANALYSIS ONLY -- see :class:`OracleLeak`.

    Its role is to stop a channel-specific failure being misreported as a
    property of the scene, and to expose whether the channel family contains
    a materially better measurement for a stratum. Returns None for an empty
    set rather than a zero: no measurement is not a perfect measurement.
    """
    if context not in ORACLE_CONTEXTS:
        raise OracleLeak(
            f"oracle envelope requested in context {context!r}; permitted only in "
            f"{sorted(ORACLE_CONTEXTS)}. D* is computed from the ground truth "
            f"and is not an inference-time score."
        )
    rows = list(measurements or ())
    if not rows:
        return None
    for row in rows:
        if not row.get("channel_id") or row.get("ned") is None:
            raise ValueError(
                "every measurement must carry a channel_id and a ned; an "
                "envelope over unlabelled measurements is the defect this "
                "layer exists to fix"
            )
    best = min(rows, key=lambda row: (float(row["ned"]), str(row["channel_id"])))
    return {
        "schema": CHANNEL_SCHEMA,
        "oracle": True,
        "context": context,
        "best_ned": float(best["ned"]),
        "best_channel_id": str(best["channel_id"]),
        "channels": len(rows),
        ## Per-channel regret, D_{e,c} - D*. The quantity a learned channel
        ## selector is eventually trained against (S7); reported here only
        ## as a diagnostic, and still oracle-derived.
        "regret": {
            str(row["channel_id"]): round(float(row["ned"]) - float(best["ned"]), 6)
            for row in rows
        },
    }


## Equivalence-class criterion. Consistency between two channels may only be
## required where they are empirically equivalent for a stratum: a strict
## garbling must not be forced to agree with the richer channel, because its
## disagreement is evidence about channel quality rather than a violation.
##
## All three `preregistered`, registered in docs/threshold-register.md, and
## fixed before the first report was read.
EQUIV_MARGIN = 0.05      ## largest mean NED difference still called equivalent
EQUIV_ALPHA = 0.05       ## two-sided exact sign test
EQUIV_MIN_PAIRS = 8      ## below this the verdict is underpowered, not equivalent


def sign_test(differences: Sequence[float]) -> float:
    """Two-sided exact sign test p-value over paired differences.

    Exact rather than normal-approximated because the strata here have
    single-digit region counts, where the approximation is not merely
    imprecise but systematically anticonservative. Ties contribute no
    information and are dropped, which is the standard treatment.
    """
    import math

    nonzero = [value for value in differences if value != 0]
    n = len(nonzero)
    if n == 0:
        return 1.0
    positive = sum(1 for value in nonzero if value > 0)
    tail = min(positive, n - positive)
    cumulative = sum(math.comb(n, k) for k in range(tail + 1)) / (2 ** n)
    return min(1.0, 2 * cumulative)


def equivalence_verdict(
    differences: Sequence[float],
    *,
    margin: float = EQUIV_MARGIN,
    alpha: float = EQUIV_ALPHA,
    min_pairs: int = EQUIV_MIN_PAIRS,
) -> dict[str, Any]:
    """Are two channels interchangeable on this stratum?

    `underpowered` is a distinct verdict and NOT a synonym for `equivalent`
    -- the same rule `decant` holds between `unknown` and `absent`. Treating
    a stratum too small to measure as one where the channels agree is
    precisely how a crop would end up teaching the encoder to agree with
    itself on the script where it is a strict garbling.

    Equivalence requires BOTH a small effect and an undetectable direction:
    a channel that is reliably worse by less than the margin is still worse,
    and the sign test is what notices.
    """
    values = [float(value) for value in differences]
    n = len(values)
    if n < min_pairs:
        return {
            "verdict": "underpowered", "pairs": n, "mean_difference": None,
            "p_value": None, "reasons": ["below_min_pairs"],
            "criterion": {
                "margin": margin, "alpha": alpha, "min_pairs": min_pairs,
            },
        }
    mean_difference = sum(values) / n
    p_value = sign_test(values)
    ## Recorded rather than inferred from the numbers by whoever reads this
    ## next. `not_equivalent` is NOT "certified different": a stratum can
    ## fail on the margin while the sign test says the direction is not even
    ## established. Both are correct grounds to refuse a tie, and they call
    ## for different follow-up -- one needs a better channel, the other
    ## needs more regions.
    reasons = []
    if abs(mean_difference) > margin:
        reasons.append("margin_exceeded")
    if p_value <= alpha:
        reasons.append("direction_detected")
    return {
        "verdict": "not_equivalent" if reasons else "equivalent",
        "pairs": n,
        "mean_difference": round(mean_difference, 6),
        "p_value": round(p_value, 6),
        "reasons": reasons or ["within_margin_and_undirected"],
        "criterion": {"margin": margin, "alpha": alpha, "min_pairs": min_pairs},
    }


def assess_manifest(instances: Any, lineage: Any = None) -> int:
    """Stamp every region with the channel that produced its reading.

    Runs on the settled instance set and changes nothing else -- the same
    contract `decant.assess_manifest` and `ticket.write_tickets` hold to. If
    it ever starts having an opinion about which channel is better, it has
    become the reliability model it exists to precede.
    """
    lineage_index = index_lineage(lineage)
    written = 0
    for inst in instances or []:
        try:
            inst.channel = channel_for(inst, lineage_index).to_dict()
            inst.channel_id = inst.channel["channel_id"]
            written += 1
        except Exception:
            ## Provenance may never fail a detection. A region that could not
            ## be stamped keeps whatever it had, which is None.
            continue
    return written


def manifest_channels(instances: Any) -> dict[str, Any]:
    """Channel coverage for one manifest, for the metrics endpoint."""
    counts: dict[str, int] = {}
    unregistered = 0
    stamped = 0
    total = 0
    for inst in instances or []:
        total += 1
        channel = getattr(inst, "channel", None)
        if not isinstance(channel, dict):
            continue
        stamped += 1
        route = str(channel.get("route") or UNREGISTERED)
        counts[route] = counts.get(route, 0) + 1
        if not channel.get("registered", False):
            unregistered += 1
    return {
        "schema": CHANNEL_SCHEMA,
        "regions": total,
        "stamped": stamped,
        "coverage": stamped / total if total else 0.0,
        "routes": counts,
        "unregistered": unregistered,
    }
