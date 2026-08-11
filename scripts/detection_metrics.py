"""Polygon-aware, stratified detection metrics with a failure taxonomy (P1.1).

The harness this replaces reported one number per fixture: mean best IoU,
plus a match count at a single 0.5 threshold. That is enough to say
detection got worse and never enough to say *how*, so every regression
turned into an afternoon of looking at overlays. Three changes:

**Polygon IoU, not bounding-box IoU.** Ground truth is axis-aligned, but
detections are quads, and two fixtures in the corpus are rotated plaques
(`rue-des-martyrs` ~-8 deg, `quai-des-orfevres` ~-16.6 deg). The
axis-aligned box of a rotated quad is substantially larger than the quad,
so box IoU understates overlap on exactly the assets where geometry is
hardest. A detection without a polygon degrades to its box and is *counted*
as degraded, so the report says how much of itself is genuinely polygonal.

**A failure taxonomy, not a match count.** A missed region, a clipped box,
a fragmented one and an over-merged one all decrement recall identically
and need four different fixes. They are separated here.

**Lineage, not inference.** Whether two regions were merged is recorded in
`okara`'s candidate graph. Reconstructing it from final geometry produced
five phantom defects once already, so over-merges are read from the merge
parents and only *localised* with geometry. The distinction the lineage
buys is real and actionable: a box spanning two GT regions with one raw
ancestor is the detector emitting a too-large box, while the same box with
two raw ancestors is the merge policy firing wrongly. Same symptom,
different code, different fix.

Nothing here does I/O or imports the pipeline; it is all pure functions over
geometry so it can be tested without running a detector.
"""
from __future__ import annotations

import random
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from typing import Any

Point = tuple[float, float]
Poly = list[Point]

# A GT region is "touched" by a candidate at or above this overlap. Below it
# the candidate is somewhere else in the image and says nothing about this
# region -- used to separate "missed entirely" from "detected badly".
TOUCH_IOU = 0.10

# Standard operating points. 0.75 is reported alongside 0.5 because a corpus
# can hold recall flat at 0.5 while boxes steadily degrade, and only the
# stricter threshold shows it.
THRESHOLDS = (0.5, 0.75)

# Fraction of the GT region's area a candidate must cover to be a plausible
# clip rather than an unrelated neighbour.
CLIP_MIN_CONTAINMENT = 0.60


# --------------------------------------------------------------- geometry

def rect_to_polygon(box: Sequence[float]) -> Poly:
    """(x, y, w, h) -> closed quad, clockwise."""
    x, y, w, h = (float(v) for v in box)
    return [(x, y), (x + w, y), (x + w, y + h), (x, y + h)]


def polygon_area(poly: Sequence[Point]) -> float:
    """Shoelace. Absolute value, so winding order does not matter."""
    if len(poly) < 3:
        return 0.0
    total = 0.0
    for i in range(len(poly)):
        x1, y1 = poly[i]
        x2, y2 = poly[(i + 1) % len(poly)]
        total += x1 * y2 - x2 * y1
    return abs(total) / 2.0


def clip_polygon(subject: Sequence[Point], clip: Sequence[Point]) -> Poly:
    """Sutherland-Hodgman. `clip` must be convex; quads and rects are.

    Returns the intersection polygon, empty when they do not overlap.
    """
    if len(subject) < 3 or len(clip) < 3:
        return []

    # Normalise winding so the inside test has a consistent sign. Without
    # this a counter-clockwise clip polygon silently returns the empty
    # intersection for every input, which reads as "no overlap anywhere".
    if _signed_area(clip) < 0:
        clip = list(reversed(list(clip)))

    output: Poly = [tuple(map(float, p)) for p in subject]
    for i in range(len(clip)):
        if not output:
            return []
        a = clip[i]
        b = clip[(i + 1) % len(clip)]
        inputs, output = output, []
        for j in range(len(inputs)):
            cur = inputs[j]
            prev = inputs[j - 1]
            cur_in = _is_inside(cur, a, b)
            prev_in = _is_inside(prev, a, b)
            if cur_in:
                if not prev_in:
                    crossing = _intersect(prev, cur, a, b)
                    if crossing is not None:
                        output.append(crossing)
                output.append(cur)
            elif prev_in:
                crossing = _intersect(prev, cur, a, b)
                if crossing is not None:
                    output.append(crossing)
    return output


def _signed_area(poly: Sequence[Point]) -> float:
    total = 0.0
    for i in range(len(poly)):
        x1, y1 = poly[i]
        x2, y2 = poly[(i + 1) % len(poly)]
        total += x1 * y2 - x2 * y1
    return total / 2.0


def _is_inside(point: Point, a: Point, b: Point) -> bool:
    return (b[0] - a[0]) * (point[1] - a[1]) - (b[1] - a[1]) * (point[0] - a[0]) >= 0


def _intersect(p1: Point, p2: Point, a: Point, b: Point) -> Point | None:
    x1, y1 = p1
    x2, y2 = p2
    x3, y3 = a
    x4, y4 = b
    denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
    if abs(denom) < 1e-12:
        return None
    t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / denom
    return (x1 + t * (x2 - x1), y1 + t * (y2 - y1))


def intersection_area(a: Sequence[Point], b: Sequence[Point]) -> float:
    return polygon_area(clip_polygon(a, b))


def polygon_iou(a: Sequence[Point], b: Sequence[Point]) -> float:
    inter = intersection_area(a, b)
    if inter <= 0.0:
        return 0.0
    union = polygon_area(a) + polygon_area(b) - inter
    return inter / union if union > 0 else 0.0


def containment(inner: Sequence[Point], outer: Sequence[Point]) -> float:
    """Fraction of `inner`'s area that lies inside `outer`."""
    area = polygon_area(inner)
    return intersection_area(inner, outer) / area if area > 0 else 0.0


def axis_aligned(poly: Sequence[Point]) -> Poly:
    """The axis-aligned bounding box of a polygon, as a polygon."""
    xs = [p[0] for p in poly]
    ys = [p[1] for p in poly]
    return rect_to_polygon((min(xs), min(ys), max(xs) - min(xs), max(ys) - min(ys)))


def is_axis_aligned(poly: Sequence[Point], tolerance: float = 0.5) -> bool:
    """Is this polygon already its own bounding box?"""
    return abs(polygon_area(poly) - polygon_area(axis_aligned(poly))) <= tolerance


def harmonize(
    candidate: Sequence[Point], gt: Sequence[Point]
) -> tuple[Poly, Poly]:
    """Compare like with like.

    MEASURED, 2026-08-10. Scoring a tight rotated quad against an
    axis-aligned ground-truth box is not a stricter measurement, it is a
    different one, and it penalises the detector for being MORE precise: the
    GT box of a rotated plaque contains large empty corners the quad
    correctly excludes, so the union inflates and IoU falls. On the frozen
    corpus this dropped `latin_plaque` recall at IoU 0.75 from 1.000 to
    0.455 -- a 55-point swing produced entirely by the annotation's
    geometry, with no change to detection.

    So when the ground truth is axis-aligned, the candidate is reduced to
    its own bounding box and both are compared in that space. Polygon-vs-
    polygon scoring only becomes meaningful once the annotations are
    polygonal too, which is P0.3b; until then this keeps the comparison
    honest instead of quietly measuring the annotation format.
    """
    if is_axis_aligned(gt):
        return axis_aligned(candidate), list(gt)
    return list(candidate), list(gt)


# ------------------------------------------------------------------ inputs

@dataclass(frozen=True)
class Candidate:
    """One detection, with whatever geometry it actually has."""

    candidate_id: str
    polygon: Poly
    ## True when `polygon` is the axis-aligned box because the detector gave
    ## no quad. Tracked so a report can say what fraction of its own numbers
    ## are genuinely polygon-scored rather than quietly box-scored.
    degraded_to_box: bool = False
    ## Distinct detector-level ancestors from okara's candidate graph. >1
    ## means a merge was RECORDED, not inferred from the final shape.
    ##
    ## None means UNKNOWN, and is the default on purpose. Defaulting to 1
    ## would silently attribute every over-merge to the detector, which is a
    ## specific accusation made from no evidence -- and it would be made most
    ## often precisely when lineage is broken, i.e. when it is least earned.
    ## Absent evidence gets its own outcome instead; see
    ## OVER_MERGED_UNATTRIBUTED.
    raw_ancestor_count: int | None = None
    text: str | None = None
    confidence: float | None = None


@dataclass(frozen=True)
class GroundTruthRegion:
    region_id: str
    polygon: Poly
    text: str | None = None
    language: str | None = None
    vertical: bool = False

    @property
    def height(self) -> float:
        ys = [p[1] for p in self.polygon]
        return max(ys) - min(ys)


# ------------------------------------------------------------- taxonomy

MATCHED = "matched"
MISSED = "missed"
CLIPPED = "clipped"
FRAGMENTED = "fragmented"
OVER_MERGED_POLICY = "over_merged_policy"      # merge op fired wrongly
OVER_MERGED_DETECTOR = "over_merged_detector"  # detector emitted one big box
OVER_MERGED_UNATTRIBUTED = "over_merged_unattributed"  # no lineage to say which
DISPLACED = "displaced"                        # overlaps, but none of the above

OUTCOMES = (
    MATCHED, MISSED, CLIPPED, FRAGMENTED,
    OVER_MERGED_POLICY, OVER_MERGED_DETECTOR, OVER_MERGED_UNATTRIBUTED,
    DISPLACED,
)

# Every way a region can be over-merged, however attributed. Grouped so a
# report can count the defect without pretending to know its cause.
OVER_MERGED_ANY = (
    OVER_MERGED_POLICY, OVER_MERGED_DETECTOR, OVER_MERGED_UNATTRIBUTED,
)


@dataclass
class RegionOutcome:
    region_id: str
    outcome: str
    best_iou: float
    best_candidate_id: str | None = None
    detail: str = ""
    contributing_candidates: list[str] = field(default_factory=list)


def classify_region(
    gt: GroundTruthRegion,
    candidates: Sequence[Candidate],
    all_gt: Sequence[GroundTruthRegion],
    threshold: float = 0.5,
) -> RegionOutcome:
    """Decide what happened to one GT region, most-specific outcome first.

    Order matters and is not arbitrary. A match is a match regardless of how
    it was reached. Then over-merge, because a candidate spanning two GT
    regions is a distinct defect even though it also looks clipped from each
    region's point of view -- classifying it as clipped would hide the more
    serious problem and suggest the wrong fix.
    """
    def iou_against(c: Candidate) -> float:
        cand_poly, gt_poly = harmonize(c.polygon, gt.polygon)
        return polygon_iou(gt_poly, cand_poly)

    scored = sorted(
        ((iou_against(c), c) for c in candidates),
        key=lambda pair: pair[0],
        reverse=True,
    )
    best_iou, best = (scored[0] if scored else (0.0, None))

    if best is not None and best_iou >= threshold:
        return RegionOutcome(gt.region_id, MATCHED, best_iou, best.candidate_id)

    touching = [(iou, c) for iou, c in scored if iou >= TOUCH_IOU]
    if not touching:
        # Nothing overlaps meaningfully. Distinguish "a candidate covers this
        # region but scores below TOUCH_IOU because it is enormous" from
        # "nothing is here at all" -- the first is still an over-merge.
        swallowing = [
            c for c in candidates
            if containment(gt.polygon, c.polygon) >= CLIP_MIN_CONTAINMENT
        ]
        if swallowing:
            return _over_merge_outcome(gt, swallowing, all_gt, best_iou)
        return RegionOutcome(gt.region_id, MISSED, best_iou, detail="no candidate overlaps")

    # Over-merge: some candidate also covers another GT region.
    spanning = [
        c for _, c in touching
        if _regions_covered(c, all_gt) >= 2
    ]
    if spanning:
        return _over_merge_outcome(gt, spanning, all_gt, best_iou)

    # Fragmented: several candidates each sit inside this region and together
    # cover most of it, but none reaches the threshold alone.
    inside = [c for _, c in touching if containment(c.polygon, gt.polygon) >= 0.80]
    if len(inside) >= 2:
        covered = _union_containment(inside, gt)
        if covered >= threshold:
            return RegionOutcome(
                gt.region_id, FRAGMENTED, best_iou, best.candidate_id,
                detail=f"{len(inside)} candidates cover {covered:.0%} of the region",
                contributing_candidates=[c.candidate_id for c in inside],
            )

    # Clipped: one candidate is mostly inside the region but misses part of it.
    if best is not None and containment(best.polygon, gt.polygon) >= 0.90:
        covered = containment(gt.polygon, best.polygon)
        if covered < threshold:
            return RegionOutcome(
                gt.region_id, CLIPPED, best_iou, best.candidate_id,
                detail=f"candidate lies inside the region but covers only {covered:.0%}",
            )

    return RegionOutcome(
        gt.region_id, DISPLACED, best_iou,
        best.candidate_id if best else None,
        detail="overlaps without matching, clipping, fragmenting or merging",
    )


def _over_merge_outcome(
    gt: GroundTruthRegion,
    spanning: Sequence[Candidate],
    all_gt: Sequence[GroundTruthRegion],
    best_iou: float,
) -> RegionOutcome:
    """Attribute an over-merge to the merge policy or to the detector.

    This is the distinction the candidate lineage exists to make. Inferring
    it from the final box is what produced five phantom defects previously,
    so it is read from the recorded ancestor count and nothing else.
    """
    culprit = max(spanning, key=lambda c: polygon_area(c.polygon))
    covered = _regions_covered(culprit, all_gt)
    ancestors = culprit.raw_ancestor_count

    if ancestors is None:
        return RegionOutcome(
            gt.region_id, OVER_MERGED_UNATTRIBUTED, best_iou, culprit.candidate_id,
            detail=(
                f"candidate spans {covered} annotated regions; no lineage "
                f"available, so whether a merge operation or the detector "
                f"produced it is unknown"
            ),
        )
    if ancestors >= 2:
        return RegionOutcome(
            gt.region_id, OVER_MERGED_POLICY, best_iou, culprit.candidate_id,
            detail=(
                f"candidate merges {ancestors} detector "
                f"proposals and spans {covered} annotated regions"
            ),
        )
    return RegionOutcome(
        gt.region_id, OVER_MERGED_DETECTOR, best_iou, culprit.candidate_id,
        detail=(
            f"single detector proposal spans {covered} annotated regions "
            f"(no merge operation involved)"
        ),
    )


def _regions_covered(candidate: Candidate, all_gt: Sequence[GroundTruthRegion]) -> int:
    return sum(
        1 for g in all_gt
        if containment(g.polygon, candidate.polygon) >= CLIP_MIN_CONTAINMENT
    )


def _union_containment(candidates: Sequence[Candidate], gt: GroundTruthRegion) -> float:
    """Approximate union coverage by summing non-overlapping contributions.

    Exact polygon union is not worth the dependency here: candidates that
    fragment a region are near-disjoint by construction, and the value is
    only compared against a threshold.
    """
    total = sum(intersection_area(c.polygon, gt.polygon) for c in candidates)
    overlap = 0.0
    for i, a in enumerate(candidates):
        for b in candidates[i + 1:]:
            overlap += intersection_area(a.polygon, b.polygon)
    area = polygon_area(gt.polygon)
    return max(0.0, (total - overlap)) / area if area > 0 else 0.0


# --------------------------------------------------------------- aggregation

def bootstrap_ci(
    values: Sequence[float],
    iterations: int = 2000,
    confidence: float = 0.95,
    seed: int = 20260810,
) -> tuple[float, float]:
    """Percentile bootstrap CI for a mean. Seeded, so reports are comparable.

    With 72 regions these intervals are wide. That is the finding, not a
    defect in the method -- reporting a point estimate alone from a corpus
    this size implies a precision it does not have.
    """
    if not values:
        return (float("nan"), float("nan"))
    if len(values) == 1:
        return (float(values[0]), float(values[0]))

    rng = random.Random(seed)
    n = len(values)
    means = []
    for _ in range(iterations):
        means.append(sum(values[rng.randrange(n)] for _ in range(n)) / n)
    means.sort()
    lo = means[int((1 - confidence) / 2 * iterations)]
    hi = means[min(iterations - 1, int((1 + confidence) / 2 * iterations))]
    return (round(lo, 4), round(hi, 4))


def summarize(
    outcomes: Sequence[RegionOutcome],
    partial_gt: bool,
    candidate_count: int,
) -> dict[str, Any]:
    """Aggregate outcomes into a reportable block.

    Precision is deliberately absent when the annotation is partial. Nine of
    the eleven corpus fixtures are partial, which means unannotated text is
    neither credited nor penalised, and any precision computed over them
    counts correct detections of unannotated text as false positives. A
    number that punishes the pipeline for working is worse than no number.
    """
    n = len(outcomes)
    counts = {outcome: 0 for outcome in OUTCOMES}
    for o in outcomes:
        counts[o.outcome] += 1

    hits = [1.0 if o.outcome == MATCHED else 0.0 for o in outcomes]
    recall = sum(hits) / n if n else 0.0
    ious = [o.best_iou for o in outcomes]

    block: dict[str, Any] = {
        "gt_regions": n,
        "candidates": candidate_count,
        "recall": round(recall, 4),
        "recall_ci95": bootstrap_ci(hits),
        "mean_best_iou": round(sum(ious) / n, 4) if n else 0.0,
        "mean_best_iou_ci95": bootstrap_ci(ious),
        "outcomes": counts,
        "partial_gt": partial_gt,
    }

    if partial_gt:
        block["precision"] = None
        block["precision_note"] = (
            "not computed: annotation is partial, so correct detections of "
            "unannotated text would be scored as false positives"
        )
    else:
        matched_candidates = len({
            o.best_candidate_id for o in outcomes
            if o.outcome == MATCHED and o.best_candidate_id
        })
        block["precision"] = (
            round(matched_candidates / candidate_count, 4) if candidate_count else 0.0
        )
    return block


def evaluate_fixture(
    gt_regions: Sequence[GroundTruthRegion],
    candidates: Sequence[Candidate],
    partial_gt: bool,
    thresholds: Iterable[float] = THRESHOLDS,
) -> dict[str, Any]:
    """Full per-fixture report at every threshold."""
    gt_axis_aligned = all(is_axis_aligned(g.polygon) for g in gt_regions)
    report: dict[str, Any] = {
        "candidates_with_polygons": (
            round(
                sum(0.0 if c.degraded_to_box else 1.0 for c in candidates) / len(candidates),
                3,
            )
            if candidates else None
        ),
        # Which space the IoU was actually computed in. Reported because the
        # two are not comparable: see harmonize().
        "scoring_mode": "axis_aligned" if gt_axis_aligned else "polygon",
        "scoring_note": (
            "ground truth is axis-aligned, so candidates are reduced to their "
            "bounding boxes; polygon-vs-box IoU would penalise a tight quad "
            "for the GT box's empty corners. Polygon scoring unlocks with "
            "polygonal annotations (P0.3b)."
            if gt_axis_aligned else
            "ground truth is polygonal; candidate quads scored directly"
        ),
        "by_threshold": {},
    }
    for threshold in thresholds:
        outcomes = [
            classify_region(gt, candidates, gt_regions, threshold)
            for gt in gt_regions
        ]
        block = summarize(outcomes, partial_gt, len(candidates))
        block["regions"] = [
            {
                "region_id": o.region_id,
                "outcome": o.outcome,
                "best_iou": round(o.best_iou, 4),
                "candidate": o.best_candidate_id,
                "detail": o.detail,
            }
            for o in outcomes
        ]
        report["by_threshold"][f"iou_{threshold}"] = block
    return report


def aggregate_strata(
    per_fixture: dict[str, dict[str, Any]],
    strata: dict[str, str],
    threshold_key: str = "iou_0.5",
) -> dict[str, Any]:
    """Roll fixtures into their declared strata.

    Region-weighted, not fixture-weighted: a stratum's recall is over its
    regions, so an 18-region fixture does not carry the same weight as a
    2-region one just because both are one file.
    """
    buckets: dict[str, dict[str, Any]] = {}
    for name, report in per_fixture.items():
        stratum = strata.get(name, "unassigned")
        block = report["by_threshold"][threshold_key]
        bucket = buckets.setdefault(
            stratum,
            {"fixtures": [], "gt_regions": 0, "matched": 0, "hits": [],
             "outcomes": {o: 0 for o in OUTCOMES}},
        )
        bucket["fixtures"].append(name)
        bucket["gt_regions"] += block["gt_regions"]
        bucket["matched"] += block["outcomes"][MATCHED]
        bucket["hits"].extend(
            1.0 if r["outcome"] == MATCHED else 0.0 for r in block["regions"]
        )
        for outcome, count in block["outcomes"].items():
            bucket["outcomes"][outcome] += count

    out: dict[str, Any] = {}
    for stratum, bucket in sorted(buckets.items()):
        hits = bucket.pop("hits")
        n = bucket["gt_regions"]
        out[stratum] = {
            **bucket,
            "recall": round(bucket["matched"] / n, 4) if n else 0.0,
            "recall_ci95": bootstrap_ci(hits),
            # An interval this wide over this few regions cannot support a
            # floor claim; saying so is the whole point of reporting it.
            "sufficient_sample": n >= 30,
        }
    return out


def floor_violations(strata_report: dict[str, Any], floor: float = 0.80) -> list[str]:
    """Strata below the declared recall floor, with sample-size honesty."""
    out = []
    for name, block in sorted(strata_report.items()):
        if block["recall"] < floor:
            qualifier = "" if block["sufficient_sample"] else " [insufficient sample]"
            out.append(
                f"{name}: recall {block['recall']:.3f} < {floor:.2f} "
                f"over {block['gt_regions']} regions{qualifier}"
            )
    return out
