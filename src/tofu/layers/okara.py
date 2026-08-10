## 🍢 okara — the pulp the straining left behind, kept rather than thrown out
## vieuxtiful
"""
When soy milk is strained the curd goes forward and the okara -- the bean
pulp -- stays in the cloth. A kitchen that throws it away has decided the
milk was the only thing worth having; one that keeps it can still tell what
the beans were.

This is the candidate lineage graph, and it exists because the pipeline was
throwing the okara away.

THE DEFECT IT ANSWERS, and the one it disproved. Tracing every ground-truth
region on images/la-bastille-1789.jpeg from the detector's own proposals to
the finished manifest, by candidate id:

    region     best raw node   raw IoU   state       final IoU
    BASTILLE   raw_-4c1ca167     0.796   active          0.796
    EN         raw_-8cb0b8d7     0.736   active          0.186   <-- LOST
    1789       raw_-16ec630a     0.572   active          0.572
    ANTOINE    raw_-b66376e9     0.434   active          0.434
    RUE        raw_-b66376e9     0.215   active          0.215
    St         raw_-b66376e9     0.123   active          0.123

Exactly one region is destroyed between the detector and the manifest. `EN`
arrives at IoU 0.736 and ships at 0.186, and an independent ablation
confirms the stage: turning the zoom pass off improves that one target and
no other.

Note what RUE, St and ANTOINE share -- a single raw node. CRAFT emitted ONE
component spanning all three, and the pipeline preserved it faithfully,
0.215 to 0.215. That is raw over-grouping, not pipeline destruction.

This module exists because an earlier attribution said otherwise. It
re-ran getDetBoxes at permissive thresholds the pipeline never uses, found
RUE there at IoU 0.735, and concluded ToFU had over-merged six regions
3-7x. Five of the six were artefacts of comparing against a detector
configuration that does not run. Hence run_kind and DetectorConfig below:
"raw" is not a sufficient provenance label when an image can have several
raw proposal sets.

That is what makes this a lineage problem rather than a threshold problem
in both directions. Un-merging a corrupted final rectangle means INVENTING
the parts back; with the parts retained it is SELECTION among geometry the
detector already produced. And a claim about what the pipeline lost is only
checkable against the proposals the pipeline actually had.

THE INVARIANT, and it is the whole design:

    A merge may CREATE a candidate. It must never DESTROY its inputs.

A merged node records its parents; the parents survive with their original
geometry, marked `merged` rather than deleted. Nothing in this module edits
a node after creation -- a transform makes a new node and points back. So
the graph is append-only and a stage cannot silently rewrite history it
disagrees with.

WHAT IS NOT HERE. No selection, no scoring, no opinion about which candidate
should win. This records what happened; choosing among the alternatives is a
later, separately-measured decision. Building the chooser first would repeat
the mistake the scene veto made -- a policy nobody could audit -- one layer
further in.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

## Where a candidate came from. `raw_craft` is the detector's own output --
## the geometry that later stages are answerable to -- and `final` is what
## the manifest shipped. Everything between is a transform that must justify
## itself against its parents.
STAGES = (
    "raw_craft", "multipass_union", "merge_detections", "merge_vertical_columns",
    "prune_contained_fragments", "zoom", "rescue", "final",
)

## What became of a candidate. Only `active` reaches the manifest; the rest
## are retained and selectable, which is the point.
SUPPRESSION = ("active", "merged", "pruned", "replaced")


## Whether a proposal came from the pipeline that actually runs, or from a
## diagnostic re-invocation of the detector at other settings.
##
## This distinction has already cost one wrong conclusion. An earlier
## attribution pass re-ran getDetBoxes at permissive thresholds the pipeline
## never uses, and reported that la-bastille's RUE arrived at IoU 0.735 and
## was then over-merged 3.4x by ToFU. The production ancestry says RUE, St
## and ANTOINE share ONE raw component at IoU 0.215/0.123/0.434 which the
## pipeline preserved unchanged -- the over-merge is CRAFT's, and five of the
## six "pipeline-destroyed" regions never existed.
##
## "Raw" is not a sufficient provenance label. There can be several raw
## proposal sets for one image, and mixing them makes claims about what the
## pipeline lost unfalsifiable.
RUN_KINDS = ("production", "diagnostic")


@dataclass(frozen=True)
class DetectorConfig:
    """Exactly which detector invocation produced a raw proposal.

    Two runs are comparable only if these match. Recorded on every raw node
    rather than once per graph because a single detect() makes several
    detector calls -- three threshold rungs, plus any rescue pass with a
    different language set -- and a node that cannot name its own pass
    cannot be checked against another run's.
    """
    engine: str = "unknown"
    languages: Tuple[str, ...] = ()
    pass_tag: Optional[str] = None
    text_threshold: Optional[float] = None
    low_text: Optional[float] = None
    link_threshold: Optional[float] = None
    mag_ratio: Optional[float] = None
    canvas_size: Optional[int] = None

    def fingerprint(self) -> str:
        parts = [
            self.engine, ",".join(self.languages), str(self.pass_tag),
            *(f"{v:.4f}" if isinstance(v, float) else str(v) for v in (
                self.text_threshold, self.low_text, self.link_threshold,
                self.mag_ratio, self.canvas_size))
        ]
        return hashlib.sha1("|".join(parts).encode("utf-8")).hexdigest()[:10]


@dataclass(frozen=True)
class CandidateNode:
    """One proposal, immutable once recorded.

    Frozen deliberately. The defect this module exists for was geometry
    being changed underneath a candidate that had already been judged; a
    dataclass that cannot be mutated makes that a type error rather than a
    debugging session.

    Nothing mutates a node after insertion -- not even suppression, which is
    held in a separate map on the graph. An earlier draft of this reached in
    with object.__setattr__ to attach a suppression reason, which is exactly
    the history-rewriting the frozen decorator was chosen to prevent.
    """
    candidate_id: str
    image_id: str
    stage: str
    geometry: Tuple[float, float, float, float]      # x, y, w, h
    polygon: Optional[Tuple[Tuple[float, float], ...]] = None
    parent_candidate_ids: Tuple[str, ...] = ()
    operation_reason: Optional[str] = None
    operation_features: Optional[Dict[str, Any]] = None
    text: Optional[str] = None
    confidence: Optional[float] = None
    run_kind: str = "production"
    detector: Optional[DetectorConfig] = None
    ## Re-evaluated against the scene surfaces using THIS node's geometry,
    ## never inherited. A merge or an expansion physically moves the box, so
    ## a parent's verdict describes a different shape and copying it forward
    ## would assert something nobody measured. Always populated with an
    ## explicit state -- `evaluated`, `no_surfaces`, `filter_disabled` or
    ## `not_evaluated` -- and never left None.
    scene_eligibility: Optional[Dict[str, Any]] = None

    @property
    def area(self) -> float:
        return float(self.geometry[2] * self.geometry[3])


def _stable_id(
    image_id: str, stage: str, geometry: Sequence[float],
    parents: Sequence[str] = (), salt: str = "", run_kind: str = "production",
) -> str:
    """Deterministic id from content, not a counter or a uuid.

    Determinism is a stated requirement: the same pixels and the same
    configuration must produce identical ids, or two runs cannot be compared
    and a regression test cannot name a node. A uuid or a counter would make
    every run incomparable to every other, and would do so silently.

    Canonicalization rules, each of which is a way this has been got wrong:
      * geometry rounded to a fixed 2dp, so float formatting cannot differ
        between hosts;
      * parents SORTED, so an unordered set iterated in a different order
        does not mint a different id for the same merge;
      * run_kind included, so a diagnostic re-run of the detector can never
        collide with the production node at the same coordinates;
      * nothing process-local -- no address, no wall clock, no counter.
    """
    key = "|".join((
        image_id, stage, run_kind,
        ",".join(f"{float(v):.2f}" for v in geometry),
        ",".join(sorted(parents)),
        salt,
    ))
    return f"{stage[:4]}-{hashlib.sha1(key.encode('utf-8')).hexdigest()[:12]}"


class CandidateGraph:
    """Append-only DAG of every proposal and every transform between them."""

    def __init__(self, image_id: str = "asset", run_kind: str = "production",
                 eligibility: Optional[Any] = None):
        """`eligibility` is a callable (x, y, w, h) -> dict, or None.

        Injected rather than imported, for two reasons. It keeps this module
        free of any knowledge of cicerone (which imports this one), and more
        importantly it makes re-evaluation STRUCTURAL: add() calls it for
        every node, so no call site can create a candidate and forget to
        evaluate it. The alternative -- asking fifteen call sites to pass a
        verdict -- fails the first time someone adds a sixteenth.
        """
        if run_kind not in RUN_KINDS:
            raise ValueError(f"unknown run_kind {run_kind!r}")
        self.image_id = image_id
        self.run_kind = run_kind
        self._eligibility = eligibility
        self._nodes: Dict[str, CandidateNode] = {}
        self._children: Dict[str, List[str]] = {}
        ## Suppression is the ONE mutable fact, and it lives here rather than
        ## on the node for that reason: marking a parent `merged` must not
        ## require touching the node whose immutability everything else
        ## depends on.
        self._state: Dict[str, str] = {}
        self._suppression_reason: Dict[str, str] = {}
        self._replaced_by: Dict[str, str] = {}

    # --- recording ------------------------------------------------------

    def add(
        self,
        stage: str,
        geometry: Sequence[float],
        *,
        polygon: Optional[Sequence[Sequence[float]]] = None,
        parents: Sequence[str] = (),
        reason: Optional[str] = None,
        features: Optional[Dict[str, Any]] = None,
        text: Optional[str] = None,
        confidence: Optional[float] = None,
        salt: str = "",
        detector: Optional[DetectorConfig] = None,
    ) -> str:
        """Record a proposal. Returns its id; an identical one is idempotent.

        Parents are stored in the order given -- a column merge's members are
        top-to-bottom and that ordering is meaningful -- while the ID hashes
        them sorted, so ordering carries information without making the id
        depend on it.
        """
        geom = tuple(float(v) for v in geometry)[:4]
        cid = _stable_id(self.image_id, stage, geom, parents, salt, self.run_kind)
        if cid not in self._nodes:
            self._nodes[cid] = CandidateNode(
                candidate_id=cid, image_id=self.image_id, stage=stage, geometry=geom,
                polygon=tuple(tuple(float(c) for c in p) for p in polygon) if polygon else None,
                parent_candidate_ids=tuple(parents),
                operation_reason=reason, operation_features=features,
                text=text, confidence=confidence,
                run_kind=self.run_kind, detector=detector,
                scene_eligibility=self._evaluate(geom, confidence),
            )
            self._state[cid] = "active"
            for p in parents:
                self._children.setdefault(p, []).append(cid)
        return cid

    def suppress(self, candidate_id: str, state: str, reason: Optional[str] = None,
                 replaced_by: Optional[str] = None) -> None:
        """Mark what became of a candidate. Never removes it, never edits it.

        `pruned` is included on purpose: a stage that drops a region has to
        say so and say why, or the graph would show a candidate that simply
        stops existing -- the silent loss this module exists to prevent.

        `replaced_by` names the winner for NMS and replacement operations,
        which is what lets a reviewer ask "what beat this, and was it better
        localized?" -- the audit the cross-pass NMS path needs.
        """
        if state not in SUPPRESSION:
            raise ValueError(f"unknown suppression state {state!r}")
        if candidate_id not in self._nodes:
            return
        self._state[candidate_id] = state
        if reason:
            self._suppression_reason[candidate_id] = reason
        if replaced_by:
            self._replaced_by[candidate_id] = replaced_by

    def _evaluate(self, geometry: Sequence[float],
                  confidence: Optional[float]) -> Dict[str, Any]:
        """Scene eligibility for THIS geometry, computed not inherited.

        Returns an explicit state in every branch. `not_evaluated` is a real
        answer meaning "nobody measured this", and a consumer that cannot
        distinguish it from "measured and eligible" will trust a signal that
        was never computed -- the same failure the scene veto had before it
        recorded its own verdicts.
        """
        if self._eligibility is None:
            return {"state": "not_evaluated", "would_veto": None, "vetoed": False}
        try:
            verdict = dict(self._eligibility(geometry, confidence or 0.0))
        except Exception:
            return {"state": "not_evaluated", "would_veto": None, "vetoed": False}
        verdict.setdefault("state", "evaluated")
        verdict.setdefault("vetoed", False)
        return verdict

    def suppression_reason(self, candidate_id: str) -> Optional[str]:
        return self._suppression_reason.get(candidate_id)

    def replaced_by(self, candidate_id: str) -> Optional[str]:
        return self._replaced_by.get(candidate_id)

    # --- querying -------------------------------------------------------

    def __len__(self) -> int:
        return len(self._nodes)

    def get(self, candidate_id: str) -> Optional[CandidateNode]:
        return self._nodes.get(candidate_id)

    def state(self, candidate_id: str) -> Optional[str]:
        return self._state.get(candidate_id)

    def nodes(self, stage: Optional[str] = None) -> List[CandidateNode]:
        return [n for n in self._nodes.values() if stage is None or n.stage == stage]

    def children(self, candidate_id: str) -> List[str]:
        return list(self._children.get(candidate_id, ()))

    def ancestors(self, candidate_id: str) -> List[CandidateNode]:
        """Every proposal this one was built from, nearest first."""
        seen, out, queue = {candidate_id}, [], list(
            self._nodes.get(candidate_id, CandidateNode("", "", "", (0, 0, 0, 0))).parent_candidate_ids
        )
        while queue:
            cid = queue.pop(0)
            if cid in seen or cid not in self._nodes:
                continue
            seen.add(cid)
            node = self._nodes[cid]
            out.append(node)
            queue.extend(node.parent_candidate_ids)
        return out

    def raw_ancestors(self, candidate_id: str) -> List[CandidateNode]:
        """The detector's own proposals underneath a candidate.

        These are the alternatives a repair hypothesis starts from: for
        la-bastille's over-merged rectangle they are the correctly-sized LA,
        RUE and EN boxes CRAFT produced, retrieved rather than reconstructed.
        """
        return [n for n in self.ancestors(candidate_id) if n.stage == "raw_craft"]

    def alternatives(self, candidate_id: str) -> List[CandidateNode]:
        """Selectable geometry for a candidate: itself plus every ancestor.

        The merge-reversibility guarantee, expressed as a query. A merged
        candidate's parents are always available here, which is what makes
        "do not merge this" a choice rather than an inference.
        """
        node = self._nodes.get(candidate_id)
        return ([node] if node else []) + self.ancestors(candidate_id)

    # --- audit ----------------------------------------------------------

    def conservation_report(self) -> Dict[str, Any]:
        """Does every candidate trace to a raw proposal, and vice versa?

        The invariant this module claims, checked rather than asserted.
        """
        raw = [n for n in self._nodes.values() if n.stage == "raw_craft"]
        orphans = [
            n.candidate_id for n in self._nodes.values()
            if n.stage != "raw_craft" and not self.raw_ancestors(n.candidate_id)
        ]
        terminal = [n.candidate_id for n in raw if not self._children.get(n.candidate_id)
                    and self._state.get(n.candidate_id) == "active"]
        return {
            "nodes": len(self._nodes),
            "raw": len(raw),
            "orphans": orphans,          # a node with no raw ancestry
            "raw_untraced": terminal,    # a raw node that never went anywhere
            "by_state": {s: sum(1 for v in self._state.values() if v == s)
                         for s in SUPPRESSION},
            "by_stage": {s: sum(1 for n in self._nodes.values() if n.stage == s)
                         for s in STAGES if any(n.stage == s for n in self._nodes.values())},
        }

    def to_dict(self) -> Dict[str, Any]:
        ## Nodes emitted in id order, not insertion order. A serialization
        ## that depends on dict ordering round-trips fine and compares
        ## differently between runs, which defeats the point of the
        ## deterministic ids it is carrying.
        return {
            "image_id": self.image_id,
            "run_kind": self.run_kind,
            "nodes": [
                {
                    "candidate_id": n.candidate_id, "stage": n.stage,
                    "geometry": list(n.geometry),
                    "parent_candidate_ids": list(n.parent_candidate_ids),
                    "child_candidate_ids": sorted(self.children(n.candidate_id)),
                    "suppression_state": self._state.get(n.candidate_id, "active"),
                    "suppression_reason": self._suppression_reason.get(n.candidate_id),
                    "replaced_by": self._replaced_by.get(n.candidate_id),
                    "operation_reason": n.operation_reason,
                    "operation_features": n.operation_features,
                    "text": n.text, "confidence": n.confidence,
                    "run_kind": n.run_kind,
                    "polygon": [list(p) for p in n.polygon] if n.polygon else None,
                    "detector": _detector_to_dict(n.detector),
                    "detector_fingerprint": n.detector.fingerprint() if n.detector else None,
                    "scene_eligibility": n.scene_eligibility,
                }
                for n in sorted(self._nodes.values(), key=lambda x: x.candidate_id)
            ],
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CandidateGraph":
        """Rebuild a graph from its serialized form.

        The offline hypothesis evaluator reads saved manifests from disk, so
        a graph that cannot come back is a graph that only exists inside the
        process that made it.

        JSON has no tuples. Geometry, polygons and parent lists all arrive as
        lists, and CandidateNode's equality is structural -- so every one has
        to be cast back before construction or a round-tripped node compares
        unequal to the node it was made from, silently, on a field nobody
        looks at. The frozen dataclass is built directly from kwargs for the
        same reason: there is no populate-after-construct path.

        `child_candidate_ids` is NOT read back. It is derived from parent
        edges, and trusting the serialized copy would let a hand-edited file
        describe a graph whose edges disagree with themselves.
        """
        graph = cls(data.get("image_id", "asset"), data.get("run_kind", "production"))
        for raw in data.get("nodes", []):
            cid = raw["candidate_id"]
            graph._nodes[cid] = CandidateNode(
                candidate_id=cid,
                image_id=graph.image_id,
                stage=raw["stage"],
                geometry=tuple(float(v) for v in raw["geometry"]),
                polygon=(tuple(tuple(float(c) for c in p) for p in raw["polygon"])
                         if raw.get("polygon") else None),
                parent_candidate_ids=tuple(raw.get("parent_candidate_ids") or ()),
                operation_reason=raw.get("operation_reason"),
                operation_features=raw.get("operation_features"),
                text=raw.get("text"),
                confidence=raw.get("confidence"),
                run_kind=raw.get("run_kind", graph.run_kind),
                detector=_detector_from_dict(raw.get("detector")),
                scene_eligibility=raw.get("scene_eligibility"),
            )
            graph._state[cid] = raw.get("suppression_state", "active")
            if raw.get("suppression_reason"):
                graph._suppression_reason[cid] = raw["suppression_reason"]
            if raw.get("replaced_by"):
                graph._replaced_by[cid] = raw["replaced_by"]
        # rebuild edges from parents, in the id order to_dict emits
        for node in sorted(graph._nodes.values(), key=lambda x: x.candidate_id):
            for p in node.parent_candidate_ids:
                graph._children.setdefault(p, []).append(node.candidate_id)
        return graph


def _detector_to_dict(cfg: Optional[DetectorConfig]) -> Optional[Dict[str, Any]]:
    if cfg is None:
        return None
    return {
        "engine": cfg.engine, "languages": list(cfg.languages), "pass_tag": cfg.pass_tag,
        "text_threshold": cfg.text_threshold, "low_text": cfg.low_text,
        "link_threshold": cfg.link_threshold, "mag_ratio": cfg.mag_ratio,
        "canvas_size": cfg.canvas_size,
    }


def _detector_from_dict(data: Optional[Dict[str, Any]]) -> Optional[DetectorConfig]:
    if not data:
        return None
    return DetectorConfig(
        engine=data.get("engine", "unknown"),
        languages=tuple(data.get("languages") or ()),
        pass_tag=data.get("pass_tag"),
        text_threshold=data.get("text_threshold"),
        low_text=data.get("low_text"),
        link_threshold=data.get("link_threshold"),
        mag_ratio=data.get("mag_ratio"),
        canvas_size=data.get("canvas_size"),
    )


def merge_features(a: Sequence[float], b: Sequence[float]) -> Dict[str, Any]:
    """Normalized geometry of a proposed union, for the operation record.

    Gaps are expressed in units of the smaller box's height rather than
    pixels: a 40px gap is nothing between storefront letters and enormous
    between two lines of a plaque, and a merge policy that reasons in pixels
    is really reasoning about image resolution.
    """
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    scale = max(1.0, min(ah, bh))
    hx = max(0.0, max(ax, bx) - min(ax + aw, bx + bw))
    hy = max(0.0, max(ay, by) - min(ay + ah, by + bh))
    ix, iy = max(ax, bx), max(ay, by)
    x2, y2 = min(ax + aw, bx + bw), min(ay + ah, by + bh)
    inter = (x2 - ix) * (y2 - iy) if (x2 > ix and y2 > iy) else 0.0
    union = aw * ah + bw * bh - inter
    return {
        "iou": round(inter / union, 4) if union > 0 else 0.0,
        "horizontal_gap_norm": round(hx / scale, 3),
        "vertical_gap_norm": round(hy / scale, 3),
        "area_before": round(aw * ah, 1),
        "area_after": round(union + inter, 1),
    }


## One graph per detection run. A module-level handle rather than a
## parameter threaded through fifteen call sites: the instrumented functions
## are deep inside cicerone and several are called from places that have no
## business knowing a graph exists. Set to None to disable recording, which
## is what every caller that has not opted in gets.
_active: Optional[CandidateGraph] = None


def begin(image_id: str, run_kind: str = "production",
          eligibility: Optional[Any] = None) -> CandidateGraph:
    global _active
    _active = CandidateGraph(image_id, run_kind, eligibility)
    return _active


def scene_evaluator(scene_regions, scene_filter: bool = True):
    """An eligibility callable bound to one run's surfaces.

    Built here so callers hand `begin()` a closure rather than plumbing
    surfaces through every stage, and so the state a node gets reflects the
    run's actual configuration: `no_surfaces` when the pre-pass found none,
    `filter_disabled` when the caller turned the gate off, `evaluated`
    otherwise. Each is a different fact, and a consumer that sees only
    "no verdict" cannot tell them apart.
    """
    from tofu.core.types import BBox
    from tofu.layers.cicerone import _scene_verdict

    def evaluate(geometry, confidence):
        if not scene_regions:
            return {"state": "no_surfaces", "would_veto": None, "vetoed": False}
        if not scene_filter:
            return {"state": "filter_disabled", "would_veto": None, "vetoed": False}
        x, y, w, h = geometry
        verdict = _scene_verdict(
            BBox(x=int(x), y=int(y), width=int(w), height=int(h)),
            confidence, scene_regions,
        )
        verdict["state"] = "evaluated"
        return verdict

    return evaluate


def active() -> Optional[CandidateGraph]:
    return _active


def end() -> Optional[CandidateGraph]:
    global _active
    graph, _active = _active, None
    return graph
