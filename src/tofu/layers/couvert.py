## 🍢 couvert -- the covers: places numbered in service order, renumbered
## whenever the table changes
## vieuxtiful
"""Make region ids contiguous, and rewrite every reference in the same pass.

THE INVARIANT. A persisted manifest's live regions are exactly
`r1 … rN`, in canonical localization order, with no gaps. Anything that
changes the live set or its order -- detection, a drawn region, a delete, a
merge, a reorder, a snapshot restore, an import, a forage rescue -- leaves
the manifest needing this.

WHY AN ALLOCATOR CANNOT DO THIS. Both creation paths used to number
independently: `add_region` took `max(existing)+1` counting soft-deleted
regions, `forage` took the lowest free slot. So an id depended on which code
made it, and deleting anything burned its number for good. Measured on
prem-sais-gt: seven regions numbered r9–r15, and every semantic unit still
pointing at r1–r8, which no longer existed. A cleverer allocator does not
help -- the ids are already spent by the time it runs, and lowest-free
merely reuses a number that stale metadata still refers to. Contiguity is a
property of the WHOLE manifest, so it has to be restored by one operation
that can see the whole manifest.

WHY SOFT-DELETED REGIONS LEAVE THE ORDINAL SPACE. They must not hold an
ordinal -- deleting r3 of five has to leave r1…r4 -- but they cannot be
dropped either, because `update_region` can set `excluded=False` and bring
one back. They are therefore RETIRED to `x1 … xK`, a namespace the ordinal
rule does not apply to. (`cleanse` skips excluded regions at both of its
mask loops, so nothing is erased on their behalf, despite what
`delete_region`'s docstring claims.) Un-excluding one hands it back to the
ordinal space at the next compaction, which is the correct moment.

DANGLING REFERENCES ARE DETECTED, NEVER DISGUISED. Renumbering live regions
to r1…rN while a stale `r1` reference exists elsewhere would make that
reference silently resolve to a DIFFERENT region -- turning a visible
inconsistency into an invisible one. So dangling references are found before
anything is rewritten, and the caller says what to do about them.

VALIDATION DOES NOT TRUST THIS MODULE'S OWN LIST OF REFERENCE SITES. After
rewriting the sites it knows about, it re-serializes the manifest and scans
for any surviving old id. A reference site added later, in code that has
never heard of this module, therefore fails loudly here instead of rotting
quietly the way `semantic_units` did.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Set

LIVE_PREFIX = "r"
RETIRED_PREFIX = "x"

## An id this module owns. Anything else in an id-shaped field is left
## alone: semantic units are `u1`, atoms are `g1a1`, and renaming those
## here would be this module exceeding its remit.
_OWNED_RE = re.compile(rf"^[{LIVE_PREFIX}{RETIRED_PREFIX}](\d+)$")


def owns(region_id: Any) -> bool:
    return isinstance(region_id, str) and bool(_OWNED_RE.match(region_id))


class CompactionError(RuntimeError):
    """The manifest could not be made to satisfy the invariant."""


@dataclass
class CompactionPlan:
    """What compaction WOULD do, computed before anything is touched."""

    mapping: Dict[str, str] = field(default_factory=dict)
    live: List[str] = field(default_factory=list)
    retired: List[str] = field(default_factory=list)
    ## Ids referenced somewhere in the manifest that no instance carries.
    ## Reported rather than quietly dropped: they are evidence that some
    ## earlier operation rewrote instances without rewriting references.
    dangling: Set[str] = field(default_factory=set)
    ## Old region id -> its renormalized `reading_order` (0…N-1).
    reading_orders: Dict[str, int] = field(default_factory=dict)

    @property
    def changed(self) -> bool:
        return any(old != new for old, new in self.mapping.items())


def _order_key(inst: Any, position: int):
    """Canonical localization order.

    `reading_order` first, because that is the order the pipeline and every
    export already use. Ties break on the CURRENT numeric id, then on
    original list position, so the result is total and stable -- two regions
    with the same reading_order must not swap places depending on dict
    iteration.
    """
    order = getattr(inst, "reading_order", None)
    match = _OWNED_RE.match(getattr(inst, "id", "") or "")
    return (
        order is None,
        order if order is not None else 0,
        int(match.group(1)) if match else 0,
        position,
    )


def plan(manifest: Any) -> CompactionPlan:
    """Compute the old->new map without mutating anything."""
    instances = list(getattr(manifest, "instances", None) or [])

    live_insts = [i for i in instances if not getattr(i, "excluded", False)]
    retired_insts = [i for i in instances if getattr(i, "excluded", False)]

    live_sorted = sorted(
        ((inst, pos) for pos, inst in enumerate(live_insts)),
        key=lambda pair: _order_key(pair[0], pair[1]),
    )
    retired_sorted = sorted(
        ((inst, pos) for pos, inst in enumerate(retired_insts)),
        key=lambda pair: _order_key(pair[0], pair[1]),
    )

    mapping: Dict[str, str] = {}
    live_ids: List[str] = []
    for index, (inst, _pos) in enumerate(live_sorted, start=1):
        new_id = f"{LIVE_PREFIX}{index}"
        mapping[inst.id] = new_id
        live_ids.append(new_id)
    ## `reading_order` carries the SAME kind of damage as the ids, from the
    ## same cause: prem-sais-gt's is `8, 1, 2, 3, 4, 5, 6` -- no 0, no 7,
    ## because `add_region` set it to `len(instances)` while eight
    ## since-purged regions were still counted. Renumbering ids off a field
    ## that is itself gappy would carry the corruption forward, so it is
    ## renormalized to 0…N-1 in the same order the ids take.
    plan_orders = {
        inst.id: position for position, (inst, _pos) in enumerate(live_sorted)
    }
    for index, (inst, _pos) in enumerate(retired_sorted, start=1):
        new_id = f"{RETIRED_PREFIX}{index}"
        mapping[inst.id] = new_id

    known = {getattr(i, "id", None) for i in instances}
    dangling = {
        ref for ref in _referenced_ids(manifest)
        if owns(ref) and ref not in known
    }
    return CompactionPlan(
        mapping=mapping,
        live=live_ids,
        retired=[mapping[i.id] for i, _ in retired_sorted],
        dangling=dangling,
        reading_orders=plan_orders,
    )


## ---------------------------------------------------------------- references


def _assessment_refs(block: Any) -> List[str]:
    assessment = getattr(block, "detection_assessment", None) or {}
    found: List[str] = []
    found.extend(assessment.get("region_ids") or [])
    found.extend(assessment.get("explicit_region_ids") or [])
    for item in assessment.get("evidence") or []:
        if isinstance(item, dict) and item.get("region_id"):
            found.append(item["region_id"])
    return found


def referenced_ids(manifest: Any, *, durable_only: bool = False) -> Set[str]:
    """Every region id the manifest points AT, excluding the instances' own.

    `durable_only` excludes Guided `detection_assessment`, which is DERIVED:
    `aboyeur.reconcile` rewrites it wholesale from the live regions, so a
    stale reference there is not corruption -- it is state that has not been
    recomputed yet, and recomputing it is the fix.

    Everything else IS durable. `semantic_units` and `semantic_assignment`
    are curated records that nothing regenerates, so a reference into
    nowhere there means an earlier write rebuilt the regions and left them
    behind. That is corruption, and it must not be papered over.

    Public because callers outside this module need to inspect references
    without reaching into private helpers -- the repair script did.
    """
    return _referenced(manifest, durable_only=durable_only)


def dangling_ids(manifest: Any, *, durable_only: bool = False) -> Set[str]:
    """References naming a region no instance carries."""
    known = {getattr(i, "id", None) for i in getattr(manifest, "instances", None) or []}
    return {ref for ref in referenced_ids(manifest, durable_only=durable_only)
            if ref not in known}


def _referenced_ids(manifest: Any) -> Set[str]:
    """Backwards-compatible alias for `referenced_ids`."""
    return referenced_ids(manifest)


def _referenced(manifest: Any, *, durable_only: bool = False) -> Set[str]:
    found: Set[str] = set()
    if not durable_only:
        for block in getattr(manifest, "guided_blocks", None) or []:
            found.update(_assessment_refs(block))
    for unit in getattr(manifest, "semantic_units", None) or []:
        found.update(_unit_refs(unit))
    for inst in getattr(manifest, "instances", None) or []:
        assignment = getattr(inst, "semantic_assignment", None)
        if isinstance(assignment, dict):
            target = assignment.get("semantic_region_id")
            if target:
                found.add(target)
    return {ref for ref in found if owns(ref)}


def _unit_refs(unit: Any) -> List[str]:
    data = unit if isinstance(unit, dict) else getattr(unit, "__dict__", {})
    found: List[str] = list(data.get("region_ids") or [])
    return [ref for ref in found if owns(ref)]


def _remap_list(values: Optional[Sequence[str]], mapping: Dict[str, str],
                drop_unknown: bool) -> List[str]:
    result: List[str] = []
    for value in values or []:
        if value in mapping:
            result.append(mapping[value])
        elif not owns(value):
            result.append(value)          ## not ours; leave it exactly as-is
        elif not drop_unknown:
            result.append(value)
    return result


def _apply_to_manifest(manifest: Any, mapping: Dict[str, str],
                       drop_unknown: bool,
                       reading_orders: Optional[Dict[str, int]] = None) -> None:
    for inst in getattr(manifest, "instances", None) or []:
        if reading_orders and inst.id in reading_orders:
            inst.reading_order = reading_orders[inst.id]
        if inst.id in mapping:
            inst.id = mapping[inst.id]
        assignment = getattr(inst, "semantic_assignment", None)
        if isinstance(assignment, dict):
            target = assignment.get("semantic_region_id")
            if target in mapping:
                assignment["semantic_region_id"] = mapping[target]
            elif owns(target) and drop_unknown:
                assignment.pop("semantic_region_id", None)

    for block in getattr(manifest, "guided_blocks", None) or []:
        assessment = getattr(block, "detection_assessment", None)
        if not assessment:
            continue
        assessment["region_ids"] = _remap_list(
            assessment.get("region_ids"), mapping, drop_unknown)
        assessment["explicit_region_ids"] = _remap_list(
            assessment.get("explicit_region_ids"), mapping, drop_unknown)
        evidence = []
        for item in assessment.get("evidence") or []:
            if not isinstance(item, dict):
                continue
            ref = item.get("region_id")
            if ref in mapping:
                evidence.append({**item, "region_id": mapping[ref]})
            elif owns(ref) and drop_unknown:
                continue              ## evidence naming a vanished region
            else:
                evidence.append(item)
        assessment["evidence"] = evidence

    for unit in getattr(manifest, "semantic_units", None) or []:
        if isinstance(unit, dict):
            unit["region_ids"] = _remap_list(
                unit.get("region_ids"), mapping, drop_unknown)
        elif hasattr(unit, "region_ids"):
            unit.region_ids = _remap_list(
                unit.region_ids, mapping, drop_unknown)


## ------------------------------------------------------------------ validate


def validate(manifest: Any) -> None:
    """The invariant, checked rather than assumed.

    Raises `CompactionError` with what is actually wrong, because every
    failure here means some reference in the manifest now points at the
    wrong region -- the one class of bug that is invisible from the UI.
    """
    instances = list(getattr(manifest, "instances", None) or [])
    live = [i.id for i in instances if not getattr(i, "excluded", False)]
    ## Compared as a SET, ordered numerically. The invariant is about which
    ## ordinals exist, not about the order of the instances array -- that
    ## array's order is incidental, and `reading_order` is what carries
    ## localization order. Comparing list order made a correctly compacted
    ## manifest fail whenever reading_order disagreed with array position.
    ordered = sorted(live, key=lambda rid: int(_OWNED_RE.match(rid).group(1))
                     if _OWNED_RE.match(rid) else -1)
    expected = [f"{LIVE_PREFIX}{n}" for n in range(1, len(live) + 1)]
    if ordered != expected:
        raise CompactionError(
            f"live region ids are not contiguous: {ordered} != {expected}")

    all_ids = [i.id for i in instances]
    if len(set(all_ids)) != len(all_ids):
        duplicates = sorted({i for i in all_ids if all_ids.count(i) > 1})
        raise CompactionError(f"duplicate region ids: {duplicates}")

    known = set(all_ids)
    missing = sorted(ref for ref in _referenced_ids(manifest) if ref not in known)
    if missing:
        raise CompactionError(f"references to regions that do not exist: {missing}")


def compact_region_ids(
    manifest: Any,
    *,
    on_dangling: str = "error",
) -> Dict[str, str]:
    """Renumber live regions to `r1…rN` and rewrite every reference.

    Returns the old->new map, which callers need in order to migrate the
    references they own OUTSIDE the manifest -- `tm_records.region_id` above
    all. Ids not in the map were not renamed.

    `on_dangling` decides what happens to references naming a region that no
    instance carries: `error` refuses (the default, because a dangling
    reference is a bug somewhere upstream and renumbering would hide it),
    `drop` removes them, `keep` leaves them and lets `validate` refuse.
    """
    if on_dangling not in {"error", "drop", "keep"}:
        raise ValueError(f"unknown on_dangling policy '{on_dangling}'")

    proposal = plan(manifest)
    if proposal.dangling and on_dangling == "error":
        raise CompactionError(
            "manifest already contains references to regions that do not "
            f"exist: {sorted(proposal.dangling)}. Repair them first, or pass "
            "on_dangling='drop' to discard them."
        )

    _apply_to_manifest(manifest, proposal.mapping,
                       drop_unknown=(on_dangling == "drop"),
                       reading_orders=proposal.reading_orders)

    ## The check that does not depend on this module having enumerated every
    ## reference site correctly: if any id that USED to exist still appears
    ## anywhere in the serialized manifest, something holds a reference this
    ## code does not know how to rewrite.
    renamed = {old for old, new in proposal.mapping.items() if old != new}
    if renamed:
        stale = renamed & _snapshot_ids(manifest)
        stale -= set(proposal.mapping.values())
        if stale:
            raise CompactionError(
                "region ids survive in manifest fields this compactor does "
                f"not rewrite: {sorted(stale)}. Add the site to "
                "couvert._apply_to_manifest."
            )


    validate(manifest)
    return proposal.mapping


def _snapshot_ids(manifest: Any) -> Set[str]:
    """Every owned-looking id appearing anywhere in the serialized manifest.

    Serialized rather than walked field-by-field on purpose: the point is to
    see fields this module has never been told about.
    """
    import json

    from tofu.utils.manifest_store import _manifest_to_dict

    blob = json.dumps(_manifest_to_dict(manifest), default=str)
    return {token for token in re.findall(r'"([rx]\d+)"', blob)}
