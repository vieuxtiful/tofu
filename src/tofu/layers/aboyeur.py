## 🍢 aboyeur -- the barker: calls the order, and checks each plate against it
## vieuxtiful
"""Reconcile requested Blocks against the regions actually captured.

In a brigade the aboyeur reads the ticket out and, as each plate reaches the
pass, confirms it against what was called. `mise` lays out what was ordered;
the user draws the boxes; this says, per Block, what has been found and what
is still missing.

WHAT MAKES THIS DIFFERENT FROM TEXT MATCHING.

Comparing two strings is the easy part and it is not the part that goes
wrong. What goes wrong is ownership -- who may claim which region -- and
every defect this layer is written to avoid is an ownership defect:

* **"PARIS PARIS" is two occurrences.** If one region reading PARIS can
  satisfy both Blocks, a user who asked for two is told they found two while
  one is still on the sign. So a region has AT MOST ONE Block owner,
  globally, and duplicate Blocks must consume distinct evidence.
* **`saisie` is not a Block.** It is an atom of "la première saisie". A
  region reading only `saisie` advances that Block to `partial`; it never
  becomes a thing to find in its own right, and the prompt keeps the whole
  phrase.
* **What the user said outranks what we inferred.** A region drawn while a
  Block was active belongs to that Block. No later automatic pass may
  reassign it, because the automatic pass is the one more likely to be wrong.
* **Uncertainty gets its own answer.** Fuzzy evidence produces `review`,
  never `complete`. The numeric match scores that would justify completing
  on a similarity figure are not calibrated, so completing on one would
  assert a precision this project has not measured.

GEOMETRY IS EVIDENCE, NOT A REQUIREMENT. A phrase can span two faces of a
sign, a corner, or a gap wider than any proximity threshold would allow, and
it has still been located. Nothing here rejects a match for being far from
another one.

WHAT IS NOT HERE. No prompting, no ordering of the user's next action, no
opinion about which Block to show. This answers "where does the ticket
stand"; deciding what to ask for next is the caller's, and keeping the two
apart is what lets the prompt be re-derived from persisted state after a
reload instead of replayed from a counter the reload lost.
"""

from __future__ import annotations

import time
from collections.abc import Mapping
from typing import Any, Dict, List, Optional, Sequence

from tofu.layers.mise import normalize_block_text
from tofu.utils.textmatch import (
    FUZZY_MATCH_THRESHOLD, best_span_similarity, normalize_text,
)

## The five states a Block can be in.
##
## `review` is the one that had to be added. Without it, fuzzy evidence must
## be forced into `partial` -- which strands a user who has in fact drawn the
## right box -- or into `complete`, which claims a match nobody checked. A
## separate state lets the UI ask instead of guessing.
PENDING, PARTIAL, REVIEW, COMPLETE, SKIPPED = (
    "pending", "partial", "review", "complete", "skipped",
)

## Why an atom counts as matched. `exact` is the only basis that may complete
## a Block on its own; the rest corroborate.
EXACT, FUZZY, DICTIONARY, OVERRIDE, USER = (
    "exact", "fuzzy", "dictionary", "override", "user",
)
CORROBORATING = (FUZZY, DICTIONARY)

## How a user can settle a Block themselves, and the status each implies.
## Held apart from `status` so an automatic pass can keep recording what it
## believes without ever overwriting what the user decided.
RESOLUTIONS = {"user_complete": COMPLETE, "user_skipped": SKIPPED}

## Similarity at which a region may corroborate an atom. Shared with
## translation-memory matching deliberately: it is the same question ("is
## this the same string, allowing for noise?"), and a second constant would
## be a second thing to calibrate. It gates `review`, never `complete`, so it
## cannot pass a Block on its own.
FUZZY_FLOOR = FUZZY_MATCH_THRESHOLD

## Atom kinds that are not independently locatable and so cannot hold a Block
## open. "N°" and "5.2" ARE locatable and are not in here -- only atoms that
## are punctuation and nothing else.
UNREQUIRED_KINDS = {"punctuation"}


def _required(atoms: Sequence[Any]) -> List[Any]:
    return [a for a in atoms if a.kind not in UNREQUIRED_KINDS]


def _norm(text: Optional[str]) -> str:
    """Comparison form: `mise`'s whitespace/NFC rules, then casefolded and
    stripped of separators, so "SORTIE" and "sortie ." compare equal.

    Case is folded ONLY here. `raw_text` keeps it, because signage
    capitalisation is evidence about rendering intent and discarding it at
    the source would lose it for good.
    """
    return normalize_text(normalize_block_text(text or ""))


def _region_text(region: Any) -> str:
    """What a region is taken to READ.

    A user-supplied `source_override` beats the recogniser's output: someone
    who retyped the text has said what is on the sign, and the OCR reading is
    the thing they were correcting.

    THE OVERRIDE IS A MAPPING. Every producer -- `forage`'s gt_rescue,
    `cicerone`'s arbitration, `menu` -- writes `{kind, text, ...}`, and this
    used to return that whole dict, which `_norm` then handed to
    `unicodedata.normalize`. One rescued region therefore turned every
    reconcile into a TypeError: Guided capture stamps `gt_rescue` overrides
    during detection, and the very next autosave reconciles. The bare string
    is still accepted because that is what the tests asserted and what a
    hand-edited manifest may hold, and a mapping carrying no usable text
    falls back to the read rather than erasing the evidence.
    """
    override = getattr(region, "source_override", None)
    if isinstance(override, Mapping):
        text = override.get("text")
        if isinstance(text, str) and text.strip():
            return text
    elif isinstance(override, str) and override.strip():
        return override
    return getattr(region, "text", None) or ""


def _match_region(
    atoms: Sequence[Any], matched: set, region_text: str,
) -> Optional[tuple]:
    """Which still-unmatched atoms this ONE region accounts for.

    Three ordering decisions, each of which was a bug before it was a rule:

    * **Any position, not the next one.** Atoms are matched IN ORDER in the
      sense that their order is preserved and duplicates stay distinct --
      not in the sense that the user must draw them left to right. Someone
      who boxes `saisie` first has still found part of "la première
      saisie", and a strict cursor reported that Block as `pending`.
    * **Exact before fuzzy, across the whole Block.** Otherwise a fuzzy hit
      on the first atom pre-empts an exact run later: a region reading the
      entire phrase plus a word of context would match only "la".
    * **Longest run first, earliest start first.** A region reading "la
      première" must consume two atoms rather than one, or the Block
      reports a leftover it has already found.

    A run may only span CONSECUTIVE unmatched atoms; an already-matched atom
    in the middle breaks it, so no atom is ever counted twice.
    """
    total = len(atoms)
    for exact_only in (True, False):
        for start in range(total):
            if start in matched:
                continue
            limit = start
            while limit < total and limit not in matched:
                limit += 1
            for end in range(limit, start, -1):
                run = "".join(a.text for a in atoms[start:end])
                if not _norm(run):
                    continue
                if exact_only:
                    if _norm(run) == _norm(region_text):
                        return (list(range(start, end)), EXACT)
                elif best_span_similarity(run, region_text) >= FUZZY_FLOOR:
                    return (list(range(start, end)), FUZZY)
    return None


def _prior(block: Any) -> Dict[str, Any]:
    return block.detection_assessment or {}


def explicit_ids(block: Any) -> List[str]:
    """Regions the USER tied to this Block, in the order they were drawn."""
    return list(_prior(block).get("explicit_region_ids") or [])


def reconcile(
    blocks: Sequence[Any],
    regions: Sequence[Any],
    *,
    associate: Optional[Dict[str, str]] = None,
) -> List[Dict[str, Any]]:
    """Recompute every Block's assessment from the live manifest.

    Total and idempotent: it reads the current regions plus the explicit
    associations recorded so far and rewrites each `detection_assessment`
    from scratch. Nothing accumulates, so a wrong match is corrected by the
    next run rather than surviving inside a partially-updated record -- and
    a reload can rebuild the whole prompt state without replaying history.

    `associate` adds region -> Block links (the user drew this box while that
    Block was active) before anything is inferred.
    """
    assessments: List[Dict[str, Any]] = []
    now = time.time()

    ## Soft-deleted regions are excluded from coverage: `delete_region` marks
    ## rather than splices, so a Block whose only evidence was deleted must
    ## reopen. Length of the array would never show that.
    active = [r for r in regions if not getattr(r, "excluded", False)]
    by_id = {r.id: r for r in active}

    ## Explicit ownership is settled FIRST, so inference cannot take a region
    ## the user has already spoken for.
    pinned: Dict[str, List[str]] = {}
    owner: Dict[str, str] = {}
    for block in blocks:
        for region_id in explicit_ids(block):
            if region_id in by_id and region_id not in owner:
                owner[region_id] = block.id
                pinned.setdefault(block.id, []).append(region_id)
    for region_id, block_id in (associate or {}).items():
        if region_id in by_id and owner.get(region_id) != block_id:
            ## Re-association takes the region off whoever held it, including
            ## an earlier explicit claim. Only a user action reaches here,
            ## and a user is allowed to change their mind.
            previous = owner.get(region_id)
            if previous:
                pinned[previous] = [r for r in pinned.get(previous, []) if r != region_id]
            owner[region_id] = block_id
            pinned.setdefault(block_id, []).append(region_id)

    for block in blocks:
        prior = _prior(block)
        atoms = list(block.atoms or [])
        required = _required(atoms)

        ## Explicitly-associated regions are tried first, in the order they
        ## were drawn; then anything nobody owns, in reading order.
        mine = [by_id[r] for r in pinned.get(block.id, []) if r in by_id]
        free = [r for r in active if owner.get(r.id) is None]
        free.sort(key=lambda r: (getattr(r, "reading_order", None) is None,
                                 getattr(r, "reading_order", 0) or 0, r.id))

        matched_positions: set = set()
        evidence: List[Dict[str, Any]] = []
        region_ids: List[str] = []
        for region in mine + free:
            if len(matched_positions) >= len(atoms):
                break
            result = _match_region(atoms, matched_positions, _region_text(region))
            if result is None:
                if region.id in pinned.get(block.id, []):
                    ## The user said this box belongs here. Keep the link
                    ## even though the read does not support it: an
                    ## unreadable box is exactly when they need the override.
                    region_ids.append(region.id)
                continue
            positions, basis = result
            for position in positions:
                matched_positions.add(position)
                evidence.append({
                    "atom_id": atoms[position].id, "region_id": region.id,
                    "basis": basis,
                })
            region_ids.append(region.id)
            owner[region.id] = block.id

        ## Reported in ATOM order, not discovery order, so the record does
        ## not depend on which box the user drew first.
        matched_ids = [atoms[p].id for p in sorted(matched_positions)]
        evidence.sort(key=lambda e: [a.id for a in atoms].index(e["atom_id"]))
        matched_set = set(matched_ids)
        remaining = [a for a in required if a.id not in matched_set]
        soft = any(e["basis"] in CORROBORATING for e in evidence)

        if not evidence:
            status = PENDING
        elif remaining:
            status = PARTIAL
        elif soft:
            status = REVIEW
        else:
            status = COMPLETE

        resolution = prior.get("resolution")
        if resolution in RESOLUTIONS:
            status = RESOLUTIONS[resolution]

        assessment = {
            "status": status,
            "region_ids": region_ids,
            "explicit_region_ids": [r for r in pinned.get(block.id, []) if r in by_id],
            "matched_atom_ids": matched_ids,
            "remaining_atom_ids": [a.id for a in remaining],
            "matched_text": " ".join(a.text for a in atoms if a.id in matched_set),
            "remaining_text": " ".join(a.text for a in remaining),
            "evidence": evidence,
            "resolution": resolution,
            "updated_at": now,
        }
        block.detection_assessment = assessment
        assessments.append(assessment)

    return assessments


def resolve(block: Any, resolution: Optional[str]) -> Dict[str, Any]:
    """Record a user's own verdict on a Block, or withdraw one.

    The escape hatch, and not optional polish: without it a sign the
    recogniser cannot read holds the workflow open indefinitely, and the
    user's only exit is to abandon Guided capture.
    """
    if resolution is not None and resolution not in RESOLUTIONS:
        raise ValueError(f"unknown resolution '{resolution}'")
    assessment = dict(block.detection_assessment or {})
    assessment["resolution"] = resolution
    if resolution:
        assessment["status"] = RESOLUTIONS[resolution]
    elif assessment.get("status") in (COMPLETE, SKIPPED):
        ## Withdrawing the override returns the Block to what the evidence
        ## actually supports; the next reconcile() recomputes it properly.
        assessment["status"] = PARTIAL if assessment.get("matched_atom_ids") else PENDING
    assessment.setdefault("updated_at", time.time())
    block.detection_assessment = assessment
    return assessment


def progress(blocks: Sequence[Any]) -> Dict[str, Any]:
    """Two numbers with two different jobs.

    The BAR moves on atom coverage, so drawing a box always advances
    something visible. The LABEL counts Blocks, because "6/8" means what a
    user thinks it means and a weighted fraction does not. Deriving either
    from the other produces the nearly-full bar labelled `0/1` that a
    five-atom Block would otherwise show at four atoms.

    `resolved` includes skipped Blocks -- they no longer ask anything of the
    user -- while `complete` does not, and it is `complete` that any recall
    measurement reads. Skipping has to let the workflow finish without
    inflating what was actually located.
    """
    total_atoms = matched_atoms = 0
    complete = skipped = review = 0
    for block in blocks:
        assessment = block.detection_assessment or {}
        required_ids = {a.id for a in _required(list(block.atoms or []))}
        total_atoms += len(required_ids)
        matched = required_ids & set(assessment.get("matched_atom_ids") or [])
        matched_atoms += len(matched)
        status = assessment.get("status", PENDING)
        if status == COMPLETE:
            complete += 1
        elif status == SKIPPED:
            skipped += 1
            ## A skipped Block's atoms are settled for workflow purposes.
            ## Without this the bar can never fill and "done" never looks
            ## done -- while `complete` stays untouched, so the recall
            ## numerator is unaffected.
            matched_atoms += len(required_ids - matched)
        elif status == REVIEW:
            review += 1
    return {
        "blocks_total": len(blocks),
        "blocks_complete": complete,
        "blocks_skipped": skipped,
        "blocks_review": review,
        "blocks_resolved": complete + skipped,
        "atoms_total": total_atoms,
        "atoms_matched": matched_atoms,
        "fraction": (matched_atoms / total_atoms) if total_atoms else 0.0,
    }


def active_block(blocks: Sequence[Any]) -> Optional[Any]:
    """The first Block still asking for something, in entry order.

    Here rather than in the frontend so the prompt after a reload is the same
    Block the user was looking at before it -- derived from persisted state
    rather than from a counter the reload lost.
    """
    for block in blocks:
        status = (block.detection_assessment or {}).get("status", PENDING)
        if status in (PENDING, PARTIAL, REVIEW):
            return block
    return None
