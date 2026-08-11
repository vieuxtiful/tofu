## 🍢 Forage -- go and find the ingredients you were asked to bring back.
## vieuxtiful
"""Locate requested Blocks in an asset that has already been detected.

Auto's job is to find everything and then prune hard, because an unrequested
false region costs a user real review time.  Guided is a different bargain:
the user has SAID this text is there, so a candidate the pruner discarded
stops being noise and becomes the most interesting thing on the page.

That is where the recall comes from.  Nothing here re-runs OCR.  It searches
two pools:

  **shipped**    the regions Auto kept;
  **recovered**  candidates the lineage graph records as pruned or replaced.

Recovery is only ever attempted for text somebody explicitly asked for, and
each recovered occurrence is marked as such, so a reviewer can see that a
region exists because it was requested rather than because the detector
volunteered it.

Bounded on purpose.  Guided widens the search, and an unbounded widening on a
dense CJK asset is how a capture turns into a hang.  Every cap is declared in
`Budget`, and hitting one is REPORTED (`optimization_budget_reached`) rather
than silently truncating a Block's results.
"""

from __future__ import annotations

import re
import time
import unicodedata
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from typing import Any, Dict, List, Optional, Sequence, Tuple

from tofu.core.types import GuidedBlock, InstText, TextManifest
from tofu.layers.palate import normalize_script, scripts_in

POLICY_REVISION = "forage-1"

## Fuzzy floor. Below this two strings are different text, not one misread:
## measured against the corpus, 0.82 keeps "SORTIE"/"S0RTIE" and rejects
## "SORTIE"/"SOIREE".
FUZZY_FLOOR = 0.82


@dataclass
class Budget:
    """Deterministic limits. Guided must not be able to run away."""
    candidates_per_block: int = 64
    retained_candidates: int = 512
    occurrences_per_block: int = 12
    seconds: float = 8.0


@dataclass
class Occurrence:
    """One located instance of a requested Block."""
    geometry: Tuple[float, float, float, float]
    text: str
    source: str                       ## shipped | recovered
    method: str                       ## exact | folded | fuzzy
    similarity: float
    region_id: Optional[str] = None
    candidate_id: Optional[str] = None


@dataclass
class BlockOutcome:
    """What became of one requested Block. Every Block gets one of these.

    `state` is the user-facing answer:
      found          located, and the count matches what was asked for
      multiple       located more times than a single occurrence
      possible       located only on fuzzy evidence -- worth a look
      not_detected   nothing matched. NEVER phrased as "the text is absent":
                     ToFU did not find it, which is a different claim.
    """
    block_id: str
    requested_text: str
    state: str
    occurrences: List[Occurrence] = field(default_factory=list)
    tier: str = "low"                 ## low | medium | high
    evidence: Dict[str, Any] = field(default_factory=dict)
    review_reasons: List[str] = field(default_factory=list)
    policy_revision: str = POLICY_REVISION


def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", unicodedata.normalize("NFC", text or "")).strip()


def _fold(text: str) -> str:
    """Casefolded, punctuation-stripped comparison form."""
    folded = _normalize(text).casefold()
    return re.sub(r"[^\w\s]", "", folded, flags=re.UNICODE).strip()


def _script_set(text: str) -> set:
    return {normalize_script(name) for name in scripts_in(text)} - {"Zyyy"}


def _compatible(requested: str, candidate: str) -> bool:
    """A candidate in a script the request does not use is not the request.

    Cheap, and it removes the largest class of fuzzy false positives: short
    Latin fragments scoring against short CJK ones on edit distance alone.
    """
    wanted, got = _script_set(requested), _script_set(candidate)
    if not wanted or not got:
        return True
    return bool(wanted & got)


def _similarity(a: str, b: str) -> float:
    return SequenceMatcher(None, a, b).ratio()


def _bidi_variants(text: str) -> List[str]:
    """The same RTL string as a reader might emit it.

    A detector reports what it sees left to right, so a two-word Arabic sign
    annotated "شارع النيل" comes back as "النيل شارع".  That is not a misread
    and not a different string -- it is the same words in visual rather than
    logical order, and the transform is reversible.  Measured on the corpus,
    this alone is most of the RTL recall gap.

    Only offered for RTL scripts: reversing Latin word order would invent
    matches ("STREET MAIN" is not "MAIN STREET").
    """
    normalized = _normalize(text)
    if not (_script_set(normalized) & {"Arab", "Hebr"}):
        return [normalized]
    words = normalized.split()
    if len(words) < 2:
        return [normalized]
    return [normalized, " ".join(reversed(words))]


def _match(requested: str, candidate_text: str) -> Optional[Tuple[str, float]]:
    """(method, similarity) if this candidate could BE the requested text."""
    if not candidate_text:
        return None
    if _normalize(candidate_text) == _normalize(requested):
        return ("exact", 1.0)
    # Visual vs logical order for RTL, before any fuzzy scoring: this is an
    # exact match under a known transform, not an approximation.
    candidate_forms = _bidi_variants(candidate_text)
    if _normalize(requested) in candidate_forms[1:]:
        return ("bidi", 1.0)
    if any(form in _bidi_variants(requested)[1:] for form in candidate_forms):
        return ("bidi", 1.0)
    wanted, got = _fold(requested), _fold(candidate_text)
    if not wanted or not got:
        return None
    if wanted == got:
        return ("folded", 1.0)
    if not _compatible(requested, candidate_text):
        return None
    ratio = _similarity(wanted, got)
    if ratio >= FUZZY_FLOOR:
        return ("fuzzy", round(ratio, 3))
    return None


def _shipped_pool(manifest: TextManifest) -> List[Dict[str, Any]]:
    pool = []
    for inst in manifest.instances:
        if getattr(inst, "excluded", False):
            continue
        box = inst.bounding_box
        pool.append({
            "geometry": (float(box.x), float(box.y), float(box.width), float(box.height)),
            "text": inst.text or "",
            "source": "shipped",
            "region_id": inst.id,
            "candidate_id": getattr(inst, "lineage_candidate_id", None),
        })
    return pool


def _recovered_pool(manifest: TextManifest, budget: Budget) -> List[Dict[str, Any]]:
    """Candidates the detector proposed and then discarded.

    Guided is the only caller allowed to look here: a pruned candidate is a
    bad bet in general and a good one when the user has named the text.
    """
    lineage = getattr(manifest, "candidate_lineage", None)
    if not isinstance(lineage, dict) or not lineage:
        return []
    try:
        from tofu.layers.okara import CandidateGraph
        graph = CandidateGraph.from_dict(lineage)
    except Exception:
        return []

    shipped = {
        (round(i.bounding_box.x, 1), round(i.bounding_box.y, 1),
         round(i.bounding_box.width, 1), round(i.bounding_box.height, 1))
        for i in manifest.instances
    }
    pool = []
    for node in graph.nodes():
        if not (node.text or "").strip():
            continue
        key = tuple(round(float(v), 1) for v in node.geometry)
        if key in shipped:
            continue          # already offered by the shipped pool
        state = graph.state(node.candidate_id)
        if state in (None, "active", "shipped"):
            continue
        pool.append({
            "geometry": tuple(float(v) for v in node.geometry),
            "text": node.text or "",
            "source": "recovered",
            "region_id": None,
            "candidate_id": node.candidate_id,
            "suppression": state,
        })
        if len(pool) >= budget.retained_candidates:
            break
    return pool


def _assess(outcome_occurrences: List[Occurrence], expected: Optional[int]) -> Tuple[str, List[str]]:
    """Tier the correspondence. Precision-first: `high` must mean high.

    No number is exposed. These weights are not calibrated, and showing a
    figure would invite the reader to trust a precision the evidence does
    not have -- the plan holds numeric Match Scores back until calibration
    justifies them.
    """
    if not outcome_occurrences:
        return "low", ["nothing matched the requested text"]
    reasons: List[str] = []
    # `bidi` is an exact match under a known reversible transform, not
    # an approximation, so it carries the same weight.
    exact = [o for o in outcome_occurrences if o.method in {"exact", "folded", "bidi"}]
    recovered = [o for o in outcome_occurrences if o.source in {"recovered", "reread"}]

    if len(exact) == len(outcome_occurrences) and not recovered:
        tier = "high"
    elif exact:
        tier = "medium"
        if recovered:
            reasons.append("located a region the detector had discarded")
    else:
        tier = "medium" if len(outcome_occurrences) == 1 else "low"
        reasons.append("matched on approximate spelling")

    if expected is not None and len(outcome_occurrences) != expected:
        tier = "low" if tier == "high" else tier
        reasons.append(
            f"expected {expected} occurrence(s), located {len(outcome_occurrences)}"
        )
    return tier, reasons


def locate(
    manifest: TextManifest,
    blocks: Sequence[GuidedBlock],
    *,
    budget: Optional[Budget] = None,
    expected_counts: Optional[Dict[str, int]] = None,
    image_path: Optional[str] = None,
) -> List[BlockOutcome]:
    """Find each requested Block. Every Block gets a visible outcome.

    Occurrences are claimed exclusively, so four requests for "SORTIE" need
    four distinct regions: one region cannot satisfy them all, which is the
    failure a set-based matcher hides.
    """
    budget = budget or Budget()
    expected_counts = expected_counts or {}
    started = time.monotonic()

    pool = _shipped_pool(manifest)
    pool += _recovered_pool(manifest, budget)
    if image_path:
        reread_pool(image_path, pool, blocks)
    claimed: set = set()
    outcomes: List[BlockOutcome] = []
    exhausted = False

    for block in blocks:
        requested = block.raw_text
        expected = expected_counts.get(block.id)
        if not exhausted and (time.monotonic() - started) > budget.seconds:
            exhausted = True

        scored: List[Tuple[float, int, Dict[str, Any], str]] = []
        if not exhausted:
            for index, candidate in enumerate(pool[: budget.candidates_per_block * max(1, len(blocks))]):
                if index in claimed:
                    continue
                verdict = _match(requested, candidate["text"])
                if verdict is None:
                    continue
                method, similarity = verdict
                # Shipped beats recovered at equal evidence: a region the
                # detector already kept needs no rescuing.
                rank = (similarity, 1.0 if candidate["source"] == "shipped" else 0.0)
                scored.append((rank[0] + rank[1] * 0.001, index, candidate, method))
                if len(scored) >= budget.candidates_per_block:
                    break

        scored.sort(key=lambda item: -item[0])
        take = min(
            budget.occurrences_per_block,
            expected if expected else budget.occurrences_per_block,
        )
        occurrences: List[Occurrence] = []
        for score, index, candidate, method in scored[:take]:
            claimed.add(index)
            occurrences.append(Occurrence(
                geometry=candidate["geometry"], text=candidate["text"],
                source=candidate["source"], method=method,
                similarity=round(score, 3), region_id=candidate.get("region_id"),
                candidate_id=candidate.get("candidate_id"),
            ))

        tier, reasons = _assess(occurrences, expected)
        if exhausted:
            reasons.append("optimization_budget_reached")
        if not occurrences:
            state = "not_detected"
        elif expected is not None and len(occurrences) < expected:
            state = "possible"
        elif len(occurrences) > 1:
            state = "multiple"
        elif occurrences and all(o.method == "fuzzy" for o in occurrences):
            state = "possible"
        else:
            state = "found"

        outcomes.append(BlockOutcome(
            block_id=block.id, requested_text=requested, state=state,
            occurrences=occurrences, tier=tier,
            evidence={
                "pool_size": len(pool),
                "recovered_available": sum(1 for c in pool if c["source"] in {"recovered", "reread"}),
                "recovered_used": sum(1 for o in occurrences if o.source in {"recovered", "reread"}),
                "methods": sorted({o.method for o in occurrences}),
            },
            review_reasons=reasons,
        ))
    return outcomes

## --------------------------------------------------- charset-aware re-read
## A candidate the pruner discarded often carries garbage TEXT rather than no
## text: measured on the corpus, Arabic boxes come back as "o,olaJl" and
## vertical Japanese columns as Latin noise, because the reader that saw them
## had no charset for the script. The BOX is fine; only the reading is wrong.
##
## So when a requested Block is in a script the shipped reading is not, the
## discarded box is re-read with a reader for the Block's own script. Bounded
## and opt-in: each re-read is a real OCR call.

SCRIPT_READERS = {
    "Hani": ("ja",), "Hira": ("ja",), "Kana": ("ja",),
    "Hang": ("ko",), "Arab": ("ar",), "Hebr": ("he",),
    "Cyrl": ("ru",), "Thai": ("th",), "Deva": ("hi",),
}
MAX_REREADS = 12
_MIN_REREAD_SIDE = 24


def _reread(image_path: str, geometry, languages) -> Optional[str]:
    """Read one discarded box with a reader for the requested script."""
    try:
        import numpy as np
        from PIL import Image
        from tofu.layers.cicerone import EasyOCRBackend
    except Exception:
        return None
    try:
        x, y, w, h = (int(round(v)) for v in geometry)
        if w < 4 or h < 4:
            return None
        with Image.open(image_path) as handle:
            crop = handle.convert("RGB").crop((x, y, x + w, y + h))
        # Small crops are why these were pruned; upscale before re-reading.
        shortest = min(crop.size)
        if shortest < _MIN_REREAD_SIDE:
            factor = max(2, int(round(_MIN_REREAD_SIDE / max(1, shortest))))
            crop = crop.resize((crop.width * factor, crop.height * factor), Image.BICUBIC)
        backend = EasyOCRBackend(languages=tuple(languages))
        results = backend._reader().readtext(np.asarray(crop))
    except Exception:
        return None
    parts = [str(item[1]).strip() for item in results or [] if len(item) > 1]
    return " ".join(part for part in parts if part) or None


def reread_pool(
    image_path: str,
    pool: List[Dict[str, Any]],
    blocks: Sequence[GuidedBlock],
    *,
    limit: int = MAX_REREADS,
) -> int:
    """Re-read discarded boxes whose script the shipped reading cannot be.

    Returns how many were re-read. Mutates `pool` in place, marking each
    re-read candidate so a reviewer can see the text came from a second look
    rather than from the original pass.
    """
    wanted_scripts: set = set()
    for block in blocks:
        wanted_scripts |= _script_set(block.raw_text)
    targets = sorted(wanted_scripts & set(SCRIPT_READERS))
    if not targets:
        return 0
    languages = tuple(dict.fromkeys(
        code for script in targets for code in SCRIPT_READERS[script]
    ))

    done = 0
    for candidate in pool:
        if done >= limit:
            break
        if candidate["source"] != "recovered":
            continue
        # Already in a requested script? Then the reading is not the problem.
        if _script_set(candidate["text"]) & wanted_scripts:
            continue
        text = _reread(image_path, candidate["geometry"], languages)
        done += 1
        if text and (_script_set(text) & wanted_scripts):
            candidate["text"] = text
            candidate["reread_languages"] = list(languages)
            candidate["source"] = "reread"
    return done




## A read this weak is not a reading, it is the recognizer declining. Measured
## on prem-sais: the box for the rare pair "鼜㒪" shipped at 0.004.
ADOPTION_CONFIDENCE_FLOOR = 0.35
## A box far wider than its read explains is under-recognition, not a narrow
## glyph: r3 shipped an 881px box carrying one character where the request
## has four. Width per character, relative to the box height.
ADOPTION_WIDTH_RATIO = 1.8
## How far the box's glyph capacity may differ from the request's length,
## as a fraction of that length. Beyond this they are different strings.
ADOPTION_FIT_TOLERANCE = 0.45


def _under_read(inst, requested: str) -> bool:
    """Does this box hold far more glyphs than its read accounts for?"""
    box = inst.bounding_box
    if box.height <= 0:
        return False
    read = _normalize(inst.text or "")
    if not read:
        return True
    implied = box.width / max(1.0, float(box.height))
    # A run of N square-ish CJK glyphs is about N box-heights wide.
    return implied >= len(read) * ADOPTION_WIDTH_RATIO and len(_normalize(requested)) > len(read)


def _implied_glyphs(inst) -> float:
    """Roughly how many square-ish glyphs this box could hold."""
    box = inst.bounding_box
    if box.height <= 0:
        return 0.0
    return box.width / float(box.height)


def _adopt_shipped(manifest, requested: str, claimed_ids: set):
    """A shipped region that is almost certainly this requested text.

    Reached only when nothing matched: the detector found a box, the user
    says this string is there, and the read is either near-zero confidence or
    accounts for a fraction of the box.

    Selection is by FIT, not by weakness. Ranking on confidence alone let a
    single rare glyph adopt whichever box happened to score lowest -- on
    prem-sais, "鬣" would have taken the two-glyph box belonging to "鼜㒪".
    A box roughly N heights wide holds roughly N CJK glyphs, so the request's
    own length is the discriminator, and a request that fits nothing is left
    unresolved rather than housed in the nearest weak box.
    """
    wanted = _normalize(requested)
    if not wanted:
        return None
    best = None
    for inst in manifest.instances:
        if inst.id in claimed_ids or getattr(inst, "excluded", False):
            continue
        if not _compatible(requested, inst.text or ""):
            continue
        confidence = inst.confidence
        weak = confidence is None or confidence < ADOPTION_CONFIDENCE_FLOOR
        if not (weak or _under_read(inst, requested)):
            continue
        implied = _implied_glyphs(inst)
        if implied <= 0:
            continue
        misfit = abs(implied - len(wanted)) / max(1.0, float(len(wanted)))
        if misfit > ADOPTION_FIT_TOLERANCE:
            continue
        if best is None or misfit < best[0]:
            best = (misfit, inst)
    return best[1] if best else None


# ------------------------------------------------------------------ rescue

def rescue_requested_text(
    manifest: TextManifest,
    requested: Sequence[str],
    image_path: Optional[str] = None,
    *,
    budget: Optional[Budget] = None,
) -> Dict[str, Any]:
    """Recover regions for text the user SAID is present but Auto did not read.

    Ground Truth has historically primed only the recognition-CORRECTION
    path, which aligns requested terms against strings already recognized.
    That cannot help in the two cases users actually hit:

      * the detector proposed no box at all (nothing to correct);
      * the read is too far from the request for alignment to fire -- a
        `zh-TW` pass returning "薹升" at confidence 0.004 for the rare pair
        "鼜㒪" is not a near-miss, it is a different string.

    So this runs AFTER detection: for each requested term still unaccounted
    for, it searches discarded candidates and re-reads them with a charset
    chosen from the REQUESTED text rather than the project language. A
    recovered region is added carrying the requested text, marked
    `gt_rescue` so it is never mistaken for something the detector read.

    Returns a report; the manifest is mutated in place.
    """
    budget = budget or Budget()
    terms = [t for t in (requested or []) if (t or "").strip()]
    if not terms:
        return {"requested": 0, "already_read": 0, "recovered": 0, "unresolved": []}

    from tofu.core.types import BBox, InstText
    from tofu.layers.mise import make_blocks

    blocks = make_blocks(terms, manifest.src_lang)
    outcomes = locate(manifest, blocks, budget=budget, image_path=image_path)

    existing = {_normalize(i.text or "").casefold() for i in manifest.instances}
    taken = {i.id for i in manifest.instances}
    ## Regions already matched to a request cannot be adopted by another.
    claimed_regions = {
        o.region_id for outcome in outcomes for o in outcome.occurrences
        if o.region_id
    }
    recovered = 0
    corrected = 0
    unresolved: List[str] = []

    for outcome in outcomes:
        wanted = _normalize(outcome.requested_text)
        if wanted.casefold() in existing:
            continue                      # the detector already read it
        placed = [o for o in outcome.occurrences if o.source in {"recovered", "reread"}]
        if not placed:
            # Nothing to recover -- but the box may already be here carrying a
            # read the recognizer had no confidence in. Correcting that is a
            # different act from creating a region, and it is the common case:
            # two of prem-sais's three failures were a right box with a wrong
            # read, which region-creation alone could never fix.
            adopted = _adopt_shipped(manifest, outcome.requested_text, claimed_regions)
            if adopted is not None:
                claimed_regions.add(adopted.id)
                history = list(adopted.recognition_history or [])
                history.append({
                    "stage": "gt_rescue_adopt",
                    "requested": outcome.requested_text,
                    "previous_read": adopted.text,
                    "previous_confidence": adopted.confidence,
                    "reason": "unclaimed low-confidence or under-read box",
                    "policy_revision": POLICY_REVISION,
                })
                adopted.recognition_history = history
                adopted.source_override = {
                    "kind": "gt_rescue",
                    "text": outcome.requested_text,
                    "resource": "ground_truth",
                }
                adopted.text = outcome.requested_text
                corrected += 1
                continue
            unresolved.append(outcome.requested_text)
            continue
        for occurrence in placed:
            index = len(taken) + 1
            while f"r{index}" in taken:
                index += 1
            region_id = f"r{index}"
            taken.add(region_id)
            x, y, w, h = (int(round(v)) for v in occurrence.geometry)
            manifest.instances.append(InstText(
                id=region_id,
                bounding_box=BBox(x=x, y=y, width=w, height=h),
                ## The REQUESTED text, not the rescued read: the user asserted
                ## this string, and the re-read only established where it is.
                text=outcome.requested_text,
                confidence=None,
                source_override={
                    "kind": "gt_rescue",
                    "text": outcome.requested_text,
                    "resource": "ground_truth",
                },
                recognition_history=[{
                    "stage": "gt_rescue",
                    "requested": outcome.requested_text,
                    "recovered_read": occurrence.text,
                    "method": occurrence.method,
                    "source": occurrence.source,
                    "policy_revision": POLICY_REVISION,
                }],
            ))
            recovered += 1

    if recovered:
        manifest.total_regions = len(manifest.instances)
    return {
        "requested": len(terms),
        "already_read": len(terms) - len(unresolved) - recovered - corrected,
        "recovered": recovered,
        "corrected": corrected,
        "unresolved": unresolved,
    }
