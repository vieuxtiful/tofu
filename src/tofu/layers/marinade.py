## 🍢 marinade — what the substrate does to the ink it holds
## vieuxtiful
"""Scene-indexed corruption families, for TRAINING-TIME candidate renderings.

A marinade is not one recipe: the same liquid does different things to
different materials, and how long it is left decides how far it goes. Ink on
a surface is the same story. Paper drinks it and the strokes bleed; paint on
a panel loses amplitude and fades; masonry abrades it and takes bites out of
it. `proof.degrade` already implements those eight processes -- what has
never existed is the map saying WHICH of them a given surface can produce.

WHAT THIS BUYS. The training corpus currently samples three processes
uniformly for every instance, which teaches the encoder that all damage is
equally likely everywhere. The technical paper's eq. (8) asks for
`D_{M,P}(R(g_i); z_D)` instead: corrupt the candidate rendering with the
operator family the OBSERVED surface could actually have applied, so the
gradient carries the reason for the damage rather than only its cost. Stroke
loss on corroded metal and bleed on fibrous paper are different explanations
with different transport signatures, and a model trained on one
undifferentiated blur cannot tell them apart.

THE TABLE IS SCENE'S, NOT OURS. `material_class -> processes` is imported
from `layers/scene.py`, not restated here. A corruption family and the
observation that justifies it drifting apart would mean the encoder learned
a decay Scene never claimed to see. This module adds the two axes Scene does
not carry: the inscription process, and the fallback.

UNKNOWN DEGRADES, IT DOES NOT VETO AND IT DOES NOT INVENT. Where Scene has
no material, or has one it cannot calibrate, the family is the full
unconditioned set -- exactly what training does today. That is the whole
safety property: conditioning can only ever narrow a hypothesis space that
was already being sampled, never assert a corruption nobody observed, and
never remove an instance from training.

MEASURED 2026-08-17, and the reason this module changes nothing yet: across
the 66 annotated regions of the frozen corpus, ZERO inherit a conditionable
material. Fifty land on a surface Scene classified `unknown` and sixteen land
on no surface at all, with `substrate_trust` `unmeasured` for every one. The
conditioned branch below is therefore unreachable on the only corpus that
exists. It is written, registered and tested so that it is correct on the day
reviewed material labels arrive -- not because it is doing anything today.

TRAINING ONLY. These operators corrupt CANDIDATE renderings, which are
hypotheses. Nothing generated here is an observation, and nothing generated
here may re-enter the pipeline as evidence.
"""

from __future__ import annotations

from typing import Any

from tofu.layers.proof import PROCESSES, SEVERITIES
from tofu.layers.scene import DEGRADATION_PRIORS

DEGRADATION_FAMILY_SCHEMA = "scene-degradation-family-v1"

## The fallback: every process `proof.degrade` implements. Used whenever the
## surface cannot narrow it, which on the frozen corpus is every region.
UNCONDITIONED = tuple(PROCESSES)

## How the inscription process narrows a material's family further.
##
## Intersected with the material's own priors rather than replacing them: a
## painted sign that was ENGRAVED can still fade, but it cannot bleed, and
## the material knows about fading while the process knows about bleeding.
## Every entry `reasoned` from what the process physically does to a mark,
## and NONE is measured -- Scene does not infer inscription process at all
## today, so this axis is unreachable in practice and is written for the day
## it is not.
PROCESS_FAMILIES: dict[str, tuple[str, ...]] = {
    ## Ink sits on top and can spread, lift or lose amplitude.
    "paint": ("fade", "bleed", "abrasion"),
    "print": ("fade", "bleed", "resample", "compression"),
    ## The mark is cut into the substrate: it cannot bleed, and what
    ## destroys it is wear and things sitting in it.
    "engraving": ("abrasion", "speckle", "occlusion"),
    "carving": ("abrasion", "speckle", "occlusion"),
    ## Emissive or displayed: no substrate chemistry at all, only the
    ## imaging chain.
    "display": ("blur", "resample", "compression"),
    "neon": ("blur", "compression"),
}

## Material classes whose evidence may condition anything. Deliberately
## excludes `unknown` and `textured_unknown`: a class whose name says it
## could not be identified is not a material, and conditioning on it would
## dress a failed classification as physical knowledge.
CONDITIONABLE = frozenset(DEGRADATION_PRIORS) - {"unknown", "textured_unknown"}


def _unconditioned(material: Any, process: Any, reasons: list[str]) -> dict[str, Any]:
    return {
        "schema": DEGRADATION_FAMILY_SCHEMA,
        "processes": UNCONDITIONED,
        "severities": tuple(SEVERITIES),
        "conditioned": False,
        "material_class": material,
        "inscription_process": process,
        "reasons": reasons,
    }


def family_for(
    material_class: Any = None,
    inscription_process: Any = None,
    substrate_trust: Any = None,
) -> dict[str, Any]:
    """The operator family this surface licenses, and why.

    Returns the processes alongside `conditioned` and a reason, because a
    caller must be able to tell a NARROWED family from the full set that
    happens to be short. A bare tuple cannot say which of the two it is, and
    the difference is the entire claim being made.

    `substrate_trust` is honoured when supplied: Scene reports `unmeasured`
    when it never sampled glyph-excluded substrate, and a material class
    derived from a crop that still contains the glyphs is a measurement of
    the ink, not of what the ink sits on.
    """
    reasons: list[str] = []
    material = str(material_class or "unknown")

    if material not in CONDITIONABLE:
        reasons.append(f"material_not_conditionable:{material}")
    if substrate_trust is not None and substrate_trust != "measured":
        ## Not a veto -- it degrades to the unconditioned set, which is the
        ## same thing an absent material does.
        reasons.append(f"substrate_{substrate_trust}")
    if reasons:
        return _unconditioned(material, inscription_process, reasons)

    processes = tuple(DEGRADATION_PRIORS.get(material, ()))
    process = str(inscription_process) if inscription_process else None
    if process in PROCESS_FAMILIES:
        narrowed = tuple(
            name for name in processes if name in PROCESS_FAMILIES[process]
        )
        ## An empty intersection means the material and the process disagree
        ## about what can happen here. That is a contradiction in the
        ## evidence, not a licence to assert an empty family -- fall back
        ## rather than train on nothing.
        if narrowed:
            processes = narrowed
            reasons.append(f"narrowed_by_process:{process}")
        else:
            ## Keeping the material's family here would privilege one of two
            ## contradicting sources arbitrarily. Fall back to the full set:
            ## incoherent evidence is evidence of nothing in particular.
            return _unconditioned(
                material, inscription_process,
                [*reasons, f"process_contradicts_material:{process}"],
            )
    elif process:
        reasons.append(f"process_unregistered:{process}")

    if not processes:
        return _unconditioned(
            material, inscription_process,
            [*reasons, "material_has_no_registered_priors"],
        )

    return {
        "schema": DEGRADATION_FAMILY_SCHEMA,
        "processes": processes,
        "severities": tuple(SEVERITIES),
        "conditioned": True,
        "material_class": material,
        "inscription_process": inscription_process,
        "reasons": reasons or ["conditioned_on_material"],
    }


def family_for_observation(observation: Any) -> dict[str, Any]:
    """The family for one `SurfaceObservationRef`, as Scene attaches it.

    Reads `calibration_status` as well: an unfitted classifier's label is a
    heuristic guess, and the programme's standing rule is that uncalibrated
    Scene evidence is MISSING evidence rather than weak evidence.
    """
    record = observation if isinstance(observation, dict) else {}
    status = record.get("calibration_status")
    if status not in (None, "calibrated"):
        return _unconditioned(
            record.get("material_class"), record.get("inscription_process"),
            [f"calibration_{status}"],
        )
    return family_for(
        record.get("material_class"),
        record.get("inscription_process"),
        record.get("substrate_trust"),
    )
