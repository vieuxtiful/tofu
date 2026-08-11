## 🍢 proof — the printer's proof: what the text looked like before the world got at it
## vieuxtiful
"""Canonical glyph rendering, physical degradation, and hard negatives.

A proof is the reference impression a printer pulls before the run: the mark
as intended, against which every later copy is judged. This renders that
impression for a ground-truth string, then degrades it the way a real sign
degrades, and produces the pairs a degradation-invariant matcher trains on.

WHY RENDER AT ALL. Ground truth arrives as a STRING and the observation is
PIXELS, and the two cannot be compared until one is turned into the other.
Turning pixels into a string is the recognizer's job and is exactly what
fails on degraded text -- so this goes the other way, which is the direction
that stays available when the read has already failed.

FOUR THINGS THIS PROVIDES, and they are the deterministic half of Phase 1:

  render()          the clean mark, at a fair rasterisation budget
  degrade()         what the world does to a mark, by named process
  hard_negatives()  the confusions worth training against, MINED rather
                    than imagined
  match()           the cheapest thing that could possibly work -- rank
                    candidates by silhouette, no training involved

WHAT IT DOES NOT DO. It does not DECIDE. `match` returns support and margin
and stops there; turning those into accept/review/reject is a calibrated
judgement and the thresholds for it do not exist yet.

`match` is deliberately the dumb baseline. Phase 1 proposes a trained
Siamese encoder, and an encoder with no number to beat is an encoder nobody
can evaluate -- this project has an explicit history of interventions that
looked obviously right in the abstract and lost on the measurement
(`docs/measured-dead-ends.md` lists nine).

ON THE DEGRADATION CATALOGUE. Every entry corresponds to a physical process
that actually marks signage, not to a convenient image operation:

    fade          paint/ink loses contrast against its substrate
    bleed         ink spreads into paper fibre or a porous surface
    abrasion      the mark is worn away — stroke thins, then breaks
    blur          optical: focus, motion, or distance
    resample      the mark was photographed at too few pixels
    compression   JPEG, which is what most source photographs are
    occlusion     something is in front of part of the mark
    speckle       sensor noise, grain, corrosion pitting

Naming them after the process rather than the operation is not decoration:
the process is what a material model will later condition on, and a
catalogue keyed to "gaussian blur sigma 1.4" cannot be conditioned on
anything.
"""

from __future__ import annotations

import hashlib
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

## Reused rather than reimplemented. `font_matching._render_mask` renders
## large and reduces to the measured height so every face gets the same
## antialiasing budget, which is a subtlety worth having exactly once: two
## renderers that disagree about rasterisation would make every comparison
## between them meaningless.
from tofu.layers.font_matching import _render_mask as _canonical_mask

## The silhouette confusions cicerone has actually observed, kept as the
## seed of the negative set. Imported rather than copied so a pair added
## there is trained against here without anyone remembering to.
from tofu.layers.cicerone import _GLYPH_CONFUSIONS

## Default render height, in pixels. Matches EasyOCRBackend.MIN_CROP_HEIGHT,
## which is the height below which the recognizer's own reads degrade -- so a
## proof rendered here is at least as legible as anything the pipeline works
## with. `reasoned`; registered in docs/threshold-register.md.
PROOF_HEIGHT = 40

## Degradation severities. Three rungs rather than a continuum because the
## point is to span the range, not to sample it densely; a continuum invites
## fitting a threshold to whichever value happened to help.
LIGHT, MODERATE, SEVERE = "light", "moderate", "severe"
SEVERITIES = (LIGHT, MODERATE, SEVERE)

PROCESSES = (
    "fade", "bleed", "abrasion", "blur", "resample", "compression",
    "occlusion", "speckle",
)


def render(text: str, font_path: str, height: int = PROOF_HEIGHT):
    """The clean mark for `text` in `font_path`, as a boolean mask.

    Returns None when the face cannot render the string -- a missing font, a
    codepoint the face has no glyph for. None means "cannot say", and a
    caller must not read it as "does not match": a face without the glyph is
    evidence about the FACE, not about the text.
    """
    try:
        return _canonical_mask(font_path, text, height)
    except Exception:
        return None


def _rng(seed: Any):
    """Deterministic per-sample randomness.

    Seeded from the caller's key so a training set is reproducible: an
    encoder trained on irreproducible degradations cannot be compared with
    the next one, and comparing arms is the whole point.
    """
    import numpy as np
    digest = hashlib.sha256(str(seed).encode("utf-8")).digest()
    return np.random.default_rng(int.from_bytes(digest[:8], "big"))


def degrade(mask, process: str, severity: str = MODERATE, seed: Any = 0):
    """Apply one named physical process to a glyph mask.

    Takes and returns a float image in [0, 1] rather than a boolean mask:
    every one of these processes destroys the binarity of the mark, and a
    generator that hands back a clean boolean has modelled nothing. That is
    the whole difficulty being simulated.
    """
    try:
        import cv2
        import numpy as np
    except ImportError:
        return None
    if mask is None:
        return None
    if process not in PROCESSES:
        raise ValueError(f"unknown process {process!r}")
    if severity not in SEVERITIES:
        raise ValueError(f"unknown severity {severity!r}")

    rung = SEVERITIES.index(severity) + 1          # 1, 2, 3
    image = mask.astype(np.float32)
    height, width = image.shape[:2]
    rng = _rng((process, severity, seed))

    if process == "fade":
        ## Contrast loss toward the substrate. The mark survives in shape
        ## and dies in amplitude, which is why a shape-based matcher can
        ## still work where a threshold-based reader cannot.
        return image * (1.0 - 0.25 * rung)

    if process == "bleed":
        ## Ink spreading into fibre: strokes thicken and holes close, which
        ## is how 'e' becomes 'o' on absorbent stock.
        kernel = np.ones((1 + 2 * rung,) * 2, np.uint8)
        return cv2.dilate(image, kernel, iterations=1)

    if process == "abrasion":
        ## Wear removes ink. Erode, then punch holes -- a worn mark is not
        ## uniformly thinner, it is thinner AND broken, and only the second
        ## defeats connected-component reasoning.
        kernel = np.ones((1 + 2 * rung,) * 2, np.uint8)
        worn = cv2.erode(image, kernel, iterations=1)
        holes = rng.random((height, width)) < (0.03 * rung)
        worn[holes] = 0.0
        return worn

    if process == "blur":
        sigma = 0.6 * rung
        return cv2.GaussianBlur(image, (0, 0), sigma)

    if process == "resample":
        ## Photographed at too few pixels, then viewed at full size. The
        ## information is gone before any enhancement runs.
        factor = 1 + rung
        small = cv2.resize(image, (max(1, width // factor), max(1, height // factor)),
                           interpolation=cv2.INTER_AREA)
        return cv2.resize(small, (width, height), interpolation=cv2.INTER_LINEAR)

    if process == "compression":
        quality = max(5, 40 - 12 * rung)
        ok, buffer = cv2.imencode(
            ".jpg", (image * 255).astype(np.uint8),
            [int(cv2.IMWRITE_JPEG_QUALITY), quality],
        )
        if not ok:
            return image
        return cv2.imdecode(buffer, cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255.0

    if process == "occlusion":
        ## Something in front of the mark. Placed as a band rather than
        ## scattered: real occluders are objects, and an object takes a
        ## contiguous bite out of a word.
        out = image.copy()
        band = max(1, int(height * 0.15 * rung))
        top = int(rng.integers(0, max(1, height - band)))
        out[top:top + band, :] = 0.0
        return out

    ## speckle
    noise = rng.normal(0.0, 0.08 * rung, size=image.shape).astype(np.float32)
    return np.clip(image + noise, 0.0, 1.0)


def recipes(processes: Optional[Sequence[str]] = None,
            severities: Sequence[str] = SEVERITIES) -> List[Tuple[str, str]]:
    """Every (process, severity) worth generating, in a stable order."""
    return [(p, s) for p in (processes or PROCESSES) for s in severities]


def _confusable_partners(text: str) -> List[str]:
    """Strings one observed silhouette confusion away from `text`.

    Only single-glyph substitutions from the MEASURED set. Deliberately not
    every Unicode confusable: cicerone's note is explicit that its pairs are
    silhouette confusions -- the two glyphs occupy the same stroke skeleton
    in a tight crop -- and pairs that change a word ('rn'/'m') belong to a
    different mechanism with different evidence.
    """
    partners: List[str] = []
    for index, char in enumerate(text):
        for pair in _GLYPH_CONFUSIONS:
            if char in pair:
                for other in pair:
                    if other != char:
                        partners.append(text[:index] + other + text[index + 1:])
    return partners


def hard_negatives(
    text: str,
    corrections: Optional[Iterable[Dict[str, Any]]] = None,
    limit: int = 8,
) -> List[Dict[str, str]]:
    """Wrong readings worth training against, with where each came from.

    Two sources, and the order matters. Recorded corrections come first --
    `memory.remember_corrections` stores `{ocr_text, corrected_text}` for
    every region a human or a later pass moved away from the recognizer's
    read, so each one is a confusion that ACTUALLY HAPPENED in this system,
    on this kind of material. Silhouette pairs come second and are
    synthetic.

    Provenance is returned alongside every negative because the two are not
    equally informative, and a training set that cannot say which is which
    cannot be re-weighted later.
    """
    seen = {text}
    out: List[Dict[str, str]] = []

    for record in corrections or ():
        corrected = (record.get("corrected_text") or "").strip()
        misread = (record.get("ocr_text") or "").strip()
        if corrected != text or not misread or misread in seen:
            continue
        seen.add(misread)
        out.append({"text": misread, "source": "observed_correction"})
        if len(out) >= limit:
            return out

    for partner in _confusable_partners(text):
        if partner in seen:
            continue
        seen.add(partner)
        out.append({"text": partner, "source": "silhouette_confusion"})
        if len(out) >= limit:
            break
    return out


def training_pairs(
    text: str,
    font_path: str,
    corrections: Optional[Iterable[Dict[str, Any]]] = None,
    height: int = PROOF_HEIGHT,
    processes: Optional[Sequence[str]] = None,
) -> Optional[Dict[str, Any]]:
    """One training example: the proof, its degradations, and its confusions.

    Positives are the SAME string under degradation -- what invariance has
    to be learned. Negatives are a DIFFERENT string, degraded identically --
    which is the part that matters, because a negative rendered clean while
    positives are degraded teaches the encoder to detect degradation rather
    than identity. That is the trap this shape exists to avoid.
    """
    anchor = render(text, font_path, height)
    if anchor is None:
        return None

    positives = []
    for process, severity in recipes(processes):
        image = degrade(anchor, process, severity, seed=(text, process, severity))
        if image is not None:
            positives.append({"image": image, "process": process, "severity": severity})

    negatives = []
    for negative in hard_negatives(text, corrections):
        rendered = render(negative["text"], font_path, height)
        if rendered is None:
            continue
        negatives.append({"image": rendered.astype("float32"), "text": negative["text"],
                          "source": negative["source"], "process": None,
                          "severity": None})
        for process, severity in recipes(processes):
            image = degrade(rendered, process, severity,
                            seed=(negative["text"], process, severity))
            if image is not None:
                negatives.append({
                    "image": image, "text": negative["text"],
                    "source": negative["source"],
                    "process": process, "severity": severity,
                })

    return {
        "schema": 1,
        "text": text,
        "font": font_path,
        "anchor": anchor,
        "positives": positives,
        "negatives": negatives,
    }


## --- the deterministic baseline an encoder has to beat ---
##
## Phase 1 of the Vision program proposes a trained Siamese glyph encoder.
## Before one earns its place there has to be a number to beat, and this
## project has an explicit history of interventions that looked obviously
## right in the abstract and lost on the measurement
## (`docs/measured-dead-ends.md` lists nine).
##
## So: the cheapest thing that could possibly work. Render each candidate
## string, compare silhouettes, rank. No training, no data, no weights to
## calibrate -- which also means no way for it to quietly memorise the
## evaluation set.
##
## The comparison is `font_matching._visual_score`, reused rather than
## rewritten. It already combines dice, chamfer, projection and aspect, and
## a second shape comparator that disagreed with the first would make every
## number produced by either meaningless.


def _score(observed, candidate) -> Tuple[float, Dict[str, float]]:
    from tofu.layers.font_matching import _visual_score
    return _visual_score(observed, candidate, typographic=False)


def match(
    observed,
    candidates: Sequence[str],
    font_path: str,
    height: int = PROOF_HEIGHT,
) -> Optional[Dict[str, Any]]:
    """Rank ground-truth candidates by how well each explains the ink.

    Returns support for every candidate AND the margin between the best two,
    because those answer different questions: support says whether anything
    fits, margin says whether the winner is distinguishable from the
    runner-up. A confident-looking top score with no margin is exactly the
    case that should reach a human, and a single scalar cannot express it.

    Returns None when nothing could be rendered -- "cannot say", never
    "no match".
    """
    if observed is None or not candidates:
        return None

    scored: List[Dict[str, Any]] = []
    for text in candidates:
        rendered = render(text, font_path, height)
        if rendered is None:
            ## A face that cannot draw the candidate is evidence about the
            ## face. Recorded, not scored, so it cannot be mistaken for a
            ## candidate that was considered and rejected.
            scored.append({"text": text, "support": None, "components": None,
                           "unrenderable": True})
            continue
        support, components = _score(observed, rendered)
        scored.append({"text": text, "support": round(float(support), 4),
                       "components": components, "unrenderable": False})

    ranked = sorted(
        [s for s in scored if s["support"] is not None],
        key=lambda s: s["support"], reverse=True,
    )
    if not ranked:
        return None

    best = ranked[0]
    runner_up = ranked[1]["support"] if len(ranked) > 1 else None
    return {
        "schema": 1,
        "best": best["text"],
        "support": best["support"],
        "margin": (round(best["support"] - runner_up, 4)
                   if runner_up is not None else None),
        "candidates": scored,
        ## Deliberately no verdict. Turning support and margin into
        ## accept/review/reject is a calibrated decision, and the thresholds
        ## for it do not exist yet -- inventing them here would be the same
        ## error `docs/threshold-register.md` was written to expose.
    }
