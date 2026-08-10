## 🍢 ticket — the docket that follows each plate to the pass
## vieuxtiful
"""
Every dish leaving a kitchen carries a ticket: what it is, what was done to
it, who is waiting on it. Nobody reads most of them. They exist so that when
something comes back, there is a record of what went out.

This writes the ticket for each candidate region — the feature vector a
review-priority ranker will need — and writes it NOW, before any ranker
exists and before the frontend can report what a reviewer did with the
region. That ordering is deliberate and it is the whole point of the module.

WHY BEFORE THE RANKER. Removing the scene-membership veto recovered four
ground-truth regions across the fixture corpus and grew the candidate set
from 99 to 134. Those 35 extra candidates are real detections, not junk —
garbage fraction barely moved, 0.043 to 0.049 — but they are work a human
must now triage, and review rate on the dense scenes rose with them
(gemini-street 0.444 to 0.622). The right control for that is ORDERING, not
another veto: nothing here may hide, discard, or collapse a candidate.

But a ranker cannot be built until its effect on review effort is
measurable, and that measurement needs reviewer outcomes the application
does not yet record. So the sequence is: keep every candidate, log what each
one looked like, wait for the outcomes, then rank. Building the ranker first
would repeat the veto's original error one layer up — a policy nobody can
price.

Logging the features now rather than with the ranker is what makes the wait
cheap: on the day reviewer outcomes start arriving they join against tickets
already written, instead of starting a collection period from zero.

WHAT IS ON THE TICKET, and where each field comes from:

  surface_overlap        scene_eligibility, already computed by the
                         eligibility gate. A NEGATIVE ranking signal only:
                         non-membership is a decent reason to look at
                         something later and a bad reason to discard it.
  recognition_confidence the recognizer's own estimate. Weak on its own --
                         cicerone's note records a correctly-scripted CJK
                         read at 0.015 against a wrong-charset garbage read
                         at 0.087 -- so it ranks, it does not decide.
  area / aspect_ratio    geometry plausibility. A candidate 400x the median
                         area, or one 30:1, is usually a merge artifact.
  neighbour_iou          overlap with the nearest other candidate, which is
                         how duplicates and fragments announce themselves.
  craft_region / craft_affinity
                         the detector's own evidence at this candidate's
                         coordinates, and the most informative fields here --
                         they separate "the detector is blind to this" from
                         "the detector saw it and something downstream went
                         wrong". Requires an extra CRAFT forward pass, so
                         they are opt-in via TOFU_LOG_CRAFT_SCORES and are
                         None otherwise. Present-but-None rather than absent,
                         for the same reason scene_eligibility records an
                         explicit unknown: a consumer must not be able to
                         confuse "not measured" with "measured low".

None of these are combined into a score here. Weighting them is the ranker's
job, and the ranker does not exist yet.
"""

from __future__ import annotations

import os
from statistics import median
from typing import Any, Dict, List, Optional, Sequence

from tofu.core.types import BBox, ImageLike, InstText

## Sampling CRAFT's score maps costs one extra forward pass over the image.
## That is amortized across every candidate rather than paid per region, but
## it is still real work on a CPU deployment, so it is opt-in: turn it on for
## a data-collection run, leave it off in production until the ranker needs it.
LOG_CRAFT_SCORES = os.environ.get("TOFU_LOG_CRAFT_SCORES", "").strip().lower() in {
    "1", "true", "yes", "on",
}

## Dilation ring around a candidate when sampling the score maps, matching
## scripts/eval_detector_evidence.py. CRAFT's response is not perfectly
## registered to a box, and a peak one pixel outside is evidence, not absence.
SAMPLE_DILATE = 0.15


def _iou(a: BBox, b: BBox) -> float:
    ix, iy = max(a.x, b.x), max(a.y, b.y)
    x2, y2 = min(a.x + a.width, b.x + b.width), min(a.y + a.height, b.y + b.height)
    if x2 <= ix or y2 <= iy:
        return 0.0
    inter = (x2 - ix) * (y2 - iy)
    union = a.width * a.height + b.width * b.height - inter
    return inter / union if union > 0 else 0.0


def _craft_maps(asset: ImageLike, languages: Sequence[str]):
    """CRAFT's region and affinity maps plus the image->map scale, or None.

    Replicates easyocr.detection.test_net's preprocessing so the tensor read
    here is the one the pipeline's own detector consumed. Fails open: a
    missing model, an unreadable asset or a torch error yields None and the
    ticket simply carries no detector evidence.
    """
    try:
        import cv2
        import numpy as np
        import torch
        from easyocr import imgproc

        import easyocr

        reader = easyocr.Reader(list(languages) or ["en"], gpu=False)
        img = imgproc.loadImage(str(asset)) if isinstance(asset, (str, os.PathLike)) else asset
        resized, ratio, _ = imgproc.resize_aspect_ratio(
            img, 2560, interpolation=cv2.INTER_LINEAR, mag_ratio=1.0,
        )
        x = np.transpose(imgproc.normalizeMeanVariance(resized), (2, 0, 1))
        with torch.no_grad():
            y, _ = reader.detector(torch.from_numpy(np.array([x])).to("cpu"))
        out = y[0]
        return out[:, :, 0].cpu().data.numpy(), out[:, :, 1].cpu().data.numpy(), ratio / 2.0
    except Exception:
        return None


def _peak(m, bbox: BBox, scale: float) -> Optional[float]:
    dx, dy = bbox.width * SAMPLE_DILATE, bbox.height * SAMPLE_DILATE
    x0 = int(max(0, (bbox.x - dx) * scale))
    y0 = int(max(0, (bbox.y - dy) * scale))
    x1 = int(min(m.shape[1], (bbox.x + bbox.width + dx) * scale))
    y1 = int(min(m.shape[0], (bbox.y + bbox.height + dy) * scale))
    if x1 <= x0 or y1 <= y0:
        return None
    patch = m[y0:y1, x0:x1]
    return round(float(patch.max()), 4) if patch.size else None


def write_tickets(
    instances: List[InstText],
    asset: ImageLike = None,
    languages: Sequence[str] = (),
) -> int:
    """Attach a review-feature ticket to every candidate. Changes nothing else.

    Returns the number of tickets written. Never reorders, never filters,
    never edits text -- if this function starts having an opinion about which
    candidates matter, it has become the ranker it exists to defer.
    """
    if not instances:
        return 0

    areas = [
        float(i.bounding_box.width * i.bounding_box.height)
        for i in instances if i.bounding_box is not None
    ]
    median_area = median(areas) if areas else 1.0

    maps = None
    if LOG_CRAFT_SCORES and asset is not None:
        maps = _craft_maps(asset, languages)

    written = 0
    for inst in instances:
        box = inst.bounding_box
        if box is None:
            continue
        neighbour = 0.0
        for other in instances:
            if other is inst or other.bounding_box is None:
                continue
            neighbour = max(neighbour, _iou(box, other.bounding_box))

        area = float(box.width * box.height)
        eligibility = inst.scene_eligibility or {}
        region_peak = affinity_peak = None
        if maps is not None:
            region_map, affinity_map, scale = maps
            region_peak = _peak(region_map, box, scale)
            affinity_peak = _peak(affinity_map, box, scale)

        inst.review_features = {
            "candidate_id": inst.id,
            # negative signal for ORDER, never a discard rule
            "surface_overlap": eligibility.get("surface_overlap"),
            "outside_surfaces": eligibility.get("outside_surfaces"),
            "would_veto": eligibility.get("would_veto"),
            "recognition_confidence": round(float(inst.confidence or 0), 4),
            "area": area,
            "area_vs_median": round(area / median_area, 3) if median_area else None,
            "aspect_ratio": round(box.width / box.height, 3) if box.height else None,
            "neighbour_iou": round(neighbour, 3),
            "character_count": len((inst.text or "").strip()),
            # None means NOT MEASURED, not measured-low. See module note.
            "craft_region_peak": region_peak,
            "craft_affinity_peak": affinity_peak,
            "craft_scores_enabled": bool(maps is not None),
        }
        written += 1
    return written
