"""Bounded store of recent background evidence, kept per track.

The larder is where you keep what you will need again shortly -- not the whole
harvest. Temporal reconstruction only ever looks a few frames back, so holding
more than that costs memory proportional to the clip for no gain.

Two corrections over the deque this replaces:

1. It stores the RAW crop, before erasure. The old history stored erased crops
   and then compared the current raw crop against one of them, so the measured
   "background stability" was dominated by the very text being removed --
   measured 24.15 against an intended 2.31 on a near-static background, just
   over the 24 threshold, so the temporal path never once fired.

2. It ALIGNS before taking a median. The old code called cv2.resize on stored
   patches, which is not alignment; on any camera move it smeared a translating
   background into the fill. Alignment is estimated on the ring OUTSIDE the
   mask, never on masked pixels, so the text cannot influence its own removal.
"""
from __future__ import annotations

from collections import deque
from typing import Any, Deque, Dict, List, Optional, Tuple
from tofu.core.types import ImageLike

## Three frames is what the median needs to reject a single outlier. More would
## widen the window over which the background must stay still, which on real
## footage costs more than the extra sample buys.
LARDER_DEPTH = 3

## Mean absolute difference, per channel, between the current crop and an
## aligned neighbour. Above this the background is genuinely different -- a
## lighting change, a parallax shift, a passing occluder -- and its pixels are
## not evidence about what sits behind the text.
EVIDENCE_TOLERANCE = 24.0

## ECC refuses to converge on flat crops; below this much structure, alignment
## is meaningless and a pure translation estimate is the honest answer.
MIN_ALIGNMENT_STD = 4.0


class BackgroundLarder:
    """Recent raw crops per track, bounded and evictable."""

    def __init__(self, depth: int = LARDER_DEPTH) -> None:
        self.depth = max(1, int(depth))
        self._store: Dict[str, Deque[Tuple[int, Any]]] = {}

    def __contains__(self, track_id: str) -> bool:
        return bool(self._store.get(track_id))

    def remember(self, track_id: str, frame_index: int, crop: ImageLike) -> None:
        if crop is None or not getattr(crop, "size", 0):
            return
        self._store.setdefault(track_id, deque(maxlen=self.depth)).append((frame_index, crop.copy()))

    def forget(self, track_id: str) -> None:
        self._store.pop(track_id, None)

    def retain(self, track_ids: Any) -> None:
        """Drop tracks that are no longer on screen. Without this the larder
        grows with the number of tracks in the clip rather than the number
        visible at once."""
        keep = set(track_ids)
        for track_id in [t for t in self._store if t not in keep]:
            del self._store[track_id]

    def frames(self, track_id: str) -> List[int]:
        return [index for index, _ in self._store.get(track_id, ())]

    def evidence(self, track_id: str, current: Any, mask: Optional[Any] = None
                 ) -> Optional[Dict[str, Any]]:
        """Aligned temporal median for this crop, or None if the evidence is unfit.

        `mask` marks the pixels being erased; alignment is estimated on
        everything else. Returning None is a real answer -- it means "the
        neighbours do not tell us what is behind this text" -- and the caller
        should fall back to spatial reconstruction rather than fill with
        something plausible-looking.
        """
        import numpy as np
        history = self._store.get(track_id)
        if not history or current is None or not current.size:
            return None

        aligned: List[Any] = []
        alignment: List[Dict[str, Any]] = []
        for frame_index, patch in history:
            warped, model = _align(patch, current, mask)
            if warped is None:
                continue
            difference = _masked_mean_difference(warped, current, mask)
            if difference > EVIDENCE_TOLERANCE:
                ## exposure change, parallax, or something moved through: reject
                ## rather than average an unrelated background into the fill
                continue
            aligned.append(warped)
            alignment.append({"frame": frame_index, "model": model, "difference": difference})
        if not aligned:
            return None
        median = np.median(np.stack(aligned), axis=0).astype(current.dtype)
        spread = float(np.mean([entry["difference"] for entry in alignment]))
        return {"patch": median,
                "frames": [entry["frame"] for entry in alignment],
                "alignment": alignment,
                ## more corroborating frames and closer agreement mean stronger
                ## evidence; this is what the quality gate is handed
                "confidence": max(0.0, min(1.0, (len(aligned) / self.depth) *
                                           (1.0 - spread / EVIDENCE_TOLERANCE)))}


def _masked_mean_difference(left: Any, right: Any, mask: Optional[Any]) -> float:
    import numpy as np
    difference = np.abs(left.astype(np.int16) - right.astype(np.int16))
    if difference.ndim == 3:
        difference = difference.mean(axis=2)
    if mask is not None and mask.shape == difference.shape:
        outside = mask == 0
        if outside.any():
            return float(difference[outside].mean())
    return float(difference.mean())


def _align(patch: ImageLike, current: Any, mask: Optional[Any]) -> Tuple[Optional[Any], str]:
    """Warp `patch` into `current`'s frame using the background around the text.

    ECC (Evangelidis & Psarakis 2008) maximizes correlation directly, needs no
    feature correspondences -- which matter here, because the interesting region
    is a thin ring around a masked hole -- and degrades predictably: when it
    cannot converge we say so instead of returning a confident wrong warp.
    """
    import cv2
    import numpy as np
    if patch.shape != current.shape:
        return None, "shape_mismatch"
    grey_patch = _grey(patch)
    grey_current = _grey(current)
    if min(float(grey_patch.std()), float(grey_current.std())) < MIN_ALIGNMENT_STD:
        ## too flat to align, and too flat for misalignment to matter
        return patch, "identity_flat"
    ## estimate on background only: letting the glyphs vote would align the text
    ## to itself and defeat the point
    ecc_mask = None
    if mask is not None and mask.shape == grey_patch.shape:
        ecc_mask = (mask == 0).astype(np.uint8) * 255
        if int(ecc_mask.sum()) == 0:
            ecc_mask = None
    warp = np.eye(2, 3, dtype=np.float32)
    try:
        criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 1e-4)
        cv2.findTransformECC(grey_current, grey_patch, warp, cv2.MOTION_EUCLIDEAN,
                             criteria, ecc_mask, 5)
    except cv2.error:
        return patch, "identity_unconverged"
    warped = cv2.warpAffine(patch, warp, (patch.shape[1], patch.shape[0]),
                            flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP,
                            borderMode=cv2.BORDER_REPLICATE)
    return warped, "euclidean"


def _grey(patch: ImageLike) -> Any:
    import cv2
    import numpy as np
    if patch.ndim == 3:
        return cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY).astype(np.float32)
    return patch.astype(np.float32)
