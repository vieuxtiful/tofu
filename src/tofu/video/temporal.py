"""Pure temporal decisions used by workers and deterministic tests."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from difflib import SequenceMatcher
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from .types import RenderKeyframe, TrackObservation

## Distinct readings kept per track before the weakest is evicted. Eight covers
## the realistic spread -- a genuine sign plus OCR variants of it -- while
## keeping the summary O(1) in clip length.
CONSENSUS_TOP_K = 8


def bbox_iou(a: Dict[str, float], b: Dict[str, float]) -> float:
    ax1, ay1 = a["x"], a["y"]
    bx1, by1 = b["x"], b["y"]
    ax2, ay2 = ax1 + a["width"], ay1 + a["height"]
    bx2, by2 = bx1 + b["width"], by1 + b["height"]
    area = max(0.0, min(ax2, bx2) - max(ax1, bx1)) * max(0.0, min(ay2, by2) - max(ay1, by1))
    union = a["width"] * a["height"] + b["width"] * b["height"] - area
    return area / union if union > 0 else 0.0


def text_similarity(a: Optional[str], b: Optional[str]) -> float:
    if not a or not b:
        return 0.5
    return SequenceMatcher(None, a.casefold().strip(), b.casefold().strip()).ratio()


def association_score(previous: TrackObservation, current: TrackObservation) -> float:
    """Score observations, with an absolute shot-boundary barrier."""
    if previous.shot_id != current.shot_id:
        return 0.0
    old_text = next((c.text for c in previous.ocr_candidates if c.accepted), None)
    new_text = next((c.text for c in current.ocr_candidates if c.accepted), None)
    return 0.65 * bbox_iou(previous.bbox, current.bbox) + 0.35 * text_similarity(old_text, new_text)


def should_run_ocr(*, is_keyframe: bool, track_birth: bool = False,
                   scene_cut: bool = False, confidence: float = 1.0,
                   visual_change: float = 0.0, recovered_from_occlusion: bool = False,
                   forced: bool = False) -> Optional[str]:
    for active, reason in (
        (forced, "forced"), (scene_cut, "scene_cut"), (track_birth, "track_birth"),
        (recovered_from_occlusion, "occlusion_recovery"),
        (confidence < 0.55, "confidence_decay"),
        (visual_change > 0.35, "visual_change"), (is_keyframe, "keyframe"),
    ):
        if active:
            return reason
    return None


@dataclass
class ConsensusAccumulator:
    """Streaming, bounded form of the temporal vote.

    The batch version kept every observation of a track alive until the end of
    the clip purely to reduce it to three numbers per distinct string -- a slow
    way to carry a dict, and unbounded in clip length. Folding as we go keeps
    the same arithmetic in O(K).

    Variants past the top K are evicted Misra-Gries / Space-Saving style. The
    evicted weight is KEPT in `evicted_mass` and stays in the confidence
    denominator, so dropping losers can only understate the winner's share,
    never inflate it. A summary that got more confident by forgetting evidence
    would be worse than no summary.
    """
    scores: Dict[str, float] = field(default_factory=dict)
    counts: Dict[str, int] = field(default_factory=dict)
    display: Dict[str, str] = field(default_factory=dict)
    observed: int = 0
    evicted_mass: float = 0.0
    evicted_variants: int = 0

    def fold(self, observation: TrackObservation) -> None:
        folded = False
        sharp = max(0.25, min(2.0, (observation.sharpness or 100.0) / 100.0))
        for candidate in observation.ocr_candidates:
            key = candidate.text.casefold().strip()
            if not key:
                continue
            self.scores[key] = self.scores.get(key, 0.0) + max(0.0, candidate.confidence) * sharp
            self.counts[key] = self.counts.get(key, 0) + 1
            self.display.setdefault(key, candidate.text)
            folded = True
        if folded:
            self.observed += 1
        while len(self.scores) > CONSENSUS_TOP_K:
            weakest = min(self.scores, key=self.scores.get)
            self.evicted_mass += self.scores.pop(weakest)
            self.counts.pop(weakest, None)
            self.display.pop(weakest, None)
            self.evicted_variants += 1

    def verdict(self) -> Dict[str, Any]:
        if not self.scores:
            return {"text": None, "confidence": 0.0, "evidence_count": 0, "disagreement": False}
        winner = max(self.scores, key=self.scores.get)
        total = sum(self.scores.values()) + self.evicted_mass
        return {"text": self.display[winner],
                "confidence": self.scores[winner] / total if total else 0.0,
                "evidence_count": self.counts[winner],
                "disagreement": len(self.scores) > 1 or self.evicted_variants > 0}

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Optional[Dict[str, Any]]) -> "ConsensusAccumulator":
        data = data or {}
        return cls(scores={str(k): float(v) for k, v in (data.get("scores") or {}).items()},
                   counts={str(k): int(v) for k, v in (data.get("counts") or {}).items()},
                   display={str(k): str(v) for k, v in (data.get("display") or {}).items()},
                   observed=int(data.get("observed", 0)),
                   evicted_mass=float(data.get("evicted_mass", 0.0)),
                   evicted_variants=int(data.get("evicted_variants", 0)))


def temporal_consensus(observations: Iterable[TrackObservation]) -> Dict[str, Any]:
    """Quality-weighted vote; repeated weak frames cannot swamp one clear crop.

    One implementation, folded in a loop: the streaming and batch paths cannot
    drift apart into two subtly different votes.
    """
    accumulator = ConsensusAccumulator()
    for observation in observations:
        accumulator.fold(observation)
    return accumulator.verdict()


def settle(track_rows: Iterable[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Close consensus over persisted tracks, off the heat: no decoder, no database.

    Finalization must not depend on what happens to be in the analyzer's
    memory. After a resume the earlier chunks' tracks were never in this
    process at all -- they are rows. Reading the accumulator off the row makes
    finalization idempotent and re-runnable over a partially resumed job.

    Returns (settled track dicts, review issues).
    """
    settled: List[Dict[str, Any]] = []
    issues: List[Dict[str, Any]] = []
    for row in track_rows:
        track = dict(row)
        verdict = ConsensusAccumulator.from_dict(track.get("consensus_state")).verdict()
        track["source_text"] = verdict["text"]
        track["consensus_confidence"] = verdict["confidence"]
        ## 'excluded' is a reviewer's decision, and the compositor honours it by
        ## skipping the region entirely. Recomputing over it would silently put
        ## burned-in text back into the export.
        if track.get("status") != "excluded":
            track["status"] = ("review_required"
                               if verdict["disagreement"] or verdict["confidence"] < .7
                               else "recognized")
        if track["status"] == "review_required":
            issues.append({"track_id": track.get("id"), "frame_index": track.get("start_frame"),
                           "code": "ocr_disagreement", "severity": "review",
                           "detail": "Temporal OCR evidence requires review"})
        settled.append(track)
    return settled, issues


def resolve_keyframes(frame_index: int, base: Dict[str, Any],
                      keyframes: Sequence[RenderKeyframe]) -> Dict[str, Any]:
    """Resolve track/range overrides then interpolate adjacent frame keys."""
    out = dict(base)
    ordered = sorted(keyframes, key=lambda k: (k.frame_index, k.created_at))
    for key in ordered:
        applies = key.scope == "track" or (
            key.scope == "range" and key.frame_index <= frame_index <= (key.end_frame or key.frame_index)
        )
        if applies:
            for name in ("bbox", "quad", "opacity", "style", "effects", "mask_path"):
                value = getattr(key, name)
                if value is not None:
                    out[name] = value
    frame_keys = [k for k in ordered if k.scope == "frame"]
    before = max((k for k in frame_keys if k.frame_index <= frame_index), key=lambda k: k.frame_index, default=None)
    after = min((k for k in frame_keys if k.frame_index >= frame_index), key=lambda k: k.frame_index, default=None)
    if before and after and before is not after and before.bbox and after.bbox:
        alpha = (frame_index - before.frame_index) / (after.frame_index - before.frame_index)
        out["bbox"] = {n: before.bbox[n] + (after.bbox[n] - before.bbox[n]) * alpha
                       for n in ("x", "y", "width", "height")}
    elif before:
        for name in ("bbox", "quad", "opacity", "style", "effects", "mask_path"):
            value = getattr(before, name)
            if value is not None:
                out[name] = value
    return out
