"""Typed progress events emitted by the canonical ToFU pipeline.

The core owns this contract.  Transport adapters (SSE, CLI, desktop) may
serialize or display events, but must not duplicate pipeline orchestration.
"""

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Callable, Dict, Optional


class PipelineEventStatus(str, Enum):
    STARTED = "started"
    COMPLETED = "completed"
    WARNING = "warning"
    FAILED = "failed"
    PAUSED = "paused"


@dataclass(frozen=True)
class PipelineEvent:
    """One transport-neutral pipeline progress notification."""

    stage: str
    operation: str
    status: PipelineEventStatus
    progress: Optional[float] = None
    payload: Dict[str, Any] = field(default_factory=dict)
    duration_ms: Optional[int] = None
    timestamp: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    def to_dict(self) -> Dict[str, Any]:
        """Return an SSE/JSON-ready representation."""
        result = asdict(self)
        result["status"] = self.status.value
        return result


PipelineObserver = Callable[[PipelineEvent], None]
