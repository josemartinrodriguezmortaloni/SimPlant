"""Training result value object."""

from __future__ import annotations

from dataclasses import dataclass

from .metrics import EpochMetrics


@dataclass(frozen=True, slots=True)
class TrainingResult:
    """Result of a complete training process (immutable).

    Produced by ``Session.train()`` or ``Session.resume()``.
    """

    final_epoch: int
    final_metrics: EpochMetrics
    stop_reason: str
    elapsed_seconds: float

    def __post_init__(self) -> None:
        if not isinstance(self.final_epoch, int) or isinstance(self.final_epoch, bool):
            raise TypeError(
                f"final_epoch must be an int, got {type(self.final_epoch).__name__}"
            )
        if self.final_epoch < 0:
            raise ValueError(
                f"final_epoch must be non-negative, got {self.final_epoch}"
            )

        if not isinstance(self.final_metrics, EpochMetrics):
            raise TypeError(
                f"final_metrics must be EpochMetrics, "
                f"got {type(self.final_metrics).__name__}"
            )

        if not isinstance(self.stop_reason, str) or not self.stop_reason.strip():
            raise ValueError("stop_reason must be a non-empty string")

        if not isinstance(self.elapsed_seconds, (int, float)):
            raise TypeError(
                f"elapsed_seconds must be a float, "
                f"got {type(self.elapsed_seconds).__name__}"
            )
        elapsed = float(self.elapsed_seconds)
        if elapsed < 0.0:
            raise ValueError(f"elapsed_seconds must be non-negative, got {elapsed}")
        object.__setattr__(self, "elapsed_seconds", elapsed)
