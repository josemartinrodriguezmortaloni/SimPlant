"""Checkpoint data value object."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone

from .hyperparameters import HyperParameters


@dataclass(frozen=True, slots=True)
class CheckpointData:
    """Training checkpoint data (immutable).

    Contains everything needed to resume training from a saved point.
    ``agent_state`` is an opaque ``bytes`` blob produced by
    ``IAgent.save_state()``.
    """

    checkpoint_id: str
    epoch: int
    agent_state: bytes
    hyperparameters: HyperParameters
    created_at: datetime = field(
        default_factory=lambda: datetime.now(timezone.utc),
    )

    def __post_init__(self) -> None:
        # Runtime guards — Python no enforce tipos en dataclasses,
        # así que validamos explícitamente (fail-fast).
        if not isinstance(self.checkpoint_id, str) or not self.checkpoint_id.strip():  # pyright: ignore[reportUnnecessaryIsInstance]
            raise ValueError("checkpoint_id must be a non-empty string")

        if not isinstance(self.epoch, int) or isinstance(self.epoch, bool):  # pyright: ignore[reportUnnecessaryIsInstance]
            raise TypeError(f"epoch must be an int, got {type(self.epoch).__name__}")
        if self.epoch < 0:
            raise ValueError(f"epoch must be non-negative, got {self.epoch}")

        if not isinstance(self.agent_state, bytes):  # pyright: ignore[reportUnnecessaryIsInstance]
            raise TypeError(
                f"agent_state must be bytes, got {type(self.agent_state).__name__}"
            )

        if not isinstance(self.hyperparameters, HyperParameters):  # pyright: ignore[reportUnnecessaryIsInstance]
            raise TypeError(
                f"hyperparameters must be HyperParameters, "
                f"got {type(self.hyperparameters).__name__}"
            )

        if not isinstance(self.created_at, datetime):  # pyright: ignore[reportUnnecessaryIsInstance]
            raise TypeError(
                f"created_at must be a datetime, got {type(self.created_at).__name__}"
            )
