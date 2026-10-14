"""Hyperparameters value object."""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Any


@dataclass(frozen=True, slots=True)
class HyperParameters:
    """Agent hyperparameters configuration (immutable).

    Core hyperparameters shared by all RL agents.  Algorithm-specific
    knobs go in ``extras`` (stored as a read-only mapping).

    Parameters
    ----------
    learning_rate:
        Optimiser step size.  Must be in ``(0, 1]``.
    batch_s
ize:
        Mini-batch size for gradient updates.  Must be a positive int.
    gamma:
        Discount factor.  Must be in ``(0, 1]``.
    buffer_capacity:
        Maximum transitions held in the experience buffer.
        Must be strictly greater than *batch_size*.
    extras:
        Optional ``dict`` of algorithm-specific parameters
        (e.g. ``{"policy_delay": 2}``).  Converted to a read-only
        ``MappingProxyType`` at construction time.
    """

    learning_rate: float
    batch_size: int
    gamma: float
    buffer_capacity: int
    extras: dict[str, Any] | None = None  # stored as MappingProxyType at runtime

    def __post_init__(self) -> None:
        # --- learning_rate ---
        if not isinstance(self.learning_rate, (int, float)):
            raise TypeError(
                f"learning_rate must be a float, got {type(self.learning_rate).__name__}"
            )
        lr = float(self.learning_rate)
        if not (0.0 < lr <= 1.0):
            raise ValueError(f"learning_rate must be in (0, 1.0], got {lr}")
        object.__setattr__(self, "learning_rate", lr)

        # --- batch_size ---
        if not isinstance(self.batch_size, int) or isinstance(self.batch_size, bool):
            raise TypeError(
                f"batch_size must be an int, got {type(self.batch_size).__name__}"
            )
        if self.batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {self.batch_size}")

        # --- gamma ---
        if not isinstance(self.gamma, (int, float)):
            raise TypeError(f"gamma must be a float, got {type(self.gamma).__name__}")
        gamma = float(self.gamma)
        if not (0.0 < gamma <= 1.0):
            raise ValueError(f"gamma must be in (0, 1.0], got {gamma}")
        object.__setattr__(self, "gamma", gamma)

        # --- buffer_capacity ---
        if not isinstance(self.buffer_capacity, int) or isinstance(
            self.buffer_capacity, bool
        ):
            raise TypeError(
                f"buffer_capacity must be an int, got {type(self.buffer_capacity).__name__}"
            )
        if self.buffer_capacity <= self.batch_size:
            raise ValueError(
                f"buffer_capacity ({self.buffer_capacity}) must be greater "
                f"than batch_size ({self.batch_size})"
            )

        # --- extras → MappingProxyType (read-only) ---
        if self.extras is None:
            object.__setattr__(self, "extras", MappingProxyType({}))
        elif isinstance(self.extras, dict):
            object.__setattr__(self, "extras", MappingProxyType(dict(self.extras)))
        elif isinstance(self.extras, MappingProxyType):
            pass  # already immutable
        else:
            raise TypeError(
                f"extras must be a dict or None, got {type(self.extras).__name__}"
            )
