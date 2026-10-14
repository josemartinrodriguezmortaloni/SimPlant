"""Step result value object."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True, slots=True)
class StepResult:
    """Result of a single simulation step (immutable).

    Contains the observation, reward, and termination flags returned
    by the environment after executing an action.
    """

    observation: NDArray[np.floating[Any]]
    reward: float
    terminated: bool
    truncated: bool

    def __post_init__(self) -> None:
        # --- observation: defensive copy + freeze ---
        obs = np.asarray(self.observation, dtype=np.float64).copy()
        obs.flags.writeable = False
        object.__setattr__(self, "observation", obs)

        # --- reward: must be finite ---
        if not isinstance(self.reward, (int, float)):
            raise TypeError(
                f"reward must be a finite float, got {type(self.reward).__name__}"
            )
        reward = float(self.reward)
        if not math.isfinite(reward):
            raise ValueError(f"reward must be finite, got {reward}")
        object.__setattr__(self, "reward", reward)

        # --- flags ---
        if not isinstance(self.terminated, bool):
            raise TypeError(
                f"terminated must be bool, got {type(self.terminated).__name__}"
            )
        if not isinstance(self.truncated, bool):
            raise TypeError(
                f"truncated must be bool, got {type(self.truncated).__name__}"
            )
        if self.terminated and self.truncated:
            raise ValueError("terminated and truncated cannot both be True")

    # -- convenience ---------------------------------------------------------

    @property
    def done(self) -> bool:
        """Episode is over (either terminated or truncated)."""
        return self.terminated or self.truncated

    # -- identity (ndarray-aware) --------------------------------------------

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, StepResult):
            return NotImplemented
        return (
            np.array_equal(self.observation, other.observation)
            and self.reward == other.reward
            and self.terminated == other.terminated
            and self.truncated == other.truncated
        )

    def __hash__(self) -> int:
        return hash(
            (
                self.observation.tobytes(),
                self.reward,
                self.terminated,
                self.truncated,
            )
        )
