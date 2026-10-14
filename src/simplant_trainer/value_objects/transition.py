"""Transition value object for experience buffers."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray


def _freeze(value: Any, name: str) -> NDArray[np.floating[Any]]:
    """Defensive-copy *value* to a non-writeable float64 array."""
    arr = np.asarray(value, dtype=np.float64).copy()
    arr.flags.writeable = False
    return arr


@dataclass(frozen=True, slots=True)
class Transition:
    """A single (s, a, r, s', done) transition for experience buffers (immutable).

    Represents one step of agent-environment interaction, stored in
    replay or rollout buffers for training.
    """

    state: NDArray[np.floating[Any]]
    action: NDArray[np.floating[Any]]
    reward: float
    next_state: NDArray[np.floating[Any]]
    terminated: bool

    def __post_init__(self) -> None:
        # --- arrays: defensive copy + freeze ---
        state = _freeze(self.state, "state")
        object.__setattr__(self, "state", state)

        action = _freeze(self.action, "action")
        object.__setattr__(self, "action", action)

        next_state = _freeze(self.next_state, "next_state")
        object.__setattr__(self, "next_state", next_state)

        if state.shape != next_state.shape:
            raise ValueError(
                f"state shape {state.shape} must match "
                f"next_state shape {next_state.shape}"
            )

        # --- reward: must be finite ---
        if not isinstance(self.reward, (int, float)):
            raise TypeError(
                f"reward must be a finite float, got {type(self.reward).__name__}"
            )
        reward = float(self.reward)
        if not math.isfinite(reward):
            raise ValueError(f"reward must be finite, got {reward}")
        object.__setattr__(self, "reward", reward)

        # --- terminated flag ---
        if not isinstance(self.terminated, bool):
            raise TypeError(
                f"terminated must be bool, got {type(self.terminated).__name__}"
            )

    # -- identity (ndarray-aware) --------------------------------------------

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Transition):
            return NotImplemented
        return (
            np.array_equal(self.state, other.state)
            and np.array_equal(self.action, other.action)
            and self.reward == other.reward
            and np.array_equal(self.next_state, other.next_state)
            and self.terminated == other.terminated
        )

    def __hash__(self) -> int:
        return hash(
            (
                self.state.tobytes(),
                self.action.tobytes(),
                self.reward,
                self.next_state.tobytes(),
                self.terminated,
            )
        )
