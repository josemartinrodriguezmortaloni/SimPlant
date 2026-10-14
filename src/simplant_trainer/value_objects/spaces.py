"""Continuous space definitions for observations and actions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _validate_shape(raw: Any) -> tuple[int, ...]:
    """Normalize and validate a shape to a tuple of positive integers."""
    if isinstance(raw, int):
        raw = (raw,)
    if isinstance(raw, (list, np.ndarray)):
        raw = tuple(int(s) for s in raw)
    if not isinstance(raw, tuple):
        raise TypeError(f"shape must be a tuple of ints, got {type(raw).__name__}")
    if len(raw) == 0:
        raise ValueError("shape must have at least one dimension")
    if not all(isinstance(s, int) and s > 0 for s in raw):
        raise ValueError(f"all shape dimensions must be positive integers, got {raw}")
    return raw


def _to_immutable_array(
    value: Any,
    shape: tuple[int, ...],
    name: str,
) -> NDArray[np.floating[Any]]:
    """Convert *value* to a float64 ndarray, broadcast to *shape*, freeze."""
    arr = np.asarray(value, dtype=np.float64)
    try:
        arr = np.broadcast_to(arr, shape).copy()
    except ValueError as exc:
        raise ValueError(
            f"{name} with shape {arr.shape} cannot broadcast to {shape}"
        ) from exc
    arr.flags.writeable = False
    return arr


# ---------------------------------------------------------------------------
# ObservationSpace
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ObservationSpace:
    """Continuous observation space (immutable).

    Defines the valid range for each observation dimension.
    Bounds are inclusive: ``low[i] <= obs[i] <= high[i]``.
    Scalar bounds are broadcast to every dimension automatically.
    """

    shape: tuple[int, ...]
    low: NDArray[np.floating[Any]]
    high: NDArray[np.floating[Any]]

    def __post_init__(self) -> None:
        shape = _validate_shape(self.shape)
        object.__setattr__(self, "shape", shape)

        low = _to_immutable_array(self.low, shape, "low")
        object.__setattr__(self, "low", low)

        high = _to_immutable_array(self.high, shape, "high")
        object.__setattr__(self, "high", high)

        if not np.all(low < high):
            bad = np.argwhere(low >= high).flatten()
            raise ValueError(
                f"low must be strictly less than high at every index. "
                f"Violations at indices: {bad.tolist()}"
            )

    # -- queries -------------------------------------------------------------

    def contains(self, value: NDArray[Any]) -> bool:
        """Return ``True`` if *value* falls within bounds."""
        arr = np.asarray(value)
        if arr.shape != self.shape:
            return False
        return bool(np.all(arr >= self.low) and np.all(arr <= self.high))

    # -- identity ------------------------------------------------------------

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, ObservationSpace):
            return NotImplemented
        return (
            self.shape == other.shape
            and np.array_equal(self.low, other.low)
            and np.array_equal(self.high, other.high)
        )

    def __hash__(self) -> int:
        return hash((self.shape, self.low.tobytes(), self.high.tobytes()))

    def __repr__(self) -> str:
        return (
            f"ObservationSpace(shape={self.shape}, "
            f"low=[{self.low.min():.4g}, {self.low.max():.4g}], "
            f"high=[{self.high.min():.4g}, {self.high.max():.4g}])"
        )


# ---------------------------------------------------------------------------
# ActionSpace
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ActionSpace:
    """Continuous action space (immutable).

    Defines the valid range for each action dimension.
    Bounds are inclusive: ``low[i] <= action[i] <= high[i]``.
    Scalar bounds are broadcast to every dimension automatically.
    """

    shape: tuple[int, ...]
    low: NDArray[np.floating[Any]]
    high: NDArray[np.floating[Any]]

    def __post_init__(self) -> None:
        shape = _validate_shape(self.shape)
        object.__setattr__(self, "shape", shape)

        low = _to_immutable_array(self.low, shape, "low")
        object.__setattr__(self, "low", low)

        high = _to_immutable_array(self.high, shape, "high")
        object.__setattr__(self, "high", high)

        if not np.all(low < high):
            bad = np.argwhere(low >= high).flatten()
            raise ValueError(
                f"low must be strictly less than high at every index. "
                f"Violations at indices: {bad.tolist()}"
            )

    # -- queries -------------------------------------------------------------

    def contains(self, value: NDArray[Any]) -> bool:
        """Return ``True`` if *value* falls within bounds."""
        arr = np.asarray(value)
        if arr.shape != self.shape:
            return False
        return bool(np.all(arr >= self.low) and np.all(arr <= self.high))

    def clip(self, value: NDArray[Any]) -> NDArray[np.floating[Any]]:
        """Clip *value* to space bounds.  Always returns a **new** array."""
        return np.clip(np.asarray(value, dtype=np.float64), self.low, self.high)

    # -- identity ------------------------------------------------------------

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, ActionSpace):
            return NotImplemented
        return (
            self.shape == other.shape
            and np.array_equal(self.low, other.low)
            and np.array_equal(self.high, other.high)
        )

    def __hash__(self) -> int:
        return hash((self.shape, self.low.tobytes(), self.high.tobytes()))

    def __repr__(self) -> str:
        return (
            f"ActionSpace(shape={self.shape}, "
            f"low=[{self.low.min():.4g}, {self.low.max():.4g}], "
            f"high=[{self.high.min():.4g}, {self.high.max():.4g}])"
        )
