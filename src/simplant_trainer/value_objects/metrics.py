"""Metrics value objects for training and evaluation."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime, timezone


@dataclass(frozen=True, slots=True)
class EpochMetrics:
    """Metrics from a single training epoch (immutable).

    Automatically timestamped at construction if *timestamp* is omitted.
    """

    epoch: int
    mean_reward: float
    policy_loss: float
    value_loss: float
    entropy: float
    timestamp: datetime = field(
        default_factory=lambda: datetime.now(timezone.utc),
    )

    def __post_init__(self) -> None:
        # --- epoch ---
        if not isinstance(self.epoch, int) or isinstance(self.epoch, bool):
            raise TypeError(f"epoch must be an int, got {type(self.epoch).__name__}")
        if self.epoch < 0:
            raise ValueError(f"epoch must be non-negative, got {self.epoch}")

        # --- finite floats ---
        for name in ("mean_reward", "policy_loss", "value_loss"):
            val = getattr(self, name)
            if not isinstance(val, (int, float)):
                raise TypeError(f"{name} must be a float, got {type(val).__name__}")
            fval = float(val)
            if not math.isfinite(fval):
                raise ValueError(f"{name} must be finite, got {fval}")
            object.__setattr__(self, name, fval)

        # --- entropy ---
        if not isinstance(self.entropy, (int, float)):
            raise TypeError(
                f"entropy must be a float, got {type(self.entropy).__name__}"
            )
        entropy = float(self.entropy)
        if not math.isfinite(entropy) or entropy < 0.0:
            raise ValueError(
                f"entropy must be a non-negative finite float, got {entropy}"
            )
        object.__setattr__(self, "entropy", entropy)

        # --- timestamp ---
        if not isinstance(self.timestamp, datetime):
            raise TypeError(
                f"timestamp must be a datetime, got {type(self.timestamp).__name__}"
            )


@dataclass(frozen=True, slots=True)
class EvaluationReport:
    """Result of evaluating a trained agent without learning (immutable)."""

    total_reward: float
    mean_reward: float
    std_reward: float
    episodes: int
    constraint_violations: int

    def __post_init__(self) -> None:
        # --- finite floats ---
        for name in ("total_reward", "mean_reward"):
            val = getattr(self, name)
            if not isinstance(val, (int, float)):
                raise TypeError(f"{name} must be a float, got {type(val).__name__}")
            object.__setattr__(self, name, float(val))

        # --- std_reward >= 0 ---
        if not isinstance(self.std_reward, (int, float)):
            raise TypeError(
                f"std_reward must be a float, got {type(self.std_reward).__name__}"
            )
        std = float(self.std_reward)
        if std < 0.0:
            raise ValueError(f"std_reward must be non-negative, got {std}")
        object.__setattr__(self, "std_reward", std)

        # --- episodes > 0 ---
        if not isinstance(self.episodes, int) or isinstance(self.episodes, bool):
            raise TypeError(
                f"episodes must be an int, got {type(self.episodes).__name__}"
            )
        if self.episodes <= 0:
            raise ValueError(f"episodes must be positive, got {self.episodes}")

        # --- constraint_violations >= 0 ---
        if not isinstance(self.constraint_violations, int) or isinstance(
            self.constraint_violations, bool
        ):
            raise TypeError(
                f"constraint_violations must be an int, "
                f"got {type(self.constraint_violations).__name__}"
            )
        if self.constraint_violations < 0:
            raise ValueError(
                f"constraint_violations must be non-negative, "
                f"got {self.constraint_violations}"
            )
