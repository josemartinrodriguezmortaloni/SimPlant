"""Stop criteria value object for training control."""

from __future__ import annotations

from dataclasses import dataclass

_STAGNATION_THRESHOLD: float = 1e-6


@dataclass(frozen=True, slots=True)
class StopCriteria:
    """Training stop criteria (immutable).

    At least one of *max_epochs* or *target_reward* must be set.
    Three stopping conditions are evaluated in order:

    1. **Max epochs** — ``epoch >= max_epochs``.
    2. **Target reward** — mean reward over the last *patience* epochs
       meets or exceeds *target_reward*.
    3. **Early stopping** — reward stagnation detected (variation over
       the last *patience* epochs < 1e-6).
    """

    max_epochs: int | None = None
    target_reward: float | None = None
    patience: int = 10

    def __post_init__(self) -> None:
        if self.max_epochs is None and self.target_reward is None:
            raise ValueError("at least one of max_epochs or target_reward must be set")

        if self.max_epochs is not None:
            if not isinstance(self.max_epochs, int) or isinstance(
                self.max_epochs, bool
            ):
                raise TypeError(
                    f"max_epochs must be an int, got {type(self.max_epochs).__name__}"
                )
            if self.max_epochs <= 0:
                raise ValueError(f"max_epochs must be positive, got {self.max_epochs}")

        if self.target_reward is not None:
            if not isinstance(self.target_reward, (int, float)):
                raise TypeError(
                    f"target_reward must be a float, "
                    f"got {type(self.target_reward).__name__}"
                )
            object.__setattr__(self, "target_reward", float(self.target_reward))

        if not isinstance(self.patience, int) or isinstance(self.patience, bool):
            raise TypeError(
                f"patience must be an int, got {type(self.patience).__name__}"
            )
        if self.patience <= 0:
            raise ValueError(f"patience must be positive, got {self.patience}")

    # -- evaluation ----------------------------------------------------------

    def is_met(self, epoch: int, reward_history: list[float]) -> bool:
        """Return ``True`` if any stopping condition is satisfied."""
        met, _ = self.check(epoch, reward_history)
        return met

    def check(self, epoch: int, reward_history: list[float]) -> tuple[bool, str]:
        """Evaluate all conditions and return ``(should_stop, reason)``.

        *reason* is an empty string when *should_stop* is ``False``.
        """
        # 1. Max epochs
        if self.max_epochs is not None and epoch >= self.max_epochs:
            return True, f"reached max_epochs ({self.max_epochs})"

        # 2. Target reward (averaged over patience window)
        if self.target_reward is not None and len(reward_history) >= self.patience:
            window = reward_history[-self.patience :]
            mean_recent = sum(window) / len(window)
            if mean_recent >= self.target_reward:
                return True, (
                    f"target_reward ({self.target_reward}) achieved "
                    f"(mean over last {self.patience}: {mean_recent:.4f})"
                )

        # 3. Early stopping — reward stagnation
        if len(reward_history) >= self.patience:
            window = reward_history[-self.patience :]
            improvement = max(window) - min(window)
            if improvement < _STAGNATION_THRESHOLD:
                return True, (
                    f"early stopping: reward stagnated "
                    f"(variation {improvement:.2e} < {_STAGNATION_THRESHOLD:.0e} "
                    f"over last {self.patience} epochs)"
                )

        return False, ""
