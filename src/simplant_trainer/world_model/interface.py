from typing import Protocol

from torch import Tensor


class ITransitionModel(Protocol):
    """Predice el próximo estado dado estado + acción."""

    def predict(self, state: Tensor, action: Tensor) -> Tensor:
        """RETORNAR next_state predicho."""
        ...

    def train_step(self, states: Tensor, actions: Tensor, next_states: Tensor) -> float:
        """Entrenar con un batch. RETORNAR loss."""
        ...


class IRewardModel(Protocol):
    """Predice la recompensa inmediata dado estado + acción."""

    def predict(self, state: Tensor, action: Tensor) -> Tensor:
        """RETORNAR reward predicho."""
        ...

    def train_step(self, states: Tensor, actions: Tensor, rewards: Tensor) -> float:
        """Entrenar con un batch. RETORNAR loss."""
        ...
