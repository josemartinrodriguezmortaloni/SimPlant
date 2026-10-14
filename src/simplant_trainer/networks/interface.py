from collections.abc import Iterator
from typing import Protocol

from torch import Tensor
from torch.nn import Parameter


class INetwork(Protocol):
    """Contrato para cualquier red neuronal configurable."""

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass de la red."""
        ...

    def parameters(self) -> Iterator[Parameter]:
        """Parámetros entrenables de la red."""
        ...

    def state_dict(self) -> dict:
        """Estado serializable de la red (pesos y biases)."""
        ...

    def load_state_dict(self, state: dict) -> None:
        """Restaurar estado de la red desde un diccionario."""
        ...
