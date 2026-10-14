"""Multi-Layer Perceptron — configurable building block for all networks.

Used as the backbone for policy, value function, transition model, and
reward model.  The ``layers`` parameter defines the **full** topology
(input → hidden → output).  Agent factories are responsible for
prepending ``obs_dim`` and appending ``action_dim`` before constructing
this class.

Examples
--------
>>> mlp = MLP(layers=[4, 256, 256, 2])          # input=4, 2 hidden, output=2
>>> mlp = MLP(layers=[8, 64, 32, 1], activation="tanh")
>>> out = mlp.forward(torch.randn(1, 4))
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Final

import torch
from torch import Tensor, nn
from torch.nn import Parameter

_ACTIVATION_REGISTRY: Final[dict[str, type[nn.Module]]] = {
    "relu": nn.ReLU,
    "tanh": nn.Tanh,
    "sigmoid": nn.Sigmoid,
    "elu": nn.ELU,
}

_VALID_ACTIVATIONS: Final[frozenset[str]] = frozenset(_ACTIVATION_REGISTRY)


def _resolve_activation(name: str) -> nn.Module:
    """Return an activation module instance, or raise ``ValueError``."""
    try:
        return _ACTIVATION_REGISTRY[name]()
    except KeyError:
        msg = (
            f"Unknown activation {name!r}. Valid options: {sorted(_VALID_ACTIVATIONS)}"
        )
        raise ValueError(msg) from None


class MLP(nn.Module):
    """Multi-Layer Perceptron configurable.

    Fundamental building block for policy networks, value functions,
    transition models, and reward models.

    Parameters
    ----------
    layers:
        Full list of layer sizes, **including** input and output dims.
        Must have at least 2 elements (input dim, output dim).
    activation:
        Activation applied after every hidden layer.
        One of ``"relu"``, ``"tanh"``, ``"sigmoid"``, ``"elu"``.
    output_activation:
        Activation applied after the **last** layer.
        ``None`` means linear (no activation).

    Raises
    ------
    ValueError
        If ``layers`` has fewer than 2 elements or an unknown activation
        is requested.
    TypeError
        If ``layers`` elements are not positive integers.
    """

    def __init__(
        self,
        layers: list[int],
        activation: str = "relu",
        output_activation: str | None = None,
    ) -> None:
        super().__init__()

        # --- validation ---------------------------------------------------
        if not isinstance(layers, list) or len(layers) < 2:
            msg = (
                "'layers' must be a list with at least 2 elements "
                f"(input_dim, output_dim). Got: {layers!r}"
            )
            raise ValueError(msg)

        for i, size in enumerate(layers):
            if isinstance(size, bool) or not isinstance(size, int) or size <= 0:
                msg = (
                    f"Every element of 'layers' must be a positive int. "
                    f"Got layers[{i}]={size!r}"
                )
                raise TypeError(msg)

        if activation not in _VALID_ACTIVATIONS:
            msg = (
                f"Unknown activation {activation!r}. "
                f"Valid options: {sorted(_VALID_ACTIVATIONS)}"
            )
            raise ValueError(msg)

        if (
            output_activation is not None
            and output_activation not in _VALID_ACTIVATIONS
        ):
            msg = (
                f"Unknown output_activation {output_activation!r}. "
                f"Valid options: {sorted(_VALID_ACTIVATIONS)}"
            )
            raise ValueError(msg)

        # --- build sequential layers --------------------------------------
        modules: list[nn.Module] = []
        num_layers = len(layers) - 1

        for i in range(num_layers):
            modules.append(nn.Linear(layers[i], layers[i + 1]))

            is_last = i == num_layers - 1

            if not is_last:
                modules.append(_resolve_activation(activation))
            elif output_activation is not None:
                modules.append(_resolve_activation(output_activation))

        self._net = nn.Sequential(*modules)

    # --- INetwork protocol ------------------------------------------------

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass through the network."""
        return self._net(x)

    # parameters() and state_dict() / load_state_dict() are inherited from
    # nn.Module — no need to reimplement.  They already satisfy INetwork.

    def __repr__(self) -> str:
        return f"MLP({self._net})"
