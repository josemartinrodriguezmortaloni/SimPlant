"""Networks — configurable neural network building blocks.

This module provides concrete ``nn.Module`` implementations that satisfy
the :class:`~simplant_trainer.networks.interface.INetwork` protocol.
"""

from .interface import INetwork
from .mlp import MLP

__all__ = [
    "INetwork",
    "MLP",
]
