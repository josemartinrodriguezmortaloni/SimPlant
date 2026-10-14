"""World Model capability — learned environment dynamics for Model-Based RL."""

from simplant_trainer.world_model.interface import IRewardModel, ITransitionModel
from simplant_trainer.world_model.world_model import WorldModel

__all__ = ["ITransitionModel", "IRewardModel", "WorldModel"]
