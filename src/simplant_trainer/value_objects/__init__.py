"""Value Objects — immutable, self-validating domain primitives.

All value objects in this package are frozen dataclasses that validate
their invariants at construction time (fail-fast).  Once created, they
cannot be modified.
"""

from .checkpoint_data import CheckpointData
from .env_metadata import EnvMetadata
from .hyperparameters import HyperParameters
from .metrics import EpochMetrics, EvaluationReport
from .model_version import ModelVersion
from .spaces import ActionSpace, ObservationSpace
from .step_result import StepResult
from .stop_criteria import StopCriteria
from .training_result import TrainingResult
from .transition import Transition

__all__ = [
    "ActionSpace",
    "CheckpointData",
    "EnvMetadata",
    "EpochMetrics",
    "EvaluationReport",
    "HyperParameters",
    "ModelVersion",
    "ObservationSpace",
    "StepResult",
    "StopCriteria",
    "TrainingResult",
    "Transition",
]
