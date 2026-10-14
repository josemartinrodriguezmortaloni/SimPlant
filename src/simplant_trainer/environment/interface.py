from typing import Protocol

from numpy import ndarray

from simplant_trainer.value_objects.env_metadata import EnvMetadata
from simplant_trainer.value_objects.step_result import StepResult


class IEnvironment(Protocol):
    def reset(self) -> ndarray:
        """Reiniciar el entorno y retornar la observación inicial."""
        ...

    def step(self, action: ndarray) -> StepResult:
        """Ejecutar una acción. RETORNAR (obs, reward, terminated, truncated)."""
        ...

    def metadata(self) -> EnvMetadata:
        """RETORNAR metadata del entorno (ej. espacio de acción, espacio de observación)."""
        ...

    def close(self) -> None:
        """Cerrar recursos del entorno (ej. conexiones a simuladores)."""
        ...
