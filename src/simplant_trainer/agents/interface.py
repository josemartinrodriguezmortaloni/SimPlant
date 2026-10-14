from typing import Protocol

from numpy import ndarray

from simplant_trainer.value_objects.metrics import EpochMetrics
from simplant_trainer.value_objects.transition import Transition


class IAgent(Protocol):
    def select_action(
        self, observation: ndarray, deterministic: bool = False
    ) -> ndarray:
        """Elegir acción dado un estado.
        Si deterministic=True, usar la media de la política (sin exploración).
        Si deterministic=False, samplear de la distribución."""
        ...

    def observe(self, transition: Transition) -> None:
        """Observar una transición (estado, acción, recompensa, siguiente estado)."""
        ...

    def update(self) -> EpochMetrics | None:
        """Actualizar pesos de las redes neuronales.
        RETORNAR métricas si hubo update, None si no hay suficientes datos."""
        ...

    def save_state(self) -> bytes:
        """Serializar estado completo del agente (pesos + optimizer + buffer)."""
        ...

    def load_state(self, state: bytes) -> None:
        """Restaurar estado del agente desde bytes serializados."""
        ...

    def close(self) -> None:
        """Cerrar recursos (ej. sesiones de TensorFlow)."""
        ...
