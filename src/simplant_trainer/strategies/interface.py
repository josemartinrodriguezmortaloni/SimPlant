from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

from simplant_trainer.agents.interface import IAgent
from simplant_trainer.environment.interface import IEnvironment

if TYPE_CHECKING:
    from simplant_trainer.world_model.world_model import WorldModel


class IStrategy(Protocol):
    """Contrato para estrategias de entrenamiento.

    La estrategia define CÓMO se ejecuta una época de entrenamiento.
    Permite desacoplar el loop de entrenamiento del algoritmo de RL.
    """

    def run_epoch(
        self,
        agent: IAgent,
        env: IEnvironment,
        world_model: WorldModel | None,
    ) -> list[float]:
        """Ejecutar una época completa.
        RETORNAR lista de rewards por episodio en esta época."""
        ...
