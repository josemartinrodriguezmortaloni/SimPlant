"""Adapter for external SimPlant environments."""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

import numpy as np
from numpy import ndarray

from simplant_trainer.value_objects.env_metadata import EnvMetadata
from simplant_trainer.value_objects.spaces import ActionSpace, ObservationSpace
from simplant_trainer.value_objects.step_result import StepResult

# ---------------------------------------------------------------------------
# Protocol: contrato mínimo que debe cumplir el env externo
# ---------------------------------------------------------------------------


class _SpaceLike(Protocol):
    """Lo mínimo que necesitamos de un espacio externo."""

    @property
    def shape(self) -> tuple[int, ...]: ...

    @property
    def low(self) -> ndarray: ...

    @property
    def high(self) -> ndarray: ...


@runtime_checkable
class SimPlantEnv(Protocol):
    """Contrato mínimo que debe cumplir cualquier entorno externo
    para poder ser adaptado con ``SimPlantEnvAdapter``.

    No necesitás heredar de esto — es structural typing (duck typing).
    Si tu entorno tiene estos atributos y métodos, funciona.
    """

    @property
    def name(self) -> str: ...

    @property
    def observation_space(self) -> _SpaceLike: ...

    @property
    def action_space(self) -> _SpaceLike: ...

    def reset(self) -> ndarray | tuple[ndarray, Any]: ...

    def step(self, action: ndarray) -> tuple[Any, ...]: ...

    def close(self) -> None: ...


# ---------------------------------------------------------------------------
# Adapter
# ---------------------------------------------------------------------------


class SimPlantEnvAdapter:
    """Adapta un entorno SimPlant externo a ``IEnvironment`` (Patrón Adapter).

    El adapter traduce la interfaz del entorno externo al contrato
    interno de la librería, asegurando que las observaciones sean
    inmutables y las acciones estén validadas.
    """

    def __init__(self, simplant_env: SimPlantEnv) -> None:
        self._env: SimPlantEnv | None = simplant_env

        obs_space = ObservationSpace(
            shape=simplant_env.observation_space.shape,
            low=simplant_env.observation_space.low,
            high=simplant_env.observation_space.high,
        )
        act_space = ActionSpace(
            shape=simplant_env.action_space.shape,
            low=simplant_env.action_space.low,
            high=simplant_env.action_space.high,
        )

        self._metadata = EnvMetadata(
            env_id=simplant_env.name,
            observation_space=obs_space,
            action_space=act_space,
        )

    # --- IEnvironment -----------------------------------------------------

    def reset(self) -> ndarray:
        """Reiniciar el entorno y retornar la observación inicial."""
        env = self._require_env()
        result = env.reset()

        # Adaptar: el env externo puede retornar ndarray o (ndarray, info)
        observation = result[0] if isinstance(result, tuple) else result

        obs = np.asarray(observation, dtype=np.float64).copy()
        obs.flags.writeable = False
        return obs

    def step(self, action: ndarray) -> StepResult:
        """Ejecutar una acción, validándola contra el espacio de acciones."""
        env = self._require_env()

        if not self._metadata.action_space.contains(action):
            raise ValueError(
                f"Action is out of bounds for action space "
                f"{self._metadata.action_space}"
            )

        result = env.step(action)

        if len(result) < 3:
            raise ValueError(
                f"step() must return a tuple of at least 3 elements "
                f"(obs, reward, done). Got {len(result)} elements"
            )

        obs = result[0]
        reward = float(result[1])

        # Adaptar: puede venir (obs, r, done) o (obs, r, terminated, truncated)
        if len(result) >= 4 and isinstance(result[3], bool):
            terminated = bool(result[2])
            truncated = bool(result[3])
        else:
            terminated = bool(result[2])
            truncated = False

        return StepResult(
            observation=obs,
            reward=reward,
            terminated=terminated,
            truncated=truncated,
        )

    def metadata(self) -> EnvMetadata:
        """Retornar metadata del entorno (inmutable, construida en __init__)."""
        return self._metadata

    def close(self) -> None:
        """Liberar recursos del entorno externo."""
        if self._env is not None:
            self._env.close()
            self._env = None

    # --- internal ---------------------------------------------------------

    def _require_env(self) -> SimPlantEnv:
        """Garantizar que el entorno no fue cerrado."""
        if self._env is None:
            raise RuntimeError("Environment is closed")
        return self._env
