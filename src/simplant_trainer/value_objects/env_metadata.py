"""Environment metadata value object."""

from __future__ import annotations

import re
from dataclasses import dataclass

from .spaces import ActionSpace, ObservationSpace

_ENV_ID_PATTERN = re.compile(r"^[a-zA-Z0-9][a-zA-Z0-9_-]*$")


@dataclass(frozen=True, slots=True)
class EnvMetadata:
    """Identity and contract of a simulation environment (immutable).

    Captures the environment's unique identifier together with its
    observation and action space definitions.  Used throughout the library
    to validate compatibility between agents, environments, and exported
    models.
    """

    env_id: str
    observation_space: ObservationSpace
    action_space: ActionSpace

    def __post_init__(self) -> None:
        # Runtime guards — Python no enforce tipos en dataclasses,
        # así que validamos explícitamente (fail-fast).
        if not isinstance(self.env_id, str) or not self.env_id:  # pyright: ignore[reportUnnecessaryIsInstance]
            raise ValueError("env_id must be a non-empty string")
        if not _ENV_ID_PATTERN.match(self.env_id):
            raise ValueError(
                f"env_id must start with an alphanumeric character and contain "
                f"only alphanumerics, hyphens or underscores. Got: '{self.env_id}'"
            )
        if not isinstance(self.observation_space, ObservationSpace):  # pyright: ignore[reportUnnecessaryIsInstance]
            raise TypeError(
                f"observation_space must be an ObservationSpace, "
                f"got {type(self.observation_space).__name__}"
            )
        if not isinstance(self.action_space, ActionSpace):  # pyright: ignore[reportUnnecessaryIsInstance]
            raise TypeError(
                f"action_space must be an ActionSpace, "
                f"got {type(self.action_space).__name__}"
            )
