"""Agent evaluation without learning."""

from __future__ import annotations

import numpy as np

from simplant_trainer.agents.interface import IAgent
from simplant_trainer.environment.interface import IEnvironment
from simplant_trainer.value_objects.metrics import EvaluationReport


class Evaluator:
    """Evaluate a trained agent in the environment **without learning**.

    The agent acts deterministically (mean of the policy, no
    exploration noise).  Observations that fall outside the declared
    observation space are counted as constraint violations.

    Parameters
    ----------
    env:
        The environment to evaluate in.
    agent:
        The trained agent to evaluate.
    """

    def __init__(self, env: IEnvironment, agent: IAgent) -> None:
        self._env = env
        self._agent = agent

    def evaluate(self, episodes: int = 100) -> EvaluationReport:
        """Run *episodes* rollouts and return aggregate metrics.

        Raises
        ------
        ValueError
            If *episodes* is not a positive integer.
        """
        if not isinstance(episodes, int) or episodes <= 0:
            raise ValueError(f"episodes must be a positive int, got {episodes}")

        total_rewards: list[float] = []
        total_violations = 0

        obs_space = self._env.metadata().observation_space

        for _ in range(episodes):
            state = self._env.reset()
            episode_reward = 0.0

            while True:
                action = self._agent.select_action(state, deterministic=True)
                result = self._env.step(action)

                episode_reward += result.reward
                state = result.observation

                if not obs_space.contains(state):
                    total_violations += 1

                if result.done:
                    break

            total_rewards.append(episode_reward)

        rewards = np.array(total_rewards, dtype=np.float64)

        return EvaluationReport(
            total_reward=float(rewards.sum()),
            mean_reward=float(rewards.mean()),
            std_reward=float(rewards.std()),
            episodes=episodes,
            constraint_violations=total_violations,
        )
