"""Model-Free strategy — learn only from real environment interactions."""

from __future__ import annotations

from simplant_trainer.agents.interface import IAgent
from simplant_trainer.environment.interface import IEnvironment
from simplant_trainer.value_objects.transition import Transition


class ModelFreeStrategy:
    """Default strategy: the agent learns exclusively from real interactions.

    Each epoch runs a fixed number of environment steps.  Completed
    episodes contribute their cumulative reward to the returned list;
    episodes that are still running when the step budget is exhausted
    are silently continued in the next epoch.

    Parameters
    ----------
    steps_per_epoch:
        Number of environment steps executed per ``run_epoch`` call.
        Must be a positive integer.
    """

    def __init__(self, steps_per_epoch: int = 2048) -> None:
        if not isinstance(steps_per_epoch, int) or steps_per_epoch <= 0:
            raise ValueError(
                f"steps_per_epoch must be a positive int, got {steps_per_epoch}"
            )
        self._steps_per_epoch = steps_per_epoch

    def run_epoch(
        self,
        agent: IAgent,
        env: IEnvironment,
        world_model: object | None,
    ) -> list[float]:
        """Execute one epoch of real-environment interaction.

        *world_model* is accepted for protocol compatibility but ignored
        (Model-Free does not use a learned dynamics model).

        Returns
        -------
        list[float]
            Cumulative reward of each episode that completed within this
            epoch.  May be empty if no episode finished.
        """
        episode_rewards: list[float] = []
        state = env.reset()
        episode_reward = 0.0

        for _ in range(self._steps_per_epoch):
            action = agent.select_action(state, deterministic=False)
            result = env.step(action)

            transition = Transition(
                state=state,
                action=action,
                reward=result.reward,
                next_state=result.observation,
                terminated=result.terminated,
            )
            agent.observe(transition)

            episode_reward += result.reward
            state = result.observation

            if result.done:
                episode_rewards.append(episode_reward)
                state = env.reset()
                episode_reward = 0.0

        return episode_rewards
