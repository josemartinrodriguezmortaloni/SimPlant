"""Training session — builder + orchestrator."""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import TYPE_CHECKING

from simplant_trainer.agents.interface import IAgent
from simplant_trainer.checkpoint.interface import ICheckpointStore
from simplant_trainer.checkpoint.safetensors_store import SafeTensorsCheckpointStore
from simplant_trainer.environment.interface import IEnvironment
from simplant_trainer.strategies.interface import IStrategy
from simplant_trainer.strategies.model_free import ModelFreeStrategy
from simplant_trainer.tracking.in_memory import InMemoryTracker
from simplant_trainer.tracking.interface import IMetricsTracker
from simplant_trainer.value_objects.checkpoint_data import CheckpointData
from simplant_trainer.value_objects.hyperparameters import HyperParameters
from simplant_trainer.value_objects.metrics import EpochMetrics
from simplant_trainer.value_objects.stop_criteria import StopCriteria
from simplant_trainer.value_objects.training_result import TrainingResult

if TYPE_CHECKING:
    from simplant_trainer.world_model.world_model import WorldModel

logger = logging.getLogger(__name__)

_FALLBACK_HP = HyperParameters(
    learning_rate=3e-4,
    batch_size=64,
    gamma=0.99,
    buffer_capacity=2048,
)


class Session:
    """Builder + orchestrator for a training run.

    Replaces the old monolithic ``Trainer``.  The user wires custom
    components via builder methods and delegates epoch execution to an
    :class:`IStrategy`.

    Usage::

        session = (
            Session(agent, env)
            .use_tracker(my_tracker)
            .use_strategy(my_strategy)
        )
        result = session.train(max_epochs=500, target_reward=95.0)

    Default components (overridable):

    ==================  ======================================
    Component           Default
    ==================  ======================================
    tracker             ``InMemoryTracker(max_entries=5000)``
    checkpointer        ``SafeTensorsCheckpointStore("./checkpoints")``
    strategy            ``ModelFreeStrategy(steps_per_epoch=2048)``
    world_model         auto-detected from agent, or ``None``
    hyperparameters     auto-detected from agent, or fallback
    ==================  ======================================
    """

    def __init__(self, agent: IAgent, env: IEnvironment) -> None:
        self._agent = agent
        self._env = env

        # --- defaults (all overridable via builder) ---
        self._tracker: IMetricsTracker = InMemoryTracker(max_entries=5000)
        self._checkpointer: ICheckpointStore = SafeTensorsCheckpointStore(
            Path("./checkpoints"),
        )
        self._strategy: IStrategy = ModelFreeStrategy()
        self._world_model: WorldModel | None = getattr(agent, "world_model", None)
        self._hyperparameters: HyperParameters | None = getattr(
            agent,
            "hyperparameters",
            None,
        )

    # === builder methods (return self for chaining) =======================

    def use_tracker(self, tracker: IMetricsTracker) -> Session:
        """Replace the default metrics tracker."""
        self._tracker = tracker
        return self

    def use_checkpointer(self, store: ICheckpointStore) -> Session:
        """Replace the default checkpoint store."""
        self._checkpointer = store
        return self

    def use_strategy(self, strategy: IStrategy) -> Session:
        """Replace the default training strategy."""
        self._strategy = strategy
        return self

    def use_world_model(self, model: WorldModel) -> Session:
        """Attach (or replace) a world model for model-based strategies."""
        self._world_model = model
        return self

    def use_hyperparameters(self, hp: HyperParameters) -> Session:
        """Explicitly set hyperparameters for checkpoint metadata."""
        self._hyperparameters = hp
        return self

    # === training entry-points ============================================

    def train(
        self,
        max_epochs: int | None = None,
        target_reward: float | None = None,
        patience: int = 10,
        stop: StopCriteria | None = None,
        checkpoint_every: int = 50,
    ) -> TrainingResult:
        """Run a full training session from scratch.

        Either pass a pre-built *stop* criteria, **or** the individual
        ``max_epochs`` / ``target_reward`` / ``patience`` parameters
        (which are forwarded to :class:`StopCriteria`).

        Returns
        -------
        TrainingResult
            Immutable summary of the completed run.
        """
        criteria = self._build_criteria(stop, max_epochs, target_reward, patience)
        return self._run_loop(
            criteria, start_epoch=0, checkpoint_every=checkpoint_every
        )

    def resume(
        self,
        checkpoint_id: str,
        max_epochs: int | None = None,
        target_reward: float | None = None,
        patience: int = 10,
        stop: StopCriteria | None = None,
        checkpoint_every: int = 50,
    ) -> TrainingResult:
        """Resume training from a previously saved checkpoint.

        Loads the agent state from *checkpoint_id* and continues the
        training loop from the checkpoint's epoch.
        """
        ckpt = self._checkpointer.load(checkpoint_id)
        self._agent.load_state(ckpt.agent_state)
        logger.info(
            "Resumed from checkpoint '%s' (epoch %d)", checkpoint_id, ckpt.epoch
        )

        criteria = self._build_criteria(stop, max_epochs, target_reward, patience)
        return self._run_loop(
            criteria,
            start_epoch=ckpt.epoch,
            checkpoint_every=checkpoint_every,
        )

    # === internal =========================================================

    @staticmethod
    def _build_criteria(
        stop: StopCriteria | None,
        max_epochs: int | None,
        target_reward: float | None,
        patience: int,
    ) -> StopCriteria:
        if stop is not None:
            return stop
        return StopCriteria(
            max_epochs=max_epochs,
            target_reward=target_reward,
            patience=patience,
        )

    def _run_loop(
        self,
        criteria: StopCriteria,
        start_epoch: int,
        checkpoint_every: int,
    ) -> TrainingResult:
        start_time = time.monotonic()
        epoch = start_epoch
        reward_history: list[float] = []
        last_metrics: EpochMetrics | None = None

        while not criteria.is_met(epoch, reward_history):
            epoch += 1

            # 1. Delegate experience collection to the strategy
            epoch_rewards = self._strategy.run_epoch(
                self._agent,
                self._env,
                self._world_model,
            )

            # 2. Gradient step
            raw_metrics = self._agent.update()

            # 3. Record metrics when the agent updated AND episodes finished
            if raw_metrics is not None and epoch_rewards:
                mean_reward = sum(epoch_rewards) / len(epoch_rewards)
                metrics = EpochMetrics(
                    epoch=epoch,
                    mean_reward=mean_reward,
                    policy_loss=raw_metrics.policy_loss,
                    value_loss=raw_metrics.value_loss,
                    entropy=raw_metrics.entropy,
                )
                self._tracker.record(metrics)
                reward_history.append(mean_reward)
                last_metrics = metrics

            # 4. Periodic checkpoint
            if checkpoint_every > 0 and epoch % checkpoint_every == 0:
                self._save_checkpoint(epoch)

        elapsed = time.monotonic() - start_time
        _, stop_reason = criteria.check(epoch, reward_history)

        if last_metrics is None:
            last_metrics = EpochMetrics(
                epoch=epoch,
                mean_reward=0.0,
                policy_loss=0.0,
                value_loss=0.0,
                entropy=0.0,
            )

        logger.info(
            "Training finished — epoch %d, reason: %s (%.1fs)",
            epoch,
            stop_reason,
            elapsed,
        )

        return TrainingResult(
            final_epoch=epoch,
            final_metrics=last_metrics,
            stop_reason=stop_reason or "training complete",
            elapsed_seconds=elapsed,
        )

    def _save_checkpoint(self, epoch: int) -> None:
        hp = self._hyperparameters or _FALLBACK_HP
        ckpt = CheckpointData(
            checkpoint_id=f"epoch_{epoch}",
            epoch=epoch,
            agent_state=self._agent.save_state(),
            hyperparameters=hp,
        )
        self._checkpointer.save(ckpt)
        logger.debug("Checkpoint saved: epoch_%d", epoch)
