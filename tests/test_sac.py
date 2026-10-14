"""Tests for agents.SACAgent — Phase 6."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from simplant_trainer.agents.buffer import BoundedReplayBuffer
from simplant_trainer.agents.sac import QNetwork, SACAgent, SquashedGaussianActor
from simplant_trainer.value_objects.env_metadata import EnvMetadata
from simplant_trainer.value_objects.hyperparameters import HyperParameters
from simplant_trainer.value_objects.metrics import EpochMetrics
from simplant_trainer.value_objects.spaces import ActionSpace, ObservationSpace
from simplant_trainer.value_objects.transition import Transition


# ---------------------------------------------------------------------------
# Constants & Helpers
# ---------------------------------------------------------------------------

OBS_DIM = 4
ACTION_DIM = 2


def _make_env_metadata(
    obs_dim: int = OBS_DIM, action_dim: int = ACTION_DIM
) -> EnvMetadata:
    return EnvMetadata(
        env_id="test-env",
        observation_space=ObservationSpace(shape=(obs_dim,), low=-10.0, high=10.0),
        action_space=ActionSpace(shape=(action_dim,), low=-1.0, high=1.0),
    )


def _make_hp(**overrides: object) -> HyperParameters:
    defaults: dict = dict(
        learning_rate=1e-3,
        batch_size=8,
        gamma=0.99,
        buffer_capacity=100,
        extras={"tau": 0.005},
    )
    defaults.update(overrides)
    return HyperParameters(**defaults)


def _make_agent(
    obs_dim: int = OBS_DIM, action_dim: int = ACTION_DIM, **hp_overrides: object
) -> SACAgent:
    env_meta = _make_env_metadata(obs_dim, action_dim)
    hp = _make_hp(**hp_overrides)
    return SACAgent(env_meta, hp)


def _random_transition(
    obs_dim: int = OBS_DIM, action_dim: int = ACTION_DIM
) -> Transition:
    return Transition(
        state=np.random.randn(obs_dim),
        action=np.random.randn(action_dim),
        reward=float(np.random.randn()),
        next_state=np.random.randn(obs_dim),
        terminated=False,
    )


def _fill_agent_buffer(
    agent: SACAgent,
    n: int,
    obs_dim: int = OBS_DIM,
    action_dim: int = ACTION_DIM,
) -> None:
    for _ in range(n):
        agent.observe(_random_transition(obs_dim, action_dim))


# ---------------------------------------------------------------------------
# SquashedGaussianActor — construction
# ---------------------------------------------------------------------------


class TestSquashedGaussianActorConstruction:
    """Verify actor builds correctly."""

    def test_creates_with_valid_dims(self) -> None:
        actor = SquashedGaussianActor(OBS_DIM, ACTION_DIM)
        assert actor is not None

    def test_has_mu_and_log_std_layers(self) -> None:
        actor = SquashedGaussianActor(OBS_DIM, ACTION_DIM)
        assert hasattr(actor, "mu_layer")
        assert hasattr(actor, "log_std_layer")

    def test_forward_output_shapes(self) -> None:
        actor = SquashedGaussianActor(OBS_DIM, ACTION_DIM)
        obs = torch.randn(1, OBS_DIM)
        mu, log_std = actor.forward(obs)
        assert mu.shape == (1, ACTION_DIM)
        assert log_std.shape == (1, ACTION_DIM)


# ---------------------------------------------------------------------------
# SquashedGaussianActor — sample
# ---------------------------------------------------------------------------


class TestSquashedGaussianActorSample:
    """Verify stochastic sampling."""

    def test_returns_action_and_log_prob(self) -> None:
        actor = SquashedGaussianActor(OBS_DIM, ACTION_DIM)
        obs = torch.randn(1, OBS_DIM)
        result = actor.sample(obs)
        assert len(result) == 2

    def test_action_bounded(self) -> None:
        """Actions must be in [-1, 1] due to tanh squashing."""
        actor = SquashedGaussianActor(OBS_DIM, ACTION_DIM)
        obs = torch.randn(32, OBS_DIM)
        action, _ = actor.sample(obs)
        assert action.min() >= -1.0
        assert action.max() <= 1.0

    def test_action_shape(self) -> None:
        actor = SquashedGaussianActor(OBS_DIM, ACTION_DIM)
        obs = torch.randn(8, OBS_DIM)
        action, _ = actor.sample(obs)
        assert action.shape == (8, ACTION_DIM)

    def test_log_prob_shape(self) -> None:
        actor = SquashedGaussianActor(OBS_DIM, ACTION_DIM)
        obs = torch.randn(8, OBS_DIM)
        _, log_prob = actor.sample(obs)
        assert log_prob.shape == (8,)


# ---------------------------------------------------------------------------
# SquashedGaussianActor — deterministic
# ---------------------------------------------------------------------------


class TestSquashedGaussianActorDeterministic:
    """Verify deterministic action selection."""

    def test_returns_tensor(self) -> None:
        actor = SquashedGaussianActor(OBS_DIM, ACTION_DIM)
        obs = torch.randn(1, OBS_DIM)
        action = actor.deterministic_action(obs)
        assert isinstance(action, torch.Tensor)

    def test_bounded(self) -> None:
        actor = SquashedGaussianActor(OBS_DIM, ACTION_DIM)
        obs = torch.randn(32, OBS_DIM)
        action = actor.deterministic_action(obs)
        assert action.min() >= -1.0
        assert action.max() <= 1.0

    def test_shape(self) -> None:
        actor = SquashedGaussianActor(OBS_DIM, ACTION_DIM)
        obs = torch.randn(8, OBS_DIM)
        action = actor.deterministic_action(obs)
        assert action.shape == (8, ACTION_DIM)


# ---------------------------------------------------------------------------
# QNetwork — construction
# ---------------------------------------------------------------------------


class TestQNetworkConstruction:
    """Verify Q-network builds correctly."""

    def test_creates_with_valid_dims(self) -> None:
        q = QNetwork(OBS_DIM, ACTION_DIM)
        assert q is not None

    def test_has_qnet_attribute(self) -> None:
        q = QNetwork(OBS_DIM, ACTION_DIM)
        assert hasattr(q, "qnet")


# ---------------------------------------------------------------------------
# QNetwork — forward
# ---------------------------------------------------------------------------


class TestQNetworkForward:
    """Verify Q-value computation."""

    def test_output_shape(self) -> None:
        q = QNetwork(OBS_DIM, ACTION_DIM)
        obs = torch.randn(1, OBS_DIM)
        action = torch.randn(1, ACTION_DIM)
        value = q.forward(obs, action)
        assert value.shape == (1,)

    def test_batch_forward(self) -> None:
        q = QNetwork(OBS_DIM, ACTION_DIM)
        obs = torch.randn(16, OBS_DIM)
        action = torch.randn(16, ACTION_DIM)
        value = q.forward(obs, action)
        assert value.shape == (16,)

    def test_no_nan(self) -> None:
        q = QNetwork(OBS_DIM, ACTION_DIM)
        obs = torch.randn(32, OBS_DIM)
        action = torch.randn(32, ACTION_DIM)
        value = q.forward(obs, action)
        assert not torch.isnan(value).any()
        assert not torch.isinf(value).any()


# ---------------------------------------------------------------------------
# SACAgent — construction
# ---------------------------------------------------------------------------


class TestSACAgentConstruction:
    """Verify SACAgent initializes correctly."""

    def test_creates_successfully(self) -> None:
        agent = _make_agent()
        assert agent is not None

    def test_has_actor_and_critics(self) -> None:
        agent = _make_agent()
        assert isinstance(agent.actor, SquashedGaussianActor)
        assert isinstance(agent.q1, QNetwork)
        assert isinstance(agent.q2, QNetwork)

    def test_has_target_networks(self) -> None:
        agent = _make_agent()
        assert isinstance(agent.q1_target, QNetwork)
        assert isinstance(agent.q2_target, QNetwork)

    def test_has_optimizers(self) -> None:
        agent = _make_agent()
        assert isinstance(agent.actor_optimizer, torch.optim.Adam)
        assert isinstance(agent.q1_optimizer, torch.optim.Adam)
        assert isinstance(agent.q2_optimizer, torch.optim.Adam)
        assert isinstance(agent.alpha_optimizer, torch.optim.Adam)

    def test_buffer_created(self) -> None:
        agent = _make_agent()
        assert isinstance(agent.buffer, BoundedReplayBuffer)


# ---------------------------------------------------------------------------
# SACAgent — select_action
# ---------------------------------------------------------------------------


class TestSelectAction:
    """Verify action selection interface."""

    def test_returns_ndarray(self) -> None:
        agent = _make_agent()
        obs = np.random.randn(OBS_DIM)
        action = agent.select_action(obs)
        assert isinstance(action, np.ndarray)

    def test_correct_shape(self) -> None:
        agent = _make_agent()
        obs = np.random.randn(OBS_DIM)
        action = agent.select_action(obs)
        assert action.shape == (ACTION_DIM,)

    def test_deterministic_mode(self) -> None:
        agent = _make_agent()
        obs = np.random.randn(OBS_DIM)
        action = agent.select_action(obs, deterministic=True)
        assert isinstance(action, np.ndarray)
        assert action.shape == (ACTION_DIM,)

    def test_values_are_finite(self) -> None:
        agent = _make_agent()
        obs = np.random.randn(OBS_DIM)
        action = agent.select_action(obs)
        assert np.all(np.isfinite(action))


# ---------------------------------------------------------------------------
# SACAgent — observe
# ---------------------------------------------------------------------------


class TestObserve:
    """Verify transition storage."""

    def test_buffer_grows(self) -> None:
        agent = _make_agent()
        agent.observe(_random_transition())
        assert len(agent.buffer) == 1

    def test_multiple_observations(self) -> None:
        agent = _make_agent()
        _fill_agent_buffer(agent, 5)
        assert len(agent.buffer) == 5


# ---------------------------------------------------------------------------
# SACAgent — update
# ---------------------------------------------------------------------------


class TestUpdate:
    """Verify gradient update step."""

    def test_returns_none_when_insufficient_data(self) -> None:
        agent = _make_agent(batch_size=8)
        _fill_agent_buffer(agent, 3)
        assert agent.update() is None

    def test_returns_epoch_metrics_when_ready(self) -> None:
        agent = _make_agent(batch_size=8)
        _fill_agent_buffer(agent, 16)
        result = agent.update()
        assert isinstance(result, EpochMetrics)

    def test_metrics_have_finite_values(self) -> None:
        agent = _make_agent(batch_size=8)
        _fill_agent_buffer(agent, 16)
        metrics = agent.update()
        assert metrics is not None
        assert np.isfinite(metrics.policy_loss)
        assert np.isfinite(metrics.value_loss)
        assert np.isfinite(metrics.entropy)

    def test_entropy_non_negative(self) -> None:
        agent = _make_agent(batch_size=8)
        _fill_agent_buffer(agent, 16)
        metrics = agent.update()
        assert metrics is not None
        assert metrics.entropy >= 0.0

    def test_multiple_updates(self) -> None:
        agent = _make_agent(batch_size=8)
        _fill_agent_buffer(agent, 32)
        for _ in range(3):
            result = agent.update()
            assert isinstance(result, EpochMetrics)


# ---------------------------------------------------------------------------
# SACAgent — gradient flow
# ---------------------------------------------------------------------------


class TestGradientFlow:
    """Verify weights change after gradient step."""

    def test_actor_weights_change_after_update(self) -> None:
        agent = _make_agent(batch_size=8)
        _fill_agent_buffer(agent, 16)

        w_before = {
            k: v.clone() for k, v in agent.actor.state_dict().items()
        }

        agent.update()

        changed = False
        for k, v in agent.actor.state_dict().items():
            if not torch.equal(w_before[k], v):
                changed = True
                break
        assert changed, "Actor weights did not change after update"

    def test_critic_weights_change_after_update(self) -> None:
        agent = _make_agent(batch_size=8)
        _fill_agent_buffer(agent, 16)

        w_before = {
            k: v.clone() for k, v in agent.q1.state_dict().items()
        }

        agent.update()

        changed = False
        for k, v in agent.q1.state_dict().items():
            if not torch.equal(w_before[k], v):
                changed = True
                break
        assert changed, "Critic Q1 weights did not change after update"


# ---------------------------------------------------------------------------
# SACAgent — serialisation (safetensors)
# ---------------------------------------------------------------------------


class TestSerialisation:
    """Verify safetensors round-trip."""

    def test_save_returns_bytes(self) -> None:
        agent = _make_agent()
        data = agent.save_state()
        assert isinstance(data, bytes)
        assert len(data) > 0

    def test_load_restores_predictions(self) -> None:
        """After save -> load, select_action must produce identical output."""
        agent = _make_agent()
        obs = np.random.randn(OBS_DIM)

        action_before = agent.select_action(obs, deterministic=True)
        saved = agent.save_state()

        # Create fresh agent, load state
        agent2 = _make_agent()
        agent2.load_state(saved)

        action_after = agent2.select_action(obs, deterministic=True)
        np.testing.assert_array_almost_equal(action_before, action_after)

    def test_round_trip_preserves_networks(self) -> None:
        agent = _make_agent()
        saved = agent.save_state()

        agent2 = _make_agent()
        agent2.load_state(saved)

        for k in agent.actor.state_dict():
            torch.testing.assert_close(
                agent.actor.state_dict()[k],
                agent2.actor.state_dict()[k],
            )

    def test_no_unsafe_format_in_bytes(self) -> None:
        """Safetensors output must NOT contain unsupported serialization opcodes."""
        agent = _make_agent()
        data = agent.save_state()
        # Safetensors starts with a JSON header length (8 bytes LE),
        # not with protocol markers from other formats
        assert not data[:2] == b"\x80\x05"
        assert not data[:2] == b"\x80\x04"


# ---------------------------------------------------------------------------
# SACAgent — IAgent protocol conformance
# ---------------------------------------------------------------------------


class TestProtocolConformance:
    """SACAgent must structurally satisfy IAgent."""

    def test_satisfies_iagent(self) -> None:
        agent = _make_agent()
        assert callable(getattr(agent, "select_action", None))
        assert callable(getattr(agent, "observe", None))
        assert callable(getattr(agent, "update", None))
        assert callable(getattr(agent, "save_state", None))
        assert callable(getattr(agent, "load_state", None))
        assert callable(getattr(agent, "close", None))


# ---------------------------------------------------------------------------
# SACAgent — resource management
# ---------------------------------------------------------------------------


class TestResourceManagement:
    """Verify close() and cleanup."""

    def test_close_nils_networks(self) -> None:
        agent = _make_agent()
        agent.close()
        assert agent.actor is None
        assert agent.q1 is None
        assert agent.q2 is None
        assert agent.q1_target is None
        assert agent.q2_target is None

    def test_close_nils_buffer(self) -> None:
        agent = _make_agent()
        _fill_agent_buffer(agent, 5)
        agent.close()
        assert agent.buffer is None

    def test_close_is_idempotent(self) -> None:
        """Calling close() multiple times must not raise."""
        agent = _make_agent()
        agent.close()
        agent.close()


# ---------------------------------------------------------------------------
# SACAgent — no NaN/Inf
# ---------------------------------------------------------------------------


class TestNoNaN:
    """Verify forward passes produce finite values."""

    def test_forward_actor_finite(self) -> None:
        agent = _make_agent()
        obs = torch.randn(8, OBS_DIM)
        mu, log_std = agent.actor.forward(obs)
        assert not torch.isnan(mu).any()
        assert not torch.isnan(log_std).any()
        assert not torch.isinf(mu).any()
        assert not torch.isinf(log_std).any()

    def test_forward_critic_finite(self) -> None:
        agent = _make_agent()
        obs = torch.randn(8, OBS_DIM)
        action = torch.randn(8, ACTION_DIM)
        q1_val = agent.q1.forward(obs, action)
        q2_val = agent.q2.forward(obs, action)
        assert not torch.isnan(q1_val).any()
        assert not torch.isnan(q2_val).any()
        assert not torch.isinf(q1_val).any()
        assert not torch.isinf(q2_val).any()
