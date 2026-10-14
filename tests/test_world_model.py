"""Tests for world_model.WorldModel — Phase 5."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from simplant_trainer.networks.mlp import MLP
from simplant_trainer.value_objects.transition import Transition
from simplant_trainer.world_model.world_model import WorldModel


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

OBS_DIM = 4
ACT_DIM = 2
INPUT_DIM = OBS_DIM + ACT_DIM  # transition/reward nets receive cat(s, a)


def _make_world_model(
    *,
    lr: float = 1e-3,
    buffer_capacity: int = 100,
) -> WorldModel:
    """Create a small WorldModel suitable for tests."""
    transition = MLP(layers=[INPUT_DIM, 32, OBS_DIM])
    reward = MLP(layers=[INPUT_DIM, 32, 1])
    return WorldModel(
        transition=transition,
        reward=reward,
        lr=lr,
        buffer_capacity=buffer_capacity,
    )


def _random_transition() -> Transition:
    """Create a random Transition with matching shapes."""
    return Transition(
        state=np.random.randn(OBS_DIM),
        action=np.random.randn(ACT_DIM),
        reward=float(np.random.randn()),
        next_state=np.random.randn(OBS_DIM),
        terminated=False,
    )


def _fill_buffer(wm: WorldModel, n: int) -> None:
    """Record *n* random transitions into the world model."""
    for _ in range(n):
        wm.record(_random_transition())


# ---------------------------------------------------------------------------
# Construction — happy path
# ---------------------------------------------------------------------------


class TestConstruction:
    """Verify WorldModel builds correctly."""

    def test_creates_with_defaults(self) -> None:
        wm = _make_world_model()
        assert wm.buffer_size == 0
        assert wm.buffer_capacity == 100

    def test_custom_lr_and_capacity(self) -> None:
        wm = _make_world_model(lr=5e-4, buffer_capacity=500)
        assert wm.buffer_capacity == 500

    def test_repr_contains_key_info(self) -> None:
        wm = _make_world_model()
        r = repr(wm)
        assert "WorldModel" in r
        assert "transition=" in r
        assert "reward=" in r


# ---------------------------------------------------------------------------
# Construction — validation
# ---------------------------------------------------------------------------


class TestConstructionValidation:
    """Construction must fail fast on invalid inputs."""

    def test_zero_lr(self) -> None:
        with pytest.raises(ValueError, match="lr must be positive"):
            _make_world_model(lr=0)

    def test_negative_lr(self) -> None:
        with pytest.raises(ValueError, match="lr must be positive"):
            _make_world_model(lr=-1e-3)

    def test_zero_capacity(self) -> None:
        with pytest.raises(ValueError, match="buffer_capacity"):
            _make_world_model(buffer_capacity=0)

    def test_negative_capacity(self) -> None:
        with pytest.raises(ValueError, match="buffer_capacity"):
            _make_world_model(buffer_capacity=-10)


# ---------------------------------------------------------------------------
# Prediction
# ---------------------------------------------------------------------------


class TestPrediction:
    """Verify predict_transition, predict_reward, and predict."""

    def test_predict_transition_shape(self) -> None:
        wm = _make_world_model()
        state = torch.randn(1, OBS_DIM)
        action = torch.randn(1, ACT_DIM)
        ns = wm.predict_transition(state, action)
        assert ns.shape == (1, OBS_DIM)

    def test_predict_reward_shape(self) -> None:
        wm = _make_world_model()
        state = torch.randn(1, OBS_DIM)
        action = torch.randn(1, ACT_DIM)
        r = wm.predict_reward(state, action)
        assert r.shape == (1, 1)

    def test_predict_combined_returns_tuple(self) -> None:
        wm = _make_world_model()
        state = torch.randn(1, OBS_DIM)
        action = torch.randn(1, ACT_DIM)
        ns, r = wm.predict(state, action)
        assert ns.shape == (1, OBS_DIM)
        assert r.shape == (1, 1)

    def test_predict_batch(self) -> None:
        """Batched inputs must work."""
        wm = _make_world_model()
        state = torch.randn(16, OBS_DIM)
        action = torch.randn(16, ACT_DIM)
        ns, r = wm.predict(state, action)
        assert ns.shape == (16, OBS_DIM)
        assert r.shape == (16, 1)

    def test_predict_no_grad(self) -> None:
        """Predictions must not accumulate gradients."""
        wm = _make_world_model()
        state = torch.randn(1, OBS_DIM)
        action = torch.randn(1, ACT_DIM)
        ns = wm.predict_transition(state, action)
        assert not ns.requires_grad

    def test_predict_no_nan(self) -> None:
        wm = _make_world_model()
        state = torch.randn(32, OBS_DIM)
        action = torch.randn(32, ACT_DIM)
        ns, r = wm.predict(state, action)
        assert not torch.isnan(ns).any()
        assert not torch.isnan(r).any()


# ---------------------------------------------------------------------------
# Data collection (record)
# ---------------------------------------------------------------------------


class TestRecord:
    """Verify the internal ring-buffer."""

    def test_buffer_grows(self) -> None:
        wm = _make_world_model(buffer_capacity=10)
        assert wm.buffer_size == 0
        _fill_buffer(wm, 5)
        assert wm.buffer_size == 5

    def test_buffer_caps_at_capacity(self) -> None:
        wm = _make_world_model(buffer_capacity=10)
        _fill_buffer(wm, 15)
        assert wm.buffer_size == 10  # ring-buffer wraps

    def test_lazy_init(self) -> None:
        wm = _make_world_model()
        assert wm._states is None  # noqa: SLF001
        wm.record(_random_transition())
        assert wm._states is not None  # noqa: SLF001


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------


class TestTrainStep:
    """Verify train_step optimises both sub-models."""

    def test_returns_loss_dict(self) -> None:
        wm = _make_world_model(buffer_capacity=100)
        _fill_buffer(wm, 50)
        losses = wm.train_step(batch_size=16)
        assert "transition_loss" in losses
        assert "reward_loss" in losses
        assert isinstance(losses["transition_loss"], float)
        assert isinstance(losses["reward_loss"], float)

    def test_losses_are_finite(self) -> None:
        wm = _make_world_model()
        _fill_buffer(wm, 50)
        losses = wm.train_step(batch_size=16)
        assert np.isfinite(losses["transition_loss"])
        assert np.isfinite(losses["reward_loss"])

    def test_multiple_steps_reduce_loss(self) -> None:
        """After many gradient steps, losses should decrease."""
        # Use a learnable pattern: next_state = state (identity)
        wm = _make_world_model(lr=1e-2, buffer_capacity=200)
        for _ in range(200):
            s = np.random.randn(OBS_DIM) * 0.1
            wm.record(
                Transition(
                    state=s,
                    action=np.zeros(ACT_DIM),
                    reward=0.0,
                    next_state=s,  # identity — easy to learn
                    terminated=False,
                )
            )

        initial_loss = wm.train_step(batch_size=32)["transition_loss"]
        for _ in range(100):
            wm.train_step(batch_size=32)
        final_loss = wm.train_step(batch_size=32)["transition_loss"]

        assert final_loss < initial_loss

    def test_empty_buffer_raises(self) -> None:
        wm = _make_world_model()
        with pytest.raises(ValueError, match="Not enough data"):
            wm.train_step(batch_size=16)

    def test_insufficient_data_raises(self) -> None:
        wm = _make_world_model()
        _fill_buffer(wm, 5)
        with pytest.raises(ValueError, match="Not enough data"):
            wm.train_step(batch_size=10)

    def test_zero_batch_size_raises(self) -> None:
        wm = _make_world_model()
        _fill_buffer(wm, 10)
        with pytest.raises(ValueError, match="batch_size must be positive"):
            wm.train_step(batch_size=0)

    def test_negative_batch_size_raises(self) -> None:
        wm = _make_world_model()
        _fill_buffer(wm, 10)
        with pytest.raises(ValueError, match="batch_size must be positive"):
            wm.train_step(batch_size=-5)


# ---------------------------------------------------------------------------
# Imagination
# ---------------------------------------------------------------------------


class TestImagine:
    """Verify synthetic trajectory generation."""

    def test_returns_correct_number_of_transitions(self) -> None:
        wm = _make_world_model()
        state = torch.randn(1, OBS_DIM)
        policy = lambda s: torch.randn(1, ACT_DIM)  # noqa: E731

        result = wm.imagine(state, policy, steps=5)
        assert len(result) == 5

    def test_all_elements_are_transitions(self) -> None:
        wm = _make_world_model()
        state = torch.randn(1, OBS_DIM)
        policy = lambda s: torch.randn(1, ACT_DIM)  # noqa: E731

        for t in wm.imagine(state, policy, steps=3):
            assert isinstance(t, Transition)

    def test_transitions_chain_correctly(self) -> None:
        """Each transition's next_state must be the next transition's state."""
        wm = _make_world_model()
        state = torch.randn(1, OBS_DIM)
        policy = lambda s: torch.randn(1, ACT_DIM)  # noqa: E731

        trajectory = wm.imagine(state, policy, steps=4)
        for i in range(len(trajectory) - 1):
            np.testing.assert_array_almost_equal(
                trajectory[i].next_state, trajectory[i + 1].state
            )

    def test_all_terminated_false(self) -> None:
        """Imagined transitions are never terminated."""
        wm = _make_world_model()
        state = torch.randn(1, OBS_DIM)
        policy = lambda s: torch.randn(1, ACT_DIM)  # noqa: E731

        for t in wm.imagine(state, policy, steps=3):
            assert t.terminated is False

    def test_shapes_match_obs_and_action(self) -> None:
        wm = _make_world_model()
        state = torch.randn(1, OBS_DIM)
        policy = lambda s: torch.randn(1, ACT_DIM)  # noqa: E731

        t = wm.imagine(state, policy, steps=1)[0]
        assert t.state.shape == (OBS_DIM,)
        assert t.action.shape == (ACT_DIM,)
        assert t.next_state.shape == (OBS_DIM,)

    def test_policy_receives_numpy_action(self) -> None:
        """policy_fn may return ndarray — WorldModel must handle it."""
        wm = _make_world_model()
        state = torch.randn(1, OBS_DIM)
        policy = lambda s: np.random.randn(1, ACT_DIM)  # noqa: E731

        result = wm.imagine(state, policy, steps=2)
        assert len(result) == 2

    def test_zero_steps_raises(self) -> None:
        wm = _make_world_model()
        state = torch.randn(1, OBS_DIM)
        with pytest.raises(ValueError, match="steps must be positive"):
            wm.imagine(state, lambda s: torch.randn(1, ACT_DIM), steps=0)

    def test_negative_steps_raises(self) -> None:
        wm = _make_world_model()
        state = torch.randn(1, OBS_DIM)
        with pytest.raises(ValueError, match="steps must be positive"):
            wm.imagine(state, lambda s: torch.randn(1, ACT_DIM), steps=-3)


# ---------------------------------------------------------------------------
# Serialisation (save_state / load_state)
# ---------------------------------------------------------------------------


class TestSerialisation:
    """Verify safetensors round-trip."""

    def test_save_returns_bytes(self) -> None:
        wm = _make_world_model()
        data = wm.save_state()
        assert isinstance(data, bytes)
        assert len(data) > 0

    def test_load_restores_predictions(self) -> None:
        """After save → load, predictions must be identical."""
        wm = _make_world_model()
        state = torch.randn(1, OBS_DIM)
        action = torch.randn(1, ACT_DIM)

        ns_before, r_before = wm.predict(state, action)
        saved = wm.save_state()

        # Create fresh model, load state
        wm2 = _make_world_model()
        wm2.load_state(saved)

        ns_after, r_after = wm2.predict(state, action)
        torch.testing.assert_close(ns_before, ns_after)
        torch.testing.assert_close(r_before, r_after)

    def test_no_pickle_in_bytes(self) -> None:
        """Safetensors output must NOT contain pickle opcodes."""
        wm = _make_world_model()
        data = wm.save_state()
        # Pickle starts with \\x80 followed by protocol version
        # safetensors starts with a JSON header length (8 bytes LE)
        assert not data[:2] == b"\x80\x05"  # pickle protocol 5
        assert not data[:2] == b"\x80\x04"  # pickle protocol 4


# ---------------------------------------------------------------------------
# Resource management
# ---------------------------------------------------------------------------


class TestResourceManagement:
    """Verify close() and context manager."""

    def test_close_releases_buffers(self) -> None:
        wm = _make_world_model()
        _fill_buffer(wm, 10)
        assert wm.buffer_size == 10

        wm.close()
        assert wm.buffer_size == 0
        assert wm._states is None  # noqa: SLF001

    def test_context_manager(self) -> None:
        with _make_world_model() as wm:
            _fill_buffer(wm, 5)
            assert wm.buffer_size == 5
        # After __exit__, buffers should be released
        assert wm.buffer_size == 0

    def test_close_is_idempotent(self) -> None:
        """Calling close() multiple times must not raise."""
        wm = _make_world_model()
        _fill_buffer(wm, 5)
        wm.close()
        wm.close()  # second call — no error


# ---------------------------------------------------------------------------
# Gradient flow (sanity check)
# ---------------------------------------------------------------------------


class TestGradientFlow:
    """Verify train_step actually updates weights."""

    def test_weights_change_after_train_step(self) -> None:
        wm = _make_world_model()
        _fill_buffer(wm, 50)

        # Snapshot transition net weights
        w_before = {
            k: v.clone()
            for k, v in wm._transition_net.state_dict().items()  # noqa: SLF001
        }

        wm.train_step(batch_size=16)

        for k, v in wm._transition_net.state_dict().items():  # noqa: SLF001
            assert not torch.equal(w_before[k], v), (
                f"Weight {k} did not change after train_step"
            )
