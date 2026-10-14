"""Tests for agents.BoundedReplayBuffer — Phase 6."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from simplant_trainer.agents.buffer import BoundedReplayBuffer
from simplant_trainer.value_objects.transition import Transition


# ---------------------------------------------------------------------------
# Constants & Helpers
# ---------------------------------------------------------------------------

OBS_DIM = 4
ACTION_DIM = 2


def _random_transition(
    obs_dim: int = OBS_DIM, action_dim: int = ACTION_DIM
) -> Transition:
    """Create a random Transition with matching shapes."""
    return Transition(
        state=np.random.randn(obs_dim),
        action=np.random.randn(action_dim),
        reward=float(np.random.randn()),
        next_state=np.random.randn(obs_dim),
        terminated=False,
    )


def _fill_buffer(buf: BoundedReplayBuffer, n: int) -> None:
    """Record *n* random transitions into the buffer."""
    for _ in range(n):
        buf.add(_random_transition())


# ---------------------------------------------------------------------------
# Construction — happy path
# ---------------------------------------------------------------------------


class TestConstruction:
    """Verify BoundedReplayBuffer builds correctly."""

    def test_happy_path(self) -> None:
        buf = BoundedReplayBuffer(capacity=100, obs_shape=(OBS_DIM,), action_shape=(ACTION_DIM,))
        assert len(buf) == 0
        assert buf.capacity == 100

    def test_multidimensional_shapes(self) -> None:
        buf = BoundedReplayBuffer(capacity=50, obs_shape=(4, 4), action_shape=(2, 3))
        assert buf.obs_shape == (4, 4)
        assert buf.action_shape == (2, 3)
        assert buf.obs_arr.shape == (50, 4, 4)
        assert buf.actions_arr.shape == (50, 2, 3)

    def test_capacity_one(self) -> None:
        buf = BoundedReplayBuffer(capacity=1, obs_shape=(OBS_DIM,), action_shape=(ACTION_DIM,))
        buf.add(_random_transition())
        assert len(buf) == 1
        sample = buf.sample(1)
        assert sample["obs"].shape == (1, OBS_DIM)


# ---------------------------------------------------------------------------
# Construction — validation
# ---------------------------------------------------------------------------


class TestConstructionValidation:
    """Construction must fail fast on invalid inputs."""

    def test_zero_capacity_raises(self) -> None:
        with pytest.raises(ValueError, match="positive"):
            BoundedReplayBuffer(capacity=0, obs_shape=(OBS_DIM,), action_shape=(ACTION_DIM,))

    def test_negative_capacity_raises(self) -> None:
        with pytest.raises(ValueError, match="positive"):
            BoundedReplayBuffer(capacity=-5, obs_shape=(OBS_DIM,), action_shape=(ACTION_DIM,))

    def test_invalid_obs_shape_raises(self) -> None:
        with pytest.raises(ValueError, match="at least one dimension"):
            BoundedReplayBuffer(capacity=10, obs_shape=(), action_shape=(ACTION_DIM,))

    def test_invalid_action_shape_raises(self) -> None:
        with pytest.raises(TypeError):
            BoundedReplayBuffer(capacity=10, obs_shape=(OBS_DIM,), action_shape="bad")  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Add
# ---------------------------------------------------------------------------


class TestAdd:
    """Verify add() stores transitions correctly."""

    def test_single_add_increments_size(self) -> None:
        buf = BoundedReplayBuffer(capacity=10, obs_shape=(OBS_DIM,), action_shape=(ACTION_DIM,))
        buf.add(_random_transition())
        assert len(buf) == 1

    def test_multiple_adds(self) -> None:
        buf = BoundedReplayBuffer(capacity=10, obs_shape=(OBS_DIM,), action_shape=(ACTION_DIM,))
        _fill_buffer(buf, 5)
        assert len(buf) == 5

    def test_data_stored_correctly(self) -> None:
        """Add a known transition and verify it can be sampled back."""
        buf = BoundedReplayBuffer(capacity=10, obs_shape=(OBS_DIM,), action_shape=(ACTION_DIM,))
        state = np.ones(OBS_DIM) * 42.0
        action = np.ones(ACTION_DIM) * 7.0
        t = Transition(
            state=state,
            action=action,
            reward=3.14,
            next_state=state * 2,
            terminated=True,
        )
        buf.add(t)
        sample = buf.sample(1)

        np.testing.assert_array_almost_equal(sample["obs"].numpy()[0], state)
        np.testing.assert_array_almost_equal(sample["actions"].numpy()[0], action)
        assert abs(sample["rewards"].item() - 3.14) < 1e-5
        assert sample["dones"].item() == 1.0


# ---------------------------------------------------------------------------
# Sample
# ---------------------------------------------------------------------------


class TestSample:
    """Verify sample() returns correct batches."""

    def test_returns_dict_with_correct_keys(self) -> None:
        buf = BoundedReplayBuffer(capacity=20, obs_shape=(OBS_DIM,), action_shape=(ACTION_DIM,))
        _fill_buffer(buf, 10)
        sample = buf.sample(4)
        assert set(sample.keys()) == {"obs", "next_state", "actions", "rewards", "dones"}

    def test_correct_batch_shapes(self) -> None:
        buf = BoundedReplayBuffer(capacity=20, obs_shape=(OBS_DIM,), action_shape=(ACTION_DIM,))
        _fill_buffer(buf, 10)
        sample = buf.sample(4)
        assert sample["obs"].shape == (4, OBS_DIM)
        assert sample["next_state"].shape == (4, OBS_DIM)
        assert sample["actions"].shape == (4, ACTION_DIM)
        assert sample["rewards"].shape == (4,)
        assert sample["dones"].shape == (4,)

    def test_batch_size_equals_buffer_size(self) -> None:
        """Edge case: sample the entire buffer."""
        buf = BoundedReplayBuffer(capacity=10, obs_shape=(OBS_DIM,), action_shape=(ACTION_DIM,))
        _fill_buffer(buf, 10)
        sample = buf.sample(10)
        assert sample["obs"].shape == (10, OBS_DIM)

    def test_raises_when_batch_exceeds_size(self) -> None:
        buf = BoundedReplayBuffer(capacity=20, obs_shape=(OBS_DIM,), action_shape=(ACTION_DIM,))
        _fill_buffer(buf, 3)
        with pytest.raises(ValueError, match="exceeds buffer size"):
            buf.sample(5)

    def test_samples_are_torch_float(self) -> None:
        """All tensors in sample must be float32."""
        buf = BoundedReplayBuffer(capacity=20, obs_shape=(OBS_DIM,), action_shape=(ACTION_DIM,))
        _fill_buffer(buf, 10)
        sample = buf.sample(4)
        for key, tensor in sample.items():
            assert isinstance(tensor, torch.Tensor)
            assert tensor.dtype == torch.float32


# ---------------------------------------------------------------------------
# Ring buffer semantics
# ---------------------------------------------------------------------------


class TestRingBuffer:
    """Verify circular overwrite behaviour."""

    def test_size_caps_at_capacity(self) -> None:
        buf = BoundedReplayBuffer(capacity=10, obs_shape=(OBS_DIM,), action_shape=(ACTION_DIM,))
        _fill_buffer(buf, 15)
        assert len(buf) == 10

    def test_oldest_data_overwritten(self) -> None:
        """With capacity 3, the 4th transition overwrites the 1st."""
        buf = BoundedReplayBuffer(capacity=3, obs_shape=(OBS_DIM,), action_shape=(ACTION_DIM,))

        # Add 3 transitions with known states
        for i in range(3):
            t = Transition(
                state=np.full(OBS_DIM, float(i)),
                action=np.zeros(ACTION_DIM),
                reward=0.0,
                next_state=np.zeros(OBS_DIM),
                terminated=False,
            )
            buf.add(t)

        # Add a 4th — should overwrite index 0 (state=[0,0,0,0])
        t4 = Transition(
            state=np.full(OBS_DIM, 99.0),
            action=np.zeros(ACTION_DIM),
            reward=0.0,
            next_state=np.zeros(OBS_DIM),
            terminated=False,
        )
        buf.add(t4)

        # All 3 stored states should be [1,...], [2,...], [99,...]
        stored = set()
        for idx in range(3):
            stored.add(buf.obs_arr[idx][0])
        assert 0.0 not in stored
        assert 99.0 in stored

    def test_pointer_wraps_correctly(self) -> None:
        buf = BoundedReplayBuffer(capacity=5, obs_shape=(OBS_DIM,), action_shape=(ACTION_DIM,))
        _fill_buffer(buf, 5)
        assert buf._ptr == 0  # noqa: SLF001


# ---------------------------------------------------------------------------
# Length tracking
# ---------------------------------------------------------------------------


class TestLength:
    """Verify __len__ tracks buffer size correctly."""

    def test_empty_buffer_length_zero(self) -> None:
        buf = BoundedReplayBuffer(capacity=10, obs_shape=(OBS_DIM,), action_shape=(ACTION_DIM,))
        assert len(buf) == 0

    def test_length_tracks_adds_up_to_capacity(self) -> None:
        buf = BoundedReplayBuffer(capacity=5, obs_shape=(OBS_DIM,), action_shape=(ACTION_DIM,))
        for i in range(1, 8):
            buf.add(_random_transition())
            assert len(buf) == min(i, 5)
