"""World Model — learned environment dynamics for Model-Based RL.

Composes a *transition model* (predicts next state) and a *reward model*
(predicts immediate reward) to enable Dyna-style data augmentation and
planning-based action selection.

The WorldModel is **opt-in**: when passed to an agent or session the
training loop switches from Model-Free to Model-Based.  When omitted,
the system operates in pure Model-Free mode.

Examples
--------
>>> from simplant_trainer.networks.mlp import MLP
>>> transition = MLP(layers=[6, 128, 128, 4])   # obs_dim=4, act_dim=2
>>> reward     = MLP(layers=[6, 64, 64, 1])
>>> wm = WorldModel(transition=transition, reward=reward, lr=1e-3)
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np
import torch
from numpy.typing import NDArray
from safetensors import torch as safetensors_torch
from torch import Tensor, nn, optim

from simplant_trainer.networks.interface import INetwork
from simplant_trainer.value_objects.transition import Transition


class WorldModel:
    """Learned environment model: predicts transitions and rewards.

    Composed of two :class:`INetwork` instances — one for state
    dynamics and one for reward prediction.  Maintains an internal
    ring-buffer of real transitions used exclusively to train **these**
    networks (separate from the agent's experience buffer).

    Parameters
    ----------
    transition:
        Network that predicts the next observation given
        ``cat(state, action)`` as input.  Output dim must equal
        ``obs_dim``.
    reward:
        Network that predicts scalar reward given
        ``cat(state, action)`` as input.  Output dim must be 1.
    lr:
        Learning rate for both Adam optimisers.
    buffer_capacity:
        Maximum number of real transitions stored for training.
        Oldest entries are overwritten in a circular fashion.

    Raises
    ------
    ValueError
        If *lr* or *buffer_capacity* are non-positive.
    """

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    def __init__(
        self,
        transition: INetwork,
        reward: INetwork,
        *,
        lr: float = 1e-3,
        buffer_capacity: int = 10_000,
    ) -> None:
        if lr <= 0:
            msg = f"lr must be positive, got {lr}"
            raise ValueError(msg)
        if not isinstance(buffer_capacity, int) or buffer_capacity <= 0:
            msg = f"buffer_capacity must be a positive integer, got {buffer_capacity!r}"
            raise ValueError(msg)

        self._transition_net: nn.Module = transition  # type: ignore[assignment]
        self._reward_net: nn.Module = reward  # type: ignore[assignment]
        self._lr = lr

        self._transition_optim = optim.Adam(transition.parameters(), lr=lr)
        self._reward_optim = optim.Adam(reward.parameters(), lr=lr)

        # Internal ring-buffer — lazily allocated on first record().
        self._capacity = buffer_capacity
        self._ptr: int = 0
        self._size: int = 0
        self._states: NDArray[np.floating[Any]] | None = None
        self._actions: NDArray[np.floating[Any]] | None = None
        self._rewards: NDArray[np.floating[Any]] | None = None
        self._next_states: NDArray[np.floating[Any]] | None = None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _init_buffers(
        self, obs_shape: tuple[int, ...], action_shape: tuple[int, ...]
    ) -> None:
        """Pre-allocate numpy arrays for the ring-buffer."""
        self._states = np.zeros((self._capacity, *obs_shape), dtype=np.float64)
        self._actions = np.zeros((self._capacity, *action_shape), dtype=np.float64)
        self._rewards = np.zeros(self._capacity, dtype=np.float64)
        self._next_states = np.zeros((self._capacity, *obs_shape), dtype=np.float64)

    @property
    def _device(self) -> torch.device:
        """Infer device from the transition network's first parameter."""
        return next(self._transition_net.parameters()).device

    # ------------------------------------------------------------------
    # Prediction (inference — no gradients)
    # ------------------------------------------------------------------

    @torch.no_grad()
    def predict_transition(self, state: Tensor, action: Tensor) -> Tensor:
        """Predict the next state given *state* and *action*.

        Both tensors are concatenated along the last dimension and
        forwarded through the transition network.  Runs under
        ``torch.no_grad()``.
        """
        inp = torch.cat([state, action], dim=-1)
        return self._transition_net.forward(inp)

    @torch.no_grad()
    def predict_reward(self, state: Tensor, action: Tensor) -> Tensor:
        """Predict the immediate reward given *state* and *action*.

        Runs under ``torch.no_grad()``.
        """
        inp = torch.cat([state, action], dim=-1)
        return self._reward_net.forward(inp)

    @torch.no_grad()
    def predict(self, state: Tensor, action: Tensor) -> tuple[Tensor, Tensor]:
        """Predict both next-state and reward in one call.

        Returns ``(next_state, reward)`` — more efficient than calling
        :meth:`predict_transition` and :meth:`predict_reward` separately
        because the concatenation is computed only once.
        """
        inp = torch.cat([state, action], dim=-1)
        return (
            self._transition_net.forward(inp),
            self._reward_net.forward(inp),
        )

    # ------------------------------------------------------------------
    # Data collection
    # ------------------------------------------------------------------

    def record(self, transition: Transition) -> None:
        """Store a real transition for later model training.

        The internal ring-buffer is lazily allocated on the first call
        (shapes inferred from the transition).
        """
        if self._states is None:
            self._init_buffers(transition.state.shape, transition.action.shape)

        assert self._states is not None  # mypy narrowing
        assert self._actions is not None
        assert self._rewards is not None
        assert self._next_states is not None

        self._states[self._ptr] = transition.state
        self._actions[self._ptr] = transition.action
        self._rewards[self._ptr] = transition.reward
        self._next_states[self._ptr] = transition.next_state

        self._ptr = (self._ptr + 1) % self._capacity
        self._size = min(self._size + 1, self._capacity)

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def train_step(self, batch_size: int) -> dict[str, float]:
        """Train both sub-models on a random mini-batch.

        Samples *batch_size* transitions uniformly from the ring-buffer,
        then performs one gradient step for each network using MSE loss.

        Returns
        -------
        dict
            ``{"transition_loss": float, "reward_loss": float}``

        Raises
        ------
        ValueError
            If *batch_size* is non-positive or exceeds the current
            buffer occupancy.
        """
        if batch_size <= 0:
            msg = f"batch_size must be positive, got {batch_size}"
            raise ValueError(msg)
        if self._size < batch_size:
            msg = f"Not enough data to sample: have {self._size}, need {batch_size}"
            raise ValueError(msg)

        assert self._states is not None  # guaranteed by _size > 0
        assert self._actions is not None
        assert self._rewards is not None
        assert self._next_states is not None

        indices = np.random.choice(self._size, batch_size, replace=False)
        device = self._device

        states = torch.from_numpy(self._states[indices].copy()).float().to(device)
        actions = torch.from_numpy(self._actions[indices].copy()).float().to(device)
        rewards = (
            torch.from_numpy(self._rewards[indices].copy())
            .float()
            .unsqueeze(-1)
            .to(device)
        )
        next_states = (
            torch.from_numpy(self._next_states[indices].copy()).float().to(device)
        )

        inp = torch.cat([states, actions], dim=-1)

        # --- transition model -----------------------------------------
        predicted_next = self._transition_net.forward(inp)
        transition_loss = nn.functional.mse_loss(predicted_next, next_states)

        self._transition_optim.zero_grad()
        transition_loss.backward()
        self._transition_optim.step()

        # --- reward model ---------------------------------------------
        predicted_reward = self._reward_net.forward(inp)
        reward_loss = nn.functional.mse_loss(predicted_reward, rewards)

        self._reward_optim.zero_grad()
        reward_loss.backward()
        self._reward_optim.step()

        return {
            "transition_loss": transition_loss.item(),
            "reward_loss": reward_loss.item(),
        }

    # ------------------------------------------------------------------
    # Imagination (synthetic trajectory generation)
    # ------------------------------------------------------------------

    def imagine(
        self,
        initial_state: Tensor,
        policy_fn: Callable[[Tensor], Tensor],
        steps: int,
    ) -> list[Transition]:
        """Roll out a synthetic trajectory using the learned model.

        Starting from *initial_state*, repeatedly:

        1. Query *policy_fn* for an action.
        2. Predict the next state and reward.
        3. Wrap into a :class:`Transition` (always ``terminated=False``).

        Parameters
        ----------
        initial_state:
            Starting observation tensor.  May be batched ``(1, obs_dim)``
            or flat ``(obs_dim,)``.
        policy_fn:
            Callable that maps a state tensor to an action tensor.
        steps:
            Number of imagination steps to unroll.

        Returns
        -------
        list[Transition]
            A list of *steps* imagined transitions (numpy-backed VOs).

        Raises
        ------
        ValueError
            If *steps* is non-positive.
        """
        if steps <= 0:
            msg = f"steps must be positive, got {steps}"
            raise ValueError(msg)

        state = initial_state
        imagined: list[Transition] = []

        for _ in range(steps):
            action = policy_fn(state)

            # policy_fn may return ndarray — normalise to Tensor
            if isinstance(action, np.ndarray):
                action = torch.from_numpy(action).float().to(state.device)

            next_state, reward = self.predict(state, action)

            # Squeeze batch dim if present for the Transition VO
            s_np = self._to_numpy_1d(state)
            a_np = self._to_numpy_1d(action)
            ns_np = self._to_numpy_1d(next_state)
            r_val = float(
                reward.squeeze().item() if reward.numel() == 1 else reward.mean().item()
            )

            imagined.append(
                Transition(
                    state=s_np,
                    action=a_np,
                    reward=r_val,
                    next_state=ns_np,
                    terminated=False,
                )
            )

            state = next_state

        return imagined

    # ------------------------------------------------------------------
    # Serialisation (safetensors — no pickle)
    # ------------------------------------------------------------------

    def save_state(self) -> bytes:
        """Serialise network weights to bytes via *safetensors*.

        Optimizer momentum is **not** persisted (fresh Adam is
        recreated on :meth:`load_state`).  This avoids pickle entirely.
        """
        tensors: dict[str, Tensor] = {}
        for key, val in self._transition_net.state_dict().items():
            tensors[f"transition.{key}"] = val
        for key, val in self._reward_net.state_dict().items():
            tensors[f"reward.{key}"] = val
        return safetensors_torch.save(tensors)

    def load_state(self, state: bytes) -> None:
        """Restore network weights from *safetensors* bytes.

        Optimizers are recreated with fresh momentum.
        """
        all_tensors = safetensors_torch.load(state)

        transition_state = {
            k.removeprefix("transition."): v
            for k, v in all_tensors.items()
            if k.startswith("transition.")
        }
        reward_state = {
            k.removeprefix("reward."): v
            for k, v in all_tensors.items()
            if k.startswith("reward.")
        }

        self._transition_net.load_state_dict(transition_state)
        self._reward_net.load_state_dict(reward_state)

        # Recreate optimisers — fresh momentum, warm weights.
        self._transition_optim = optim.Adam(
            self._transition_net.parameters(), lr=self._lr
        )
        self._reward_optim = optim.Adam(self._reward_net.parameters(), lr=self._lr)

    # ------------------------------------------------------------------
    # Resource management
    # ------------------------------------------------------------------

    def close(self) -> None:
        """Release GPU memory and internal buffers."""
        self._states = None
        self._actions = None
        self._rewards = None
        self._next_states = None
        self._size = 0
        self._ptr = 0
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def __enter__(self) -> WorldModel:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    @property
    def buffer_size(self) -> int:
        """Number of transitions currently stored in the ring-buffer."""
        return self._size

    @property
    def buffer_capacity(self) -> int:
        """Maximum capacity of the ring-buffer."""
        return self._capacity

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _to_numpy_1d(tensor: Tensor) -> NDArray[np.floating[Any]]:
        """Detach, move to CPU, squeeze batch dim, return numpy."""
        arr = tensor.detach().cpu().numpy()
        if arr.ndim > 1 and arr.shape[0] == 1:
            arr = arr.squeeze(0)
        return arr

    def __repr__(self) -> str:
        return (
            f"WorldModel("
            f"transition={self._transition_net!r}, "
            f"reward={self._reward_net!r}, "
            f"lr={self._lr}, "
            f"buffer={self._size}/{self._capacity})"
        )
