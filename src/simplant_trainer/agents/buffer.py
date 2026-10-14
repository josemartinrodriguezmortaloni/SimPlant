import torch
import numpy as np
from simplant_trainer.value_objects.spaces import _validate_shape
from simplant_trainer.value_objects.transition import Transition

class BoundedReplayBuffer:
    def __init__(self, capacity: int, obs_shape: tuple, action_shape: tuple):
        self._ptr = 0
        self._size = 0
        if capacity <= 0:
            raise ValueError("Capacity must be positive")
        self.capacity = capacity
        self.obs_shape = _validate_shape(obs_shape)
        self.action_shape = _validate_shape(action_shape)
        self.obs_arr = np.zeros((self.capacity, *obs_shape), dtype=np.float64)
        self.next_states_arr = np.zeros((self.capacity, *obs_shape), dtype=np.float64)
        self.actions_arr = np.zeros((self.capacity, *action_shape), dtype=np.float64)
        self.rewards_arr = np.zeros((self.capacity, ), dtype=np.float64)
        self.dones_arr = np.zeros((self.capacity, ), dtype=np.float64)

    def add(self, transition: Transition):
        self.obs_arr[self._ptr] = transition.state
        self.next_states_arr[self._ptr] = transition.next_state
        self.actions_arr[self._ptr] = transition.action
        self.rewards_arr[self._ptr] = transition.reward
        self.dones_arr[self._ptr] = transition.terminated

        self._ptr = (self._ptr + 1) %  self.capacity
        self._size = min(self._size + 1, self.capacity)


    def sample(self, batch_size: int) -> dict:
        if batch_size > self._size:
            raise ValueError(f"batch_size ({batch_size}) exceeds buffer size ({self._size})")
        indexes = np.random.choice(self._size, batch_size, replace=False)
        return {
            "obs": torch.from_numpy(self.obs_arr[indexes]).float(),
            "next_state": torch.from_numpy(self.next_states_arr[indexes]).float(),
            "actions": torch.from_numpy(self.actions_arr[indexes]).float(),
            "rewards": torch.from_numpy(self.rewards_arr[indexes]).float(),
            "dones": torch.from_numpy(self.dones_arr[indexes]).float(),
        }

    def __len__(self) -> int:
        return self._size
 
