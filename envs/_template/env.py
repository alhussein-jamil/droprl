"""Task entrypoint — copy this folder to envs/<your_task>/ and implement."""

from __future__ import annotations

from typing import Any

import gymnasium as gym
import numpy as np

ENV_ID = "my_task-v0"


class MyTaskEnv(gym.Env):
    """Minimal Gymnasium env. DropRL wraps this for RLlib automatically."""

    metadata = {"render_modes": []}

    def __init__(self, config: dict[str, Any]):
        super().__init__()
        self._cfg = config
        self.observation_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)
        self.action_space = gym.spaces.Discrete(2)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        return np.zeros(1, dtype=np.float32), {}

    def step(self, action):
        obs = np.zeros(1, dtype=np.float32)
        return obs, 0.0, False, False, {}


def make_env(config: dict[str, Any] | None = None) -> gym.Env:
    return MyTaskEnv(config or {})
