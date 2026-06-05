from __future__ import annotations

from typing import Any

import gymnasium as gym

from droprl.envs.base import BaseEnv


class RllibGymWrapper(gym.Env):
    """Wrap a framework-native BaseEnv as a Gymnasium Env for RLlib."""

    metadata = {"render_modes": []}

    def __init__(self, env: BaseEnv):
        self._env = env
        self.observation_space = env.observation_space
        self.action_space = env.action_space

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        return self._env.reset(seed=seed, options=options)

    def step(self, action: Any):
        res = self._env.step(action)
        return res.observation, res.reward, res.terminated, res.truncated, res.info

    def render(self):
        return self._env.render()

    def close(self):
        return self._env.close()
