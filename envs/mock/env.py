from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import gymnasium as gym
import numpy as np

from droprl.envs.base import BaseEnv, StepResult


@dataclass(frozen=True)
class MockEnvConfig:
    horizon: int = 50
    target: float = 0.0
    start: float = 5.0
    step_size: float = 1.0
    render_width: int = 320
    render_height: int = 240
    render_x_range: float = 20.0


class MockCounterEnv(BaseEnv):
    """1D counter env: move left/right toward a target."""

    metadata = {"render_modes": ["rgb_array"]}

    def __init__(self, config: MockEnvConfig):
        self._cfg = config
        self._t = 0
        self._pos = float(config.start)
        self.render_mode = "rgb_array"

        self._observation_space = gym.spaces.Box(
            low=np.array([-1000.0], dtype=np.float32),
            high=np.array([1000.0], dtype=np.float32),
            dtype=np.float32,
        )
        self._action_space = gym.spaces.Discrete(2)

    @property
    def observation_space(self) -> gym.Space:
        return self._observation_space

    @property
    def action_space(self) -> gym.Space:
        return self._action_space

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[np.ndarray, dict[str, Any]]:
        if seed is not None:
            np.random.seed(seed)
        self._t = 0
        self._pos = float(self._cfg.start)
        return self._obs(), {"pos": self._pos, "target": float(self._cfg.target)}

    def step(self, action: int) -> StepResult:
        delta = -self._cfg.step_size if int(action) == 0 else self._cfg.step_size
        self._pos += delta
        self._t += 1

        reward = -abs(self._pos - float(self._cfg.target))
        truncated = self._t >= int(self._cfg.horizon)
        info = {
            "pos": self._pos,
            "t": self._t,
            "target": float(self._cfg.target),
            "distance": abs(self._pos - float(self._cfg.target)),
        }
        return StepResult(
            observation=self._obs(),
            reward=float(reward),
            terminated=False,
            truncated=bool(truncated),
            info=info,
        )

    def render(self) -> np.ndarray:
        w = int(self._cfg.render_width)
        h = int(self._cfg.render_height)
        img = np.full((h, w, 3), 32, dtype=np.uint8)

        mid_y = h // 2
        img[mid_y - 1 : mid_y + 1, :] = [64, 64, 64]

        span = float(self._cfg.render_x_range)

        def _x(value: float) -> int:
            t = (float(value) + span) / (2.0 * span)
            return int(np.clip(t, 0.0, 1.0) * (w - 1))

        x_target = _x(self._cfg.target)
        x_agent = _x(self._pos)
        img[:, x_target : x_target + 2] = [180, 70, 70]

        radius = 6
        for dy in range(-radius, radius + 1):
            for dx in range(-radius, radius + 1):
                if dx * dx + dy * dy > radius * radius:
                    continue
                y, x = mid_y + dy, x_agent + dx
                if 0 <= y < h and 0 <= x < w:
                    img[y, x] = [70, 200, 90]

        return img

    def _obs(self) -> np.ndarray:
        return np.array([self._pos], dtype=np.float32)


ENV_ID = "mock-v0"


def make_env(config: dict[str, Any] | None = None) -> BaseEnv:
    cfg = config or {}
    env_cfg = MockEnvConfig(
        horizon=int(cfg.get("horizon", 50)),
        target=float(cfg.get("target", 0.0)),
        start=float(cfg.get("start", 5.0)),
        step_size=float(cfg.get("step_size", 1.0)),
        render_width=int(cfg.get("render_width", 320)),
        render_height=int(cfg.get("render_height", 240)),
        render_x_range=float(cfg.get("render_x_range", 20.0)),
    )
    return MockCounterEnv(env_cfg)
