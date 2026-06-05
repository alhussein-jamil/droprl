from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Protocol

import gymnasium as gym


@dataclass(frozen=True)
class StepResult:
    observation: Any
    reward: float
    terminated: bool
    truncated: bool
    info: dict[str, Any]


class BaseEnv(ABC):
    """Framework-native environment interface.

    The framework stays minimal. If your env is naturally a Gymnasium env, prefer
    implementing Gymnasium directly and using `GymAdapterEnv`.
    """

    @abstractmethod
    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[Any, dict[str, Any]]:
        raise NotImplementedError

    @abstractmethod
    def step(self, action: Any) -> StepResult:
        raise NotImplementedError

    @property
    @abstractmethod
    def observation_space(self) -> gym.Space:
        raise NotImplementedError

    @property
    @abstractmethod
    def action_space(self) -> gym.Space:
        raise NotImplementedError

    def render(self) -> Any:
        return None

    def close(self) -> None:
        return None


class GymLike(Protocol):
    observation_space: gym.Space
    action_space: gym.Space

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[Any, dict[str, Any]]: ...
    def step(self, action: Any) -> tuple[Any, float, bool, bool, dict[str, Any]]: ...
    def render(self) -> Any: ...
    def close(self) -> None: ...


class GymAdapterEnv(BaseEnv):
    def __init__(self, env: GymLike):
        self._env = env

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[Any, dict[str, Any]]:
        return self._env.reset(seed=seed, options=options)

    def step(self, action: Any) -> StepResult:
        obs, reward, terminated, truncated, info = self._env.step(action)
        return StepResult(
            observation=obs,
            reward=float(reward),
            terminated=bool(terminated),
            truncated=bool(truncated),
            info=dict(info),
        )

    @property
    def observation_space(self) -> gym.Space:
        return self._env.observation_space

    @property
    def action_space(self) -> gym.Space:
        return self._env.action_space

    def render(self) -> Any:
        return self._env.render()

    def close(self) -> None:
        self._env.close()
