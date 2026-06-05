from __future__ import annotations

import gymnasium as gym

from droprl.config import load_experiment
from droprl.envs.registry import load_env


def test_load_cartpole_env() -> None:
    env = load_env("cartpole", {"gym_id": "CartPole-v1"})
    obs, _ = env.reset()
    assert obs is not None
    result = env.step(env.action_space.sample())
    assert result.reward is not None


def test_cartpole_is_gymnasium() -> None:
    env = load_env("cartpole", {})
    inner = getattr(env, "_env", env)
    assert isinstance(inner, gym.Env)


def test_load_cartpole_experiment() -> None:
    cfg = load_experiment(task="cartpole", train="Cartpole")
    assert cfg["env"]["config"]["gym_id"] == "CartPole-v1"
    assert cfg["training"]["environment"]["normalize_actions"] is False
