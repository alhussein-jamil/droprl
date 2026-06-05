"""Gymnasium CartPole — standard env integration smoke test."""

from __future__ import annotations

from typing import Any

import gymnasium as gym

ENV_ID = "cartpole-v0"


def make_env(config: dict[str, Any] | None = None) -> gym.Env:
    cfg = config or {}
    gym_id = str(cfg.get("gym_id", "CartPole-v1"))
    render_mode = cfg.get("render_mode")
    if render_mode:
        return gym.make(gym_id, render_mode=render_mode)
    return gym.make(gym_id)
