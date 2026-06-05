"""Gymnasium Pendulum — continuous control example for SAC."""

from __future__ import annotations

from typing import Any

import gymnasium as gym

ENV_ID = "pendulum-v0"


def make_env(config: dict[str, Any] | None = None) -> gym.Env:
    cfg = config or {}
    gym_id = str(cfg.get("gym_id", "Pendulum-v1"))
    render_mode = cfg.get("render_mode")
    if render_mode:
        return gym.make(gym_id, render_mode=render_mode)
    return gym.make(gym_id)
