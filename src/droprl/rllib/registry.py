from __future__ import annotations

import logging
from typing import Any

import gymnasium as gym
from ray.tune.registry import register_env

from droprl.envs.base import BaseEnv
from droprl.envs.registry import list_envs
from droprl.rllib.env_wrapper import RllibGymWrapper

log = logging.getLogger(__name__)


def _default_env_id(env_dir_name: str) -> str:
    return f"{env_dir_name}-v0"


def _register_one(name: str) -> str:
    """Register a single task with RLlib without importing its env module."""
    env_id = _default_env_id(name)

    def _factory(env_config: dict[str, Any], *, _name: str = name):
        from droprl.envs.base import GymAdapterEnv
        from droprl.envs.registry import _load_module_from_path, _resolve_env_entrypoint

        spec = _resolve_env_entrypoint(_name)
        module = _load_module_from_path(f"droprl_user_env_{_name}", spec.entrypoint)
        env_obj = module.make_env(env_config or {})
        if isinstance(env_obj, gym.Env):
            return env_obj
        if isinstance(env_obj, BaseEnv):
            return RllibGymWrapper(env_obj)
        return RllibGymWrapper(GymAdapterEnv(env_obj))

    register_env(env_id, _factory)
    log.info("Registered env '%s' as RLlib id '%s'", name, env_id)
    return env_id


def register_rllib_envs(*, tasks: list[str] | None = None) -> dict[str, str]:
    """Register envs with Ray. Only imports task code when an env instance is created.

    Pass ``tasks=['mock']`` to avoid loading optional deps from other env folders.
    """
    names = list(tasks) if tasks is not None else list_envs()
    mapping: dict[str, str] = {}
    for name in names:
        mapping[name] = _register_one(name)
    return mapping
