from __future__ import annotations

import logging
from typing import Any

import gymnasium as gym
from ray.tune.registry import register_env

from droprl.envs.base import BaseEnv
from droprl.envs.registry import list_envs
from droprl.rllib.env_wrapper import RllibGymWrapper

log = logging.getLogger(__name__)


def _maybe_env_id_from_module(env_dir_name: str) -> str:
    # For RLlib, use stable ids. Default: "<folder>-v0".
    return f"{env_dir_name}-v0"


def register_rllib_envs() -> dict[str, str]:
    """Register all discovered `envs/<name>` with Ray under RLlib env ids.

    Returns a mapping: { env_folder_name: rllib_env_id }.
    """
    from droprl.envs.registry import _load_module_from_path, _resolve_env_entrypoint

    mapping: dict[str, str] = {}
    for name in list_envs():
        spec = _resolve_env_entrypoint(name)
        module = _load_module_from_path(f"droprl_env_meta_{name}", spec.entrypoint)
        env_id = getattr(module, "ENV_ID", None) or _maybe_env_id_from_module(name)

        def _factory(env_config: dict[str, Any], *, _name: str = name):
            # Import here to avoid Ray worker serialization issues.
            from droprl.envs.base import GymAdapterEnv
            from droprl.envs.registry import _load_module_from_path, _resolve_env_entrypoint

            spec2 = _resolve_env_entrypoint(_name)
            module2 = _load_module_from_path(f"droprl_user_env_{_name}", spec2.entrypoint)
            env_obj = module2.make_env(env_config or {})
            if isinstance(env_obj, gym.Env):
                return env_obj
            if isinstance(env_obj, BaseEnv):
                return RllibGymWrapper(env_obj)
            # If it's gym-like but not a subclass, wrap as BaseEnv first.
            return RllibGymWrapper(GymAdapterEnv(env_obj))

        register_env(env_id, _factory)
        mapping[name] = env_id
        log.info("Registered env '%s' as RLlib id '%s'", name, env_id)
    return mapping
