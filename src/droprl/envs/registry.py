from __future__ import annotations

import importlib.util
import logging
import sys
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Any

import gymnasium as gym

from droprl.config import abs_path_str
from droprl.envs.base import BaseEnv, GymAdapterEnv

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class EnvSpec:
    name: str
    env_dir: Path
    entrypoint: Path


def _repo_root() -> Path:
    # src/droprl/envs/registry.py -> droprl -> src -> repo root
    return Path(__file__).resolve().parents[3]


def envs_root() -> Path:
    return _repo_root() / "envs"


def list_envs() -> list[str]:
    root = envs_root()
    if not root.is_dir():
        return []
    names: list[str] = []
    for p in sorted(root.iterdir()):
        if not p.is_dir():
            continue
        if p.name.startswith("_") or p.name.startswith("."):
            continue
        if (p / "env.py").is_file() or (p / "__init__.py").is_file():
            names.append(p.name)
    return names


def _load_module_from_path(module_name: str, path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location(module_name, abs_path_str(path))
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot import {module_name} from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def _resolve_env_entrypoint(name: str) -> EnvSpec:
    d = envs_root() / name
    if not d.is_dir():
        raise FileNotFoundError(f"Unknown env '{name}'. Expected {d} to exist.")
    if (d / "env.py").is_file():
        return EnvSpec(name=name, env_dir=d, entrypoint=d / "env.py")
    if (d / "__init__.py").is_file():
        return EnvSpec(name=name, env_dir=d, entrypoint=d / "__init__.py")
    raise FileNotFoundError(f"Env '{name}' must provide env.py or __init__.py under {d}.")


def load_env(name: str, config: dict[str, Any] | None = None) -> BaseEnv:
    """Load `envs/<name>/env.py` and call `make_env(config)`."""
    spec = _resolve_env_entrypoint(name)
    module = _load_module_from_path(f"droprl_user_env_{name}", spec.entrypoint)
    if not hasattr(module, "make_env"):
        raise AttributeError(f"{spec.entrypoint} must define make_env(config).")

    make_env = module.make_env
    if not callable(make_env):
        raise TypeError(f"make_env in {spec.entrypoint} must be callable.")

    env_obj = make_env(config or {})
    if isinstance(env_obj, BaseEnv):
        return env_obj
    if isinstance(env_obj, gym.Env) or _looks_gym_like(env_obj):
        return GymAdapterEnv(env_obj)  # type: ignore[arg-type]
    raise TypeError(
        "make_env(config) must return a BaseEnv or a Gymnasium-like env "
        f"(got {type(env_obj).__name__})."
    )


def _looks_gym_like(obj: Any) -> bool:
    required = ("reset", "step", "observation_space", "action_space")
    return all(hasattr(obj, name) for name in required)


def load_env_factory(name: str, config: dict[str, Any] | None = None) -> Callable[[], BaseEnv]:
    """Return a zero-arg factory usable by frameworks that create env instances lazily."""
    cfg = dict(config or {})
    return lambda: load_env(name, cfg)
