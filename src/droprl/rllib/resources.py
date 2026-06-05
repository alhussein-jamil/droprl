from __future__ import annotations

import multiprocessing
from copy import deepcopy
from typing import Any

_AUTO = frozenset({None, "auto", "???"})


def _cpu_count() -> int:
    return max(1, multiprocessing.cpu_count())


def resolve_ray_config(ray_cfg: dict[str, Any] | None) -> dict[str, Any]:
    cfg = deepcopy(ray_cfg or {})
    if cfg.get("num_cpus") in _AUTO:
        cfg["num_cpus"] = _cpu_count()
    return cfg


def resolve_env_runners(env_runners: dict[str, Any] | None) -> dict[str, Any]:
    cfg = deepcopy(env_runners or {})
    if cfg.get("num_env_runners") in _AUTO:
        cfg["num_env_runners"] = _cpu_count()
    if cfg.get("num_gpus_per_env_runner") in _AUTO:
        cfg["num_gpus_per_env_runner"] = 0
    return cfg


def resolve_training_section(training: dict[str, Any]) -> dict[str, Any]:
    section = deepcopy(training)
    section["env_runners"] = resolve_env_runners(section.get("env_runners"))
    return section
