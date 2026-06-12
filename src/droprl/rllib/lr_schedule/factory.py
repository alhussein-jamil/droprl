from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from droprl.rllib.lr_schedule.base import LRScheduler, legacy_lr_state_path, lr_state_path
from droprl.rllib.lr_schedule.cosine import build_cosine_scheduler
from droprl.rllib.lr_schedule.dynamic import build_dynamic_scheduler
from droprl.rllib.lr_schedule.fixed import build_fixed_scheduler
from droprl.rllib.lr_schedule.linear import build_linear_scheduler

log = logging.getLogger(__name__)

SCHEDULER_TYPES = ("fixed", "dynamic", "cosine", "linear")


def list_schedulers() -> tuple[str, ...]:
    return SCHEDULER_TYPES


def resolve_scheduler_type(run_cfg: dict[str, Any]) -> str:
    """Resolve scheduler type from ``run.lr_schedule`` (with ``dynamic_lr`` fallback)."""
    lr_cfg = run_cfg.get("lr_schedule")
    if isinstance(lr_cfg, str):
        return lr_cfg
    if isinstance(lr_cfg, dict):
        return str(lr_cfg.get("type", "fixed"))

    dynamic_cfg = run_cfg.get("dynamic_lr", {})
    if bool(dynamic_cfg.get("enabled", False)):
        return "dynamic"
    return "fixed"


def resolve_scheduler_params(run_cfg: dict[str, Any], sched_type: str) -> dict[str, Any]:
    lr_cfg = run_cfg.get("lr_schedule")
    if isinstance(lr_cfg, dict):
        params = dict(lr_cfg)
        params.pop("type", None)
        return params

    if sched_type == "dynamic":
        return dict(run_cfg.get("dynamic_lr", {}))
    return {}


def build_lr_scheduler(
    run_cfg: dict[str, Any],
    training_cfg: dict[str, Any],
    *,
    total_iters: int,
) -> LRScheduler:
    sched_type = resolve_scheduler_type(run_cfg)
    if sched_type not in SCHEDULER_TYPES:
        raise ValueError(
            f"Unknown lr_schedule type '{sched_type}'. Choose one of: {', '.join(SCHEDULER_TYPES)}"
        )

    params = resolve_scheduler_params(run_cfg, sched_type)
    training_lr = float(training_cfg.get("lr", 1.0e-3))

    if sched_type == "fixed":
        return build_fixed_scheduler(params, training_lr)
    if sched_type == "dynamic":
        return build_dynamic_scheduler(params, training_lr)
    if sched_type == "cosine":
        return build_cosine_scheduler(params, training_lr, total_iters=total_iters)
    return build_linear_scheduler(params, training_lr, total_iters=total_iters)


def load_lr_state(scheduler: LRScheduler, run_dir: Path) -> bool:
    path = lr_state_path(run_dir)
    if scheduler.load_state(path):
        return True

    legacy = legacy_lr_state_path(run_dir)
    if scheduler.name == "dynamic" and scheduler.load_state(legacy):
        log.info("Loaded legacy LR state from %s", legacy)
        return True
    return False
