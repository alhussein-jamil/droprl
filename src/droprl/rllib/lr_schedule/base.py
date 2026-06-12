from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

log = logging.getLogger(__name__)

LR_STATE_FILE = "lr_schedule.json"
LEGACY_LR_STATE_FILE = "dynamic_lr.json"


@runtime_checkable
class LRScheduler(Protocol):
    """Iteration-driven learning-rate schedule used by ``scripts/train.py``."""

    @property
    def name(self) -> str: ...

    @property
    def lr(self) -> float: ...

    def observe(self, iteration: int, reward: float) -> float: ...

    def load_state(self, path: Path) -> bool: ...

    def save_state(self, path: Path) -> None: ...


def apply_lr_to_trainer(trainer: Any, lr: float) -> None:
    """Push LR into every policy optimizer (local + remote rollout workers)."""

    def _update(worker) -> None:
        policy = worker.get_policy()
        if policy is None:
            return
        policy.config["lr"] = lr
        if hasattr(policy, "cur_lr"):
            policy.cur_lr = lr
        for opt in getattr(policy, "_optimizers", ()):
            for group in opt.param_groups:
                group["lr"] = lr

    trainer.env_runner_group.foreach_env_runner(_update, local_env_runner=True)


def lr_state_path(run_dir: Path) -> Path:
    return run_dir / LR_STATE_FILE


def legacy_lr_state_path(run_dir: Path) -> Path:
    return run_dir / LEGACY_LR_STATE_FILE


def _resolve_lr_bounds(cfg: dict[str, Any], training_lr: float) -> tuple[float, float, float]:
    initial = float(cfg.get("initial_lr", training_lr))

    min_lr = cfg.get("min_lr")
    if min_lr is None:
        min_lr = initial * float(cfg.get("min_lr_ratio", 0.01))
    min_lr = float(min_lr)

    max_lr = cfg.get("max_lr")
    if max_lr is None:
        max_lr = initial * float(cfg.get("max_lr_ratio", 1.0))
    max_lr = float(max_lr)

    return initial, min_lr, max(max_lr, min_lr)


def _float_cfg(cfg: dict[str, Any], key: str, default: float) -> float:
    return float(cfg.get(key, default))


def _int_cfg(cfg: dict[str, Any], key: str, default: int) -> int:
    return int(cfg.get(key, default))
