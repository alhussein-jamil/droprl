from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from droprl.rllib.lr_schedule.base import _resolve_lr_bounds


class LinearLRScheduler:
    """Linear decay from ``initial_lr`` down to ``min_lr`` over ``total_iters``."""

    name = "linear"

    def __init__(
        self,
        *,
        initial_lr: float,
        min_lr: float,
        total_iters: int,
        warmup_iters: int = 0,
    ) -> None:
        self.initial_lr = float(initial_lr)
        self.min_lr = float(min_lr)
        self.total_iters = max(1, int(total_iters))
        self.warmup_iters = max(0, int(warmup_iters))
        self.current_lr = self.initial_lr

    @property
    def lr(self) -> float:
        return self.current_lr

    def _lr_for_iteration(self, iteration: int) -> float:
        if iteration < self.warmup_iters:
            if self.warmup_iters <= 0:
                return self.initial_lr
            progress = iteration / self.warmup_iters
            return self.min_lr + progress * (self.initial_lr - self.min_lr)

        if self.total_iters <= self.warmup_iters:
            return self.min_lr

        decay_iters = self.total_iters - self.warmup_iters
        t = min(max(iteration - self.warmup_iters, 0), decay_iters)
        progress = t / decay_iters
        return self.initial_lr - progress * (self.initial_lr - self.min_lr)

    def observe(self, iteration: int, reward: float) -> float:
        del reward
        self.current_lr = self._lr_for_iteration(iteration)
        return self.current_lr

    def load_state(self, path: Path) -> bool:
        if not path.is_file():
            return False
        try:
            data = json.loads(path.read_text())
            if data.get("type", self.name) != self.name:
                return False
            self.current_lr = float(data.get("current_lr", self.initial_lr))
        except (json.JSONDecodeError, OSError, KeyError, TypeError, ValueError):
            return False
        return True

    def save_state(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(
                {"type": self.name, "current_lr": self.current_lr},
                indent=2,
            )
        )


def build_linear_scheduler(
    cfg: dict[str, Any],
    training_lr: float,
    *,
    total_iters: int,
) -> LinearLRScheduler:
    initial, min_lr, _ = _resolve_lr_bounds(cfg, training_lr)
    return LinearLRScheduler(
        initial_lr=initial,
        min_lr=min_lr,
        total_iters=total_iters,
        warmup_iters=int(cfg.get("warmup_iters", 0)),
    )
