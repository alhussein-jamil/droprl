from __future__ import annotations

import json
from pathlib import Path
from typing import Any


class FixedLRScheduler:
    """Constant learning rate from ``training.lr`` (or ``initial_lr`` override)."""

    name = "fixed"

    def __init__(self, *, initial_lr: float) -> None:
        self.initial_lr = float(initial_lr)
        self.current_lr = self.initial_lr

    @property
    def lr(self) -> float:
        return self.current_lr

    def observe(self, iteration: int, reward: float) -> float:
        del iteration, reward
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


def build_fixed_scheduler(cfg: dict[str, Any], training_lr: float) -> FixedLRScheduler:
    initial = float(cfg.get("initial_lr", training_lr))
    return FixedLRScheduler(initial_lr=initial)
