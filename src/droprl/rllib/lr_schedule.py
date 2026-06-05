from __future__ import annotations

import json
import logging
from collections import deque
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)

LR_STATE_FILE = "dynamic_lr.json"


@dataclass
class DynamicLRState:
    current_lr: float
    peak_reward_mean: float
    last_adjust_iter: int
    reward_history: list[float] | None = None


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


class DynamicLRController:
    """Scale-free LR from relative reward trend over a rolling window."""

    def __init__(
        self,
        *,
        initial_lr: float,
        min_lr: float,
        max_lr: float,
        enabled: bool = True,
        window_iters: int = 300,
        plateau_rel_improvement: float = 0.015,
        breakthrough_rel_improvement: float = 0.08,
        min_reward_scale: float = 1.0,
        decay_factor: float = 0.75,
        boost_factor: float = 1.15,
        cooldown_iters: int = 250,
    ) -> None:
        self.initial_lr = float(initial_lr)
        self.min_lr = float(min_lr)
        self.max_lr = float(max_lr)
        self.enabled = bool(enabled)
        self.window_iters = max(2, int(window_iters))
        self.plateau_rel_improvement = float(plateau_rel_improvement)
        self.breakthrough_rel_improvement = float(breakthrough_rel_improvement)
        self.min_reward_scale = float(min_reward_scale)
        self.decay_factor = float(decay_factor)
        self.boost_factor = float(boost_factor)
        self.cooldown_iters = int(cooldown_iters)

        self.current_lr = self.initial_lr
        self.peak_reward_mean = float("-inf")
        self._last_adjust_iter = -(10**9)
        self._history: deque[float] = deque(maxlen=self.window_iters)

    @property
    def lr(self) -> float:
        return self.current_lr

    def observe(self, iteration: int, reward: float) -> float:
        reward = float(reward)
        self.peak_reward_mean = max(self.peak_reward_mean, reward)
        self._history.append(reward)

        if not self.enabled or len(self._history) < self.window_iters:
            return self.current_lr
        if iteration - self._last_adjust_iter < self.cooldown_iters:
            return self.current_lr

        window = list(self._history)
        delta = window[-1] - window[0]
        window_mean = sum(window) / len(window)
        scale = max(abs(window_mean), self.min_reward_scale)
        rel_pct = 100.0 * delta / scale
        prev_lr = self.current_lr

        if (delta / scale) >= self.breakthrough_rel_improvement:
            self.current_lr = min(self.current_lr * self.boost_factor, self.max_lr)
            if self.current_lr > prev_lr * 1.01:
                self._last_adjust_iter = iteration
                log.info(
                    "dynamic_lr boost iter=%d delta=%+.3g mean=%.3g rel=%+.1f%% lr %.2e -> %.2e",
                    iteration,
                    delta,
                    window_mean,
                    rel_pct,
                    prev_lr,
                    self.current_lr,
                )
        elif (delta / scale) < self.plateau_rel_improvement:
            self.current_lr = max(self.current_lr * self.decay_factor, self.min_lr)
            if self.current_lr < prev_lr * 0.99:
                self._last_adjust_iter = iteration
                log.info(
                    "dynamic_lr decay iter=%d delta=%+.3g mean=%.3g rel=%+.1f%% lr %.2e -> %.2e",
                    iteration,
                    delta,
                    window_mean,
                    rel_pct,
                    prev_lr,
                    self.current_lr,
                )

        return self.current_lr

    def load_state(self, path: Path) -> bool:
        if not path.is_file():
            return False
        try:
            data = json.loads(path.read_text())
            peak = data.get("peak_reward_mean", data.get("best_reward", float("-inf")))
            state = DynamicLRState(
                current_lr=float(data["current_lr"]),
                peak_reward_mean=float(peak),
                last_adjust_iter=int(data["last_adjust_iter"]),
                reward_history=data.get("reward_history"),
            )
        except (json.JSONDecodeError, OSError, KeyError, TypeError, ValueError) as exc:
            log.warning("Ignoring invalid LR state %s: %s", path, exc)
            return False

        self.current_lr = state.current_lr
        self.peak_reward_mean = state.peak_reward_mean
        self._last_adjust_iter = state.last_adjust_iter
        if state.reward_history:
            self._history = deque(state.reward_history, maxlen=self.window_iters)
        log.info(
            "Restored dynamic_lr lr=%.2e peak_mean=%.3g history=%d/%d",
            self.current_lr,
            self.peak_reward_mean,
            len(self._history),
            self.window_iters,
        )
        return True

    def save_state(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        state = DynamicLRState(
            current_lr=self.current_lr,
            peak_reward_mean=self.peak_reward_mean,
            last_adjust_iter=self._last_adjust_iter,
            reward_history=list(self._history),
        )
        path.write_text(json.dumps(asdict(state), indent=2))


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


def build_dynamic_lr_controller(
    run_cfg: dict[str, Any], training_cfg: dict[str, Any]
) -> DynamicLRController:
    cfg = run_cfg.get("dynamic_lr", {})
    training_lr = float(training_cfg.get("lr", 1.0e-3))
    initial, min_lr, max_lr = _resolve_lr_bounds(cfg, training_lr)

    return DynamicLRController(
        initial_lr=initial,
        min_lr=min_lr,
        max_lr=max_lr,
        enabled=bool(cfg.get("enabled", False)),
        window_iters=int(cfg.get("window_iters", 300)),
        plateau_rel_improvement=float(cfg.get("plateau_rel_improvement", 0.015)),
        breakthrough_rel_improvement=float(cfg.get("breakthrough_rel_improvement", 0.08)),
        min_reward_scale=float(cfg.get("min_reward_scale", 1.0)),
        decay_factor=float(cfg.get("decay_factor", 0.75)),
        boost_factor=float(cfg.get("boost_factor", 1.15)),
        cooldown_iters=int(cfg.get("cooldown_iters", 250)),
    )


def lr_state_path(run_dir: Path) -> Path:
    return run_dir / LR_STATE_FILE
