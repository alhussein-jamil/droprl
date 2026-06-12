from __future__ import annotations

import json
import logging
from collections import deque
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from droprl.rllib.lr_schedule.base import (
    _float_cfg,
    _int_cfg,
    _resolve_lr_bounds,
)

log = logging.getLogger(__name__)


@dataclass
class DynamicLRState:
    current_lr: float
    peak_reward_mean: float
    last_adjust_iter: int
    reward_history: list[float]
    ema_fast: float | None = None
    ema_slow: float | None = None
    regression_iters: int = 0


class DynamicLRScheduler:
    """Reward-driven LR for on-policy RL.

    Regimes (evaluated each cooldown window once the history buffer is full):

    - **recovery**: below peak (rolling or all-time) with positive EMA trend → mild boost
    - **warm_restart**: sustained regression without recovery → LR floor at initial × ratio
    - **breakthrough**: near rolling peak with strong positive trend → boost
    - **plateau**: near rolling peak with flat trend → decay
    """

    name = "dynamic"

    def __init__(
        self,
        *,
        initial_lr: float,
        min_lr: float,
        max_lr: float,
        enabled: bool = True,
        window_iters: int = 300,
        trend_fast_iters: int = 50,
        trend_slow_iters: int | None = None,
        plateau_rel_improvement: float = 0.015,
        breakthrough_rel_improvement: float = 0.08,
        min_reward_scale: float = 1.0,
        peak_near_ratio: float = 0.90,
        regression_drop_ratio: float = 0.15,
        recovery_min_trend: float = 0.02,
        decay_factor: float = 0.75,
        boost_factor: float = 1.15,
        recovery_boost_factor: float = 1.10,
        restart_lr_ratio: float = 0.5,
        restart_after_iters: int = 400,
        cooldown_iters: int = 250,
    ) -> None:
        self.initial_lr = float(initial_lr)
        self.min_lr = float(min_lr)
        self.max_lr = float(max_lr)
        self.enabled = bool(enabled)
        self.window_iters = max(2, int(window_iters))
        self.trend_fast_iters = max(2, int(trend_fast_iters))
        slow = self.window_iters if trend_slow_iters is None else int(trend_slow_iters)
        self.trend_slow_iters = max(self.trend_fast_iters + 1, slow)
        self.plateau_rel_improvement = float(plateau_rel_improvement)
        self.breakthrough_rel_improvement = float(breakthrough_rel_improvement)
        self.min_reward_scale = float(min_reward_scale)
        self.peak_near_ratio = float(peak_near_ratio)
        self.regression_drop_ratio = float(regression_drop_ratio)
        self.recovery_min_trend = float(recovery_min_trend)
        self.decay_factor = float(decay_factor)
        self.boost_factor = float(boost_factor)
        self.recovery_boost_factor = float(recovery_boost_factor)
        self.restart_lr_ratio = float(restart_lr_ratio)
        self.restart_after_iters = max(1, int(restart_after_iters))
        self.cooldown_iters = int(cooldown_iters)

        self._fast_alpha = 2.0 / (self.trend_fast_iters + 1)
        self._slow_alpha = 2.0 / (self.trend_slow_iters + 1)

        self.current_lr = self.initial_lr
        self.peak_reward_mean = float("-inf")
        self._last_adjust_iter = -(10**9)
        self._regression_iters = 0
        self._history: deque[float] = deque(maxlen=self.window_iters)
        self._ema_fast = float("-inf")
        self._ema_slow = float("-inf")

    @property
    def lr(self) -> float:
        return self.current_lr

    def _update_emas(self, reward: float) -> None:
        if self._ema_fast == float("-inf"):
            self._ema_fast = reward
            self._ema_slow = reward
            return
        self._ema_fast = self._fast_alpha * reward + (1.0 - self._fast_alpha) * self._ema_fast
        self._ema_slow = self._slow_alpha * reward + (1.0 - self._slow_alpha) * self._ema_slow

    def _rebuild_emas_from_history(self, history: list[float]) -> None:
        self._ema_fast = float("-inf")
        self._ema_slow = float("-inf")
        for reward in history:
            self._update_emas(float(reward))

    def _trend_metrics(self) -> tuple[float, float, float, float, float]:
        delta = self._ema_fast - self._ema_slow
        scale = max(abs(self._ema_slow), self.min_reward_scale)
        rel_pct = 100.0 * delta / scale
        recent_peak = max(self._history)
        recent_peak_ratio = self._ema_fast / max(recent_peak, self.min_reward_scale)
        all_time_peak_ratio = self._ema_fast / max(self.peak_reward_mean, self.min_reward_scale)
        return delta, scale, rel_pct, recent_peak_ratio, all_time_peak_ratio

    def _below_peak(self, recent_peak_ratio: float, all_time_peak_ratio: float) -> bool:
        threshold = 1.0 - self.regression_drop_ratio
        return recent_peak_ratio < threshold or all_time_peak_ratio < threshold

    def _in_recovery(
        self,
        rel: float,
        recent_peak_ratio: float,
        all_time_peak_ratio: float,
    ) -> bool:
        return self._below_peak(recent_peak_ratio, all_time_peak_ratio) and (
            rel >= self.recovery_min_trend
        )

    def _track_regression(
        self,
        rel: float,
        recent_peak_ratio: float,
        all_time_peak_ratio: float,
    ) -> None:
        if self._below_peak(recent_peak_ratio, all_time_peak_ratio) and (
            rel < self.recovery_min_trend
        ):
            self._regression_iters += 1
        else:
            self._regression_iters = 0

    def _apply_lr_change(
        self,
        *,
        iteration: int,
        prev_lr: float,
        reason: str,
        delta: float,
        rel_pct: float,
        recent_peak_ratio: float,
        all_time_peak_ratio: float,
    ) -> None:
        self._last_adjust_iter = iteration
        log.info(
            "lr_schedule[%s] %s iter=%d trend=%+.3g ema=%.3g peak=%.3g "
            "recent_ratio=%.2f all_time_ratio=%.2f regression_iters=%d rel=%+.1f%% "
            "lr %.2e -> %.2e",
            self.name,
            reason,
            iteration,
            delta,
            self._ema_fast,
            self.peak_reward_mean,
            recent_peak_ratio,
            all_time_peak_ratio,
            self._regression_iters,
            rel_pct,
            prev_lr,
            self.current_lr,
        )

    def _maybe_boost(
        self,
        *,
        iteration: int,
        prev_lr: float,
        factor: float,
        reason: str,
        delta: float,
        rel_pct: float,
        recent_peak_ratio: float,
        all_time_peak_ratio: float,
    ) -> bool:
        self.current_lr = min(self.current_lr * factor, self.max_lr)
        if self.current_lr <= prev_lr * 1.01:
            self.current_lr = prev_lr
            return False
        self._apply_lr_change(
            iteration=iteration,
            prev_lr=prev_lr,
            reason=reason,
            delta=delta,
            rel_pct=rel_pct,
            recent_peak_ratio=recent_peak_ratio,
            all_time_peak_ratio=all_time_peak_ratio,
        )
        return True

    def observe(self, iteration: int, reward: float) -> float:
        reward = float(reward)
        self.peak_reward_mean = max(self.peak_reward_mean, reward)
        self._history.append(reward)
        self._update_emas(reward)

        if not self.enabled or len(self._history) < self.window_iters:
            return self.current_lr
        if iteration - self._last_adjust_iter < self.cooldown_iters:
            return self.current_lr

        delta, scale, rel_pct, recent_peak_ratio, all_time_peak_ratio = self._trend_metrics()
        rel = delta / scale
        self._track_regression(rel, recent_peak_ratio, all_time_peak_ratio)
        prev_lr = self.current_lr

        if self._regression_iters >= self.restart_after_iters:
            target = min(self.initial_lr * self.restart_lr_ratio, self.max_lr)
            if self.current_lr < target * 0.99:
                self.current_lr = max(target, self.min_lr)
                self._regression_iters = 0
                self._apply_lr_change(
                    iteration=iteration,
                    prev_lr=prev_lr,
                    reason="warm_restart",
                    delta=delta,
                    rel_pct=rel_pct,
                    recent_peak_ratio=recent_peak_ratio,
                    all_time_peak_ratio=all_time_peak_ratio,
                )
            return self.current_lr

        if self._in_recovery(rel, recent_peak_ratio, all_time_peak_ratio):
            self._maybe_boost(
                iteration=iteration,
                prev_lr=prev_lr,
                factor=self.recovery_boost_factor,
                reason="recovery",
                delta=delta,
                rel_pct=rel_pct,
                recent_peak_ratio=recent_peak_ratio,
                all_time_peak_ratio=all_time_peak_ratio,
            )
            return self.current_lr

        if rel >= self.breakthrough_rel_improvement and (recent_peak_ratio >= self.peak_near_ratio):
            self._maybe_boost(
                iteration=iteration,
                prev_lr=prev_lr,
                factor=self.boost_factor,
                reason="breakthrough",
                delta=delta,
                rel_pct=rel_pct,
                recent_peak_ratio=recent_peak_ratio,
                all_time_peak_ratio=all_time_peak_ratio,
            )
            return self.current_lr

        if recent_peak_ratio >= self.peak_near_ratio and abs(rel) < self.plateau_rel_improvement:
            self.current_lr = max(self.current_lr * self.decay_factor, self.min_lr)
            if self.current_lr < prev_lr * 0.99:
                self._apply_lr_change(
                    iteration=iteration,
                    prev_lr=prev_lr,
                    reason="plateau",
                    delta=delta,
                    rel_pct=rel_pct,
                    recent_peak_ratio=recent_peak_ratio,
                    all_time_peak_ratio=all_time_peak_ratio,
                )

        return self.current_lr

    def load_state(self, path: Path) -> bool:
        if not path.is_file():
            return False
        try:
            data = json.loads(path.read_text())
            state = DynamicLRState(
                current_lr=float(data["current_lr"]),
                peak_reward_mean=float(data["peak_reward_mean"]),
                last_adjust_iter=int(data["last_adjust_iter"]),
                reward_history=list(data.get("reward_history") or []),
                ema_fast=_optional_float(data.get("ema_fast")),
                ema_slow=_optional_float(data.get("ema_slow")),
                regression_iters=int(data.get("regression_iters", 0)),
            )
        except (json.JSONDecodeError, OSError, KeyError, TypeError, ValueError) as exc:
            log.warning("Ignoring invalid LR state %s: %s", path, exc)
            return False

        self.current_lr = state.current_lr
        self.peak_reward_mean = state.peak_reward_mean
        self._last_adjust_iter = state.last_adjust_iter
        self._regression_iters = state.regression_iters
        if state.reward_history:
            self._history = deque(state.reward_history, maxlen=self.window_iters)
        if state.ema_fast is not None and state.ema_slow is not None:
            self._ema_fast = state.ema_fast
            self._ema_slow = state.ema_slow
        elif state.reward_history:
            self._rebuild_emas_from_history(state.reward_history)

        log.info(
            "Restored lr_schedule[%s] lr=%.2e peak_mean=%.3g ema=%.3g history=%d/%d "
            "regression_iters=%d",
            self.name,
            self.current_lr,
            self.peak_reward_mean,
            self._ema_fast,
            len(self._history),
            self.window_iters,
            self._regression_iters,
        )
        return True

    def save_state(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        state = DynamicLRState(
            current_lr=self.current_lr,
            peak_reward_mean=self.peak_reward_mean,
            last_adjust_iter=self._last_adjust_iter,
            reward_history=list(self._history),
            ema_fast=self._ema_fast if self._ema_fast != float("-inf") else None,
            ema_slow=self._ema_slow if self._ema_slow != float("-inf") else None,
            regression_iters=self._regression_iters,
        )
        payload = asdict(state)
        payload["type"] = self.name
        path.write_text(json.dumps(payload, indent=2))


def _optional_float(value: Any) -> float | None:
    if value is None:
        return None
    return float(value)


def build_dynamic_scheduler(cfg: dict[str, Any], training_lr: float) -> DynamicLRScheduler:
    initial, min_lr, max_lr = _resolve_lr_bounds(cfg, training_lr)

    return DynamicLRScheduler(
        initial_lr=initial,
        min_lr=min_lr,
        max_lr=max_lr,
        enabled=bool(cfg.get("enabled", True)),
        window_iters=_int_cfg(cfg, "window_iters", 300),
        trend_fast_iters=_int_cfg(cfg, "trend_fast_iters", 50),
        trend_slow_iters=cfg.get("trend_slow_iters"),
        plateau_rel_improvement=_float_cfg(cfg, "plateau_rel_improvement", 0.015),
        breakthrough_rel_improvement=_float_cfg(cfg, "breakthrough_rel_improvement", 0.08),
        min_reward_scale=_float_cfg(cfg, "min_reward_scale", 1.0),
        peak_near_ratio=_float_cfg(cfg, "peak_near_ratio", 0.90),
        regression_drop_ratio=_float_cfg(cfg, "regression_drop_ratio", 0.15),
        recovery_min_trend=_float_cfg(cfg, "recovery_min_trend", 0.02),
        decay_factor=_float_cfg(cfg, "decay_factor", 0.75),
        boost_factor=_float_cfg(cfg, "boost_factor", 1.15),
        recovery_boost_factor=_float_cfg(cfg, "recovery_boost_factor", 1.10),
        restart_lr_ratio=_float_cfg(cfg, "restart_lr_ratio", 0.5),
        restart_after_iters=_int_cfg(cfg, "restart_after_iters", 400),
        cooldown_iters=_int_cfg(cfg, "cooldown_iters", 250),
    )


# Backward-compatible alias used by existing tests and callers.
DynamicLRController = DynamicLRScheduler


def build_dynamic_lr_controller(
    run_cfg: dict[str, Any], training_cfg: dict[str, Any]
) -> DynamicLRScheduler:
    from droprl.rllib.lr_schedule.factory import resolve_scheduler_params

    cfg = resolve_scheduler_params(run_cfg, "dynamic") or run_cfg.get("dynamic_lr", run_cfg)
    training_lr = float(training_cfg.get("lr", 1.0e-3))
    return build_dynamic_scheduler(cfg, training_lr)
