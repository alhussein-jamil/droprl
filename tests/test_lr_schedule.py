from __future__ import annotations

import json
from pathlib import Path

from droprl.rllib.lr_schedule import (
    DynamicLRController,
    build_dynamic_lr_controller,
    build_lr_scheduler,
    resolve_scheduler_type,
)


def _controller(**overrides: object) -> DynamicLRController:
    defaults = {
        "initial_lr": 1e-3,
        "min_lr": 1e-5,
        "max_lr": 1e-3,
        "window_iters": 20,
        "trend_fast_iters": 5,
        "trend_slow_iters": 20,
        "cooldown_iters": 0,
        "breakthrough_rel_improvement": 0.08,
        "plateau_rel_improvement": 0.015,
        "peak_near_ratio": 0.90,
        "regression_drop_ratio": 0.15,
        "recovery_min_trend": 0.02,
        "recovery_boost_factor": 1.10,
        "restart_lr_ratio": 0.5,
        "restart_after_iters": 400,
    }
    defaults.update(overrides)
    return DynamicLRController(**defaults)


def _warmup(controller: DynamicLRController, rewards: list[float], start: int = 0) -> None:
    for i, reward in enumerate(rewards, start=start):
        controller.observe(i, reward)


def test_recovery_boost_after_regression() -> None:
    controller = _controller(cooldown_iters=250)
    peak_phase = [520.0 + (i % 5) for i in range(25)]
    regression = [300.0 - i * 3 for i in range(15)]
    dip_recovery = [220.0 + i * 5 for i in range(25)]
    _warmup(controller, peak_phase + regression + dip_recovery)

    controller._last_adjust_iter = -(10**9)
    lr_before = controller.lr
    lr_after = controller.observe(
        len(peak_phase) + len(regression) + len(dip_recovery),
        360.0,
    )

    assert controller.peak_reward_mean >= 520.0
    assert controller._ema_fast < controller.peak_reward_mean * 0.85
    assert lr_after > lr_before * 1.05


def test_no_decay_during_recovery_climb() -> None:
    controller = _controller(cooldown_iters=0, min_lr=1e-6)
    peak = [400.0] * 20
    drop = [250.0 - i for i in range(10)]
    climb = [220.0 + i * 5 for i in range(20)]
    _warmup(controller, peak + drop + climb)

    lr_before = controller.lr
    lr_after = controller.observe(50, 330.0)
    assert lr_after >= lr_before


def test_warm_restart_after_sustained_regression() -> None:
    controller = _controller(
        cooldown_iters=0,
        restart_after_iters=5,
        restart_lr_ratio=0.5,
        min_lr=1e-6,
    )
    peak = [400.0] * 20
    flat_low = [180.0 + (i % 2) for i in range(25)]
    _warmup(controller, peak + flat_low)

    assert controller.lr >= controller.initial_lr * 0.5 * 0.99


def test_breakthrough_near_peak_with_positive_trend() -> None:
    controller = _controller(cooldown_iters=0)
    baseline = [350.0] * 15
    climb = [350.0 + i * 3 for i in range(1, 16)]
    _warmup(controller, baseline + climb)

    lr = controller.observe(30, 410.0)
    assert lr > 1e-3 * 0.99


def test_decay_on_plateau_near_peak() -> None:
    controller = _controller(cooldown_iters=0)
    climb = [300.0 + i * 4 for i in range(20)]
    plateau = [380.0 + (i % 3) * 0.1 for i in range(20)]
    _warmup(controller, climb + plateau)

    lr = controller.observe(40, 380.1)
    assert lr < 1e-3


def test_no_decay_during_regression_before_restart() -> None:
    controller = _controller(cooldown_iters=250, restart_after_iters=400)
    peak = [390.0] * 20
    drift_down = [390.0 - i * 3 for i in range(20)]
    _warmup(controller, peak + drift_down)

    assert controller.lr >= 7.0e-4


def test_state_roundtrip_preserves_controller_state(tmp_path: Path) -> None:
    controller = _controller()
    _warmup(controller, [100.0 + i for i in range(25)])

    path = tmp_path / "lr_schedule.json"
    controller.save_state(path)

    restored = _controller()
    assert restored.load_state(path)
    assert restored.current_lr == controller.current_lr
    assert restored.peak_reward_mean == controller.peak_reward_mean
    assert restored._ema_fast == controller._ema_fast
    assert restored._ema_slow == controller._ema_slow
    assert restored._regression_iters == controller._regression_iters

    data = json.loads(path.read_text())
    assert set(data) == {
        "type",
        "current_lr",
        "peak_reward_mean",
        "last_adjust_iter",
        "reward_history",
        "ema_fast",
        "ema_slow",
        "regression_iters",
    }
    assert data["type"] == "dynamic"


def test_load_rebuilds_emas_from_history_when_missing(tmp_path: Path) -> None:
    state = {
        "current_lr": 5e-4,
        "peak_reward_mean": 400.0,
        "last_adjust_iter": 100,
        "reward_history": [300.0, 310.0, 320.0, 330.0, 340.0],
        "regression_iters": 2,
    }
    path = tmp_path / "dynamic_lr.json"
    path.write_text(json.dumps(state))

    controller = _controller(window_iters=5, trend_slow_iters=5, trend_fast_iters=2)
    assert controller.load_state(path)

    expected = _controller(window_iters=5, trend_slow_iters=5, trend_fast_iters=2)
    expected._rebuild_emas_from_history(state["reward_history"])
    assert controller._ema_fast == expected._ema_fast
    assert controller._ema_slow == expected._ema_slow
    assert controller._regression_iters == 2


def test_build_from_config() -> None:
    controller = build_dynamic_lr_controller(
        {
            "lr_schedule": {
                "type": "dynamic",
                "peak_near_ratio": 0.85,
                "trend_fast_iters": 40,
                "recovery_boost_factor": 1.12,
            }
        },
        {"lr": 2e-3},
    )
    assert controller.peak_near_ratio == 0.85
    assert controller.trend_fast_iters == 40
    assert controller.recovery_boost_factor == 1.12
    assert controller.initial_lr == 2e-3


def test_resolve_scheduler_type_from_lr_schedule() -> None:
    assert resolve_scheduler_type({"lr_schedule": "cosine"}) == "cosine"
    assert resolve_scheduler_type({"lr_schedule": {"type": "linear"}}) == "linear"
    assert resolve_scheduler_type({"lr_schedule": {"type": "fixed"}}) == "fixed"


def test_resolve_scheduler_type_legacy_dynamic_lr() -> None:
    assert resolve_scheduler_type({"dynamic_lr": {"enabled": True}}) == "dynamic"
    assert resolve_scheduler_type({"dynamic_lr": {"enabled": False}}) == "fixed"
    assert resolve_scheduler_type({}) == "fixed"


def test_build_lr_scheduler_fixed() -> None:
    scheduler = build_lr_scheduler(
        {"lr_schedule": "fixed"},
        {"lr": 3e-4},
        total_iters=100,
    )
    assert scheduler.name == "fixed"
    assert scheduler.observe(0, 1.0) == 3e-4
    assert scheduler.observe(50, 999.0) == 3e-4


def test_build_lr_scheduler_cosine_decays() -> None:
    scheduler = build_lr_scheduler(
        {
            "lr_schedule": {
                "type": "cosine",
                "min_lr_ratio": 0.0,
            }
        },
        {"lr": 1e-3},
        total_iters=100,
    )
    start = scheduler.observe(0, 0.0)
    end = scheduler.observe(99, 0.0)
    assert start == 1e-3
    assert end < start * 0.1


def test_build_lr_scheduler_linear_decays() -> None:
    scheduler = build_lr_scheduler(
        {
            "lr_schedule": {
                "type": "linear",
                "min_lr_ratio": 0.0,
            }
        },
        {"lr": 1e-3},
        total_iters=100,
    )
    start = scheduler.observe(0, 0.0)
    end = scheduler.observe(99, 0.0)
    assert start == 1e-3
    assert end < start * 0.1
