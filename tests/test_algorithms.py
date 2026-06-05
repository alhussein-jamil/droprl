from __future__ import annotations

import pytest

from droprl.config import load_experiment
from droprl.rllib.algorithms import (
    build_algorithm,
    default_train_name,
    list_algorithms,
    resolve_algorithm,
)


def test_list_algorithms() -> None:
    assert "ppo" in list_algorithms()
    assert "sac" in list_algorithms()
    assert "dqn" in list_algorithms()


def test_default_train_name() -> None:
    assert default_train_name("mock") == "Mock"
    assert default_train_name("cartpole") == "Cartpole"


def test_resolve_algorithm_from_train_config() -> None:
    cfg = load_experiment(task="mock", train="Mock")
    assert resolve_algorithm(cfg) == "ppo"


def test_unknown_algorithm_raises() -> None:
    with pytest.raises(ValueError, match="Unknown algorithm"):
        resolve_algorithm({"algorithm": "not_a_real_algo"})


def test_build_algorithm_resolves_auto_env_runners() -> None:
    """build_algorithm should resolve num_env_runners: auto before RLlib setup."""
    import ray

    from droprl.rllib.registry import register_rllib_envs

    cfg = load_experiment(task="mock", train="Mock")
    env_map = register_rllib_envs(tasks=["mock"])
    cfg.setdefault("training", {}).setdefault("environment", {})["env"] = env_map["mock"]
    cfg["training"]["env_runners"]["num_env_runners"] = 0

    ray.init(ignore_reinit_error=True, num_cpus=1, include_dashboard=False)
    try:
        algo = build_algorithm(cfg, task="mock")
        try:
            assert algo.get_policy() is not None
        finally:
            algo.stop()
    finally:
        ray.shutdown()
