from __future__ import annotations

from droprl.config import list_tasks, list_train_configs, load_experiment


def test_list_tasks_includes_mock() -> None:
    assert "mock" in list_tasks()


def test_list_train_configs_includes_mock() -> None:
    assert "Mock" in list_train_configs()


def test_load_experiment_mock() -> None:
    cfg = load_experiment(task="mock", train="Mock")
    assert cfg["env"]["name"] == "mock"
    assert cfg["algorithm"] == "ppo"
    assert "training" in cfg
    assert cfg["training"]["framework"]["framework"] == "torch"


def test_load_experiment_cartpole_dqn() -> None:
    cfg = load_experiment(task="cartpole", train="CartpoleDQN")
    assert cfg["algorithm"] == "dqn"
    assert cfg["training"]["training"]["n_step"] == 3


def test_load_experiment_pendulum_sac() -> None:
    cfg = load_experiment(task="pendulum", train="PendulumSAC")
    assert cfg["algorithm"] == "sac"
    assert cfg["env"]["config"]["gym_id"] == "Pendulum-v1"
