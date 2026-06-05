from __future__ import annotations

from droprl.config import list_tasks, list_train_configs, load_experiment


def test_list_tasks_includes_mock() -> None:
    assert "mock" in list_tasks()


def test_list_train_configs_includes_mock_ppo() -> None:
    assert "MockPPO" in list_train_configs()


def test_load_experiment_mock() -> None:
    cfg = load_experiment(task="mock", train="MockPPO")
    assert cfg["env"]["name"] == "mock"
    assert "training" in cfg
    assert cfg["training"]["framework"]["framework"] == "torch"
