from __future__ import annotations

from droprl.envs.registry import list_envs, load_env


def test_list_envs_discovers_mock() -> None:
    names = list_envs()
    assert "mock" in names


def test_load_mock_env_step() -> None:
    env = load_env("mock", {"horizon": 5, "target": 0, "start": 1})
    obs = env.reset()
    assert obs is not None
    result = env.step(0)
    assert result.reward is not None
