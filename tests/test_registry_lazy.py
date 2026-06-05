from __future__ import annotations

from droprl.rllib.registry import register_rllib_envs


def test_register_single_task_does_not_import_cassie() -> None:
    mapping = register_rllib_envs(tasks=["mock"])
    assert mapping == {"mock": "mock-v0"}
