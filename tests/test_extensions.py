from __future__ import annotations

from droprl.tasks.extensions import discover_callbacks, load_task_symbol


def test_discover_callbacks_none_for_mock() -> None:
    assert discover_callbacks("mock") is None


def test_discover_callbacks_cassie() -> None:
    cls = discover_callbacks("cassie")
    assert cls is not None
    assert cls.__name__ == "Callbacks"


def test_load_task_symbol_cassie() -> None:
    cls = load_task_symbol("cassie", "callbacks.Callbacks")
    assert cls.__name__ == "Callbacks"
