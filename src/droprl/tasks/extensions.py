from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Any

from droprl.config import repo_root
from droprl.rllib.runtime import ensure_task_path

CALLBACKS_FILE = "callbacks.py"
CALLBACKS_CLASS = "Callbacks"


def _task_dir(task: str) -> Path:
    return repo_root() / "envs" / task


def _load_module_from_path(module_name: str, path: Path) -> Any:
    if module_name in sys.modules:
        return sys.modules[module_name]

    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load module from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def load_task_symbol(task: str, qualname: str) -> type:
    """Load ``module.Class`` from ``envs/<task>/`` (task-local PYTHONPATH)."""
    if "." not in qualname:
        raise ValueError(f"Expected dotted path module.Class, got {qualname!r}")

    module_name, attr_name = qualname.rsplit(".", 1)
    env_dir = ensure_task_path(task)
    module_path = env_dir / f"{module_name}.py"
    if not module_path.is_file():
        raise FileNotFoundError(f"Task module not found: {module_path}")

    module = _load_module_from_path(module_name, module_path)
    try:
        symbol = getattr(module, attr_name)
    except AttributeError as exc:
        raise AttributeError(f"{qualname} not found in envs/{task}/{module_name}.py") from exc
    if not isinstance(symbol, type):
        raise TypeError(f"{qualname} must be a class, got {type(symbol).__name__}")
    return symbol


def discover_callbacks(task: str) -> type | None:
    """Return RLlib callback class if ``envs/<task>/callbacks.py`` defines ``Callbacks``."""
    callbacks_path = _task_dir(task) / CALLBACKS_FILE
    if not callbacks_path.is_file():
        return None
    return load_task_symbol(task, f"callbacks.{CALLBACKS_CLASS}")
