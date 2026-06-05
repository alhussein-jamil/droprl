from __future__ import annotations

import sys
from os import environ
from pathlib import Path
from typing import Any

from droprl.config import abs_path_str, repo_root


def _path_list_sep() -> str:
    return ";" if sys.platform == "win32" else ":"


def pythonpath_for_task(task: str) -> str:
    root = repo_root()
    paths = [root / "src", root / "envs" / task]
    parts = [abs_path_str(path) for path in paths]
    existing = environ.get("PYTHONPATH", "")
    if existing:
        parts.append(existing)
    return _path_list_sep().join(parts)


def build_runtime_env(task: str) -> dict[str, Any]:
    """Ray worker environment so env-local modules (callbacks, cassie.py, etc.) import."""
    return {
        "env_vars": {
            "PYTHONPATH": pythonpath_for_task(task),
            "PYTHONWARNINGS": "ignore::DeprecationWarning",
        }
    }


def ensure_task_path(task: str) -> Path:
    env_dir = repo_root() / "envs" / task
    env_dir_str = abs_path_str(env_dir)
    if env_dir_str not in sys.path:
        sys.path.insert(0, env_dir_str)
    return env_dir
