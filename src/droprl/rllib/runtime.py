from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

from droprl.config import repo_root


def pythonpath_for_task(task: str) -> str:
    root = repo_root()
    paths = [
        str(root / "src"),
        str(root / "envs" / task),
    ]
    existing = os.environ.get("PYTHONPATH", "")
    if existing:
        paths.append(existing)
    return os.pathsep.join(paths)


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
    if str(env_dir) not in sys.path:
        sys.path.insert(0, str(env_dir))
    return env_dir
