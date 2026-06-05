from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def load_yaml(path: str | Path) -> dict[str, Any]:
    p = Path(path)
    if not p.is_file():
        return {}
    data = yaml.safe_load(p.read_text())
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise TypeError(f"Expected mapping at {p}, got {type(data).__name__}")
    return data


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def abs_path_str(path: str | Path) -> str:
    """Absolute path string for APIs (RLlib, subprocess) that require ``str``."""
    return str(Path(path).resolve())


def configs_dir() -> Path:
    return repo_root() / "configs"


def list_tasks() -> list[str]:
    root = repo_root() / "envs"
    if not root.is_dir():
        return []
    return sorted(
        p.name
        for p in root.iterdir()
        if p.is_dir() and (p / "env.py").is_file() and not p.name.startswith("_")
    )


def list_train_configs() -> list[str]:
    root = configs_dir() / "train"
    if not root.is_dir():
        return []
    return sorted(p.stem for p in root.glob("*.yaml") if not p.stem.startswith("_"))


def load_task_config(task: str) -> dict[str, Any]:
    path = repo_root() / "envs" / task / "config.yaml"
    if not path.is_file():
        raise FileNotFoundError(f"Missing task config: {path}")
    raw = load_yaml(path)
    if "config" in raw:
        return {"env": {"name": task, "config": raw["config"]}}
    return {"env": {"name": task, "config": raw}}


def load_experiment(*, task: str, train: str) -> dict[str, Any]:
    """Merge root + task + train configs."""
    root_cfg = load_yaml(configs_dir() / "config.yaml")
    task_cfg = load_task_config(task)
    train_path = configs_dir() / "train" / f"{train}.yaml"
    if not train_path.is_file():
        raise FileNotFoundError(f"Missing train config: {train_path}")
    train_cfg = load_yaml(train_path)
    return deep_merge(deep_merge(root_cfg, task_cfg), train_cfg)
