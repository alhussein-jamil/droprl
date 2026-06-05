from __future__ import annotations

import json
from pathlib import Path

from droprl.runs.checkpoints import (
    TrainingState,
    best_dir,
    is_usable_checkpoint,
    latest_dir,
    read_state,
    write_state,
)


def test_training_state_roundtrip(tmp_path: Path) -> None:
    state = TrainingState(iteration=3, best_reward=1.5, best_iteration=2)
    write_state(tmp_path, state, checkpoint_iteration=3)
    loaded = read_state(tmp_path)
    assert loaded.iteration == 3
    assert loaded.best_reward == 1.5
    assert loaded.best_iteration == 2

    raw = json.loads((tmp_path / "training_state.json").read_text())
    assert raw["checkpoint_iteration"] == 3


def test_is_usable_checkpoint_npz(tmp_path: Path) -> None:
    ckpt = tmp_path / "checkpoint_best"
    ckpt.mkdir()
    (ckpt / "policy_weights.npz").write_bytes(b"")
    assert is_usable_checkpoint(ckpt)


def test_checkpoint_dirs(tmp_path: Path) -> None:
    assert latest_dir(tmp_path).name == "checkpoint_latest"
    assert best_dir(tmp_path).name == "checkpoint_best"
