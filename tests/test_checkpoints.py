from __future__ import annotations

import json
from pathlib import Path

from droprl.runs.checkpoints import (
    TrainingState,
    best_dir,
    find_resumable_run,
    is_usable_checkpoint,
    latest_backup_dir,
    latest_dir,
    read_state,
    rotate_latest_backups,
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


def _fake_checkpoint(path: Path, *, label: str) -> None:
    path.mkdir(parents=True, exist_ok=True)
    (path / "policy_weights.npz").write_bytes(b"")
    (path / "checkpoint_meta.json").write_text(f'{{"label": "{label}"}}')


def test_rotate_latest_backups_disabled(tmp_path: Path) -> None:
    _fake_checkpoint(latest_dir(tmp_path), label="current")
    rotate_latest_backups(tmp_path, keep=0)
    assert not latest_backup_dir(tmp_path, 0).exists()


def test_find_resumable_run_prefers_newest_not_highest_iter(tmp_path: Path) -> None:
    task_root = tmp_path / "cassie"
    old_run = task_root / "20260608_163021"
    new_run = task_root / "20260609_074238"
    for run_dir, iteration in ((old_run, 9000), (new_run, 100)):
        _fake_checkpoint(latest_dir(run_dir), label=str(iteration))
        write_state(run_dir, TrainingState(iteration=iteration))

    assert find_resumable_run(tmp_path, "cassie") == new_run


def test_rotate_latest_backups_ring_buffer(tmp_path: Path) -> None:
    _fake_checkpoint(latest_dir(tmp_path), label="v1")
    rotate_latest_backups(tmp_path, keep=2)
    assert latest_backup_dir(tmp_path, 0).is_dir()
    assert (
        '"label": "v1"'
        in latest_backup_dir(tmp_path, 0).joinpath("checkpoint_meta.json").read_text()
    )

    _fake_checkpoint(latest_dir(tmp_path), label="v2")
    rotate_latest_backups(tmp_path, keep=2)
    assert (
        '"label": "v2"'
        in latest_backup_dir(tmp_path, 0).joinpath("checkpoint_meta.json").read_text()
    )
    assert (
        '"label": "v1"'
        in latest_backup_dir(tmp_path, 1).joinpath("checkpoint_meta.json").read_text()
    )
    assert not latest_backup_dir(tmp_path, 2).exists()

    _fake_checkpoint(latest_dir(tmp_path), label="v3")
    rotate_latest_backups(tmp_path, keep=2)
    assert (
        '"label": "v3"'
        in latest_backup_dir(tmp_path, 0).joinpath("checkpoint_meta.json").read_text()
    )
    assert (
        '"label": "v2"'
        in latest_backup_dir(tmp_path, 1).joinpath("checkpoint_meta.json").read_text()
    )
