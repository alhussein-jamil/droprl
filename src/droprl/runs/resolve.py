from __future__ import annotations

import time
from pathlib import Path

from droprl.runs.checkpoints import (
    checkpoint_for_resume,
    find_resumable_run,
)


def resolve_run_dir(
    runs_root: Path,
    task: str,
    *,
    name: str | None,
    clean: bool,
) -> tuple[Path, str, bool]:
    """Pick run directory. Returns (run_dir, run_name, will_resume)."""
    task_root = runs_root / task
    task_root.mkdir(parents=True, exist_ok=True)

    if clean:
        run_name = name or time.strftime("%Y%m%d_%H%M%S")
        return task_root / run_name, run_name, False

    if name:
        run_dir = task_root / name
        resume = checkpoint_for_resume(run_dir) is not None
        return run_dir, name, resume

    existing = find_resumable_run(runs_root, task)
    if existing is not None:
        return existing, existing.name, True

    run_name = time.strftime("%Y%m%d_%H%M%S")
    return task_root / run_name, run_name, False
