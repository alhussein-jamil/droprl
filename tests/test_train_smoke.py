from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]


@pytest.mark.slow
def test_mock_train_one_iter(tmp_path: Path) -> None:
    env = os.environ.copy()
    env["RAY_DISABLE_IMPORT_WARNING"] = "1"
    cmd = [
        sys.executable,
        str(REPO / "scripts" / "train.py"),
        "--task",
        "mock",
        "--train",
        "MockPPO",
        "--output",
        str(tmp_path),
        "--name",
        "ci_smoke",
        "--iters",
        "1",
        "--clean",
    ]
    result = subprocess.run(
        cmd,
        cwd=REPO,
        env=env,
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert result.returncode == 0, result.stderr[-2000:]
    run_dir = tmp_path / "mock" / "ci_smoke"
    assert (run_dir / "checkpoint_latest").is_dir()
    assert (run_dir / "training_state.json").is_file()
