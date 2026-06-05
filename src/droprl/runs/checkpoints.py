from __future__ import annotations

import json
import shutil
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

CHECKPOINT_LATEST = "checkpoint_latest"
CHECKPOINT_BEST = "checkpoint_best"
STATE_FILE = "training_state.json"
POLICY_WEIGHTS_FILE = "policy_weights.npz"
CHECKPOINT_META_FILE = "checkpoint_meta.json"


@dataclass(frozen=True)
class TrainingState:
    iteration: int = 0
    best_reward: float = float("-inf")
    best_iteration: int = 0


def read_state(run_dir: Path) -> TrainingState:
    path = run_dir / STATE_FILE
    if not path.is_file():
        return TrainingState()
    raw = json.loads(path.read_text())
    return TrainingState(
        iteration=int(raw.get("iteration", 0)),
        best_reward=float(raw.get("best_reward", float("-inf"))),
        best_iteration=int(raw.get("best_iteration", 0)),
    )


def write_state(
    run_dir: Path,
    state: TrainingState,
    *,
    checkpoint_iteration: int | None = None,
) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "iteration": int(state.iteration),
        "best_reward": float(state.best_reward),
        "best_iteration": int(state.best_iteration),
        "updated_at": time.time(),
    }
    if checkpoint_iteration is not None:
        payload["checkpoint_iteration"] = int(checkpoint_iteration)
    (run_dir / STATE_FILE).write_text(json.dumps(payload, indent=2))


def _replace_dir(dest: Path) -> None:
    if dest.exists():
        shutil.rmtree(dest)
    dest.mkdir(parents=True, exist_ok=True)


def is_usable_checkpoint(path: Path) -> bool:
    if not path.is_dir():
        return False
    if (path / POLICY_WEIGHTS_FILE).is_file():
        return True
    return (path / "policies").is_dir() or (path / "rllib_checkpoint.json").is_file()


def checkpoint_for_render(run_dir: Path, *, prefer: str = "latest") -> Path | None:
    latest = latest_dir(run_dir)
    best = best_dir(run_dir)
    if prefer == "latest":
        if is_usable_checkpoint(latest):
            return latest
        if is_usable_checkpoint(best):
            return best
        return None
    if is_usable_checkpoint(best):
        return best
    if is_usable_checkpoint(latest):
        return latest
    return None


def _algo_training_iteration(algo: Any, *, loop_iteration: int | None) -> int | None:
    iteration = getattr(algo, "iteration", None)
    if iteration is not None:
        return int(iteration)
    if loop_iteration is not None:
        return int(loop_iteration) + 1
    return None


def _write_checkpoint_meta(dest: Path, *, training_iteration: int | None) -> None:
    if training_iteration is None:
        return
    (dest / CHECKPOINT_META_FILE).write_text(
        json.dumps({"training_iteration": int(training_iteration)}, indent=2)
    )


def save_checkpoint(
    algo: Any,
    dest: Path,
    *,
    loop_iteration: int | None = None,
) -> Path:
    """Save an RLlib Algorithm checkpoint into `dest` (replacing it)."""
    training_iteration = _algo_training_iteration(algo, loop_iteration=loop_iteration)
    _replace_dir(dest)
    algo.save(checkpoint_dir=str(dest.resolve()))
    _write_checkpoint_meta(dest, training_iteration=training_iteration)
    try:
        from droprl.rllib.loader import save_policy_artifacts

        save_policy_artifacts(algo, dest)
    except Exception:
        pass
    return dest


def latest_dir(run_dir: Path) -> Path:
    return run_dir / CHECKPOINT_LATEST


def best_dir(run_dir: Path) -> Path:
    return run_dir / CHECKPOINT_BEST


def task_runs_root(runs_root: Path, task: str) -> Path:
    return Path(runs_root) / task


def _run_has_checkpoint(run_dir: Path) -> bool:
    return is_usable_checkpoint(latest_dir(run_dir)) or is_usable_checkpoint(best_dir(run_dir))


def read_checkpoint_iteration(checkpoint: Path) -> int | None:
    """Read RLlib ``training_iteration`` stored in an algorithm checkpoint."""
    meta = checkpoint / CHECKPOINT_META_FILE
    if meta.is_file():
        try:
            value = json.loads(meta.read_text()).get("training_iteration")
            return int(value) if value is not None else None
        except (json.JSONDecodeError, OSError, TypeError, ValueError):
            pass

    pkl = checkpoint / "algorithm_state.pkl"
    if not pkl.is_file():
        return None
    try:
        import cloudpickle

        state = cloudpickle.loads(pkl.read_bytes())
        value = state.get("training_iteration")
        return int(value) if value is not None else None
    except Exception:
        return None


def checkpoint_for_resume(run_dir: Path) -> Path | None:
    """Pick the freshest checkpoint for resume (latest or best)."""
    state = read_state(run_dir)
    candidates: list[tuple[int, Path]] = []

    latest = latest_dir(run_dir)
    if is_usable_checkpoint(latest):
        ckpt_iter = read_checkpoint_iteration(latest)
        # Unknown latest age must not beat a known-good best checkpoint.
        score = ckpt_iter if ckpt_iter is not None else 0
        candidates.append((score, latest))

    best = best_dir(run_dir)
    if is_usable_checkpoint(best):
        ckpt_iter = read_checkpoint_iteration(best)
        score = ckpt_iter if ckpt_iter is not None else state.best_iteration + 1
        candidates.append((score, best))

    if not candidates:
        return None
    return max(candidates, key=lambda item: item[0])[1]


def sync_algo_progress(algo: Any, *, loop_iteration: int) -> None:
    """Align RLlib counters so TensorBoard steps continue after resume."""
    # RLlib logs ``training_iteration = self.iteration`` in results.
    algo._iteration = int(loop_iteration) + 1


def find_resumable_run(runs_root: Path, task: str) -> Path | None:
    """Return the newest run directory under runs/<task>/ with a checkpoint."""
    root = task_runs_root(runs_root, task)
    if not root.is_dir():
        return None

    candidates = [d for d in root.iterdir() if d.is_dir() and _run_has_checkpoint(d)]
    if not candidates:
        return None

    def _sort_key(path: Path) -> tuple[int, float]:
        state = read_state(path)
        return (state.iteration, path.stat().st_mtime)

    return max(candidates, key=_sort_key)


def find_latest_run(
    runs_root: Path,
    task: str,
    *,
    name: str | None = None,
) -> Path | None:
    """Return a run directory for rendering (latest by mtime unless ``name`` is set)."""
    root = task_runs_root(runs_root, task)
    if name:
        run_dir = root / name
        return run_dir if run_dir.is_dir() else None

    if not root.is_dir():
        return None

    candidates = [d for d in root.iterdir() if d.is_dir() and is_usable_checkpoint(best_dir(d))]
    if not candidates:
        candidates = [d for d in root.iterdir() if d.is_dir() and _run_has_checkpoint(d)]
    if not candidates:
        return None
    return max(candidates, key=lambda path: path.stat().st_mtime)


def resolve_render_paths(
    runs_root: Path,
    task: str,
    *,
    name: str | None = None,
    checkpoint: Path | str | None = None,
    output: Path | str | None = None,
    prefer: str = "best",
) -> tuple[Path, Path, Path]:
    """Resolve (run_dir, checkpoint, output_mp4) for ``make render`` defaults."""
    if checkpoint:
        ckpt = Path(checkpoint)
        if not is_usable_checkpoint(ckpt):
            raise FileNotFoundError(f"Not a usable checkpoint: {ckpt}")
        run_dir = ckpt.parent
        out = Path(output) if output else run_dir / "simulations" / "render_best.mp4"
        return run_dir, ckpt, out

    run_dir = find_latest_run(runs_root, task, name=name)
    if run_dir is None:
        raise FileNotFoundError(
            f"No run with checkpoint_best under {task_runs_root(runs_root, task)}"
        )

    ckpt = checkpoint_for_render(run_dir, prefer=prefer)
    if ckpt is None:
        raise FileNotFoundError(f"No usable checkpoint in {run_dir}")

    if output:
        out = Path(output)
    else:
        label = "best" if prefer == "best" else "latest"
        out = run_dir / "simulations" / f"render_{label}.mp4"
    return run_dir, ckpt, out


def restore_checkpoint(algo: Any, checkpoint: Path) -> bool:
    """Restore an RLlib algorithm from a full RLlib checkpoint directory."""
    import os

    from droprl.rllib.loader import is_rllib_checkpoint

    if not is_rllib_checkpoint(checkpoint):
        return False

    algo.restore(os.path.abspath(checkpoint.resolve()))
    return True
