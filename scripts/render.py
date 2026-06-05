#!/usr/bin/env python3

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from droprl.config import load_experiment
from droprl.logging import setup_logging
from droprl.rllib.algorithms import default_train_name
from droprl.rllib.render import render_checkpoint
from droprl.runs.checkpoints import resolve_render_paths


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Render a checkpoint rollout to MP4.",
    )
    p.add_argument("--task", type=str, default="mock")
    p.add_argument(
        "--train",
        type=str,
        default=None,
        help="Train config name (default: capitalized task name)",
    )
    p.add_argument(
        "--runs-root",
        type=str,
        default="runs",
        help="Runs root (default: runs)",
    )
    p.add_argument(
        "--name",
        type=str,
        default=None,
        help="Run directory name (default: latest run for task)",
    )
    p.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Checkpoint path (default: checkpoint_best in latest run)",
    )
    p.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output MP4 path (default: <run>/simulations/render_best.mp4)",
    )
    p.add_argument(
        "--latest",
        action="store_true",
        help="Use checkpoint_latest instead of checkpoint_best",
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()
    setup_logging(level=logging.INFO, colorize=True)
    log = logging.getLogger("droprl.render")

    prefer = "latest" if args.latest else "best"
    try:
        run_dir, checkpoint, output = resolve_render_paths(
            Path(args.runs_root),
            args.task,
            name=args.name,
            checkpoint=args.checkpoint,
            output=args.output,
            prefer=prefer,
        )
    except FileNotFoundError as exc:
        raise SystemExit(str(exc)) from exc

    log.info("run=%s checkpoint=%s output=%s", run_dir.name, checkpoint, output)

    train_name = args.train or default_train_name(args.task)
    config = load_experiment(task=args.task, train=train_name)
    path = render_checkpoint(
        config=config,
        env_name=args.task,
        checkpoint=checkpoint,
        output_path=output,
    )
    if path is None:
        log.warning("No video written (env may not support rgb_array rendering)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
