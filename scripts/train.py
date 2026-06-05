#!/usr/bin/env python3

from __future__ import annotations

import argparse
import logging
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import ray
from ray.rllib.algorithms.ppo import PPOConfig

from droprl.config import list_tasks, list_train_configs, load_experiment
from droprl.logging import setup_logging
from droprl.rllib.lr_schedule import (
    apply_lr_to_trainer,
    build_dynamic_lr_controller,
    lr_state_path,
)
from droprl.rllib.registry import register_rllib_envs
from droprl.rllib.resources import resolve_ray_config, resolve_training_section
from droprl.rllib.run_config import (
    chkpt_every_iters,
    chkpt_freq_seconds,
    render_every_seconds,
    resolve_end_iter,
)
from droprl.rllib.runtime import build_runtime_env
from droprl.rllib.tune_logger import TrainLogger
from droprl.runs.checkpoints import (
    TrainingState,
    best_dir,
    checkpoint_for_render,
    checkpoint_for_resume,
    latest_dir,
    read_checkpoint_iteration,
    read_state,
    restore_checkpoint,
    save_checkpoint,
    sync_algo_progress,
    write_state,
)
from droprl.runs.resolve import resolve_run_dir
from droprl.tasks.extensions import discover_callbacks


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train a task with RLlib PPO.")
    p.add_argument(
        "--task",
        type=str,
        default="mock",
        help=f"Task name (envs/<task>/). Available: {list_tasks()}",
    )
    p.add_argument(
        "--train",
        type=str,
        default="MockPPO",
        help=f"Train config (configs/train/<name>.yaml). Available: {list_train_configs()}",
    )
    p.add_argument("--output", type=str, default="runs", help="Output directory")
    p.add_argument("--name", type=str, default=None, help="Run name (directory name)")
    p.add_argument("--iters", type=int, default=None, help="Override run.iters")
    p.add_argument(
        "--clean",
        action="store_true",
        help="Start a new run (do not resume the latest checkpoint)",
    )
    return p.parse_args()


def _build_ppo(cfg: dict[str, Any], *, task: str):
    training = cfg.get("training", {})
    env_id = training.get("environment", {}).get("env")
    if not env_id:
        raise ValueError("config.training.environment.env must be set to the RLlib env id.")

    builder = (
        PPOConfig()
        .api_stack(
            enable_rl_module_and_learner=False,
            enable_env_runner_and_connector_v2=False,
        )
        .environment(**training.get("environment", {}))
        .env_runners(**training.get("env_runners", {}))
        .debugging(logger_creator=_logger_creator(training.get("logdir")))
        .training(**training.get("training", {}))
        .framework(**training.get("framework", {}))
        .resources(**training.get("resources", {}))
        .fault_tolerance(**training.get("fault_tolerance", {}))
    )

    callbacks_cls = discover_callbacks(task)
    if callbacks_cls is not None:
        builder = builder.callbacks(callbacks_class=callbacks_cls)

    return builder.build_algo()


def _logger_creator(logdir: str | None):
    def _creator(config):
        if not logdir:
            return TrainLogger(config, Path.cwd().as_posix(), trial=None)
        Path(logdir).mkdir(parents=True, exist_ok=True)
        return TrainLogger(config, str(logdir), trial=None)

    return _creator


def _spawn_render(
    *,
    task: str,
    train: str,
    checkpoint: Path,
    run_dir: Path,
    label: str,
) -> None:
    sim_dir = run_dir / "simulations"
    sim_dir.mkdir(parents=True, exist_ok=True)
    output = sim_dir / f"run_{label}.mp4"
    cmd = [
        sys.executable,
        str(Path(__file__).resolve().parent / "render.py"),
        "--task",
        task,
        "--train",
        train,
        "--checkpoint",
        str(checkpoint.resolve()),
        "--output",
        str(output),
    ]
    logging.getLogger("droprl.train").info("Rendering checkpoint (subprocess): %s", " ".join(cmd))
    repo_root = Path(__file__).resolve().parents[1]
    result = subprocess.run(cmd, cwd=repo_root)
    if result.returncode != 0:
        logging.getLogger("droprl.train").error(
            "Checkpoint render failed (exit %s)", result.returncode
        )


def _save_latest_checkpoint(
    algo: Any,
    *,
    run_dir: Path,
    iteration: int,
    state: TrainingState,
    lr_controller: Any,
    lr_path: Path,
) -> None:
    save_checkpoint(algo, latest_dir(run_dir), loop_iteration=iteration)
    lr_controller.save_state(lr_path)
    write_state(
        run_dir,
        state,
        checkpoint_iteration=iteration,
    )


def main() -> int:
    args = parse_args()
    setup_logging(level=logging.INFO)
    log = logging.getLogger("droprl.train")

    config = load_experiment(task=args.task, train=args.train)
    runs_root = Path(args.output)
    runs_root.mkdir(parents=True, exist_ok=True)

    env_map = register_rllib_envs()
    if args.task not in env_map:
        raise SystemExit(f"Unknown task '{args.task}'. Available: {sorted(env_map)}")

    run_dir, run_name, will_resume = resolve_run_dir(
        runs_root,
        args.task,
        name=args.name,
        clean=args.clean,
    )
    run_dir.mkdir(parents=True, exist_ok=True)
    logdir = run_dir / "ray_results"
    logdir.mkdir(parents=True, exist_ok=True)

    chosen_env_id = env_map[args.task]
    env_cfg = dict((config.get("env") or {}).get("config") or {})

    training_section = config.setdefault("training", {})
    training_section.update(resolve_training_section(training_section))
    training_section.setdefault("environment", {})
    training_section["environment"]["env"] = chosen_env_id
    training_section["environment"]["env_config"] = env_cfg
    training_section["logdir"] = str(logdir)

    ray_cfg = resolve_ray_config(config.get("ray", {}))
    ray.init(
        ignore_reinit_error=True,
        runtime_env=build_runtime_env(args.task),
        **ray_cfg,
    )

    algo = _build_ppo(config, task=args.task)
    run_cfg = config.get("run", {})
    training_cfg = training_section.get("training", {})
    chkpt_every = chkpt_every_iters(run_cfg)
    chkpt_freq = chkpt_freq_seconds(run_cfg)
    render_every = render_every_seconds(run_cfg)
    render_on_best = bool(run_cfg.get("render_on_best", False))

    start_iter = 0
    state = read_state(run_dir)
    if will_resume:
        resume_ckpt = checkpoint_for_resume(run_dir)
        try:
            if resume_ckpt is not None and restore_checkpoint(algo, resume_ckpt):
                start_iter = state.iteration + 1
                ckpt_iter = read_checkpoint_iteration(resume_ckpt)
                if ckpt_iter is not None and ckpt_iter < start_iter:
                    sync_algo_progress(algo, loop_iteration=state.iteration)
                    log.warning(
                        "Checkpoint %s is from iteration %d but training_state is at "
                        "%d; aligned RLlib counters for continuous logging",
                        resume_ckpt.name,
                        ckpt_iter,
                        state.iteration,
                    )
                log.info("Resumed from %s at iter %d", resume_ckpt, start_iter)
        except Exception as exc:
            log.warning(
                "Failed to restore %s (%s); starting fresh",
                resume_ckpt,
                exc,
            )
            start_iter = 0
            state = TrainingState()

    lr_controller = build_dynamic_lr_controller(run_cfg, training_cfg)
    lr_path = lr_state_path(run_dir)
    if will_resume and start_iter > 0:
        lr_controller.load_state(lr_path)
    apply_lr_to_trainer(algo, lr_controller.lr)

    end_iter, limit_mode = resolve_end_iter(
        run_cfg,
        start_iter=start_iter,
        cli_iters=args.iters,
    )
    if start_iter >= end_iter:
        log.warning("Nothing to train from iteration %d (limit=%d)", start_iter, end_iter)
        algo.stop()
        ray.shutdown()
        return 0

    last_render_time = time.time()
    last_checkpoint_time = time.time()
    last_ckpt_iter = -1
    stop_after_iter = False
    shutting_down = False

    def _request_stop(signum: int, _frame: Any) -> None:
        nonlocal stop_after_iter, shutting_down
        if shutting_down:
            log.error("Forced exit")
            raise SystemExit(128 + signum)
        stop_after_iter = True
        shutting_down = True
        log.warning(
            "Stop signal %s — finishing this iteration, then saving checkpoint_latest",
            signum,
        )

    signal.signal(signal.SIGINT, _request_stop)
    signal.signal(signal.SIGTERM, _request_stop)

    log.info(
        "task=%s train=%s run=%s resume=%s start=%d end=%d limit=%s "
        "ray_cpus=%s ray_gpus=%s env_runners=%s chkpt_every=%d chkpt_freq=%ds "
        "render_every=%ds dynamic_lr=%s lr=%.2e",
        args.task,
        args.train,
        run_name,
        start_iter > 0,
        start_iter,
        end_iter,
        limit_mode,
        ray_cfg.get("num_cpus"),
        ray_cfg.get("num_gpus"),
        training_section.get("env_runners", {}).get("num_env_runners"),
        chkpt_every,
        int(chkpt_freq),
        int(render_every),
        run_cfg.get("dynamic_lr", {}).get("enabled", False),
        lr_controller.lr,
    )

    try:
        for i in range(start_iter, end_iter):
            try:
                result = algo.train()
            except KeyboardInterrupt:
                stop_after_iter = True
                shutting_down = True
                log.warning("KeyboardInterrupt — saving checkpoint_latest after current work")
                break

            r = result.get("env_runners", {}).get("episode_return_mean")
            reward = float(r) if r is not None else float("-inf")
            lr = lr_controller.observe(i, reward)
            apply_lr_to_trainer(algo, lr)
            log.info(
                "iter=%d episode_return_mean=%.6g best=%.6g lr=%.2e",
                i,
                reward,
                state.best_reward,
                lr,
            )

            is_new_best = reward >= state.best_reward
            state = TrainingState(
                iteration=int(i),
                best_reward=float(reward if is_new_best else state.best_reward),
                best_iteration=int(i if is_new_best else state.best_iteration),
            )
            now = time.time()
            if chkpt_every > 0 and i % chkpt_every == 0:
                _save_latest_checkpoint(
                    algo,
                    run_dir=run_dir,
                    iteration=i,
                    state=state,
                    lr_controller=lr_controller,
                    lr_path=lr_path,
                )
                last_checkpoint_time = now
                last_ckpt_iter = i
            elif chkpt_freq > 0 and now - last_checkpoint_time >= chkpt_freq:
                _save_latest_checkpoint(
                    algo,
                    run_dir=run_dir,
                    iteration=i,
                    state=state,
                    lr_controller=lr_controller,
                    lr_path=lr_path,
                )
                last_checkpoint_time = now
                last_ckpt_iter = i
            else:
                write_state(
                    run_dir,
                    state,
                    checkpoint_iteration=last_ckpt_iter if last_ckpt_iter >= 0 else None,
                )

            if is_new_best:
                save_checkpoint(algo, best_dir(run_dir), loop_iteration=i)
                lr_controller.save_state(lr_path)

            if stop_after_iter:
                if last_ckpt_iter != i:
                    _save_latest_checkpoint(
                        algo,
                        run_dir=run_dir,
                        iteration=i,
                        state=state,
                        lr_controller=lr_controller,
                        lr_path=lr_path,
                    )
                    last_ckpt_iter = i
                log.info("Training stopped cleanly at iteration %d", i)
                break

            should_render = False
            render_label = f"iter_{i}"
            render_prefer = "latest"

            if render_every > 0 and now - last_render_time >= render_every:
                should_render = True
            if is_new_best and render_on_best and i > 0:
                should_render = True
                render_label = "best"
                render_prefer = "best"

            if should_render:
                ckpt = checkpoint_for_render(run_dir, prefer=render_prefer)
                if ckpt is not None:
                    last_render_time = now
                    _spawn_render(
                        task=args.task,
                        train=args.train,
                        checkpoint=ckpt,
                        run_dir=run_dir,
                        label=render_label,
                    )
                else:
                    log.debug("Skipping render (no checkpoint yet)")

        else:
            _save_latest_checkpoint(
                algo,
                run_dir=run_dir,
                iteration=state.iteration,
                state=state,
                lr_controller=lr_controller,
                lr_path=lr_path,
            )
    finally:
        if stop_after_iter and last_ckpt_iter < state.iteration:
            _save_latest_checkpoint(
                algo,
                run_dir=run_dir,
                iteration=state.iteration,
                state=state,
                lr_controller=lr_controller,
                lr_path=lr_path,
            )
    algo.stop()
    ray.shutdown()
    log.info("Run complete: %s", run_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
