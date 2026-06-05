from __future__ import annotations

import logging
import os
from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np
from ray.rllib.algorithms.ppo import PPO, PPOConfig

log = logging.getLogger(__name__)

POLICY_WEIGHTS_FILE = "policy_weights.npz"


def policy_weights_path(checkpoint_path: Path | str) -> Path:
    return Path(checkpoint_path) / POLICY_WEIGHTS_FILE


def is_rllib_checkpoint(checkpoint_path: Path | str) -> bool:
    ckpt = Path(checkpoint_path)
    return (ckpt / "rllib_checkpoint.json").is_file() or (ckpt / "policies").is_dir()


def _eval_env_runners_config(env_runners: dict[str, Any]) -> dict[str, Any]:
    cfg = deepcopy(env_runners)
    cfg.update(
        num_env_runners=1,
        num_envs_per_env_runner=1,
        num_gpus_per_env_runner=0,
    )
    if "rollout_fragment_length" not in cfg:
        cfg["rollout_fragment_length"] = 200
    return cfg


def save_policy_weights(trainer: PPO, checkpoint_path: Path | str) -> None:
    path = policy_weights_path(checkpoint_path)
    weights = trainer.get_policy().get_weights()
    arrays = {k: np.asarray(v) for k, v in weights.items()}
    np.savez_compressed(path, **arrays)
    log.info("Policy weights exported to %s", path)


def load_ppo_trainer(training_section: dict[str, Any], checkpoint_path: Path | str) -> PPO:
    section = deepcopy(training_section)
    section["framework"] = {"framework": "torch"}
    env_runners = _eval_env_runners_config(section.get("env_runners", {}))

    trainer = (
        PPOConfig()
        .api_stack(
            enable_rl_module_and_learner=False,
            enable_env_runner_and_connector_v2=False,
        )
        .environment(**section.get("environment", {}))
        .env_runners(**env_runners)
        .training(**section.get("training", {}))
        .framework(**section.get("framework", {}))
        .resources(num_gpus=0)
        .build_algo()
    )

    ckpt = Path(checkpoint_path)
    npz = policy_weights_path(ckpt)

    if is_rllib_checkpoint(ckpt):
        trainer.restore(os.path.abspath(ckpt))
        log.info("Restored RLlib checkpoint from %s", ckpt.resolve())
        if npz.is_file():
            weights = {k: arr for k, arr in np.load(npz).items()}
            trainer.get_policy().set_weights(weights)
            log.info("Applied policy weights from %s", npz)
        return trainer

    if npz.is_file():
        weights = {k: arr for k, arr in np.load(npz).items()}
        trainer.get_policy().set_weights(weights)
        log.warning(
            "Loaded weights only from %s; observation filters may be wrong if used in training",
            npz,
        )
        return trainer

    trainer.restore(os.path.abspath(ckpt))
    return trainer
