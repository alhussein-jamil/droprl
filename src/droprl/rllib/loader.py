from __future__ import annotations

import json
import logging
from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np
from ray.rllib.algorithms.ppo import PPO, PPOConfig
from ray.rllib.connectors.agent.mean_std_filter import MeanStdObservationFilterAgentConnector
from ray.rllib.utils.filter import RunningStat

log = logging.getLogger(__name__)

POLICY_WEIGHTS_FILE = "policy_weights.npz"
OBS_FILTER_FILE = "obs_filter.json"


def policy_weights_path(checkpoint_path: Path | str) -> Path:
    return Path(checkpoint_path) / POLICY_WEIGHTS_FILE


def obs_filter_path(checkpoint_path: Path | str) -> Path:
    return Path(checkpoint_path) / OBS_FILTER_FILE


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


def build_ppo_trainer(training_section: dict[str, Any]) -> PPO:
    """Build a single-env-runner PPO trainer for eval/render."""
    section = deepcopy(training_section)
    section["framework"] = {"framework": "torch"}
    env_runners = _eval_env_runners_config(section.get("env_runners", {}))

    return (
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


def _json_safe(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {k: _json_safe(v) for k, v in value.items()}
    if isinstance(value, list | tuple):
        return [_json_safe(v) for v in value]
    return value


def _obs_filter_connectors(trainer: PPO) -> list[MeanStdObservationFilterAgentConnector]:
    policy = trainer.get_policy()
    connectors = getattr(policy, "agent_connectors", None)
    if connectors is None:
        return []
    return connectors[MeanStdObservationFilterAgentConnector]


def _save_obs_filter(trainer: PPO, path: Path) -> None:
    connectors = _obs_filter_connectors(trainer)
    if not connectors:
        return
    _name, params = connectors[0].to_state()
    path.write_text(json.dumps(_json_safe(params), indent=2))


def _load_obs_filter(trainer: PPO, path: Path) -> None:
    connectors = _obs_filter_connectors(trainer)
    if not connectors:
        if path.is_file():
            log.warning("Found %s but policy has no observation filter connector", path)
        return
    if not path.is_file():
        return

    params = json.loads(path.read_text())
    connector = connectors[0]
    connector.filter.shape = params["shape"]
    connector.filter.no_preprocessor = params["no_preprocessor"]
    connector.filter.demean = params["demean"]
    connector.filter.destd = params["destd"]
    connector.filter.clip = params["clip"]

    import tree

    running_stats = [RunningStat.from_state(s) for s in params["running_stats"]]
    connector.filter.running_stats = tree.unflatten_as(connector.filter.shape, running_stats)
    buffer = [RunningStat.from_state(s) for s in params["buffer"]]
    connector.filter.buffer = tree.unflatten_as(connector.filter.shape, buffer)
    log.info("Restored observation filter from %s", path)


def save_policy_artifacts(trainer: PPO, checkpoint_path: Path | str) -> None:
    """Export portable policy weights and observation-filter state."""
    ckpt = Path(checkpoint_path)
    weights = trainer.get_policy().get_weights()
    arrays = {k: np.asarray(v) for k, v in weights.items()}
    np.savez_compressed(policy_weights_path(ckpt), **arrays)
    log.info("Policy weights exported to %s", policy_weights_path(ckpt))
    _save_obs_filter(trainer, obs_filter_path(ckpt))


def load_policy_artifacts(trainer: PPO, checkpoint_path: Path | str) -> None:
    """Load portable artifacts for eval/render (no RLlib pickle restore)."""
    ckpt = Path(checkpoint_path)
    npz = policy_weights_path(ckpt)
    if not npz.is_file():
        raise FileNotFoundError(
            f"Missing {POLICY_WEIGHTS_FILE} in {ckpt}. "
            "Re-save checkpoints from a current DropRL training run."
        )

    weights = {k: arr for k, arr in np.load(npz).items()}
    trainer.get_policy().set_weights(weights)
    log.info("Loaded policy weights from %s", npz)
    _load_obs_filter(trainer, obs_filter_path(ckpt))
