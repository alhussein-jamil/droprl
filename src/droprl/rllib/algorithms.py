from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any

from ray.rllib.algorithms.algorithm import Algorithm
from ray.rllib.algorithms.algorithm_config import AlgorithmConfig
from ray.rllib.algorithms.dqn import DQNConfig
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.algorithms.sac import SACConfig

from droprl.rllib.resources import resolve_training_section
from droprl.tasks.extensions import discover_callbacks

log = logging.getLogger(__name__)

DEFAULT_ALGORITHM = "ppo"

_ALGORITHM_CONFIGS: dict[str, type[AlgorithmConfig]] = {
    "ppo": PPOConfig,
    "sac": SACConfig,
    "dqn": DQNConfig,
}


def list_algorithms() -> list[str]:
    return sorted(_ALGORITHM_CONFIGS)


def resolve_algorithm(config: dict[str, Any]) -> str:
    """Return normalized algorithm id from merged experiment config."""
    name = str(config.get("algorithm", DEFAULT_ALGORITHM)).strip().lower()
    if name not in _ALGORITHM_CONFIGS:
        raise ValueError(f"Unknown algorithm '{name}'. Supported: {', '.join(list_algorithms())}")
    return name


def default_train_name(task: str) -> str:
    """Default train config stem for a task (e.g. ``cassie`` → ``Cassie``)."""
    return task[:1].upper() + task[1:] if task else task


def build_algorithm(
    config: dict[str, Any],
    *,
    task: str,
    logger_creator: Callable[..., Any] | None = None,
) -> Algorithm:
    """Build an RLlib Algorithm from merged experiment config."""
    algorithm = resolve_algorithm(config)
    training = resolve_training_section(config.get("training", {}))
    env_id = training.get("environment", {}).get("env")
    if not env_id:
        raise ValueError("config.training.environment.env must be set to the RLlib env id.")

    config_cls = _ALGORITHM_CONFIGS[algorithm]
    builder = (
        config_cls()
        .api_stack(
            enable_rl_module_and_learner=False,
            enable_env_runner_and_connector_v2=False,
        )
        .environment(**training.get("environment", {}))
        .env_runners(**training.get("env_runners", {}))
        .training(**training.get("training", {}))
        .framework(**training.get("framework", {}))
        .resources(**training.get("resources", {}))
    )

    fault_tolerance = training.get("fault_tolerance")
    if fault_tolerance:
        builder = builder.fault_tolerance(**fault_tolerance)

    if logger_creator is not None:
        builder = builder.debugging(logger_creator=logger_creator)

    callbacks_cls = discover_callbacks(task)
    if callbacks_cls is not None:
        builder = builder.callbacks(callbacks_class=callbacks_cls)

    log.info("Building RLlib algorithm '%s' for task '%s'", algorithm, task)
    return builder.build_algo()
