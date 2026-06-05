from __future__ import annotations

import logging
from copy import deepcopy
from pathlib import Path
from typing import Any

import gymnasium as gym
import mediapy as media
import numpy as np
import ray
from ray.rllib.algorithms.ppo import PPO
from ray.rllib.connectors.agent.mean_std_filter import MeanStdObservationFilterAgentConnector
from ray.rllib.utils.typing import ActionConnectorDataType, AgentConnectorDataType

from droprl.envs.base import BaseEnv
from droprl.envs.registry import load_env
from droprl.rllib.env_wrapper import RllibGymWrapper
from droprl.rllib.loader import build_ppo_trainer, load_policy_artifacts
from droprl.rllib.registry import register_rllib_envs
from droprl.runs.checkpoints import is_usable_checkpoint

log = logging.getLogger(__name__)


def _to_gym_env(env_obj: Any) -> gym.Env:
    if isinstance(env_obj, gym.Env):
        return env_obj
    if isinstance(env_obj, BaseEnv):
        return RllibGymWrapper(env_obj)
    raise TypeError(f"Unsupported env type for render: {type(env_obj).__name__}")


def _obs_filter(policy) -> MeanStdObservationFilterAgentConnector | None:
    connectors = getattr(policy, "agent_connectors", None)
    if connectors is None:
        return None
    found = connectors[MeanStdObservationFilterAgentConnector]
    return found[0] if found else None


def _apply_obs_filter(filter_conn: MeanStdObservationFilterAgentConnector | None, obs):
    if filter_conn is None:
        return obs
    acd = AgentConnectorDataType(env_id="eval", agent_id=0, data={"obs": obs})
    return filter_conn([acd])[0].data["obs"]


def _apply_action_connectors(policy, raw_action, states, fetches):
    action_conns = getattr(policy, "action_connectors", None)
    if not action_conns:
        return raw_action
    ac_data = ActionConnectorDataType(
        env_id="eval",
        agent_id=0,
        input_dict={},
        output=(raw_action, states, fetches),
    )
    return action_conns(ac_data).output[0]


def rollout_frames(
    trainer: PPO,
    env: gym.Env,
    *,
    max_steps: int,
    render_fps: int,
    sim_fps: int,
) -> list[np.ndarray]:
    policy = trainer.get_policy()
    obs_filter = _obs_filter(policy)
    sim_steps_per_frame = max(1, int(round(sim_fps / float(render_fps))))

    obs, _ = env.reset()
    terminated = truncated = False
    frames: list[np.ndarray] = []
    step = 0
    next_capture = 0

    while not (terminated or truncated) and step < max_steps:
        obs_in = _apply_obs_filter(obs_filter, obs)
        raw_action, states, fetches = policy.compute_single_action(obs_in, explore=False)
        action = _apply_action_connectors(policy, raw_action, states, fetches)
        obs, _reward, terminated, truncated, _info = env.step(action)

        if step >= next_capture:
            frame = env.render()
            if isinstance(frame, np.ndarray) and frame.size > 0:
                frames.append(frame)
            next_capture += sim_steps_per_frame
        step += 1

    return frames


def render_checkpoint(
    *,
    config: dict[str, Any],
    env_name: str,
    checkpoint: Path | str,
    output_path: Path | str,
    init_ray: bool = True,
) -> Path | None:
    checkpoint = Path(checkpoint)
    if not is_usable_checkpoint(checkpoint):
        raise FileNotFoundError(f"Not a usable checkpoint: {checkpoint}")

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    run_cfg = config.get("run", {})
    render_fps = int(run_cfg.get("render_fps", 30))
    max_steps = int(run_cfg.get("render_max_steps", 10_000))

    env_cfg = deepcopy((config.get("env") or {}).get("config") or {})
    env_cfg["is_training"] = False
    sim_fps = int(env_cfg.get("sim_fps") or run_cfg.get("sim_fps") or render_fps)

    if init_ray:
        from droprl.rllib.runtime import build_runtime_env

        ray.init(
            ignore_reinit_error=True,
            num_cpus=1,
            include_dashboard=False,
            runtime_env=build_runtime_env(env_name),
        )

    training_section = deepcopy(config.get("training", {}))
    training_section.setdefault("environment", {})
    env_map = register_rllib_envs(tasks=[env_name])
    if env_name not in env_map:
        raise ValueError(f"Unknown env '{env_name}'. Available: {sorted(env_map)}")
    training_section["environment"]["env"] = env_map[env_name]
    training_section["environment"]["env_config"] = env_cfg

    trainer = build_ppo_trainer(training_section)
    load_policy_artifacts(trainer, checkpoint)
    env = _to_gym_env(load_env(env_name, env_cfg))
    if hasattr(env, "render_mode"):
        env.render_mode = "rgb_array"

    frames = rollout_frames(
        trainer,
        env,
        max_steps=max_steps,
        render_fps=render_fps,
        sim_fps=sim_fps,
    )
    env.close()
    trainer.stop()

    if not frames:
        log.warning("No frames captured for env '%s'; skipping video write", env_name)
        return None

    media.write_video(output_path, frames, fps=render_fps)
    log.info("Render saved: %s", output_path.resolve())
    return output_path
