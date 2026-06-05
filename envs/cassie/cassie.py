import logging as log
from collections import OrderedDict
from copy import deepcopy
from pathlib import Path
from typing import TYPE_CHECKING, Any

import cv2
import mujoco as m
import numpy as np
from constants import (
    DEFAULT_CONFIG,
    LEFT_CONTACT_IDX,
    LEFT_FOOT,
    PELVIS,
    RIGHT_CONTACT_IDX,
    RIGHT_FOOT,
    sensors,
)
from functions import (
    mirror_symmetric_action,
    mirror_symmetric_obs,
    p_between_von_mises,
    pelvis_linvel_world,
    quat_to_rpy,
)
from gymnasium.envs.mujoco.mujoco_env import MujocoEnv
from gymnasium.spaces import Box
from health import check_health
from observations import build_observation
from rewards import RewardCalculator

if TYPE_CHECKING:
    import numpy.typing as npt


class CassieEnv(MujocoEnv):
    metadata = {
        "render_modes": ["human", "rgb_array", "depth_array"],
        "render_fps": 0,
    }
    mujoco_dt = 0.0005

    def __init__(
        self,
        env_config: dict[str, Any] | None = None,
        model_dir: str | Path = "assets/model",
    ):
        config = deepcopy(DEFAULT_CONFIG)
        if env_config:
            config.update(env_config)

        # --- Configuration Parameters ---
        self.symmetric_regulation: str = config["symmetric_regulation"]
        assert self.symmetric_regulation in ["alternate", "random", "none"]
        self._terminate_when_unhealthy: bool = config["terminate_when_unhealthy"]
        self._healthy_pelvis_z_range: tuple[float, float] = config["pelvis_height"]
        self._healthy_feet_distance_x: float = config["feet_distance_x"]
        self._healthy_feet_distance_y: float = config["feet_distance_y"]
        self._healthy_feet_distance_z: float = config["feet_distance_z"]
        self._healthy_dis_to_pelvis: float = config["feet_pelvis_height"]
        self._healthy_feet_height: float = config["feet_height"]
        self._require_grounded_after_first_contact: bool = bool(
            config.get("require_grounded_after_first_contact", True)
        )
        self._max_sim_time: int = config["max_simulation_time"]
        self.dt_per_cycle: float = config["dt_per_cycle"]
        self.max_roll: float = config["max_roll"]
        self.max_pitch: float = config["max_pitch"]
        self.max_yaw: float = config["max_yaw"]
        self.training: bool = config["is_training"]
        self._log_step_metrics: bool = config.get("log_step_metrics", not config["is_training"])
        self.r: float = config["r"]
        self.kappa: float = config["kappa"]
        self._command = np.asarray(config["command"], dtype=np.float32)
        if self._command.shape != (2,):
            raise ValueError(f"env_config.command must be length-2 [vx, vy], got {self._command!r}")
        self.model_file: str = config["model"]
        self._reset_noise_scale: float = config["reset_noise_scale"]
        # Velocity reset noise can be tuned separately from position noise;
        # large initial qvel makes static-balance learning extremely hard.
        self._reset_noise_vel_scale: float = float(
            config.get("reset_noise_vel_scale", config["reset_noise_scale"])
        )
        self._force_max_norm: float = config["force_max_norm"]
        self._push_prob_per_second: float = float(config["push_prob_per_second"])
        self._current_push_force: np.ndarray = np.zeros(3, dtype=np.float32)
        self._current_push_full: np.ndarray = np.zeros(6, dtype=np.float64)
        # Low-pass (EMA) filter on actions before they reach MuJoCo.
        # filtered = alpha * new + (1 - alpha) * previous
        # alpha=1.0 disables the filter; lower values smooth more aggressively.
        self.action_filter_alpha: float = float(config.get("action_filter_alpha", 1.0))
        self.render_width: int = config["width"]
        self.render_height: int = config["height"]
        self.sim_fps: int = config["sim_fps"]
        self.local_render_mode: str = config["render_mode"]

        self.reward_coeffs = {k: v for k, v in config.items() if k.startswith("r_")}

        # --- State ---
        self.a_swing: float = 0.0
        self.a_stance: float = self.r
        self.b_swing: float = self.a_stance
        self.b_stance: float = 1.0
        self._pushing: int = -1
        self.phi: float = 0.0
        self.steps: int = 0
        self.previous_action: npt.NDArray[np.float32] = np.zeros(10)
        self.command: npt.NDArray[np.float32] = self._command.copy()
        self.contact: bool = False
        self.symmetric_turn: bool = False
        self.init_rpy: npt.NDArray[np.float32] | None = None
        self.contact_force_left_foot: npt.NDArray[np.float64] = np.zeros(6, dtype=np.float64)
        self.contact_force_right_foot: npt.NDArray[np.float64] = np.zeros(6, dtype=np.float64)
        self.obs: npt.NDArray[np.float32] | None = None
        self._sym_obs_buf = np.zeros(49, dtype=np.float32)
        self._sym_act_buf = np.zeros(10, dtype=np.float32)
        # Cached sensor readings (refreshed once per physics step). Avoids
        # repeated string lookups into the model when the same data is
        # consumed by reward, health, and observation builders.
        self._sensor_cache: dict[str, np.ndarray] | None = None
        self._sensor_name_lookup: dict[str, list] | None = None

        # Used by health checks (set after check_health call)
        self.done_n: float = 0.0
        self.isdone: str = "not done"

        # --- Observation Space ---
        observation_space_dict = OrderedDict(
            [
                ("actuatorpos", Box(-180.0, 180.0, shape=(10,))),
                ("jointpos", Box(-180.0, 180.0, shape=(6,))),
                ("framequat", Box(-1.0, 1.0, shape=(4,))),
                ("gyro", Box(-np.inf, np.inf, shape=(3,))),
                ("accelerometer", Box(-np.inf, np.inf, shape=(3,))),
                ("magnetometer", Box(-np.inf, np.inf, shape=(3,))),
                ("command", Box(-np.inf, np.inf, shape=(2,))),
                ("contact_forces", Box(-np.inf, np.inf, shape=(6,))),
                ("clock", Box(-1.0, 1.0, shape=(2,))),
                ("previous_action", Box(-np.inf, np.inf, shape=(10,))),
            ]
        )
        _obs_low = np.concatenate([s.low for s in observation_space_dict.values()])
        _obs_high = np.concatenate([s.high for s in observation_space_dict.values()])
        observation_space = Box(low=_obs_low, high=_obs_high, dtype=np.float32)

        # --- MujocoEnv Initialization ---
        frame_skip = int((1.0 / self.sim_fps) // self.mujoco_dt)
        self.metadata["render_fps"] = int(np.round(1 / self.mujoco_dt / frame_skip))

        MujocoEnv.__init__(
            self,
            model_path=str(Path(model_dir).absolute() / f"{self.model_file}.xml"),
            frame_skip=frame_skip,
            render_mode="rgb_array",
            observation_space=observation_space,
            width=self.render_width,
            height=self.render_height,
        )

        self.action_space = Box(
            self.action_space.low.astype(np.float32),
            self.action_space.high.astype(np.float32),
            dtype=np.float32,
        )

        # --- Post-Init ---
        self.steps_per_cycle = int(self.dt_per_cycle / self.dt)
        if self.steps_per_cycle <= 0:
            log.warning("steps_per_cycle calculated as %d, setting to 1", self.steps_per_cycle)
            self.steps_per_cycle = 1

        self._push_duration: int = int(config["push_duration"] / self.dt)

        # Precompute Von Mises values
        phis = np.linspace(0, 1, self.steps_per_cycle, endpoint=False)
        self.von_mises_values_swing = np.array(
            [
                p_between_von_mises(a=self.a_swing, b=self.b_swing, kappa=self.kappa, x=p)
                for p in phis
            ],
            dtype=np.float32,
        )
        self.von_mises_values_stance = np.array(
            [
                p_between_von_mises(a=self.a_stance, b=self.b_stance, kappa=self.kappa, x=p)
                for p in phis
            ],
            dtype=np.float32,
        )

        self._reward_calc = RewardCalculator(
            reward_coeffs=self.reward_coeffs,
            cmd_scale=np.abs(self._command),
            action_space_high=self.action_space.high,
            action_space_low=self.action_space.low,
            steps_per_cycle=self.steps_per_cycle,
            von_mises_values_swing=self.von_mises_values_swing,
            von_mises_values_stance=self.von_mises_values_stance,
        )

    quat_to_rpy = staticmethod(quat_to_rpy)

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def total_simulation_time(self):
        return self.steps * self.dt

    @property
    def sensor_data(self) -> dict[str, np.ndarray]:
        if self._sensor_cache is None:
            self._refresh_sensor_cache()
        return self._sensor_cache

    def _refresh_sensor_cache(self) -> None:
        """Recompute the per-step sensor dictionary.

        Resolves sensor name -> sensor object lookups exactly once and
        thereafter reuses the cached objects, avoiding the expensive
        string-keyed lookup into the MuJoCo model on every step.
        """
        if self._sensor_name_lookup is None:
            self._sensor_name_lookup = {
                group: [self.data.sensor(s) for s in sensors[group]] for group in sensors.keys()
            }
        self._sensor_cache = {
            group: np.concatenate([s.data for s in sensor_objs])
            for group, sensor_objs in self._sensor_name_lookup.items()
        }

    def _foot_in_contact(self, foot_body: int, contact_force: np.ndarray) -> bool:
        return (
            self.data.xpos[foot_body, 2] < self._healthy_feet_height
            or np.linalg.norm(contact_force) > 100
        )

    @property
    def left_foot_contact(self) -> bool:
        return self._foot_in_contact(LEFT_FOOT, self.contact_force_left_foot)

    @property
    def right_foot_contact(self) -> bool:
        return self._foot_in_contact(RIGHT_FOOT, self.contact_force_right_foot)

    @property
    def foot_contact(self) -> bool:
        return self.left_foot_contact or self.right_foot_contact

    @property
    def is_healthy(self):
        pelvis_rpy = self.quat_to_rpy(self.sensor_data["framequat"])
        healthy, reason, code = check_health(
            data=self.data,
            pelvis_z_range=self._healthy_pelvis_z_range,
            feet_distance_x=self._healthy_feet_distance_x,
            feet_distance_y=self._healthy_feet_distance_y,
            feet_distance_z=self._healthy_feet_distance_z,
            dis_to_pelvis=self._healthy_dis_to_pelvis,
            max_roll=self.max_roll,
            max_pitch=self.max_pitch,
            max_yaw=self.max_yaw,
            contact=self.contact,
            init_rpy=self.init_rpy,
            pelvis_rpy=pelvis_rpy,
            left_foot_contact=self.left_foot_contact,
            right_foot_contact=self.right_foot_contact,
            require_grounded_after_first_contact=self._require_grounded_after_first_contact,
        )
        self.isdone = reason
        self.done_n = code
        return healthy

    @property
    def terminated(self):
        return (not self.is_healthy) if self._terminate_when_unhealthy else False

    @property
    def truncated(self):
        return self.total_simulation_time > self._max_sim_time

    # ------------------------------------------------------------------
    # Symmetric regulation
    # ------------------------------------------------------------------

    def update_symmetric_turn(self):
        if self.symmetric_regulation == "alternate":
            self.symmetric_turn ^= True
        elif self.symmetric_regulation == "random":
            self.symmetric_turn = self.np_random.random() < 0.5
        elif self.symmetric_regulation == "none":
            self.symmetric_turn = False

    def _set_obs(self) -> None:
        self.obs = build_observation(
            sensor_data=self.sensor_data,
            command=self.command,
            contact_force_left_foot=self.contact_force_left_foot,
            contact_force_right_foot=self.contact_force_right_foot,
            phi=self.phi,
            previous_action=self.previous_action,
        )

    # ------------------------------------------------------------------
    # Step
    # ------------------------------------------------------------------

    def step(
        self, action: "npt.NDArray[np.float32]"
    ) -> tuple["npt.NDArray[np.float32]", float, bool, bool, dict[str, Any]]:
        self.update_symmetric_turn()
        if self.symmetric_turn:
            mirror_symmetric_action(self._sym_act_buf, action)
            act = self._sym_act_buf
        else:
            act = action
        # Low-pass filter the action in motor-units before it reaches MuJoCo.
        if self.action_filter_alpha < 1.0:
            act = (
                self.action_filter_alpha * act
                + (1.0 - self.action_filter_alpha) * self.previous_action
            ).astype(act.dtype)
        self.do_simulation(act, self.frame_skip)

        # Refresh sensor cache once per step; reused by reward + obs below.
        self._refresh_sensor_cache()

        # Feet contact forces
        self._update_contact_forces()

        if self.foot_contact:
            self.contact = True

        terminated = self.terminated

        sensors_now = self._sensor_cache
        if self._reward_calc.use_foot_pitch:
            left_foot_rpy = self.quat_to_rpy(np.ascontiguousarray(self.data.xquat[LEFT_FOOT]))
            right_foot_rpy = self.quat_to_rpy(np.ascontiguousarray(self.data.xquat[RIGHT_FOOT]))
        else:
            left_foot_rpy = None
            right_foot_rpy = None

        reward, metrics = self._reward_calc.compute(
            action=act,
            previous_action=self.previous_action,
            qvel=self.data.qvel,
            pelvis_linvel_world=pelvis_linvel_world(self.data.cvel, PELVIS),
            command=self.command,
            pelvis_quat=sensors_now["framequat"],
            pelvis_ang_vel=sensors_now["gyro"],
            left_foot_rpy=left_foot_rpy,
            right_foot_rpy=right_foot_rpy,
            contact_force_left=self.contact_force_left_foot,
            contact_force_right=self.contact_force_right_foot,
            phi=self.phi,
            training=self.training,
        )

        self._set_obs()
        if self.symmetric_turn:
            mirror_symmetric_obs(self._sym_obs_buf, self.obs)
            observation = self._sym_obs_buf
        else:
            observation = self.obs

        self.steps += 1
        self.phi = (self.phi + 1.0 / self.steps_per_cycle) % 1
        self.previous_action = act

        info = self._build_step_info(metrics, reward)

        # Random push perturbation
        self._apply_push()

        self.obs = observation
        return observation, reward, terminated, self.truncated, info

    # ------------------------------------------------------------------
    # Reset
    # ------------------------------------------------------------------

    def reset_model(self, seed: int = None) -> "npt.NDArray[np.float32]":
        self.done_n = 0.0
        noise_low = -self._reset_noise_scale
        noise_high = self._reset_noise_scale

        self.previous_action = np.zeros(10, dtype=np.float32)
        self.contact_force_left_foot = np.zeros(6, dtype=np.float64)
        self.contact_force_right_foot = np.zeros(6, dtype=np.float64)
        self.phi = 0.0
        self.steps = 0
        self.contact = False
        self.command = self._command.copy()
        self._pushing = -1
        # Reset the symmetric-regulation flag so episodes don't start mid-toggle
        # when the previous one ended on an odd step. This keeps the per-episode
        # parity unbiased.
        self.symmetric_turn = False

        qpos = self.init_qpos + self.np_random.uniform(
            low=noise_low, high=noise_high, size=self.model.nq
        )
        qvel = self.init_qvel + self.np_random.uniform(
            low=-self._reset_noise_vel_scale,
            high=self._reset_noise_vel_scale,
            size=self.model.nv,
        )

        self.set_state(qpos, qvel)
        self.data.xfrc_applied[PELVIS] = np.zeros(6)
        # set_state already calls mj_forward; no need to call it again.

        self._refresh_sensor_cache()
        self.init_rpy = self.quat_to_rpy(self.sensor_data["framequat"])
        self._set_obs()

        if self.obs is None:
            raise RuntimeError("Observation was not set during reset.")

        self.update_symmetric_turn()
        if self.symmetric_turn:
            mirror_symmetric_obs(self._sym_obs_buf, self.obs)
            return self._sym_obs_buf
        return self.obs

    # ------------------------------------------------------------------
    # Render
    # ------------------------------------------------------------------

    def render(self) -> "bool | npt.NDArray[np.uint8]":
        frame = super().render()

        if self.local_render_mode == "rgb_array":
            return frame

        height, width = frame.shape[:2]
        aspect_ratio = width / height
        initial_width = int(720 * aspect_ratio)
        cv2.imshow("Cassie", cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        cv2.resizeWindow("Cassie", initial_width, 720)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q") or key == 27:
            self.is_running = False
            cv2.destroyAllWindows()
            return None
        elif key == ord("r"):
            cv2.resizeWindow("Cassie", 800, 600)

        return True

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _update_contact_forces(self):
        # Reuse pre-allocated buffers; mj_contactForce writes into result in-place.
        self.contact_force_left_foot.fill(0.0)
        self.contact_force_right_foot.fill(0.0)

        ncon = self.data.ncon
        if ncon == 0:
            return

        # Vectorized access: data.contact is a struct array; .geom2 returns a
        # numpy view of length ncon (no Python list construction).
        geom2 = self.data.contact.geom2[:ncon]

        left_idx = np.where(geom2 == LEFT_CONTACT_IDX)[0]
        if left_idx.size:
            m.mj_contactForce(
                self.model,
                self.data,
                int(left_idx[0]),
                self.contact_force_left_foot,
            )

        right_idx = np.where(geom2 == RIGHT_CONTACT_IDX)[0]
        if right_idx.size:
            m.mj_contactForce(
                self.model,
                self.data,
                int(right_idx[0]),
                self.contact_force_right_foot,
            )

    def _build_step_info(self, metrics: dict, reward: float) -> dict:
        if self.training and not self._log_step_metrics:
            return {}

        info: dict[str, Any] = {}
        info["custom_metrics"] = {
            "distance": self.data.qpos[0],
            "height": self.data.qpos[2],
        }

        if "rewards" in metrics:
            for k, v in metrics["rewards"].items():
                info["custom_metrics"][k] = v
        if "coefficients" in metrics:
            for k, v in metrics["coefficients"].items():
                info["custom_metrics"][k] = v

        exp_keys = ["q_left_frc", "q_right_frc", "q_left_spd", "q_right_spd"]
        if "after_exponential" in metrics:
            for key in exp_keys:
                if key in metrics["after_exponential"]:
                    info["custom_metrics"][f"exp_{key}"] = metrics["after_exponential"][key]

        # NOTE: previously stored the full ``metrics`` dict here, but it was
        # serialized through Ray on every step from every worker -- a major
        # throughput killer. The reward components above already cover what
        # the callbacks need.
        return info

    def _apply_push(self):
        if self._push_prob_per_second <= 0.0 and self._pushing < 0:
            self._current_push_force[:] = 0.0
            return

        p_step = min(max(self._push_prob_per_second * float(self.dt), 0.0), 1.0)

        if np.random.uniform(0, 1) < p_step and self._pushing == -1:
            self._pushing = 0
            # Sample a single force vector for the entire push duration.
            random_force_xy = np.random.uniform(-self._force_max_norm, self._force_max_norm, size=2)
            self._current_push_full[:] = 0.0
            self._current_push_full[0:2] = random_force_xy
            self._current_push_force[:] = self._current_push_full[:3]

        if -1 < self._pushing < self._push_duration:
            # Apply the same sampled force for the whole active push window.
            self.data.xfrc_applied[PELVIS] = self._current_push_full
            self._pushing += 1
        else:
            self._pushing = -1
            self._current_push_force[:] = 0.0
            self._current_push_full[:] = 0.0
