import logging as log
from typing import TYPE_CHECKING, Any

import numba as nb
import numpy as np
from constants import (
    FORWARD_QUARTERNIONS,
    OMEGA,
    THETA_LEFT,
    THETA_RIGHT,
    c_stance_spd,
    c_swing_frc,
    gravity,
    mass,
)
from functions import action_dist

if TYPE_CHECKING:
    import numpy.typing as npt

REWARD_COMPONENT_KEYS = ("r_biped", "r_cmd", "r_smooth", "r_feet_parallel")
_EXP_INFO_KEYS = ("q_left_frc", "q_right_frc", "q_left_spd", "q_right_spd")


class RewardCalculator:
    """Handles all reward computation for the Cassie environment."""

    def __init__(
        self,
        reward_coeffs: dict[str, float],
        cmd_scale: "npt.NDArray[np.float32]",
        action_space_high: "npt.NDArray[np.float32]",
        action_space_low: "npt.NDArray[np.float32]",
        steps_per_cycle: int,
        von_mises_values_swing: "npt.NDArray[np.float32]",
        von_mises_values_stance: "npt.NDArray[np.float32]",
    ):
        self.reward_coeffs = reward_coeffs
        self.action_space_high = action_space_high
        self.action_space_low = action_space_low
        self.steps_per_cycle = steps_per_cycle
        self.von_mises_values_swing = von_mises_values_swing
        self.von_mises_values_stance = von_mises_values_stance
        self.use_foot_pitch = reward_coeffs.get("r_feet_parallel", 0.0) != 0.0
        self._reward_values = np.zeros(len(REWARD_COMPONENT_KEYS), dtype=np.float32)
        self._coeff_values = np.array(
            [reward_coeffs.get(k, 0.0) for k in REWARD_COMPONENT_KEYS],
            dtype=np.float32,
        )

        torque_max_norm = float(np.linalg.norm(action_space_high))
        action_dist_max = float(np.sqrt(action_space_high.size))
        self.exponents_ranges = {
            "q_vx": (0.0, max(float(cmd_scale[0]), 0.5)),
            "q_vy": (0.0, max(float(cmd_scale[1]), 0.5)),
            "q_left_frc": (0.0, gravity * mass),
            "q_right_frc": (0.0, gravity * mass),
            "q_left_spd": (0.0, 40.0),
            "q_right_spd": (0.0, 40.0),
            "q_action": (0.0, action_dist_max),
            "q_pelvis_acc": (0.0, 25.0),
            "q_orientation": (0.0, 1.2),
            "q_torque": (0.0, torque_max_norm),
            "q_left_foot_pitch": (0.0, np.pi / 2),
            "q_right_foot_pitch": (0.0, np.pi / 2),
        }

    def _normalize_quantity(self, name: str, q: float) -> float:
        lo, hi = self.exponents_ranges[name]
        rng = hi - lo + 1e-6
        return np.clip((q - lo) / rng, 0.0, 1.0)

    def _exp(self, name: str, q: float) -> float:
        return np.exp(-OMEGA * self._normalize_quantity(name, q))

    @staticmethod
    @nb.jit(nopython=True, cache=True)
    def _weighted_sum(
        rewards: "npt.NDArray[np.float32]", coeffs: "npt.NDArray[np.float32]"
    ) -> float:
        return np.sum(rewards * coeffs) / np.sum(coeffs)

    def normalize_reward(self, rewards: dict[str, float]) -> float:
        for i, key in enumerate(REWARD_COMPONENT_KEYS):
            self._reward_values[i] = rewards[key]

        if np.sum(self._coeff_values) == 0:
            log.warning("Sum of reward coefficients is zero, returning raw sum.")
            return float(np.sum(self._reward_values))

        return self._weighted_sum(self._reward_values, self._coeff_values)

    def _phase_coeff(self, phi: float, values: "npt.NDArray[np.float32]", scale: float) -> float:
        idx = int(round((phi % 1.0) * self.steps_per_cycle)) % self.steps_per_cycle
        return scale * values[idx]

    def _phase_coefficient_force(self, phi: float) -> float:
        return self._phase_coeff(phi, self.von_mises_values_swing, c_swing_frc)

    def _phase_coefficient_speed(self, phi: float) -> float:
        return self._phase_coeff(phi, self.von_mises_values_stance, c_stance_spd)

    def compute(
        self,
        action: "npt.NDArray[np.float32]",
        previous_action: "npt.NDArray[np.float32]",
        qvel: "npt.NDArray[np.float64]",
        pelvis_linvel_world: "npt.NDArray[np.float64]",
        command: "npt.NDArray[np.float32]",
        pelvis_quat: "npt.NDArray[np.float32]",
        pelvis_ang_vel: "npt.NDArray[np.float32]",
        left_foot_rpy: "npt.NDArray[np.float32] | None",
        right_foot_rpy: "npt.NDArray[np.float32] | None",
        contact_force_left: "npt.NDArray[np.float32]",
        contact_force_right: "npt.NDArray[np.float32]",
        phi: float,
        *,
        training: bool = False,
    ) -> tuple[float, dict[str, Any]]:
        left_frc_norm = float(np.linalg.norm(contact_force_left))
        right_frc_norm = float(np.linalg.norm(contact_force_right))

        exp_vx = self._exp("q_vx", abs(pelvis_linvel_world[0] - command[0]))
        exp_vy = self._exp("q_vy", abs(pelvis_linvel_world[1] - command[1]))
        exp_left_frc = self._exp("q_left_frc", left_frc_norm)
        exp_right_frc = self._exp("q_right_frc", right_frc_norm)
        exp_left_spd = self._exp("q_left_spd", abs(qvel[12]))
        exp_right_spd = self._exp("q_right_spd", abs(qvel[25]))
        exp_action = self._exp(
            "q_action",
            action_dist(
                action.reshape(1, -1),
                previous_action.reshape(1, -1),
                self.action_space_high,
                self.action_space_low,
            )[0],
        )
        exp_pelvis_acc = self._exp("q_pelvis_acc", np.linalg.norm(pelvis_ang_vel))
        exp_torque = self._exp("q_torque", np.linalg.norm(action))
        exp_orientation = self._exp(
            "q_orientation", np.linalg.norm(pelvis_quat - FORWARD_QUARTERNIONS)
        )

        if self.use_foot_pitch and left_foot_rpy is not None and right_foot_rpy is not None:
            left_pitch = abs(left_foot_rpy[1]) if left_frc_norm > 0.01 else 0.0
            right_pitch = abs(right_foot_rpy[1]) if right_frc_norm > 0.01 else 0.0
            exp_left_pitch = self._exp("q_left_foot_pitch", left_pitch)
            exp_right_pitch = self._exp("q_right_foot_pitch", right_pitch)
        else:
            exp_left_pitch = 1.0
            exp_right_pitch = 1.0

        c_frc = self._phase_coefficient_force
        c_spd = self._phase_coefficient_speed

        r_cmd = (exp_vx + 0.8 * exp_vy + 0.2 * exp_orientation) / 2.0
        r_smooth = (exp_action + 0.5 * exp_torque + 0.5 * exp_pelvis_acc) / 2.0
        r_biped = (
            c_frc(phi + THETA_LEFT) * exp_left_frc
            + c_frc(phi + THETA_RIGHT) * exp_right_frc
            + c_spd(phi + THETA_LEFT) * exp_left_spd
            + c_spd(phi + THETA_RIGHT) * exp_right_spd
        ) / 2.0
        r_feet_parallel = (exp_left_pitch + exp_right_pitch) / 2.0

        rewards: dict[str, float] = {
            "r_biped": r_biped,
            "r_cmd": r_cmd,
            "r_smooth": r_smooth,
            "r_feet_parallel": r_feet_parallel,
        }

        total_reward = self.normalize_reward(rewards)
        total_reward += self.reward_coeffs.get("r_bias", 0.0)

        if training:
            return total_reward, {"rewards": rewards}

        coeff = {
            "c_frc_left": c_frc(phi + THETA_LEFT),
            "c_frc_right": c_frc(phi + THETA_RIGHT),
            "c_spd_left": c_spd(phi + THETA_LEFT),
            "c_spd_right": c_spd(phi + THETA_RIGHT),
        }
        after_exponential = {
            "q_vx": exp_vx,
            "q_vy": exp_vy,
            "q_left_frc": exp_left_frc,
            "q_right_frc": exp_right_frc,
            "q_left_spd": exp_left_spd,
            "q_right_spd": exp_right_spd,
            "q_action": exp_action,
            "q_pelvis_acc": exp_pelvis_acc,
            "q_torque": exp_torque,
            "q_orientation": exp_orientation,
            "q_left_foot_pitch": exp_left_pitch,
            "q_right_foot_pitch": exp_right_pitch,
        }
        raw_quantities = {
            "q_vx": abs(pelvis_linvel_world[0] - command[0]),
            "q_vy": abs(pelvis_linvel_world[1] - command[1]),
            "q_left_frc": left_frc_norm,
            "q_right_frc": right_frc_norm,
            "q_left_spd": abs(qvel[12]),
            "q_right_spd": abs(qvel[25]),
            "q_action": action_dist(
                action.reshape(1, -1),
                previous_action.reshape(1, -1),
                self.action_space_high,
                self.action_space_low,
            )[0],
            "q_pelvis_acc": np.linalg.norm(pelvis_ang_vel),
            "q_torque": np.linalg.norm(action),
            "q_orientation": np.linalg.norm(pelvis_quat - FORWARD_QUARTERNIONS),
            "q_left_foot_pitch": (abs(left_foot_rpy[1]) if left_frc_norm > 0.01 else 0.0)
            if left_foot_rpy is not None
            else 0.0,
            "q_right_foot_pitch": (abs(right_foot_rpy[1]) if right_frc_norm > 0.01 else 0.0)
            if right_foot_rpy is not None
            else 0.0,
        }
        normalized_quantities = {
            k: self._normalize_quantity(k, v) for k, v in raw_quantities.items()
        }

        return total_reward, {
            "raw_quantities": raw_quantities,
            "normalized_quantities": normalized_quantities,
            "after_exponential": after_exponential,
            "coefficients": coeff,
            "rewards": rewards,
            "total_reward": total_reward,
        }
