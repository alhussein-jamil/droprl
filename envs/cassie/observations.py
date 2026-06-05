from typing import TYPE_CHECKING

import numpy as np
from constants import OBS_SLICES, sensors

if TYPE_CHECKING:
    import numpy.typing as npt


def _mirror_actuators(values: "npt.NDArray[np.float32]") -> "npt.NDArray[np.float32]":
    """Swap left/right leg actuators (5 each) and negate hip-roll/yaw."""
    mirrored = np.concatenate([values[5:], values[:5]])
    mirrored[0] = -mirrored[0]
    mirrored[1] = -mirrored[1]
    mirrored[5] = -mirrored[5]
    mirrored[6] = -mirrored[6]
    return mirrored


def build_observation(
    sensor_data: dict[str, np.ndarray],
    command: "npt.NDArray[np.float32]",
    contact_force_left_foot: "npt.NDArray[np.float32]",
    contact_force_right_foot: "npt.NDArray[np.float32]",
    phi: float,
    previous_action: "npt.NDArray[np.float32]",
) -> "npt.NDArray[np.float32]":
    """Constructs the observation vector from sensor data and internal state."""
    clock_signal = np.array([np.sin(2 * np.pi * phi), np.cos(2 * np.pi * phi)])

    sensor_readings_np = np.concatenate([sensor_data[group] for group in sensors.keys()])

    return np.concatenate(
        [
            sensor_readings_np,
            command,
            contact_force_left_foot[:3],
            contact_force_right_foot[:3],
            clock_signal,
            previous_action,
        ]
    ).astype(np.float32)


def get_symmetric_obs(obs: "npt.NDArray[np.float32]") -> "npt.NDArray[np.float32]":
    """Returns the symmetric (mirrored) observation for alternating gait."""
    s = OBS_SLICES
    symmetric_obs = obs.copy()

    symmetric_obs[s["actuatorpos"]] = _mirror_actuators(obs[s["actuatorpos"]])

    jp = s["jointpos"]
    symmetric_obs[jp] = np.concatenate([obs[jp][3:], obs[jp][:3]])

    fq = s["framequat"]
    symmetric_obs[fq.start + 1] = -obs[fq.start + 1]
    symmetric_obs[fq.start + 3] = -obs[fq.start + 3]

    gyro = s["gyro"]
    symmetric_obs[gyro.start] = -obs[gyro.start]
    symmetric_obs[gyro.start + 2] = -obs[gyro.start + 2]

    accel = s["accelerometer"]
    symmetric_obs[accel.start + 1] = -obs[accel.start + 1]

    mag = s["magnetometer"]
    symmetric_obs[mag.start + 1] = -obs[mag.start + 1]

    cmd = s["command"]
    symmetric_obs[cmd.start + 1] = -obs[cmd.start + 1]

    cf = s["contact_forces"]
    left_force = obs[cf.start + 3 : cf.start + 6].copy()
    right_force = obs[cf.start : cf.start + 3].copy()
    left_force[1] = -left_force[1]
    right_force[1] = -right_force[1]
    symmetric_obs[cf.start : cf.start + 3] = left_force
    symmetric_obs[cf.start + 3 : cf.start + 6] = right_force

    clk = s["clock"]
    symmetric_obs[clk] = -obs[clk]

    symmetric_obs[s["previous_action"]] = _mirror_actuators(obs[s["previous_action"]])

    return symmetric_obs


def symmetric_action(action: "npt.NDArray[np.float32]") -> "npt.NDArray[np.float32]":
    """Swap left/right actuator commands for symmetric gait."""
    return _mirror_actuators(np.asarray(action, dtype=np.float32))
