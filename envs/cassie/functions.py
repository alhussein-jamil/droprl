from typing import TYPE_CHECKING

import numba as nb
import numpy as np
from scipy import stats

if TYPE_CHECKING:
    import numpy.typing as npt


def pelvis_linvel_world(cvel: "npt.NDArray[np.float64]", body_id: int) -> "npt.NDArray[np.float64]":
    """Pelvis linear velocity in the world frame (m/s): [vx, vy, vz].

    MuJoCo ``data.cvel`` stores rotational velocity in indices 0:3 and linear
    velocity in 3:6, both in global coordinates. Do not use ``qvel[0:2]`` for
    command tracking — those are root joint rates, not Cartesian pelvis speed.
    """
    return cvel[body_id, 3:6]


def p_between_von_mises(a, b, kappa, x):
    # Calculate the CDF values for A and B at x
    cdf_a = stats.vonmises.cdf(2 * np.pi * x, kappa, loc=2 * np.pi * a)
    cdf_b = stats.vonmises.cdf(2 * np.pi * x, kappa, loc=2 * np.pi * b)

    # Calculate the probability of A < x < B
    p_between = np.abs(cdf_b - cdf_a)

    return p_between


@nb.jit(nopython=True, cache=True)
def mod(a, b):
    """
    Computes `a mod b` centered around zero.

    The result `r` satisfies:
    - `r = a - k*b` for some integer `k`
    - `-abs(b)/2 <= r < abs(b)/2` if b > 0
    - `-abs(b)/2 < r <= abs(b)/2` if b < 0

    Args:
        a: The dividend. Can be a scalar or NumPy array.
        b: The divisor. Must be non-zero. Can be a scalar or NumPy array (if broadcasting is intended).

    Returns:
        The result of the centered modulo operation. Matches the type of `a / b`.
    """
    # Using the formula: (a + b/2) % b - b/2
    # Ensure floating point division
    b_half = b / 2.0
    # Calculate result using standard modulo operator (%) which works correctly in Numba/NumPy
    # Note: The behavior of % with negative numbers matches Python's definition (sign of divisor).
    result = (a + b_half) % b - b_half
    return result


@nb.jit(nopython=True, cache=True)
def action_dist(
    a: "npt.NDArray[np.float64]",
    b: "npt.NDArray[np.float64]",
    actions_high: "npt.NDArray[np.float64]",
    actions_low: "npt.NDArray[np.float64]",
) -> "npt.NDArray[np.float64]":
    diff = a - b

    diff /= actions_high - actions_low
    diff = np.sum(np.square(diff), axis=1)

    return np.sqrt(diff)


@nb.jit(nopython=True, cache=True)
def mirror_symmetric_obs(out: "npt.NDArray[np.float32]", obs: "npt.NDArray[np.float32]") -> None:
    """In-place sagittal mirror of the 49-dim Cassie observation into *out*."""
    for i in range(49):
        out[i] = obs[i]

    # actuatorpos: swap legs and negate hip-roll/yaw
    for i in range(5):
        out[i] = obs[i + 5]
        out[i + 5] = obs[i]
    out[0] = -out[0]
    out[1] = -out[1]
    out[5] = -out[5]
    out[6] = -out[6]

    # jointpos: swap left/right (3 each)
    for i in range(3):
        out[10 + i] = obs[13 + i]
        out[13 + i] = obs[10 + i]

    # framequat x, z
    out[17] = -obs[17]
    out[19] = -obs[19]

    # gyro x, z
    out[20] = -obs[20]
    out[22] = -obs[22]

    # accelerometer y
    out[24] = -obs[24]

    # magnetometer y
    out[27] = -obs[27]

    # command y
    out[30] = -obs[30]

    # contact forces: swap feet, negate tangent y
    for i in range(3):
        out[31 + i] = obs[34 + i]
        out[34 + i] = obs[31 + i]
    out[32] = -out[32]
    out[35] = -out[35]

    # clock
    out[37] = -obs[37]
    out[38] = -obs[38]

    # previous_action
    for i in range(5):
        out[39 + i] = obs[44 + i]
        out[44 + i] = obs[39 + i]
    out[39] = -out[39]
    out[40] = -out[40]
    out[44] = -out[44]
    out[45] = -out[45]


@nb.jit(nopython=True, cache=True)
def mirror_symmetric_action(
    out: "npt.NDArray[np.float32]", action: "npt.NDArray[np.float32]"
) -> None:
    for i in range(5):
        out[i] = action[i + 5]
        out[i + 5] = action[i]
    out[0] = -out[0]
    out[1] = -out[1]
    out[5] = -out[5]
    out[6] = -out[6]


@nb.jit(nopython=True, cache=True)
def quat_to_rpy(
    quaternion: "npt.NDArray[np.float32]", radians: bool = True
) -> "npt.NDArray[np.float32]":
    """Convert quaternion (w, x, y, z) to roll, pitch, yaw."""
    w, x, y, z = quaternion[0], quaternion[1], quaternion[2], quaternion[3]
    sinr_cosp = 2.0 * (w * x + y * z)
    cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
    roll = np.arctan2(sinr_cosp, cosr_cosp)

    sinp = 2.0 * (w * y - z * x)
    if np.abs(sinp) >= 1:
        pitch = np.copysign(np.pi / 2.0, sinp)
    else:
        pitch = np.arcsin(sinp)

    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    yaw = np.arctan2(siny_cosp, cosy_cosp)

    rpy = np.array([roll, pitch, yaw], dtype=np.float32)
    return rpy if radians else np.degrees(rpy)


def apply_f_to_nested_dict(f, nested_dict):
    """
    Applies f to all values in a nested dict
    """
    for k, v in nested_dict.items():
        if isinstance(v, dict):
            apply_f_to_nested_dict(f, v)
        elif isinstance(v, list):
            for i in range(len(v)):
                v[i] = f(v[i])
        elif isinstance(v, float):
            nested_dict[k] = f(v)
