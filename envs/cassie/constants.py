from collections import OrderedDict

import numpy as np

DEFAULT_CONFIG = {
    "symmetric_regulation": "alternate",
    "dt_per_cycle": 1.0,
    "r": 0.6,
    "kappa": 25,
    "command": [0.5, 0.0],
    "terminate_when_unhealthy": True,
    "max_simulation_time": 10.0,
    "pelvis_height": [0.5, 1.25],
    "feet_distance_x": 0.8,
    "feet_distance_y": 0.6,
    "feet_distance_z": 0.5,
    "feet_pelvis_height": 0.15,
    "feet_height": 0.15,
    "require_grounded_after_first_contact": True,
    "model": "cassie",
    "render_mode": "rgb_array",
    "reset_noise_scale": 0.08,
    "force_max_norm": 5.0,
    "push_prob_per_second": 0.0,
    "push_duration": 0.1,
    "r_bias": -0.1,
    "r_biped": 0.7,
    "r_cmd": 0.5,
    "r_smooth": 0.3,
    "r_feet_parallel": 0.1,
    "is_training": True,
    "max_roll": 2.0,
    "max_pitch": 2.0,
    "max_yaw": 10.0,
    "width": 1920,
    "height": 1080,
    "sim_fps": 40,
}


THETA_LEFT = 0.5
THETA_RIGHT = 0

FORWARD_QUARTERNIONS = np.array([1, 0, 0, 0])

c_swing_frc = +1
c_stance_frc = 0
c_swing_spd = 0
c_stance_spd = +1

OMEGA = 3.0

# MuJoCo body / geom indices
PELVIS = 1
RIGHT_FOOT = 13
LEFT_FOOT = 25
RIGHT_CONTACT_IDX = 49
LEFT_CONTACT_IDX = 33

DEFAULT_CAMERA_CONFIG = {
    "trackbodyid": 0,
    "distance": 4.0,
    "lookat": np.array((0.0, 0.0, 0.85)),
    "elevation": -20.0,
}

actuator_speed_ranges = {
    "left-hip-roll": [-4.5, 4.5],
    "left-hip-yaw": [-4.5, 4.5],
    "left-hip-pitch": [-12.2, 12.2],
    "left-knee": [-12.2, 12.2],
    "left-foot": [-0.9, 0.9],
    "right-hip-roll": [-4.5, 4.5],
    "right-hip-yaw": [-4.5, 4.5],
    "right-hip-pitch": [-12.2, 12.2],
    "right-knee": [-12.2, 12.2],
    "right-foot": [-0.9, 0.9],
}

mass = 33.8502
gravity = 9.81

sensors = OrderedDict(
    {
        "actuatorpos": [
            "left-hip-roll-input",
            "left-hip-yaw-input",
            "left-hip-pitch-input",
            "left-knee-input",
            "left-foot-input",
            "right-hip-roll-input",
            "right-hip-yaw-input",
            "right-hip-pitch-input",
            "right-knee-input",
            "right-foot-input",
        ],
        "jointpos": [
            "left-shin-output",
            "left-tarsus-output",
            "left-foot-output",
            "right-shin-output",
            "right-tarsus-output",
            "right-foot-output",
        ],
        "framequat": ["pelvis-orientation"],
        "gyro": ["pelvis-angular-velocity"],
        "accelerometer": ["pelvis-linear-acceleration"],
        "magnetometer": ["pelvis-magnetometer"],
    }
)

# Flat observation layout (must match build_observation concatenation order)
OBS_SLICES = {
    "actuatorpos": slice(0, 10),
    "jointpos": slice(10, 16),
    "framequat": slice(16, 20),
    "gyro": slice(20, 23),
    "accelerometer": slice(23, 26),
    "magnetometer": slice(26, 29),
    "command": slice(29, 31),
    "contact_forces": slice(31, 37),
    "clock": slice(37, 39),
    "previous_action": slice(39, 49),
}

OBS_COMPONENT_NAMES = {
    "actuatorpos": [
        "left-hip-roll",
        "left-hip-yaw",
        "left-hip-pitch",
        "left-knee",
        "left-foot",
        "right-hip-roll",
        "right-hip-yaw",
        "right-hip-pitch",
        "right-knee",
        "right-foot",
    ],
    "jointpos": [
        "left-shin",
        "left-tarsus",
        "left-foot-output",
        "right-shin",
        "right-tarsus",
        "right-foot-output",
    ],
    "framequat": ["w", "x", "y", "z"],
    "gyro": ["x", "y", "z"],
    "accelerometer": ["x", "y", "z"],
    "magnetometer": ["x", "y", "z"],
    "command": ["vx_world", "vy_world"],
    "contact_forces": ["left_x", "left_y", "left_z", "right_x", "right_y", "right_z"],
    "clock": ["sin", "cos"],
}
