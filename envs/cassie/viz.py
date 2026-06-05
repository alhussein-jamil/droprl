"""Lightweight overlays for evaluation / render videos."""

from __future__ import annotations

import cv2
import mujoco as m
import numpy as np
from constants import PELVIS


def _project_to_render(
    env,
    point: np.ndarray,
    frame_shape: tuple[int, ...],
) -> tuple[int, int] | None:
    """Project a world point to pixel coords using the live render camera."""
    h, w = int(frame_shape[0]), int(frame_shape[1])
    renderer = getattr(env, "mujoco_renderer", None)
    if renderer is None:
        return None

    viewer = renderer._get_viewer("rgb_array")
    m.mjv_updateScene(
        env.model,
        env.data,
        viewer.vopt,
        viewer.pert,
        viewer.cam,
        m.mjtCatBit.mjCAT_ALL,
        viewer.scn,
    )
    cam = viewer.scn.camera[0]
    pos = np.asarray(cam.pos, dtype=np.float64)
    forward = np.asarray(cam.forward, dtype=np.float64)
    up = np.asarray(cam.up, dtype=np.float64)
    forward /= np.linalg.norm(forward) + 1e-12
    up /= np.linalg.norm(up) + 1e-12
    right = np.cross(forward, up)
    right /= np.linalg.norm(right) + 1e-12
    up = np.cross(right, forward)

    rel = np.asarray(point, dtype=np.float64) - pos
    depth = float(np.dot(rel, forward))
    if depth <= 1e-6:
        return None

    xc = float(np.dot(rel, right))
    yc = float(np.dot(rel, up))
    half_h = abs(float(cam.frustum_top))
    if half_h <= 1e-12:
        return None
    half_w = half_h * (w / h)
    u = w * (0.5 + 0.5 * xc / (depth * half_w))
    v = h * (0.5 - 0.5 * yc / (depth * half_h))
    u_i, v_i = int(round(u)), int(round(v))
    return u_i, v_i


def screen_force_direction(
    env,
    force: np.ndarray,
    frame_shape: tuple[int, ...],
    *,
    z_offset: float = 0.0,
    arrow_m: float = 0.45,
) -> tuple[float, float] | None:
    """Unit vector in pixel space aligned with the applied push force."""
    h, w = frame_shape[0], frame_shape[1]
    _ = h, w

    origin = _pelvis_arrow_origin(env, z_offset=z_offset)
    direction = np.asarray(force, dtype=np.float64)
    n = float(np.linalg.norm(direction))
    if n <= 1e-8:
        return None
    tip = origin + (direction / n) * arrow_m

    p0 = _project_to_render(env, origin, frame_shape)
    p1 = _project_to_render(env, tip, frame_shape)
    if p0 is None or p1 is None:
        return None

    dx, dy = float(p1[0] - p0[0]), float(p1[1] - p0[1])
    length = float(np.hypot(dx, dy))
    if length < 1e-6:
        return None
    return dx / length, dy / length


def _pelvis_arrow_origin(env, *, z_offset: float) -> np.ndarray:
    """World point at the pelvis body (where ``xfrc_applied`` is set)."""
    return np.asarray(env.data.xpos[PELVIS], dtype=np.float64).copy() + np.array(
        [0.0, 0.0, z_offset], dtype=np.float64
    )


def _clamp_arrow_end(
    p0: tuple[int, int],
    p1: tuple[int, int],
    frame_shape: tuple[int, ...],
    *,
    min_px: float,
    max_px: float,
) -> tuple[int, int]:
    dx, dy = float(p1[0] - p0[0]), float(p1[1] - p0[1])
    length = float(np.hypot(dx, dy))
    if length < 1e-6:
        return p1
    if length < min_px:
        scale = min_px / length
    elif length > max_px:
        scale = max_px / length
    else:
        return p1
    return int(p0[0] + dx * scale), int(p0[1] + dy * scale)


def draw_push_indicator(
    frame: np.ndarray,
    env,
    *,
    active: bool = True,
    z_offset: float = 0.0,
    arrow_m: float = 0.45,
    label: bool = False,
) -> np.ndarray:
    """Arrow from the pelvis showing the applied push (screen-projected)."""
    if not active or int(getattr(env, "_pushing", -1)) < 0:
        return frame

    force = np.asarray(getattr(env, "_current_push_force", np.zeros(3)), dtype=np.float64)
    mag = float(np.linalg.norm(force))
    if mag <= 1e-6:
        return frame

    h, w = frame.shape[:2]

    origin = _pelvis_arrow_origin(env, z_offset=z_offset)
    direction = force / mag
    tip_world = origin + direction * arrow_m

    p0 = _project_to_render(env, origin, frame.shape)
    p1 = _project_to_render(env, tip_world, frame.shape)
    if p0 is None or p1 is None:
        return frame

    min_px = 0.10 * min(w, h)
    max_px = 0.20 * min(w, h)
    p1 = _clamp_arrow_end(p0, p1, frame.shape, min_px=min_px, max_px=max_px)

    out = frame.copy()
    shadow = (20, 20, 24)
    shaft = (245, 248, 255)
    line_t = max(2, int(round(min(w, h) * 0.0016)))
    dot_r = max(3, int(round(min(w, h) * 0.0024)))

    cv2.arrowedLine(
        out,
        p0,
        p1,
        shadow,
        thickness=line_t + 2,
        tipLength=0.22,
        line_type=cv2.LINE_AA,
    )
    cv2.arrowedLine(
        out,
        p0,
        p1,
        shaft,
        thickness=line_t,
        tipLength=0.22,
        line_type=cv2.LINE_AA,
    )
    cv2.circle(out, p0, dot_r + 2, shadow, -1, lineType=cv2.LINE_AA)
    cv2.circle(out, p0, dot_r, shaft, -1, lineType=cv2.LINE_AA)

    if label:
        text = f"Push {mag:.1f} N"
        font = cv2.FONT_HERSHEY_DUPLEX
        scale = max(0.45, min(w, h) / 2400.0)
        thick = max(1, int(round(scale * 2)))
        (tw, th), _ = cv2.getTextSize(text, font, scale, thick)
        lx = int(np.clip(p0[0] - tw // 2, 8, w - tw - 8))
        ly = int(np.clip(p0[1] - th - dot_r - 10, th + 4, h - 8))
        cv2.rectangle(
            out,
            (lx - 8, ly - th - 6),
            (lx + tw + 8, ly + 6),
            (28, 32, 42),
            -1,
        )
        cv2.putText(out, text, (lx, ly), font, scale, (255, 210, 180), thick, cv2.LINE_AA)

    return out
