"""3D MuJoCo scene overlays injected during ``CassieEnv.render()``."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import mujoco as m
import numpy as np

if TYPE_CHECKING:
    from gymnasium.envs.mujoco.mujoco_rendering import BaseRender


@dataclass(frozen=True)
class ArrowSpec:
    origin: np.ndarray
    tip: np.ndarray
    width: float
    rgba: tuple[float, float, float, float]
    emission: float = 0.25


def _f64_vec3(point: np.ndarray) -> np.ndarray:
    return np.asarray(point, dtype=np.float64).reshape(3, 1)


def _f32_rgba4(rgba: tuple[float, float, float, float]) -> np.ndarray:
    return np.asarray(rgba, dtype=np.float32).reshape(4, 1)


def pelvis_origin(
    data,
    pelvis_id: int,
    *,
    z_offset: float = 0.2,
) -> np.ndarray:
    origin = np.asarray(data.xpos[pelvis_id], dtype=np.float64).copy()
    origin[2] += z_offset
    return origin


def world_xy_unit(vec2: np.ndarray) -> np.ndarray | None:
    direction = np.array([vec2[0], vec2[1], 0.0], dtype=np.float64)
    norm = float(np.linalg.norm(direction))
    if norm < 1e-8:
        return None
    return direction / norm


def command_arrow_tip(
    origin: np.ndarray,
    command: np.ndarray,
    *,
    min_length: float = 0.28,
    max_length: float = 0.62,
) -> np.ndarray | None:
    unit = world_xy_unit(command)
    if unit is None:
        return None
    mag = float(np.linalg.norm(command[:2]))
    length = min_length + (max_length - min_length) * min(mag, 1.0)
    return origin + unit * length


def push_arrow_tip(
    origin: np.ndarray,
    force: np.ndarray,
    force_max_norm: float,
    *,
    min_length: float = 0.32,
    max_length: float = 0.72,
) -> np.ndarray | None:
    unit = world_xy_unit(force[:2])
    if unit is None:
        return None
    mag = float(np.linalg.norm(force[:2]))
    if mag < 1e-6:
        return None
    scale = min(mag / max(force_max_norm, 1e-6), 1.0)
    length = min_length + (max_length - min_length) * scale
    return origin + unit * length


def add_arrow_to_scene(scn: m.MjvScene, spec: ArrowSpec) -> None:
    if scn.ngeom >= scn.maxgeom:
        return

    geom = scn.geoms[scn.ngeom]
    m.mjv_initGeom(
        geom,
        m.mjtGeom.mjGEOM_ARROW,
        np.zeros((3, 1)),
        np.zeros((3, 1)),
        np.eye(3, dtype=np.float64).reshape(9, 1),
        _f32_rgba4(spec.rgba),
    )
    geom.category = m.mjtCatBit.mjCAT_DECOR
    geom.emission = spec.emission
    m.mjv_connector(
        geom,
        m.mjtGeom.mjGEOM_ARROW,
        spec.width,
        _f64_vec3(spec.origin),
        _f64_vec3(spec.tip),
    )
    scn.ngeom += 1


def inject_scene_arrows(scn: m.MjvScene, specs: list[ArrowSpec]) -> None:
    for spec in specs:
        add_arrow_to_scene(scn, spec)


def install_render_arrow_hook(viewer: BaseRender, env: Any) -> None:
    """Inject arrows after ``mjv_updateScene`` without Gymnasium's marker API."""
    if getattr(viewer, "_cassie_render_arrow_hook", False):
        return
    viewer._cassie_render_arrow_hook = True
    viewer._cassie_arrow_env = env
    original_render = viewer.render
    original_update_scene = m.mjv_updateScene

    def mjv_update_scene_with_arrows(*args, **kwargs):
        original_update_scene(*args, **kwargs)
        scn = args[6] if len(args) > 6 else kwargs["scn"]
        specs = getattr(viewer._cassie_arrow_env, "_pending_arrow_specs", ())
        inject_scene_arrows(scn, list(specs))

    def render_with_arrows(*args, **kwargs):
        m.mjv_updateScene = mjv_update_scene_with_arrows
        try:
            return original_render(*args, **kwargs)
        finally:
            m.mjv_updateScene = original_update_scene

    viewer.render = render_with_arrows
