"""Deprecated module name — 3D arrows live in ``CassieEnv.render()`` via ``scene_viz``."""

from scene_viz import (
    ArrowSpec,
    add_arrow_to_scene,
    command_arrow_tip,
    inject_scene_arrows,
    install_render_arrow_hook,
    pelvis_origin,
    push_arrow_tip,
)

__all__ = [
    "ArrowSpec",
    "add_arrow_to_scene",
    "command_arrow_tip",
    "inject_scene_arrows",
    "install_render_arrow_hook",
    "pelvis_origin",
    "push_arrow_tip",
]
