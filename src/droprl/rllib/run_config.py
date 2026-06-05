from __future__ import annotations

from typing import Any


def run_int(run_cfg: dict[str, Any], *keys: str, default: int = 0) -> int:
    for key in keys:
        if key in run_cfg:
            return int(run_cfg[key])
    return default


def chkpt_every_iters(run_cfg: dict[str, Any]) -> int:
    return run_int(run_cfg, "chkpt_every_iters", "checkpoint_every", default=250)


def chkpt_freq_seconds(run_cfg: dict[str, Any]) -> float:
    return float(run_int(run_cfg, "chkpt_freq", "checkpoint_every_seconds", default=2400))


def render_every_seconds(run_cfg: dict[str, Any]) -> float:
    return float(run_int(run_cfg, "render_every", "render_every_seconds", default=600))


def resolve_end_iter(
    run_cfg: dict[str, Any],
    *,
    start_iter: int,
    cli_iters: int | None,
) -> tuple[int, str]:
    """Return (end_iter_exclusive, limit_mode) for the training loop."""
    if cli_iters is not None:
        return start_iter + cli_iters, "iters"
    if "epochs" in run_cfg:
        return int(run_cfg["epochs"]), "epochs"
    return start_iter + int(run_cfg.get("iters", 10)), "iters"
