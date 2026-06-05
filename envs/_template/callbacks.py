"""Optional RLlib callbacks — delete this file if your task does not need hooks."""

from ray.rllib.algorithms.callbacks import DefaultCallbacks


class Callbacks(DefaultCallbacks):
    """Subclass DefaultCallbacks. DropRL auto-loads this when present."""

    pass
