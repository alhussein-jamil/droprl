from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

_ENV_DIR = Path(__file__).resolve().parent
_ENV_DIR_STR = str(_ENV_DIR)
if _ENV_DIR_STR not in sys.path:
    sys.path.insert(0, _ENV_DIR_STR)

ENV_ID = "cassie-v0"
MODEL_DIR = _ENV_DIR / "assets" / "model"


def make_env(config: dict[str, Any] | None = None):
    from cassie import CassieEnv  # noqa: E402

    return CassieEnv(env_config=config or {}, model_dir=MODEL_DIR)
