from __future__ import annotations

import logging
import sys
from typing import Literal

_LEVEL_COLORS = {
    "DEBUG": "\033[36m",
    "INFO": "\033[32m",
    "WARNING": "\033[33m",
    "ERROR": "\033[31m",
    "CRITICAL": "\033[35m",
}
_RESET = "\033[0m"


class _ColorFormatter(logging.Formatter):
    def __init__(self, *, colorize: bool, fmt: str):
        super().__init__(fmt)
        self._colorize = colorize

    def format(self, record: logging.LogRecord) -> str:
        if self._colorize:
            color = _LEVEL_COLORS.get(record.levelname, "")
            if color:
                record.levelname = f"{color}{record.levelname}{_RESET}"
        return super().format(record)


def setup_logging(
    *,
    level: int | str = logging.INFO,
    fmt: str = "%(levelname)s | %(message)s",
    stream: Literal["stdout", "stderr"] = "stdout",
    colorize: bool | None = None,
) -> None:
    out = sys.stdout if stream == "stdout" else sys.stderr
    use_color = out.isatty() if colorize is None else colorize

    root = logging.getLogger()
    root.handlers.clear()
    root.setLevel(level)

    handler = logging.StreamHandler(out)
    handler.setFormatter(_ColorFormatter(colorize=use_color, fmt=fmt))
    root.addHandler(handler)

    logging.getLogger("ray").setLevel(logging.WARNING)
    logging.getLogger("ray.rllib").setLevel(logging.WARNING)
