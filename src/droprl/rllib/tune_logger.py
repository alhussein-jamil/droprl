from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

import cloudpickle
import numpy as np
from ray.air.constants import EXPR_PARAM_FILE, EXPR_PARAM_PICKLE_FILE, EXPR_RESULT_FILE
from ray.tune.logger.logger import Logger
from ray.tune.result import TIME_TOTAL_S, TIMESTEPS_TOTAL, TRAINING_ITERATION
from ray.tune.utils import flatten_dict
from ray.tune.utils.util import SafeFallbackEncoder

if TYPE_CHECKING:
    pass

_VALID_SUMMARY_TYPES = (int, float, np.float32, np.float64, np.int32, np.int64)


class TrainLogger(Logger):
    """Writes result.json and TensorBoard events into the run logdir."""

    def _init(self) -> None:
        from tensorboardX import SummaryWriter

        self._result_file = Path(self.logdir, EXPR_RESULT_FILE).open("a")
        self._tb_writer = SummaryWriter(self.logdir, flush_secs=30)
        self.update_config(self.config)

    def on_result(self, result: dict[str, Any]) -> None:
        json.dump(result, self._result_file, cls=SafeFallbackEncoder)
        self._result_file.write("\n")
        self._result_file.flush()

        step = result.get(TIMESTEPS_TOTAL) or result[TRAINING_ITERATION]
        payload = result.copy()
        for key in ("config", "pid", "timestamp", TIME_TOTAL_S, TRAINING_ITERATION):
            payload.pop(key, None)

        for attr, value in flatten_dict(payload, delimiter="/").items():
            full_attr = f"ray/tune/{attr}"
            if isinstance(value, _VALID_SUMMARY_TYPES) and not np.isnan(value):
                self._tb_writer.add_scalar(full_attr, value, global_step=step)
            elif isinstance(value, np.ndarray) and value.ndim == 3 and value.size > 0:
                self._tb_writer.add_image(full_attr, value, global_step=step)

        self._tb_writer.flush()

    def update_config(self, config: dict[str, Any]) -> None:
        self.config = config
        config_out = Path(self.logdir, EXPR_PARAM_FILE)
        config_out.write_text(json.dumps(config, indent=2, sort_keys=True, cls=SafeFallbackEncoder))
        with Path(self.logdir, EXPR_PARAM_PICKLE_FILE).open("wb") as handle:
            cloudpickle.dump(config, handle)

    def close(self) -> None:
        if not self._result_file.closed:
            self._result_file.close()
        self._tb_writer.close()

    def flush(self) -> None:
        if not self._result_file.closed:
            self._result_file.flush()
        self._tb_writer.flush()
