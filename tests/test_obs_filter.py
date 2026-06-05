from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import tree
from ray.rllib.utils.filter import RunningStat

from droprl.rllib.loader import _json_safe, _restore_filter_shape


def test_obs_filter_shape_roundtrip() -> None:
    original = np.array([49])
    payload = _json_safe({"shape": original})
    restored = _restore_filter_shape(payload["shape"])
    assert isinstance(restored, np.ndarray)
    assert restored.tolist() == [49]

    rs = RunningStat()
    rs.mean_array = np.zeros(49, dtype=np.float32)
    rs.std_array = np.ones(49, dtype=np.float32)
    rs.num_pushes = 1
    stats = [rs]
    unflat = tree.unflatten_as(restored, stats)
    assert isinstance(unflat, RunningStat)
    assert unflat.mean_array.shape == (49,)


def test_obs_filter_json_file_roundtrip(tmp_path: Path) -> None:
    stat = RunningStat()
    stat.mean_array = np.zeros(49, dtype=np.float32)
    stat.std_array = np.ones(49, dtype=np.float32)
    stat.num_pushes = 10
    empty = RunningStat()
    empty.mean_array = np.zeros(49, dtype=np.float32)
    empty.std_array = np.ones(49, dtype=np.float32)
    path = tmp_path / "obs_filter.json"
    path.write_text(
        json.dumps(
            _json_safe(
                {
                    "shape": [49],
                    "no_preprocessor": False,
                    "demean": True,
                    "destd": True,
                    "clip": 10.0,
                    "running_stats": [stat.to_state()],
                    "buffer": [empty.to_state()],
                }
            ),
            indent=2,
        )
    )
    params = json.loads(path.read_text())
    shape = _restore_filter_shape(params["shape"])
    stats = [RunningStat.from_state(s) for s in params["running_stats"]]
    assert isinstance(tree.unflatten_as(shape, stats), RunningStat)
