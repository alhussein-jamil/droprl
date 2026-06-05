# DropRL

**Drop an environment folder. Train with PPO. Ship it.**

[![ci](https://github.com/alhussein-jamil/droprl/actions/workflows/ci.yml/badge.svg?branch=master)](https://github.com/alhussein-jamil/droprl/actions/workflows/ci.yml)
[![Python 3.10–3.12](https://img.shields.io/badge/python-3.10--3.12-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Ray RLlib](https://img.shields.io/badge/RLlib-2.44.1-orange.svg)](https://docs.ray.io/en/latest/rllib/index.html)

DropRL is a minimal, env-first reinforcement learning framework built on [Ray RLlib](https://docs.ray.io/en/latest/rllib/index.html). Add a task under `envs/<name>/`, point PPO at it, and go — no boilerplate projects, no per-env Makefiles.

Includes a **mock** env for fast iteration and a full **Cassie** locomotion task (MuJoCo).

## Why DropRL?

| | DropRL | Typical RL repo |
|---|---|---|
| Add a new task | Copy `envs/_template/` | Fork, refactor, wire callbacks |
| Resume training | `make train` | Custom checkpoint scripts |
| Render best policy | `make render` | Manual checkpoint paths |
| Config | YAML merge: global → env → train | Scattered Python constants |

## Quick start

Requires **Python 3.10–3.12** (Ray 2.44.1).

```bash
git clone https://github.com/alhussein-jamil/droprl.git
cd droprl
make install
make train TASK=mock ITERS=5
make tensorboard
```

## Layout

```
DropRL/
├── configs/
│   ├── config.yaml              # global defaults (ray, run, dynamic_lr)
│   └── train/<Task>PPO.yaml     # PPO hyperparameters per task
├── envs/
│   ├── _template/               # copy to scaffold a new task
│   └── <task>/
│       ├── env.py               # ENV_ID + make_env(config)
│       ├── config.yaml          # task parameters
│       ├── callbacks.py         # optional RLlib hooks (auto-discovered)
│       ├── requirements.txt     # optional extra deps
│       └── assets/              # optional meshes, models
├── scripts/
│   ├── train.py
│   └── render.py
└── src/droprl/
```

Config merge order: `configs/config.yaml` → `envs/<task>/config.yaml` → `configs/train/<Train>.yaml`.

## Commands

```bash
make install                         # venv + core deps
make install-env TASK=cassie         # + env-specific deps (MuJoCo, etc.)
make train TASK=mock                 # resume latest checkpoint
make clean-train TASK=cassie         # fresh run
make render TASK=cassie              # MP4 from checkpoint_best (latest run)
make tensorboard                     # http://localhost:6006
make test                            # unit tests
make lint                            # ruff check + format
make pre-commit-install              # git hooks
```

`TRAIN` defaults to `<Task>PPO` (e.g. `mock` → `MockPPO`).

### Train & resume

```bash
make train TASK=mock ITERS=5
make train TASK=mock NAME=my_run     # resume a specific run
```

Without `NAME`, `make train` picks the newest run with a checkpoint and continues from the saved iteration.

Ctrl+C finishes the current iteration, saves `checkpoint_latest`, then exits.

### Render

```bash
make render TASK=cassie
make render TASK=cassie NAME=my_run
make render TASK=cassie LATEST=1      # checkpoint_latest
```

Output: `runs/<task>/<run>/simulations/render_best.mp4`.

## Add a new task

Scaffold from `envs/_template/`:

```bash
cp -r envs/_template envs/my_task
cp configs/train/_template.yaml configs/train/MyTaskPPO.yaml
make train TASK=my_task ITERS=5
```

### Task contract (`envs/<task>/`)

| File | Required | Purpose |
|------|----------|---------|
| `env.py` | yes | `ENV_ID` + `make_env(config)` |
| `config.yaml` | yes | Parameters merged into `env_config` |
| `callbacks.py` | no | `class Callbacks(DefaultCallbacks)` — auto-discovered |
| `requirements.txt` | no | Per-task pip deps |
| `assets/` | no | Models, meshes, data |

`make_env` may return a **Gymnasium** env (recommended) or a **`droprl.envs.base.BaseEnv`**.

Train configs (`configs/train/<Task>PPO.yaml`) hold **PPO hyperparameters only** — no task wiring.

## Features

- **RLlib PPO** with dynamic LR, checkpoint resume, TensorBoard logging
- **`num_env_runners: auto`** — uses all CPUs
- **Periodic renders** and **Ctrl+C checkpoint** (CassieRobot-style)
- **Per-env `requirements.txt`** — Cassie pins MuJoCo without polluting core deps

## Development

```bash
make install
pip install -e ".[dev]"
make pre-commit-install
make lint
make test
```

## License

MIT — see [LICENSE](LICENSE).
