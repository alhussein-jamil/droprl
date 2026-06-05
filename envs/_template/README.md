# Task template

Copy this directory to `envs/<your_task>/` and replace placeholders.

## Required files

| File | Contract |
|------|----------|
| `env.py` | `ENV_ID: str` and `make_env(config: dict) -> gym.Env \| BaseEnv` |
| `config.yaml` | `config:` mapping passed to `make_env` |

## Optional files

| File | Contract |
|------|----------|
| `callbacks.py` | `class Callbacks(DefaultCallbacks)` — auto-wired during training |
| `requirements.txt` | Extra pip deps (`make install-env TASK=...`) |
| `assets/` | Meshes, models, textures |

## After scaffolding

1. Copy `configs/train/_template.yaml` → `configs/train/<YourTask>PPO.yaml`
2. `make install-env TASK=<your_task>` if you added `requirements.txt`
3. `make train TASK=<your_task> ITERS=5`

Train configs contain **algorithm settings only**. Task-specific code stays in this folder.
