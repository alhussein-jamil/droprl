.PHONY: install install-env install-all train clean-train render tensorboard \
	lint format test pre-commit-install help

PYTHON ?= python3.10
VENV ?= .venv
PIP := $(VENV)/bin/pip
PY := $(VENV)/bin/python
RUFF := $(VENV)/bin/ruff
PYTEST := $(VENV)/bin/pytest
PRE_COMMIT := $(VENV)/bin/pre-commit
TENSORBOARD := $(VENV)/bin/tensorboard

TASK ?= mock
TRAIN ?= $(shell $(PYTHON) -c "print('$(TASK)'.capitalize() + 'PPO')")
ITERS ?=
NAME ?=
OUTPUT ?= runs
LOGDIR ?= runs
PORT ?= 6006

install:
	$(PYTHON) -m venv $(VENV)
	$(PIP) install -U pip
	$(PIP) install -r requirements.txt

install-env: install
	@test -f envs/$(TASK)/requirements.txt || \
		(echo "No envs/$(TASK)/requirements.txt" && exit 1)
	$(PIP) install -r envs/$(TASK)/requirements.txt

install-all: install
	@for req in envs/*/requirements.txt; do \
		echo ">> $$req"; \
		$(PIP) install -r "$$req"; \
	done

train:
	$(PY) scripts/train.py --task $(TASK) --train $(TRAIN) --output $(OUTPUT) \
		$(if $(ITERS),--iters $(ITERS),) \
		$(if $(NAME),--name $(NAME),)

clean-train:
	$(PY) scripts/train.py --task $(TASK) --train $(TRAIN) --output $(OUTPUT) --clean \
		$(if $(ITERS),--iters $(ITERS),) \
		$(if $(NAME),--name $(NAME),)

render:
	$(PY) scripts/render.py --task $(TASK) --train $(TRAIN) --runs-root $(OUTPUT) \
		$(if $(NAME),--name $(NAME),) \
		$(if $(CHECKPOINT),--checkpoint $(CHECKPOINT),) \
		$(if $(RENDER_OUT),--output $(RENDER_OUT),) \
		$(if $(LATEST),--latest,)

tensorboard:
	@test -x "$(TENSORBOARD)" || (echo "Run make install first" && exit 1)
	$(TENSORBOARD) --logdir $(LOGDIR) --port $(PORT) --bind_all

lint:
	@test -x "$(RUFF)" || (echo "Run: pip install -e '.[dev]'" && exit 1)
	$(RUFF) check src scripts tests envs/mock envs/cartpole
	$(RUFF) format --check src scripts tests envs/mock envs/cartpole

format:
	@test -x "$(RUFF)" || (echo "Run: pip install -e '.[dev]'" && exit 1)
	$(RUFF) check --fix src scripts tests envs/mock envs/cartpole
	$(RUFF) format src scripts tests envs/mock envs/cartpole

test:
	@test -x "$(PYTEST)" || (echo "Run: pip install -e '.[dev]'" && exit 1)
	$(PYTEST) tests/ -m "not slow" -v

test-all:
	@test -x "$(PYTEST)" || (echo "Run: pip install -e '.[dev]'" && exit 1)
	$(PYTEST) tests/ -v

pre-commit-install:
	@test -x "$(PRE_COMMIT)" || (echo "Run: pip install -e '.[dev]'" && exit 1)
	$(PRE_COMMIT) install

help:
	@echo "DropRL — drop-in RL environments"
	@echo ""
	@echo "install / install-env TASK= / install-all"
	@echo "train TASK=          resume latest checkpoint"
	@echo "clean-train TASK=    new run, no restore"
	@echo "render TASK=         checkpoint_best from latest run"
	@echo "tensorboard [LOGDIR=runs] [PORT=6006]"
	@echo "lint / format / test / pre-commit-install"
