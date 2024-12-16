#!/bin/bash
apt update && apt install -y vim tmux
curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.local/bin/env
# We need to build the packages without flash-attn first, and then build flash-attn with dependencies
uv sync --no-install-package flash-attn
uv sync --no-build-isolation
source .venv/bin/activate
wandb login
huggingface-cli login