#!/bin/bash
apt update && apt install -y vim tmux
curl -LsSf https://astral.sh/uv/install.sh | sh
uv sync --no-build-isolation
source .venv/bin/activate