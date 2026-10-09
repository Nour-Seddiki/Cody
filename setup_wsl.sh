#!/usr/bin/env bash
# One-time setup of the training env inside WSL Ubuntu (run from PowerShell:
#   wsl -d Ubuntu -- bash setup_wsl.sh)
set -euo pipefail
export PATH="$HOME/.local/bin:$PATH"
# the connection here times out on big wheels at uv's defaults and drops now and then; finished
# wheels stay in uv's cache, so each retry only fetches what is still missing
export UV_HTTP_TIMEOUT=600 UV_HTTP_RETRIES=10 UV_CONCURRENT_DOWNLOADS=2
retry() {
    for i in $(seq 1 20); do
        "$@" && return 0
        echo "attempt $i failed - retrying in 15s"
        sleep 15
    done
    return 1
}

command -v uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh

uv venv ~/venvs/cody --python 3.12 --allow-existing
PY=~/venvs/cody/bin/python
retry uv pip install --python $PY torch --index-url https://download.pytorch.org/whl/cu128
retry uv pip install --python $PY "transformers==5.16.1" peft accelerate bitsandbytes \
    flash-linear-attention huggingface_hub safetensors
$PY -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"
