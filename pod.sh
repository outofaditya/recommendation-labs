#!/usr/bin/env bash
# tunes fism then runs every tuned model and its seed repeats on a gpu pod
set -euo pipefail
cd "$(dirname "$0")"
export PATH="$HOME/.local/bin:$PATH" OMP_NUM_THREADS=1
command -v uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
uv sync --no-dev
uv run python -c "import torch; assert torch.cuda.is_available()"

# containers report host cores so the cpu quota wins when set
cpus=$(nproc)
read -r quota period </sys/fs/cgroup/cpu.max || quota=max
[ "$quota" != max ] && cpus=$((quota / period))
# each worker needs up to 4 gb of gpu memory
memory=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1)
workers=$((cpus < memory / 4096 ? cpus : memory / 4096))
echo "workers $workers"

uv run python scripts/tune.py --workers "$workers" FISM
uv run python scripts/models.py --tuned --workers "$workers"
uv run python scripts/seeds.py --workers "$workers"
