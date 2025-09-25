#!/usr/bin/env bash
set -euo pipefail

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3}

NUM_GPUS=$(python - <<'PY'
import os
print(len(os.environ.get('CUDA_VISIBLE_DEVICES','0').split(',')))
PY
)

torchrun --nproc_per_node=${NUM_GPUS} train.py --config configs/default.yaml
