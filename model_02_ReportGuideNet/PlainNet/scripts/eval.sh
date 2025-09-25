#!/usr/bin/env bash
set -euo pipefail

CKPT=${1:-runs/exp01/best.pth}
python evaluate.py --config configs/default.yaml --ckpt "$CKPT"
