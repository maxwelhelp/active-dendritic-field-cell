#!/usr/bin/env bash
set -euo pipefail
cd /home/maxwelhelp/test/sience/experiments/math_search/WORKING_BEST/active-dendritic-field-cell-repo
TORCH_DISABLE_ADDR2LINE=1 /home/maxwelhelp/main/bin/python -u adfc/life_task_moe_gpu.py \
  --out results/015_life_task_moe_gpu_big \
  --steps 6000 \
  --agents 768 \
  --food 192 \
  --seq-len 48 \
  --nodes 160 \
  --dim 256 \
  --motor-nodes 12 \
  --degree 16 \
  --log-every 25
