#!/usr/bin/env bash
set -euo pipefail
cd /home/maxwelhelp/test/sience/experiments/math_search/WORKING_BEST/active-dendritic-field-cell-repo
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
TORCH_DISABLE_ADDR2LINE=1 /home/maxwelhelp/main/bin/python -u adfc/life_task_moe_gpu_pygame.py \
  --out results/015_life_task_moe_gpu_visual_big \
  --steps 30000 \
  --agents 192 \
  --food 64 \
  --seq-len 20 \
  --nodes 64 \
  --dim 96 \
  --motor-nodes 6 \
  --degree 10 \
  --log-every 25
