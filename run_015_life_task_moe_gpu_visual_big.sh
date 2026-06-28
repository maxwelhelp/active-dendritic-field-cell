#!/usr/bin/env bash
set -euo pipefail
cd /home/maxwelhelp/test/sience/experiments/math_search/WORKING_BEST/active-dendritic-field-cell-repo
TORCH_DISABLE_ADDR2LINE=1 /home/maxwelhelp/main/bin/python -u adfc/life_task_moe_gpu_pygame.py \
  --out results/015_life_task_moe_gpu_visual_big \
  --steps 30000 \
  --agents 512 \
  --food 128 \
  --seq-len 48 \
  --nodes 128 \
  --dim 192 \
  --motor-nodes 10 \
  --degree 14 \
  --log-every 25
