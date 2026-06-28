#!/usr/bin/env bash
set -euo pipefail
cd /home/maxwelhelp/test/sience/experiments/math_search/WORKING_BEST/active-dendritic-field-cell-repo
/home/maxwelhelp/main/bin/python -u adfc/life_task_moe_gpu.py --out results/015_life_task_moe_gpu --steps 2000 --agents 256 --food 64 --log-every 50
