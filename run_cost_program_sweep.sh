#!/usr/bin/env bash
set -euo pipefail
for L in 0.00 0.02 0.05 0.10; do
  OUT="results/cost_router_program_L${L}"
  /home/maxwelhelp/main/bin/python -u adfc/graph_adfc_worm_cost_router.py \
    --out "$OUT" \
    --steps 120 \
    --batch 192 \
    --eval-batch 224 \
    --tasks program \
    --mixed-tasks program \
    --models always_typed,cost_router \
    --log-every 40 \
    --cost-lambda "$L"
done
