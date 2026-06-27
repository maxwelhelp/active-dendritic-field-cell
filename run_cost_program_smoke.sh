#!/usr/bin/env bash
set -euo pipefail
/home/maxwelhelp/main/bin/python -u adfc/graph_adfc_worm_cost_router.py \
  --out results/cost_router_program_smoke \
  --steps 60 \
  --batch 192 \
  --eval-batch 224 \
  --tasks program \
  --mixed-tasks program \
  --models graph,always_typed,cost_router \
  --log-every 20 \
  --cost-lambda 0.10
