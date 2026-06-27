#!/usr/bin/env bash
set -euo pipefail
OUT="results/009_metabolic_mixed_s240"
mkdir -p "$OUT"
/home/maxwelhelp/main/bin/python -u adfc/graph_adfc_worm_metabolic.py --out "$OUT" --steps 240 --batch 192 --eval-batch 256 --seq-len 56 --tasks mixed --mixed-tasks route,order,kv,program --models always_typed,cost_router,metabolic,metabolic_hard --cost-lambda 0.02 --target-lambda 0.04 --channel-lambda 0.03 --edge-lambda 0.04 --activity-lambda 0.03 --homeo-lambda 0.01 --homeo-target 0.08 --log-every 40 > "$OUT/run.log" 2>&1
cat "$OUT/winners.csv"
cat "$OUT/summary_rows.csv"
