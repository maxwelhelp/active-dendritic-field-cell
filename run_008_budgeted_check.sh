#!/usr/bin/env bash
set -euo pipefail
OUT="results/008_budgeted_topk_mixed_s240"
mkdir -p "$OUT"
/home/maxwelhelp/main/bin/python -u adfc/graph_adfc_worm_budgeted_router.py --out "$OUT" --steps 240 --batch 192 --eval-batch 256 --seq-len 56 --tasks mixed --mixed-tasks route,order,kv,program --models graph,always_typed,cost_router,budget_topk,budget_topk_hard --top-k 2 --channel-dropout 0.12 --cost-lambda 0.02 --target-lambda 0.08 --balance-lambda 0.04 --entropy-penalty 0.005 --log-every 40 > "$OUT/run.log" 2>&1
cat "$OUT/winners.csv"
cat "$OUT/summary_rows.csv"
