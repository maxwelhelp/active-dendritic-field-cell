#!/usr/bin/env bash
set -euo pipefail
OUT="results/011_task_moe_mixed_s200"
mkdir -p "$OUT"
/home/maxwelhelp/main/bin/python -u adfc/graph_adfc_worm_task_moe.py --out "$OUT" --steps 200 --batch 192 --eval-batch 256 --seq-len 56 --tasks mixed --mixed-tasks route,order,kv,program --models always_typed,task_moe_utility --log-every 40 > "$OUT/run.log" 2>&1
cat "$OUT/winners.csv"
cat "$OUT/summary_rows.csv"
