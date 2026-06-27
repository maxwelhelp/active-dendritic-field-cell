#!/usr/bin/env bash
set -euo pipefail
python -u adfc/adfc_v6_orderkernel.py \
  --out results/reproduce_v6_order \
  --tasks order_compare \
  --models mean_pool,adfc3,adfc6
