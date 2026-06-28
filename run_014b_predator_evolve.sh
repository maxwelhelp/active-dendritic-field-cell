#!/usr/bin/env bash
set -euo pipefail
mkdir -p results/014_predator_morph
python3 -u adfc/ecosim_predator_morph.py \
  --visual pygame \
  --steps 35000 \
  --agents 65 \
  --min-agents 20 \
  --initial-energy 82 \
  --prey-init 65 \
  --prey-max 110 \
  --prey-spawn 0.12 \
  --prey-energy 20 \
  --metabolism 0.018 \
  --hunger-penalty 0.015 \
  --reproduce-energy 62 \
  --child-energy 22 \
  --reproduce-cooldown 150 \
  --csv results/014_predator_morph/stats_evolve.csv
