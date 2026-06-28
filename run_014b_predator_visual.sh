#!/usr/bin/env bash
set -euo pipefail
mkdir -p results/014_predator_morph
python3 -u adfc/ecosim_predator_morph.py --visual pygame --steps 25000 --agents 55 --prey-init 45 --prey-spawn 0.055 --csv results/014_predator_morph/stats.csv
