#!/usr/bin/env bash
set -euo pipefail
mkdir -p results/014_predator_morph
python3 -u adfc/ecosim_predator_morph.py --visual none --steps 6000 --log-every 100 --csv results/014_predator_morph/stats_headless.csv
