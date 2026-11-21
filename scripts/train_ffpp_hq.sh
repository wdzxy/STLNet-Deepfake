#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
python scripts/train.py --config configs/base.yaml \
  --train-csv data/train.csv --val-csv data/val.csv \
  --output runs/ffpp_hq "$@"
