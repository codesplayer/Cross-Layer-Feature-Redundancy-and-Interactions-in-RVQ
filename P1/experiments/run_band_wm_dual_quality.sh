#!/usr/bin/env bash
# band 同 token：仅 LSD@16k（wm↔orig / recon↔orig / wm↔recon）
set -euo pipefail

ORIG="${ORIG:-/data1/home/ynian/codes/data/fleurs/select_n2000}"
OUT="${OUT:-/data1/home/ynian/codes/eval/add_exp/result/band_wm_dual_quality}"

python -m P1.experiments.run_band_wm_dual_quality \
  --orig-dir "$ORIG" \
  --out-dir "$OUT" \
  --metric-sr 16000 \
  --num-workers 8 \
  --npu-ids 0,1,2,3,4,5,6,7 \
  --cpu-threads 2
