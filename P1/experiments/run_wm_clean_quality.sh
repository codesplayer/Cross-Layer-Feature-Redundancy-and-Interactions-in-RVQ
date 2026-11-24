#!/usr/bin/env bash
# 嵌印干净语音 vs 原始：LSD/MCD/spk/emo/DNSMOS
set -euo pipefail

ORIG="${ORIG:-/data1/home/ynian/codes/data/fleurs/select_n2000}"
RECON="${RECON:-/data1/home/ynian/codes/data/fleurs/wm_n2000}"
OUT="${OUT:-/data1/home/ynian/codes/eval/add_exp/result/wm_clean_quality_n2000}"

python -m P1.experiments.run_wm_clean_quality \
  --orig-dir "$ORIG" \
  --recon-root "$RECON" \
  --out-dir "$OUT" \
  --wav-subdir wav_wm_clean \
  --methods audioseal,band,wavmark \
  --sr 16000 \
  --num-workers 8 \
  --npu-ids 0,1,2,3,4,5,6,7 \
  --cpu-threads 2
