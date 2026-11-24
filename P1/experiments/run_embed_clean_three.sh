#!/usr/bin/env bash
# 三方法干净嵌印 → {OUT}/{method}/wav_wm_clean/{stem}_wm.wav
set -euo pipefail

IN="${IN:-/data1/home/ynian/codes/data/fleurs/select_n2000}"
OUT="${OUT:-/data1/home/ynian/codes/data/fleurs/wm_n2000}"

python -m P1.experiments.run_embed_clean_three \
  --wav-dir "$IN" \
  --out-dir "$OUT" \
  --methods audioseal,band,wavmark \
  --num-workers 8 \
  --npu-ids 0,1,2,3,4,5,6,7 \
  --cpu-threads 2
