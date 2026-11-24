#!/usr/bin/env bash
# L9 · 10 ABA key 闭集溯源：rare vs random 对照（主 cos）
set -euo pipefail

WAV="${WAV:-/data1/home/ynian/codes/codec_an/exp_watermark/data/mix_1500}"
OUT="${OUT:-/data1/home/ynian/codes/eval/add_exp/result/l9_rare10_key_gap_matrix}"

python -m P1.experiments.run_l9_rare10_key_gap_matrix \
  --wav-dir "$WAV" \
  --out-dir "$OUT" \
  --key-mode both \
  --n-keys 10 \
  --seed 0 \
  --rebuild-keys \
  --num-workers 8 \
  --npu-ids 0,1,2,3,4,5,6,7 \
  --cpu-threads 2
