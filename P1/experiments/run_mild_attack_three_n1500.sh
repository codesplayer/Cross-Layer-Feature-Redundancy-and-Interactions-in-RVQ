#!/usr/bin/env bash
# 宽松攻击 × audioseal / wavmark / band · 可改 WAV / OUT
set -euo pipefail

WAV="${WAV:-/data1/home/ynian/codes/codec_an/exp_watermark/data/mix_1500}"
OUT="${OUT:-/data1/home/ynian/codes/eval/add_exp/result/mild_attack_three_n1500}"

python -m P1.experiments.run_mild_attack_three_n1500 \
  --wav-dir "$WAV" \
  --out-dir "$OUT" \
  --methods audioseal,wavmark,band \
  --num-workers 8 \
  --npu-ids 0,1,2,3,4,5,6,7 \
  --cpu-threads 2
