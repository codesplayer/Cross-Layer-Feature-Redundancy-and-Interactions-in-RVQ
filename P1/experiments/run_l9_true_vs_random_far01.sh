#!/usr/bin/env bash
# 三个路径：1500 定标 / 2040 评测 / 输出目录；每条语音存 per_clip json
set -euo pipefail

WAV_1500="${WAV_1500:-/data1/home/ynian/codes/codec_an/exp_watermark/data/mix_1500}"
WAV_2040="${WAV_2040:-/data1/home/ynian/codes/data/fleurs/select_n2000}"
OUT="${OUT:-/data1/home/ynian/codes/eval/add_exp/result/l9_true_vs_random_far01}"
TABLE="${TABLE:-/data1/home/ynian/codes/codec_an/exp_watermark/codec_an/anas/watermark_rare_tokens/rare_aba.json}"

python -m P1.experiments.run_l9_true_vs_random_far01 \
  --wav-1500 "$WAV_1500" \
  --wav-2040 "$WAV_2040" \
  --out-dir "$OUT" \
  --table "$TABLE" \
  --seed 0 \
  --group-size 20 \
  --target-far 0.01 \
  --stage all \
  --num-workers 8 \
  --npu-ids 0,1,2,3,4,5,6,7 \
  --cpu-threads 2
