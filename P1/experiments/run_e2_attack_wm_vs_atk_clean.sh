#!/usr/bin/env bash
# E2：只算 attack(clean)；mild+anchor × audioseal/wavmark/band
# 攻击水印复用已有 mild_attack_three / anchor_attack_three 结果
set -euo pipefail

WAV="${WAV:-/data1/home/ynian/codes/codec_an/exp_watermark/data/mix_1500}"
OUT="${OUT:-/data1/home/ynian/codes/eval/add_exp/result/e2_attack_clean_detect}"
WM_MILD="${WM_MILD:-/data1/home/ynian/codes/eval/add_exp/result/mild_attack_three_n1500}"
WM_ANCHOR="${WM_ANCHOR:-/data1/home/ynian/codes/eval/add_exp/result/anchor_attack_three_n1500}"

python -m P1.experiments.run_e2_attack_wm_vs_atk_clean \
  --wav-dir "$WAV" \
  --out-dir "$OUT" \
  --suites mild,anchor \
  --methods audioseal,wavmark,band \
  --wm-mild-root "$WM_MILD" \
  --wm-anchor-root "$WM_ANCHOR" \
  --target-far 0.01 \
  --num-workers 8 \
  --npu-ids 0,1,2,3,4,5,6,7 \
  --cpu-threads 2
