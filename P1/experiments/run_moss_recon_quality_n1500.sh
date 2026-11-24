#!/usr/bin/env bash
# MOSS 重构 + LSD/MCD/spk/emo/DNSMOS；保存 wav_recon
set -euo pipefail

WAV="${WAV:-/data1/home/ynian/codes/codec_an/exp_watermark/data/mix_1500}"
OUT="${OUT:-/data1/home/ynian/codes/eval/add_exp/result/moss_recon_quality_n1500}"

python -m P1.experiments.run_moss_recon_quality_n1500 \
  --wav-dir "$WAV" \
  --out-dir "$OUT" \
  --num-workers 8 \
  --npu-ids 0,1,2,3,4,5,6,7 \
  --cpu-threads 2
