#!/usr/bin/env bash
# Latent-Mark 抗 MOSS / Qwen codec 实验
# 需先：conda activate latentmark（或已安装 snac/torchaudio + water_marks 的 moss/qwen）
set -euo pipefail

WAV_DIR="${WAV_DIR:-/data1/home/ynian/codes/codec_an/exp_watermark/data/mix_1500}"
OUT="${OUT:-/data1/home/ynian/codes/eval/add_exp/result/latentmark_vs_moss_qwen}"
DEVICE="${DEVICE:-cuda}"
N_CALIB="${N_CALIB:-40}"
N_TEST="${N_TEST:-50}"
LATENTMARK_SRC="${LATENTMARK_SRC:-/data1/home/ynian/codes/Latent-Mark/watermark_research/src}"

python -m P1.experiments.run_latentmark_vs_moss_qwen \
  --wav-dir "${WAV_DIR}" \
  --n-calib "${N_CALIB}" \
  --n-test "${N_TEST}" \
  --codecs moss,qwen \
  --views snac_24 \
  --axis cluster \
  --device "${DEVICE}" \
  --latentmark-src "${LATENTMARK_SRC}" \
  --out-dir "${OUT}"
