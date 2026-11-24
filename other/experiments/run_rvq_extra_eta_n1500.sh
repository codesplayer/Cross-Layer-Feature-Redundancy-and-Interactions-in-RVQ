#!/usr/bin/env bash
# mix_1500 · MOSS + Qwen + DAC · η_extra / LSD / MCD · 8 进程 × npu 0–7
# 每条语音一个 JSON：${OUT}/{moss,qwen,dac}/{stem}.json
set -euo pipefail

ROOT_WM="${ROOT_WM:-/data1/home/ynian/codes/water_marks}"
ROOT_CA="${ROOT_CA:-/data1/home/ynian/codes/codec_an}"
WAV="${WAV:-/data1/home/ynian/codes/codec_an/exp_watermark/data/mix_1500}"
OUT="${OUT:-/data1/home/ynian/codes/eval/add_exp/result/rvq_extra_eta_n1500}"
PY="${PY:-python}"

export PYTHONPATH="${ROOT_WM}:${ROOT_CA}:${PYTHONPATH:-}"
export PYTHONIOENCODING=utf-8

cd "${ROOT_WM}"
echo "[launch] wav=${WAV} out=${OUT} workers=8"
"${PY}" -m other.experiments.run_rvq_extra_eta_n1500 \
  --codecs moss,qwen,dac \
  --wav-dir "${WAV}" \
  --out_dir "${OUT}" \
  --num-workers 8 \
  --npu-ids 0,1,2,3,4,5,6,7 \
  --cpu-threads 2 \
  "$@"
