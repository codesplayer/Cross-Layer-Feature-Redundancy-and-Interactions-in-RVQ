#!/usr/bin/env bash
# mix_1500 · MOSS + Qwen + DAC · 实验 A–F · 8 进程 × npu 0–7
# 每个实验一个文件夹，每条语音一个 JSON：
#   ${OUT}/{A,B,C,D,E,F}/{moss,qwen,dac}/{stem}.json
set -euo pipefail

ROOT_WM="${ROOT_WM:-/data1/home/ynian/codes/water_marks}"
ROOT_CA="${ROOT_CA:-/data1/home/ynian/codes/codec_an}"
WAV="${WAV:-/data1/home/ynian/codes/codec_an/exp_watermark/data/mix_1500}"
OUT="${OUT:-/data1/home/ynian/codes/eval/add_exp/result/rvq_af_n1500}"
PY="${PY:-python}"

export PYTHONPATH="${ROOT_WM}:${ROOT_CA}:${PYTHONPATH:-}"
export PYTHONIOENCODING=utf-8

cd "${ROOT_WM}"
echo "[launch] wav=${WAV} out=${OUT} workers=8"
"${PY}" -m other.experiments.run_rvq_af_n1500 \
  --codecs moss,qwen,dac \
  --experiments A,B,C,D,E,F \
  --wav-dir "${WAV}" \
  --out_dir "${OUT}" \
  --num-workers 8 \
  --npu-ids 0,1,2,3,4,5,6,7 \
  --cpu-threads 2 \
  "$@"
