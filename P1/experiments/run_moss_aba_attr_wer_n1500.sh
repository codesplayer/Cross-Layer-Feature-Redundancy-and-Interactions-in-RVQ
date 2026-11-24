#!/usr/bin/env bash
# 对 moss_aba_emotion_wav_n1500 的语音计算说话人余弦、情感余弦。本轮不算 WER。8×NPU。


WAV=/data1/home/ynian/codes/codec_an/exp_watermark/data/mix_1500
WAV_ROOT=/data1/home/ynian/codes/eval/add_exp/result/moss_aba_emotion_wav_n1500
OUT=/data1/home/ynian/codes/eval/add_exp/result/moss_aba_attr_n1500

python -m P1.experiments.run_moss_aba_attr_wer_n1500 \
  --wav-dir "${WAV}" \
  --wav-root "${WAV_ROOT}" \
  --out_dir "${OUT}" \
  --num-workers 8 \
  --npu-ids 0,1,2,3,4,5,6,7 \
  --cpu-threads 2
