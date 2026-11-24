#!/usr/bin/env bash


AUDIO_ROOT=/data1/home/ynian/codes/eval/audios
TEXT_JSON=/data1/home/ynian/codes/eval/add_exp/mix_1500_audio_text.json
OUT=/data1/home/ynian/codes/eval/audios/result/baselines_asr_wer_n1500
WHISPER=/data1/home/ynian/pred_models/whisper-large-v3
PARAFORMER=/data1/home/ynian/pred_models/paraformer_zh

python -m P1.experiments.run_baselines_asr_wer_n1500 \
  --audio-root "${AUDIO_ROOT}" \
  --text-json "${TEXT_JSON}" \
  --out_dir "${OUT}" \
  --whisper "${WHISPER}" \
  --paraformer "${PARAFORMER}" \
  --num-workers 8 \
  --npu-ids 0,1,2,3,4,5,6,7 \
  --cpu-threads 2
