#!/usr/bin/env bash
# 75 组 ABA 语音：中文 Paraformer / 英文 Whisper，相对 mix text 算 WER+CER。8×NPU。


WAV_ROOT=/data1/home/ynian/codes/eval/add_exp/result/moss_aba_emotion_wav_n1500
TEXT_JSON=/data1/home/ynian/codes/codec_an/exp_watermark/data/mix_1500_audio_text.json
OUT=/data1/home/ynian/codes/eval/add_exp/result/moss_aba_asr_wer_n1500
WHISPER=/data1/home/ynian/pred_models/whisper-large-v3
PARAFORMER=/data1/home/ynian/pred_models/paraformer_zh

python -m P1.experiments.run_moss_aba_asr_wer_n1500 \
  --wav-root "${WAV_ROOT}" \
  --text-json "${TEXT_JSON}" \
  --out_dir "${OUT}" \
  --whisper "${WHISPER}" \
  --paraformer "${PARAFORMER}" \
  --num-workers 8 \
  --npu-ids 0,1,2,3,4,5,6,7 \
  --cpu-threads 2 \
