#!/usr/bin/env bash
# mix_1500 · MOSS ABA 单层 / 二层 / L4–L14 丢层 · 保存语音 · 8×NPU

#名称	内容
#clean
#不写水印的干净重构。各层都是原码，用来和后面的含水印语音对比。
#single
#只替换一层。目录名是 L04、L05 这种，该层用自己的 rare ABA。
#pair
#只替换两层。目录名是 L04_L05 这种，两层各用自己的 rare ABA。
#band_L04_L14
#L4 到 L14 全部写入，而且每一层都用 L9 的同一对 ABA。
#skip
#仍在 L4–L14 里写 L9 的 ABA，但目录名里的那一层或两层不写、保持原码。skip/L04 是只留出 L4；skip/L04_L05 是留出 L4 和 L5。
#conditions.json
#这次实际使用的层名单：单层、两层组合、L4–L14，以及 skip 里留出的那些组。
#logs
#8 个进程各自的运行日志。
#wav
#启动时自动建的空目录，这次的语音没有写在这里。

WAV=/data1/home/ynian/codes/codec_an/exp_watermark/data/mix_1500
OUT=/data1/home/ynian/codes/eval/add_exp/result/moss_aba_emotion_wav_n1500

echo "[launch] wav=${WAV} out=${OUT}"
python -m P1.experiments.run_moss_aba_emotion_wav_n1500 \
  --wav-dir "${WAV}" \
  --out_dir "${OUT}" \
  --num-workers 8 \
  --npu-ids 0,1,2,3,4,5,6,7 \
  --cpu-threads 2
