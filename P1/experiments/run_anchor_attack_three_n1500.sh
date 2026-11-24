#!/usr/bin/env bash
# P1：Anchor 十类攻击 × audioseal + wavmark + band × mix_1500 × 8×NPU
set -euo pipefail
export PYTHONPATH="${PYTHONPATH:-$(cd "$(dirname "$0")/../.." && pwd)}"
# AudioSeal/moshi torch.compile → Inductor 不支持 NPU
export TORCHDYNAMO_DISABLE="${TORCHDYNAMO_DISABLE:-1}"
export TORCH_COMPILE_DISABLE="${TORCH_COMPILE_DISABLE:-1}"

python -m P1.experiments.run_anchor_attack_three_n1500 \
  --methods audioseal,wavmark,band \
  --num-workers 8 \
  --npu-ids 0,1,2,3,4,5,6,7 \
  --cpu-threads 1
