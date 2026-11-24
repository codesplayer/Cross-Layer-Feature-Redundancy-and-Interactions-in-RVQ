"""P1 多通道检测的精简占位。完整级联检测请用历史 codec_an/watermark 结果；新实验优先 detect_aba。"""

from __future__ import annotations


def compact_out(det: dict) -> dict:
    scores = det.get("scores") or {}
    return {"scores": {k: float(scores.get(k) or 0.0) for k in ("aba_embed", "slide_v3", "wave")}}


def score_wav_channels(*args, **kwargs):
    raise NotImplementedError(
        "完整 score_wav_channels 未迁入独立包；P1 请用 P1.detect.detect_aba / base.detect_p1_cosine"
    )


def cascade_decide_scores(*args, **kwargs):
    raise NotImplementedError("cascade_decide_scores 未迁入；见 codec_an/watermark/future_exp/_multi_channel.py")
