"""音频读写与 MOSS 编解码、长度对齐。

约定：中间 wav 一律以 ``subtype="FLOAT"`` 写盘，避免 PCM 量化污染残差统计。
"""

from __future__ import annotations

import numpy as np
import soundfile as sf

from .moss_io import decode_moss, encode_moss, read_wav  # noqa: F401


def write_wav_float(path, wav, sr) -> None:
    """以 FLOAT subtype 写单声道 wav。"""
    sf.write(str(path), np.asarray(wav, dtype=np.float32).reshape(-1), int(sr), subtype="FLOAT")


def match_len(ref, wav):
    """把 wav 截断/零填充到与 ref 等长（MOSS 解码长度常与输入差几帧）。"""
    n = int(np.asarray(ref).reshape(-1).shape[0])
    w = np.asarray(wav, dtype=np.float32).reshape(-1)
    if w.shape[0] == n:
        return w
    if w.shape[0] > n:
        return w[:n]
    return np.pad(w, (0, n - w.shape[0]))


def clean_decode(model, wav):
    """encode→decode 往返（干净基线），长度对齐到原 wav。"""
    return np.asarray(match_len(wav, decode_moss(model, encode_moss(model, wav))), dtype=np.float32).reshape(-1)


def _zq_full(q, ids_1t, max_time: int, mask):
    """按给定 token 序列取整层 z_q，并对齐到 max_time（comp-sub 嵌入用）。"""
    import torch

    from .residual_feat import layer_zq_from_ids

    z_q = layer_zq_from_ids(q, ids_1t)
    if z_q.shape[-1] < max_time:
        pad = torch.zeros(
            (1, z_q.shape[1], max_time - z_q.shape[-1]),
            device=z_q.device,
            dtype=z_q.dtype,
        )
        z_q_full = torch.cat([z_q, pad], dim=-1)
    else:
        z_q_full = z_q[..., :max_time]
    return z_q_full * mask.unsqueeze(1)
