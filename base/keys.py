"""密钥图案与载体向量。"""

from __future__ import annotations

import numpy as np
import torch

from .residual_feat import layer_zq_from_ids

from .config import KEY


def keyed_bits(dim: int, key: int = KEY) -> np.ndarray:
    """P3 抗克隆图案 b ∈ {±1}^D。检测端用同一 key 再生，不是干净语音。"""
    rng = np.random.default_rng(int(key))
    return (rng.integers(0, 2, size=int(dim)) * 2 - 1).astype(np.float64)


def pn_seq(n: int, key: int = KEY) -> np.ndarray:
    """P2 时间 PN 序列 s ∈ {±1}^n；P3 仪器也用同一形式。"""
    rng = np.random.default_rng(int(key))
    return (rng.integers(0, 2, size=int(n)) * 2 - 1).astype(np.float64)


def carrier_w(model, layer: int, token: int) -> np.ndarray:
    """P2 静态码字方向：层 layer 某 token 的 z_q（密钥选定的载体 v）。"""
    q = model.quantizer.quantizers[int(layer)]
    device = next(q.parameters()).device
    ids = torch.tensor([[int(token)]], dtype=torch.long, device=device)
    with torch.no_grad():
        z = layer_zq_from_ids(q, ids)
    return z[0, :, 0].detach().float().cpu().numpy().astype(np.float64)
