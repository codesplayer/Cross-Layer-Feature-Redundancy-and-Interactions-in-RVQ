"""P1 共用 RVQ / 几何小工具（从旧 n3 探针抽出，供规模脚本引用）。"""

from __future__ import annotations

import numpy as np
import torch

from base.residual_feat import encode_hidden


def n_codes_of(q) -> int:
    cb = getattr(q, "codebook", None)
    if cb is not None and hasattr(cb, "weight"):
        return int(cb.weight.shape[0])
    if cb is not None and hasattr(cb, "embed"):
        return int(cb.embed.shape[0])
    if hasattr(q, "codebook_size"):
        return int(q.codebook_size)
    return 1024


def rvq_trace(model, wav: np.ndarray) -> dict:
    """返回 k [L,T]、q [L,D,T]、r [L,D,T]。r 是当前层减去 q 之后的残差。"""
    hidden, lengths = encode_hidden(model, wav)
    quantizer = model.quantizer
    z = quantizer.input_proj(hidden)
    batch, _, max_time = z.shape
    t_use = min(int(lengths[0].item()), int(max_time))
    device = z.device
    mask = torch.arange(max_time, device=device).expand(batch, max_time) < t_use
    residual = z.clone().float()
    ks, qs, rs = [], [], []
    with torch.no_grad():
        for q in quantizer.quantizers:
            z_q, indices, _ = q(residual * mask.unsqueeze(1))
            ks.append(indices[0, :t_use].detach().long().cpu().numpy())
            qs.append(z_q[0, :, :t_use].detach().float().cpu().numpy())
            residual = residual - z_q * mask.unsqueeze(1)
            rs.append(residual[0, :, :t_use].detach().float().cpu().numpy())
    return {
        "k": np.stack(ks, axis=0),
        "q": np.stack(qs, axis=0),
        "r": np.stack(rs, axis=0),
        "T": int(t_use),
    }


def frame_cosine(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """逐帧余弦。[D,T] → [T]。两边都接近 0 时记为 1。"""
    a64 = a.astype(np.float64)
    b64 = b.astype(np.float64)
    an = np.linalg.norm(a64, axis=0)
    bn = np.linalg.norm(b64, axis=0)
    dot = np.sum(a64 * b64, axis=0)
    out = np.empty(a.shape[1], dtype=np.float64)
    both = (an < 1e-8) & (bn < 1e-8)
    ok = (~both) & (an * bn > 1e-8)
    out[both] = 1.0
    out[~both & ~ok] = 0.0
    out[ok] = dot[ok] / (an[ok] * bn[ok])
    return out


def pearson(a: np.ndarray, b: np.ndarray) -> float:
    x = np.asarray(a, dtype=np.float64).reshape(-1)
    y = np.asarray(b, dtype=np.float64).reshape(-1)
    x = x - x.mean()
    y = y - y.mean()
    denom = float(np.linalg.norm(x) * np.linalg.norm(y))
    if denom <= 1e-12:
        return float("nan")
    return float(np.dot(x, y) / denom)


def uncentered_cos(a: np.ndarray, b: np.ndarray) -> float:
    x = np.asarray(a, dtype=np.float64).reshape(-1)
    y = np.asarray(b, dtype=np.float64).reshape(-1)
    denom = float(np.linalg.norm(x) * np.linalg.norm(y))
    if denom <= 1e-12:
        return float("nan")
    return float(np.dot(x, y) / denom)
