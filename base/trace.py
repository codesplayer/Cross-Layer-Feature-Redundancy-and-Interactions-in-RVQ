"""逐层量化前残差追踪——所有残差域检测的基础。"""

from __future__ import annotations

import torch

from .residual_feat import encode_hidden  # noqa: F401


def trace_enter(model, wav):
    """逐层量化前残差。

    返回 (enters, qs, t_use)：
      enters [L][D,T]  每层量化器**输入**残差（检测在这里取特征）；
      qs     [L][D,T]  每层量化器输出 z_q；
      t_use  int       有效帧数。
    """
    hidden, lengths = encode_hidden(model, wav)
    quantizer = model.quantizer
    z = quantizer.input_proj(hidden).float()
    batch, _, max_time = z.shape
    t_use = min(int(lengths[0].item()), int(max_time))
    device = z.device
    mask = torch.arange(max_time, device=device).expand(batch, max_time) < t_use
    residual = z.clone().float()
    enters, qs = [], []
    with torch.no_grad():
        for q in quantizer.quantizers:
            enters.append(residual[0, :, :t_use].detach().float().cpu().numpy())
            z_q, indices, _ = q(residual * mask.unsqueeze(1))
            qs.append(z_q[0, :, :t_use].detach().float().cpu().numpy())
            residual = residual - z_q * mask.unsqueeze(1)
    return enters, qs, int(t_use)
