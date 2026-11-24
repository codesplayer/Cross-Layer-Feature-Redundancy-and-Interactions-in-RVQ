"""MOSS RVQ walk / 简易波形 LSD（从旧 n3 探针抽出，供 n1500 脚本引用）。"""

from __future__ import annotations

import numpy as np
import torch


def walk_rvq(model, hidden, lengths, delta_at: dict[int, torch.Tensor] | None = None):
    """返回 codes [n_q, T]、各层 z_q [D, T]、各层量化前残差 [D, T]。"""
    quantizer = model.quantizer
    n_q = len(quantizer.quantizers)
    z = quantizer.input_proj(hidden)
    batch, _, max_time = z.shape
    t_use = min(int(lengths[0].item()), int(max_time))
    mask = torch.arange(max_time, device=z.device).expand(batch, max_time) < t_use
    residual = z.clone().float()
    codes = np.zeros((n_q, t_use), dtype=np.int64)
    zqs: list[np.ndarray] = []
    pres: list[np.ndarray] = []
    delta_at = delta_at or {}
    with torch.no_grad():
        for i in range(n_q):
            pres.append(residual[0, :, :t_use].detach().float().cpu().numpy())
            masked = residual * mask.unsqueeze(1)
            if i in delta_at:
                delta = delta_at[i]
                if delta.shape[-1] < residual.shape[-1]:
                    pad = torch.zeros(
                        delta.shape[0],
                        delta.shape[1],
                        residual.shape[-1] - delta.shape[-1],
                        device=delta.device,
                        dtype=delta.dtype,
                    )
                    delta = torch.cat([delta, pad], dim=-1)
                elif delta.shape[-1] > residual.shape[-1]:
                    delta = delta[..., : residual.shape[-1]]
                z_q_i, indices, _ = quantizer.quantizers[i]((masked - delta) * mask.unsqueeze(1))
                residual = residual - delta - z_q_i * mask.unsqueeze(1) + delta
            else:
                z_q_i, indices, _ = quantizer.quantizers[i](masked)
                residual = residual - z_q_i * mask.unsqueeze(1)
            codes[i] = indices[0, :t_use].detach().long().cpu().numpy()
            zqs.append(z_q_i[0, :, :t_use].detach().float().cpu().numpy())
    return codes, zqs, pres


def wave_lsd(a: np.ndarray, b: np.ndarray, sr: int) -> float:
    import librosa

    n = min(int(a.shape[0]), int(b.shape[0]))
    sa = np.abs(librosa.stft(np.asarray(a[:n], dtype=np.float32), n_fft=1024, hop_length=256)) + 1e-8
    sb = np.abs(librosa.stft(np.asarray(b[:n], dtype=np.float32), n_fft=1024, hop_length=256)) + 1e-8
    t = min(sa.shape[1], sb.shape[1])
    diff = 20.0 * np.log10(sa[:, :t]) - 20.0 * np.log10(sb[:, :t])
    return float(np.sqrt(np.mean(diff ** 2)))
