"""把 L 层水印特征按权重 β 加到该层 encode 特征上。

peel L0..L−1 之后，层 L 的 encode 残差是 r。
水印特征 f = q_wm（该层 ABA 的 z_q）。

  r_seen = r + β f

β=0：不加。β 越大，encode 特征上的水印越强。
从 L 起按 r_seen 正常量化。不把水印 token 写进 L 层码本。
检测再编码后看同一层的 z_q。
"""

from __future__ import annotations

import numpy as np
import torch

from .moss_io import decode_moss
from .io import match_len as _match_len
from .rvq_helpers import _zq_full, embed_layer_tokens
from .residual_feat import encode_hidden


def _decode(model, wav: np.ndarray, codes: np.ndarray) -> np.ndarray:
    return np.asarray(_match_len(wav, decode_moss(model, codes)), dtype=np.float32).reshape(-1)


def _flat_cos(a: torch.Tensor, b: torch.Tensor) -> float:
    x = a.detach().float().reshape(-1)
    y = b.detach().float().reshape(-1)
    n = min(int(x.numel()), int(y.numel()))
    if n <= 0:
        return 0.0
    x = x[:n]
    y = y[:n]
    denom = torch.linalg.vector_norm(x) * torch.linalg.vector_norm(y)
    if float(denom) <= 0:
        return 0.0
    return float(torch.dot(x, y).item() / float(denom))


def beta_embed_layer(
    model,
    wav: np.ndarray,
    table: dict,
    *,
    wm_layer: int = 9,
    beta: float = 1.0,
    period: int = 3,
) -> dict:
    """把 β·q_wm 加到层 L 的 encode 残差上。β 越大水印越强。"""
    beta = float(np.clip(beta, 0.0, 1.0))
    L = int(wm_layer)
    tokens_orig, mod_wm, plan, n_changed = embed_layer_tokens(
        model, wav, table, L, period=int(period)
    )
    hidden, lengths = encode_hidden(model, wav)
    quantizer = model.quantizer
    n_q = len(quantizer.quantizers)
    device = hidden.device
    z = quantizer.input_proj(hidden)
    batch, _, max_time = z.shape
    t_codes = int(mod_wm.shape[1])
    t_use = min(int(lengths[0].item()), t_codes, int(max_time), int(tokens_orig.shape[1]))
    mask = torch.arange(max_time, device=device).expand(batch, max_time) < t_use

    new_codes = np.array(tokens_orig, dtype=np.int64, copy=True)
    residual = z.clone().float()

    with torch.no_grad():
        for i in range(L):
            q = quantizer.quantizers[i]
            ids = (
                torch.from_numpy(np.ascontiguousarray(tokens_orig[i, :t_use]))
                .long()
                .view(1, t_use)
                .to(device)
            )
            residual = residual - _zq_full(q, ids, max_time, mask)

        qL = quantizer.quantizers[L]
        ids_wm = (
            torch.from_numpy(np.ascontiguousarray(mod_wm[L, :t_use]))
            .long()
            .view(1, t_use)
            .to(device)
        )
        f_wm = _zq_full(qL, ids_wm, max_time, mask)
        # L 层特征按权重 β 加到 encode 残差上
        seen = residual + beta * f_wm
        z_q_L, indices, _ = qL(seen * mask.unsqueeze(1))
        idx = indices[0, :t_use].detach().long().cpu().numpy()
        new_codes[L, :t_use] = idx
        residual = seen - z_q_L * mask.unsqueeze(1)
        obs_cos = _flat_cos(z_q_L * mask.unsqueeze(1), f_wm)

        changed: dict = {}
        for i in range(L + 1, n_q):
            q = quantizer.quantizers[i]
            masked = residual * mask.unsqueeze(1)
            z_q_i, indices_i, _ = q(masked)
            idx_i = indices_i[0, :t_use].detach().long().cpu().numpy()
            old = new_codes[i, :t_use].copy()
            new_codes[i, :t_use] = idx_i
            residual = residual - z_q_i * mask.unsqueeze(1)
            changed[int(i)] = {
                "n_diff": int(np.sum(old != idx_i)),
                "T": int(t_use),
                "frac_diff": float(np.mean(old != idx_i)),
            }

    agree_wm = float(np.mean(new_codes[L, :t_use] == mod_wm[L, :t_use])) if t_use else 0.0
    agree_orig = float(np.mean(new_codes[L, :t_use] == tokens_orig[L, :t_use])) if t_use else 0.0
    meta = {
        "t_use": int(t_use),
        "wm_layer": L,
        "beta": beta,
        "delta_norm": float(torch.linalg.vector_norm(f_wm).item()),
        "codebook_obs_cos": obs_cos,
        "post_layers_changed": changed,
        "mean_frac_diff_post": float(
            np.mean([v["frac_diff"] for v in changed.values()]) if changed else 0.0
        ),
        "L_agree_wm": agree_wm,
        "L_agree_orig": agree_orig,
        "formula": "r_seen = r_encode + beta*q_wm at L; quantize L..; codes[L] not assigned",
    }
    return {
        "tokens_orig": tokens_orig,
        "tokens_wm_plan": mod_wm,
        "wav": _decode(model, wav, new_codes),
        "codes": new_codes,
        "meta": meta,
        "plan": plan,
        "n_changed": int(n_changed),
        "wm_layer": L,
        "beta": beta,
    }
