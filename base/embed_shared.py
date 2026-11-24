"""嵌入算子。配对检测见 ``detect.py`` / ``works.py``。

统一范式：**剥浅层 → 在目标层残差注入 → 顺序重量化到末层 → 解码**。
三条工作禁止串用嵌入与检测：

- **P1**：ABA token 替换（``watermark`` 包，不在本模块）。
- **P2**：``embed_p2`` = 静态码字方向 ``static`` + 时间 PN ``time_pn``（抗变速）。
- **P3 抗克隆**：``embed_sign`` / ``embed_sign_multilayer``（二值图案 b）。
- **P3 仪器**（不是水印系统）：``embed_pn`` 多抽头探针。
"""

from __future__ import annotations

import numpy as np
import torch

from .moss_io import decode_moss
from .residual_feat import encode_hidden

from .io import match_len


def _rvq_head(model, wav):
    """公共前段：encode→input_proj→mask。返回 (quantizer, z, t_use, mask, device)。"""
    hidden, lengths = encode_hidden(model, wav)
    quantizer = model.quantizer
    device = hidden.device
    z = quantizer.input_proj(hidden).float()
    batch, _, max_time = z.shape
    t_use = min(int(lengths[0].item()), int(max_time))
    mask = torch.arange(max_time, device=device).expand(batch, max_time) < t_use
    return quantizer, z, t_use, mask, device


def _flip_frame(r: np.ndarray, b: np.ndarray, max_flips: int, gamma: float) -> np.ndarray:
    """对单帧残差 r [D,T]，把每帧 |r_d| 最小的不匹配维翻到 b_d*gamma*帧rms。"""
    rms = np.sqrt((r ** 2).mean(axis=0)) + 1e-12  # [T]
    sign_r = np.sign(r)
    mismatch = sign_r != b[:, None]
    order = np.argsort(np.abs(r), axis=0)  # 每帧按 |r_d| 升序
    r_new = r.copy()
    for t in range(r.shape[1]):
        cnt = 0
        for d in order[:, t]:
            if cnt >= max_flips:
                break
            if mismatch[d, t]:
                r_new[d, t] = b[d] * gamma * rms[t]
                cnt += 1
    return r_new


def embed_sign(model, wav: np.ndarray, b: np.ndarray, layer: int, max_flips: int, gamma: float) -> np.ndarray:
    """P3 抗克隆：单层符号翻转。b 由密钥再生（keyed_bits），不是干净语音。"""
    quantizer, z, t_use, mask, device = _rvq_head(model, wav)
    residual = z.clone().float()
    codes = []
    with torch.no_grad():
        for i, q in enumerate(quantizer.quantizers):
            if i == layer:
                r = residual[0, :, :t_use].detach().float().cpu().numpy().astype(np.float64)
                r_new = _flip_frame(r, b, int(max_flips), float(gamma))
                residual[0, :, :t_use] = torch.from_numpy(r_new).to(device=device, dtype=residual.dtype)
            z_q, indices, _ = q(residual * mask.unsqueeze(1))
            codes.append(indices[0, :t_use].detach().long().cpu().numpy())
            residual = residual - z_q * mask.unsqueeze(1)
    new_codes = np.stack(codes, axis=0)
    return np.asarray(match_len(wav, decode_moss(model, new_codes)), dtype=np.float32).reshape(-1)


def embed_sign_multilayer(model, wav: np.ndarray, b: np.ndarray, flips: dict, gamma: float) -> np.ndarray:
    """P3 抗克隆：多层符号翻转。``flips={layer: K}``，层间重量化。b 由密钥再生。"""
    quantizer, z, t_use, mask, device = _rvq_head(model, wav)
    residual = z.clone().float()
    codes = []
    with torch.no_grad():
        for i, q in enumerate(quantizer.quantizers):
            if i in flips:
                r = residual[0, :, :t_use].detach().float().cpu().numpy().astype(np.float64)
                r_new = _flip_frame(r, b, int(flips[i]), float(gamma))
                residual[0, :, :t_use] = torch.from_numpy(r_new).to(device=device, dtype=residual.dtype)
            z_q, indices, _ = q(residual * mask.unsqueeze(1))
            codes.append(indices[0, :t_use].detach().long().cpu().numpy())
            residual = residual - z_q * mask.unsqueeze(1)
    new_codes = np.stack(codes, axis=0)
    return np.asarray(match_len(wav, decode_moss(model, new_codes)), dtype=np.float32).reshape(-1)


def embed_pn(model, wav: np.ndarray, v: np.ndarray, layer: int, beta: float, s: np.ndarray, n_taps: int = 4):
    """P3 仪器（流动地图）：PN 扩频并记录抽头。水印系统请用 embed_p2(kind='time_pn')。"""
    quantizer, z, t_use, mask, device = _rvq_head(model, wav)
    vt = torch.from_numpy(np.asarray(v, dtype=np.float32)).to(device=device, dtype=z.dtype)
    sign = torch.from_numpy(np.asarray(s[:t_use], dtype=np.float32)).to(device=device, dtype=z.dtype)
    inj = z.new_zeros((1, z.shape[1], z.shape[2]))
    inj[0, :, :t_use] = float(beta) * vt.unsqueeze(1) * sign.unsqueeze(0)
    residual = z.clone().float()
    codes = []
    taps = {}
    with torch.no_grad():
        for i, q in enumerate(quantizer.quantizers):
            if i == layer:
                residual = residual + inj * mask.unsqueeze(1)
            if layer <= i <= layer + n_taps - 1:
                taps[f"enter_L{i}"] = residual[0, :, :t_use].detach().float().cpu().numpy()
            z_q, indices, _ = q(residual * mask.unsqueeze(1))
            codes.append(indices[0, :t_use].detach().long().cpu().numpy())
            residual = residual - z_q * mask.unsqueeze(1)
    new_codes = np.stack(codes, axis=0)
    out = np.asarray(match_len(wav, decode_moss(model, new_codes)), dtype=np.float32).reshape(-1)
    return out, taps


P2_KINDS = ("time_pn", "static")
P2_POST_MODES = ("freeze", "sub", "nosub")


def _make_inj(z, t_use, mask, device, v, w, beta, s, kind: str):
    """构造注入张量 [1,D,T]，与 ``embed_carrier`` 五种 kind 一致。"""

    def to_t(x):
        return torch.from_numpy(np.asarray(x, dtype=np.float32)).to(device=device, dtype=z.dtype)

    inj = z.new_zeros((1, z.shape[1], z.shape[2]))
    if kind == "time_pn":
        inj[0, :, :t_use] = float(beta) * to_t(v).unsqueeze(1) * to_t(s[:t_use]).unsqueeze(0)
    elif kind == "space_pn":
        inj[0, :, :t_use] = float(beta) * to_t(w).unsqueeze(1)
    elif kind == "spacetime":
        inj[0, :, :t_use] = float(beta) * to_t(w).unsqueeze(1) * to_t(s[:t_use]).unsqueeze(0)
    elif kind == "static":
        inj[0, :, :t_use] = float(beta) * to_t(v).unsqueeze(1)
    elif kind == "single":
        t_star = t_use // 2
        inj[0, :, t_star] = float(beta) * to_t(w)
    else:
        raise ValueError(kind)
    return inj * mask.unsqueeze(1)


def _clean_codes(quantizer, z, t_use, mask):
    """无注入的干净 RVQ 码流 [n_q, T]。"""
    residual = z.clone().float()
    codes = []
    with torch.no_grad():
        for q in quantizer.quantizers:
            z_q, indices, _ = q(residual * mask.unsqueeze(1))
            codes.append(indices[0, :t_use].detach().long().cpu().numpy())
            residual = residual - z_q * mask.unsqueeze(1)
    return np.stack(codes, axis=0)


def embed_carrier(model, wav: np.ndarray, v: np.ndarray, w: np.ndarray, layer: int, beta: float, s: np.ndarray, kind: str):
    """残差载体注入（对照实验五种 kind 的底层实现）。

    P2 水印系统只用 ``time_pn`` / ``static``，请走 ``embed_p2``。
    kind: time_pn(β·v·s_t) / space_pn(β·w) / spacetime(β·w·s_t) / static(β·v) / single(单帧 β·w)。
    默认后层处理 = 正确残差（sub）：量化该层后减 z_q，再顺序重量化到末层。
    """
    out, _, _ = embed_carrier_cond(model, wav, v, w, layer, beta, s, kind, mode="sub")
    return out


def embed_carrier_cond(
    model,
    wav: np.ndarray,
    v: np.ndarray,
    w: np.ndarray,
    layer: int,
    beta: float,
    s: np.ndarray,
    kind: str,
    mode: str = "sub",
    obs_beta: float | None = None,
):
    """条件残差对照的载体注入。

    ``mode``（``obs_beta is None`` 时）::

      - freeze：只改选定层码字，后层保持干净编码的原码字
      - sub：正确残差——减该层新 z_q，再从下一层量化到末层
      - nosub：错误残差消融——仍重量化后层，但不减该层贡献

    ``obs_beta`` 非空时覆盖 ``mode``：固定码流本层为水印码，残差按
    ``(1−β) q_orig + β q_wm`` 剥离后再量化后层。β=1 ≡ sub；β=0 表示
    后层残差看不到扰动 Δ。返回 ``(wav, codes, meta)``。
    """
    from .rvq_helpers import _zq_full

    if mode not in P2_POST_MODES and obs_beta is None:
        raise ValueError(f"mode must be freeze|sub|nosub, got {mode!r}")

    quantizer, z, t_use, mask, device = _rvq_head(model, wav)
    max_time = int(z.shape[2])
    L = int(layer)
    n_q = len(quantizer.quantizers)
    inj = _make_inj(z, t_use, mask, device, v, w, beta, s, kind)
    orig_codes = _clean_codes(quantizer, z, t_use, mask)
    new_codes = np.array(orig_codes, dtype=np.int64, copy=True)
    changed: dict = {}
    effective = str(mode)
    beta_used = None if obs_beta is None else float(np.clip(obs_beta, 0.0, 1.0))
    if beta_used is not None:
        effective = f"obs_beta{beta_used:g}"

    residual = z.clone().float()
    with torch.no_grad():
        for i, q in enumerate(quantizer.quantizers):
            if i < L:
                ids = (
                    torch.from_numpy(np.ascontiguousarray(orig_codes[i, :t_use]))
                    .long()
                    .view(1, t_use)
                    .to(device)
                )
                residual = residual - _zq_full(q, ids, max_time, mask)
                continue

            if i == L:
                residual = residual + inj
                z_q, indices, _ = q(residual * mask.unsqueeze(1))
                new_codes[i, :t_use] = indices[0, :t_use].detach().long().cpu().numpy()
                if effective == "freeze":
                    break
                if effective == "nosub":
                    continue
                if beta_used is not None:
                    ids_orig = (
                        torch.from_numpy(np.ascontiguousarray(orig_codes[L, :t_use]))
                        .long()
                        .view(1, t_use)
                        .to(device)
                    )
                    z_q_orig = _zq_full(q, ids_orig, max_time, mask)
                    blended = (1.0 - beta_used) * z_q_orig + beta_used * (z_q * mask.unsqueeze(1))
                    residual = residual - blended
                else:
                    # sub
                    residual = residual - z_q * mask.unsqueeze(1)
                continue

            # post layers: requantize
            z_q, indices, _ = q(residual * mask.unsqueeze(1))
            idx = indices[0, :t_use].detach().long().cpu().numpy()
            old = new_codes[i, :t_use].copy()
            new_codes[i, :t_use] = idx
            residual = residual - z_q * mask.unsqueeze(1)
            changed[int(i)] = {
                "n_diff": int(np.sum(old != idx)),
                "T": int(t_use),
                "frac_diff": float(np.mean(old != idx)),
            }

    out = np.asarray(match_len(wav, decode_moss(model, new_codes)), dtype=np.float32).reshape(-1)
    meta = {
        "layer": L,
        "kind": kind,
        "mode": effective,
        "obs_beta": beta_used,
        "t_use": int(t_use),
        "n_q": int(n_q),
        "post_layers_changed": changed,
        "mean_frac_diff_post": float(
            np.mean([v["frac_diff"] for v in changed.values()]) if changed else 0.0
        ),
        "token_unchanged_at_L": bool(np.all(new_codes[L, :t_use] == orig_codes[L, :t_use])),
    }
    return out, new_codes, meta


def embed_p2(model, wav: np.ndarray, v: np.ndarray, layer: int, beta: float, s: np.ndarray, kind: str) -> np.ndarray:
    """P2 抗变速水印：只允许 static（静态码字方向）与 time_pn（时间 PN）。

    检测必须用 ``detect_p2``（本层 tap0）。不要用符号 Jaccard，也不要用多抽头。
    默认后层 = 正确残差（sub）。条件残差对照请用 ``embed_p2_cond``。
    """
    if kind not in P2_KINDS:
        raise ValueError(f"P2 embed kind must be time_pn|static, got {kind!r}")
    unused_w = np.zeros_like(np.asarray(v, dtype=np.float64))
    return embed_carrier(model, wav, v, unused_w, layer, beta, s, kind)


def embed_p2_cond(
    model,
    wav: np.ndarray,
    v: np.ndarray,
    layer: int,
    beta: float,
    s: np.ndarray,
    kind: str,
    mode: str = "sub",
    obs_beta: float | None = None,
):
    """P2 条件残差对照：``freeze`` / ``sub`` / ``nosub``（可选 ``obs_beta``）。

    检测必须用 ``detect_p2``。返回 ``(wav, codes, meta)``。
    """
    if kind not in P2_KINDS:
        raise ValueError(f"P2 embed kind must be time_pn|static, got {kind!r}")
    unused_w = np.zeros_like(np.asarray(v, dtype=np.float64))
    return embed_carrier_cond(
        model, wav, v, unused_w, layer, beta, s, kind, mode=mode, obs_beta=obs_beta
    )
