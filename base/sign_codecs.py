"""P3 符号翻转 / 检测：MOSS、DAC、Qwen 共用残差接口。

嵌入仍是剥浅层 → 在目标层残差翻转或加 α·v → 后层按正确残差重量化 → 解码。
检测是重编码后该层量化前残差对密钥图案 b 的 Jaccard。b 由 keyed_bits 再生。
"""

from __future__ import annotations

import numpy as np
import torch

from .codec_dac import (
    decode_codes as dac_decode,
    layer_zq_from_codes as dac_layer_zq,
    wav_to_batch as dac_wav_to_batch,
)
from .codec_qwen import (
    decode_codes_kt,
    embeddings as qwen_embeddings,
    layer_zq_from_ids as qwen_layer_zq,
    waveform_values as qwen_waveform_values,
)
from .detect_shared import jaccard_and_agree
from .embed_shared import _flip_frame
from .io import match_len
from .keys import carrier_w, keyed_bits
from .moss_io import decode_moss
from .residual_feat import encode_hidden
from .trace import trace_enter


def n_q_of(codec: str, handle) -> int:
    codec = str(codec).lower()
    if codec == "moss":
        return int(len(handle.quantizer.quantizers))
    if codec == "qwen":
        return int(handle.model.encoder_valid_num_quantizers)
    return int(handle.quantizer.n_codebooks)


def _np(x: torch.Tensor) -> np.ndarray:
    return x.detach().float().cpu().numpy()


def _put_frame(residual: torch.Tensor, r_new: np.ndarray) -> torch.Tensor:
    out = residual.clone()
    out[0] = torch.from_numpy(np.asarray(r_new, dtype=np.float32)).to(
        device=residual.device, dtype=residual.dtype
    )
    return out


def _align_dt(r: np.ndarray, dim: int) -> tuple[np.ndarray, bool]:
    r = np.asarray(r, dtype=np.float64)
    if r.ndim != 2:
        raise ValueError(f"residual rank {r.ndim}")
    if r.shape[0] == dim:
        return r, False
    if r.shape[1] == dim:
        return r.T, True
    raise ValueError(f"residual {r.shape} vs dim {dim}")


def _b_at(b, layer: int) -> np.ndarray:
    if isinstance(b, dict):
        return np.asarray(b[int(layer)], dtype=np.float64).reshape(-1)
    return np.asarray(b, dtype=np.float64).reshape(-1)


def _flip_residual(residual: torch.Tensor, b: np.ndarray, k: int, gamma: float) -> torch.Tensor:
    b = np.asarray(b, dtype=np.float64).reshape(-1)
    r, trans = _align_dt(_np(residual[0]), int(b.size))
    r_new = _flip_frame(r, b, min(int(k), int(b.size)), float(gamma))
    if trans:
        r_new = r_new.T
    return _put_frame(residual, r_new)


def _add_static(residual: torch.Tensor, v: np.ndarray, alpha: float) -> torch.Tensor:
    v = np.asarray(v, dtype=np.float64).reshape(-1)
    r, trans = _align_dt(_np(residual[0]), int(v.size))
    r_new = r + float(alpha) * v[:, None]
    if trans:
        r_new = r_new.T
    return _put_frame(residual, r_new)


def _vec_from_zq(arr: np.ndarray) -> np.ndarray:
    a = np.asarray(arr, dtype=np.float64)
    if a.ndim == 1:
        return a
    if a.ndim == 2:
        if a.shape[1] == 1:
            return a[:, 0]
        if a.shape[0] == 1:
            return a[0]
        return a.mean(axis=-1)
    return a.reshape(-1)


def codebook_vec(codec: str, handle, layer: int, token: int) -> np.ndarray:
    codec = str(codec).lower()
    L = int(layer)
    tok = int(token)
    if codec == "moss":
        return np.asarray(carrier_w(handle, L, tok), dtype=np.float64).reshape(-1)
    if codec == "dac":
        device = next(handle.parameters()).device
        n_q = int(handle.quantizer.n_codebooks)
        codes = torch.zeros((n_q, 1), dtype=torch.long, device=device)
        codes[L, 0] = tok
        return _vec_from_zq(dac_layer_zq(handle, codes, L))
    device = next(handle.model.parameters()).device
    ids = torch.tensor([tok], dtype=torch.long, device=device)
    return _vec_from_zq(qwen_layer_zq(handle, ids, L))


def residual_dim(codec: str, handle, layer: int = 0) -> int:
    return int(codebook_vec(codec, handle, int(layer), 0).size)


def bits_for(codec: str, handle, layer: int, key: int) -> np.ndarray:
    return keyed_bits(residual_dim(codec, handle, int(layer)), int(key))


def _moss_head(model, wav):
    hidden, lengths = encode_hidden(model, wav)
    quantizer = model.quantizer
    device = hidden.device
    z = quantizer.input_proj(hidden).float()
    batch, _, max_time = z.shape
    t_use = min(int(lengths[0].item()), int(max_time))
    mask = torch.arange(max_time, device=device).expand(batch, max_time) < t_use
    return quantizer, z, t_use, mask, device


def _moss_embed(model, wav, *, flips=None, statics=None, b=None, gamma=0.4):
    flips = {int(k): int(v) for k, v in (flips or {}).items()}
    statics = {int(k): v for k, v in (statics or {}).items()}
    quantizer, z, t_use, mask, device = _moss_head(model, wav)
    residual = z.clone().float()
    codes = []
    with torch.no_grad():
        for i, q in enumerate(quantizer.quantizers):
            if i in flips:
                residual = _flip_residual(residual, _b_at(b, i), flips[i], gamma)
            if i in statics:
                v, alpha = statics[i]
                residual = _add_static(residual, v, alpha)
            z_q, indices, _ = q(residual * mask.unsqueeze(1))
            codes.append(indices[0, :t_use].detach().long().cpu().numpy())
            residual = residual - z_q * mask.unsqueeze(1)
    new_codes = np.stack(codes, axis=0)
    return np.asarray(match_len(wav, decode_moss(model, new_codes)), dtype=np.float32).reshape(-1)


def _moss_enters(model, wav) -> list[np.ndarray]:
    enters, _, _ = trace_enter(model, wav)
    return [np.asarray(e, dtype=np.float64) for e in enters]


def _dac_embed(model, wav, sr, *, flips=None, statics=None, b=None, gamma=0.4):
    flips = {int(k): int(v) for k, v in (flips or {}).items()}
    statics = {int(k): v for k, v in (statics or {}).items()}
    x = dac_wav_to_batch(model, wav, sr)
    with torch.inference_mode():
        z = model.encoder(x).float()
        residual = z.clone()
        codes = []
        for i, q in enumerate(model.quantizer.quantizers):
            if i in flips:
                residual = _flip_residual(residual, _b_at(b, i), flips[i], gamma)
            if i in statics:
                v, alpha = statics[i]
                residual = _add_static(residual, v, alpha)
            z_q_i, _cmt, _cbl, indices, _ze = q(residual)
            codes.append(indices[0].long())
            residual = residual - z_q_i
        new_codes = torch.stack(codes, 0)
    return np.asarray(match_len(wav, dac_decode(model, new_codes)), dtype=np.float32).reshape(-1)


def _dac_enters(model, wav, sr) -> list[np.ndarray]:
    x = dac_wav_to_batch(model, wav, sr)
    enters = []
    with torch.inference_mode():
        residual = model.encoder(x).float()
        for q in model.quantizer.quantizers:
            enters.append(np.asarray(_np(residual[0]), dtype=np.float64))
            z_q_i, _cmt, _cbl, indices, _ze = q(residual)
            residual = residual - z_q_i
    return enters


def _qwen_branches(tok, n_q: int):
    q = tok.model.encoder.quantizer
    n_sem = int(q.num_semantic_quantizers)
    sem = list(range(0, min(n_sem, n_q)))
    ac = list(range(n_sem, n_q))
    return q, n_sem, sem, ac


def _qwen_embed(tok, wav, sr, *, flips=None, statics=None, b=None, gamma=0.4):
    flips = {int(k): int(v) for k, v in (flips or {}).items()}
    statics = {int(k): v for k, v in (statics or {}).items()}
    n_q = n_q_of("qwen", tok)
    q, n_sem, sem_layers, ac_layers = _qwen_branches(tok, n_q)
    values = qwen_waveform_values(tok, wav, sr)
    rows: dict[int, torch.Tensor] = {}
    with torch.inference_mode():
        emb = qwen_embeddings(tok, values)

        def run_branch(rvq, global_layers):
            if not global_layers:
                return
            h = emb
            if rvq.input_proj is not None:
                h = rvq.input_proj(h)
            residual = h
            for loc, gL in enumerate(global_layers):
                layer_m = rvq.layers[loc]
                if gL in flips:
                    residual = _flip_residual(residual, _b_at(b, gL), flips[gL], gamma)
                if gL in statics:
                    v, alpha = statics[gL]
                    residual = _add_static(residual, v, alpha)
                idx = layer_m.encode(residual)
                if isinstance(idx, (tuple, list)):
                    idx = idx[0]
                z_q = layer_m.decode(idx)
                rows[gL] = idx[0].long()
                residual = residual - z_q

        run_branch(q.semantic_residual_vector_quantizer, sem_layers)
        run_branch(q.acoustic_residual_vector_quantizer, ac_layers)
        T = int(next(iter(rows.values())).shape[-1])
        device = next(iter(rows.values())).device
        codes = torch.zeros((n_q, T), dtype=torch.long, device=device)
        for gL, row in rows.items():
            t = min(T, int(row.shape[-1]))
            codes[gL, :t] = row[:t]
    wm, _ = decode_codes_kt(tok, codes)
    return np.asarray(match_len(wav, wm), dtype=np.float32).reshape(-1)


def _qwen_enters(tok, wav, sr) -> list[np.ndarray]:
    n_q = n_q_of("qwen", tok)
    q, n_sem, sem_layers, ac_layers = _qwen_branches(tok, n_q)
    values = qwen_waveform_values(tok, wav, sr)
    enters: list[np.ndarray | None] = [None] * n_q
    with torch.inference_mode():
        emb = qwen_embeddings(tok, values)

        def run_branch(rvq, global_layers):
            if not global_layers:
                return
            h = emb
            if rvq.input_proj is not None:
                h = rvq.input_proj(h)
            residual = h
            for loc, gL in enumerate(global_layers):
                layer_m = rvq.layers[loc]
                enters[gL] = np.asarray(_np(residual[0]), dtype=np.float64)
                idx = layer_m.encode(residual)
                if isinstance(idx, (tuple, list)):
                    idx = idx[0]
                z_q = layer_m.decode(idx)
                residual = residual - z_q

        run_branch(q.semantic_residual_vector_quantizer, sem_layers)
        run_branch(q.acoustic_residual_vector_quantizer, ac_layers)
    return [e if e is not None else np.zeros((1, 1), dtype=np.float64) for e in enters]


def embed_sign_codec(codec, handle, wav, sr, b, layer: int, k: int, gamma: float) -> np.ndarray:
    codec = str(codec).lower()
    flips = {int(layer): int(k)}
    if codec == "moss":
        return _moss_embed(handle, wav, flips=flips, b=b, gamma=gamma)
    if codec == "dac":
        return _dac_embed(handle, wav, sr, flips=flips, b=b, gamma=gamma)
    return _qwen_embed(handle, wav, sr, flips=flips, b=b, gamma=gamma)


def embed_sign_multi_codec(codec, handle, wav, sr, b, flips: dict, gamma: float) -> np.ndarray:
    codec = str(codec).lower()
    flips = {int(k): int(v) for k, v in flips.items()}
    if codec == "moss":
        return _moss_embed(handle, wav, flips=flips, b=b, gamma=gamma)
    if codec == "dac":
        return _dac_embed(handle, wav, sr, flips=flips, b=b, gamma=gamma)
    return _qwen_embed(handle, wav, sr, flips=flips, b=b, gamma=gamma)


def embed_static_codec(codec, handle, wav, sr, layer: int, v, alpha: float) -> np.ndarray:
    codec = str(codec).lower()
    statics = {int(layer): (np.asarray(v, dtype=np.float64).reshape(-1), float(alpha))}
    if codec == "moss":
        return _moss_embed(handle, wav, statics=statics)
    if codec == "dac":
        return _dac_embed(handle, wav, sr, statics=statics)
    return _qwen_embed(handle, wav, sr, statics=statics)


def enter_all(codec, handle, wav, sr) -> list[np.ndarray]:
    codec = str(codec).lower()
    if codec == "moss":
        return _moss_enters(handle, wav)
    if codec == "dac":
        return _dac_enters(handle, wav, sr)
    return _qwen_enters(handle, wav, sr)


def detect_sign_codec(codec, handle, wav, sr, b, layer: int) -> dict:
    enters = enter_all(codec, handle, wav, sr)
    bb = _b_at(b, int(layer))
    r, _ = _align_dt(np.asarray(enters[int(layer)], dtype=np.float64), int(bb.size))
    jac, agree = jaccard_and_agree(r, bb)
    return {"jaccard": float(jac), "agree": float(agree)}


def detect_sign_layers(codec, handle, wav, sr, b, layers) -> dict:
    enters = enter_all(codec, handle, wav, sr)
    out = {}
    for L in layers:
        bb = _b_at(b, int(L))
        r, _ = _align_dt(np.asarray(enters[int(L)], dtype=np.float64), int(bb.size))
        jac, agree = jaccard_and_agree(r, bb)
        out[int(L)] = {"jaccard": float(jac), "agree": float(agree)}
    return out
