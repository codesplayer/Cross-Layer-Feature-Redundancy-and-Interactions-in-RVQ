"""SNAC 24kHz 编解码封装（本地权重 ``E:/pred_model/snac_24khz``）。

依赖 pip 包 ``snac``。多分辨率 RVQ：``vq_strides`` 默认 [4,2,1]，
各层 code 时序长度不同；``z_q`` 经 ``repeat_interleave`` 对齐到 encoder 帧率。
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from base.config import first_existing

SNAC_PATH_DEFAULT = first_existing(
    Path(r"E:\pred_model\snac_24khz"),
    Path("/data1/home/ynian/pred_models/snac_24khz"),
    Path(r"D:\pred_model\snac_24khz"),
    "",
)


def load_snac(device: str = "cuda:0", model_path: str | None = None):
    """加载 SNAC；``model_path`` 空则用本地默认目录 / hub 名。"""
    from snac import SNAC

    path = str(model_path).strip() if model_path else ""
    if not path:
        path = str(SNAC_PATH_DEFAULT or "hubertsiuzdak/snac_24khz")
    model = SNAC.from_pretrained(path)
    model.to(device).eval()
    return model


def wav_to_batch(model, wav: np.ndarray, sr: int) -> torch.Tensor:
    """``[1,1,T]`` float，必要时重采样到 model.sampling_rate。"""
    import librosa

    wav = np.asarray(wav, dtype=np.float32).reshape(-1)
    target = int(model.sampling_rate)
    if int(sr) != target:
        wav = librosa.resample(wav, orig_sr=int(sr), target_sr=target).astype(np.float32)
    device = next(model.parameters()).device
    return torch.from_numpy(wav).view(1, 1, -1).to(device)


@torch.no_grad()
def encode_codes(model, wav: np.ndarray, sr: int) -> list[np.ndarray]:
    """返回各层 codes（长度可能不同）：list of [T_l] int64。"""
    x = wav_to_batch(model, wav, sr)
    length = int(x.shape[-1])
    x = model.preprocess(x)
    z = model.encoder(x)
    _zq, codes = model.quantizer(z)
    out = []
    for c in codes:
        out.append(c[0].detach().long().cpu().numpy().astype(np.int64))
    return out


@torch.no_grad()
def decode_codes(model, codes_list: list[np.ndarray] | list[torch.Tensor], wav_len: int | None = None) -> np.ndarray:
    """``codes_list``: 每层 [T_l] → 波形 float32 1-D。"""
    device = next(model.parameters()).device
    codes_t = []
    for c in codes_list:
        t = torch.as_tensor(c, device=device, dtype=torch.long)
        if t.dim() == 1:
            t = t.unsqueeze(0)
        codes_t.append(t)
    z_q = model.quantizer.from_codes(codes_t)
    y = model.decoder(z_q)
    out = y[0, 0].detach().float().cpu().numpy().astype(np.float32).reshape(-1)
    if wav_len is not None and wav_len > 0:
        out = out[: int(wav_len)]
    return out


@torch.no_grad()
def layer_zq_from_ids(model, layer: int) -> np.ndarray:
    """该层全部码字的 ``out_proj`` 向量 ``[K, D]``（未做 temporal repeat）。"""
    q = model.quantizer.quantizers[int(layer)]
    weight = q.codebook.weight.detach().float()
    k = int(weight.shape[0])
    device = weight.device
    ids = torch.arange(k, device=device, dtype=torch.long).view(1, -1)
    z_p = q.decode_code(ids)
    z_q = q.out_proj(z_p).float()
    if z_q.dim() == 3:
        arr = z_q[0].transpose(0, 1).detach().cpu().numpy()
    else:
        arr = z_q.view(k, -1).detach().cpu().numpy()
    return np.asarray(arr, dtype=np.float64)
