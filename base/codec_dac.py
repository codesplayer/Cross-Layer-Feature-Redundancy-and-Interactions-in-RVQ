"""Descript DAC 24kHz 编解码与层 z_q（本仓库自包含；依赖 pip 包 ``dac``）。"""

from __future__ import annotations

import numpy as np
import torch


def load_dac(device: str = "cuda:0", model_path: str | None = None):
    """``model_path`` 空则 ``dac.utils.download(model_type='24khz')``。"""
    import dac

    path = str(model_path).strip() if model_path else ""
    if not path:
        path = dac.utils.download(model_type="24khz")
    model = dac.DAC.load(path)
    model.to(device).eval()
    return model


def wav_to_batch(model, wav: np.ndarray, sr: int) -> torch.Tensor:
    from audiotools import AudioSignal

    wav = np.asarray(wav, dtype=np.float32).reshape(-1)
    sig = AudioSignal(wav, sample_rate=int(sr))
    sig.to(model.device)
    return model.preprocess(sig.audio_data, int(sr))


def encode_codes(model, wav: np.ndarray, sr: int) -> torch.Tensor:
    x = wav_to_batch(model, wav, sr)
    with torch.inference_mode():
        _z, codes, _lat, _cmt, _cbl = model.encode(x)
    return codes[0].contiguous()


def decode_codes(model, codes_nt: torch.Tensor) -> np.ndarray:
    codes = codes_nt.unsqueeze(0)
    with torch.inference_mode():
        z_q, _zp, _ = model.quantizer.from_codes(codes)
        y = model.decode(z_q)
    out = y[0, 0].detach().float().cpu().numpy()
    peak = float(np.max(np.abs(out))) if out.size else 0.0
    if peak > 1.0:
        out = (out / peak).astype(np.float32)
    return np.asarray(out, dtype=np.float32).reshape(-1)


def layer_zq_from_codes(model, codes_nt: torch.Tensor, layer: int) -> np.ndarray:
    q = model.quantizer.quantizers[int(layer)]
    ids = codes_nt[int(layer)].long().view(1, -1)
    with torch.inference_mode():
        z_p = q.decode_code(ids)
        z_q = q.out_proj(z_p)
    return z_q[0].detach().float().cpu().numpy()
