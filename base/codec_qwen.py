"""Qwen3 speech_tokenizer 编解码与层 z_q（本仓库自包含；依赖 pip 包 ``qwen_tts``）。

说明
----
波形读入/重采样统一用 **librosa**。若环境 ``import torchaudio`` 因 CUDA .so 失败，
请安装与当前 torch 版本匹配的 **CPU** 版 torchaudio（NPU 机常见做法），例如::

    python -c "import torch; print(torch.__version__)"
    pip install --force-reinstall torchaudio==<与torch同主版本> \\
        --index-url https://download.pytorch.org/whl/cpu
"""

from __future__ import annotations

from pathlib import Path

import librosa
import numpy as np
import torch

TOKENIZER_PATH = r"E:\pred_model\Qwen3-TTS-12Hz-1.7B-Base\speech_tokenizer"


def load_audio_librosa(path: str | Path, target_sr: int | None = None) -> tuple[np.ndarray, int]:
    """全程 librosa 读音频（不走 torchaudio/sox）。"""
    wav, sr = librosa.load(str(path), sr=None, mono=True)
    wav = np.asarray(wav, dtype=np.float32).reshape(-1)
    sr = int(sr)
    if target_sr is not None and int(target_sr) != sr:
        wav = librosa.resample(wav, orig_sr=sr, target_sr=int(target_sr)).astype(np.float32)
        sr = int(target_sr)
    return wav, sr


def resample_librosa(wav: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    wav = np.asarray(wav, dtype=np.float32).reshape(-1)
    if int(orig_sr) == int(target_sr):
        return wav
    return librosa.resample(wav, orig_sr=int(orig_sr), target_sr=int(target_sr)).astype(np.float32)


def load_qwen_tokenizer(path: str, device: str, *, attn_implementation: str = "eager"):
    """加载 Qwen3 speech_tokenizer。

    默认 ``attn_implementation="eager"``，关闭 FlashAttention。
    """
    from qwen_tts.inference.qwen3_tts_tokenizer import Qwen3TTSTokenizer

    tok = Qwen3TTSTokenizer.from_pretrained(
        path,
        device_map=device,
        dtype=torch.bfloat16,
        attn_implementation=str(attn_implementation),
    )
    tok.model.eval()
    return tok


def waveform_values(tokenizer, wav: np.ndarray, sr: int) -> torch.Tensor:
    """用 librosa 重采样到 tokenizer 输入率，再走 feature_extractor（避开 tokenizer.load_audio）。"""
    target_sr = int(tokenizer.feature_extractor.sampling_rate)
    wav_rs = resample_librosa(np.asarray(wav, dtype=np.float32).reshape(-1), int(sr), target_sr)
    inputs = tokenizer.feature_extractor(
        raw_audio=[wav_rs],
        sampling_rate=target_sr,
        return_tensors="pt",
    )
    values = inputs["input_values"].to(tokenizer.device).to(tokenizer.model.dtype)
    if values.ndim == 2:
        values = values.unsqueeze(1)
    return values


def embeddings(tokenizer, values: torch.Tensor) -> torch.Tensor:
    enc = tokenizer.model.encoder
    hidden = enc.encoder(values)
    out = enc.encoder_transformer(hidden.transpose(1, 2), return_dict=True)
    states = out.last_hidden_state if hasattr(out, "last_hidden_state") else out[0]
    states = states.transpose(1, 2)
    return enc.downsample(states)


def encode_codes_kt(tokenizer, wav: np.ndarray, sr: int) -> torch.Tensor:
    n_q = int(tokenizer.model.encoder_valid_num_quantizers)
    values = waveform_values(tokenizer, wav, sr)
    with torch.inference_mode():
        emb = embeddings(tokenizer, values)
        codes_kbt = tokenizer.model.encoder.quantizer.encode(emb, n_q)
    return codes_kbt[:, 0, :].contiguous()


def decode_codes_kt(tokenizer, codes_kt: torch.Tensor) -> tuple[np.ndarray, int]:
    code_tk = codes_kt.transpose(0, 1).contiguous()
    with torch.inference_mode():
        wavs, out_sr = tokenizer.decode({"audio_codes": [code_tk]})
    out = np.asarray(wavs[0], dtype=np.float32).reshape(-1)
    peak = float(np.max(np.abs(out))) if out.size else 0.0
    if peak > 1.0:
        out = (out / peak).astype(np.float32)
    return out, int(out_sr)


def _rvq_local(tokenizer, layer: int):
    q = tokenizer.model.encoder.quantizer
    n_sem = int(q.num_semantic_quantizers)
    layer = int(layer)
    if layer < n_sem:
        return q.semantic_residual_vector_quantizer, layer
    return q.acoustic_residual_vector_quantizer, layer - n_sem


def layer_zq_from_ids(tokenizer, ids_t: torch.Tensor, layer: int) -> np.ndarray:
    rvq, loc = _rvq_local(tokenizer, layer)
    ids = ids_t.long().view(1, -1)
    with torch.inference_mode():
        z = rvq.layers[loc].decode(ids)
        if rvq.output_proj is not None:
            z = rvq.output_proj(z)
    return z[0].detach().float().cpu().numpy()
