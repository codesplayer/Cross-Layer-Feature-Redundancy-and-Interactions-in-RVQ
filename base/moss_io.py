"""MOSS 加载 / 编解码封装（供嵌入与检测共用）。"""

from __future__ import annotations

import os
import random

import numpy as np

SEED = 42


def seeding_enabled() -> bool:
    v = os.environ.get("WATERMARK_NO_SEED", "").strip().lower()
    return v not in {"1", "true", "yes"}


def set_seed(seed: int = SEED) -> None:
    if not seeding_enabled():
        return
    os.environ["PYTHONHASHSEED"] = str(int(seed))
    random.seed(int(seed))
    np.random.seed(int(seed))
    try:
        import torch

        torch.manual_seed(int(seed))
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(int(seed))
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except Exception:
        pass


def _compat_transformers():
    """transformers 4.57 仍用 PretrainedConfig；MOSS 远程代码写的是 PreTrainedConfig。"""
    import transformers.configuration_utils as cu

    if not hasattr(cu, "PreTrainedConfig") and hasattr(cu, "PretrainedConfig"):
        cu.PreTrainedConfig = cu.PretrainedConfig


def load_moss(model_path: str, device: str = "npu"):
    import torch
    from transformers import AutoModel

    set_seed(SEED)
    _compat_transformers()
    if str(device).startswith("npu"):
        try:
            import torch_npu  # noqa: F401

            if ":" in str(device):
                torch.npu.set_device(int(str(device).split(":", 1)[1]))
        except Exception:
            pass
    kwargs = {"trust_remote_code": True, "local_files_only": True}
    model = AutoModel.from_pretrained(model_path, **kwargs).eval()
    try:
        model = model.to(device)
    except Exception:
        pass
    sr = getattr(model, "sampling_rate", None) or getattr(model.config, "sample_rate", 24000)
    return model, int(sr)


def read_wav(path, target_sr: int) -> np.ndarray:
    """读 wav/mp3/flac/ogg/mp4：优先 librosa（避开 torchaudio/sox）。mp4 需要 ffmpeg。"""
    import librosa

    path = str(path)
    wav, sr = librosa.load(path, sr=None, mono=True)
    wav = np.asarray(wav, dtype=np.float32).reshape(-1)
    if int(sr) != int(target_sr):
        wav = librosa.resample(wav, orig_sr=int(sr), target_sr=int(target_sr)).astype(np.float32)
    return wav


def write_wav(path, wav: np.ndarray, sr: int) -> None:
    import soundfile as sf

    sf.write(str(path), np.asarray(wav, dtype=np.float32).reshape(-1), int(sr))


def encode_moss(model, wav_np: np.ndarray) -> np.ndarray:
    """wav -> tokens [num_quantizers, T] (int64)."""
    import torch

    set_seed(SEED)
    device = next(model.parameters()).device
    wav = torch.from_numpy(np.ascontiguousarray(wav_np)).float().view(1, 1, -1).to(device)
    with torch.no_grad():
        out = model.encode(wav)
    codes = out.audio_codes.detach().cpu()
    if codes.ndim == 3:
        if codes.shape[1] == 1:
            codes = codes.squeeze(1)
        elif codes.shape[0] == 1:
            codes = codes.squeeze(0)
        else:
            codes = codes[:, 0, :]
    return codes.long().numpy()


def decode_moss(model, tokens: np.ndarray) -> np.ndarray:
    """tokens [num_quantizers, T] -> wav (float32 1-D)."""
    import torch

    set_seed(SEED)
    device = next(model.parameters()).device
    codes = torch.from_numpy(np.asarray(tokens, dtype=np.int64)).unsqueeze(1).to(device)
    with torch.no_grad():
        out = model.decode(codes)
    return np.asarray(out.audio.detach().float().cpu().numpy()).reshape(-1)
