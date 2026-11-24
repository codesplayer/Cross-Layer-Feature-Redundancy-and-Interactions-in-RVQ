"""语音克隆封装（P3 独占管线）。

当前实现：VoxCPM2。多 cloner 泛化（Qwen3-TTS / OmniVoice / MiMo-Audio）为后续待跑项。

官方 voxcpm 的 auto 只走 cuda→mps→cpu，Ascend 上会落到 CPU+bfloat16，
再叠 Xbyak / 混用 NPU 内核，报 illegal immediate 或 Broken promise。
加载前把 resolve_runtime_device 补上当前 npu:N，from_pretrained 仍不传 device。
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np

from .config import TEXTS, VOXCPM2_PATH
from .io import write_wav_float

_VOX_PATCHED = False


def _pin_device(device: str | None) -> None:
    if not device or not str(device).startswith("npu"):
        return
    import torch
    import torch_npu  # noqa: F401

    if ":" in str(device):
        torch.npu.set_device(int(str(device).split(":", 1)[1]))


def _npu_sync() -> None:
    import torch

    if hasattr(torch, "npu"):
        try:
            torch.npu.synchronize()
        except Exception:
            pass
    if torch.cuda.is_available():
        try:
            torch.cuda.synchronize()
        except Exception:
            pass


def _current_npu(device: str | None = None) -> str | None:
    if device and str(device).startswith("npu"):
        d = str(device)
        return d if ":" in d else "npu:0"
    import torch

    try:
        import torch_npu  # noqa: F401
    except Exception:
        pass
    if not hasattr(torch, "npu"):
        return None
    try:
        if not torch.npu.is_available():
            return None
        return f"npu:{int(torch.npu.current_device())}"
    except Exception:
        return "npu:0"


def _install_voxcpm_npu_patch() -> None:
    """让官方 voxcpm 的 auto 选中当前 NPU，而不是落到 CPU。"""
    global _VOX_PATCHED
    if _VOX_PATCHED:
        return
    os.environ.setdefault("TORCH_COMPILE_DISABLE", "1")
    os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")
    import voxcpm.model.utils as utils

    orig = utils.resolve_runtime_device

    def resolve_runtime_device(device=None, configured_device: str = "cuda"):
        explicit = None if device is None else str(device).strip().lower()
        npu = _current_npu()
        if explicit is None or explicit in {"", "auto"}:
            if npu:
                return npu
            return orig(device, configured_device)
        if explicit.startswith("npu"):
            return explicit if ":" in explicit else (npu or "npu:0")
        return orig(device, configured_device)

    utils.resolve_runtime_device = resolve_runtime_device
    for mod_name in ("voxcpm.model.voxcpm2", "voxcpm.model.voxcpm"):
        try:
            mod = __import__(mod_name, fromlist=["resolve_runtime_device"])
        except Exception:
            continue
        if hasattr(mod, "resolve_runtime_device"):
            mod.resolve_runtime_device = resolve_runtime_device
    _VOX_PATCHED = True


def prepare_clone_prompt(src_path, dst_path, target_sr: int, *, min_sec: float = 1.0, max_sec: float = 20.0) -> str:
    """把提示音写成 VoxCPM2 能吃的 48 kHz PCM_16，避免 NPU 上 illegal immediate。"""
    import librosa
    import soundfile as sf

    target_sr = int(target_sr) or 48000
    y, sr = librosa.load(str(src_path), sr=None, mono=True)
    y = np.nan_to_num(np.asarray(y, dtype=np.float32).reshape(-1), nan=0.0, posinf=0.0, neginf=0.0)
    if int(sr) != target_sr:
        y = librosa.resample(y, orig_sr=int(sr), target_sr=target_sr).astype(np.float32)
    peak = float(np.max(np.abs(y))) if y.size else 0.0
    if peak > 1.0:
        y = (y / peak).astype(np.float32)
    n_min = int(min_sec * target_sr)
    n_max = int(max_sec * target_sr)
    if y.size < n_min:
        y = np.pad(y, (0, n_min - int(y.size)))
    elif y.size > n_max:
        y = y[:n_max]
    Path(dst_path).parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(dst_path), y.astype(np.float32), target_sr, subtype="PCM_16")
    return str(dst_path)


class VoxCPM2Cloner:
    """懒加载 VoxCPM2，克隆多条后 ``free()`` 释放显存。

    不要和 codec / WavLM 同时占同一张 NPU：combo 里在 embed worker 退出后再开 clone pass。
    """

    def __init__(self, path: str = VOXCPM2_PATH, device: str | None = None):
        self.path = path
        self.device = device
        self.tts = None
        self.sample_rate = 48000
        self._load()

    def _load(self) -> None:
        _pin_device(self.device)
        _npu_sync()
        _install_voxcpm_npu_patch()
        from voxcpm import VoxCPM

        self.tts = VoxCPM.from_pretrained(
            self.path, load_denoiser=False, local_files_only=True, optimize=False
        )
        npu = _current_npu(self.device)
        model = getattr(self.tts, "tts_model", None)
        got = str(getattr(model, "device", "") or "")
        if npu and model is not None and not got.startswith("npu"):
            try:
                model.to(npu)
                model.device = npu
                if hasattr(model, "config"):
                    model.config.device = npu
                got = npu
            except Exception:
                pass
        print(f"voxcpm device={got} want={npu}", flush=True)
        self.sample_rate = int(getattr(model, "sample_rate", None) or 48000)

    def clone_text(
        self,
        text: str,
        ref_wav_path,
        out_path,
        cfg_value: float = 2.0,
        inference_timesteps: int = 10,
    ):
        """同文本克隆：prompt/reference 都是 ref_wav_path，文本由调用方提供。"""
        text = str(text).strip()
        if not text:
            raise ValueError("empty clone text")
        _pin_device(self.device)
        src = Path(ref_wav_path)
        prompt = src.with_name(src.stem + "_48k16.wav")
        prepare_clone_prompt(src, prompt, self.sample_rate)
        try:
            _npu_sync()
            gen_kw = dict(
                text=text,
                prompt_wav_path=str(prompt),
                prompt_text=text,
                reference_wav_path=str(prompt),
                cfg_value=float(cfg_value),
                inference_timesteps=int(inference_timesteps),
            )
            try:
                out = self.tts.generate(**gen_kw, retry_badcase=False)
            except TypeError:
                out = self.tts.generate(**gen_kw)
        finally:
            try:
                prompt.unlink(missing_ok=True)
            except TypeError:
                if prompt.exists():
                    prompt.unlink()
            except Exception:
                pass
        write_wav_float(out_path, np.asarray(out, dtype=np.float32).reshape(-1), self.sample_rate)
        return out_path

    def clone(self, name: str, ref_wav_path, out_path, cfg_value: float = 2.0, inference_timesteps: int = 10):
        """用 ref_wav_path 作 prompt+reference，按 TEXTS[name] 生成克隆语音并写盘。"""
        return self.clone_text(TEXTS[name], ref_wav_path, out_path, cfg_value, inference_timesteps)

    def reload(self) -> None:
        try:
            self.free()
        except Exception:
            pass
        self._load()

    def free(self) -> None:
        import gc

        import torch

        if getattr(self, "tts", None) is not None:
            del self.tts
            self.tts = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        if hasattr(torch, "npu"):
            try:
                torch.npu.empty_cache()
            except Exception:
                pass
        _npu_sync()
