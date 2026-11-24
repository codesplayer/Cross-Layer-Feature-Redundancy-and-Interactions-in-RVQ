"""宽松攻击套件（§4.1.4 / ATTACK_PROTOCOL 正式可比口径）。

与 Anchor 严苛档隔离。默认参数：crop keep=0.7，AWGN SNR=20 dB，speed=0.9。
"""

from __future__ import annotations

import shutil
import subprocess
import tempfile
from functools import partial
from pathlib import Path
from typing import Callable

import numpy as np

from .attack import add_noise, crop_mid, mp3_roundtrip, speed_wav
from .moss_io import decode_moss, encode_moss

AttackFn = Callable[[np.ndarray, int], np.ndarray]

# 与 ATTACK_PROTOCOL §B 一致（不含 identity）
DEFAULT_MILD_KEYS = (
    "resample_16k",
    "aac_128k",
    "reencode",
    "resample_48k",
    "resample_44100",
    "crop_mid",
    "noise",
    "bandpass",
    "boost",
    "duck",
    "echo",
    "highpass",
    "lowpass",
    "pink_noise",
    "white_noise",
    "smooth",
    "speed",
)


def resample_roundtrip(wav: np.ndarray, sr: int, mid_sr: int) -> np.ndarray:
    import librosa

    wav = np.asarray(wav, dtype=np.float32).reshape(-1)
    mid = librosa.resample(wav, orig_sr=int(sr), target_sr=int(mid_sr))
    back = librosa.resample(mid, orig_sr=int(mid_sr), target_sr=int(sr))
    n = wav.shape[0]
    if back.shape[0] >= n:
        return back[:n].astype(np.float32)
    return np.pad(back, (0, n - back.shape[0])).astype(np.float32)


def _sos_filter(wav: np.ndarray, sr: int, btype: str, *cuts: float) -> np.ndarray:
    from scipy.signal import butter, sosfiltfilt

    wav = np.asarray(wav, dtype=np.float32).reshape(-1)
    nyq = 0.5 * float(sr)
    wn = [max(1e-6, min(0.999, float(c) / nyq)) for c in cuts]
    if btype == "bandpass":
        sos = butter(5, wn, btype="bandpass", output="sos")
    else:
        sos = butter(5, wn[0], btype=btype, output="sos")
    return sosfiltfilt(sos, wav).astype(np.float32)


def bandpass(wav: np.ndarray, sr: int, lo: float = 300.0, hi: float = 8000.0) -> np.ndarray:
    return _sos_filter(wav, sr, "bandpass", lo, hi)


def highpass(wav: np.ndarray, sr: int, cutoff_hz: float = 500.0) -> np.ndarray:
    return _sos_filter(wav, sr, "highpass", float(cutoff_hz))


def lowpass(wav: np.ndarray, sr: int, cutoff_hz: float = 5000.0) -> np.ndarray:
    return _sos_filter(wav, sr, "lowpass", float(cutoff_hz))


def boost(wav: np.ndarray, factor: float = 1.2) -> np.ndarray:
    return np.clip(np.asarray(wav, dtype=np.float32).reshape(-1) * float(factor), -1.0, 1.0).astype(np.float32)


def duck(wav: np.ndarray, factor: float = 0.8) -> np.ndarray:
    return (np.asarray(wav, dtype=np.float32).reshape(-1) * float(factor)).astype(np.float32)


def echo(wav: np.ndarray, sr: int, delay_s: float = 0.3, volume: float = 0.3) -> np.ndarray:
    wav = np.asarray(wav, dtype=np.float32).reshape(-1)
    n = int(wav.shape[0])
    delay = max(1, int(round(float(delay_s) * int(sr))))
    out = wav.copy()
    if delay < n:
        out[delay:] = out[delay:] + float(volume) * wav[: n - delay]
    peak = float(np.max(np.abs(out))) or 1.0
    ref = float(np.max(np.abs(wav))) or 1.0
    return (out / peak * ref).astype(np.float32)


def pink_noise(wav: np.ndarray, std: float = 0.01, seed: int = 42) -> np.ndarray:
    wav = np.asarray(wav, dtype=np.float32).reshape(-1)
    rng = np.random.default_rng(seed)
    n = int(wav.shape[0])
    num_rows = 16
    arr = rng.standard_normal((num_rows, n // num_rows + 1))
    pink = np.cumsum(arr, axis=1).reshape(-1)[:n]
    pink = pink / (np.max(np.abs(pink)) + 1e-8)
    return (wav + pink.astype(np.float32) * float(std)).astype(np.float32)


def white_noise(wav: np.ndarray, std: float = 0.001, seed: int = 42) -> np.ndarray:
    wav = np.asarray(wav, dtype=np.float32).reshape(-1)
    rng = np.random.default_rng(seed)
    return (wav + rng.normal(0.0, float(std), size=wav.shape).astype(np.float32)).astype(np.float32)


def smooth(wav: np.ndarray, window: int = 5) -> np.ndarray:
    wav = np.asarray(wav, dtype=np.float32).reshape(-1)
    w = max(1, int(window))
    ker = np.ones(w, dtype=np.float64) / float(w)
    return np.convolve(wav, ker, mode="same").astype(np.float32)


def aac_roundtrip(wav: np.ndarray, sr: int, bitrate: str = "128k") -> np.ndarray:
    """ffmpeg 内置 aac 往返。"""
    import soundfile as sf

    from .io import match_len

    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise RuntimeError("need ffmpeg for aac_128k")
    with tempfile.TemporaryDirectory(prefix="aacrt_") as td:
        td = Path(td)
        src, enc, dst = td / "in.wav", td / "mid.m4a", td / "out.wav"
        sf.write(str(src), np.asarray(wav, dtype=np.float32).reshape(-1), int(sr), subtype="FLOAT")
        subprocess.run(
            [ffmpeg, "-y", "-loglevel", "error", "-i", str(src), "-c:a", "aac", "-b:a", bitrate, str(enc)],
            check=True,
        )
        subprocess.run(
            [ffmpeg, "-y", "-loglevel", "error", "-i", str(enc), "-ar", str(int(sr)), str(dst)],
            check=True,
        )
        out, _ = sf.read(str(dst), dtype="float32")
        return np.asarray(match_len(wav, np.asarray(out, dtype=np.float32).reshape(-1)), dtype=np.float32)


def moss_reencode(model, wav: np.ndarray) -> np.ndarray:
    return np.asarray(decode_moss(model, encode_moss(model, wav)), dtype=np.float32).reshape(-1)


def build_mild_attacks(
    keys: list[str] | None = None,
    *,
    moss_model=None,
    crop_keep: float = 0.7,
    snr_db: float = 20.0,
    speed_factor: float = 0.9,
) -> list[tuple[str, AttackFn, dict]]:
    """返回 [(key, fn(wav, sr), meta), ...]。``reencode`` 需要 ``moss_model``。"""
    reg: dict[str, tuple[AttackFn, dict]] = {
        "resample_16k": (
            lambda w, s: resample_roundtrip(w, s, 16000),
            {"source": "mild", "params": "24→16→24"},
        ),
        "resample_48k": (
            lambda w, s: resample_roundtrip(w, s, 48000),
            {"source": "mild", "params": "24→48→24"},
        ),
        "resample_44100": (
            lambda w, s: resample_roundtrip(w, s, 44100),
            {"source": "mild", "params": "24→44.1→24"},
        ),
        "aac_128k": (partial(aac_roundtrip, bitrate="128k"), {"source": "mild", "params": "aac 128k"}),
        "mp3_128k": (partial(mp3_roundtrip, bitrate="128k"), {"source": "mild", "params": "mp3 128k"}),
        "crop_mid": (
            lambda w, s: crop_mid(w, s, keep=float(crop_keep)),
            {"source": "mild", "params": f"keep={crop_keep}"},
        ),
        "noise": (
            lambda w, s: add_noise(w, float(snr_db), seed=42),
            {"source": "mild", "params": f"AWGN SNR={snr_db}dB"},
        ),
        "bandpass": (lambda w, s: bandpass(w, s, 300.0, 8000.0), {"source": "mild", "params": "300-8000 Hz"}),
        "boost": (lambda w, s: boost(w, 1.2), {"source": "mild", "params": "x1.2"}),
        "duck": (lambda w, s: duck(w, 0.8), {"source": "mild", "params": "x0.8"}),
        "echo": (lambda w, s: echo(w, s, 0.3, 0.3), {"source": "mild", "params": "delay=0.3s vol=0.3"}),
        "highpass": (lambda w, s: highpass(w, s, 500.0), {"source": "mild", "params": "500 Hz"}),
        "lowpass": (lambda w, s: lowpass(w, s, 5000.0), {"source": "mild", "params": "5000 Hz"}),
        "pink_noise": (lambda w, s: pink_noise(w, 0.01), {"source": "mild", "params": "std=0.01"}),
        "white_noise": (lambda w, s: white_noise(w, 0.001), {"source": "mild", "params": "std=0.001"}),
        "smooth": (lambda w, s: smooth(w, 5), {"source": "mild", "params": "win=5"}),
        "speed": (
            lambda w, s: speed_wav(w, s, factor=float(speed_factor)),
            {"source": "mild", "params": f"factor={speed_factor}"},
        ),
    }
    if moss_model is not None:
        reg["reencode"] = (
            lambda w, s: moss_reencode(moss_model, w),
            {"source": "mild", "params": "MOSS encode→decode"},
        )

    use = list(keys) if keys is not None else list(DEFAULT_MILD_KEYS)
    out: list[tuple[str, AttackFn, dict]] = []
    for k in use:
        if k == "reencode" and k not in reg:
            raise SystemExit("mild attack 'reencode' needs MOSS model loaded")
        if k not in reg:
            raise SystemExit(f"unknown mild attack {k}; choose from {sorted(set(reg) | {'reencode'})}")
        fn, meta = reg[k]
        out.append((k, fn, meta))
    return out
