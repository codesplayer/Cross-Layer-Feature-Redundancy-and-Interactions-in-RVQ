"""攻击 / 信道降级算子。"""

from __future__ import annotations

import numpy as np


def add_noise(wav: np.ndarray, snr_db: float, seed: int = 0) -> np.ndarray:
    """加性高斯白噪到指定 SNR（dB）。"""
    rng = np.random.default_rng(seed)
    rms = float(np.sqrt(np.mean(wav.astype(np.float64) ** 2)) + 1e-12)
    sigma = rms * 10.0 ** (-float(snr_db) / 20.0)
    return (wav + rng.normal(0.0, sigma, size=wav.shape)).astype(np.float32)


def resample_roundtrip(wav: np.ndarray) -> np.ndarray:
    """24k → 16k → 24k 重采样往返。"""
    from scipy.signal import resample_poly

    down = resample_poly(wav, 2, 3)  # 24k -> 16k
    up = resample_poly(down, 3, 2)  # 16k -> 24k
    return np.asarray(up[: wav.shape[0]], dtype=np.float32)


def stretch(wav: np.ndarray, rate: float) -> np.ndarray:
    """变速（librosa time_stretch，不改音高）。rate<1 变慢、>1 变快。"""
    import librosa

    rate = max(0.5, min(2.0, float(rate)))
    return np.asarray(
        librosa.effects.time_stretch(np.asarray(wav, dtype=np.float32).reshape(-1), rate=rate),
        dtype=np.float32,
    )


def speed_wav(wav: np.ndarray, sr: int | None = None, factor: float = 0.9) -> np.ndarray:
    """与历史 ``speed_wav`` 同义；``sr`` 保留兼容未使用。"""
    return stretch(wav, float(factor))


def crop_wav(wav: np.ndarray, keep_ratio: float = 0.7, where: str = "mid") -> np.ndarray:
    """按比例裁剪；where=mid/head/tail。"""
    wav = np.asarray(wav, dtype=np.float32).reshape(-1)
    n = int(wav.shape[0])
    keep = max(1, int(round(n * float(keep_ratio))))
    keep = min(keep, n)
    where = str(where).lower()
    if where in {"head", "start"}:
        return wav[:keep].copy()
    if where in {"tail", "end"}:
        return wav[n - keep :].copy()
    start = max(0, (n - keep) // 2)
    return wav[start : start + keep].copy()


def crop_mid(wav: np.ndarray, sr: int | None = None, keep: float = 0.7) -> np.ndarray:
    """中心裁剪（默认 keep=0.7）；``sr`` 保留签名兼容。"""
    return crop_wav(wav, keep_ratio=float(keep), where="mid")


def mp3_roundtrip(wav: np.ndarray, sr: int, bitrate: str = "128k", tmpdir=None) -> np.ndarray:
    """MP3 编解码往返（经 ffmpeg，中间 wav 用 FLOAT）。"""
    import subprocess
    import tempfile
    from pathlib import Path

    import soundfile as sf

    from .io import match_len

    tmp = Path(tmpdir) if tmpdir else Path(tempfile.mkdtemp(prefix="mp3rt_"))
    tmp.mkdir(parents=True, exist_ok=True)
    src = tmp / "in.wav"
    mp3 = tmp / "mid.mp3"
    dst = tmp / "out.wav"
    sf.write(str(src), np.asarray(wav, dtype=np.float32).reshape(-1), int(sr), subtype="FLOAT")
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(src), "-b:a", bitrate, str(mp3)], check=True)
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(mp3), "-ar", str(int(sr)), str(dst)], check=True)
    out, _ = sf.read(str(dst), dtype="float32")
    return np.asarray(match_len(wav, np.asarray(out, dtype=np.float32).reshape(-1)), dtype=np.float32)
