"""Anchor Watermark (AAAI-26) 十类严苛失真（water_marks 自包含）。

原文 Distortions::
  32 kHz RS, 100 Hz HP, 1000 Hz LP, 10% AS, 9 kbps MP3, 8-bit RC,
  median-15 MF, SNR=5 dB GN, crop keep=60% CP, +0.5 semitone PS.
"""

from __future__ import annotations

from functools import partial
from typing import Callable

import numpy as np

from .attack import add_noise, crop_wav, mp3_roundtrip

AttackFn = Callable[[np.ndarray, int], np.ndarray]

DEFAULT_ANCHOR_KEYS = (
    "anchor_rs",
    "anchor_hp",
    "anchor_lp",
    "anchor_as",
    "anchor_mc",
    "anchor_rc",
    "anchor_mf",
    "anchor_gn",
    "anchor_cp",
    "anchor_ps",
)


def resample_to_mid(wav: np.ndarray, sr: int, mid_sr: int = 32000) -> np.ndarray:
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
    wn = [max(1e-6, min(0.999, c / nyq)) for c in cuts]
    if btype == "bandpass":
        sos = butter(5, wn, btype="bandpass", output="sos")
    elif len(wn) == 1:
        sos = butter(5, wn[0], btype=btype, output="sos")
    else:
        raise ValueError(btype)
    return sosfiltfilt(sos, wav).astype(np.float32)


def highpass(wav: np.ndarray, sr: int, cutoff_hz: float = 100.0) -> np.ndarray:
    return _sos_filter(wav, sr, "highpass", float(cutoff_hz))


def lowpass(wav: np.ndarray, sr: int, cutoff_hz: float = 1000.0) -> np.ndarray:
    return _sos_filter(wav, sr, "lowpass", float(cutoff_hz))


def amplitude_scale(wav: np.ndarray, factor: float = 0.9) -> np.ndarray:
    return (np.asarray(wav, dtype=np.float32).reshape(-1) * float(factor)).astype(np.float32)


def bit_depth_requant(wav: np.ndarray, bits: int = 8) -> np.ndarray:
    wav = np.clip(np.asarray(wav, dtype=np.float32).reshape(-1), -1.0, 1.0)
    levels = (1 << max(2, int(bits))) - 1
    q = np.round((wav + 1.0) * 0.5 * levels) / levels
    return (q * 2.0 - 1.0).astype(np.float32)


def median_filter(wav: np.ndarray, window: int = 15) -> np.ndarray:
    from scipy.ndimage import median_filter as _mf

    wav = np.asarray(wav, dtype=np.float32).reshape(-1)
    w = max(1, int(window))
    if w % 2 == 0:
        w += 1
    return _mf(wav, size=w, mode="nearest").astype(np.float32)


def pitch_shift(wav: np.ndarray, sr: int, n_steps: float = 0.5) -> np.ndarray:
    import librosa

    wav = np.asarray(wav, dtype=np.float32).reshape(-1)
    return librosa.effects.pitch_shift(wav, sr=int(sr), n_steps=float(n_steps)).astype(np.float32)


def build_anchor_attacks(keys: list[str] | None = None) -> list[tuple[str, AttackFn, dict]]:
    reg: dict[str, tuple[AttackFn, dict]] = {
        "anchor_rs": (lambda w, s: resample_to_mid(w, s, 32000), {"paper": "RS", "params": "32kHz RT"}),
        "anchor_hp": (lambda w, s: highpass(w, s, 100.0), {"paper": "HP", "params": "100 Hz"}),
        "anchor_lp": (lambda w, s: lowpass(w, s, 1000.0), {"paper": "LP", "params": "1000 Hz"}),
        "anchor_as": (lambda w, s: amplitude_scale(w, 0.9), {"paper": "AS", "params": "×0.9"}),
        "anchor_mc": (partial(mp3_roundtrip, bitrate="9k"), {"paper": "MC", "params": "mp3 9k"}),
        "anchor_rc": (lambda w, s: bit_depth_requant(w, 8), {"paper": "RC", "params": "8-bit"}),
        "anchor_mf": (lambda w, s: median_filter(w, 15), {"paper": "MF", "params": "win=15"}),
        "anchor_gn": (lambda w, s: add_noise(w, 5.0, seed=42), {"paper": "GN", "params": "SNR=5dB"}),
        "anchor_cp": (lambda w, s: crop_wav(w, 0.6, "mid"), {"paper": "CP", "params": "keep=0.6"}),
        "anchor_ps": (lambda w, s: pitch_shift(w, s, 0.5), {"paper": "PS", "params": "+0.5 st"}),
    }
    use = list(keys) if keys is not None else list(DEFAULT_ANCHOR_KEYS)
    out = []
    for k in use:
        if k not in reg:
            raise SystemExit(f"unknown Anchor attack {k}; choose from {sorted(reg)}")
        fn, meta = reg[k]
        out.append((k, fn, meta))
    return out
