"""Wav 域毫秒裁剪（盲搜 crop 用）。"""

from __future__ import annotations

import math

import numpy as np

FRAME_MS = 40.0
PERIOD_MS = 120.0  # 3 × 40ms


def ms_to_samples(ms: float, sr: int) -> int:
    return max(0, int(round(float(ms) * int(sr) / 1000.0)))


def crop_wav_ms_range(wav: np.ndarray, sr: int, start_ms: float, end_ms: float | None) -> np.ndarray:
    """wav 上 [start_ms, end_ms)；end_ms=None 表示到末尾。"""
    wav = np.asarray(wav, dtype=np.float32).reshape(-1)
    n = int(wav.shape[0])
    i0 = ms_to_samples(start_ms, sr)
    i1 = n if end_ms is None else min(n, ms_to_samples(end_ms, sr))
    i0 = max(0, min(i0, n - 1))
    i1 = max(i0 + 1, i1)
    return wav[i0:i1].copy()


def align_ms_length(length_ms: float, period_ms: float = PERIOD_MS) -> float:
    """向下取 period 整倍数毫秒。"""
    p = float(period_ms)
    if p <= 0:
        return float(length_ms)
    n = int(float(length_ms) // p)
    return float(max(p, n * p)) if n >= 1 else float(min(length_ms, p))


def ms_steps(period_ms: float, step_ms: float) -> list[float]:
    step = max(0.1, float(step_ms))
    period = float(period_ms)
    n = int(math.floor(period / step + 1e-9))
    out = [round(i * step, 6) for i in range(max(1, n))]
    if out[0] != 0.0:
        out.insert(0, 0.0)
    return out
