"""对数谱距离 LSD（自 codec_an/eval 迁入，无外部项目依赖）。"""

from __future__ import annotations

import warnings

import librosa
import numpy as np

STFT_PRESETS = {
    16000: {"n_fft": 512, "hop_length": 256, "win_length": 512},
    22050: {"n_fft": 706, "hop_length": 353, "win_length": 706},
    24000: {"n_fft": 768, "hop_length": 384, "win_length": 768},
    44100: {"n_fft": 1411, "hop_length": 705, "win_length": 1411},
}


def _stft_params(sr, n_fft=None, hop_length=None, win_length=None):
    if n_fft is None and hop_length is None and win_length is None:
        preset = STFT_PRESETS.get(sr)
        if preset is None:
            win_length = int(round(0.032 * sr))
            hop_length = win_length // 2
            n_fft = win_length
        else:
            return preset.copy()
    win_length = win_length or n_fft or STFT_PRESETS.get(sr, {}).get("win_length", 512)
    n_fft = n_fft or win_length
    hop_length = hop_length or (win_length // 2)
    return {"n_fft": n_fft, "hop_length": hop_length, "win_length": win_length}


class CalcuLSD:
    def __init__(
        self,
        sr=16000,
        n_fft=None,
        hop_length=None,
        win_length=None,
        eps=1e-8,
        max_length_diff=None,
    ):
        self.sr = sr
        self.stft_cfg = _stft_params(sr, n_fft, hop_length, win_length)
        self.eps = eps
        self.max_length_diff = max_length_diff or self.stft_cfg["hop_length"]

    def _load_mono(self, signal):
        if isinstance(signal, str):
            y, _ = librosa.load(signal, sr=self.sr, mono=True)
            return y.astype(np.float64)
        return np.asarray(signal, dtype=np.float64).squeeze()

    @staticmethod
    def _length_diff_ms(len_ref, len_deg, sr):
        return abs(len_ref - len_deg) / sr * 1000.0

    def _warn_length_diff(self, len_ref, len_deg):
        diff_samples = abs(len_ref - len_deg)
        if diff_samples <= self.max_length_diff:
            return
        diff_ms = diff_samples / self.sr * 1000.0
        warnings.warn(
            f"参考与重建音频长度差 {diff_samples} 样本 ({diff_ms:.2f} ms)，"
            f"超过阈值 {self.max_length_diff} 样本 "
            f"({self.max_length_diff / self.sr * 1000:.2f} ms)。"
            f"请检查音频是否对齐。",
            stacklevel=3,
        )

    def compute(self, ref, deg):
        ref = self._load_mono(ref)
        deg = self._load_mono(deg)
        length_diff_ms = self._length_diff_ms(len(ref), len(deg), self.sr)
        self._warn_length_diff(len(ref), len(deg))
        ref_spec = np.abs(librosa.stft(ref, **self.stft_cfg)) ** 2
        deg_spec = np.abs(librosa.stft(deg, **self.stft_cfg)) ** 2
        min_frames = min(ref_spec.shape[1], deg_spec.shape[1])
        ref_spec = ref_spec[:, :min_frames]
        deg_spec = deg_spec[:, :min_frames]
        log_ratio = 10.0 * np.log10((ref_spec + self.eps) / (deg_spec + self.eps))
        frame_lsd = np.sqrt(np.mean(log_ratio**2, axis=0))
        return float(np.mean(frame_lsd)), length_diff_ms

    def lsd_file(self, ref_path, deg_path):
        return self.compute(ref_path, deg_path)
