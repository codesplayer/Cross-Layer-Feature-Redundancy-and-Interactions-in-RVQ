"""波形相似度：余弦、相关、MSE、简单指纹。"""

from __future__ import annotations

import numpy as np

FINGERPRINT_KEYS = [
    "energy",
    "rms",
    "zcr",
    "spectral_centroid",
    "spectral_bandwidth",
    "spectral_rolloff",
] + [f"mfcc_{i}" for i in range(13)]


def fingerprint_vector(fp: dict, keys: list[str] | None = None) -> np.ndarray:
    keys = keys or FINGERPRINT_KEYS
    return np.array([float((fp or {}).get(k, 0.0) or 0.0) for k in keys], dtype=np.float64)


def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na < 1e-12 or nb < 1e-12:
        return 0.0
    return float(np.clip(np.dot(a, b) / (na * nb), -1.0, 1.0))


def zcosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float64).reshape(-1)
    b = np.asarray(b, dtype=np.float64).reshape(-1)
    a = a - a.mean()
    b = b - b.mean()
    return cosine_sim(a, b)


def match_template(fp: dict, template: dict) -> float:
    sim = zcosine_sim(fingerprint_vector(fp), fingerprint_vector(template))
    return float((sim + 1.0) / 2.0)


def wav_fingerprint(wav: np.ndarray, sr: int) -> dict:
    wav = np.asarray(wav, dtype=np.float64).reshape(-1)
    if wav.size < 8:
        return {k: 0.0 for k in FINGERPRINT_KEYS}
    energy = float(np.mean(wav ** 2))
    rms = float(np.sqrt(max(energy, 0.0)))
    zcr = float(np.mean(np.abs(np.diff(np.sign(wav))) / 2.0))
    spec = np.abs(np.fft.rfft(wav))
    freqs = np.fft.rfftfreq(len(wav), 1.0 / max(int(sr), 1))
    ssum = float(spec.sum())
    if ssum > 0:
        centroid = float(np.sum(freqs * spec) / ssum)
        bandwidth = float(np.sqrt(np.sum(((freqs - centroid) ** 2) * spec) / ssum))
        cum = np.cumsum(spec)
        rolloff = float(freqs[int(np.searchsorted(cum, 0.85 * cum[-1]))]) if cum[-1] > 0 else 0.0
    else:
        centroid, bandwidth, rolloff = 0.0, 0.0, 0.0
    fp = {
        "energy": energy,
        "rms": rms,
        "zcr": zcr,
        "spectral_centroid": centroid,
        "spectral_bandwidth": bandwidth,
        "spectral_rolloff": rolloff,
    }
    try:
        import librosa

        mfcc = librosa.feature.mfcc(y=wav.astype(np.float32), sr=int(sr), n_mfcc=13)
        means = mfcc.mean(axis=1)
        for i, v in enumerate(means):
            fp[f"mfcc_{i}"] = float(v)
    except Exception:
        for i in range(13):
            fp[f"mfcc_{i}"] = 0.0
    return fp


def residual_stats(ref_wav: np.ndarray, mod_wav: np.ndarray, sr: int) -> dict:
    n = min(len(ref_wav), len(mod_wav))
    if n < 2:
        return {"residual_energy": 0.0, "mse": 0.0, "corr": 0.0}
    ref, mod = np.asarray(ref_wav[:n], dtype=np.float64), np.asarray(mod_wav[:n], dtype=np.float64)
    residual = mod - ref
    energy = float(np.mean(residual ** 2))
    mse = float(np.mean((ref - mod) ** 2))
    corr = 0.0
    if np.std(ref) > 1e-12 and np.std(mod) > 1e-12:
        corr = float(np.corrcoef(ref, mod)[0, 1])
    return {
        "residual_energy": energy,
        "residual_rms": float(np.sqrt(max(energy, 0.0))),
        "mse": mse,
        "corr": corr,
        "residual_fp": wav_fingerprint(residual, sr),
    }


def _pad_or_trim(x: np.ndarray, n: int) -> np.ndarray:
    x = np.asarray(x).reshape(-1)
    n = int(n)
    if len(x) == n:
        return x
    if len(x) > n:
        return x[:n]
    out = np.zeros(n, dtype=x.dtype)
    out[: len(x)] = x
    return out
