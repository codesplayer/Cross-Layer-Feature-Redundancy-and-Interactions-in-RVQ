"""检测统计量。配对嵌入见 ``embed.py`` / ``works.py``。

三条工作禁止串用：

- **P1** ``detect_p1_cosine``：量化后 band_zq 与密钥模板的特征 cosine。不需要原语音。
- **P2** ``detect_p2``：本层 tap0；time_pn 用 Pearson(proj, s)，static 用帧均余弦。不需要原语音。
- **P3 抗克隆** ``detect_sign``：残差符号与密钥再生的图案 b 做 Jaccard。b 不是干净语音。
- **P3 仪器** ``tap_stat``：多抽头 Pearson，用于流动地图，不是 P2 检测器。
"""

from __future__ import annotations

import numpy as np

from .residual_feat import band_zq_of, feat_cosine  # noqa: F401

from .embed_shared import P2_KINDS
from .trace import trace_enter


def detect_p1_cosine(feat, template) -> dict:
    """P1：band_zq 特征与密钥模板的 cosine / zcosine。模板由密钥生成，不需要干净语音。"""
    return feat_cosine(feat, template)


def jaccard_and_agree(r: np.ndarray, b: np.ndarray) -> tuple[float, float]:
    """r [D,T]，b [D]∈{±1}。返回逐帧 Jaccard 均值与一致率均值。"""
    pos = r > 0  # [D,T]
    bp = (b > 0)[:, None]
    inter = (pos & bp).sum(axis=0)
    union = (pos | bp).sum(axis=0)
    jac = inter / np.maximum(union, 1)
    agree = (pos == bp).mean(axis=0)
    return float(np.mean(jac)), float(np.mean(agree))


def pearson(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    x = x - x.mean()
    y = y - y.mean()
    d = float(np.sqrt((x * x).sum() * (y * y).sum()))
    return float((x * y).sum() / d) if d > 1e-12 else 0.0


def proj_series(enter: np.ndarray, direction: np.ndarray) -> np.ndarray:
    vn = np.asarray(direction, dtype=np.float64)
    vn = vn / (np.linalg.norm(vn) + 1e-12)
    return (enter.astype(np.float64) * vn[:, None]).sum(axis=0)


def frame_cos(enter: np.ndarray, direction: np.ndarray) -> np.ndarray:
    vn = np.asarray(direction, dtype=np.float64)
    vn = vn / (np.linalg.norm(vn) + 1e-12)
    e = enter.astype(np.float64)
    en = np.linalg.norm(e, axis=0) + 1e-12
    return (e * vn[:, None]).sum(axis=0) / en


def carrier_stat(kind: str, enter: np.ndarray, v: np.ndarray, w: np.ndarray, s: np.ndarray) -> float:
    """五种载体对照用的统计量。P2 水印请用 ``detect_p2``（只接受 time_pn/static）。"""
    t_star = enter.shape[1] // 2
    if kind == "time_pn":
        return pearson(proj_series(enter, v), s[: enter.shape[1]])
    if kind == "spacetime":
        return pearson(proj_series(enter, w), s[: enter.shape[1]])
    if kind == "space_pn":
        return float(np.mean(frame_cos(enter, w)))
    if kind == "static":
        return float(np.mean(frame_cos(enter, v)))
    if kind == "single":
        return float(frame_cos(enter[:, t_star : t_star + 1], w)[0])
    raise ValueError(kind)


def tap_stat(enter: np.ndarray, v: np.ndarray, s: np.ndarray) -> float:
    """P3 仪器：enter [D,T]；proj_t = <r_t, v>/|v| 与 s 的 Pearson。"""
    vn = np.asarray(v, dtype=np.float64)
    vn = vn / (np.linalg.norm(vn) + 1e-12)
    proj = (enter.astype(np.float64) * vn[:, None]).sum(axis=0)
    t = min(proj.shape[0], s.shape[0])
    return pearson(proj[:t], s[:t])


def detect_p2(model, wav: np.ndarray, v: np.ndarray, layer: int, s: np.ndarray, kind: str) -> dict:
    """P2 抗变速检测：只看本层 tap0。time_pn → Pearson；static → 帧均余弦。

    不需要干净语音。密钥给出 v（码字方向）和/或 s（PN 序列）。
    """
    if kind not in P2_KINDS:
        raise ValueError(f"P2 detect kind must be time_pn|static, got {kind!r}")
    enters, _, _ = trace_enter(model, wav)
    enter = enters[int(layer)]
    if kind == "time_pn":
        score = pearson(proj_series(enter, v), s[: enter.shape[1]])
    else:
        score = float(np.mean(frame_cos(enter, v)))
    return {"score": float(score), "kind": kind, "layer": int(layer)}


def detect_sign(model, wav: np.ndarray, b: np.ndarray, layer: int = 1) -> dict:
    """P3 抗克隆：残差符号对密钥图案 b 的 Jaccard。

    b 必须用 ``keyed_bits(D, key)`` 从密钥再生，禁止用原语音/干净水印当参考。
    """
    enters, _, _ = trace_enter(model, wav)
    jac, agree = jaccard_and_agree(enters[layer].astype(np.float64), b)
    return {"jaccard": jac, "agree": agree}


class LayerNull:
    """P3 抗克隆：逐层 Jaccard 零分布 + Stouffer 合并。b 同样由密钥再生。"""

    def __init__(self, model, null_wavs, layers, b: np.ndarray):
        self.layers = [int(l) for l in layers]
        self.b = np.asarray(b, dtype=np.float64)
        null_jac = {L: [] for L in self.layers}
        for w in null_wavs:
            enters, _, _ = trace_enter(model, w)
            for L in self.layers:
                jac, _ = jaccard_and_agree(enters[L].astype(np.float64), self.b)
                null_jac[L].append(jac)
        self.null_jac = null_jac
        self.n = len(null_wavs)
        self.mu = {L: float(np.mean(null_jac[L])) for L in self.layers}
        self.sd = {L: float(np.std(null_jac[L]) + 1e-12) for L in self.layers}

    def z_layer(self, L: int, jac: float) -> float:
        return (jac - self.mu[int(L)]) / self.sd[int(L)]

    def _combined_null(self, layers) -> tuple[float, float]:
        tvals = []
        for i in range(self.n):
            tvals.append(sum((self.null_jac[L][i] - self.mu[L]) / self.sd[L] for L in layers))
        return float(np.mean(tvals)), float(np.std(tvals) + 1e-12)

    def detect(self, model, wav, layers) -> dict:
        layers = [int(l) for l in layers]
        enters, _, _ = trace_enter(model, wav)
        zl, jl = {}, {}
        for L in layers:
            jac, _ = jaccard_and_agree(enters[L].astype(np.float64), self.b)
            zl[L] = self.z_layer(L, jac)
            jl[L] = jac
        t = sum(zl.values())
        mu, sd = self._combined_null(layers)
        return {
            "z_per_layer": {str(k): float(v) for k, v in zl.items()},
            "jac_per_layer": {str(k): float(v) for k, v in jl.items()},
            "z_comb": float((t - mu) / sd),
        }
