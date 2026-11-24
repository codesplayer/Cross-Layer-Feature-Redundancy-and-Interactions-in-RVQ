"""P1 band 多通道打分（本仓库自包含；不依赖 codec_an）。

主通道 ``slide_v3`` 与 codec_an ``score_wav_channels`` 默认一致：win=18, hop=3。
盲搜辅分 ``channel_max = max(slide_aba_win3_h1, slide_ab_win2_h1)``。
"""

from __future__ import annotations

import numpy as np

from .moss_io import encode_moss
from .plan import resolve_wm_layers
from .residual_feat import (
    aba_embed_template,
    band_zq_layers_of,
    feat_cosine,
    query_feat_of,
    slide_feat_vs_cycle,
)

ILLUSTRATIVE_TH_ABA = 0.15

DEFAULT_SPEED_FACTORS = (
    0.833333,
    0.869565,
    0.909091,
    0.952381,
    1.0,
    1.052632,
    1.08,
    1.111111,
    1.14,
    1.176471,
    1.25,
)

P1_BAND_L9 = {
    "key": "band_L9",
    "high_start": 9,
    "high_end": 9,
    "layers": [9],
    "query_space": "band",
}


def profile_score_cfg(table: dict, profile: dict | None = None) -> dict:
    profile = dict(profile or P1_BAND_L9)
    return {
        "table": table,
        "high_start": int(profile["high_start"]),
        "high_end": int(profile["high_end"]),
        "layers": profile.get("layers"),
        "query_space": str(profile.get("query_space") or "band"),
    }


def score_band_once(
    model,
    wav: np.ndarray,
    table: dict,
    *,
    high_start: int = 9,
    high_end: int = 9,
    layers=None,
    n_peel: int | None = None,
    period: int = 3,
    hop: int = 3,
    slide_win: int = 18,
    query_space: str = "band",
) -> dict:
    """对一条 wav 打 band 检测分（aba / slide_v3 / channel_max）。

    ``slide_v3`` / ``slide18``：与 codec_an ``score_wav_channels`` 默认一致（win=18, hop=3）。
    ``aba3`` / ``ab2`` / ``channel_max``：短窗盲搜通道（win=period/2, hop=1），与 codec_an 副通道一致。
    不 decode、不算 wave（攻击 cascade 默认不用 wave）。
    """
    wav = np.asarray(wav, dtype=np.float32).reshape(-1)
    tokens = encode_moss(model, wav)
    n_q, T = tokens.shape
    last = int(n_q - 1) if high_end is None else int(high_end)
    wm_layers = resolve_wm_layers(n_q, high_start, last, layers)
    n_peel = int(min(wm_layers) if n_peel is None else n_peel)
    space = str(query_space or "band").strip().lower()
    hop_use = int(hop) if hop is not None else int(period)
    win = int(slide_win)

    if space == "band":
        feat_q = band_zq_layers_of(model, wav, wm_layers)
    else:
        feat_q = query_feat_of(model, wav, n_peel, last, space)

    t_feat = int(feat_q.shape[-1])
    token_tmpl = aba_embed_template(
        model, t_feat, n_peel, last, table, layers=wm_layers, period=int(period)
    )
    aba = feat_cosine(feat_q, token_tmpl)
    aba_cycle = aba_embed_template(
        model, int(period), n_peel, last, table, layers=wm_layers, period=int(period)
    )
    # codec_an 主通道 slide_v3
    slide = slide_feat_vs_cycle(
        feat_q, aba_cycle, win=win, hop=hop_use, threshold=ILLUSTRATIVE_TH_ABA
    )
    # codec_an 副通道：短窗 ABA / AB（盲搜 channel_max）
    slide_aba = slide_feat_vs_cycle(
        feat_q, aba_cycle, win=int(period), hop=1, threshold=ILLUSTRATIVE_TH_ABA
    )
    ab_cycle = np.asarray(aba_cycle[:, :2], dtype=np.float32)
    slide_ab = slide_feat_vs_cycle(
        feat_q, ab_cycle, win=2, hop=1, threshold=ILLUSTRATIVE_TH_ABA
    )
    aba3 = float(slide_aba.get("max") or 0.0)
    ab2 = float(slide_ab.get("max") or 0.0)
    slide_max = float(slide.get("max") or 0.0)
    return {
        "T": int(T),
        "aba_embed": float(aba["cosine"]),
        "slide18": slide_max,
        "slide_v3": slide_max,
        "wave": 0.0,
        "aba3": aba3,
        "ab2": ab2,
        "channel_max": float(max(aba3, ab2)),
        "layers": list(wm_layers),
        "query_space": space,
        "slide_win": win,
        "hop": hop_use,
    }


def score_wav_once(
    model, wav, sr, cfg, cluster_dir=None, n_peel=None, period=3, hop=3, slide_win=18, **_kwargs
) -> dict:
    """兼容旧签名；``cluster_dir`` / ``sr`` 在 band 盲搜路径中未使用。"""
    return score_band_once(
        model,
        wav,
        cfg["table"],
        high_start=int(cfg["high_start"]),
        high_end=cfg.get("high_end"),
        layers=cfg.get("layers"),
        n_peel=n_peel,
        period=int(period),
        hop=int(hop),
        slide_win=int(slide_win),
        query_space=str(cfg.get("query_space") or "band"),
    )
