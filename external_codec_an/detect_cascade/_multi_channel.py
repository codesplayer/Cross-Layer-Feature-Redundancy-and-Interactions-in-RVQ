"""多通道打分：高层波形、两路 residual_6、ABA embedding / 簇质心。

生产检测仍是高层波形余弦。这里的通道和融合规则只给未来实验用，不要改默认 CLI。
3 句干净数据上的绝对分不在同一量级，禁止两路都用 0.65 做 AND。
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from watermark.future_exp._score_variants import windowed_high_cosine
from watermark.isolate import rebuild_high_residual
from watermark.moss_io import encode_moss
from watermark.plan import build_rare_cycle_plan, resize_plan, resolve_wm_layers
from watermark.rare_tokens import DEFAULT_CLUSTER_DIR
from watermark.residual_feat import (
    aba_embed_template,
    centroids_to_sum,
    cluster_cycle_grid,
    feat_cosine,
    query_feat_of,
    slide_feat_vs_cycle,
    windowed_feat_cosine,
)

# 3 句干净数据上的示意门限，不是规模 ROC 结论。
# 仅用于级联 flags；定标 ROC 用 ROC_KEYS（含 residual / slide_v3）。
ILLUSTRATIVE_TH = {
    "wave": 0.65,
    "residual_pair": 0.50,
    "aba_embed": 0.15,
    "slide_v3": 0.15,
    "centroid_member": 0.10,
    "centroid_kmeans": 0.12,
}

CHANNEL_KEYS = tuple(ILLUSTRATIVE_TH.keys())

# 与选层 DETECT_KEYS 对齐：定标 / 攻击报全通道 ROC
ROC_KEYS = (
    "wave",
    "residual",
    "residual_pair",
    "aba_embed",
    "centroid_member",
    "centroid_kmeans",
    "slide_v3",
)


def compact_channel(sim: dict) -> dict:
    return {
        "cosine": float(sim.get("cosine") or 0.0),
        "zcosine": float(sim.get("zcosine") or 0.0),
        "T": int(sim.get("T") or 0),
        "D": int(sim.get("D") or 0),
    }


def cascade_decide_scores(
    wave: float,
    aba: float,
    slide: float,
    t_wave: float,
    t_aba: float,
    t_slide: float,
) -> tuple[bool, str, bool]:
    """级联：主判 aba_embed ∨ slide_v3；主判不过再用 wave 兜底。

    返回 (present, reason, ran_wave_stage)。
    reason: latent_embed | latent_slide | wave_fallback | absent
    """
    if float(aba) >= float(t_aba):
        return True, "latent_embed", False
    if float(slide) >= float(t_slide):
        return True, "latent_slide", False
    if float(wave) >= float(t_wave):
        return True, "wave_fallback", True
    return False, "absent", True


def fuse_channels(scores: dict, thresholds: dict | None = None) -> dict:
    """级联门控：主判 embed/slide，次判 wave。

    二值「有水印」= (aba≥T_aba ∨ slide≥T_slide) ∨ wave≥T_wave（仅主判均失败时看 wave）。
    主判的价值是潜空间/滑窗对 orig 基线更稳；wave 作兜底，避免主判边界漏检。
    禁止多路共用 0.65 做 AND，禁止加权混合分。
    """
    th = dict(ILLUSTRATIVE_TH)
    if thresholds:
        th.update({k: float(v) for k, v in thresholds.items() if k in th})
    flags = {k: float(scores.get(k) or 0.0) >= float(th[k]) for k in CHANNEL_KEYS}
    slide_flag = float(scores.get("slide_v3") or 0.0) >= float(th["slide_v3"])
    present, reason, ran_wave = cascade_decide_scores(
        scores.get("wave") or 0.0,
        scores.get("aba_embed") or 0.0,
        scores.get("slide_v3") or 0.0,
        th["wave"],
        th["aba_embed"],
        th["slide_v3"],
    )
    embed_primary = bool(flags["aba_embed"])
    slide_primary = bool(slide_flag) and not embed_primary
    wave_fallback = reason == "wave_fallback"
    return {
        "thresholds": th,
        "flags": flags,
        "slide_v3_flag": slide_flag,
        "embed_primary": embed_primary,
        "slide_primary": slide_primary,
        "wave_fallback": wave_fallback,
        "wave_only": wave_fallback,
        "residual_pair_only": bool(flags["residual_pair"]),
        "aba_embed_only": embed_primary,
        "cascaded_present": present,
        "cascaded_reason": reason,
        "cascaded_ran_wave": ran_wave,
        "cascaded_ran_v3": ran_wave,
        "or_wave_embed": bool(present),
        "or_any": bool(any(flags.values()) or slide_flag),
        "wave_and_embed_sep": bool(flags["wave"] and flags["aba_embed"]),
        "wave_and_pair_sep": bool(flags["wave"] and flags["residual_pair"]),
        "note": (
            "cascade: primary aba_embed OR slide_v3, secondary wave; "
            "thresholds illustrative until Sprint 1 ROC"
        ),
    }


def score_wav_channels(
    model,
    wav: np.ndarray,
    sr: int,
    table: dict,
    high_start: int,
    high_end: int | None,
    cluster_dir: str | Path | None = None,
    n_peel: int | None = None,
    period: int = 3,
    hop: int | None = None,
    slide_win: int = 18,
    layers=None,
    query_space: str = "residual",
) -> dict:
    """对一条 wav 打全部通道。嵌入不变；查询各自取特征。

    query_space:
      residual — 现行 residual_n（剥浅层后含更深宿主）
      band — recon(0..high_end) − recon(0..n_peel-1)，只留替换层 z_q
    """
    cluster_dir = Path(cluster_dir or DEFAULT_CLUSTER_DIR)
    wav = np.asarray(wav, dtype=np.float32).reshape(-1)
    tokens = encode_moss(model, wav)
    n_q, T = tokens.shape
    last = int(n_q - 1) if high_end is None else int(high_end)
    wm_layers = resolve_wm_layers(n_q, high_start, last, layers)
    n_peel = int(min(wm_layers) if n_peel is None else n_peel)
    space = str(query_space or "residual").strip().lower()
    if space not in ("residual", "band"):
        raise ValueError(f"query_space must be residual|band, got {query_space!r}")

    def query_feat(audio: np.ndarray) -> np.ndarray:
        if space == "band":
            from watermark.residual_feat import band_zq_layers_of

            # 稀疏写入时只加写入层 z_q，避免 min..max 宿主冲淡模板
            return band_zq_layers_of(model, audio, wm_layers)
        return query_feat_of(model, audio, n_peel, last, space)

    plan = build_rare_cycle_plan(
        int(T),
        n_q=int(n_q),
        high_start=int(high_start),
        high_end=last,
        table=table,
        layers=wm_layers,
        period=int(period),
    )
    plan = resize_plan(plan, int(T))
    scored = rebuild_high_residual(
        model,
        tokens,
        plan,
        int(sr),
        high_start=int(high_start),
        high_end=last,
        layers=wm_layers,
    )
    high_enc = np.asarray(scored["high_enc"], dtype=np.float32).reshape(-1)
    high_pal = np.asarray(scored["high_pal"], dtype=np.float32).reshape(-1)
    fa = query_feat(high_enc)
    fb = query_feat(high_pal)
    pair = feat_cosine(fa, fb)

    feat_q = query_feat(wav)
    t_feat = int(feat_q.shape[-1])
    token_tmpl = aba_embed_template(
        model, t_feat, n_peel, last, table, layers=wm_layers, period=int(period)
    )
    aba = feat_cosine(feat_q, token_tmpl)
    cgrid = cluster_cycle_grid(t_feat, n_q, n_peel, last, table, layers=wm_layers)
    member = feat_cosine(
        feat_q,
        centroids_to_sum(
            model, cgrid, n_peel, last, "member", cluster_dir, layers=wm_layers
        ),
    )
    kmeans = feat_cosine(
        feat_q,
        centroids_to_sum(
            model, cgrid, n_peel, last, "kmeans", cluster_dir, layers=wm_layers
        ),
    )
    hop_use = int(hop) if hop is not None else int(period)
    aba_cycle = aba_embed_template(
        model, int(period), n_peel, last, table, layers=wm_layers, period=int(period)
    )

    slide = slide_feat_vs_cycle(
        feat_q,
        aba_cycle,
        win=int(slide_win),
        hop=hop_use,
        threshold=ILLUSTRATIVE_TH["aba_embed"],
    )
    slide_aba = slide_feat_vs_cycle(
        feat_q,
        aba_cycle,
        win=int(period),
        hop=1,
        threshold=ILLUSTRATIVE_TH["aba_embed"],
    )
    ab_cycle = np.asarray(aba_cycle[:, :2], dtype=np.float32)
    slide_ab = slide_feat_vs_cycle(
        feat_q,
        ab_cycle,
        win=2,
        hop=1,
        threshold=ILLUSTRATIVE_TH["aba_embed"],
    )
    scores = {
        "wave": float(scored["sim_high_only"]["cosine"]),
        "residual": float((scored.get("sim_residual") or {}).get("cosine") or 0.0),
        "residual_pair": float(pair["cosine"]),
        "aba_embed": float(aba["cosine"]),
        "centroid_member": float(member["cosine"]),
        "centroid_kmeans": float(kmeans["cosine"]),
        "slide_v3": float(slide["max"]),
    }
    return {
        "T": int(T),
        "n_peel": int(n_peel),
        "high_end": int(last),
        "query_space": space,
        "layers": list(wm_layers),
        "token_agree": scored["token_agree"],
        "sim_high_only": scored["sim_high_only"],
        "sim_residual": scored["sim_residual"],
        "residual_pair": compact_channel(pair),
        "aba_embed": compact_channel(aba),
        "centroid_member": compact_channel(member),
        "centroid_kmeans": compact_channel(kmeans),
        "scores": scores,
        "fusion": fuse_channels(scores),
        "window_wave": windowed_high_cosine(
            high_enc,
            high_pal,
            int(T),
            period=int(period),
            hop=hop_use,
            threshold=ILLUSTRATIVE_TH["wave"],
        ),
        "window_residual_pair": windowed_feat_cosine(
            fa, fb, period=int(period), hop=hop_use, threshold=ILLUSTRATIVE_TH["residual_pair"]
        ),
        "window_aba_embed": windowed_feat_cosine(
            feat_q,
            token_tmpl,
            period=int(period),
            hop=hop_use,
            threshold=ILLUSTRATIVE_TH["aba_embed"],
        ),
        "slide_v3": slide,
        "slide_aba_win3_h1": slide_aba,
        "slide_ab_win2_h1": slide_ab,
    }


FEAT_KEYS = (
    "aba_embed",
    "centroid_member",
    "centroid_kmeans",
    "slide_v3",
)


def score_feat_query(
    model,
    feat_q: np.ndarray,
    n_q: int,
    n_peel: int,
    last: int,
    table: dict,
    cluster_dir: str | Path | None = None,
    period: int = 3,
    hop: int | None = None,
    slide_win: int = 18,
    layers=None,
) -> dict:
    """只打特征通道：aba / 质心 / slide。不 decode、不算 wave / residual wav / residual_pair。"""
    cluster_dir = Path(cluster_dir or DEFAULT_CLUSTER_DIR)
    wm_layers = resolve_wm_layers(int(n_q), int(n_peel), int(last), layers)
    t_feat = int(np.asarray(feat_q).shape[-1])
    token_tmpl = aba_embed_template(
        model, t_feat, int(n_peel), int(last), table, layers=wm_layers, period=int(period)
    )
    aba = feat_cosine(feat_q, token_tmpl)
    cgrid = cluster_cycle_grid(t_feat, int(n_q), int(n_peel), int(last), table, layers=wm_layers)
    member = feat_cosine(
        feat_q,
        centroids_to_sum(model, cgrid, int(n_peel), int(last), "member", cluster_dir, layers=wm_layers),
    )
    kmeans = feat_cosine(
        feat_q,
        centroids_to_sum(model, cgrid, int(n_peel), int(last), "kmeans", cluster_dir, layers=wm_layers),
    )
    hop_use = int(hop) if hop is not None else int(period)
    aba_cycle = aba_embed_template(
        model, int(period), int(n_peel), int(last), table, layers=wm_layers, period=int(period)
    )
    slide = slide_feat_vs_cycle(
        feat_q,
        aba_cycle,
        win=int(slide_win),
        hop=hop_use,
        threshold=ILLUSTRATIVE_TH["aba_embed"],
    )
    return {
        "aba_embed": float(aba["cosine"]),
        "centroid_member": float(member["cosine"]),
        "centroid_kmeans": float(kmeans["cosine"]),
        "slide_v3": float(slide["max"]),
    }


def score_feat_residual(
    model,
    wav: np.ndarray,
    table: dict,
    layer: int,
    cluster_dir: str | Path | None = None,
    period: int = 3,
    hop: int | None = None,
    slide_win: int = 18,
) -> dict:
    """单层 L，residual 查询：剥 L 之前，特征含该层及更深宿主。"""
    from watermark.residual_feat import residual_feat_of

    L = int(layer)
    wav = np.asarray(wav, dtype=np.float32).reshape(-1)
    feat = residual_feat_of(model, wav, L)
    n_q = int(len(model.quantizer.quantizers))
    return score_feat_query(
        model, feat, n_q, L, L, table, cluster_dir, period, hop, slide_win, layers=[L]
    )


def score_feat_band(
    model,
    wav: np.ndarray,
    table: dict,
    layer: int | None = None,
    cluster_dir: str | Path | None = None,
    period: int = 3,
    hop: int | None = None,
    slide_win: int = 18,
    layers=None,
) -> dict:
    """band 查询：只留写入层 z_q。单层传 `layer`；连续/稀疏段传 `layers`。

    稀疏段只累加写入层（与 ABA 模板同一层集）；连续段等价于 min..max 的 z_q 和。
    """
    from watermark.residual_feat import band_zq_layers_of

    if layers is not None:
        wm_layers = sorted({int(x) for x in layers})
        if not wm_layers:
            raise ValueError("score_feat_band: empty layers")
        lo, hi = int(wm_layers[0]), int(wm_layers[-1])
    else:
        if layer is None:
            raise ValueError("score_feat_band needs layer or layers")
        lo = hi = int(layer)
        wm_layers = [lo]
    wav = np.asarray(wav, dtype=np.float32).reshape(-1)
    feat = band_zq_layers_of(model, wav, wm_layers)
    n_q = int(len(model.quantizer.quantizers))
    return score_feat_query(
        model, feat, n_q, lo, hi, table, cluster_dir, period, hop, slide_win, layers=wm_layers
    )


def _score_of(block: dict, key: str) -> float:
    scores = block.get("scores") if isinstance(block.get("scores"), dict) else block
    return float((scores or {}).get(key) or 0.0)


def channel_roc(rows: list[dict], key: str) -> dict:
    from watermark.future_exp._common import roc_and_eer

    pos = [_score_of(r.get("wm") or {}, key) for r in rows]
    neg = [_score_of(r.get("orig") or {}, key) for r in rows]
    return roc_and_eer(pos, neg)
