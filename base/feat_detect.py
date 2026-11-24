"""L* 层特征统计检测（类 Hidden in Plain Tokens，但在特征空间）。

论文：绿/红名单 + 再编码后数绿 token → 二项 p-value。
这里：绿码本向量 bank + 该层 ``z_q`` 帧余弦是否过阈 → 绿帧比例 → 二项 p-value。

``H0``：每帧独立以概率 ``gamma`` 越过阈值（由干净句标定，或固定）。
``H1``：水印句绿帧率显著高于 ``gamma``。
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .rare_tokens import cluster_members, layer_aba, load_layer_mapping
from .residual_feat import layer_zq_from_ids, layer_zq_only_of


def get_binomial_pval(x: float, n: float, p: float) -> float:
    """右尾：观察到 >= x 次成功时，在 Binomial(n,p) 下的 p-value。"""
    from scipy import special

    if n <= 0:
        return 1.0
    p = float(min(max(p, 1e-12), 1.0 - 1e-12))
    x = float(min(max(x, 0.0), n))
    # P(X >= x) = I_p(x, n-x+1) regularized incomplete beta (survival)
    return float(special.betainc(x, 1.0 + n - x, p))


def _zq_of_token(model, layer: int, token_id: int) -> np.ndarray:
    import torch

    device = next(model.parameters()).device
    q = model.quantizer.quantizers[int(layer)]
    ids = torch.tensor([[int(token_id)]], dtype=torch.long, device=device)
    z = layer_zq_from_ids(q, ids)  # [1, D, 1]
    return z[0, :, 0].detach().float().cpu().numpy()


def build_green_bank(
    model,
    table: dict,
    *,
    wm_layer: int = 9,
    cluster_dir: str | Path | None = None,
    expand_cluster: bool = True,
) -> dict:
    """绿特征 = rare ABA 码字向量；可选扩到同簇其它 token（类论文的簇稳健）。"""
    packed = layer_aba(table, int(wm_layer))
    if not packed:
        raise ValueError(f"no rare ABA for layer {wm_layer}")
    aba = [int(x) for x in packed[0]]
    token_ids = list(dict.fromkeys(aba))
    mapping = {}
    if expand_cluster and cluster_dir and Path(cluster_dir).exists():
        mapping = load_layer_mapping(int(wm_layer), cluster_dir)
        if mapping:
            members = cluster_members(mapping)
            clusters = {int(mapping[t]) for t in token_ids if int(t) in mapping}
            for cid in clusters:
                for t in members.get(int(cid), []):
                    if int(t) not in token_ids:
                        token_ids.append(int(t))

    vecs = []
    for tid in token_ids:
        v = _zq_of_token(model, int(wm_layer), int(tid))
        n = float(np.linalg.norm(v) + 1e-12)
        vecs.append((v / n).astype(np.float32))
    bank = np.stack(vecs, axis=0) if vecs else np.zeros((1, 1), dtype=np.float32)
    return {
        "wm_layer": int(wm_layer),
        "aba": aba,
        "token_ids": token_ids,
        "n_green_tokens": len(token_ids),
        "expand_cluster": bool(expand_cluster),
        "bank": bank,  # [G, D] L2-normalized
    }


def _frame_max_cos(feat_dt: np.ndarray, bank: np.ndarray) -> np.ndarray:
    """feat [D,T], bank [G,D] → per-frame max cosine [T]."""
    f = np.asarray(feat_dt, dtype=np.float64)
    b = np.asarray(bank, dtype=np.float64)
    if f.ndim != 2 or b.ndim != 2 or f.size == 0 or b.size == 0:
        return np.zeros((0,), dtype=np.float64)
    # normalize frames
    norms = np.linalg.norm(f, axis=0, keepdims=True) + 1e-12
    fn = f / norms
    # bank rows already unit; cos[g,t] = bank @ fn
    cos = b @ fn
    return cos.max(axis=0)


def detect_layer_feat(
    model,
    wav: np.ndarray,
    green: dict,
    *,
    tau: float,
    gamma: float,
    detect_p: float = 0.05,
) -> dict:
    """对该层特征做绿帧统计 + 二项检验。"""
    layer = int(green["wm_layer"])
    feat = layer_zq_only_of(model, wav, layer)
    scores = _frame_max_cos(feat, green["bank"])
    t = int(scores.size)
    hits = scores >= float(tau)
    n_green = int(np.sum(hits))
    green_rate = float(n_green / t) if t else float("nan")
    pval = get_binomial_pval(n_green, t, float(gamma))
    return {
        "wm_layer": layer,
        "T": t,
        "tau": float(tau),
        "gamma": float(gamma),
        "n_green": n_green,
        "green_rate": green_rate,
        "pval": float(pval),
        "detected": bool(pval < float(detect_p)),
        "mean_max_cos": float(np.mean(scores)) if t else float("nan"),
        "p50_max_cos": float(np.median(scores)) if t else float("nan"),
        "p90_max_cos": float(np.quantile(scores, 0.9)) if t else float("nan"),
    }


def calibrate_tau_gamma(
    orig_scores: list[np.ndarray],
    *,
    gamma_target: float = 0.25,
) -> tuple[float, float]:
    """用干净句的 max-cos 分布标定阈值：使平均绿帧率 ≈ gamma_target。"""
    all_s = np.concatenate([np.asarray(s, dtype=np.float64).reshape(-1) for s in orig_scores])
    if all_s.size == 0:
        return 0.5, float(gamma_target)
    # tau = quantile such that P(score >= tau) ≈ gamma_target
    tau = float(np.quantile(all_s, 1.0 - float(gamma_target)))
    # empirical gamma under this tau
    gamma = float(np.mean(all_s >= tau))
    gamma = min(max(gamma, 1e-4), 1.0 - 1e-4)
    return tau, gamma
