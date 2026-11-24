"""RVQ 中高层连续特征：查询 residual_n、替换带宽 band_zq、ABA embedding / 簇质心累加。

不依赖 token 对齐。

两种查询（6–14 时 n_peel=6, band_end=14）：

- residual_n：剥掉 L0..L_{n-1} 后的整段残差，含 L_n 及更深层（含未写入的 L15+ 宿主）
- band_zq：recon(L0..band_end) − recon(L0..n_peel-1) = Σ z_q(L_n..L_end)，与 ABA 模板同一空间

上线通道默认仍用 residual（`score_wav_channels(query_space="residual")`）。
带宽对照：`query_space="band"`，或 `run_speed_palindrome_feat` / `run_channel_band_gap_n3`。
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from .moss_io import SEED, set_seed
from .plan import apply_plan, build_rare_cycle_plan, resolve_wm_layers
from .rare_tokens import cluster_members, layer_aba, load_layer_mapping
from .wave import cosine_sim, zcosine_sim


def encode_hidden(model, wav_np: np.ndarray):
    """wav → encoder_hidden_states [B, D, T], lengths [B]。"""
    set_seed(SEED)
    device = next(model.parameters()).device
    wav = torch.from_numpy(np.ascontiguousarray(wav_np)).float().view(1, 1, -1).to(device)
    with torch.no_grad():
        out = model.encode(wav, return_dict=True)
    if isinstance(out, (tuple, list)):
        raise RuntimeError("encode(..., return_dict=True) 未返回 hidden")
    e = out.encoder_hidden_states
    if e is None:
        raise RuntimeError("encoder_hidden_states is None")
    lengths = out.audio_codes_lengths
    if lengths is None:
        lengths = torch.full((e.shape[0],), e.shape[-1], device=e.device, dtype=torch.long)
    return e, lengths


def rvq_residual_after(model, hidden: torch.Tensor, lengths: torch.Tensor, n_peel: int) -> np.ndarray:
    """RVQ 只跑完前 n_peel 层，返回之后的整段残差 [D, T']。"""
    quantizer = model.quantizer
    n_peel = max(0, min(int(n_peel), len(quantizer.quantizers)))
    z = quantizer.input_proj(hidden)
    batch, _, max_time = z.shape
    mask = torch.arange(max_time, device=z.device).expand(batch, max_time) < lengths.unsqueeze(1)
    residual = z.clone().float()
    with torch.no_grad():
        for i in range(n_peel):
            masked = residual * mask.unsqueeze(1)
            z_q_i, _, _ = quantizer.quantizers[i](masked)
            residual = residual - z_q_i * mask.unsqueeze(1)
        residual = residual * mask.unsqueeze(1)
        t_use = int(lengths[0].item())
        feat = residual[0, :, :t_use].detach().float().cpu().numpy()
    return feat


def residual_feat_of(model, wav_np: np.ndarray, n_peel: int) -> np.ndarray:
    hidden, lengths = encode_hidden(model, wav_np)
    return rvq_residual_after(model, hidden, lengths, n_peel)


def rvq_layer_zq_only(
    model, hidden: torch.Tensor, lengths: torch.Tensor, layer: int
) -> np.ndarray:
    """只保留第 ``layer`` 层的量化向量 [D, T]。

    残差先减去 L0..layer-1，量化得到该层 z_q 后继续减去，
    因此 L0..layer-1 与更深层都不进入返回值。
    """
    quantizer = model.quantizer
    n_q = len(quantizer.quantizers)
    layer = max(0, min(int(layer), n_q - 1))
    z = quantizer.input_proj(hidden)
    batch, _, max_time = z.shape
    mask = torch.arange(max_time, device=z.device).expand(batch, max_time) < lengths.unsqueeze(1)
    residual = z.clone().float()
    kept = None
    with torch.no_grad():
        t_use = int(lengths[0].item())
        for i in range(n_q):
            masked = residual * mask.unsqueeze(1)
            z_q_i, _, _ = quantizer.quantizers[i](masked)
            if i == layer:
                kept = (z_q_i * mask.unsqueeze(1))[0, :, :t_use].detach().float().cpu().numpy()
            residual = residual - z_q_i * mask.unsqueeze(1)
    if kept is None:
        return np.zeros((int(z.shape[1]), t_use), dtype=np.float32)
    return kept


def layer_zq_only_of(model, wav_np: np.ndarray, layer: int) -> np.ndarray:
    hidden, lengths = encode_hidden(model, wav_np)
    return rvq_layer_zq_only(model, hidden, lengths, layer)


def speech_minus_other_layers(
    model, hidden: torch.Tensor, lengths: torch.Tensor, keep_layer: int
) -> np.ndarray:
    """量化器入口的语音特征，减去除 keep_layer 以外每一层的 z_q。

    剩下的是该层 z_q，再加上 32 层都量化完之后的残差。不是 token 查表。
    """
    quantizer = model.quantizer
    n_q = len(quantizer.quantizers)
    keep = max(0, min(int(keep_layer), n_q - 1))
    z = quantizer.input_proj(hidden)
    batch, _, max_time = z.shape
    mask = torch.arange(max_time, device=z.device).expand(batch, max_time) < lengths.unsqueeze(1)
    speech = z.clone().float()
    residual = speech.clone()
    dropped = None
    with torch.no_grad():
        t_use = int(lengths[0].item())
        for i in range(n_q):
            masked = residual * mask.unsqueeze(1)
            z_q_i, _, _ = quantizer.quantizers[i](masked)
            if i != keep:
                term = z_q_i * mask.unsqueeze(1)
                dropped = term if dropped is None else dropped + term
            residual = residual - z_q_i * mask.unsqueeze(1)
        feat = speech if dropped is None else speech - dropped
        feat = (feat * mask.unsqueeze(1))[0, :, :t_use].detach().float().cpu().numpy()
    return feat


def speech_minus_other_layers_of(model, wav_np: np.ndarray, keep_layer: int) -> np.ndarray:
    hidden, lengths = encode_hidden(model, wav_np)
    return speech_minus_other_layers(model, hidden, lengths, keep_layer)


def rvq_residual_and_band(
    model,
    hidden: torch.Tensor,
    lengths: torch.Tensor,
    n_peel: int,
    band_end: int,
) -> tuple[np.ndarray, np.ndarray]:
    """一次 RVQ：residual_n 与 L_peel..L_end 的 z_q 之和。"""
    quantizer = model.quantizer
    n_q = len(quantizer.quantizers)
    peel = max(0, min(int(n_peel), n_q))
    end = max(peel - 1, min(int(band_end), n_q - 1))
    z = quantizer.input_proj(hidden)
    batch, _, max_time = z.shape
    mask = torch.arange(max_time, device=z.device).expand(batch, max_time) < lengths.unsqueeze(1)
    residual = z.clone().float()
    acc = None
    peel_feat = None
    with torch.no_grad():
        t_use = int(lengths[0].item())
        for i in range(end + 1):
            if i == peel:
                peel_feat = (residual * mask.unsqueeze(1))[0, :, :t_use].detach().float().cpu().numpy()
            masked = residual * mask.unsqueeze(1)
            z_q_i, _, _ = quantizer.quantizers[i](masked)
            residual = residual - z_q_i * mask.unsqueeze(1)
            if i >= peel:
                term = z_q_i * mask.unsqueeze(1)
                acc = term if acc is None else acc + term
        if peel_feat is None:
            peel_feat = (residual * mask.unsqueeze(1))[0, :, :t_use].detach().float().cpu().numpy()
        if acc is None:
            band = np.zeros((int(z.shape[1]), t_use), dtype=np.float32)
        else:
            band = acc[0, :, :t_use].detach().float().cpu().numpy()
    return peel_feat, band


def rvq_band_zq(
    model, hidden: torch.Tensor, lengths: torch.Tensor, layer_start: int, layer_end: int
) -> np.ndarray:
    """L_start..L_end 的量化贡献之和 [D, T']。

    等价于「前 (end+1) 层重建 − 前 start 层重建」：
    sum_{i=start}^{end} z_q^{(i)} = recon_{0..end} − recon_{0..start-1}。
    6–14 时就是 L0–L14 减去 L0–L5，去掉浅层宿主，也不含 L15+。
    """
    _residual, band = rvq_residual_and_band(model, hidden, lengths, layer_start, layer_end)
    return band


def band_zq_of(model, wav_np: np.ndarray, layer_start: int, layer_end: int) -> np.ndarray:
    hidden, lengths = encode_hidden(model, wav_np)
    return rvq_band_zq(model, hidden, lengths, layer_start, layer_end)


def band_zq_layers_of(model, wav_np: np.ndarray, layers) -> np.ndarray:
    """只累加指定写入层的 z_q（稀疏组合用）。连续段时与 band_zq_of(min,max) 等价。"""
    from .moss_io import encode_moss

    layer_ids = sorted({int(x) for x in layers})
    if not layer_ids:
        raise ValueError("band_zq_layers_of: empty layers")
    codes = encode_moss(model, wav_np)
    return codes_to_zq_sum(model, codes, layer_ids[0], layer_ids[-1], layers=layer_ids)


def residual_and_band_of(
    model, wav_np: np.ndarray, n_peel: int, band_end: int
) -> tuple[np.ndarray, np.ndarray]:
    """一次 encode + 一次 RVQ：residual_n（含更深宿主）和替换带宽 z_q 之和。"""
    hidden, lengths = encode_hidden(model, wav_np)
    return rvq_residual_and_band(model, hidden, lengths, n_peel, band_end)


def query_feat_of(
    model,
    wav_np: np.ndarray,
    n_peel: int,
    band_end: int,
    space: str = "residual",
) -> np.ndarray:
    """检测查询特征。space=residual 为现行；band 为替换层 z_q 之和。"""
    key = str(space or "residual").strip().lower()
    if key == "band":
        return band_zq_of(model, wav_np, n_peel, band_end)
    if key == "residual":
        return residual_feat_of(model, wav_np, n_peel)
    raise ValueError(f"query space must be residual|band, got {space!r}")


def feat_cosine(a: np.ndarray, b: np.ndarray) -> dict:
    t = min(int(a.shape[-1]), int(b.shape[-1]))
    if t <= 0 or a.ndim != 2 or b.ndim != 2:
        return {"cosine": 0.0, "zcosine": 0.0, "T": 0, "D": 0}
    aa = np.asarray(a[:, :t], dtype=np.float64).reshape(-1)
    bb = np.asarray(b[:, :t], dtype=np.float64).reshape(-1)
    return {
        "cosine": float(cosine_sim(aa, bb)),
        "zcosine": float(zcosine_sim(aa, bb)),
        "T": int(t),
        "D": int(a.shape[0]),
    }


def aba_code_grid(
    T: int,
    n_q: int,
    high_start: int,
    high_end: int | None,
    table: dict,
    layers=None,
    period: int = 3,
) -> np.ndarray:
    codes = np.zeros((int(n_q), int(T)), dtype=np.int64)
    plan = build_rare_cycle_plan(
        int(T),
        n_q=int(n_q),
        high_start=int(high_start),
        high_end=high_end,
        table=table,
        layers=layers,
        period=int(period),
    )
    mod, _ = apply_plan(codes, plan)
    return mod


def layer_zq_from_ids(quantizer, ids_1t: torch.Tensor) -> torch.Tensor:
    """码字 [1, T] → 与 RVQ residual 同空间的 z_q [1, D, T]。"""
    z = quantizer.decode_code(ids_1t).float()
    weight = getattr(quantizer.out_proj, "weight", None)
    if weight is not None and int(z.shape[1]) == int(weight.shape[1]):
        z = quantizer.out_proj(z).float()
    return z


def codes_to_zq_sum(
    model, codes: np.ndarray, high_start: int, high_end: int | None, layers=None
) -> np.ndarray:
    n_q, T = codes.shape
    layer_ids = resolve_wm_layers(n_q, high_start, high_end, layers)
    device = next(model.parameters()).device
    acc = None
    with torch.no_grad():
        for L in layer_ids:
            q = model.quantizer.quantizers[L]
            ids = torch.from_numpy(np.ascontiguousarray(codes[L])).long().view(1, T).to(device)
            z = layer_zq_from_ids(q, ids)
            acc = z if acc is None else acc + z
    if acc is None:
        return np.zeros((1, T), dtype=np.float32)
    return acc[0].detach().float().cpu().numpy()


def aba_embed_template(
    model, T: int, n_peel: int, tmpl_end: int | None, table: dict, layers=None, period: int = 3
) -> np.ndarray:
    n_q = int(len(model.quantizer.quantizers))
    grid = aba_code_grid(
        int(T), n_q, n_peel, tmpl_end, table, layers=layers, period=int(period)
    )
    return codes_to_zq_sum(model, grid, n_peel, tmpl_end, layers=layers)


def cluster_cycle_grid(
    T: int, n_q: int, high_start: int, high_end: int | None, table: dict, layers=None
) -> np.ndarray:
    layer_ids = resolve_wm_layers(n_q, high_start, high_end, layers)
    grid = np.full((int(n_q), int(T)), -1, dtype=np.int64)
    for L in layer_ids:
        picked = layer_aba(table, L)
        if not picked:
            continue
        _toks, cs = picked
        if not cs:
            continue
        grid[L] = [int(cs[t % len(cs)]) for t in range(int(T))]
    return grid


def load_kmeans_centroids(cluster_dir: str | Path, layer: int) -> dict[int, np.ndarray]:
    p = Path(cluster_dir) / f"layer_{int(layer):02d}_prototypes.json"
    if not p.exists():
        return {}
    raw = json.loads(p.read_text(encoding="utf-8"))
    out = {}
    for cid, item in (raw.get("clusters") or {}).items():
        c = item.get("centroid")
        if c is None:
            continue
        out[int(cid)] = np.asarray(c, dtype=np.float32).reshape(-1)
    return out


@torch.no_grad()
def member_mean_zq(model, layer: int, members: list[int]) -> torch.Tensor | None:
    if not members:
        return None
    q = model.quantizer.quantizers[int(layer)]
    device = next(model.parameters()).device
    acc = None
    for tok in members:
        ids = torch.tensor([[int(tok)]], device=device, dtype=torch.long)
        z = layer_zq_from_ids(q, ids)
        acc = z if acc is None else acc + z
    return acc / float(len(members))


@torch.no_grad()
def kmeans_proj_zq(model, layer: int, vec8: np.ndarray) -> torch.Tensor | None:
    q = model.quantizer.quantizers[int(layer)]
    device = next(model.parameters()).device
    v = torch.from_numpy(np.ascontiguousarray(vec8)).float().view(1, -1, 1).to(device)
    weight = getattr(q.out_proj, "weight", None)
    if weight is None:
        return v
    in_ch = int(weight.shape[1])
    if int(v.shape[1]) == in_ch:
        return q.out_proj(v).float()
    if int(v.shape[1]) == int(weight.shape[0]):
        return v
    return None


def centroids_to_sum(
    model,
    cluster_grid: np.ndarray,
    high_start: int,
    high_end: int | None,
    kind: str,
    cluster_dir: str | Path,
    layers=None,
) -> np.ndarray:
    n_q, T = cluster_grid.shape
    layer_ids = resolve_wm_layers(n_q, high_start, high_end, layers)
    device = next(model.parameters()).device
    cluster_dir = Path(cluster_dir)
    acc = None
    cache: dict[tuple[int, int], torch.Tensor | None] = {}
    with torch.no_grad():
        for L in layer_ids:
            mapping = load_layer_mapping(L, cluster_dir)
            members = cluster_members(mapping)
            km = load_kmeans_centroids(cluster_dir, L) if kind == "kmeans" else {}
            vecs: list[torch.Tensor | None] = []
            for t in range(T):
                cid = int(cluster_grid[L, t])
                if cid < 0:
                    vecs.append(None)
                    continue
                key = (L, cid)
                if key not in cache:
                    if kind == "member":
                        cache[key] = member_mean_zq(model, L, members.get(cid) or [])
                    elif cid not in km:
                        cache[key] = None
                    else:
                        cache[key] = kmeans_proj_zq(model, L, km[cid])
                vecs.append(cache[key])
            if all(v is None for v in vecs):
                continue
            D = next(v.shape[1] for v in vecs if v is not None)
            layer_map = torch.zeros(1, D, T, device=device, dtype=torch.float32)
            for t, zc in enumerate(vecs):
                if zc is not None:
                    layer_map[:, :, t] = zc[:, :, 0]
            acc = layer_map if acc is None else acc + layer_map
    if acc is None:
        return np.zeros((1, T), dtype=np.float32)
    return acc[0].detach().float().cpu().numpy()


def score_wav_vs_aba(model, wav: np.ndarray, table: dict, n_peel: int, tmpl_end: int | None) -> dict:
    feat = residual_feat_of(model, wav, n_peel)
    tmpl = aba_embed_template(model, int(feat.shape[-1]), n_peel, tmpl_end, table)
    if int(feat.shape[0]) != int(tmpl.shape[0]):
        raise RuntimeError(
            f"residual D={feat.shape[0]} vs ABA-embed D={tmpl.shape[0]}；空间没对齐"
        )
    n_q = int(len(model.quantizer.quantizers))
    sim = feat_cosine(feat, tmpl)
    sim["tmpl_end"] = int(n_q - 1) if tmpl_end is None else int(tmpl_end)
    sim["tmpl_layers"] = int(sim["tmpl_end"] - n_peel + 1)
    return sim


def windowed_feat_cosine(
    a: np.ndarray,
    b: np.ndarray,
    period: int = 3,
    hop: int | None = None,
    threshold: float = 0.5,
) -> dict:
    """沿时间轴按 ABA 周期切 [D, T]，每窗余弦后汇总（同一 T 对齐，不是裁剪滑动）。"""
    if a.ndim != 2 or b.ndim != 2:
        return {"n_windows": 0, "mean": 0.0, "median": 0.0, "frac_above": 0.0, "scores": []}
    t = min(int(a.shape[-1]), int(b.shape[-1]))
    period = max(1, int(period))
    hop = int(hop) if hop is not None else period
    hop = max(1, hop)
    scores = []
    for t0 in range(0, t - period + 1, hop):
        aa = np.asarray(a[:, t0 : t0 + period], dtype=np.float64).reshape(-1)
        bb = np.asarray(b[:, t0 : t0 + period], dtype=np.float64).reshape(-1)
        scores.append(float(cosine_sim(aa, bb)))
    arr = np.asarray(scores, dtype=np.float64)
    if arr.size == 0:
        whole = feat_cosine(a, b)
        arr = np.asarray([float(whole["cosine"])], dtype=np.float64)
    return {
        "n_windows": int(arr.size),
        "period": int(period),
        "hop": int(hop),
        "mean": float(arr.mean()),
        "median": float(np.median(arr)),
        "min": float(arr.min()),
        "max": float(arr.max()),
        "frac_above": float(np.mean(arr >= float(threshold))),
        "threshold": float(threshold),
        "scores": [float(x) for x in arr.tolist()[:64]],
    }


def tile_cycle(cycle: np.ndarray, length: int, phase: int = 0) -> np.ndarray:
    """把一个 ABA 周期 [D, P] 按 phase 循环铺到 length。"""
    cycle = np.asarray(cycle)
    if cycle.ndim != 2 or int(cycle.shape[-1]) <= 0:
        return np.zeros((int(cycle.shape[0]) if cycle.ndim == 2 else 1, int(length)), dtype=np.float32)
    period = int(cycle.shape[-1])
    idx = (np.arange(int(length), dtype=np.int64) + int(phase)) % period
    return cycle[:, idx]


def slide_feat_vs_cycle(
    feat: np.ndarray,
    cycle: np.ndarray,
    win: int = 18,
    hop: int = 3,
    threshold: float = 0.15,
) -> dict:
    """查询 residual 短窗 vs 循环 ABA embedding，试 period 个相位。

    裁剪后不要整段对按新 T 铺的模板；只要有一窗过 threshold 即印记尚存。
    默认 win=18 帧 ≈ 1.5 s（T≈50 / 4 s）。
    """
    if feat.ndim != 2 or cycle.ndim != 2 or int(feat.shape[0]) != int(cycle.shape[0]):
        return {
            "n_scores": 0,
            "n_t0": 0,
            "max": 0.0,
            "median": 0.0,
            "mean": 0.0,
            "min": 0.0,
            "frac_above": 0.0,
            "best_t0": 0,
            "best_phase": 0,
            "win": int(win),
            "hop": int(hop),
            "period": 0,
            "threshold": float(threshold),
            "scores": [],
            "windows": [],
            "t0_best": [],
        }
    t = int(feat.shape[-1])
    win = max(1, min(int(win), t))
    hop = max(1, int(hop))
    period = max(1, int(cycle.shape[-1]))
    scores = []
    meta = []
    for t0 in range(0, t - win + 1, hop):
        q = feat[:, t0 : t0 + win]
        for phase in range(period):
            tmpl = tile_cycle(cycle, win, phase)
            c = float(cosine_sim(q.reshape(-1).astype(np.float64), tmpl.reshape(-1).astype(np.float64)))
            scores.append(c)
            meta.append((t0, phase, c))
    arr = np.asarray(scores, dtype=np.float64)
    if arr.size == 0:
        tmpl = tile_cycle(cycle, t, 0)
        c = float(cosine_sim(feat.reshape(-1).astype(np.float64), tmpl.reshape(-1).astype(np.float64)))
        arr = np.asarray([c], dtype=np.float64)
        meta = [(0, 0, c)]
    best_i = int(np.argmax(arr))
    best_t0, best_phase, _ = meta[best_i]
    # 按 t0 汇总：每个时间起点取最佳相位（便于看 ABA/BAB 相位与裁剪定位）
    by_t0: dict[int, dict] = {}
    for t0, phase, c in meta:
        prev = by_t0.get(int(t0))
        if prev is None or float(c) > float(prev["score"]):
            by_t0[int(t0)] = {"t0": int(t0), "phase": int(phase), "score": float(c)}
    t0_curve = [by_t0[k] for k in sorted(by_t0)]
    return {
        "n_scores": int(arr.size),
        "n_t0": int(len(t0_curve)),
        "win": int(win),
        "hop": int(hop),
        "period": int(period),
        "mean": float(arr.mean()),
        "median": float(np.median(arr)),
        "min": float(arr.min()),
        "max": float(arr.max()),
        "frac_above": float(np.mean(arr >= float(threshold))),
        "threshold": float(threshold),
        "best_t0": int(best_t0),
        "best_phase": int(best_phase),
        # 全量保留，供定标后 / 1500 评估 / 裁剪分析（勿截断）
        "scores": [float(x) for x in arr.tolist()],
        "windows": [
            {"t0": int(t0), "phase": int(phase), "score": float(c)} for t0, phase, c in meta
        ],
        "t0_best": t0_curve,
    }


def blind_period_slide(
    feat: np.ndarray,
    aba_cycle: np.ndarray,
    period: int = 3,
    threshold: float = 0.15,
) -> dict:
    """盲同步：不知裁切边界时，枚举起/止 trim 到 period 整倍数，再 slide ABA/AB。

    适用于攻击者任意裁剪后只剩一段 wav、检测端不知道裁了多少的情况。
    在 residual 帧轴上尝试 start_phase/end_trim ∈ {0..period-1}，
    取中间最长 period 整倍数段，分别用 ABA(win=period) 与 AB(win=2) 滑动，取全局 max。
    """
    if feat.ndim != 2 or aba_cycle.ndim != 2:
        return {"max": 0.0, "best": {}, "n_trials": 0, "trials_top5": []}
    feat = np.asarray(feat, dtype=np.float32)
    t = int(feat.shape[-1])
    period = max(1, int(period))
    ab_cycle = np.asarray(aba_cycle[:, : min(2, int(aba_cycle.shape[-1]))], dtype=np.float32)
    best_score = -1.0
    best: dict = {}
    trials: list[dict] = []
    modes = (
        ("aba3", aba_cycle, period, 1),
        ("ab2", ab_cycle, 2, 1),
    )
    for start_p in range(period):
        for end_p in range(period):
            t_end = t - int(end_p)
            if t_end <= int(start_p):
                continue
            seg = feat[:, int(start_p) : t_end]
            aligned = (int(seg.shape[-1]) // period) * period
            if aligned < period:
                continue
            seg = seg[:, :aligned]
            for mode, cycle, win, hop in modes:
                if int(cycle.shape[-1]) < 1:
                    continue
                sl = slide_feat_vs_cycle(
                    seg, cycle, win=int(win), hop=int(hop), threshold=float(threshold)
                )
                rec = {
                    "start_phase": int(start_p),
                    "end_trim": int(end_p),
                    "aligned_T": int(aligned),
                    "mode": mode,
                    "max": float(sl["max"]),
                    "best_t0": sl.get("best_t0"),
                    "best_phase": sl.get("best_phase"),
                }
                trials.append(rec)
                if float(sl["max"]) > best_score:
                    best_score = float(sl["max"])
                    best = dict(rec)
    trials.sort(key=lambda x: -float(x["max"]))
    return {
        "max": float(best_score if best_score >= 0 else 0.0),
        "best": best,
        "n_trials": int(len(trials)),
        "trials_top5": trials[:5],
    }
