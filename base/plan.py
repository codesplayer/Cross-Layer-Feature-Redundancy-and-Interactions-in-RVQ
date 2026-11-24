"""全局低频 ABA 嵌入计划：只改选定层，其余层不动。

选定层可以是连续区间 high_start..high_end，也可以是跨层集合（如 4,8,12）。
"""

from __future__ import annotations

import numpy as np

CODEBOOK_SIZE = 1024
PALINDROME_LEN = 3
HIGH_START = 6
HIGH_END = 16


def cycle_fill_row(T: int, aba) -> tuple[list[int], list[int], list[int]]:
    """整层按 ABA 循环铺满。"""
    aba = [int(x) for x in (aba or [])]
    T = int(T)
    n = len(aba)
    if T <= 0 or n <= 0:
        return [], [], []
    positions = list(range(T))
    tokens = [aba[i % n] for i in positions]
    starts = list(range(0, max(0, T - n + 1), n))
    return positions, tokens, starts


def is_palindrome_seq(toks, length: int | None = None) -> bool:
    t = [int(x) for x in (toks or [])]
    n = int(length) if length is not None else PALINDROME_LEN
    return len(t) == n and n >= 3 and (n % 2 == 1) and t == t[::-1] and len(set(t)) >= 2


def is_cluster_palindrome(clusters, length: int | None = None) -> bool:
    c = [int(x) for x in (clusters or [])]
    n = int(length) if length is not None else PALINDROME_LEN
    return (
        len(c) == n
        and n >= 3
        and (n % 2 == 1)
        and c == c[::-1]
        and len(set(c)) >= 2
        and min(c) >= 0
    )


def fallback_aba(layer: int, codebook: int = CODEBOOK_SIZE, period: int = 3) -> list[int]:
    codebook = max(2, int(codebook))
    a = (int(layer) * 17) % codebook
    b = (a + 97) % codebook
    if b == a:
        b = (a + 1) % codebook
    return extend_aba_motif([int(a), int(b), int(a)], period)


def extend_aba_motif(aba, period: int) -> list[int]:
    """把稀有表上的 ABA=[A,B,A] 扩成奇数长度回文：3→ABA，5→ABABA，7→ABABABA。"""
    period = int(period)
    if period < 3 or period % 2 == 0:
        raise ValueError(f"period must be odd and >= 3, got {period}")
    aba = [int(x) for x in (aba or [])]
    if len(aba) < 2:
        raise ValueError("aba motif needs at least two symbols")
    a, b = int(aba[0]), int(aba[1])
    half = period // 2
    left = [a if i % 2 == 0 else b for i in range(half)]
    center = a if half % 2 == 0 else b
    return left + [center] + left[::-1]


def apply_plan(tokens: np.ndarray, plan: list[dict]) -> tuple[np.ndarray, int]:
    """tokens [N, T] -> 修改后的副本；返回 (mod, n_changed)。"""
    mod = tokens.copy()
    n_q, T = mod.shape
    changed = 0
    for item in plan:
        L = int(item["layer"])
        if L >= n_q:
            continue
        for p, tok in zip(item["positions"], item["tokens"]):
            if 0 <= int(p) < T and int(mod[L, int(p)]) != int(tok):
                mod[L, int(p)] = int(tok)
                changed += 1
    return mod, changed


def resize_plan(plan: list[dict], T: int) -> list[dict]:
    """检测时按当前帧长循环铺同一段 ABA。"""
    T = int(T)
    out = []
    for item in plan:
        it = dict(item)
        motif = [int(x) for x in (it.get("run_tokens") or it.get("tokens") or [])]
        if T <= 0 or not motif:
            it["positions"] = []
            it["tokens"] = []
            out.append(it)
            continue
        block_pos, block_toks, starts = cycle_fill_row(T, motif)
        it["tokens"] = block_toks
        it["positions"] = block_pos
        # keep full motif (may be period 3/5/7); do not truncate to PALINDROME_LEN
        it["run_tokens"] = list(motif)
        it["run_starts"] = starts
        it["n_runs"] = len(starts)
        it["period"] = int(len(motif)) if motif else int(it.get("period") or PALINDROME_LEN)
        out.append(it)
    return out


def resolve_wm_layers(
    n_q: int,
    high_start: int = HIGH_START,
    high_end: int | None = HIGH_END,
    layers=None,
) -> list[int]:
    """水印层列表。`layers` 优先；否则 high_start..high_end（含）。"""
    n_q = int(n_q)
    if layers is not None:
        out = sorted({int(L) for L in layers if 0 <= int(L) < n_q})
        if not out:
            raise ValueError("empty watermark layer set")
        return out
    high_start = int(high_start)
    last = int(n_q - 1) if high_end is None else int(high_end)
    last = min(max(high_start, last), n_q - 1)
    return list(range(high_start, last + 1))


def combo_label(layers) -> str:
    ls = [int(x) for x in layers]
    if not ls:
        return ""
    if ls == list(range(ls[0], ls[-1] + 1)):
        return f"{ls[0]}-{ls[-1]}"
    return ",".join(str(x) for x in ls)


def build_rare_cycle_plan(
    T: int,
    n_q: int = 32,
    high_start: int = HIGH_START,
    high_end: int | None = HIGH_END,
    table: dict | None = None,
    layers=None,
    period: int = 3,
) -> list[dict]:
    """只改选定层（含）；整层铺全局低频簇 ABA。其余层不动。

    `period`：回文周期长度，奇数，默认 3；5→ABABA，7→ABABABA（由稀有表 ABA 扩展）。
    """
    from .rare_tokens import ensure_rare_table, layer_aba

    T = int(T)
    n_q = int(n_q)
    period = int(period)
    if period < 3 or period % 2 == 0:
        raise ValueError(f"period must be odd and >= 3, got {period}")
    layer_ids = resolve_wm_layers(n_q, high_start, high_end, layers)
    if table is None:
        table = ensure_rare_table()
    plan: list[dict] = []
    for L in layer_ids:
        picked = layer_aba(table, L)
        extra = {}
        if picked:
            toks, cs = picked
            pattern = "rare_cluster_aba"
            item_src = ((table.get("layers") or {}).get(str(L)) or {})
            extra = {
                "token_freq": list(item_src.get("token_freq") or []),
                "cluster_freq": list(item_src.get("cluster_freq") or []),
            }
        else:
            toks = fallback_aba(L, period=3)
            cs = []
            pattern = "fallback_aba"
        # 表内已是当前周期的回文（如 ABCBA）则原样铺；否则把长度 3 的 ABA 扩成 ABABA。
        if len(toks) == int(period) and is_palindrome_seq(toks, length=int(period)):
            if not (cs and is_cluster_palindrome(cs, length=int(period))):
                cs = []
        else:
            if not is_palindrome_seq(toks, length=3):
                toks = fallback_aba(L, period=3)
                cs = []
                pattern = "fallback_aba"
                extra = {}
            toks = extend_aba_motif(toks, period)
            if cs and is_cluster_palindrome(cs, length=3):
                cs = extend_aba_motif(cs, period)
            else:
                cs = []
        block_pos, block_toks, starts = cycle_fill_row(T, toks)
        item = {
            "layer": int(L),
            "kind": "palindrome_cycle",
            "role": "payload",
            "fill": "cycle",
            "period": int(period),
            "cluster": cs[0] if cs else None,
            "cluster_seq": list(cs),
            "pattern": pattern,
            "n_unique": len(set(toks)),
            "n_unique_clusters": len(set(cs)),
            "run_tokens": list(toks),
            "run_starts": starts,
            "n_runs": len(starts),
            "positions": block_pos,
            "tokens": block_toks,
        }
        item.update(extra)
        plan.append(item)
    return plan
