"""从语料转移统计里筛「簇少用、token 少用」的全局水印 ABA。

所有句子共用同一张表：嵌入整层铺 ABA，检测也用这张表，不必每句现算、现存。
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np

DEFAULT_CLUSTER_DIR = r"/data1/home/ynian/codes/codec_an/exp_watermark/codec_an/codec_plot/embedding_clusters/embedding_clusters_normalize"
DEFAULT_TRANS = r"d:\data\codec_an\codec_plot\codec_layers\analysis\trans_count.npy"
DEFAULT_RARE_PATH = r"/data1/home/ynian/codes/codec_an/exp_watermark/codec_an/anas/watermark_rare_tokens/rare_aba.json"
CODEBOOK_SIZE = 1024
N_LAYERS = 32


def load_layer_mapping(layer: int, cluster_dir: str | Path = DEFAULT_CLUSTER_DIR) -> dict[int, int]:
    p = Path(cluster_dir) / f"layer_{int(layer):02d}_index2cluster.json"
    if not p.exists():
        return {}
    raw = json.loads(p.read_text(encoding="utf-8"))
    return {int(k): int(v) for k, v in raw.items()}


def token_freq_from_trans(trans: np.ndarray) -> np.ndarray:
    """trans [n_layers, K, K] → 每层每个 token 作转移起点的次数 [n_layers, K]。"""
    trans = np.asarray(trans)
    if trans.ndim != 3:
        raise ValueError(f"trans must be [n_layers, K, K], got {trans.shape}")
    return trans.sum(axis=2).astype(np.int64)


def cluster_members(mapping: dict[int, int]) -> dict[int, list[int]]:
    by_c: dict[int, list[int]] = defaultdict(list)
    for tok, cid in mapping.items():
        if int(cid) >= 0:
            by_c[int(cid)].append(int(tok))
    return {c: sorted(v) for c, v in by_c.items()}


def cluster_freq_from_token_freq(
    tok_freq: np.ndarray,
    mapping: dict[int, int],
) -> dict[int, int]:
    out: dict[int, int] = defaultdict(int)
    n = int(tok_freq.shape[0])
    for tok, cid in mapping.items():
        tok_i, cid_i = int(tok), int(cid)
        if cid_i < 0:
            continue
        if 0 <= tok_i < n:
            out[cid_i] += int(tok_freq[tok_i])
        else:
            out[cid_i] += 0
    return dict(out)


def _rarest_token(members: list[int], tok_freq: np.ndarray, used: set[int]) -> int | None:
    cands = [int(t) for t in members if int(t) not in used]
    if not cands:
        cands = [int(t) for t in members]
    if not cands:
        return None
    n = int(tok_freq.shape[0])

    def key(t: int):
        f = int(tok_freq[t]) if 0 <= t < n else 0
        return (f, t)

    return min(cands, key=key)


def select_rare_aba_layer(
    tok_freq: np.ndarray,
    mapping: dict[int, int],
    used_tokens: set[int] | None = None,
) -> dict | None:
    """一层选两个最少用的簇，各取该簇里最少用的 token，拼 ABA。"""
    used = set(int(x) for x in (used_tokens or []))
    members = cluster_members(mapping)
    cf = cluster_freq_from_token_freq(tok_freq, mapping)
    if len(members) < 2:
        return None
    ranked = []
    for cid, mem in members.items():
        tok = _rarest_token(mem, tok_freq, used)
        if tok is None:
            continue
        n = int(tok_freq.shape[0])
        tf = int(tok_freq[tok]) if 0 <= tok < n else 0
        ranked.append((int(cf.get(cid, 0)), tf, int(cid), int(tok)))
    ranked.sort()
    picked = []
    seen_c, seen_t = set(), set(used)
    for cfreq, tf, cid, tok in ranked:
        if cid in seen_c or tok in seen_t:
            continue
        picked.append((cid, tok, cfreq, tf))
        seen_c.add(cid)
        seen_t.add(tok)
        if len(picked) >= 2:
            break
    if len(picked) < 2:
        return None
    (c0, t0, cf0, tf0), (c1, t1, cf1, tf1) = picked[0], picked[1]
    total_tok = int(tok_freq.sum()) or 1
    total_c = int(sum(cf.values())) or 1
    return {
        "aba": [int(t0), int(t1), int(t0)],
        "clusters": [int(c0), int(c1), int(c0)],
        "token_freq": [int(tf0), int(tf1)],
        "cluster_freq": [int(cf0), int(cf1)],
        "token_share": [float(tf0) / total_tok, float(tf1) / total_tok],
        "cluster_share": [float(cf0) / total_c, float(cf1) / total_c],
        "n_clusters": int(len(members)),
        "n_tokens": int(tok_freq.shape[0]),
    }


def build_rare_table(
    trans_path: str | Path = DEFAULT_TRANS,
    cluster_dir: str | Path = DEFAULT_CLUSTER_DIR,
    n_layers: int = N_LAYERS,
) -> dict:
    trans = np.load(str(trans_path))
    freq = token_freq_from_trans(trans)
    n_layers = min(int(n_layers), int(freq.shape[0]))
    used: set[int] = set()
    layers = {}
    for L in range(n_layers):
        mapping = load_layer_mapping(L, cluster_dir)
        picked = select_rare_aba_layer(freq[L], mapping, used_tokens=used)
        if not picked:
            continue
        used |= set(picked["aba"])
        layers[str(L)] = picked
    return {
        "trans_path": str(trans_path),
        "cluster_dir": str(cluster_dir),
        "n_layers": n_layers,
        "codebook_size": int(freq.shape[1]),
        "note": "global rare-cluster + rare-token ABA; embed and detect share this table",
        "layers": layers,
    }


def save_rare_table(table: dict, path: str | Path = DEFAULT_RARE_PATH) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(table, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def load_rare_table(path: str | Path = DEFAULT_RARE_PATH) -> dict:
    path = Path(path)
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def ensure_rare_table(
    path: str | Path = DEFAULT_RARE_PATH,
    trans_path: str | Path = DEFAULT_TRANS,
    cluster_dir: str | Path = DEFAULT_CLUSTER_DIR,
    rebuild: bool = False,
) -> dict:
    path = Path(path)
    if path.exists() and not rebuild:
        return load_rare_table(path)
    table = build_rare_table(trans_path, cluster_dir)
    save_rare_table(table, path)
    return table


def layer_aba(table: dict, layer: int) -> tuple[list[int], list[int]] | None:
    layers = (table or {}).get("layers") or {}
    item = layers.get(str(int(layer))) or layers.get(int(layer))
    if not item:
        return None
    aba = [int(x) for x in (item.get("aba") or [])]
    cs = [int(x) for x in (item.get("clusters") or [])]
    if len(aba) < 3 or len(aba) % 2 == 0 or aba != aba[::-1]:
        return None
    if len(cs) != len(aba):
        cs = []
    return aba, cs


def main():
    import argparse

    p = argparse.ArgumentParser(description="筛低频簇/低频 token 作为全局水印 ABA")
    p.add_argument("--trans", type=str, default=DEFAULT_TRANS)
    p.add_argument("--cluster_dir", type=str, default=DEFAULT_CLUSTER_DIR)
    p.add_argument("--out", type=str, default=DEFAULT_RARE_PATH)
    args = p.parse_args()
    table = build_rare_table(args.trans, args.cluster_dir)
    save_rare_table(table, args.out)
    print("saved", args.out, "layers", len(table.get("layers") or {}))
    for L in range(6, min(12, table.get("n_layers", 0))):
        item = table["layers"].get(str(L))
        if not item:
            continue
        print(
            f"  L{L} aba={item['aba']} clusters={item['clusters']} "
            f"tok_freq={item['token_freq']} cluster_freq={item['cluster_freq']}"
        )


if __name__ == "__main__":
    main()
