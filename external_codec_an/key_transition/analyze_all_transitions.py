"""
聚合所有 *_trans.npz，分析全局 index 转移倾向。

用法:
    python analyze_all_transitions.py \
        --input_dir d:/data/codec_an/codec_plot/codec_layers \
        --output_dir d:/data/codec_an/codec_plot/codec_layers/analysis \
        --topk 20 --codebook_size 1024
"""

import argparse
import json
import os
from pathlib import Path

import numpy as np
from tqdm import tqdm


def _lang_of(name):
    base = os.path.basename(name)
    if base.startswith("EN_"):
        return "EN"
    if base.startswith("ZH_"):
        return "ZH"
    return None


def accumulate_trans(files, codebook_size=1024, n_layers=32):
    """稀疏累加所有 trans.npz，返回 dense [N,K,K] 或稀疏 dict。"""
    # 用 dict 累加避免一开始就占 32*1024*1024*8 ≈ 256MB * 3
    # 但最终还是要 dense 做统计。对 K=1024, N=32, int64 ≈ 256MB 可接受。
    trans = np.zeros((n_layers, codebook_size, codebook_size), dtype=np.int64)
    n_ok = 0
    for f in tqdm(files, desc="Accumulate"):
        try:
            npz = np.load(f)
            for key in npz.keys():
                if not key.startswith("layer_"):
                    continue
                l = int(key.split("_")[1])
                if l >= n_layers:
                    continue
                arr = npz[key]  # [3, nnz]
                src, dst, cnt = arr[0], arr[1], arr[2]
                # 过滤越界
                mask = (src >= 0) & (src < codebook_size) & (dst >= 0) & (dst < codebook_size)
                np.add.at(trans[l], (src[mask], dst[mask]), cnt[mask])
            n_ok += 1
        except Exception as e:
            print(f"Skip {f}: {e}")
    return trans, n_ok


def analyze_layer(mat, topk=20):
    """分析单层转移矩阵。"""
    K = mat.shape[0]
    total = int(mat.sum())
    diag = int(np.trace(mat))
    off = total - diag

    # 去掉对角后的 top-k
    m = mat.astype(np.int64).copy()
    np.fill_diagonal(m, 0)
    flat = m.ravel()
    if flat.max() == 0:
        return {
            "total": total,
            "diag": diag,
            "off_diag": off,
            "top_transitions": [],
            "mean_abs_delta": 0.0,
            "median_abs_delta": 0.0,
            "local_ratio_16": 0.0,
            "local_ratio_64": 0.0,
            "entropy": 0.0,
            "concentration_top10": 0.0,
        }

    idx = np.argsort(flat)[::-1][:topk]
    top = []
    for i in idx:
        if flat[i] <= 0:
            break
        src, dst = divmod(int(i), K)
        top.append({"src": src, "dst": dst, "count": int(flat[i]), "delta": abs(src - dst)})

    # 迁移距离分布（只看 off-diag）
    src_idx, dst_idx = np.nonzero(m)
    counts = m[src_idx, dst_idx]
    deltas = np.abs(src_idx.astype(np.int64) - dst_idx.astype(np.int64))
    # 加权
    mean_delta = float(np.average(deltas, weights=counts)) if counts.sum() > 0 else 0.0
    # 近似 median：展开太贵，用分位数近似
    order = np.argsort(deltas)
    cum = np.cumsum(counts[order])
    med_idx = np.searchsorted(cum, counts.sum() / 2)
    median_delta = float(deltas[order[min(med_idx, len(deltas) - 1)]])

    local_16 = float(counts[deltas <= 16].sum() / max(counts.sum(), 1))
    local_64 = float(counts[deltas <= 64].sum() / max(counts.sum(), 1))

    # 转移集中度：top10 占比
    top10_sum = sum(t["count"] for t in top[:10])
    concentration = float(top10_sum / max(off, 1))

    # 熵（行归一化后平均）
    row_sum = mat.sum(axis=1, keepdims=True).astype(np.float64)
    row_sum = np.maximum(row_sum, 1)
    p = mat.astype(np.float64) / row_sum
    # 只对有质量的行
    active = (mat.sum(axis=1) > 0)
    if active.any():
        p_active = p[active]
        ent = -np.sum(p_active * np.log(p_active + 1e-12), axis=1)
        mean_ent = float(ent.mean())
    else:
        mean_ent = 0.0

    return {
        "total": total,
        "diag": diag,
        "off_diag": off,
        "top_transitions": top,
        "mean_abs_delta": mean_delta,
        "median_abs_delta": median_delta,
        "local_ratio_16": local_16,
        "local_ratio_64": local_64,
        "entropy": mean_ent,
        "concentration_top10": concentration,
    }


def classify_tendency(stats):
    """根据统计结果判断迁移倾向。"""
    local64 = stats["local_ratio_64"]
    mean_d = stats["mean_abs_delta"]
    conc = stats["concentration_top10"]
    if local64 > 0.5 and mean_d < 80:
        return "簇内/局部迁移为主"
    if local64 > 0.25 and mean_d < 200:
        return "局部 + 近簇间混合"
    if conc > 0.15:
        return "有明显偏好对，但跨距较大"
    return "接近随机/远距离迁移"


def analyze_group(trans, tag, topk, out_dir):
    n_layers = trans.shape[0]
    layer_stats = []
    print(f"\n=== {tag} ===")
    for l in range(n_layers):
        s = analyze_layer(trans[l], topk=topk)
        s["layer"] = l
        s["tendency"] = classify_tendency(s)
        layer_stats.append(s)
        print(
            f"Layer {l:2d}: off={s['off_diag']:8d}  "
            f"mean|Δ|={s['mean_abs_delta']:6.1f}  "
            f"local64={s['local_ratio_16']:.3f}/{s['local_ratio_64']:.3f}  "
            f"top10conc={s['concentration_top10']:.3f}  "
            f"-> {s['tendency']}"
        )
        if s["top_transitions"]:
            t0 = s["top_transitions"][0]
            print(f"         top1: {t0['src']} -> {t0['dst']} (count={t0['count']}, |Δ|={t0['delta']})")

    # 分层汇总
    shallow = layer_stats[0:4]
    mid = layer_stats[4:12]
    deep = layer_stats[12:]

    def avg(xs, key):
        return float(np.mean([x[key] for x in xs])) if xs else 0.0

    summary = {
        "tag": tag,
        "n_layers": n_layers,
        "shallow_0_3": {
            "mean_abs_delta": avg(shallow, "mean_abs_delta"),
            "local_ratio_64": avg(shallow, "local_ratio_64"),
            "concentration_top10": avg(shallow, "concentration_top10"),
            "tendency": classify_tendency({
                "local_ratio_64": avg(shallow, "local_ratio_64"),
                "mean_abs_delta": avg(shallow, "mean_abs_delta"),
                "concentration_top10": avg(shallow, "concentration_top10"),
            }),
        },
        "mid_4_11": {
            "mean_abs_delta": avg(mid, "mean_abs_delta"),
            "local_ratio_64": avg(mid, "local_ratio_64"),
            "concentration_top10": avg(mid, "concentration_top10"),
            "tendency": classify_tendency({
                "local_ratio_64": avg(mid, "local_ratio_64"),
                "mean_abs_delta": avg(mid, "mean_abs_delta"),
                "concentration_top10": avg(mid, "concentration_top10"),
            }),
        },
        "deep_12_plus": {
            "mean_abs_delta": avg(deep, "mean_abs_delta"),
            "local_ratio_64": avg(deep, "local_ratio_64"),
            "concentration_top10": avg(deep, "concentration_top10"),
            "tendency": classify_tendency({
                "local_ratio_64": avg(deep, "local_ratio_64"),
                "mean_abs_delta": avg(deep, "mean_abs_delta"),
                "concentration_top10": avg(deep, "concentration_top10"),
            }),
        },
        "layers": layer_stats,
    }

    path = os.path.join(out_dir, f"transition_analysis{tag}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"Saved {path}")

    # 保存 dense 转移矩阵（可选，较大）
    npy_path = os.path.join(out_dir, f"trans_count{tag}.npy")
    np.save(npy_path, trans)
    print(f"Saved {npy_path}")

    return summary


def plot_summary(summaries, out_dir):
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    for tag, summary in summaries.items():
        layers = [s["layer"] for s in summary["layers"]]
        mean_d = [s["mean_abs_delta"] for s in summary["layers"]]
        local64 = [s["local_ratio_64"] for s in summary["layers"]]
        conc = [s["concentration_top10"] for s in summary["layers"]]
        axes[0, 0].plot(layers, mean_d, marker="o", label=tag)
        axes[0, 1].plot(layers, local64, marker="o", label=tag)
        axes[1, 0].plot(layers, conc, marker="o", label=tag)

    axes[0, 0].set_title("Mean |src-dst| per layer")
    axes[0, 0].set_xlabel("Layer")
    axes[0, 0].set_ylabel("|Δ|")
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    axes[0, 1].set_title("Local migration ratio (|Δ|≤64)")
    axes[0, 1].set_xlabel("Layer")
    axes[0, 1].set_ylabel("Ratio")
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    axes[1, 0].set_title("Top-10 transition concentration")
    axes[1, 0].set_xlabel("Layer")
    axes[1, 0].set_ylabel("Concentration")
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    # 分层倾向柱状
    tags = list(summaries.keys())
    x = np.arange(3)
    width = 0.25
    for i, tag in enumerate(tags):
        s = summaries[tag]
        vals = [
            s["shallow_0_3"]["local_ratio_64"],
            s["mid_4_11"]["local_ratio_64"],
            s["deep_12_plus"]["local_ratio_64"],
        ]
        axes[1, 1].bar(x + i * width, vals, width, label=tag)
    axes[1, 1].set_xticks(x + width)
    axes[1, 1].set_xticklabels(["shallow 0-3", "mid 4-11", "deep 12+"])
    axes[1, 1].set_title("Local ratio by layer group")
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    path = os.path.join(out_dir, "transition_tendency.png")
    plt.savefig(path, dpi=200)
    plt.close()
    print(f"Saved {path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--topk", type=int, default=20)
    parser.add_argument("--codebook_size", type=int, default=1024)
    parser.add_argument("--n_layers", type=int, default=32)
    args = parser.parse_args()

    out_dir = args.output_dir or os.path.join(args.input_dir, "analysis")
    os.makedirs(out_dir, exist_ok=True)

    files = sorted(Path(args.input_dir).glob("*_trans.npz"))
    en_files = [f for f in files if _lang_of(f.name) == "EN"]
    zh_files = [f for f in files if _lang_of(f.name) == "ZH"]
    print(f"Found {len(files)} trans files (EN={len(en_files)}, ZH={len(zh_files)})")

    summaries = {}
    for tag, flist in [("", files), ("_EN", en_files), ("_ZH", zh_files)]:
        if not flist:
            continue
        trans, n_ok = accumulate_trans(flist, args.codebook_size, args.n_layers)
        print(f"Accumulated {n_ok}/{len(flist)} files for {tag or 'ALL'}")
        summaries[tag or "ALL"] = analyze_group(trans, tag, args.topk, out_dir)

    plot_summary(summaries, out_dir)

    # 打印分组结论
    print("\n========== 转移倾向总结 ==========")
    for tag, s in summaries.items():
        print(f"\n[{tag}]")
        print(f"  浅层(0-3): {s['shallow_0_3']['tendency']}  "
              f"mean|Δ|={s['shallow_0_3']['mean_abs_delta']:.1f}  "
              f"local64={s['shallow_0_3']['local_ratio_64']:.3f}")
        print(f"  中层(4-11): {s['mid_4_11']['tendency']}  "
              f"mean|Δ|={s['mid_4_11']['mean_abs_delta']:.1f}  "
              f"local64={s['mid_4_11']['local_ratio_64']:.3f}")
        print(f"  深层(12+): {s['deep_12_plus']['tendency']}  "
              f"mean|Δ|={s['deep_12_plus']['mean_abs_delta']:.1f}  "
              f"local64={s['deep_12_plus']['local_ratio_64']:.3f}")


if __name__ == "__main__":
    main()
