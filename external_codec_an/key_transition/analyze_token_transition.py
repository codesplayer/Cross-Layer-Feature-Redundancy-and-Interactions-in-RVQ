"""
分析 token_stability 生成的 index 转移矩阵，看漂移有没有倾向性。

支持两种输入:
  1. 全局 dense:  trans_count.npy, 形状 [N, K, K]
  2. 单条稀疏:  *_trans.npz, 每层一个 [3, nnz] 数组

用法:
    python analysi/analyze_token_transition.py \
        --trans_npy .../token_stability/.../trans_count.npy \
        --topk 10 --layer 0

    # 或看某条音频
    python analysi/analyze_token_transition.py \
        --trans_npz .../token_stability/.../EN_xxx_trans.npz \
        --topk 10 --layer 0
"""

import argparse
import numpy as np


def top_transitions_dense(trans, layer, topk=10):
    mat = trans[layer].astype(np.int64).copy()
    np.fill_diagonal(mat, 0)
    flat_idx = np.argsort(mat.ravel())[::-1][:topk]
    K = mat.shape[0]
    out = []
    for idx in flat_idx:
        src, dst = divmod(int(idx), K)
        cnt = int(mat[src, dst])
        if cnt > 0:
            out.append((src, dst, cnt))
    return out


def top_transitions_sparse(npz, layer, topk=10):
    key = f"layer_{layer}"
    if key not in npz:
        return []
    arr = npz[key]  # [3, nnz]
    src, dst, cnt = arr[0], arr[1], arr[2]
    order = np.argsort(cnt)[::-1][:topk]
    return [(int(src[i]), int(dst[i]), int(cnt[i])) for i in order if cnt[i] > 0]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--trans_npy", type=str, default=None,
                        help="全局 dense 转移矩阵 trans_count.npy")
    parser.add_argument("--trans_npz", type=str, default=None,
                        help="单条音频的稀疏转移 *_trans.npz")
    parser.add_argument("--topk", type=int, default=10)
    parser.add_argument("--layer", type=int, default=None)
    args = parser.parse_args()

    if not args.trans_npy and not args.trans_npz:
        raise SystemExit("need --trans_npy or --trans_npz")

    if args.trans_npz:
        npz = np.load(args.trans_npz)
        layers = [args.layer] if args.layer is not None else [
            int(k.split("_")[1]) for k in npz.keys() if k.startswith("layer_")
        ]
        for l in sorted(layers):
            print(f"\n=== Layer {l} (sparse) ===")
            rows = top_transitions_sparse(npz, l, args.topk)
            if not rows:
                print("  (no drift)")
                continue
            for src, dst, cnt in rows:
                print(f"  {src:>5} -> {dst:<5}  count={cnt}")
        return

    trans = np.load(args.trans_npy)
    n_layers, K, _ = trans.shape
    print(f"layers={n_layers}, codebook={K}")
    layers = [args.layer] if args.layer is not None else range(n_layers)
    for l in layers:
        print(f"\n=== Layer {l} ===")
        rows = top_transitions_dense(trans, l, args.topk)
        if not rows:
            print("  (no drift)")
            continue
        for src, dst, cnt in rows:
            print(f"  {src:>5} -> {dst:<5}  count={cnt}")


if __name__ == "__main__":
    main()
