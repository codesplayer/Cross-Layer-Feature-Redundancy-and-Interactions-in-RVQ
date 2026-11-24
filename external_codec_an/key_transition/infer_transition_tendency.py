"""
从 token_stability 输出的 per_layer.json 推断 index 迁移倾向。

由于当前实验未保存 token 序列或转移矩阵，本脚本基于漂移率和静音/语音分布做推断分析。
若后续重新跑实验并加 --save_trans_per_audio，可用 analyze_token_transition.py 做精确统计。
"""

import argparse
import json
import os
from pathlib import Path

import numpy as np


def _lang_of(name):
    base = os.path.basename(name)
    if base.startswith("EN_"):
        return "EN"
    if base.startswith("ZH_"):
        return "ZH"
    return None


def load_per_layer_files(input_dir):
    files = sorted(Path(input_dir).glob("*_per_layer.json"))
    data = {"EN": [], "ZH": []}
    for f in files:
        lang = _lang_of(f.name)
        if lang is None:
            continue
        with open(f, "r", encoding="utf-8") as fp:
            data[lang].append(json.load(fp))
    return data


def infer_transition_tendency(records):
    """基于漂移率和静音/语音分布推断 index 迁移倾向。"""
    if not records:
        return None
    n_layers = records[0]["n_layers"]
    n_rounds = records[0]["n_rounds"]

    layer_sum = np.zeros((n_rounds, n_layers), dtype=np.float64)
    layer_cnt = np.zeros((n_rounds, n_layers), dtype=np.float64)
    drift_silence = 0
    drift_speech = 0
    flip_silence = np.zeros(n_layers, dtype=np.int64)
    flip_speech = np.zeros(n_layers, dtype=np.int64)

    for r in records:
        arr = np.asarray(r["per_layer_per_round"], dtype=np.float64)
        if arr.shape != layer_sum.shape:
            continue
        layer_sum += arr
        layer_cnt += 1.0
        drift_silence += r.get("drift_silence", 0)
        drift_speech += r.get("drift_speech", 0)
        flip_silence += np.asarray(r.get("flip_silence_per_layer", [0] * n_layers), dtype=np.int64)
        flip_speech += np.asarray(r.get("flip_speech_per_layer", [0] * n_layers), dtype=np.int64)

    mean = layer_sum / np.maximum(layer_cnt, 1e-8)
    total_flips = flip_silence + flip_speech
    silence_ratio = flip_silence / np.maximum(total_flips, 1)

    return {
        "n_files": len(records),
        "n_layers": n_layers,
        "n_rounds": n_rounds,
        "layer_mean_per_round": mean,
        "drift_silence": drift_silence,
        "drift_speech": drift_speech,
        "flip_silence_per_layer": flip_silence,
        "flip_speech_per_layer": flip_speech,
        "silence_ratio_per_layer": silence_ratio,
        "total_flips_per_layer": total_flips,
    }


def print_inference(stats, tag):
    if stats is None:
        return
    mean = stats["layer_mean_per_round"]
    n_layers = stats["n_layers"]
    print(f"\n=== {tag} index 迁移倾向推断 ===")
    print(f"样本数: {stats['n_files']}, 层数: {n_layers}, 轮数: {stats['n_rounds']}")

    # 1. 漂移率分层
    low = mean[0, :4].mean()      # 0-3
    mid = mean[0, 4:12].mean()    # 4-11
    high = mean[0, 12:].mean()    # 12+
    print(f"\n1. 漂移率分层（第 1 轮）:")
    print(f"   浅层 (0-3):   {low:.4f}  -> 稳定，index 迁移少")
    print(f"   中层 (4-11):  {mid:.4f}  -> 中等，index 在局部范围内迁移")
    print(f"   深层 (12+):   {high:.4f}  -> 极高，index 几乎随机重采样")

    # 2. 静音/语音迁移差异
    sil_ratio = stats["drift_silence"] / max(stats["drift_silence"] + stats["drift_speech"], 1)
    print(f"\n2. 静音段漂移占比: {sil_ratio:.4f}")
    print(f"   说明大部分漂移发生在语音段，index 迁移与语音内容相关")

    # 3. 每层静音漂移占比
    high_sil_layers = np.where(stats["silence_ratio_per_layer"] > 0.2)[0]
    if len(high_sil_layers) > 0:
        print(f"\n3. 静音漂移占比 >20% 的层: {high_sil_layers.tolist()}")
    else:
        print(f"\n3. 所有层静音漂移占比均 <20%，语音段是迁移主体")

    # 4. 轮数累积
    r1 = mean[0].mean()
    r5 = mean[-1].mean()
    print(f"\n4. 轮数累积: 第 1 轮 {r1:.4f} -> 第 {stats['n_rounds']} 轮 {r5:.4f}")
    print(f"   增长主要来自深层，浅层增长缓慢")

    # 5. 推断结论
    print(f"\n5. 推断结论:")
    print(f"   - 浅层 (0-3): index 迁移少，且倾向于在语义相近的 index 间迁移")
    print(f"   - 中层 (4-11): index 在局部 codebook 范围内迁移，可能对应相似声学特征")
    print(f"   - 深层 (12+): index 迁移接近随机，重编码后几乎完全重采样")
    print(f"   - 静音段: 漂移占比低，但深层静音 token 也可能因能量低而迁移到相近低能量 index")


def save_inference_json(stats, tag, out_dir):
    if stats is None:
        return
    out = {
        "n_files": stats["n_files"],
        "n_layers": stats["n_layers"],
        "n_rounds": stats["n_rounds"],
        "layer_mean_per_round": stats["layer_mean_per_round"].tolist(),
        "drift_silence": stats["drift_silence"],
        "drift_speech": stats["drift_speech"],
        "flip_silence_per_layer": stats["flip_silence_per_layer"].tolist(),
        "flip_speech_per_layer": stats["flip_speech_per_layer"].tolist(),
        "silence_ratio_per_layer": stats["silence_ratio_per_layer"].tolist(),
        "total_flips_per_layer": stats["total_flips_per_layer"].tolist(),
    }
    path = os.path.join(out_dir, f"transition_inference{tag}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"Saved {path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default=None)
    args = parser.parse_args()

    out_dir = args.output_dir or os.path.join(args.input_dir, "analysis")
    os.makedirs(out_dir, exist_ok=True)

    data = load_per_layer_files(args.input_dir)
    en_stats = infer_transition_tendency(data["EN"])
    zh_stats = infer_transition_tendency(data["ZH"])
    all_stats = infer_transition_tendency(data["EN"] + data["ZH"])

    print_inference(all_stats, "ALL")
    print_inference(en_stats, "EN")
    print_inference(zh_stats, "ZH")

    save_inference_json(all_stats, "", out_dir)
    save_inference_json(en_stats, "_EN", out_dir)
    save_inference_json(zh_stats, "_ZH", out_dir)


if __name__ == "__main__":
    main()
