"""Sprint 0.2：多层组合 Pareto（两阶段 + 8 进程）。

阶段 1（先跑）连续筛选：起点 4/6/8，右端到 8/10/12/14/16，全是连续段。
阶段 2（连续段选定后再跑）：只以 **`4-14`** 为底做全量 drop（`6-14` 与其高度重合，不另跑）。
去掉 1/2/3 个中间层（保留两端 L4、L14）。例如底 4–14 去掉 L5 → 4,6-14。

嵌入与检测用同一层集合。跨层时中间未选层不改，也不进 V1 high-only。

落盘只写每种组合的均值（与单层 `layer_mean` 同口径：cosine / corr / mse / LSD + 各通道 orig/wm/gap），
不写逐条语音、不写 per_clip。

例：
  python -m watermark.residual.experiments.run_layer_pareto --phase contig --n-files 200 --num-workers 8
  python -m watermark.residual.experiments.run_layer_pareto --phase drop --base 4-14 --drop-k 1,2,3 --n-files 1500 --num-workers 8
"""

from __future__ import annotations

import argparse
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from itertools import combinations
from pathlib import Path

import numpy as np

from watermark.future_exp._common import (
    EXP_ROOT,
    add_model_args,
    dump_json,
    embed_wav,
    list_audio,
    load_model,
    prepare_run,
    read_wav,
)
from watermark.future_exp._multi_channel import score_wav_channels
from watermark.isolate import match_wave
from watermark.moss_io import SEED, set_seed
from watermark.plan import combo_label
from watermark.rare_tokens import DEFAULT_CLUSTER_DIR, ensure_rare_table

DETECT_KEYS = (
    "wave",
    "residual",
    "residual_pair",
    "aba_embed",
    "centroid_member",
    "centroid_kmeans",
    "slide_v3",
)
# 与 run_layer_sim_sweep 日志 / README 一致
DETECT_LABELS = {
    "wave": "V1 wave",
    "residual": "residual wav",
    "residual_pair": "V2 residual_pair",
    "aba_embed": "V3 aba_embed",
    "centroid_member": "centroid_member",
    "centroid_kmeans": "centroid_kmeans",
    "slide_v3": "slide_v3",
}
GAP_LOG_KEYS = {
    "wave": "gapW",
    "residual": "gapR",
    "residual_pair": "gapP",
    "aba_embed": "gapA",
    "centroid_member": "gapM",
    "centroid_kmeans": "gapK",
    "slide_v3": "gapS",
}

# 单层结论：主嵌入不超过 L19；0.2 连续筛选做到 16。
LAYER_POOL = (4, 16)
CONTIG_STARTS = (4, 6, 8)
CONTIG_ENDS = (8, 10, 12, 14, 16)


def _limit_cpu_threads(n: int = 1) -> None:
    n = max(1, int(n))
    for key in (
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
        "BLIS_NUM_THREADS",
        "TORCH_NUM_THREADS",
    ):
        os.environ[key] = str(n)
    try:
        import torch

        torch.set_num_threads(n)
        torch.set_num_interop_threads(1)
    except Exception:
        pass


def contig_combos(
    starts=CONTIG_STARTS, ends=CONTIG_ENDS, pool=LAYER_POOL
) -> list[list[int]]:
    """4–16 内的连续段：从 4/6/8 收到 8/10/12/14/16。"""
    lo, hi = int(pool[0]), int(pool[1])
    out = []
    for s in starts:
        for e in ends:
            if lo <= int(s) < int(e) <= hi:
                out.append(list(range(int(s), int(e) + 1)))
    return _dedupe(out)


def drop_from_base(base: list[int], ks, keep_ends: bool = True) -> list[list[int]]:
    """从连续底集去掉 k 个中间层。默认保留两端，只删中间。含 drop-0（原底集）。"""
    base = sorted({int(x) for x in base})
    if len(base) < 3:
        return [base] if base else []
    ends = {base[0], base[-1]} if keep_ends else set()
    mid = [x for x in base if x not in ends]
    out = [list(base)]
    for k in ks:
        k = int(k)
        if k <= 0 or k > len(mid):
            continue
        for dropped in combinations(mid, k):
            keep = [x for x in base if x not in dropped]
            if keep:
                out.append(keep)
    return _dedupe(out)


def parse_drop_k(spec: str) -> list[int]:
    ks = []
    for part in (spec or "").split(","):
        part = part.strip()
        if not part:
            continue
        ks.append(int(part))
    return sorted({k for k in ks if k > 0})


def parse_combos(spec: str) -> list[list[int]]:
    """`4-16;4,6-16` → 层列表。也兼容旧写法 `6-16,8-16`。"""
    text = (spec or "").strip()
    if not text:
        return []
    if ";" in text:
        parts = [p.strip() for p in text.split(";")]
    elif any(ch.isalpha() for ch in text):
        raise SystemExit("combo spec must be like 4-16;4,6,8,10,12,14,16")
    elif "," in text and "-" in text and not _looks_like_skip_list(text):
        parts = [p.strip() for p in text.split(",")]
    else:
        parts = [text]
    out = []
    for part in parts:
        if not part:
            continue
        out.append(_parse_one_combo(part))
    return _dedupe(out)


def _looks_like_skip_list(text: str) -> bool:
    chunks = [c.strip() for c in text.split(",") if c.strip()]
    return bool(chunks) and all("-" not in c and c.lstrip("-").isdigit() for c in chunks)


def _parse_one_combo(part: str) -> list[int]:
    part = part.strip()
    if "-" in part and "," not in part:
        a, b = part.split("-", 1)
        lo, hi = int(a), int(b)
        if hi < lo:
            raise SystemExit(f"bad range {part}")
        return list(range(lo, hi + 1))
    layers = [int(x.strip()) for x in part.split(",") if x.strip()]
    if not layers:
        raise SystemExit(f"empty combo {part!r}")
    return sorted(set(layers))


def _dedupe(combos: list[list[int]]) -> list[list[int]]:
    seen = set()
    out = []
    for raw in combos:
        key = tuple(sorted({int(x) for x in raw}))
        if not key or key in seen:
            continue
        seen.add(key)
        out.append(list(key))
    return out


def _clip_to_pool(layers: list[int], lo: int, hi: int) -> list[int]:
    return [int(x) for x in layers if lo <= int(x) <= hi]


def _parse_npu_ids(s: str) -> list[int]:
    s = (s or "").strip()
    if not s:
        return list(range(8))
    return [int(x.strip()) for x in s.split(",") if x.strip() != ""]


def _auto_device_single() -> str:
    import torch

    if torch.cuda.is_available():
        return "cuda"
    try:
        import torch_npu  # noqa: F401

        if hasattr(torch, "npu") and torch.npu.is_available():
            return "npu"
    except Exception:
        pass
    return "cpu"


def _corpus_tag(stem: str) -> str:
    if stem.startswith("aishell_"):
        return "aishell"
    if stem.startswith("sub_ZH_") or stem.startswith("ZH_"):
        return "zh"
    if stem.startswith("sub_EN_") or stem.startswith("EN_"):
        return "en"
    return "other"


def lsd_of(ref, deg, sr) -> float | None:
    try:
        from eval.experiments.calcu_lsd import CalcuLSD
    except Exception:
        return None
    val, _ = CalcuLSD(sr=int(sr)).compute(ref, deg)
    return float(val)


def _channel_scores(block: dict) -> dict:
    scores = dict(block.get("scores") or {})
    scores["residual"] = float((block.get("sim_residual") or {}).get("cosine") or 0.0)
    if "wave" not in scores:
        scores["wave"] = float((block.get("sim_high_only") or {}).get("cosine") or 0.0)
    return {k: float(scores.get(k) or 0.0) for k in DETECT_KEYS}


def _mean(xs):
    xs = [float(x) for x in xs if x is not None]
    return float(np.mean(xs)) if xs else None


def _fmt_gaps(gaps: dict) -> str:
    return " ".join(
        f"{GAP_LOG_KEYS[k]}={float(gaps[k]):+.3f}" for k in DETECT_KEYS if k in gaps
    )


def _gap_aliases_from_means(item: dict) -> dict:
    """combo_mean 里增加 gapW/gapR/... 别名，便于对照单层扫日志。"""
    return {
        GAP_LOG_KEYS[k]: float(item.get(f"mean_gap_{k}") or 0.0) for k in DETECT_KEYS
    }


def _eval_one_combo(model, wav, sr, table, layers, cluster_dir, period, slide_win) -> dict:
    hs, he = int(layers[0]), int(layers[-1])
    emb = embed_wav(model, wav, table, hs, he, layers=layers)
    ovw = match_wave(emb["wav"], wav, sr)
    orig_blk = score_wav_channels(
        model,
        wav,
        sr,
        table,
        hs,
        he,
        cluster_dir=cluster_dir,
        period=int(period),
        slide_win=int(slide_win),
        layers=layers,
    )
    wm_blk = score_wav_channels(
        model,
        emb["wav"],
        sr,
        table,
        hs,
        he,
        cluster_dir=cluster_dir,
        period=int(period),
        slide_win=int(slide_win),
        layers=layers,
    )
    orig_s = _channel_scores(orig_blk)
    wm_s = _channel_scores(wm_blk)
    return {
        "layers": list(layers),
        "n_layers": len(layers),
        "n_changed": int(emb["n_changed"]),
        "orig_vs_wm": ovw,
        "lsd": lsd_of(wav, emb["wav"], sr),
        "orig_scores": orig_s,
        "wm_scores": wm_s,
        "gaps": {k: float(wm_s[k] - orig_s[k]) for k in DETECT_KEYS},
    }


def _worker_shard(payload: dict) -> dict:
    worker_id = int(payload["worker_id"])
    device = str(payload["device"])
    wav_paths = [Path(p) for p in payload["wav_paths"]]
    combos = [list(c) for c in payload["combos"]]
    cpu_threads = int(payload.get("cpu_threads") or 1)
    cluster_dir = str(payload["cluster_dir"])
    period = int(payload["period"])
    slide_win = int(payload["slide_win"])
    model_path = str(payload["model"])
    table_path = payload.get("table") or None
    no_seed = bool(payload["no_seed"])
    seed = payload.get("seed")

    _limit_cpu_threads(cpu_threads)
    if no_seed:
        os.environ["WATERMARK_NO_SEED"] = "1"
        os.environ.pop("PYTHONHASHSEED", None)
    else:
        set_seed(int(seed) if seed is not None else SEED)

    if device.startswith("npu"):
        import torch
        import torch_npu  # noqa: F401

        if ":" in device:
            idx = int(device.split(":", 1)[1])
            torch.npu.set_device(idx)

    table = ensure_rare_table(path=table_path) if table_path else ensure_rare_table()
    print(f"[w{worker_id}] device={device} n_files={len(wav_paths)} n_combos={len(combos)}", flush=True)
    model, sr = load_model(model_path, device)
    print(f"[w{worker_id}] sr={sr} ready", flush=True)

    rows = []
    for ap in wav_paths:
        name = ap.stem
        print(f"[w{worker_id}] === {name} ({_corpus_tag(name)})", flush=True)
        wav = read_wav(ap, sr)
        for layers in combos:
            rec = _eval_one_combo(
                model, wav, sr, table, layers, cluster_dir, period, slide_win
            )
            rec["name"] = name
            rec["corpus"] = _corpus_tag(name)
            rec["label"] = combo_label(layers)
            rows.append(rec)
            g = rec["gaps"]
            print(
                f"[w{worker_id}]   {name} {rec['label']:<22} "
                f"cos={rec['orig_vs_wm']['cosine']:.4f} "
                f"lsd={rec['lsd']} {_fmt_gaps(g)}",
                flush=True,
            )
    return {
        "worker_id": worker_id,
        "device": device,
        "n_files": len(wav_paths),
        "rows": rows,
    }


CORPUS_NAMES = ("aishell", "zh", "en")


def _summarize(rows: list[dict], corpus: str | None = None) -> list[dict]:
    if corpus is not None:
        rows = [r for r in rows if r.get("corpus") == corpus]
    by = {}
    for rec in rows:
        key = tuple(rec["layers"])
        by.setdefault(key, []).append(rec)
    out = []
    for layers, recs in by.items():
        layers = list(layers)
        item = {
            "label": combo_label(layers),
            "layers": layers,
            "n_layers": len(layers),
            "contiguous": layers == list(range(layers[0], layers[-1] + 1)),
            "high_start": int(layers[0]),
            "high_end": int(layers[-1]),
            "n_clips": len(recs),
            "mean_orig_vs_wm_cosine": _mean(r["orig_vs_wm"]["cosine"] for r in recs),
            "mean_orig_vs_wm_corr": _mean(r["orig_vs_wm"].get("corr") for r in recs),
            "mean_mse": _mean(r["orig_vs_wm"].get("mse") for r in recs),
            "mean_lsd": _mean(r.get("lsd") for r in recs),
            "mean_orig_high": _mean(r["orig_scores"]["wave"] for r in recs),
            "mean_wm_high": _mean(r["wm_scores"]["wave"] for r in recs),
            "mean_high_gap": _mean(r["gaps"]["wave"] for r in recs),
        }
        if corpus is not None:
            item["corpus"] = corpus
        for key in DETECT_KEYS:
            item[f"mean_gap_{key}"] = _mean(r["gaps"][key] for r in recs)
            item[f"mean_orig_{key}"] = _mean(r["orig_scores"][key] for r in recs)
            item[f"mean_wm_{key}"] = _mean(r["wm_scores"][key] for r in recs)
        item["gaps"] = _gap_aliases_from_means(item)
        out.append(item)
    out.sort(key=lambda x: (x["high_start"], x["high_end"], x["n_layers"], x["label"]))
    return out


def _summarize_all_and_by_corpus(rows: list[dict]) -> tuple[list[dict], dict[str, list[dict]]]:
    """全库综合 + aishell / zh / en 分库均值。"""
    combo_mean = _summarize(rows)
    by_corpus = {corp: _summarize(rows, corpus=corp) for corp in CORPUS_NAMES}
    return combo_mean, by_corpus


def _print_combo_table(title: str, combo_mean: list[dict]) -> None:
    print(f"\n=== {title} ===")
    print(
        "combo                  nL  n   cos    lsd     "
        "gapW   gapR   gapP   gapA   gapM   gapK   gapS"
    )
    for r in combo_mean:
        lsd_s = f"{r['mean_lsd']:.3f}" if r.get("mean_lsd") is not None else "  n/a"
        g = r.get("gaps") or _gap_aliases_from_means(r)
        print(
            f"{r['label']:<22} {r['n_layers']:>2} {r['n_clips']:>3}  "
            f"{(r['mean_orig_vs_wm_cosine'] or 0):.4f}  {lsd_s:>6}  "
            f"{g['gapW']:+.3f}  {g['gapR']:+.3f}  {g['gapP']:+.3f}  "
            f"{g['gapA']:+.3f}  {g['gapM']:+.3f}  {g['gapK']:+.3f}  {g['gapS']:+.3f}"
        )


def main():
    p = argparse.ArgumentParser(description="多层组合听感 vs 全通道检出 Pareto")
    add_model_args(p, n_files=1500)
    p.add_argument(
        "--phase",
        type=str,
        default="contig",
        choices=["contig", "drop"],
        help="contig=先连续筛选；drop=在 --base 连续段上去掉中间层",
    )
    p.add_argument(
        "--base",
        type=str,
        default="",
        help="阶段 2 的连续底集，如 4-14。contig 后默认 4-14",
    )
    p.add_argument(
        "--drop-k",
        type=str,
        default="1,2,3",
        help="阶段 2：去掉几个中间层，逗号分隔，如 1 或 1,2",
    )
    p.add_argument(
        "--keep-ends",
        action="store_true",
        default=True,
        help="去掉中间层时保留底集两端（默认开）",
    )
    p.add_argument("--no-keep-ends", action="store_true", help="去掉中间层时也允许去掉两端")
    p.add_argument(
        "--combos",
        type=str,
        default="",
        help="手工组合，分号分隔；与 --combos-only 联用可只跑指定组合",
    )
    p.add_argument(
        "--combos-only",
        action="store_true",
        help="drop 阶段只跑 --combos / --ranges，不做全量 drop-k 枚举",
    )
    p.add_argument("--ranges", type=str, default="", help="兼容旧参数：6-16,8-16")
    p.add_argument("--pool-start", type=int, default=LAYER_POOL[0])
    p.add_argument("--pool-end", type=int, default=LAYER_POOL[1])
    p.add_argument(
        "--num-workers",
        type=int,
        default=8,
        help="并行进程数；1=单卡。默认 8",
    )
    p.add_argument(
        "--npu-ids",
        type=str,
        default="0,1,2,3,4,5,6,7",
        help="NPU 卡号，逗号分隔，与 workers 按序配对",
    )
    p.add_argument("--cpu-threads", type=int, default=1)
    p.add_argument("--cluster-dir", type=str, default=str(DEFAULT_CLUSTER_DIR))
    p.add_argument("--period", type=int, default=3)
    p.add_argument("--slide-win", type=int, default=18)
    args = p.parse_args()

    if not args.out_dir:
        if args.phase == "drop" and args.base.strip():
            stem = f"layer_pareto_drop_{args.base.strip().replace(',', '_')}"
        else:
            stem = f"layer_pareto_{args.phase}"
        args.out_dir = str(EXP_ROOT.parent / stem)
    cfg = prepare_run(args, f"layer_pareto_{args.phase}")
    wavs = list_audio(args.wav_dir, args.n_files)
    if not wavs:
        raise SystemExit(f"no audio in {args.wav_dir}")

    keep_ends = not bool(args.no_keep_ends)
    lo, hi = int(args.pool_start), int(args.pool_end)
    combos: list[list[int]] = []
    drop_ks: list[int] = []
    base_layers: list[int] = []

    if args.phase == "contig":
        combos.extend(contig_combos(pool=(lo, hi)))
    else:
        if not args.base.strip():
            raise SystemExit("--phase drop 需要 --base，例如 --base 4-14（contig 后选定）")
        base_layers = _parse_one_combo(args.base.strip())
        if args.combos_only:
            if not args.combos.strip() and not args.ranges.strip():
                raise SystemExit("--combos-only 需要 --combos 或 --ranges 指定组合")
        else:
            drop_ks = parse_drop_k(args.drop_k)
            if not drop_ks:
                raise SystemExit("--drop-k 至少要有一个正整数，例如 1 或 1,2")
            combos.extend(drop_from_base(base_layers, drop_ks, keep_ends=keep_ends))

    if args.combos.strip():
        combos.extend(parse_combos(args.combos))
    if args.ranges.strip():
        spec = args.ranges
        if ";" not in spec:
            spec = spec.replace(",", ";")
        combos.extend(parse_combos(spec))

    clipped = []
    for c in combos:
        keep = _clip_to_pool(c, lo, hi)
        if keep:
            clipped.append(keep)
    combos = _dedupe(clipped)
    if not combos:
        raise SystemExit("no combos after pool clip")

    labels = [combo_label(c) for c in combos]
    by_corp = {"aishell": 0, "zh": 0, "en": 0, "other": 0}
    for ap in wavs:
        by_corp[_corpus_tag(ap.stem)] += 1

    num_workers = max(1, int(args.num_workers))
    cpu_threads = max(1, int(args.cpu_threads))
    npu_ids = _parse_npu_ids(args.npu_ids)
    if num_workers > 1 and len(npu_ids) < num_workers:
        raise SystemExit(
            f"--num-workers={num_workers} 需要至少 {num_workers} 个 --npu-ids，当前 {npu_ids}"
        )

    os.environ["WATERMARK_CPU_THREADS"] = str(cpu_threads)
    _limit_cpu_threads(cpu_threads)
    if args.no_seed:
        seed_label = None
    else:
        set_seed(SEED)
        seed_label = SEED

    out_dir = cfg["out_dir"]

    print(
        f"phase={args.phase} pool L{lo}-L{hi} n_combos={len(combos)} "
        f"n_files={len(wavs)} num_workers={num_workers}"
    )
    if args.phase == "drop":
        print(
            f"base={combo_label(base_layers)} drop_k={drop_ks} keep_ends={keep_ends} "
            f"(跨层=连续底集去掉中间层)"
        )
    print(
        f"corpus aishell={by_corp['aishell']} zh={by_corp['zh']} "
        f"en={by_corp['en']} other={by_corp['other']}"
    )
    for lab in labels:
        print("  combo", lab)

    ensure_rare_table(path=args.table)
    payloads = []
    if num_workers == 1:
        device = (args.device or "").strip() or _auto_device_single()
        print("device", device)
        shards = [wavs]
        devices = [device]
    else:
        shards = [[] for _ in range(num_workers)]
        for i, ap in enumerate(wavs):
            shards[i % num_workers].append(ap)
        devices = [f"npu:{int(npu_ids[w])}" for w in range(num_workers)]
        for wid, (dev, shard) in enumerate(zip(devices, shards)):
            print(f"worker {wid} -> {dev} files={len(shard)}")

    for wid, (dev, shard) in enumerate(zip(devices, shards)):
        payloads.append(
            {
                "worker_id": wid,
                "device": dev,
                "wav_paths": [str(p) for p in shard],
                "combos": combos,
                "model": args.model_path,
                "table": args.table,
                "no_seed": bool(args.no_seed),
                "seed": seed_label,
                "cpu_threads": cpu_threads,
                "cluster_dir": str(args.cluster_dir),
                "period": int(args.period),
                "slide_win": int(args.slide_win),
            }
        )

    clip_rows: list[dict] = []
    devices_used: list[str] = []
    if num_workers == 1:
        out = _worker_shard(payloads[0])
        clip_rows.extend(out["rows"])
        devices_used.append(out["device"])
    else:
        with ProcessPoolExecutor(max_workers=num_workers) as ex:
            futs = {ex.submit(_worker_shard, pl): pl["worker_id"] for pl in payloads}
            for fut in as_completed(futs):
                wid = futs[fut]
                try:
                    out = fut.result()
                except Exception as exc:
                    raise SystemExit(f"worker {wid} failed: {exc}") from exc
                print(
                    f"[main] worker {wid} done files={out['n_files']} rows={len(out['rows'])}",
                    flush=True,
                )
                clip_rows.extend(out["rows"])
                devices_used.append(out["device"])

    combo_mean, by_corpus = _summarize_all_and_by_corpus(clip_rows)
    del clip_rows
    combo_mean_payload = {
        "n_files": len(wavs),
        "corpus_counts": by_corp,
        "n_combos": len(combos),
        "combo_mean": combo_mean,
        "by_corpus": by_corpus,
    }
    summary = {
        "experiment": "layer_pareto",
        "phase": args.phase,
        "protocol": (
            "phase1=contiguous 4/6/8..16; "
            "phase2=drop k intermediate layers from a chosen contiguous base; "
            "embed and detect use the same layer set; "
            "metrics match layer_mean: cosine/corr/mse/LSD + per-channel orig/wm/gap; "
            "report all + aishell/zh/en corpus means"
        ),
        "note": "Only combo averages are saved; no per-clip rows.",
        "base": combo_label(base_layers) if base_layers else None,
        "drop_k": drop_ks,
        "keep_ends": keep_ends,
        "pool": [lo, hi],
        "detect_keys": list(DETECT_KEYS),
        "detect_labels": DETECT_LABELS,
        "gap_log_keys": GAP_LOG_KEYS,
        "combos": labels,
        "n_files": len(wavs),
        "corpus_counts": by_corp,
        "num_workers": num_workers,
        "device": devices_used[0] if len(devices_used) == 1 else devices_used,
        "seed": seed_label,
        "combo_mean": combo_mean,
        "by_corpus": by_corpus,
    }
    dump_json(out_dir / "summary.json", summary)
    dump_json(out_dir / "combo_mean.json", combo_mean)
    dump_json(out_dir / "combo_mean_by_corpus.json", combo_mean_payload)
    print("saved", out_dir / "summary.json")
    print("saved", out_dir / "combo_mean.json")
    print("saved", out_dir / "combo_mean_by_corpus.json")
    _print_combo_table("combo means · all", combo_mean)
    for corp in CORPUS_NAMES:
        _print_combo_table(f"combo means · {corp}", by_corpus[corp])


if __name__ == "__main__":
    import multiprocessing as mp

    try:
        mp.set_start_method("spawn", force=True)
    except RuntimeError:
        pass
    main()
