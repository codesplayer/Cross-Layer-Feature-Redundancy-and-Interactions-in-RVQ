"""Band 多层组合（contig / drop / 手工 combos）。

默认特征-only（aba/质心/slide）。`--full-channels` 时打全通道 + LSD + orig↔wm cos/corr/mse：
  wave / residual / residual_pair / aba_embed / centroid_* / slide_v3

稀疏写入时 band 查询只累加写入层 z_q（与 ABA 模板对齐）。

例（跨 B/C/D 峰层，全通道）：
  python -m watermark.band.experiments.run_layer_pareto_band \\
    --phase drop --combos-only --base 9 \\
    --combos \"4,9;9,18;4,18;4,9,18\" \\
    --full-channels --n-per-corpus 500 --num-workers 1 --device cuda \\
    --pool-start 4 --pool-end 28 \\
    --out_dir D:/data/codec_an/exp/band/layer_cross_BCD_peaks_full_n1500
"""

from __future__ import annotations

import argparse
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

from watermark.future_exp._common import (
    CORPUS_NAMES,
    add_model_args,
    auto_device,
    corpus_tag,
    dump_json,
    embed_wav,
    list_audio,
    list_audio_per_corpus,
    load_model,
    prepare_run,
    read_wav,
)
from watermark.future_exp._multi_channel import FEAT_KEYS, score_feat_band, score_wav_channels
from watermark.isolate import match_wave
from watermark.moss_io import SEED, set_seed
from watermark.plan import combo_label
from watermark.rare_tokens import DEFAULT_CLUSTER_DIR, ensure_rare_table
from watermark.residual.experiments.run_layer_pareto import (
    _clip_to_pool,
    _dedupe,
    _parse_one_combo,
    contig_combos,
    drop_from_base,
    lsd_of,
    parse_combos,
    parse_drop_k,
)

DETECT_KEYS = (
    "wave",
    "residual",
    "residual_pair",
    "aba_embed",
    "centroid_member",
    "centroid_kmeans",
    "slide_v3",
)
GAP_LOG_FULL = (
    ("gapW", "wave"),
    ("gapR", "residual"),
    ("gapP", "residual_pair"),
    ("gapA", "aba_embed"),
    ("gapM", "centroid_member"),
    ("gapK", "centroid_kmeans"),
    ("gapS", "slide_v3"),
)
GAP_LOG_FEAT = (
    ("gapA", "aba_embed"),
    ("gapM", "centroid_member"),
    ("gapK", "centroid_kmeans"),
    ("gapS", "slide_v3"),
)

DEFAULT_POOL = (4, 28)
DEFAULT_STARTS = "4,6,8,10"
DEFAULT_ENDS = "8,10,12,14,16,20,24,28"


def parse_int_csv(spec: str) -> list[int]:
    out = []
    for part in (spec or "").split(","):
        part = part.strip()
        if not part:
            continue
        out.append(int(part))
    return sorted(set(out))


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


def _parse_npu_ids(s: str) -> list[int]:
    return [int(x) for x in str(s).split(",") if x.strip() != ""]


def _mean(xs) -> float | None:
    vals = [float(x) for x in xs if x is not None]
    if not vals:
        return None
    return float(np.mean(vals))


def _channel_scores(block: dict) -> dict:
    scores = dict(block.get("scores") or {})
    scores["residual"] = float((block.get("sim_residual") or {}).get("cosine") or 0.0)
    if "wave" not in scores:
        scores["wave"] = float((block.get("sim_high_only") or {}).get("cosine") or 0.0)
    return {k: float(scores.get(k) or 0.0) for k in DETECT_KEYS}


def _fmt_gaps(gap: dict, full: bool) -> str:
    keys = GAP_LOG_FULL if full else GAP_LOG_FEAT
    return " ".join(f"{lab}={float(gap[k]):+.3f}" for lab, k in keys)


def _plain_feat(d: dict) -> dict:
    return {k: float(d[k]) for k in FEAT_KEYS}


def _eval_feat_only(model, wav, table, layers, cluster_dir, period, slide_win) -> dict:
    layers = [int(x) for x in layers]
    hs, he = int(layers[0]), int(layers[-1])
    emb = embed_wav(model, wav, table, hs, he, layers=layers, period=int(period))
    orig = score_feat_band(
        model, wav, table, layers=layers, cluster_dir=cluster_dir, period=period, slide_win=slide_win
    )
    wm = score_feat_band(
        model,
        emb["wav"],
        table,
        layers=layers,
        cluster_dir=cluster_dir,
        period=period,
        slide_win=slide_win,
    )
    orig_s, wm_s = _plain_feat(orig), _plain_feat(wm)
    return {
        "layers": list(layers),
        "n_layers": len(layers),
        "n_changed": int(emb["n_changed"]),
        "space": "band",
        "full_channels": False,
        "orig_scores": orig_s,
        "wm_scores": wm_s,
        "gaps": {k: float(wm_s[k] - orig_s[k]) for k in FEAT_KEYS},
    }


def _eval_full(model, wav, sr, table, layers, cluster_dir, period, slide_win) -> dict:
    layers = [int(x) for x in layers]
    hs, he = int(layers[0]), int(layers[-1])
    emb = embed_wav(model, wav, table, hs, he, layers=layers, period=int(period))
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
        query_space="band",
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
        query_space="band",
    )
    orig_s = _channel_scores(orig_blk)
    wm_s = _channel_scores(wm_blk)
    return {
        "layers": list(layers),
        "n_layers": len(layers),
        "n_changed": int(emb["n_changed"]),
        "space": "band",
        "full_channels": True,
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
    full = bool(payload.get("full_channels"))
    _limit_cpu_threads(int(payload.get("cpu_threads") or 1))
    if payload.get("no_seed"):
        os.environ["WATERMARK_NO_SEED"] = "1"
        os.environ.pop("PYTHONHASHSEED", None)
    else:
        set_seed(int(payload.get("seed") or SEED))

    if device.startswith("npu"):
        import torch
        import torch_npu  # noqa: F401

        if ":" in device:
            torch.npu.set_device(int(device.split(":", 1)[1]))

    table = ensure_rare_table(path=payload.get("table") or None)
    mode = "full(W/R/P/A/M/K/S+LSD+cos)" if full else "feat-only"
    print(
        f"[w{worker_id}] band {mode} device={device} n_files={len(wav_paths)} "
        f"n_combos={len(combos)}",
        flush=True,
    )
    model, sr = load_model(str(payload["model"]), device)
    print(f"[w{worker_id}] sr={sr} ready", flush=True)

    rows = []
    n_files = len(wav_paths)
    cluster_dir = str(payload["cluster_dir"])
    period = int(payload["period"])
    slide_win = int(payload["slide_win"])
    for i, ap in enumerate(wav_paths, 1):
        wav = read_wav(ap, sr)
        corp = corpus_tag(ap.stem)
        print(f"[w{worker_id}] {i}/{n_files} {corp} {ap.stem}", flush=True)
        for layers in combos:
            if full:
                rec = _eval_full(
                    model, wav, sr, table, layers, cluster_dir, period, slide_win
                )
            else:
                rec = _eval_feat_only(
                    model, wav, table, layers, cluster_dir, period, slide_win
                )
            rec["name"] = ap.stem
            rec["corpus"] = corp
            rec["label"] = combo_label(layers)
            rows.append(rec)
            extra = ""
            if full and rec.get("orig_vs_wm"):
                cos = float(rec["orig_vs_wm"].get("cosine") or 0.0)
                lsd = rec.get("lsd")
                lsd_s = f"{float(lsd):.3f}" if lsd is not None else "n/a"
                extra = f" cos={cos:.4f} lsd={lsd_s}"
            print(
                f"[w{worker_id}]   {ap.stem} {rec['label']:<22} "
                f"{_fmt_gaps(rec['gaps'], full)}{extra}",
                flush=True,
            )
    return {"worker_id": worker_id, "device": device, "n_files": n_files, "rows": rows}


def summarize(rows: list[dict], corpus: str | None = None, full: bool = False) -> list[dict]:
    if corpus is not None:
        rows = [r for r in rows if r.get("corpus") == corpus]
    by: dict[tuple, list] = {}
    for rec in rows:
        by.setdefault(tuple(rec["layers"]), []).append(rec)
    gap_keys = DETECT_KEYS if full else FEAT_KEYS
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
            "space": "band",
            "full_channels": bool(full),
        }
        if corpus is not None:
            item["corpus"] = corpus
        if full:
            item["mean_orig_vs_wm_cosine"] = _mean(
                (r.get("orig_vs_wm") or {}).get("cosine") for r in recs
            )
            item["mean_orig_vs_wm_corr"] = _mean(
                (r.get("orig_vs_wm") or {}).get("corr") for r in recs
            )
            item["mean_mse"] = _mean((r.get("orig_vs_wm") or {}).get("mse") for r in recs)
            item["mean_lsd"] = _mean(r.get("lsd") for r in recs)
        for key in gap_keys:
            item[f"mean_gap_{key}"] = _mean(r["gaps"][key] for r in recs)
            item[f"mean_orig_{key}"] = _mean(r["orig_scores"][key] for r in recs)
            item[f"mean_wm_{key}"] = _mean(r["wm_scores"][key] for r in recs)
        log = GAP_LOG_FULL if full else GAP_LOG_FEAT
        item["gaps"] = {lab: float(item[f"mean_gap_{key}"] or 0.0) for lab, key in log}
        out.append(item)
    out.sort(key=lambda x: (x["high_start"], x["high_end"], x["n_layers"], x["label"]))
    return out


def _print_combo_table(title: str, combo_mean: list[dict], full: bool) -> None:
    print(f"\n=== {title} ===")
    if full:
        print(
            "combo                  nL  n   cos    lsd     "
            "gapW   gapR   gapP   gapA   gapM   gapK   gapS"
        )
    else:
        print("combo                  nL  n    gapA   gapM   gapK   gapS")
    for r in combo_mean:
        g = r.get("gaps") or {}
        if full:
            lsd_s = f"{r['mean_lsd']:.3f}" if r.get("mean_lsd") is not None else "  n/a"
            print(
                f"{r['label']:<22} {r['n_layers']:>2} {r['n_clips']:>3}  "
                f"{(r.get('mean_orig_vs_wm_cosine') or 0):.4f}  {lsd_s:>6}  "
                f"{g.get('gapW', 0):+.3f}  {g.get('gapR', 0):+.3f}  {g.get('gapP', 0):+.3f}  "
                f"{g.get('gapA', 0):+.3f}  {g.get('gapM', 0):+.3f}  "
                f"{g.get('gapK', 0):+.3f}  {g.get('gapS', 0):+.3f}"
            )
        else:
            print(
                f"{r['label']:<22} {r['n_layers']:>2} {r['n_clips']:>3}  "
                f"{g.get('gapA', 0):+.3f}  {g.get('gapM', 0):+.3f}  "
                f"{g.get('gapK', 0):+.3f}  {g.get('gapS', 0):+.3f}"
            )


def _build_combos(args) -> tuple[list[list[int]], list[int], list[int]]:
    lo, hi = int(args.pool_start), int(args.pool_end)
    if hi < lo:
        raise SystemExit(f"bad pool L{lo}-L{hi}")
    combos: list[list[int]] = []
    drop_ks: list[int] = []
    base_layers: list[int] = []
    starts = parse_int_csv(args.contig_starts)
    ends = parse_int_csv(args.contig_ends)
    if args.phase == "contig":
        if not starts or not ends:
            raise SystemExit("--contig-starts 和 --contig-ends 不能为空")
        combos.extend(contig_combos(starts=starts, ends=ends, pool=(lo, hi)))
    else:
        if not args.base.strip():
            raise SystemExit("--phase drop 需要 --base，例如 --base 4-16")
        base_layers = _parse_one_combo(args.base.strip())
        if args.combos_only:
            if not args.combos.strip() and not args.ranges.strip():
                raise SystemExit("--combos-only 需要 --combos 或 --ranges")
        else:
            drop_ks = parse_drop_k(args.drop_k)
            if not drop_ks:
                raise SystemExit("--drop-k 至少要有一个正整数")
            keep_ends = not bool(args.no_keep_ends)
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
    return combos, base_layers, drop_ks


def main() -> None:
    p = argparse.ArgumentParser(description="band 多层 contig/drop（可选全通道+LSD+cos）")
    add_model_args(p, n_files=150)
    p.add_argument("--phase", type=str, default="contig", choices=["contig", "drop"])
    p.add_argument("--base", type=str, default="", help="drop 底集，如 4-16")
    p.add_argument("--drop-k", type=str, default="1,2,3")
    p.add_argument("--keep-ends", action="store_true", default=True)
    p.add_argument("--no-keep-ends", action="store_true")
    p.add_argument("--combos", type=str, default="", help="手工组合，分号分隔")
    p.add_argument("--combos-only", action="store_true")
    p.add_argument("--ranges", type=str, default="", help="如 4-14,4-16,4-28")
    p.add_argument("--pool-start", type=int, default=DEFAULT_POOL[0])
    p.add_argument("--pool-end", type=int, default=DEFAULT_POOL[1])
    p.add_argument("--contig-starts", type=str, default=DEFAULT_STARTS)
    p.add_argument("--contig-ends", type=str, default=DEFAULT_ENDS)
    p.add_argument("--n-per-corpus", type=int, default=50)
    p.add_argument("--num-workers", type=int, default=1)
    p.add_argument("--npu-ids", type=str, default="0,1,2,3,4,5,6,7")
    p.add_argument("--cpu-threads", type=int, default=1)
    p.add_argument("--cluster-dir", type=str, default=str(DEFAULT_CLUSTER_DIR))
    p.add_argument("--period", type=int, default=3)
    p.add_argument("--slide-win", type=int, default=18)
    p.add_argument(
        "--full-channels",
        action="store_true",
        help="全通道检测 + LSD + orig↔wm cosine/corr/mse（首次跨带稀疏实验用）",
    )
    args = p.parse_args()
    full = bool(args.full_channels)

    ntag = (
        f"n{int(args.n_per_corpus)}"
        if int(args.n_per_corpus) > 0
        else f"n{int(args.n_files)}"
    )
    if not str(args.out_dir or "").strip():
        stem = f"layer_pareto_{args.phase}_{ntag}"
        if full:
            stem = f"layer_pareto_{args.phase}_full_{ntag}"
        args.out_dir = str(Path(r"D:/data/codec_an/exp/band") / stem)
    cfg = prepare_run(args, f"layer_pareto_band_{args.phase}")
    if int(args.n_per_corpus) > 0:
        wavs = list_audio_per_corpus(args.wav_dir, int(args.n_per_corpus))
    else:
        wavs = list_audio(args.wav_dir, args.n_files)
    if not wavs:
        raise SystemExit(f"no audio in {args.wav_dir}")

    combos, base_layers, drop_ks = _build_combos(args)
    lo, hi = int(args.pool_start), int(args.pool_end)
    labels = [combo_label(c) for c in combos]
    by_corp = {c: 0 for c in (*CORPUS_NAMES, "other")}
    for ap in wavs:
        by_corp[corpus_tag(ap.stem)] += 1

    num_workers = max(1, int(args.num_workers))
    npu_ids = _parse_npu_ids(args.npu_ids)
    if num_workers > 1 and len(npu_ids) < num_workers:
        raise SystemExit(f"--num-workers={num_workers} needs >= {num_workers} --npu-ids")
    os.environ["WATERMARK_CPU_THREADS"] = str(max(1, int(args.cpu_threads)))
    _limit_cpu_threads(int(args.cpu_threads))

    mode = "full-channels+LSD+cos" if full else "feature-only"
    print(
        f"band {mode} phase={args.phase} pool L{lo}-L{hi} "
        f"n_combos={len(combos)} n_files={len(wavs)} workers={num_workers}",
        flush=True,
    )
    if args.phase == "drop":
        print(
            f"base={combo_label(base_layers) if base_layers else None} "
            f"drop_k={drop_ks} keep_ends={not bool(args.no_keep_ends)}",
            flush=True,
        )
    print(
        f"corpus aishell={by_corp['aishell']} zh={by_corp['zh']} "
        f"en={by_corp['en']} other={by_corp['other']}",
        flush=True,
    )
    for lab in labels:
        print("  combo", lab, flush=True)

    ensure_rare_table(path=args.table)
    if num_workers == 1:
        device = (args.device or "").strip() or auto_device()
        shards, devices = [wavs], [device]
    else:
        shards = [[] for _ in range(num_workers)]
        for i, ap in enumerate(wavs):
            shards[i % num_workers].append(ap)
        devices = [f"npu:{int(npu_ids[w])}" for w in range(num_workers)]

    payloads = []
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
                "seed": SEED,
                "cpu_threads": int(args.cpu_threads),
                "cluster_dir": str(args.cluster_dir),
                "period": int(args.period),
                "slide_win": int(args.slide_win),
                "full_channels": full,
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
                out = fut.result()
                print(
                    f"[main] worker {wid} done files={out['n_files']} rows={len(out['rows'])}",
                    flush=True,
                )
                clip_rows.extend(out["rows"])
                devices_used.append(out["device"])

    combo_mean = summarize(clip_rows, full=full)
    by_corpus = {
        c: summarize(clip_rows, c, full=full)
        for c in CORPUS_NAMES
        if any(r.get("corpus") == c for r in clip_rows)
    }
    del clip_rows
    summary = {
        "experiment": "layer_pareto_band",
        "phase": args.phase,
        "space": "band",
        "full_channels": full,
        "detect_keys": list(DETECT_KEYS if full else FEAT_KEYS),
        "skip": []
        if full
        else ["wave", "residual_wav", "residual_pair", "lsd", "orig_vs_wm_cosine"],
        "protocol": (
            "embed and detect use the same layer set; "
            "band query = sum z_q over written layers only; "
            + (
                "full channels + LSD + orig↔wm cosine/corr/mse"
                if full
                else "feature channels only"
            )
        ),
        "note": "Do not overwrite residual layer_pareto_*. Only combo averages are saved.",
        "base": combo_label(base_layers) if base_layers else None,
        "drop_k": drop_ks,
        "keep_ends": not bool(args.no_keep_ends),
        "pool": [lo, hi],
        "contig_starts": parse_int_csv(args.contig_starts),
        "contig_ends": parse_int_csv(args.contig_ends),
        "combos": labels,
        "n_files": len(wavs),
        "n_per_corpus": int(args.n_per_corpus) if int(args.n_per_corpus) > 0 else None,
        "corpus_counts": by_corp,
        "num_workers": num_workers,
        "device": devices_used[0] if len(devices_used) == 1 else devices_used,
        "seed": None if args.no_seed else SEED,
        "combo_mean": combo_mean,
        "by_corpus": by_corpus,
    }
    out_dir = cfg["out_dir"]
    dump_json(out_dir / "summary.json", summary)
    dump_json(out_dir / "combo_mean.json", combo_mean)
    dump_json(
        out_dir / "combo_mean_by_corpus.json",
        {
            "n_files": len(wavs),
            "corpus_counts": by_corp,
            "n_combos": len(combos),
            "combo_mean": combo_mean,
            "by_corpus": by_corpus,
        },
    )
    print("saved", out_dir / "summary.json", flush=True)
    _print_combo_table("combo means · all", combo_mean, full)
    for corp in CORPUS_NAMES:
        if corp in by_corpus:
            _print_combo_table(f"combo means · {corp}", by_corpus[corp], full)


if __name__ == "__main__":
    import multiprocessing as mp

    try:
        mp.set_start_method("spawn", force=True)
    except RuntimeError:
        pass
    main()
