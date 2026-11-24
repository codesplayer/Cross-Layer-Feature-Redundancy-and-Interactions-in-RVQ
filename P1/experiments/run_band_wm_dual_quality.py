"""band 同 token：只算 LSD@16k（与 audioseal/wavmark 的 run_wm_clean_quality 对齐）。

每条 clip 只 encode 一次::

  tokens = encode(orig)                 # MOSS 原生 sr（多为 24k）
  recon  = decode(tokens)
  wm     = decode(apply_plan(tokens, band_L9))

再统一重采样到 ``--metric-sr``（默认 **16000**）算三组 LSD：
  - ``lsd_wm_orig``：水印 vs 原始
  - ``lsd_recon_orig``：同 token 干净重构 vs 原始
  - ``lsd_wm_recon``：水印 vs 同 token 干净重构

例::

  python -m P1.experiments.run_band_wm_dual_quality \\
    --orig-dir /data1/.../fleurs/select_n2000 \\
    --out-dir /data1/.../result/band_wm_dual_quality \\
    --num-workers 8 --npu-ids 0,1,2,3,4,5,6,7
"""

from __future__ import annotations

import argparse
import json
import traceback
from pathlib import Path

import numpy as np

from base.config import EXP_ROOT, default_mix_1500, default_model_path, default_table_path
from base.io import match_len, write_wav_float
from base.lsd import CalcuLSD
from base.moss_io import decode_moss, encode_moss, read_wav, set_seed
from base.npu_parallel import (
    add_npu_args,
    build_payloads,
    ensure_spawn,
    free_device,
    limit_cpu_threads,
    open_worker_log,
    parse_npu_ids,
    pin_npu,
    run_shards,
)
from base.plan import apply_plan, build_rare_cycle_plan
from base.rare_tokens import ensure_rare_table
from base.util import dump_json, load_model
from P1.experiments.run_anchor_attack_three_n1500 import (
    list_clips,
    load_clips_from_manifest,
    save_clean_wm_wav,
)

OUT_STEM = "band_wm_dual_quality"
_OUT_CANDS = (
    Path(r"/data1/home/ynian/codes/eval/add_exp/result") / OUT_STEM,
    Path(r"E:\code\water_marks\result") / OUT_STEM,
    EXP_ROOT / OUT_STEM,
)
DEFAULT_OUT = next((p for p in _OUT_CANDS if p.parent.exists()), _OUT_CANDS[0])
# v3：仅 LSD@16k；三对照 wm↔orig / recon↔orig / wm↔recon
METRIC_VERSION = 3
DEFAULT_METRIC_SR = 16000
LSD_KEYS = ("lsd_wm_orig", "lsd_recon_orig", "lsd_wm_recon")


def _write_json(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(obj, ensure_ascii=False), encoding="utf-8")
    tmp.replace(path)


def resample_to(wav: np.ndarray, sr_from: int, sr_to: int) -> np.ndarray:
    wav = np.asarray(wav, dtype=np.float32).reshape(-1)
    if int(sr_from) == int(sr_to):
        return wav
    import librosa

    return librosa.resample(wav, orig_sr=int(sr_from), target_sr=int(sr_to)).astype(np.float32)


def lsd_pair(lsd_tool: CalcuLSD, ref: np.ndarray, deg: np.ndarray) -> float | None:
    ref = np.asarray(ref, dtype=np.float32).reshape(-1)
    deg = np.asarray(match_len(ref, deg), dtype=np.float32).reshape(-1)
    try:
        val, _ = lsd_tool.compute(ref, deg)
        return float(val)
    except Exception:
        return None


def clip_ok(path: Path) -> bool:
    if not path.is_file():
        return False
    try:
        doc = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return False
    if not isinstance(doc, dict) or doc.get("metric_version") != METRIC_VERSION or "error" in doc:
        return False
    if not doc.get("same_tokens"):
        return False
    return all(doc.get(k) is not None for k in LSD_KEYS)


def same_token_recon_and_wm(
    model,
    wav: np.ndarray,
    table: dict,
    *,
    high_start: int = 9,
    high_end: int = 9,
    layers=(9,),
    period: int = 3,
) -> dict:
    """一次 encode → 同 tokens 干净 decode + 嵌印 decode。"""
    wav = np.asarray(wav, dtype=np.float32).reshape(-1)
    tokens = encode_moss(model, wav)
    n_q, T = tokens.shape
    recon = np.asarray(match_len(wav, decode_moss(model, tokens)), dtype=np.float32).reshape(-1)
    plan = build_rare_cycle_plan(
        T,
        n_q=n_q,
        high_start=int(high_start),
        high_end=int(high_end),
        table=table,
        layers=list(layers),
        period=int(period),
    )
    mod, n_changed = apply_plan(tokens, plan)
    wm = np.asarray(match_len(wav, decode_moss(model, mod)), dtype=np.float32).reshape(-1)
    return {
        "recon": recon,
        "wm": wm,
        "n_changed": int(n_changed),
        "T": int(T),
        "n_q": int(n_q),
        "period": int(period),
    }


def _mean(xs) -> float | None:
    vals = [float(x) for x in xs if x is not None and np.isfinite(float(x))]
    return None if not vals else float(np.mean(vals))


def aggregate(per_dir: Path) -> dict:
    docs = []
    for p in sorted(per_dir.glob("*.json")):
        try:
            d = json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            continue
        if d.get("error") or d.get("metric_version") != METRIC_VERSION:
            continue
        if not d.get("same_tokens"):
            continue
        if any(d.get(k) is None for k in LSD_KEYS):
            continue
        docs.append(d)
    return {
        "method": "band",
        "n_clips": len(docs),
        "metric_version": METRIC_VERSION,
        "same_tokens": True,
        "metric_sr": DEFAULT_METRIC_SR,
        "mean": {k: _mean([d.get(k) for d in docs]) for k in LSD_KEYS},
    }


def render_md(summary: dict) -> str:
    def f(x):
        return "-" if x is None else f"{float(x):.4f}"

    m = summary.get("mean") or {}
    return (
        f"# band same-token LSD@{summary.get('metric_sr', DEFAULT_METRIC_SR)}\n\n"
        f"n={summary.get('n_clips')}  metric_v={summary.get('metric_version')}\n\n"
        f"pipeline: `tokens=encode(orig)` → `recon=decode(tokens)` / "
        f"`wm=decode(apply_plan(tokens))`；LSD 统一 resample 到 metric_sr\n\n"
        f"| compare | LSD |\n"
        f"|---|---:|\n"
        f"| wm vs orig | {f(m.get('lsd_wm_orig'))} |\n"
        f"| recon vs orig | {f(m.get('lsd_recon_orig'))} |\n"
        f"| wm vs recon | {f(m.get('lsd_wm_recon'))} |\n"
    )


def _worker(payload: dict) -> dict:
    limit_cpu_threads(int(payload.get("cpu_threads") or 1))
    device = str(payload["device"])
    pin_npu(device)
    log_fp, log = open_worker_log(payload)
    per_dir = Path(payload["per_clip_dir"])
    wm_out = Path(payload["wm_out_dir"])
    recon_out = Path(payload["recon_out_dir"])
    for d in (per_dir, wm_out, recon_out):
        d.mkdir(parents=True, exist_ok=True)
    resume = bool(payload.get("resume"))
    high_start = int(payload.get("high_start", 9))
    high_end = int(payload.get("high_end", 9))
    period = int(payload.get("period", 3))
    metric_sr = int(payload.get("metric_sr") or DEFAULT_METRIC_SR)
    n_ok = n_skip = n_err = 0

    try:
        set_seed(int(payload.get("seed") or 0) + int(payload["worker_id"]))
        model, sr = load_model(str(payload["moss_path"]), device)
        table = ensure_rare_table(str(payload["table"]))
        lsd_tool = CalcuLSD(sr=int(metric_sr))
        log(
            f"start clips={len(payload['items'])} moss_sr={sr} "
            f"metric_sr={metric_sr} same_tokens=1 lsd_only=1"
        )

        for i, (stem, orig_path) in enumerate(payload["items"]):
            safe = str(stem).replace("/", "_").replace("\\", "_")
            out_json = per_dir / f"{safe}.json"
            if resume and clip_ok(out_json):
                n_skip += 1
                n_ok += 1
                continue
            try:
                orig = np.asarray(read_wav(orig_path, int(sr)), dtype=np.float32).reshape(-1)
                pack = same_token_recon_and_wm(
                    model,
                    orig,
                    table,
                    high_start=high_start,
                    high_end=high_end,
                    layers=(9,),
                    period=period,
                )
                recon = pack["recon"]
                wm = pack["wm"]

                dest_r = recon_out / f"{safe}.wav"
                write_wav_float(dest_r, recon, int(sr))
                recon_path = str(dest_r.resolve())
                wm_path = save_clean_wm_wav(wm_out, stem, wm, int(sr))

                orig_m = resample_to(orig, int(sr), metric_sr)
                recon_m = resample_to(recon, int(sr), metric_sr)
                wm_m = resample_to(wm, int(sr), metric_sr)

                _write_json(
                    out_json,
                    {
                        "stem": stem,
                        "method": "band",
                        "same_tokens": True,
                        "src": orig_path,
                        "moss_recon_wav": recon_path,
                        "wm_wav": wm_path,
                        "sr": int(sr),
                        "metric_sr": int(metric_sr),
                        "metric_version": METRIC_VERSION,
                        "n_samples": int(orig.shape[0]),
                        "n_changed": int(pack["n_changed"]),
                        "T": int(pack["T"]),
                        "n_q": int(pack["n_q"]),
                        "period": int(pack["period"]),
                        "layers": [9],
                        "lsd_wm_orig": lsd_pair(lsd_tool, orig_m, wm_m),
                        "lsd_recon_orig": lsd_pair(lsd_tool, orig_m, recon_m),
                        "lsd_wm_recon": lsd_pair(lsd_tool, recon_m, wm_m),
                    },
                )
                n_ok += 1
            except Exception as exc:
                n_err += 1
                _write_json(
                    out_json,
                    {
                        "stem": stem,
                        "src": orig_path,
                        "error": str(exc),
                        "trace": traceback.format_exc(),
                    },
                )
                log(f"ERR {stem}: {exc}")
            if (i + 1) % 20 == 0:
                log(f"progress {i+1}/{len(payload['items'])} ok={n_ok} skip={n_skip} err={n_err}")
    finally:
        free_device()
        if log_fp:
            log_fp.close()
    return {"worker_id": payload["worker_id"], "n_ok": n_ok, "n_skip": n_skip, "n_err": n_err}


def parse_args():
    p = argparse.ArgumentParser(
        description="band same-token LSD@16k: wm↔orig / recon↔orig / wm↔recon"
    )
    p.add_argument("--orig-dir", type=str, default="", help="原始语音目录")
    p.add_argument("--manifest", type=str, default="", help="优先于 --orig-dir")
    p.add_argument("--names", type=str, default="")
    p.add_argument("--out-dir", type=str, default=str(DEFAULT_OUT))
    p.add_argument("--moss-path", type=str, default=default_model_path())
    p.add_argument("--table", type=str, default=default_table_path())
    p.add_argument("--high-start", type=int, default=9)
    p.add_argument("--high-end", type=int, default=9)
    p.add_argument("--period", type=int, default=3)
    p.add_argument(
        "--metric-sr",
        type=int,
        default=DEFAULT_METRIC_SR,
        help="LSD 统一重采样率（默认 16000，对齐 run_wm_clean_quality）",
    )
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--resume", action="store_true", default=True)
    p.add_argument("--no-resume", action="store_true")
    p.add_argument("--summary-only", action="store_true")
    add_npu_args(p, default_workers=8, default_device="npu:0")
    return p.parse_args()


def main() -> None:
    ensure_spawn()
    args = parse_args()
    if args.no_resume:
        args.resume = False

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    per_dir = out_dir / "per_clip"
    wm_out = out_dir / "wav_wm"
    recon_out = out_dir / "wav_moss_recon"

    if bool(args.summary_only):
        summary = aggregate(per_dir)
        dump_json(out_dir / "summary.json", summary)
        (out_dir / "FINDINGS.md").write_text(render_md(summary), encoding="utf-8")
        print(f"summary-only n={summary['n_clips']} -> {out_dir}", flush=True)
        return

    if str(args.manifest or "").strip():
        clips = load_clips_from_manifest(Path(args.manifest), limit=int(args.limit))
        src_desc = f"manifest={args.manifest}"
    else:
        wav_dir = str(args.orig_dir or "").strip() or default_mix_1500()
        name_set = {x.strip() for x in str(args.names).replace(" ", ",").split(",") if x.strip()} or None
        clips = list_clips(Path(wav_dir), limit=int(args.limit), names=name_set)
        src_desc = f"orig_dir={wav_dir}" + (f" names={len(name_set)}" if name_set else "")
    if not clips:
        raise SystemExit(f"no clips ({src_desc})")

    n_workers = max(1, int(args.num_workers))
    for d in (per_dir, wm_out, recon_out, out_dir / "logs"):
        d.mkdir(parents=True, exist_ok=True)

    (out_dir / "RUN.md").write_text(
        "# band same-token LSD only\n\n"
        f"- orig: `{src_desc}` n={len(clips)}\n"
        f"- pipeline: encode once → recon / wm (same tokens)\n"
        f"- layers: L{args.high_start}..L{args.high_end}  period={args.period}\n"
        f"- metric: LSD only @ metric_sr={args.metric_sr}\n"
        f"- compares: wm↔orig ; recon↔orig ; wm↔recon\n"
        f"- metric_version: {METRIC_VERSION}\n"
        f"- workers: {n_workers}\n",
        encoding="utf-8",
    )
    print(
        f"band_dual LSD-only n={len(clips)} workers={n_workers} "
        f"metric_sr={args.metric_sr} src={src_desc} -> {out_dir}",
        flush=True,
    )

    payloads = build_payloads(
        clips,
        num_workers=n_workers,
        npu_ids=parse_npu_ids(args.npu_ids),
        device_single=str(args.device),
        logs_dir=out_dir / "logs",
        common={
            "moss_path": args.moss_path,
            "table": args.table,
            "per_clip_dir": str(per_dir),
            "wm_out_dir": str(wm_out),
            "recon_out_dir": str(recon_out),
            "cpu_threads": int(args.cpu_threads),
            "resume": bool(args.resume),
            "seed": int(args.seed),
            "high_start": int(args.high_start),
            "high_end": int(args.high_end),
            "period": int(args.period),
            "metric_sr": int(args.metric_sr),
        },
    )
    outs = run_shards(_worker, payloads, num_workers=n_workers)
    dump_json(out_dir / "workers.json", outs)
    summary = aggregate(per_dir)
    summary["metric_sr"] = int(args.metric_sr)
    dump_json(out_dir / "summary.json", summary)
    (out_dir / "FINDINGS.md").write_text(render_md(summary), encoding="utf-8")
    print(
        f"done n={summary['n_clips']} err={sum(int(o.get('n_err') or 0) for o in outs)} -> {out_dir}",
        flush=True,
    )


if __name__ == "__main__":
    main()
