"""三 codec 各层模长 / 能量占编码特征的比例（mix_1500）。

与 n=3 ``moss_norm_direction_remain`` 同一套定义。z 是进入该 RVQ 栈第 0 层
之前的编码残差。第 L 层：

    模长比   = ||z_q_L|| / ||z||
    能量比   = ||z_q_L||^2 / ||z||^2

逐帧算比值再对时间平均；同时报总能量之比（分子能量和 / 分母能量和）。
Qwen 有语义 / 声学两段 RVQ，每层对比自己那一段的入口 z。

默认 1500 条、8 进程 × npu:0–7。每句一个 JSON：``{out}/{codec}/per_clip/{stem}.json``。

运行::

  $env:PYTHONPATH="e:\\code\\water_marks"
  python -m other.experiments.run_codecs_layer_energy_n1500
  python -m other.experiments.run_codecs_layer_energy_n1500 --codecs moss --n-clips 3 --num-workers 1 --device npu:0
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from base.codec_dac import wav_to_batch as dac_wav_to_batch
from base.codec_qwen import TOKENIZER_PATH as QWEN_TOKENIZER_DEFAULT
from base.codec_qwen import embeddings as qwen_embeddings
from base.codec_qwen import waveform_values as qwen_waveform_values
from base.config import EXP_ROOT, default_mix_1500, default_model_path, first_existing
from base.moss_io import read_wav
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
from base.residual_feat import encode_hidden
from base.util import dump_json
from P2.experiments.run_p2_beta_layer_inject import pick_clip_paths
from P2.experiments.run_p2_cond_residual_n1500 import load_codec

OUT_STEM = "codecs_layer_energy_n1500"
_LOCAL_EXP = Path(r"/data1/home/ynian/codes/eval/add_exp/result")
DEFAULT_OUT = (EXP_ROOT if EXP_ROOT.exists() else _LOCAL_EXP) / OUT_STEM
DEFAULT_CODECS = ("moss", "qwen", "dac")
N_CLIPS = 1500
_QWEN_PATHS = (
    Path(QWEN_TOKENIZER_DEFAULT),
    Path(r"/data1/home/ynian/pred_models/Qwen3-TTS-Tokenizer-12Hz"),
    Path(r"/data1/home/ynian/pred_models/Qwen3-TTS-12Hz-1.7B-Base/speech_tokenizer"),
    Path(r"E:\pred_model\Qwen3-TTS-12Hz-1.7B-Base\speech_tokenizer"),
)
_DAC_PATHS = (
    Path(r"/data1/home/ynian/pred_models/dac/weights_24khz_8kbps_0.0.4.pth"),
    Path(r"E:\pred_model\dac\weights_24khz_8kbps_0.0.4.pth"),
)

def walk_rvq(model, hidden, lengths, delta_at: dict[int, torch.Tensor] | None = None):
    """返回 codes [n_q, T]、各层 z_q [D, T]、各层量化前残差 [D, T]。"""
    quantizer = model.quantizer
    n_q = len(quantizer.quantizers)
    z = quantizer.input_proj(hidden)
    batch, _, max_time = z.shape
    t_use = min(int(lengths[0].item()), int(max_time))
    mask = torch.arange(max_time, device=z.device).expand(batch, max_time) < t_use
    residual = z.clone().float()
    codes = np.zeros((n_q, t_use), dtype=np.int64)
    zqs: list[np.ndarray] = []
    pres: list[np.ndarray] = []
    delta_at = delta_at or {}
    with torch.no_grad():
        for i in range(n_q):
            pres.append(residual[0, :, :t_use].detach().float().cpu().numpy())
            masked = residual * mask.unsqueeze(1)
            if i in delta_at:
                delta = delta_at[i]
                if delta.shape[-1] < residual.shape[-1]:
                    pad = torch.zeros(
                        delta.shape[0],
                        delta.shape[1],
                        residual.shape[-1] - delta.shape[-1],
                        device=delta.device,
                        dtype=delta.dtype,
                    )
                    delta = torch.cat([delta, pad], dim=-1)
                elif delta.shape[-1] > residual.shape[-1]:
                    delta = delta[..., : residual.shape[-1]]
                z_q_i, indices, _ = quantizer.quantizers[i]((masked - delta) * mask.unsqueeze(1))
                residual = residual - delta - z_q_i * mask.unsqueeze(1) + delta
            else:
                z_q_i, indices, _ = quantizer.quantizers[i](masked)
                residual = residual - z_q_i * mask.unsqueeze(1)
            codes[i] = indices[0, :t_use].detach().long().cpu().numpy()
            zqs.append(z_q_i[0, :, :t_use].detach().float().cpu().numpy())
    return codes, zqs, pres


def _fmt(x) -> str:
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return "-"
    return f"{float(x):.4f}"


def _mean(xs) -> float | None:
    vals = [float(x) for x in xs if x is not None and np.isfinite(float(x))]
    return None if not vals else float(np.mean(vals))


def clip_name(tag: str) -> str:
    return str(tag).replace("/", "_").replace("\\", "_")


def to_dt(x) -> np.ndarray:
    """[D, T]。3 维取 batch 0。"""
    if torch.is_tensor(x):
        x = x.detach().float().cpu().numpy()
    a = np.asarray(x, dtype=np.float64)
    if a.ndim == 3:
        a = a[0]
    if a.ndim != 2:
        raise ValueError(f"feature rank {a.ndim}, expected 2 or 3")
    return a


def align_dt(zq: np.ndarray, zref: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    q = to_dt(zq)
    z = to_dt(zref)
    if q.shape[0] != z.shape[0] and q.shape[0] == z.shape[1]:
        z = z.T
    if q.shape[0] != z.shape[0] and q.shape[1] == z.shape[0]:
        q = q.T
    if q.shape[0] != z.shape[0]:
        raise ValueError(f"z_q {q.shape} vs z {z.shape}")
    t = min(int(q.shape[1]), int(z.shape[1]))
    return q[:, :t], z[:, :t]


def layer_ratios(zq, zref) -> dict:
    q, z = align_dt(zq, zref)
    qn = np.linalg.norm(q, axis=0)
    zn = np.linalg.norm(z, axis=0) + 1e-20
    q2 = qn * qn
    z2 = zn * zn
    return {
        "norm_ratio": float(np.mean(qn / zn)),
        "energy_ratio": float(np.mean(q2 / z2)),
        "norm_ratio_total": float(np.sum(qn) / np.sum(zn)),
        "energy_ratio_total": float(np.sum(q2) / np.sum(z2)),
        "T": int(q.shape[1]),
        "D": int(q.shape[0]),
    }


def collect_moss(model, wav: np.ndarray):
    hidden, lengths = encode_hidden(model, wav)
    _codes, zqs, pres = walk_rvq(model, hidden, lengths)
    z0 = pres[0]
    return [np.asarray(x) for x in zqs], [z0] * len(zqs), None


def collect_dac(model, wav: np.ndarray, sr: int):
    x = dac_wav_to_batch(model, wav, sr)
    zqs = []
    with torch.inference_mode():
        residual = model.encoder(x).float()
        z0 = to_dt(residual[0])
        for q in model.quantizer.quantizers:
            z_q_i, _cmt, _cbl, _idx, _ze = q(residual)
            zqs.append(to_dt(z_q_i[0]))
            residual = residual - z_q_i
    return zqs, [z0] * len(zqs), None


def collect_qwen(tok, wav: np.ndarray, sr: int):
    n_q = int(tok.model.encoder_valid_num_quantizers)
    quant = tok.model.encoder.quantizer
    n_sem = int(quant.num_semantic_quantizers)
    zqs: list[np.ndarray | None] = [None] * n_q
    z_refs: list[np.ndarray | None] = [None] * n_q
    values = qwen_waveform_values(tok, wav, sr)
    model_dtype = tok.model.dtype
    with torch.inference_mode():
        emb = qwen_embeddings(tok, values).to(dtype=model_dtype)

        def run_branch(rvq, global_layers):
            if not global_layers:
                return
            h = emb
            if rvq.input_proj is not None:
                h = rvq.input_proj(h)
            residual = h.to(dtype=model_dtype)
            z0 = to_dt(residual[0])
            for loc, gL in enumerate(global_layers):
                layer_m = rvq.layers[loc]
                idx = layer_m.encode(residual)
                if isinstance(idx, (tuple, list)):
                    idx = idx[0]
                z_q = layer_m.decode(idx)
                zqs[gL] = to_dt(z_q[0])
                z_refs[gL] = z0
                residual = residual - z_q

        run_branch(quant.semantic_residual_vector_quantizer, list(range(0, min(n_sem, n_q))))
        run_branch(quant.acoustic_residual_vector_quantizer, list(range(n_sem, n_q)))
    if any(x is None for x in zqs):
        missing = [i for i, x in enumerate(zqs) if x is None]
        raise RuntimeError(f"qwen missing layers {missing}")
    return zqs, z_refs, int(n_sem)


def measure_clip(codec: str, handle, wav: np.ndarray, sr: int) -> dict:
    codec = str(codec).lower()
    if codec == "moss":
        zqs, refs, n_sem = collect_moss(handle, wav)
    elif codec == "dac":
        zqs, refs, n_sem = collect_dac(handle, wav, sr)
    elif codec == "qwen":
        zqs, refs, n_sem = collect_qwen(handle, wav, sr)
    else:
        raise RuntimeError(codec)
    layers = []
    for i, (zq, zref) in enumerate(zip(zqs, refs)):
        rec = layer_ratios(zq, zref)
        rec["layer"] = int(i)
        layers.append(rec)
    energy_mean = [r["energy_ratio"] for r in layers]
    energy_tot = [r["energy_ratio_total"] for r in layers]
    out = {
        "n_q": len(layers),
        "T": int(layers[0]["T"]) if layers else 0,
        "D": int(layers[0]["D"]) if layers else 0,
        "energy_sum": float(sum(energy_mean)),
        "energy_sum_total": float(sum(energy_tot)),
        "layers": layers,
    }
    if n_sem is not None:
        out["n_sem"] = int(n_sem)
        sem = layers[: int(n_sem)]
        ac = layers[int(n_sem) :]
        out["energy_sum_semantic"] = float(sum(r["energy_ratio"] for r in sem))
        out["energy_sum_acoustic"] = float(sum(r["energy_ratio"] for r in ac))
    return out


def clip_json_ok(path: Path, codec: str) -> bool:
    if not path.exists():
        return False
    try:
        doc = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return False
    if not isinstance(doc, dict) or doc.get("error"):
        return False
    if str(doc.get("codec") or "") != str(codec):
        return False
    layers = doc.get("layers")
    return isinstance(layers, list) and len(layers) > 0 and "energy_ratio" in layers[0]


def _worker(payload: dict) -> dict:
    limit_cpu_threads(int(payload.get("cpu_threads") or 1))
    pin_npu(str(payload["device"]))
    log_fp, log = open_worker_log(payload)
    codec = str(payload["codec"]).lower()
    resume = bool(payload.get("resume"))
    per_dir = Path(payload["per_dir"])
    per_dir.mkdir(parents=True, exist_ok=True)
    log(f"start {codec} device={payload['device']} clips={len(payload['items'])}")
    handle, sr = load_codec(codec, payload)
    n_ok = 0
    n_err = 0
    n_skip = 0
    for i, (tag, path_str) in enumerate(payload["items"]):
        out_p = per_dir / f"{clip_name(tag)}.json"
        if resume and clip_json_ok(out_p, codec):
            n_skip += 1
            n_ok += 1
            continue
        try:
            wav = np.asarray(read_wav(path_str, int(sr)), dtype=np.float32).reshape(-1)
            rec = measure_clip(codec, handle, wav, int(sr))
            doc = {
                "name": tag,
                "src": str(path_str),
                "codec": codec,
                "sr": int(sr),
                **rec,
            }
            dump_json(out_p, doc)
            n_ok += 1
            top = rec["layers"][0] if rec["layers"] else {}
            log(
                f"saved {tag} {i + 1}/{len(payload['items'])} "
                f"n_q={rec['n_q']} T={rec['T']} "
                f"L0_norm={top.get('norm_ratio', 0):.3f} "
                f"L0_energy={top.get('energy_ratio', 0):.3f} "
                f"energy_sum={rec['energy_sum']:.3f}"
            )
        except Exception as exc:
            n_err += 1
            log(f"FAIL {tag}: {type(exc).__name__}: {exc}")
            dump_json(
                out_p,
                {
                    "name": tag,
                    "src": str(path_str),
                    "codec": codec,
                    "error": f"{type(exc).__name__}: {exc}",
                },
            )
    del handle
    free_device()
    if log_fp is not None:
        log_fp.close()
    return {
        "worker_id": int(payload["worker_id"]),
        "codec": codec,
        "n_ok": int(n_ok),
        "n_error": int(n_err),
        "n_skip": int(n_skip),
    }


def load_docs(per_dir: Path, clips: list[tuple[str, str]], codec: str) -> list[dict]:
    docs = []
    for tag, _ in clips:
        fp = per_dir / f"{clip_name(tag)}.json"
        if not clip_json_ok(fp, codec):
            continue
        try:
            docs.append(json.loads(fp.read_text(encoding="utf-8")))
        except Exception:
            pass
    return docs


def aggregate(docs: list[dict], codec: str) -> dict:
    n_q = max((int(d.get("n_q") or 0) for d in docs), default=0)
    layers = []
    for L in range(n_q):
        layers.append(
            {
                "layer": int(L),
                "norm_ratio": _mean(
                    [r["norm_ratio"] for d in docs for r in d.get("layers") or [] if int(r["layer"]) == L]
                ),
                "energy_ratio": _mean(
                    [r["energy_ratio"] for d in docs for r in d.get("layers") or [] if int(r["layer"]) == L]
                ),
                "norm_ratio_total": _mean(
                    [
                        r["norm_ratio_total"]
                        for d in docs
                        for r in d.get("layers") or []
                        if int(r["layer"]) == L
                    ]
                ),
                "energy_ratio_total": _mean(
                    [
                        r["energy_ratio_total"]
                        for d in docs
                        for r in d.get("layers") or []
                        if int(r["layer"]) == L
                    ]
                ),
                "n": sum(1 for d in docs for r in d.get("layers") or [] if int(r["layer"]) == L),
            }
        )
    return {
        "codec": codec,
        "n_clips": len(docs),
        "n_q": n_q,
        "n_sem": next((d.get("n_sem") for d in docs if d.get("n_sem") is not None), None),
        "formula": "norm=||zq||/||z||, energy=||zq||^2/||z||^2, z=encode residual into this RVQ stack",
        "energy_sum": _mean([d.get("energy_sum") for d in docs]),
        "energy_sum_total": _mean([d.get("energy_sum_total") for d in docs]),
        "energy_sum_semantic": _mean([d.get("energy_sum_semantic") for d in docs]),
        "energy_sum_acoustic": _mean([d.get("energy_sum_acoustic") for d in docs]),
        "layers": layers,
    }


def render(summaries: list[dict]) -> str:
    lines = [
        "# 三 codec 各层模长 / 能量占比（n1500）\n\n",
        "模长比 `||z_q||/||z||`，能量比 `||z_q||^2/||z||^2`。",
        "`z` 是进入该 RVQ 栈第 0 层之前的编码残差。",
        "表里是各句逐帧比值的平均，再对句子平均。\n\n",
    ]
    for summary in summaries:
        lines.append(f"## {summary['codec']}  n={summary['n_clips']}  n_q={summary['n_q']}\n\n")
        extra = f"能量比之和 { _fmt(summary.get('energy_sum')) }"
        if summary.get("n_sem") is not None:
            extra += (
                f"（语义 {_fmt(summary.get('energy_sum_semantic'))} / "
                f"声学 {_fmt(summary.get('energy_sum_acoustic'))}，n_sem={summary['n_sem']}）"
            )
        lines.append(extra + "\n\n")
        lines.append("| 层 | 模长比 | 能量比 | 模长比(总量) | 能量比(总量) | n |\n")
        lines.append("|---:|---:|---:|---:|---:|---:|\n")
        for row in summary.get("layers") or []:
            lines.append(
                f"| {row['layer']} | {_fmt(row['norm_ratio'])} | {_fmt(row['energy_ratio'])} | "
                f"{_fmt(row['norm_ratio_total'])} | {_fmt(row['energy_ratio_total'])} | {row['n']} |\n"
            )
        lines.append("\n")
    lines.append(
        "能量比之和不是 1：各层 z_q 有重叠。Qwen 语义层相对语义入口 z，声学层相对声学入口 z。\n"
    )
    return "".join(lines)


def parse_args():
    p = argparse.ArgumentParser(description="Per-layer norm/energy share vs encode z, 3 codecs, n1500")
    p.add_argument("--wav-dir", type=str, default=default_mix_1500())
    p.add_argument("--n-clips", type=int, default=N_CLIPS)
    p.add_argument("--codecs", type=str, default=",".join(DEFAULT_CODECS))
    p.add_argument("--model_path", type=str, default=default_model_path())
    p.add_argument("--tokenizer-path", type=str, default=first_existing(*_QWEN_PATHS))
    p.add_argument("--dac-path", type=str, default=first_existing(*_DAC_PATHS))
    p.add_argument("--out_dir", type=str, default=str(DEFAULT_OUT))
    p.add_argument("--resume", action="store_true")
    add_npu_args(p, default_workers=8, default_device="npu:0")
    return p.parse_args()


def run_one_codec(codec: str, clips: list[tuple[str, str]], args, out_root: Path) -> dict:
    codec_dir = out_root / codec
    logs = codec_dir / "logs"
    per_dir = codec_dir / "per_clip"
    logs.mkdir(parents=True, exist_ok=True)
    per_dir.mkdir(parents=True, exist_ok=True)
    n_workers = max(1, int(args.num_workers))
    npu_ids = parse_npu_ids(args.npu_ids)
    print(
        f"=== {codec} layer energy clips={len(clips)} workers={n_workers} per_clip={per_dir} ===",
        flush=True,
    )
    payloads = build_payloads(
        clips,
        num_workers=n_workers,
        npu_ids=npu_ids,
        device_single=args.device,
        logs_dir=logs,
        common={
            "codec": codec,
            "model_path": args.model_path,
            "tokenizer_path": str(args.tokenizer_path),
            "dac_path": str(args.dac_path or ""),
            "per_dir": str(per_dir),
            "cpu_threads": int(args.cpu_threads),
            "resume": bool(args.resume),
        },
    )
    outs = run_shards(_worker, payloads, num_workers=n_workers)
    docs = load_docs(per_dir, clips, codec)
    summary = aggregate(docs, codec)
    dump_json(codec_dir / "summary.json", summary)
    (codec_dir / "FINDINGS.md").write_text(render([summary]), encoding="utf-8")
    n_ok = sum(int(o.get("n_ok") or 0) for o in outs)
    n_err = sum(int(o.get("n_error") or 0) for o in outs)
    print(
        f"  [{codec}] docs={len(docs)} n_ok={n_ok} n_err={n_err} -> {codec_dir}",
        flush=True,
    )
    return summary


def main() -> None:
    args = parse_args()
    codecs = [x.strip().lower() for x in str(args.codecs).split(",") if x.strip()]
    for name in codecs:
        if name not in DEFAULT_CODECS:
            raise SystemExit(f"unknown codec {name}")
    out_dir = Path(args.out_dir) if args.out_dir else DEFAULT_OUT
    out_dir.mkdir(parents=True, exist_ok=True)
    clips = pick_clip_paths(Path(args.wav_dir), int(args.n_clips))
    summaries = [run_one_codec(codec, clips, args, out_dir) for codec in codecs]
    dump_json(out_dir / "summary.json", {"codecs": summaries})
    (out_dir / "FINDINGS.md").write_text(render(summaries), encoding="utf-8")
    print(f"done -> {out_dir}", flush=True)


if __name__ == "__main__":
    ensure_spawn()
    main()
