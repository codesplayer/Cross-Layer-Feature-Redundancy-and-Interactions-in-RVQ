"""每条语音的最终额外方向残留，以及相对原波形的 LSD、MCD。

η_extra 逐帧提取正交分量、层间求和抵消后，再用整句 Encoder 能量归一化。
不先对残差做时间平均。

输出::

  {out}/{moss|qwen|dac}/{stem}.json

8 进程 × 8 NPU。已有 JSON 跳过。

例::

  python -m other.experiments.run_rvq_extra_eta_n1500 \\
    --codecs moss,qwen,dac --num-workers 8 --npu-ids 0,1,2,3,4,5,6,7
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from base.codec_dac import decode_codes, encode_codes, load_dac
from base.codec_qwen import decode_codes_kt, encode_codes_kt, load_qwen_tokenizer
from base.config import default_mix_1500, default_model_path
from base.lsd import CalcuLSD
from base.mcd import CalcuMCD
from base.moss_io import decode_moss, encode_moss, read_wav, write_wav
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
from base.util import dump_json, load_model
from other.experiments.run_feat_retain_vs_rcl_n1500 import list_mix_clips
from other.experiments.run_rvq_af_n1500 import QWEN_PATH, trace_dac, trace_moss, trace_qwen

DAC_PATH = Path("/data1/home/ynian/pred_models/dac/weights_24khz_8kbps_0.0.4.pth")
WAV_PATH = Path("/data1/home/ynian/codes/codec_an/exp_watermark/data/mix_1500")
OUT_STEM = "rvq_extra_eta_n1500"
_OUT_CANDIDATES = Path(r"/data1/home/ynian/codes/eval/add_exp/result") / OUT_STEM,
DEFAULT_OUT = Path(
    next((str(p.parent / OUT_STEM) for p in _OUT_CANDIDATES if p.parent.exists()), None)
    or _OUT_CANDIDATES[-1]
)
def compute_extra_residual(z, q_layers, valid_mask=None):
    """z: [T, D]，q_layers: [L, T, D]。逐帧正交分解后再按能量汇总。"""
    z = z.double()
    q_layers = q_layers.double()
    z_energy = (z ** 2).sum(dim=-1)
    if valid_mask is None:
        valid_mask = z_energy > 1e-12
    else:
        valid_mask = valid_mask.bool() & (z_energy > 1e-12)

    dot = (q_layers * z.unsqueeze(0)).sum(dim=-1)
    coeff = dot / z_energy.clamp_min(1e-12).unsqueeze(0)
    q_perp = q_layers - coeff.unsqueeze(-1) * z.unsqueeze(0)
    extra_final = q_perp.sum(dim=0)
    final_energy = (extra_final ** 2).sum(dim=-1)
    initial_energy = (q_perp ** 2).sum(dim=(0, 2))

    final_total = final_energy[valid_mask].sum()
    initial_total = initial_energy[valid_mask].sum()
    encoder_total = z_energy[valid_mask].sum()
    eta_extra = final_total / encoder_total
    eta_initial = initial_total / encoder_total
    cancel_ratio = 1 - final_total / initial_total.clamp_min(1e-12)

    big_q = q_layers.sum(dim=0)
    q_coeff = (big_q * z).sum(dim=-1) / z_energy.clamp_min(1e-12)
    extra_direct = big_q - q_coeff.unsqueeze(-1) * z
    equiv = torch.max(torch.abs(extra_final[valid_mask] - extra_direct[valid_mask])).item()
    return {
        "eta_extra": float(eta_extra.item()),
        "eta_initial": float(eta_initial.item()),
        "cancel_ratio": float(cancel_ratio.item()),
        "n_frames": int(valid_mask.sum().item()),
        "equiv_max_abs": float(equiv),
        "final_total": float(final_total.item()),
        "initial_total": float(initial_total.item()),
        "encoder_total": float(encoder_total.item()),
    }


def chain_eta(chain: dict) -> dict:
    """chain q 是 [L, D, T]，z 是 [D, T]。"""
    q = np.transpose(chain["q"], (0, 2, 1))
    z = np.transpose(chain["z"], (1, 0))
    stats = compute_extra_residual(torch.from_numpy(z), torch.from_numpy(q))
    stats["chain"] = chain["name"]
    stats["n_layers"] = int(chain["q"].shape[0])
    return stats


def pooled_eta(rows: list[dict]) -> dict:
    final = sum(r["final_total"] for r in rows)
    initial = sum(r["initial_total"] for r in rows)
    encoder = sum(r["encoder_total"] for r in rows)
    return {
        "eta_extra": float(final / encoder) if encoder else None,
        "eta_initial": float(initial / encoder) if encoder else None,
        "cancel_ratio": float(1.0 - final / initial) if initial else None,
        "n_frames": int(sum(r["n_frames"] for r in rows)),
        "equiv_max_abs": float(max(r["equiv_max_abs"] for r in rows)),
    }


def reconstruct(kind: str, model, wav: np.ndarray, sr: int) -> tuple[np.ndarray, int]:
    if kind == "moss":
        codes = encode_moss(model, wav)
        return decode_moss(model, codes), sr
    if kind == "dac":
        codes = encode_codes(model, wav, sr)
        return decode_codes(model, codes), sr
    codes = encode_codes_kt(model, wav, sr)
    return decode_codes_kt(model, codes)


def quality(ref: np.ndarray, deg: np.ndarray, sr: int, tmp: Path, stem: str, lsd_tool, mcd_tool) -> dict:
    ref = np.asarray(ref, dtype=np.float32).reshape(-1)
    deg = np.asarray(deg, dtype=np.float32).reshape(-1)
    n = min(ref.shape[0], deg.shape[0])
    ref, deg = ref[:n], deg[:n]
    out = {"lsd": None, "mcd": None, "mcd_mode": "pymcd_dtw"}
    try:
        val, _ = lsd_tool.compute(ref, deg)
        out["lsd"] = float(val)
    except Exception as exc:
        out["lsd_error"] = str(exc)
    tmp.mkdir(parents=True, exist_ok=True)
    rp, dp = tmp / f"{stem}_ref.wav", tmp / f"{stem}_deg.wav"
    write_wav(rp, ref, int(sr))
    write_wav(dp, deg, int(sr))
    out["mcd"] = float(mcd_tool.mcd_file(str(rp), str(dp)))
    rp.unlink(missing_ok=True)
    dp.unlink(missing_ok=True)
    return out


def write_json(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(obj, ensure_ascii=False), encoding="utf-8")
    tmp.replace(path)


def worker(payload: dict) -> dict:
    limit_cpu_threads(int(payload.get("cpu_threads", 2)))
    device = str(payload["device"])
    pin_npu(device)
    log_fp, log = open_worker_log(payload)
    out_dir = Path(payload["out_dir"])
    tmp = out_dir / "tmp" / f"w{payload['worker_id']}"
    n_ok = n_err = 0
    lsd_tool = mcd_tool = None
    try:
        for codec in payload["codecs"]:
            bundle = _load(codec, device, payload)
            if lsd_tool is None:
                lsd_tool = CalcuLSD(sr=int(bundle["sr"]))
                mcd_tool = CalcuMCD()
            for stem, path in payload["items"]:
                dest = out_dir / codec / f"{stem}.json"
                if dest.exists():
                    old = json.loads(dest.read_text(encoding="utf-8"))
                    if old.get("mcd_mode") == "pymcd_dtw" and old.get("mcd") is not None:
                        n_ok += 1
                        continue
                try:
                    wav = read_wav(path, bundle["sr"])
                    chains = bundle["trace"](wav)
                    per_chain = [chain_eta(c) for c in chains]
                    pooled = pooled_eta(per_chain)
                    rec, out_sr = reconstruct(bundle["kind"], bundle["model"], wav, bundle["sr"])
                    if int(out_sr) != int(bundle["sr"]):
                        lsd_use = CalcuLSD(sr=int(out_sr))
                    else:
                        lsd_use = lsd_tool
                    qual = quality(wav, rec, int(out_sr), tmp, f"{codec}_{stem}", lsd_use, mcd_tool)
                    write_json(
                        dest,
                        {
                            "stem": stem,
                            "codec": codec,
                            "eta_extra": pooled["eta_extra"],
                            "eta_initial": pooled["eta_initial"],
                            "cancel_ratio": pooled["cancel_ratio"],
                            "lsd": qual["lsd"],
                            "mcd": qual["mcd"],
                            "n_frames": pooled["n_frames"],
                            "equiv_max_abs": pooled["equiv_max_abs"],
                            "chains": per_chain,
                            "lsd_error": qual.get("lsd_error"),
                            "mcd_error": qual.get("mcd_error"),
                        },
                    )
                    n_ok += 1
                    if n_ok % 20 == 0:
                        log(f"{codec} ok={n_ok} {stem} eta={pooled['eta_extra']:.4f}")
                except Exception as exc:
                    n_err += 1
                    write_json(out_dir / "errors" / codec / f"{stem}.json", {"stem": stem, "codec": codec, "error": str(exc)})
                    log(f"ERR {codec} {stem}: {exc}")
            bundle.clear()
            free_device()
    finally:
        if log_fp is not None:
            log_fp.close()
    return {"worker_id": payload["worker_id"], "n_ok": n_ok, "n_error": n_err}


def _load(codec: str, device: str, payload: dict) -> dict:
    if codec == "moss":
        model, sr = load_model(payload["moss_path"], device)
        return {"kind": "moss", "model": model, "sr": sr, "trace": lambda wav: trace_moss(model, wav)}
    if codec == "dac":
        model = load_dac(device, payload.get("dac_path") or None)
        sr = int(getattr(model, "sample_rate", 24000))
        return {"kind": "dac", "model": model, "sr": sr, "trace": lambda wav: trace_dac(model, wav, sr)}
    tok = load_qwen_tokenizer(payload["qwen_path"], device)
    sr = int(tok.feature_extractor.sampling_rate)
    return {"kind": "qwen", "model": tok, "sr": sr, "trace": lambda wav: trace_qwen(tok, wav, sr)}


def _corr(xs, ys) -> float | None:
    x = np.asarray(xs, dtype=np.float64)
    y = np.asarray(ys, dtype=np.float64)
    ok = np.isfinite(x) & np.isfinite(y)
    if int(ok.sum()) < 3 or np.std(x[ok]) < 1e-12 or np.std(y[ok]) < 1e-12:
        return None
    return float(np.corrcoef(x[ok], y[ok])[0, 1])


def merge_summary(out_dir: Path, codecs: list[str]) -> dict:
    summary = {}
    for codec in codecs:
        rows = []
        for path in sorted((out_dir / codec).glob("*.json")):
            rows.append(json.loads(path.read_text(encoding="utf-8")))
        summary[codec] = {
            "n": len(rows),
            "pearson_eta_lsd": _corr([r.get("eta_extra") for r in rows], [r.get("lsd") for r in rows]),
            "pearson_eta_mcd": _corr([r.get("eta_extra") for r in rows], [r.get("mcd") for r in rows]),
            "mean_eta_extra": float(np.nanmean([r.get("eta_extra") for r in rows])) if rows else None,
            "mean_cancel_ratio": float(np.nanmean([r.get("cancel_ratio") for r in rows])) if rows else None,
            "mean_lsd": float(np.nanmean([r.get("lsd") if r.get("lsd") is not None else np.nan for r in rows])) if rows else None,
            "mean_mcd": float(np.nanmean([r.get("mcd") if r.get("mcd") is not None else np.nan for r in rows])) if rows else None,
        }
    return summary


def main() -> None:
    ensure_spawn()
    p = argparse.ArgumentParser(description=__doc__)
    add_npu_args(p)
    p.add_argument("--codecs", type=str, default="moss,qwen,dac")
    p.add_argument("--wav-dir", type=str, default=WAV_PATH)
    p.add_argument("--out_dir", type=str, default=DEFAULT_OUT)
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--names", type=str, default="")
    p.add_argument("--moss-path", type=str, default=default_model_path())
    p.add_argument("--qwen-path", type=str, default=QWEN_PATH)
    p.add_argument("--dac-path", type=str, default=DAC_PATH)
    args = p.parse_args()
    wav_dir = Path(args.wav_dir) if args.wav_dir else Path(default_mix_1500())
    out_dir = Path(args.out_dir) if args.out_dir else Path(r"/data1/home/ynian/codes/eval/add_exp/result/rvq_extra_eta_n1500")
    clips = list_mix_clips(wav_dir, int(args.limit))
    if args.names.strip():
        want = {x.strip() for x in args.names.split(",") if x.strip()}
        clips = [item for item in clips if item[0] in want]
    if not clips:
        raise SystemExit(f"no audio in {wav_dir}")
    codecs = [x.strip() for x in args.codecs.split(",") if x.strip()]
    print(f"clips={len(clips)} codecs={codecs} -> {out_dir}", flush=True)
    payloads = build_payloads(
        clips,
        num_workers=int(args.num_workers),
        npu_ids=parse_npu_ids(args.npu_ids),
        device_single=args.device,
        logs_dir=out_dir / "logs",
        common={
            "codecs": codecs,
            "out_dir": str(out_dir),
            "cpu_threads": int(args.cpu_threads),
            "moss_path": args.moss_path,
            "qwen_path": args.qwen_path,
            "dac_path": args.dac_path,
        },
    )
    outs = run_shards(worker, payloads, num_workers=int(args.num_workers))
    summary = merge_summary(out_dir, codecs)
    summary["workers"] = outs
    dump_json(out_dir / "summary.json", summary)
    print(json.dumps({k: summary[k] for k in codecs}, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
