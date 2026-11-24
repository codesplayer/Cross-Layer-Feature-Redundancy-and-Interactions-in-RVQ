"""MOSS encode→decode 重构：保存语音，并算 LSD / MCD / 说话人 / 情感 / DNSMOS。

相对**原始语音**打分。8×NPU。可指定 ``--wav-dir`` / ``--manifest`` / ``--out-dir``。

输出::

  {out}/wav_recon/{stem}.wav
  {out}/per_clip/{stem}.json
  {out}/summary.json
  {out}/FINDINGS.md

例::

  python -m P1.experiments.run_moss_recon_quality_n1500 \\
    --wav-dir /path/to/wavs --out-dir /path/to/out \\
    --num-workers 8 --npu-ids 0,1,2,3,4,5,6,7

  python -m P1.experiments.run_moss_recon_quality_n1500 \\
    --manifest /path/to/manifest.jsonl --out-dir /path/to/out
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import traceback
from pathlib import Path

import numpy as np

from base.config import (
    EXP_ROOT,
    WAVLM_SV_PATH,
    default_mix_1500,
    default_model_path,
    first_existing,
)
from base.io import match_len, write_wav_float
from base.lsd import CalcuLSD
from base.mcd import CalcuMCD
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
from base.util import dump_json, load_model
from P1.experiments.run_anchor_attack_three_n1500 import list_clips, load_clips_from_manifest

OUT_STEM = "moss_recon_quality_n1500"
_OUT_CANDS = (
    Path(r"/data1/home/ynian/codes/eval/add_exp/result") / OUT_STEM,
    Path(r"E:\code\water_marks\result") / OUT_STEM,
    EXP_ROOT / OUT_STEM,
)
DEFAULT_OUT = next((p for p in _OUT_CANDS if p.parent.exists()), _OUT_CANDS[0])

METRIC_VERSION = 1
_EVAL_FILES = (
    Path("eval") / "experiments" / "calcu_speaker_sim.py",
    Path("eval") / "experiments" / "calcu_emotion_sim.py",
    Path("eval") / "experiments" / "calcu_dnsmos.py",
)
_WAVLM = (
    Path(r"/data1/home/ynian/pred_models/wavlm-base-plus-sv"),
    Path(r"/home/ma-user/work/ynian/code/codec_an/model/wavlm-base-plus-sv"),
    Path(r"E:\code\codec_an\model\wavlm-base-plus-sv"),
    Path(WAVLM_SV_PATH),
)
_EMO = (
    Path(r"/data1/home/ynian/pred_models/emotion2vec_plus_large"),
    Path(r"/home/ma-user/work/ynian/pred_model/emotion2vec_plus_large"),
    Path(r"E:\code\codec_an\model\emotion2vec_plus_large"),
)
_DNSMOS_DIRS = (
    Path(r"/data1/home/ynian/pred_models/DNSMOS"),
    Path(r"/home/ma-user/work/ynian/code/codec_an/model/DNSMOS"),
    Path(r"E:\code\codec_an\model\DNSMOS"),
    Path(r"D:\data\codec_an\model\DNSMOS"),
)


def find_eval_root(explicit: str = "") -> Path:
    here = Path(__file__).resolve()
    candidates = [
        explicit,
        os.environ.get("CODEC_AN", ""),
        r"/data1/home/ynian/codes/eval/add_exp/water_marks",
        r"/data1/home/ynian/codes/codec_an",
        str(Path(r"E:\code\water_marks")),
        str(Path(r"E:\code\codec_an")),
    ]
    for parent in here.parents:
        candidates.append(str(parent / "codec_an"))
        candidates.append(str(parent))
    seen: set[str] = set()
    for raw in candidates:
        if not raw:
            continue
        root = str(Path(raw))
        if root in seen:
            continue
        seen.add(root)
        if all((Path(root) / rel).is_file() for rel in _EVAL_FILES):
            return Path(root)
    raise SystemExit("cannot find eval (speaker/emotion/dnsmos); set --codec-an")


def resolve_dnsmos_dir(explicit: str = "") -> Path:
    cands = [Path(explicit)] if explicit else []
    cands.extend(_DNSMOS_DIRS)
    for d in cands:
        if d and (d / "sig_bak_ovr.onnx").is_file() and (d / "model_v8.onnx").is_file():
            return d
    raise SystemExit("cannot find DNSMOS onnx; set --dnsmos-dir")


def use_eval_root(root: Path) -> None:
    text = str(root)
    if text not in sys.path:
        sys.path.insert(0, text)


def unlink_quiet(path: Path) -> None:
    try:
        path.unlink(missing_ok=True)
    except TypeError:
        try:
            if path.exists():
                path.unlink()
        except Exception:
            pass
    except Exception:
        pass


def _write_json(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(obj, ensure_ascii=False), encoding="utf-8")
    tmp.replace(path)


def clip_ok(path: Path, wav_path: Path) -> bool:
    if not path.is_file() or not wav_path.is_file():
        return False
    try:
        doc = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return False
    return (
        isinstance(doc, dict)
        and doc.get("metric_version") == METRIC_VERSION
        and doc.get("lsd") is not None
        and "error" not in doc
    )


def score_quality(
    *,
    ref_wav: np.ndarray,
    deg_wav: np.ndarray,
    sr: int,
    ref_path: str,
    tmp: Path,
    stem: str,
    lsd_tool: CalcuLSD,
    mcd_tool: CalcuMCD,
    spk,
    emo,
    dnsmos,
) -> dict:
    ref = np.asarray(ref_wav, dtype=np.float32).reshape(-1)
    deg = np.asarray(match_len(ref, deg_wav), dtype=np.float32).reshape(-1)
    out: dict = {}
    try:
        val, _ = lsd_tool.compute(ref, deg)
        out["lsd"] = float(val)
    except Exception as exc:
        out["lsd"] = None
        out["lsd_error"] = str(exc)
    rp = tmp / f"{stem}_ref.wav"
    dp = tmp / f"{stem}_deg.wav"
    write_wav_float(rp, ref, sr)
    write_wav_float(dp, deg, sr)
    try:
        out["mcd"] = float(mcd_tool.mcd_file(str(rp), str(dp)))
    except Exception as exc:
        out["mcd"] = None
        out["mcd_error"] = str(exc)
    try:
        mos = dnsmos(str(dp), 16000, False)
        out["speaker_cos"] = float(spk.sim_file(str(ref_path), str(dp)))
        out["emotion_cos"] = float(emo.sim_file(str(ref_path), str(dp)))
        out["dnsmos_ovrl"] = float(mos["OVRL"])
        out["dnsmos_sig"] = float(mos["SIG"])
        out["dnsmos_bak"] = float(mos["BAK"])
        out["dnsmos_p808"] = float(mos["P808_MOS"])
    except Exception as exc:
        out["attr_error"] = str(exc)
    finally:
        unlink_quiet(rp)
        unlink_quiet(dp)
    return out


def _worker(payload: dict) -> dict:
    limit_cpu_threads(int(payload.get("cpu_threads") or 1))
    device = str(payload["device"])
    pin_npu(device)
    log_fp, log = open_worker_log(payload)
    per_dir = Path(payload["per_clip_dir"])
    wav_out = Path(payload["wav_recon_dir"])
    tmp = Path(payload["tmp_dir"]) / f"w{payload['worker_id']}"
    per_dir.mkdir(parents=True, exist_ok=True)
    wav_out.mkdir(parents=True, exist_ok=True)
    tmp.mkdir(parents=True, exist_ok=True)
    resume = bool(payload.get("resume"))
    n_ok = n_skip = n_err = 0

    use_eval_root(Path(payload["codec_an"]))
    from eval.experiments.calcu_dnsmos import ComputeScore
    from eval.experiments.calcu_emotion_sim import CalcuEmotionSim
    from eval.experiments.calcu_speaker_sim import CalcuSpeakerSim

    try:
        set_seed(int(payload.get("seed") or 0) + int(payload["worker_id"]))
        model, sr = load_model(str(payload["moss_path"]), device)
        lsd_tool = CalcuLSD(sr=int(sr))
        mcd_tool = CalcuMCD()
        spk = CalcuSpeakerSim(model_source=str(payload["wavlm"]), device=device)
        emo = CalcuEmotionSim(model_id=str(payload["emotion2vec"]), device=device)
        dnsmos_dir = Path(payload["dnsmos_dir"])
        dnsmos = ComputeScore(
            str(dnsmos_dir / "sig_bak_ovr.onnx"),
            str(dnsmos_dir / "model_v8.onnx"),
            num_threads=1,
        )
        log(f"start clips={len(payload['items'])} sr={sr}")

        for i, (stem, path_str) in enumerate(payload["items"]):
            safe = str(stem).replace("/", "_").replace("\\", "_")
            out_json = per_dir / f"{safe}.json"
            out_wav = wav_out / f"{safe}.wav"
            if resume and clip_ok(out_json, out_wav):
                n_skip += 1
                n_ok += 1
                continue
            try:
                wav = np.asarray(read_wav(path_str, sr), dtype=np.float32).reshape(-1)
                codes = encode_moss(model, wav)
                recon = np.asarray(match_len(wav, decode_moss(model, codes)), dtype=np.float32).reshape(-1)
                write_wav_float(out_wav, recon, int(sr))
                qual = score_quality(
                    ref_wav=wav,
                    deg_wav=recon,
                    sr=int(sr),
                    ref_path=path_str,
                    tmp=tmp,
                    stem=safe,
                    lsd_tool=lsd_tool,
                    mcd_tool=mcd_tool,
                    spk=spk,
                    emo=emo,
                    dnsmos=dnsmos,
                )
                _write_json(
                    out_json,
                    {
                        "stem": stem,
                        "src": path_str,
                        "recon_wav": str(out_wav.resolve()),
                        "codec": "moss",
                        "sr": int(sr),
                        "metric_version": METRIC_VERSION,
                        "n_samples": int(wav.shape[0]),
                        **qual,
                    },
                )
                n_ok += 1
            except Exception as exc:
                n_err += 1
                _write_json(
                    out_json,
                    {"stem": stem, "src": path_str, "error": str(exc), "trace": traceback.format_exc()},
                )
                log(f"ERR {stem}: {exc}")
            if (i + 1) % 20 == 0:
                log(f"progress {i+1}/{len(payload['items'])} ok={n_ok} skip={n_skip} err={n_err}")
    finally:
        free_device()
        if log_fp:
            log_fp.close()
    return {
        "worker_id": payload["worker_id"],
        "n_ok": n_ok,
        "n_skip": n_skip,
        "n_err": n_err,
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
        docs.append(d)
    keys = ("lsd", "mcd", "speaker_cos", "emotion_cos", "dnsmos_ovrl", "dnsmos_sig", "dnsmos_bak", "dnsmos_p808")
    return {
        "codec": "moss",
        "n_clips": len(docs),
        "metric_version": METRIC_VERSION,
        "mean": {k: _mean([d.get(k) for d in docs]) for k in keys},
    }


def render_md(summary: dict) -> str:
    m = summary.get("mean") or {}

    def f(x):
        return "-" if x is None else f"{float(x):.4f}"

    return (
        f"# MOSS recon quality\n\n"
        f"n={summary.get('n_clips')}  metric_v={summary.get('metric_version')}\n\n"
        f"| LSD | MCD | speaker | emotion | DNSMOS_ovrl | DNSMOS_p808 |\n"
        f"|---|---|---|---|---|---|\n"
        f"| {f(m.get('lsd'))} | {f(m.get('mcd'))} | {f(m.get('speaker_cos'))} | "
        f"{f(m.get('emotion_cos'))} | {f(m.get('dnsmos_ovrl'))} | {f(m.get('dnsmos_p808'))} |\n"
    )


def parse_args():
    p = argparse.ArgumentParser(description="MOSS recon + LSD/MCD/spk/emo/DNSMOS")
    p.add_argument("--wav-dir", type=str, default=default_mix_1500())
    p.add_argument("--manifest", type=str, default="", help="优先于 --wav-dir")
    p.add_argument("--names", type=str, default="")
    p.add_argument("--out-dir", type=str, default=str(DEFAULT_OUT))
    p.add_argument("--moss-path", type=str, default=default_model_path())
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--resume", action="store_true", default=True)
    p.add_argument("--no-resume", action="store_true")
    p.add_argument("--codec-an", type=str, default="")
    p.add_argument("--wavlm", type=str, default=first_existing(*_WAVLM))
    p.add_argument("--emotion2vec", type=str, default=first_existing(*_EMO))
    p.add_argument("--dnsmos-dir", type=str, default="")
    p.add_argument("--summary-only", action="store_true")
    add_npu_args(p, default_workers=8, default_device="npu:0")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if args.no_resume:
        args.resume = False
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    per_dir = out_dir / "per_clip"
    wav_recon = out_dir / "wav_recon"

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
        name_set = {x.strip() for x in str(args.names).replace(" ", ",").split(",") if x.strip()} or None
        clips = list_clips(Path(args.wav_dir), limit=int(args.limit), names=name_set)
        src_desc = f"wav_dir={args.wav_dir}" + (f" names={len(name_set)}" if name_set else "")
    if not clips:
        raise SystemExit(f"no clips ({src_desc})")

    codec_an = find_eval_root(str(args.codec_an))
    dnsmos_dir = resolve_dnsmos_dir(str(args.dnsmos_dir))
    n_workers = max(1, int(args.num_workers))
    per_dir.mkdir(parents=True, exist_ok=True)
    wav_recon.mkdir(parents=True, exist_ok=True)
    logs_dir = out_dir / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)

    (out_dir / "RUN.md").write_text(
        "# MOSS recon quality\n\n"
        f"- source: `{src_desc}`\n"
        f"- n_clips: {len(clips)}\n"
        f"- metrics: LSD, MCD, speaker_cos, emotion_cos, DNSMOS (vs original)\n"
        f"- wav_recon: `{wav_recon}`\n"
        f"- workers: {n_workers}\n",
        encoding="utf-8",
    )
    print(
        f"moss_recon n={len(clips)} workers={n_workers} src={src_desc} -> {out_dir}",
        flush=True,
    )

    payloads = build_payloads(
        clips,
        num_workers=n_workers,
        npu_ids=parse_npu_ids(args.npu_ids),
        device_single=str(args.device),
        logs_dir=logs_dir,
        common={
            "moss_path": args.moss_path,
            "per_clip_dir": str(per_dir),
            "wav_recon_dir": str(wav_recon),
            "tmp_dir": str(out_dir / "tmp"),
            "cpu_threads": int(args.cpu_threads),
            "resume": bool(args.resume),
            "seed": int(args.seed),
            "codec_an": str(codec_an),
            "wavlm": args.wavlm,
            "emotion2vec": args.emotion2vec,
            "dnsmos_dir": str(dnsmos_dir),
        },
    )
    outs = run_shards(_worker, payloads, num_workers=n_workers)
    dump_json(out_dir / "workers.json", outs)
    summary = aggregate(per_dir)
    dump_json(out_dir / "summary.json", summary)
    (out_dir / "FINDINGS.md").write_text(render_md(summary), encoding="utf-8")
    print(
        f"done n={summary['n_clips']} err={sum(int(o.get('n_err') or 0) for o in outs)} -> {out_dir}",
        flush=True,
    )


if __name__ == "__main__":
    ensure_spawn()
    main()
